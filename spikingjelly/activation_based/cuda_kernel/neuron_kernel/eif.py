import numpy as np
import torch
import torch.nn.functional as F

from .... import configure
from .. import cuda_utils, tensor_cache
from ..cuda_utils import resolve_python_object
from .common import (
    _CapturedAutogradCtx,
    _decode_v_reset,
    _resolve_sg_cuda_code_fun,
    _should_stash_capture_ctx,
    _sg_obj_id,
    _stash_capture_ctx,
    _take_capture_ctx,
    cupy,
)


__all__ = ["create_fptt_kernel", "create_bptt_kernel", "multistep_eif_ptt"]

def create_fptt_kernel(hard_reset: bool, dtype: str):
    kernel_name = f"EIFNode_fptt_{'hard' if hard_reset else 'soft'}Reset_{dtype}"

    if dtype == "fp32":
        code = rf"""
        extern "C" __global__
        void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
        const float & reciprocal_tau,
        const float & delta_T,
        const float & theta_rh,
        const float & v_threshold,
        const float & v_rest, {"const float & v_reset," if hard_reset else ""}
        const int & neuron_num, const int & numel)
        """
        code += r"""
        {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < neuron_num)
        {
            const int dt = neuron_num;
            for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
            {
                const int t = index + mem_offset;
                h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_rest + delta_T * expf((v_v_seq[t] - theta_rh) / delta_T));
                if (h_seq[t] >= v_threshold)
                {
                    spike_seq[t] = 1.0f;
        """

        if hard_reset:
            code += r"""
                    v_v_seq[t + dt] = v_reset;
            """
        else:
            code += r"""
                    v_v_seq[t + dt] = h_seq[t] - v_threshold;
            """

        code += r"""
                }
                else
                {
                    spike_seq[t] = 0.0f;
                    v_v_seq[t + dt] = h_seq[t];
                }

            }
        }
        }
        """

    elif dtype == "fp16":
        code = rf"""
        #include <cuda_fp16.h>
        extern "C" __global__
        void {kernel_name}(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
        const half & reciprocal_tau,
        const half & delta_T,
        const half & theta_rh,
        const half & v_threshold,
        const half & v_rest, {"const half & v_reset," if hard_reset else ""}
        const int & neuron_num, const int & numel)
        """

        code += r"""
        {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = neuron_num >> 1;
        if (index < stride)
        {
            const int numel_2 = numel >> 1;
            const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
            const half2 delta_T_half2 = __half2half2(delta_T);
            const half2 theta_rh_half2 = __half2half2(theta_rh);
            const half2 v_threshold_half2 = __half2half2(v_threshold);
            const half2 v_rest_half2 = __half2half2(v_rest);
        """

        if hard_reset:
            code += r"""
                const half2 v_reset_half2 = __half2half2(v_reset);
            """

        code += r"""
            for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
            {
                const int t = index + mem_offset;
                h_seq[t] = __hfma2(__hfma2(h2exp(__h2div(__hsub2(v_v_seq[t], theta_rh_half2), delta_T_half2)), delta_T_half2, __hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_rest_half2)), reciprocal_tau_half2, v_v_seq[t]);

                spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
        """

        if hard_reset:
            code += r"""
                v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
            """
        else:
            code += r"""
                v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
            """

        code += r"""
            }
        }
        }
        """
    else:
        raise TypeError

    return cupy.RawKernel(
        code,
        kernel_name,
        options=configure.cuda_compiler_options,
        backend=configure.cuda_compiler_backend,
    )


def create_bptt_kernel(
    sg_cuda_code_fun, hard_reset: bool, detach_reset: bool, dtype: str
):
    kernel_name = f"EIFNode_bptt_{'hard' if hard_reset else 'soft'}Reset_{'detachReset' if detach_reset else ''}_{dtype}"

    code_grad_s_to_h = sg_cuda_code_fun(x="over_th", y="grad_s_to_h", dtype=dtype)

    if dtype == "fp32":
        code = rf"""
        extern "C" __global__
        void {kernel_name}(
        const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
        float* grad_x_seq, float* grad_v_init,
        const float & theta_rh, const float & reciprocal_delta_T,
        const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
        const float & v_threshold, {"const float & v_reset," if hard_reset else ""}
        const int & neuron_num, const int & numel)
        """

        code += r"""
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < neuron_num)
            {
                float grad_h = 0.0f;  // grad_h will be used recursively
                for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                {
                    const int t = index + mem_offset;
                    const float over_th = h_seq[t] - v_threshold;
        """
        code += code_grad_s_to_h
        if detach_reset:
            if hard_reset:
                code_grad_v_to_h = r"""
                const float grad_v_to_h = 1.0f - spike_seq[t];
                """
            else:
                code_grad_v_to_h = r"""
                const float grad_v_to_h = 1.0f;
                """
        else:
            if hard_reset:
                code_grad_v_to_h = r"""
                const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                """
            else:
                code_grad_v_to_h = r"""
                const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                """

        code += code_grad_v_to_h
        code += r"""
            grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
            grad_x_seq[t] = grad_h * reciprocal_tau;
            }
        grad_v_init[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
        }
        }
        """

    elif dtype == "fp16":
        code = rf"""
        #include <cuda_fp16.h>
        extern "C" __global__
        void {kernel_name}(
        const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
        half2* grad_x_seq, half2* grad_v_init,
        const half & theta_rh, const half & reciprocal_delta_T,
        const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
        const half & v_threshold, {"const half & v_reset," if hard_reset else ""}
        const int & neuron_num, const int & numel)
        """
        code += r"""
        {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = neuron_num >> 1;
        if (index < stride)
        {
            const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
            const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
            const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
            const half2 theta_rh_half2 = __half2half2(theta_rh);
            const half2 v_threshold_half2 = __half2half2(v_threshold);
        """

        if hard_reset:
            code += r"""
                const half2 v_reset_half2 = __half2half2(v_reset);
            """

        code += r"""
            half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
            for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
            {
                const int t = index + mem_offset;

                const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
        """
        code += code_grad_s_to_h

        if detach_reset:
            if hard_reset:
                code_grad_v_to_h = r"""
                const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                """
            else:
                code_grad_v_to_h = r"""
                const half2 grad_v_to_h = __float2half2_rn(1.0f);
                """
        else:
            if hard_reset:
                code_grad_v_to_h = r"""
                const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                """
            else:
                code_grad_v_to_h = r"""
                const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                """

        code += code_grad_v_to_h
        code += r"""
                grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
            }
        grad_v_init[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
        }
        }
        """
    else:
        raise TypeError
    return cupy.RawKernel(
        code,
        kernel_name,
        options=configure.cuda_compiler_options,
        backend=configure.cuda_compiler_backend,
    )


def _eif_forward(
    ctx,
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    tau: float,
    v_threshold: float,
    v_reset: float,
    v_rest: float,
    theta_rh: float,
    delta_T: float,
    detach_reset: bool,
    sg_cuda_code_fun,
):
    requires_grad = x_seq.requires_grad or v_init.requires_grad
    device = x_seq.get_device()
    if x_seq.dtype == torch.float32:
        dtype = "fp32"
        cp_dtype = np.float32
    elif x_seq.dtype == torch.float16:
        dtype = "fp16"
        cp_dtype = np.half
    else:
        raise NotImplementedError

    use_pad = False
    if dtype == "fp16" and v_init.numel() % 2 != 0:
        # only fp16 needs even numel because we use half2 to accelerate
        # when numel is odd, we will pad x_seq
        use_pad = True
        x_seq = F.pad(x_seq, (0, 1))  # [T, N] -> [T, N + 1]
        v_init = F.pad(v_init, (0, 1))  # [N] -> [N + 1]

    zero_shape = list(x_seq.shape)
    zero_shape[0] *= 3
    v_seq, h_seq, spike_seq = torch.split(
        torch.zeros(zero_shape, device=x_seq.device, dtype=x_seq.dtype),
        x_seq.shape[0],
    )

    v_v_seq = torch.cat((v_init.unsqueeze(0), v_seq))

    with cuda_utils.DeviceEnvironment(device):
        numel = x_seq.numel()
        neuron_num = numel // x_seq.shape[0]

        threads = configure.cuda_threads
        if dtype == "fp16":
            assert neuron_num % 2 == 0
            blocks = cuda_utils.cal_blocks(neuron_num >> 1)
            # we will take two neurons to calculate as one neuron in cuda half2
        else:
            blocks = cuda_utils.cal_blocks(neuron_num)

        cp_numel = cupy.asarray(numel)
        cp_neuron_num = cupy.asarray(neuron_num)
        cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
        cp_v_rest = cupy.asarray(v_rest, dtype=cp_dtype)
        cp_theta_rh = cupy.asarray(theta_rh, dtype=cp_dtype)
        cp_delta_T = cupy.asarray(delta_T, dtype=cp_dtype)
        cp_reciprocal_delta_T = cupy.asarray(1.0 / delta_T, dtype=cp_dtype)
        cp_reciprocal_tau = cupy.asarray(1.0 / tau, dtype=cp_dtype)
        cp_one_sub_reciprocal_tau = cupy.asarray(1.0 - 1.0 / tau, dtype=cp_dtype)

        if v_reset is None:
            cp_v_reset = None
            hard_reset = False
            (
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
                cp_reciprocal_tau,
                cp_delta_T,
                cp_theta_rh,
                cp_v_threshold,
                cp_v_rest,
                cp_neuron_num,
                cp_numel,
            ) = cuda_utils.get_contiguous(
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
                cp_reciprocal_tau,
                cp_delta_T,
                cp_theta_rh,
                cp_v_threshold,
                cp_v_rest,
                cp_neuron_num,
                cp_numel,
            )
            kernel_args = [
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
                cp_reciprocal_tau,
                cp_delta_T,
                cp_theta_rh,
                cp_v_threshold,
                cp_v_rest,
                cp_neuron_num,
                cp_numel,
            ]
        else:
            cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
            hard_reset = True
            (
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
                cp_reciprocal_tau,
                cp_delta_T,
                cp_theta_rh,
                cp_v_threshold,
                cp_v_rest,
                cp_v_reset,
                cp_neuron_num,
                cp_numel,
            ) = cuda_utils.get_contiguous(
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
                cp_reciprocal_tau,
                cp_delta_T,
                cp_theta_rh,
                cp_v_threshold,
                cp_v_rest,
                cp_v_reset,
                cp_neuron_num,
                cp_numel,
            )
            kernel_args = [
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
                cp_reciprocal_tau,
                cp_delta_T,
                cp_theta_rh,
                cp_v_threshold,
                cp_v_rest,
                cp_v_reset,
                cp_neuron_num,
                cp_numel,
            ]

        kernel = create_fptt_kernel(hard_reset, dtype)

        kernel(
            (blocks,),
            (threads,),
            cuda_utils.wrap_args_to_raw_kernel(device, *kernel_args),
        )

    if requires_grad:
        ctx.use_pad = use_pad
        if configure.save_spike_as_bool_in_neuron_kernel:
            ctx.s_shape = spike_seq.shape
            ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike_seq)
            ctx.save_for_backward(h_seq, v_v_seq)
        else:
            ctx.save_for_backward(h_seq, spike_seq, v_v_seq)
        ctx.blocks = blocks
        ctx.threads = threads
        ctx.cp_numel = cp_numel
        ctx.cp_neuron_num = cp_neuron_num
        ctx.cp_reciprocal_tau = cp_reciprocal_tau
        ctx.cp_one_sub_reciprocal_tau = cp_one_sub_reciprocal_tau
        ctx.cp_theta_rh = cp_theta_rh
        ctx.cp_reciprocal_delta_T = cp_reciprocal_delta_T
        ctx.cp_v_threshold = cp_v_threshold
        ctx.cp_v_reset = cp_v_reset
        ctx.detach_reset = detach_reset
        ctx.sg_cuda_code_fun = sg_cuda_code_fun

    if use_pad:
        return spike_seq[..., :-1], v_v_seq[1:, ..., :-1]
    else:
        return spike_seq, v_v_seq[1:,]


def _eif_backward(ctx, grad_spike_seq, grad_v_seq):
    if ctx.use_pad:
        # grad_spike_seq.shape = [T, N]
        # grad_v_seq.shape = [T, N]
        # h_seq.shape = [T, N + 1]
        # spike_seq.shape = [T, N + 1]
        grad_spike_seq = F.pad(grad_spike_seq, (0, 1))
        grad_v_seq = F.pad(grad_v_seq, (0, 1))

    device = grad_spike_seq.get_device()
    if configure.save_spike_as_bool_in_neuron_kernel:
        spike_seq = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
        h_seq, v_v_seq = ctx.saved_tensors
    else:
        h_seq, spike_seq, v_v_seq = ctx.saved_tensors
    zero_shape = list(grad_spike_seq.shape)
    zero_shape[0] += 1
    zero_data = torch.zeros(
        zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype
    )
    grad_x_seq = zero_data[0:-1]
    grad_v_init = zero_data[-1]

    if ctx.cp_v_reset is None:
        hard_reset = False
    else:
        hard_reset = True

    if grad_spike_seq.dtype == torch.float32:
        dtype = "fp32"
    elif grad_spike_seq.dtype == torch.float16:
        dtype = "fp16"
    else:
        raise NotImplementedError

    kernel = create_bptt_kernel(
        ctx.sg_cuda_code_fun, hard_reset, ctx.detach_reset, dtype
    )

    with cuda_utils.DeviceEnvironment(device):
        if hard_reset:
            (
                grad_spike_seq,
                grad_v_seq,
                h_seq,
                spike_seq,
                v_v_seq,
                grad_x_seq,
                grad_v_init,
                ctx.cp_theta_rh,
                ctx.cp_reciprocal_delta_T,
                ctx.cp_reciprocal_tau,
                ctx.cp_one_sub_reciprocal_tau,
                ctx.cp_v_threshold,
                ctx.cp_v_reset,
                ctx.cp_neuron_num,
                ctx.cp_numel,
            ) = cuda_utils.get_contiguous(
                grad_spike_seq,
                grad_v_seq,
                h_seq,
                spike_seq,
                v_v_seq,
                grad_x_seq,
                grad_v_init,
                ctx.cp_theta_rh,
                ctx.cp_reciprocal_delta_T,
                ctx.cp_reciprocal_tau,
                ctx.cp_one_sub_reciprocal_tau,
                ctx.cp_v_threshold,
                ctx.cp_v_reset,
                ctx.cp_neuron_num,
                ctx.cp_numel,
            )
            kernel_args = [
                grad_spike_seq,
                grad_v_seq,
                h_seq,
                spike_seq,
                v_v_seq,
                grad_x_seq,
                grad_v_init,
                ctx.cp_theta_rh,
                ctx.cp_reciprocal_delta_T,
                ctx.cp_reciprocal_tau,
                ctx.cp_one_sub_reciprocal_tau,
                ctx.cp_v_threshold,
                ctx.cp_v_reset,
                ctx.cp_neuron_num,
                ctx.cp_numel,
            ]
        else:
            (
                grad_spike_seq,
                grad_v_seq,
                h_seq,
                spike_seq,
                v_v_seq,
                grad_x_seq,
                grad_v_init,
                ctx.cp_theta_rh,
                ctx.cp_reciprocal_delta_T,
                ctx.cp_reciprocal_tau,
                ctx.cp_one_sub_reciprocal_tau,
                ctx.cp_v_threshold,
                ctx.cp_neuron_num,
                ctx.cp_numel,
            ) = cuda_utils.get_contiguous(
                grad_spike_seq,
                grad_v_seq,
                h_seq,
                spike_seq,
                v_v_seq,
                grad_x_seq,
                grad_v_init,
                ctx.cp_theta_rh,
                ctx.cp_reciprocal_delta_T,
                ctx.cp_reciprocal_tau,
                ctx.cp_one_sub_reciprocal_tau,
                ctx.cp_v_threshold,
                ctx.cp_neuron_num,
                ctx.cp_numel,
            )
            kernel_args = [
                grad_spike_seq,
                grad_v_seq,
                h_seq,
                spike_seq,
                v_v_seq,
                grad_x_seq,
                grad_v_init,
                ctx.cp_theta_rh,
                ctx.cp_reciprocal_delta_T,
                ctx.cp_reciprocal_tau,
                ctx.cp_one_sub_reciprocal_tau,
                ctx.cp_v_threshold,
                ctx.cp_neuron_num,
                ctx.cp_numel,
            ]

        kernel(
            (ctx.blocks,),
            (ctx.threads,),
            cuda_utils.wrap_args_to_raw_kernel(device, *kernel_args),
        )
    if ctx.use_pad:
        return (
            grad_x_seq[..., :-1],
            grad_v_init[..., :-1],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    else:
        return (
            grad_x_seq,
            grad_v_init,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.library.custom_op("sj::cupy_multistep_eif_forward", mutates_args=())
def cupy_multistep_eif_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    tau: float,
    v_threshold: float,
    v_reset: float,
    v_rest: float,
    theta_rh: float,
    delta_T: float,
    detach_reset: bool,
    sg_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sg = resolve_python_object(sg_id)
    captured_ctx = _CapturedAutogradCtx()
    out = _eif_forward(
        captured_ctx,
        x_seq,
        v_init,
        tau,
        v_threshold,
        _decode_v_reset(v_reset),
        v_rest,
        theta_rh,
        delta_T,
        detach_reset,

        _resolve_sg_cuda_code_fun(sg),
    )
    capture_id = (
        _stash_capture_ctx(captured_ctx)
        if _should_stash_capture_ctx((x_seq, v_init))
        else -1
    )
    capture_token = torch.tensor(capture_id, device=x_seq.device, dtype=torch.int64)
    return (*out, capture_token)


@torch.library.register_fake("sj::cupy_multistep_eif_forward")
def _cupy_multistep_eif_forward_fake(*args):
    x_seq = args[0]
    return (x_seq.new_empty(x_seq.shape), x_seq.new_empty(x_seq.shape), x_seq.new_empty((), dtype=torch.int64))


def _setup_ctx(ctx, inputs, output):
    capture_token = output[-1]
    if capture_token.is_meta:
        ctx.captured = None
        return
    capture_id = int(capture_token.item())
    if capture_id < 0:
        ctx.captured = None
        return
    ctx.captured = _take_capture_ctx(capture_id)


def _bw(ctx, *grad_outputs):
    if ctx.captured is None:
        raise RuntimeError("Missing captured context for backward.")
    grads = _eif_backward(ctx.captured, *grad_outputs[:-1])
    return grads[0], grads[1], None, None, None, None, None, None, None, None


torch.library.register_autograd("sj::cupy_multistep_eif_forward", _bw, setup_context=_setup_ctx)


def multistep_eif_ptt(
    x_seq,
    v_init,
    tau,
    v_threshold,
    v_reset,
    v_rest,
    theta_rh,
    delta_T,
    detach_reset,
    surrogate_function,
):
    sg_id = _sg_obj_id(surrogate_function)
    v_reset_value = float("nan") if v_reset is None else float(v_reset)
    return cupy_multistep_eif_forward(
        x_seq,
        v_init,
        tau,
        v_threshold,
        v_reset_value,
        v_rest,
        theta_rh,
        delta_T,
        detach_reset,

        sg_id,
    )[:-1]
