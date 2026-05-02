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
    _sg_obj_id,
    _stash_capture_ctx,
    _take_capture_ctx,
    cupy,
)
from .lif import create_fptt_kernel as lif_create_fptt_kernel

__all__ = ["create_fptt_kernel", "create_bptt_kernel", "multistep_plif_ptt"]

def create_fptt_kernel(decay_input: bool, hard_reset: bool, dtype: str):
    return lif_create_fptt_kernel(
        decay_input, hard_reset, dtype, kernel_name_prefix="ParametricLIFNode"
    )


def create_bptt_kernel(
    sg_cuda_code_fun,
    decay_input: bool,
    hard_reset: bool,
    detach_reset: bool,
    dtype: str,
):
    kernel_name = f"ParametricLIFNode_bptt_decayInput{decay_input}_{'hard' if hard_reset else 'soft'}Reset_{'detachReset' if detach_reset else ''}_{dtype}"

    code_grad_s_to_h = sg_cuda_code_fun(x="over_th", y="grad_s_to_h", dtype=dtype)

    if dtype == "fp32":
        code = rf"""
        extern "C" __global__
        void {kernel_name}(
        const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
        float* grad_x_seq, float* grad_v_init, float* grad_reciprocal_tau,
        const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
        const float & v_threshold, {"const float & v_reset," if hard_reset else ""}
        const int & neuron_num, const int & numel)
        """
        code += r"""
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
        """
        code += f"__shared__ float sdata[{configure.cuda_threads}];"
        code += r"""
            if (index < neuron_num)
            {
                float grad_h = 0.0f;  // grad_h will be used recursively
                sdata[threadIdx.x] = 0.0f;
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
                // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                """
            else:
                code_grad_v_to_h = r"""
                const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                """

        code += code_grad_v_to_h
        code += r"""
            grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
            // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
        """
        if decay_input:
            code += r"""
                grad_x_seq[t] = grad_h * reciprocal_tau;
                sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
            """
        else:
            if hard_reset:
                code += r"""
                    grad_x_seq[t] = grad_h;
                    sdata[threadIdx.x] += grad_h * (v_reset - v_v_seq[t]);
                """
            else:
                code += r"""
                    grad_x_seq[t] = grad_h;
                    sdata[threadIdx.x] -= grad_h * v_v_seq[t];
                """
        code += r"""
            }
        grad_v_init[index] = grad_h * one_sub_reciprocal_tau;
        }
        else
        {
            sdata[threadIdx.x] = 0.0f;
        }
        int threadx = blockDim.x;
        #pragma unroll
        for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
        {
        // Synchronize all thread before next loop
        __syncthreads();
        if (threadIdx.x < stride)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
        atomicAdd(grad_reciprocal_tau, sdata[0]);
        }
        }
        """

    elif dtype == "fp16":
        code = rf"""
        #include <cuda_fp16.h>
        extern "C" __global__
        void {kernel_name}(
        const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
        half2* grad_x_seq, half2* grad_v_init,  float* grad_reciprocal_tau,
        const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
        const half & v_threshold, {"const half & v_reset," if hard_reset else ""}
        const int & neuron_num, const int & numel)\
        // note that grad_reciprocal_tau is float to avoid overflow
        """
        code += r"""
        {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = neuron_num >> 1;

        """
        code += f"__shared__ float sdata[{configure.cuda_threads}];"
        code += r"""
        if (index < stride)
        {
            const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
            const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
            const half2 v_threshold_half2 = __half2half2(v_threshold);
        """

        if hard_reset:
            code += r"""
                const half2 v_reset_half2 = __half2half2(v_reset);
            """

        code += r"""

            half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
            sdata[threadIdx.x] = 0.0f;
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
                grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
        """
        if decay_input:
            code += r"""
                    grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                    half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                    sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
            """
        else:
            if hard_reset:
                code += r"""
                        grad_x_seq[t] = grad_h;
                        half2 temp_sum = __hmul2(grad_h, __hsub2(v_reset_half2, v_v_seq[t]));
                        sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                """
            else:
                code += r"""
                        grad_x_seq[t] = grad_h;
                        half2 temp_sum = __hmul2(grad_h, __hneg2(v_v_seq[t]));
                        sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                """
        code += r"""
            }
        grad_v_init[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
        }
        else
        {
            sdata[threadIdx.x] = 0.0f;
        }
        int threadx = blockDim.x;
        #pragma unroll
        for (int i = threadx >> 1; i > 0; i = i >> 1)
        {
        // Synchronize all thread before next loop
        __syncthreads();
        if (threadIdx.x < i)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
        /*
        The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

        The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.

        The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.

        The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.

        The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
        */

        atomicAdd(grad_reciprocal_tau, sdata[0]);

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


def _plif_forward(
    ctx,
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    reciprocal_tau: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: float,
    detach_reset: bool,
    sg_cuda_code_fun,
):
    # reciprocal_tau.dtype is float32 even when using amp
    requires_grad = x_seq.requires_grad or v_init.requires_grad
    device = x_seq.get_device()
    if x_seq.dtype == torch.float32:
        dtype = "fp32"
        cp_dtype = np.float32
    elif x_seq.dtype == torch.float16:
        dtype = "fp16"
        cp_dtype = np.half
        # assert torch.cuda.get_device_capability(device)[0] >= 7, "MultiStepParametricLIFNodePTT can not run in the current device with float16 because the 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher."

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
    tau = 1.0 / reciprocal_tau.item()

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
                cp_v_threshold,
                cp_neuron_num,
                cp_numel,
            ) = cuda_utils.get_contiguous(
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
                cp_reciprocal_tau,
                cp_v_threshold,
                cp_neuron_num,
                cp_numel,
            )
            kernel_args = [
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
                cp_reciprocal_tau,
                cp_v_threshold,
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
                cp_v_threshold,
                cp_v_reset,
                cp_neuron_num,
                cp_numel,
            ) = cuda_utils.get_contiguous(
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
                cp_reciprocal_tau,
                cp_v_threshold,
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
                cp_v_threshold,
                cp_v_reset,
                cp_neuron_num,
                cp_numel,
            ]

        kernel = create_fptt_kernel(
            decay_input, hard_reset, dtype
        )

        kernel(
            (blocks,),
            (threads,),
            cuda_utils.wrap_args_to_raw_kernel(device, *kernel_args),
        )

    if requires_grad:
        ctx.decay_input = decay_input
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
        ctx.cp_v_threshold = cp_v_threshold
        ctx.cp_v_reset = cp_v_reset
        ctx.detach_reset = detach_reset
        ctx.sg_cuda_code_fun = sg_cuda_code_fun

    if use_pad:
        return spike_seq[..., :-1], v_v_seq[1:, ..., :-1]
    else:
        return spike_seq, v_v_seq[1:,]


def _plif_backward(ctx, grad_spike_seq, grad_v_seq):
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
    grad_reciprocal_tau = torch.as_tensor(
        0.0, device=grad_spike_seq.device, dtype=torch.float32
    )

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
        ctx.sg_cuda_code_fun, ctx.decay_input, hard_reset, ctx.detach_reset, dtype
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
                grad_reciprocal_tau,
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
                grad_reciprocal_tau,
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
                grad_reciprocal_tau,
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
                grad_reciprocal_tau,
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
                grad_reciprocal_tau,
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
                grad_reciprocal_tau,
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
            grad_reciprocal_tau,
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
            grad_reciprocal_tau,
            None,
            None,
            None,
            None,
            None,
        )


_PLIF_OP_NAME = "sj::cupy_neuron_kernel_multistep_plif_forward"


@torch.library.custom_op(_PLIF_OP_NAME, mutates_args=())
def cupy_multistep_plif_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    reciprocal_tau: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: float,
    detach_reset: bool,
    sg_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sg = resolve_python_object(sg_id)
    captured_ctx = _CapturedAutogradCtx()
    out = _plif_forward(
        captured_ctx,
        x_seq,
        v_init,
        reciprocal_tau,
        decay_input,
        v_threshold,
        _decode_v_reset(v_reset),
        detach_reset,

        _resolve_sg_cuda_code_fun(sg),
    )
    capture_id = _stash_capture_ctx(captured_ctx)
    capture_token = torch.tensor(capture_id, device=x_seq.device, dtype=torch.int64)
    return (*out, capture_token)


@torch.library.register_fake(_PLIF_OP_NAME)
def _cupy_multistep_plif_forward_fake(*args):
    x_seq = args[0]
    return (x_seq.new_empty(x_seq.shape), x_seq.new_empty(x_seq.shape), x_seq.new_empty((), dtype=torch.int64))


def _setup_ctx(ctx, inputs, output):
    capture_token = output[-1]
    if capture_token.is_meta:
        ctx.captured = None
        return
    ctx.captured = _take_capture_ctx(int(capture_token.item()))


def _bw(ctx, *grad_outputs):
    if ctx.captured is None:
        raise RuntimeError("Missing captured context for backward.")
    grads = _plif_backward(ctx.captured, *grad_outputs[:-1])
    return grads[0], grads[1], grads[2], None, None, None, None, None


torch.library.register_autograd(_PLIF_OP_NAME, _bw, setup_context=_setup_ctx)


def multistep_plif_ptt(
    x_seq,
    v_init,
    reciprocal_tau,
    decay_input,
    v_threshold,
    v_reset,
    detach_reset,
    surrogate_function,
):
    sg_id = _sg_obj_id(surrogate_function)
    v_reset_value = float("nan") if v_reset is None else float(v_reset)
    return cupy_multistep_plif_forward(
        x_seq,
        v_init,
        reciprocal_tau,
        decay_input,
        v_threshold,
        v_reset_value,
        detach_reset,

        sg_id,
    )[:-1]
