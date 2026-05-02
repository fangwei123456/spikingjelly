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

__all__ = [
    "create_fptt_kernel",
    "create_bptt_kernel",
    "multistep_if_ptt",
]


def create_fptt_kernel(hard_reset: bool, dtype: str):
    kernel_name = f"IFNode_fptt_{'hard' if hard_reset else 'soft'}Reset_{dtype}"

    if dtype == "fp32":
        code = rf"""
        extern "C" __global__
        void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
        const float & v_threshold, {"const float & v_reset," if hard_reset else ""}
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
                h_seq[t] = v_v_seq[t] + x_seq[t];
                if (h_seq[t] >= v_threshold)
        """

        if hard_reset:
            code += r"""
                {
                    spike_seq[t] = 1.0f;
                    v_v_seq[t + dt] = v_reset;
                }
            """
        else:
            code += r"""
                {
                    spike_seq[t] = 1.0f;
                    v_v_seq[t + dt] = h_seq[t] - v_threshold;
                }
            """

        code += r"""
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
        const half & v_threshold, {"const half & v_reset," if hard_reset else ""}
        const int & neuron_num, const int & numel)
        """

        code += r"""
        {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = neuron_num >> 1;
        if (index < stride)
        {
            const int numel_2 = numel >> 1;
            const half2 v_threshold_half2 = __half2half2(v_threshold);
        """

        if hard_reset:
            code += r"""
                const half2 v_reset_half2 = __half2half2(v_reset);
            """

        code += r"""
            for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
            {
                const int t = index + mem_offset;
                h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

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
    kernel_name = f"IFNode_bptt_{'hard' if hard_reset else 'soft'}Reset_{'detachReset' if detach_reset else ''}_{dtype}"

    code_grad_s_to_h = sg_cuda_code_fun(x="over_th", y="grad_s_to_h", dtype=dtype)

    if dtype == "fp32":
        code = rf"""
        extern "C" __global__
        void {kernel_name}(
        const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
        float* grad_x_seq, float* grad_v_init,
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
                // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                """
            else:
                code_grad_v_to_h = r"""
                const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                """

        code += code_grad_v_to_h
        code += r"""
            grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
            // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
            grad_x_seq[t] = grad_h;
            }
        grad_v_init[index] = grad_h;
        }
        }
        """

    elif dtype == "fp16":
        code = rf"""
        #include <cuda_fp16.h>
        extern "C" __global__
        void {kernel_name}(
        const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
        half2* grad_x_seq, half2* grad_v_init,
        const half & v_threshold, {"const half & v_reset," if hard_reset else ""}
        const int & neuron_num, const int & numel)
        """
        code += r"""
        {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = neuron_num >> 1;
        if (index < stride)
        {
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
                const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                """
            else:
                code_grad_v_to_h = r"""
                const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                """

        code += code_grad_v_to_h
        code += r"""
                grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                grad_x_seq[t] = grad_h;
                }
        grad_v_init[index] = grad_h;
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


def _if_forward_impl(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
):
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
        use_pad = True
        x_seq = F.pad(x_seq, (0, 1))
        v_init = F.pad(v_init, (0, 1))

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
        else:
            blocks = cuda_utils.cal_blocks(neuron_num)

        cp_numel = cupy.asarray(numel)
        cp_neuron_num = cupy.asarray(neuron_num)
        cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)

        if v_reset is None:
            cp_v_reset = None
            hard_reset = False
            (
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
                cp_v_threshold,
                cp_neuron_num,
                cp_numel,
            ) = cuda_utils.get_contiguous(
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
                cp_v_threshold,
                cp_neuron_num,
                cp_numel,
            )
            kernel_args = [
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
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
                cp_v_threshold,
                cp_v_reset,
                cp_neuron_num,
                cp_numel,
            ) = cuda_utils.get_contiguous(
                x_seq,
                v_v_seq,
                h_seq,
                spike_seq,
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
                cp_v_threshold,
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

    if use_pad:
        spike_out = spike_seq[..., :-1]
        v_out = v_v_seq[1:, ..., :-1]
    else:
        spike_out = spike_seq
        v_out = v_v_seq[1:,]

    return {
        "spike_seq": spike_out,
        "v_seq": v_out,
        "h_seq": h_seq,
        "spike_seq_full": spike_seq,
        "use_pad": use_pad,
        "blocks": blocks,
        "threads": threads,
        "cp_numel": cp_numel,
        "cp_neuron_num": cp_neuron_num,
        "cp_v_threshold": cp_v_threshold,
        "cp_v_reset": cp_v_reset,
    }


def _if_backward_impl(
    grad_spike_seq: torch.Tensor,
    grad_v_seq: torch.Tensor,
    *,
    use_pad: bool,
    blocks: int,
    threads: int,
    cp_numel,
    cp_neuron_num,
    cp_v_threshold,
    cp_v_reset,
    h_seq: torch.Tensor,
    spike_seq_saved: torch.Tensor,
    detach_reset: bool,
    sg_cuda_code_fun,
):
    if use_pad:
        grad_spike_seq = F.pad(grad_spike_seq, (0, 1))
        grad_v_seq = F.pad(grad_v_seq, (0, 1))

    device = grad_spike_seq.get_device()
    zero_shape = list(grad_spike_seq.shape)
    zero_shape[0] += 1
    zero_data = torch.zeros(
        zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype
    )
    grad_x_seq = zero_data[0:-1]
    grad_v_init = zero_data[-1]

    hard_reset = cp_v_reset is not None

    if grad_spike_seq.dtype == torch.float32:
        dtype = "fp32"
    elif grad_spike_seq.dtype == torch.float16:
        dtype = "fp16"
    else:
        raise NotImplementedError

    kernel = create_bptt_kernel(sg_cuda_code_fun, hard_reset, detach_reset, dtype)

    with cuda_utils.DeviceEnvironment(device):
        if hard_reset:
            (
                grad_spike_seq,
                grad_v_seq,
                h_seq,
                spike_seq_saved,
                grad_x_seq,
                grad_v_init,
                cp_v_threshold,
                cp_v_reset,
                cp_neuron_num,
                cp_numel,
            ) = cuda_utils.get_contiguous(
                grad_spike_seq,
                grad_v_seq,
                h_seq,
                spike_seq_saved,
                grad_x_seq,
                grad_v_init,
                cp_v_threshold,
                cp_v_reset,
                cp_neuron_num,
                cp_numel,
            )
            kernel_args = [
                grad_spike_seq,
                grad_v_seq,
                h_seq,
                spike_seq_saved,
                grad_x_seq,
                grad_v_init,
                cp_v_threshold,
                cp_v_reset,
                cp_neuron_num,
                cp_numel,
            ]
        else:
            (
                grad_spike_seq,
                grad_v_seq,
                h_seq,
                spike_seq_saved,
                grad_x_seq,
                grad_v_init,
                cp_v_threshold,
                cp_neuron_num,
                cp_numel,
            ) = cuda_utils.get_contiguous(
                grad_spike_seq,
                grad_v_seq,
                h_seq,
                spike_seq_saved,
                grad_x_seq,
                grad_v_init,
                cp_v_threshold,
                cp_neuron_num,
                cp_numel,
            )
            kernel_args = [
                grad_spike_seq,
                grad_v_seq,
                h_seq,
                spike_seq_saved,
                grad_x_seq,
                grad_v_init,
                cp_v_threshold,
                cp_neuron_num,
                cp_numel,
            ]

        kernel(
            (blocks,),
            (threads,),
            cuda_utils.wrap_args_to_raw_kernel(device, *kernel_args),
        )

    if use_pad:
        return grad_x_seq[..., :-1], grad_v_init[..., :-1]
    return grad_x_seq, grad_v_init


_IF_OP_NAME = "sj::cupy_neuron_kernel_multistep_if_forward"


@torch.library.custom_op(_IF_OP_NAME, mutates_args=())
def cupy_multistep_if_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    detach_reset: bool,
    sg_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del detach_reset
    _ = resolve_python_object(sg_id)
    out = _if_forward_impl(x_seq, v_init, v_threshold, _decode_v_reset(v_reset))
    captured_ctx = _CapturedAutogradCtx()
    captured_ctx.use_pad = out["use_pad"]
    captured_ctx.blocks = out["blocks"]
    captured_ctx.threads = out["threads"]
    captured_ctx.cp_numel = out["cp_numel"]
    captured_ctx.cp_neuron_num = out["cp_neuron_num"]
    captured_ctx.cp_v_threshold = out["cp_v_threshold"]
    captured_ctx.cp_v_reset = out["cp_v_reset"]
    if configure.save_spike_as_bool_in_neuron_kernel:
        captured_ctx.s_shape = out["spike_seq_full"].shape
        captured_ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(
            out["spike_seq_full"]
        )
        captured_ctx.saved_tensors = (out["h_seq"],)
    else:
        captured_ctx.saved_tensors = (out["h_seq"], out["spike_seq_full"])
    capture_id = _stash_capture_ctx(captured_ctx)
    capture_token = torch.tensor(capture_id, device=x_seq.device, dtype=torch.int64)
    return out["spike_seq"], out["v_seq"], capture_token


@torch.library.register_fake(_IF_OP_NAME)
def _cupy_multistep_if_forward_fake(
    x_seq, v_init, v_threshold, v_reset, detach_reset, sg_id
):
    return (
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty((), dtype=torch.int64),
    )


def _if_bw(ctx, grad_spike_seq, grad_v_seq, grad_capture_token):
    del grad_capture_token
    if ctx.captured is None:
        raise RuntimeError("Missing captured context for IF backward.")
    captured = ctx.captured
    if configure.save_spike_as_bool_in_neuron_kernel:
        h_seq = captured.saved_tensors[0]
        spike_seq_saved = tensor_cache.BOOL_TENSOR_CACHE.get_float(
            captured.s_tk, captured.s_shape
        )
    else:
        h_seq, spike_seq_saved = captured.saved_tensors

    # sg_id is not persisted by torch ctx in register_autograd, read from saved inputs
    # through closure in _setup_if_ctx by assigning explicitly.
    sg = ctx.sg
    grad_x, grad_v_init = _if_backward_impl(
        grad_spike_seq,
        grad_v_seq,
        use_pad=captured.use_pad,
        blocks=captured.blocks,
        threads=captured.threads,
        cp_numel=captured.cp_numel,
        cp_neuron_num=captured.cp_neuron_num,
        cp_v_threshold=captured.cp_v_threshold,
        cp_v_reset=captured.cp_v_reset,
        h_seq=h_seq,
        spike_seq_saved=spike_seq_saved,
        detach_reset=ctx.detach_reset,
        sg_cuda_code_fun=_resolve_sg_cuda_code_fun(sg),
    )
    return grad_x, grad_v_init, None, None, None, None


def _setup_if_ctx(ctx, inputs, output):
    capture_token = output[2]
    if capture_token.is_meta:
        ctx.captured = None
        return
    ctx.captured = _take_capture_ctx(int(capture_token.item()))
    ctx.detach_reset = inputs[4]
    ctx.sg = resolve_python_object(inputs[5])


torch.library.register_autograd(
    _IF_OP_NAME,
    _if_bw,
    setup_context=_setup_if_ctx,
)


def multistep_if_ptt(
    x_seq,
    v_init,
    v_threshold,
    v_reset,
    detach_reset,
    surrogate_function,
):
    sg_id = _sg_obj_id(surrogate_function)
    v_reset_value = float("nan") if v_reset is None else float(v_reset)
    spike_seq, v_seq, _ = cupy_multistep_if_forward(
        x_seq,
        v_init,
        v_threshold,
        v_reset_value,
        detach_reset,
        sg_id,
    )
    return spike_seq, v_seq
