import numpy as np
import torch

from ...._neuron_layout import _empty_like

from ..strides import _launch_strided
from .base import _aligned_v_v_seq
from ..... import configure
from ... import cuda_utils, tensor_cache
from ..surrogate_code import (
    _cuda_codes_callable,
    _surrogate_cuda_code,
)
from .runtime import (
    _CapturedAutogradCtx,
    _capture_token,
    _decode_v_reset,
    _setup_capture_ctx,
    _surrogate_cuda_dtype,
    cupy,
)


def _create_fptt_kernel(hard_reset: bool, dtype: str):
    kernel_name = f"QIFNode_fptt_{'hard' if hard_reset else 'soft'}Reset_{dtype}"

    if dtype == "fp32":
        code = rf"""
        extern "C" __global__
        void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
        const float & reciprocal_tau,
        const float & v_c,
        const float & a0,
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
                h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] + a0 * (v_v_seq[t] - v_rest) * (v_v_seq[t] - v_c));
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
        const half & v_c,
        const half & a0,
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
            const half2 v_c_half2 = __half2half2(v_c);
            const half2 a0_half2 = __half2half2(a0);
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
                h_seq[t] = __hfma2(__hfma2(__hmul2(__hsub2(v_v_seq[t], v_rest_half2), __hsub2(v_v_seq[t], v_c_half2)), a0_half2, x_seq[t]), reciprocal_tau_half2, v_v_seq[t]);

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


def _create_bptt_kernel(
    sg_cuda_codes_fun, hard_reset: bool, detach_reset: bool, dtype: str
):
    kernel_name = f"QIFNode_bptt_{'hard' if hard_reset else 'soft'}Reset_{'detachReset' if detach_reset else ''}_{dtype}"

    surrogate_dtype = _surrogate_cuda_dtype(dtype)
    code_grad_s_to_h = sg_cuda_codes_fun(
        y=f"{surrogate_dtype} grad_s_to_h", x="over_th", dtype=surrogate_dtype
    )

    if dtype == "fp32":
        code = rf"""
        extern "C" __global__
        void {kernel_name}(
        const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
        float* grad_x_seq, float* grad_v_init,
        const float & a0_over_tau, const float & neg_sum_v_rest_v_c, const float & reciprocal_tau,
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
            grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (1.0f + a0_over_tau * (2.0f * v_v_seq[t + neuron_num] + neg_sum_v_rest_v_c))) * grad_v_to_h;
            grad_x_seq[t] = grad_h * reciprocal_tau;
            }
        grad_v_init[index] = grad_h * (1.0f + a0_over_tau * (2.0f * v_v_seq[index] + neg_sum_v_rest_v_c));
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
        const half & a0_over_tau, const half & neg_sum_v_rest_v_c,
        const half & reciprocal_tau,
        const half & v_threshold, {"const half & v_reset," if hard_reset else ""}
        const int & neuron_num, const int & numel)
        """
        code += r"""
        {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = neuron_num >> 1;
        if (index < stride)
        {
            const half2 a0_over_tau_half2 = __half2half2(a0_over_tau);
            const half2 neg_sum_v_rest_v_c_half2 = __half2half2(neg_sum_v_rest_v_c);
            const half2 v_threshold_half2 = __half2half2(v_threshold);
            const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
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
                grad_h = __hfma2(__hfma2(__hfma2(__hfma2(__float2half2_rn(2.0f), v_v_seq[t + stride], neg_sum_v_rest_v_c_half2), a0_over_tau_half2, __float2half2_rn(1.0f)), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));

                grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
            }
        grad_v_init[index] = __hmul2(__hfma2(__hfma2(__float2half2_rn(2.0f), v_v_seq[index], neg_sum_v_rest_v_c_half2), a0_over_tau_half2, __float2half2_rn(1.0f)), grad_h);
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


def _qif_forward(
    ctx,
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    tau: float,
    v_threshold: float,
    v_reset: float,
    v_rest: float,
    v_c: float,
    a0: float,
    detach_reset: bool,
    sg_cuda_codes_fun,
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

    h_seq = _empty_like(x_seq)
    spike_seq = _empty_like(x_seq)
    v_v_seq = _aligned_v_v_seq(x_seq)

    with cuda_utils.DeviceEnvironment(device):
        numel = x_seq.numel()
        neuron_num = numel // x_seq.shape[0]

        threads = configure.cuda_threads
        if dtype == "fp16":
            neuron_num += neuron_num % 2
            numel = neuron_num * x_seq.shape[0]
            blocks = cuda_utils.cal_blocks(neuron_num >> 1)
            # we will take two neurons to calculate as one neuron in cuda half2
        else:
            blocks = cuda_utils.cal_blocks(neuron_num)

        cp_numel = cupy.asarray(numel)
        cp_neuron_num = cupy.asarray(neuron_num)
        cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
        cp_v_rest = cupy.asarray(v_rest, dtype=cp_dtype)
        cp_v_c = cupy.asarray(v_c, dtype=cp_dtype)
        cp_a0 = cupy.asarray(a0, dtype=cp_dtype)
        cp_reciprocal_tau = cupy.asarray(1.0 / tau, dtype=cp_dtype)
        cp_a0_over_tau = cupy.asarray(a0 / tau, dtype=cp_dtype)
        cp_neg_sum_v_rest_v_c = cupy.asarray(-v_rest - v_c, dtype=cp_dtype)

        hard_reset = v_reset is not None
        cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype) if hard_reset else None
        kernel_args = {
            "x_seq": x_seq,
            "v_v_seq": v_v_seq,
            "h_seq": h_seq,
            "spike_seq": spike_seq,
            "reciprocal_tau": cp_reciprocal_tau,
            "v_c": cp_v_c,
            "a0": cp_a0,
            "v_threshold": cp_v_threshold,
            "v_rest": cp_v_rest,
            "neuron_num": cp_neuron_num,
            "numel": cp_numel,
        }
        if hard_reset:
            kernel_args["v_reset"] = cp_v_reset

        kernel = _create_fptt_kernel(hard_reset, dtype)

        _launch_strided(
            kernel.code,
            kernel.name,
            (blocks,),
            (threads,),
            kernel_args,
            initial_states={"v_init": v_init},
        )

    if requires_grad:
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
        ctx.cp_a0_over_tau = cp_a0_over_tau
        ctx.cp_neg_sum_v_rest_v_c = cp_neg_sum_v_rest_v_c
        ctx.cp_reciprocal_tau = cp_reciprocal_tau
        ctx.cp_v_threshold = cp_v_threshold
        ctx.cp_v_reset = cp_v_reset
        ctx.detach_reset = detach_reset
        ctx.sg_cuda_codes_fun = sg_cuda_codes_fun

    return spike_seq, v_v_seq[1:,]


def _qif_backward(ctx, grad_spike_seq, grad_v_seq):
    device = grad_spike_seq.get_device()
    if configure.save_spike_as_bool_in_neuron_kernel:
        spike_seq = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
        h_seq, v_v_seq = ctx.saved_tensors
    else:
        h_seq, spike_seq, v_v_seq = ctx.saved_tensors
    grad_x_seq = _empty_like(h_seq)
    grad_v_init = _empty_like(grad_spike_seq[0], sequence=False)

    hard_reset = ctx.cp_v_reset is not None

    if grad_spike_seq.dtype == torch.float32:
        dtype = "fp32"
    elif grad_spike_seq.dtype == torch.float16:
        dtype = "fp16"
    else:
        raise NotImplementedError

    kernel = _create_bptt_kernel(
        ctx.sg_cuda_codes_fun, hard_reset, ctx.detach_reset, dtype
    )

    with cuda_utils.DeviceEnvironment(device):
        kernel_args = {
            "grad_spike_seq": grad_spike_seq,
            "grad_v_seq": grad_v_seq,
            "h_seq": h_seq,
            "spike_seq": spike_seq,
            "v_v_seq": v_v_seq,
            "grad_x_seq": grad_x_seq,
            "grad_v_init": grad_v_init,
            "a0_over_tau": ctx.cp_a0_over_tau,
            "neg_sum_v_rest_v_c": ctx.cp_neg_sum_v_rest_v_c,
            "reciprocal_tau": ctx.cp_reciprocal_tau,
            "v_threshold": ctx.cp_v_threshold,
            "neuron_num": ctx.cp_neuron_num,
            "numel": ctx.cp_numel,
        }
        if hard_reset:
            kernel_args["v_reset"] = ctx.cp_v_reset

        _launch_strided(
            kernel.code, kernel.name, (ctx.blocks,), (ctx.threads,), kernel_args
        )
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


@torch.library.custom_op("sj::cupy_multistep_qif_forward", mutates_args=())
def cupy_multistep_qif_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    tau: float,
    v_threshold: float,
    v_reset: float,
    v_rest: float,
    v_c: float,
    a0: float,
    detach_reset: bool,
    surrogate_code: str,
    capture_context: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    captured_ctx = _CapturedAutogradCtx()
    out = _qif_forward(
        captured_ctx,
        x_seq,
        v_init,
        tau,
        v_threshold,
        _decode_v_reset(v_reset),
        v_rest,
        v_c,
        a0,
        detach_reset,
        _cuda_codes_callable(surrogate_code),
    )
    return (
        *out,
        _capture_token(captured_ctx, (x_seq, v_init), capture_context),
    )


@torch.library.register_fake("sj::cupy_multistep_qif_forward")
def _cupy_multistep_qif_forward_fake(*args):
    x_seq = args[0]
    return (
        _empty_like(x_seq),
        _aligned_v_v_seq(x_seq)[1:],
        torch.empty((), dtype=torch.int64),
    )


def _bw(ctx, *grad_outputs):
    if ctx.captured is None:
        raise RuntimeError("Missing captured context for backward.")
    grads = _qif_backward(ctx.captured, *grad_outputs[:-1])
    return grads[0], grads[1], None, None, None, None, None, None, None, None, None


torch.library.register_autograd(
    "sj::cupy_multistep_qif_forward", _bw, setup_context=_setup_capture_ctx
)


def qif_multi_step(
    x_seq,
    v_init,
    tau,
    v_threshold,
    v_reset,
    v_rest,
    v_c,
    a0,
    detach_reset,
    surrogate_function,
):
    surrogate_code = _surrogate_cuda_code(
        surrogate_function, _surrogate_cuda_dtype(x_seq.dtype)
    )
    v_reset_value = float("nan") if v_reset is None else float(v_reset)
    capture_context = torch.is_grad_enabled()
    return cupy_multistep_qif_forward(
        x_seq,
        v_init,
        tau,
        v_threshold,
        v_reset_value,
        v_rest,
        v_c,
        a0,
        detach_reset,
        surrogate_code,
        capture_context,
    )[:-1]
