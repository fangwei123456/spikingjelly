from typing import Callable, Optional

import numpy as np
import torch

from spikingjelly import configure

from .. import cuda_runtime as cuda_utils
from ..cuda_codegen.autograd import (
    _capture_token,
    _CapturedAutogradCtx,
    _decode_v_reset,
    _setup_capture_ctx,
    _surrogate_cuda_dtype,
    cupy,
)
from ..cuda_codegen.neuron_multi import _aligned_v_v_seq
from ..cuda_strides import _launch_strided
from ..cuda_surrogate import _cuda_codes_callable, _surrogate_cuda_code
from ..layout import _empty_like
from ..spike_compress import cache as tensor_cache


def _create_fptt_kernel(hard_reset: bool, dtype: str):
    kernel_name = f"IzhikevichNode_fptt_{'hard' if hard_reset else 'soft'}Reset_{dtype}"

    if dtype == "fp32":
        code = rf"""
        extern "C" __global__
        void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* w_w_seq, float* spike_seq,
        const float & reciprocal_tau,
        const float & a0,
        const float & v_c,
        const float & v_threshold,
        const float & v_rest,
        const float & reciprocal_tau_w,
        const float & a,
        const float & b, {"const float & v_reset," if hard_reset else ""}
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
                h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] + a0 * (v_v_seq[t] - v_rest) * (v_v_seq[t] - v_c) - w_w_seq[t]);
                const float z = w_w_seq[t] + reciprocal_tau_w * (a * (h_seq[t] - v_rest) - w_w_seq[t]);
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
                w_w_seq[t + dt] = z + b * spike_seq[t];
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
    kernel_name = f"IzhikevichNode_bptt_{'hard' if hard_reset else 'soft'}Reset_{'detachReset' if detach_reset else ''}_{dtype}"

    surrogate_dtype = _surrogate_cuda_dtype(dtype)
    code_grad_s_to_h = sg_cuda_codes_fun(
        y=f"{surrogate_dtype} grad_s_to_h", x="over_th", dtype=surrogate_dtype
    )

    if dtype == "fp32":
        code = rf"""
        extern "C" __global__
        void {kernel_name}(
        const float* grad_spike_seq, const float* grad_v_seq,
        const float* grad_w_seq, const float* h_seq,
        const float* spike_seq, const float* v_v_seq,
        float* grad_x_seq, float* grad_v_init, float* grad_w_init,
        const float & reciprocal_tau, const float & one_sub_reciprocal_tau_w,
        const float & a_over_tau_w, const float & a0_over_tau,
        const float & b, const float & neg_sum_v_rest_v_c,
        const float & v_threshold, {"const float & v_reset," if hard_reset else ""}
        const int & neuron_num, const int & numel)
        """

        code += r"""
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < neuron_num)
            {
                float grad_h = 0.0f;  // grad_h will be used recursively
                float grad_w = 0.0f;  // grad_w will be used recursively
                for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                {
                    const int t = index + mem_offset;
                    const float over_th = h_seq[t] - v_threshold;
        """
        code += code_grad_s_to_h
        if detach_reset:
            if hard_reset:
                code_grad_v_to_h = r"""
                const float grad_v_to_h = 1.0f - spike_seq[t] + v_reset * grad_s_to_h;
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
            grad_w += grad_w_seq[t];
            grad_h = grad_w * (a_over_tau_w + b * grad_s_to_h) + ((1 + a0_over_tau * (2.0f * v_v_seq[t + neuron_num] + neg_sum_v_rest_v_c)) * grad_h + grad_v_seq[t]) * grad_v_to_h + grad_spike_seq[t] * grad_s_to_h;
            grad_x_seq[t] = grad_h * reciprocal_tau;
            grad_w = -reciprocal_tau * grad_h + one_sub_reciprocal_tau_w * grad_w;
            }
        grad_v_init[index] = grad_h * (1.0f + a0_over_tau * (2.0f * v_v_seq[index] + neg_sum_v_rest_v_c));
        grad_w_init[index] = grad_w;

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


def _iz_forward(
    ctx,
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    w_init: torch.Tensor,
    tau: float,
    v_threshold: float,
    v_reset: float,
    v_rest: float,
    a: float,
    b: float,
    tau_w: float,
    v_c: float,
    a0: float,
    detach_reset: bool,
    sg_cuda_codes_fun,
):
    requires_grad = x_seq.requires_grad or v_init.requires_grad or w_init.requires_grad
    device = x_seq.get_device()
    if x_seq.dtype == torch.float32:
        dtype = "fp32"
        cp_dtype = np.float32
    else:
        raise NotImplementedError

    h_seq = _empty_like(x_seq)
    spike_seq = _empty_like(x_seq)
    v_v_seq = _aligned_v_v_seq(x_seq)
    w_w_seq = _aligned_v_v_seq(x_seq)

    with cuda_utils.DeviceEnvironment(device):
        numel = x_seq.numel()
        neuron_num = numel // x_seq.shape[0]

        threads = configure.cuda_threads

        blocks = cuda_utils.cal_blocks(neuron_num)

        cp_numel = cupy.asarray(numel)
        cp_neuron_num = cupy.asarray(neuron_num)
        cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
        cp_v_rest = cupy.asarray(v_rest, dtype=cp_dtype)
        cp_v_c = cupy.asarray(v_c, dtype=cp_dtype)
        cp_a0 = cupy.asarray(a0, dtype=cp_dtype)
        cp_a = cupy.asarray(a, dtype=cp_dtype)
        cp_b = cupy.asarray(b, dtype=cp_dtype)
        cp_reciprocal_tau = cupy.asarray(1.0 / tau, dtype=cp_dtype)
        cp_reciprocal_tau_w = cupy.asarray(1.0 / tau_w, dtype=cp_dtype)
        cp_a0_over_tau = cupy.asarray(a0 / tau, dtype=cp_dtype)
        cp_a_over_tau_w = cupy.asarray(a / tau_w, dtype=cp_dtype)
        cp_one_sub_reciprocal_tau_w = cupy.asarray(1.0 - 1.0 / tau_w, dtype=cp_dtype)
        cp_neg_sum_v_rest_v_c = cupy.asarray(-v_rest - v_c, dtype=cp_dtype)

        hard_reset = v_reset is not None
        cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype) if hard_reset else None
        kernel_args = {
            "x_seq": x_seq,
            "v_v_seq": v_v_seq,
            "h_seq": h_seq,
            "w_w_seq": w_w_seq,
            "spike_seq": spike_seq,
            "reciprocal_tau": cp_reciprocal_tau,
            "a0": cp_a0,
            "v_c": cp_v_c,
            "v_threshold": cp_v_threshold,
            "v_rest": cp_v_rest,
            "reciprocal_tau_w": cp_reciprocal_tau_w,
            "a": cp_a,
            "b": cp_b,
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
            initial_states={"v_init": v_init, "w_init": w_init},
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
        ctx.cp_reciprocal_tau = cp_reciprocal_tau
        ctx.cp_one_sub_reciprocal_tau_w = cp_one_sub_reciprocal_tau_w
        ctx.cp_a_over_tau_w = cp_a_over_tau_w
        ctx.cp_a0_over_tau = cp_a0_over_tau
        ctx.cp_b = cp_b
        ctx.cp_neg_sum_v_rest_v_c = cp_neg_sum_v_rest_v_c
        ctx.cp_v_threshold = cp_v_threshold
        ctx.cp_v_reset = cp_v_reset
        ctx.detach_reset = detach_reset
        ctx.sg_cuda_codes_fun = sg_cuda_codes_fun

    return spike_seq, v_v_seq[1:,], w_w_seq[1:,]


def _iz_backward(ctx, grad_spike_seq, grad_v_seq, grad_w_seq):
    device = grad_spike_seq.get_device()
    if configure.save_spike_as_bool_in_neuron_kernel:
        spike_seq = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
        h_seq, v_v_seq = ctx.saved_tensors
    else:
        h_seq, spike_seq, v_v_seq = ctx.saved_tensors
    grad_x_seq = _empty_like(h_seq)
    grad_v_init = _empty_like(grad_spike_seq[0], sequence=False)
    grad_w_init = _empty_like(grad_spike_seq[0], sequence=False)

    hard_reset = ctx.cp_v_reset is not None

    if grad_spike_seq.dtype == torch.float32:
        dtype = "fp32"
    else:
        raise NotImplementedError

    kernel = _create_bptt_kernel(
        ctx.sg_cuda_codes_fun, hard_reset, ctx.detach_reset, dtype
    )

    with cuda_utils.DeviceEnvironment(device):
        kernel_args = {
            "grad_spike_seq": grad_spike_seq,
            "grad_v_seq": grad_v_seq,
            "grad_w_seq": grad_w_seq,
            "h_seq": h_seq,
            "spike_seq": spike_seq,
            "v_v_seq": v_v_seq,
            "grad_x_seq": grad_x_seq,
            "grad_v_init": grad_v_init,
            "grad_w_init": grad_w_init,
            "reciprocal_tau": ctx.cp_reciprocal_tau,
            "one_sub_reciprocal_tau_w": ctx.cp_one_sub_reciprocal_tau_w,
            "a_over_tau_w": ctx.cp_a_over_tau_w,
            "a0_over_tau": ctx.cp_a0_over_tau,
            "b": ctx.cp_b,
            "neg_sum_v_rest_v_c": ctx.cp_neg_sum_v_rest_v_c,
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
        grad_w_init,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


@torch.library.custom_op(
    "sj_izhikevich::cupy_multistep_izhikevich_forward", mutates_args=()
)
def cupy_multistep_izhikevich_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    w_init: torch.Tensor,
    tau: float,
    v_threshold: float,
    v_reset: float,
    v_rest: float,
    a: float,
    b: float,
    tau_w: float,
    v_c: float,
    a0: float,
    detach_reset: bool,
    surrogate_code: str,
    capture_context: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    captured_ctx = _CapturedAutogradCtx()
    out = _iz_forward(
        captured_ctx,
        x_seq,
        v_init,
        w_init,
        tau,
        v_threshold,
        _decode_v_reset(v_reset),
        v_rest,
        a,
        b,
        tau_w,
        v_c,
        a0,
        detach_reset,
        _cuda_codes_callable(surrogate_code),
    )
    return (
        *out,
        _capture_token(
            captured_ctx,
            (x_seq, v_init, w_init),
            capture_context,
        ),
    )


@torch.library.register_fake("sj_izhikevich::cupy_multistep_izhikevich_forward")
def _cupy_multistep_izhikevich_forward_fake(*args):
    x_seq = args[0]
    return (
        _empty_like(x_seq),
        _aligned_v_v_seq(x_seq)[1:],
        _aligned_v_v_seq(x_seq)[1:],
        torch.empty((), dtype=torch.int64),
    )


def _bw(ctx, *grad_outputs):
    if ctx.captured is None:
        raise RuntimeError("Missing captured context for backward.")
    grads = _iz_backward(ctx.captured, *grad_outputs[:-1])
    return (
        grads[0],
        grads[1],
        grads[2],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "sj_izhikevich::cupy_multistep_izhikevich_forward",
    _bw,
    setup_context=_setup_capture_ctx,
)


def izhikevich_multi_step(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    w_init: torch.Tensor,
    tau: float,
    v_threshold: float,
    v_reset: Optional[float],
    v_rest: float,
    a: float,
    b: float,
    tau_w: float,
    v_c: float,
    a0: float,
    detach_reset: bool,
    surrogate_function: Callable,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - :ref:`中文 <izhikevich-multi-step-cn>` | :ref:`English <izhikevich-multi-step-en>`

    ----

    .. _izhikevich-multi-step-cn:

    * **中文**

    使用 CuPy kernel 执行 Izhikevich 神经元的多步前向传播，并返回脉冲序列、膜电位
    序列和适应变量序列。输入应为 CUDA ``float32`` 张量，支持任意非负 stride；
    序列的第零维为时间维，其余维度与初始状态一致。

    :param x_seq: 输入序列，shape 为 ``[T, N]``
    :type x_seq: torch.Tensor
    :param v_init: 初始膜电位，shape 为 ``[N]``
    :type v_init: torch.Tensor
    :param w_init: 初始适应变量，shape 为 ``[N]``
    :type w_init: torch.Tensor
    :param tau: 膜电位时间常数
    :type tau: float
    :param v_threshold: 脉冲阈值
    :type v_threshold: float
    :param v_reset: 重置电压；``None`` 表示 soft reset
    :type v_reset: Optional[float]
    :param v_rest: 静息电位
    :type v_rest: float
    :param a: 适应变量恢复系数
    :type a: float
    :param b: 脉冲触发的适应变量增量
    :type b: float
    :param tau_w: 适应变量时间常数
    :type tau_w: float
    :param v_c: 临界电位
    :type v_c: float
    :param a0: 膜电位动力学二次项系数
    :type a0: float
    :param detach_reset: 是否分离 reset 分支中的 spike
    :type detach_reset: bool
    :param surrogate_function: 替代梯度函数
    :type surrogate_function: Callable
    :return: ``(spike_seq, v_seq, w_seq)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _izhikevich-multi-step-en:

    * **English**

    Run the multi-step Izhikevich neuron forward pass with the CuPy kernel and
    return spike, membrane-voltage, and adaptation-variable sequences. Inputs
    must be CUDA ``float32`` tensors with arbitrary nonnegative strides.
    Sequence dimension zero is time; other dimensions match the initial states.

    :param x_seq: Input sequence shaped ``[T, N]``
    :type x_seq: torch.Tensor
    :param v_init: Initial membrane voltage shaped ``[N]``
    :type v_init: torch.Tensor
    :param w_init: Initial adaptation variable shaped ``[N]``
    :type w_init: torch.Tensor
    :param tau: Membrane time constant
    :type tau: float
    :param v_threshold: Spike threshold
    :type v_threshold: float
    :param v_reset: Reset voltage; ``None`` means soft reset
    :type v_reset: Optional[float]
    :param v_rest: Resting voltage
    :type v_rest: float
    :param a: Adaptation recovery coefficient
    :type a: float
    :param b: Spike-triggered adaptation increment
    :type b: float
    :param tau_w: Adaptation time constant
    :type tau_w: float
    :param v_c: Critical voltage
    :type v_c: float
    :param a0: Quadratic coefficient in the membrane-voltage dynamics
    :type a0: float
    :param detach_reset: Whether to detach spike in the reset branch
    :type detach_reset: bool
    :param surrogate_function: Surrogate-gradient function
    :type surrogate_function: Callable
    :return: ``(spike_seq, v_seq, w_seq)``
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    surrogate_code = _surrogate_cuda_code(
        surrogate_function, _surrogate_cuda_dtype(x_seq.dtype)
    )
    v_reset_value = float("nan") if v_reset is None else float(v_reset)
    capture_context = torch.is_grad_enabled()
    return cupy_multistep_izhikevich_forward(
        x_seq,
        v_init,
        w_init,
        tau,
        v_threshold,
        v_reset_value,
        v_rest,
        a,
        b,
        tau_w,
        v_c,
        a0,
        detach_reset,
        surrogate_code,
        capture_context,
    )[:-1]
