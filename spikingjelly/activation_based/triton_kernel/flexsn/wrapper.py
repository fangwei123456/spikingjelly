import torch
from typing import Optional, Tuple

from torch import autograd

try:
    import triton
except BaseException as e:
    import logging
    from .. import dummy

    logging.info(f"spikingjelly.activation_based.triton_kernel.flexsn.wrapper: {e}")
    triton = dummy.DummyImport()

from ..triton_utils import type_dict, contiguous_and_device_guard
from ..triton_utils import amp_custom_fwd, amp_custom_bwd
from .info import FlexSNInfo


__all__ = [
    "FlexSNFunction",
    "flexsn_backward",
    "flexsn_backward_ncl_bucket",
    "flexsn_forward",
    "flexsn_inference",
    "flexsn_inference_final_state",
]


_BACKWARD_SMALL_MAX_NCL = 1 << 12
_BACKWARD_MEDIUM_MAX_NCL = 1 << 17
_BACKWARD_LARGE_MAX_NCL = 1 << 20
_BACKWARD_XLARGE_MAX_NCL = 1 << 23


def flexsn_backward_ncl_bucket(ncl: int) -> int:
    """Bucket a flattened sequence size for backward-kernel tuning.

    Chinese:
        将展平后的单步元素数 ``NCL`` 映射到 backward kernel 的调优分桶。

    English:
        Map the flattened per-step element count ``NCL`` to the backward-kernel
        autotuning bucket.

    :param ncl: EN: Flattened element count per time step. Chinese: 单个时间步展平后的元素数。
    :type ncl: int
    :return: EN: Bucket index in ``[0, 4]``. Chinese: ``[0, 4]`` 范围内的分桶索引。
    :rtype: int
    """
    if ncl <= _BACKWARD_SMALL_MAX_NCL:
        return 0
    if ncl <= _BACKWARD_MEDIUM_MAX_NCL:
        return 1
    if ncl <= _BACKWARD_LARGE_MAX_NCL:
        return 2
    if ncl <= _BACKWARD_XLARGE_MAX_NCL:
        return 3
    return 4


def _num_elements_per_step(x: torch.Tensor) -> int:
    n = 1
    for dim in x.shape[1:]:
        n *= dim
    return n


def _make_grid(ncl: int):
    def grid(meta):
        return (triton.cdiv(ncl, meta["BLOCK_NCL"]),)

    return grid


def _first_non_none_tensor(tensors):
    for tensor in tensors:
        if tensor is not None:
            return tensor
    return None


def _allocate_state_grad(
    i: int,
    T: int,
    state_templates: Optional[Tuple[torch.Tensor, ...]],
    grad_state_seq_examples,
    grad_example: torch.Tensor,
) -> torch.Tensor:
    if state_templates is not None:
        return (
            torch.zeros_like(state_templates[i])
            if T == 0
            else torch.empty_like(state_templates[i])
        )

    if i < len(grad_state_seq_examples) and grad_state_seq_examples[i] is not None:
        example = grad_state_seq_examples[i]
        return (
            example.new_zeros(example.shape[1:])
            if T == 0
            else example.new_empty(example.shape[1:])
        )

    return (
        grad_example.new_zeros(grad_example.shape[1:])
        if T == 0
        else grad_example.new_empty(grad_example.shape[1:])
    )


def flexsn_inference(f, info: FlexSNInfo, *args) -> tuple:
    """Run the inference kernel for a multi-step FlexSN core.

    Chinese:
        执行 FlexSN 多步推理 kernel。

    English:
        Execute the FlexSN multi-step inference kernel.

    :param f: EN: Triton kernel callable. Chinese: Triton kernel 可调用对象。
    :param info: EN: FlexSN metadata. Chinese: FlexSN 元信息。
    :param args: EN: Input/state sequences accepted by the kernel. Chinese: kernel 接收的输入/状态序列。
    :return: EN: Output/state sequences. When ``T == 0``, returns empty tensors with the expected templates. Chinese: 输出/状态序列；当 ``T == 0`` 时, 返回符合模板的空张量。
    :rtype: tuple
    """
    x_example = args[0]
    T = x_example.shape[0]
    NCL = _num_elements_per_step(x_example)
    dtype = x_example.dtype
    outputs = [
        torch.empty_like(x_example) for _ in range(info.num_outputs + info.num_states)
    ]
    if T == 0:
        return tuple(outputs)
    grid = _make_grid(NCL)

    f[grid](
        *args,
        *outputs,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple(outputs)


def flexsn_inference_final_state(f, info: FlexSNInfo, *args) -> tuple:
    """Run the inference kernel and materialize final states.

    Chinese:
        执行带最终状态物化的 FlexSN 多步推理 kernel。

    English:
        Execute the FlexSN inference kernel and materialize final states.

    :param f: EN: Triton kernel callable. Chinese: Triton kernel 可调用对象。
    :param info: EN: FlexSN metadata. Chinese: FlexSN 元信息。
    :param args: EN: Input/state sequences accepted by the kernel. Chinese: kernel 接收的输入/状态序列。
    :return: EN: Output sequences followed by final states. When ``T == 0``, output sequences are empty, provided initial states are cloned, and missing states are zero-filled. Chinese: 输出序列后接最终状态；当 ``T == 0`` 时, 输出序列为空, 已提供的初始状态会被克隆, 缺失状态会以零填充。
    :rtype: tuple
    """
    x_example = args[0]
    T = x_example.shape[0]
    NCL = _num_elements_per_step(x_example)
    dtype = x_example.dtype
    output_seqs = [torch.empty_like(x_example) for _ in range(info.num_outputs)]
    init_states = args[info.num_inputs : info.num_inputs + info.num_states]
    final_states = [
        init_states[i].new_empty(init_states[i].shape)
        if i < len(init_states)
        else x_example.new_empty(x_example.shape[1:])
        for i in range(info.num_states)
    ]
    if T == 0:
        final_states = [
            (
                init_states[i].clone()
                if i < len(init_states)
                else x_example.new_zeros(x_example.shape[1:])
            )
            for i in range(info.num_states)
        ]
        return tuple([*output_seqs, *final_states])
    grid = _make_grid(NCL)

    f[grid](
        *args,
        *output_seqs,
        *final_states,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple([*output_seqs, *final_states])


def flexsn_forward(f, info: FlexSNInfo, *args) -> tuple:
    """Run the training forward kernel for FlexSN.

    Chinese:
        执行 FlexSN 训练前向 kernel。

    English:
        Execute the FlexSN training forward kernel.

    :param f: EN: Triton kernel callable. Chinese: Triton kernel 可调用对象。
    :param info: EN: FlexSN metadata. Chinese: FlexSN 元信息。
    :param args: EN: Input/state sequences accepted by the kernel. Chinese: kernel 接收的输入/状态序列。
    :return: EN: Forward outputs plus any saved tensors required by backward. When ``T == 0``, returns empty tensors following the expected templates. Chinese: 前向输出以及 backward 所需的保存张量；当 ``T == 0`` 时, 返回符合模板的空张量。
    :rtype: tuple
    """
    x_example = args[0]
    T = x_example.shape[0]
    NCL = _num_elements_per_step(x_example)
    returns = [torch.empty_like(x_example) for _ in range(info.num_fwd_kernel_returns)]
    dtype = x_example.dtype
    if T == 0:
        return tuple(returns)
    grid = _make_grid(NCL)

    f[grid](
        *args,
        *returns,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple(returns)


def flexsn_backward(
    f,
    info: FlexSNInfo,
    *args,
    input_templates: Optional[Tuple[torch.Tensor, ...]] = None,
    state_templates: Optional[Tuple[torch.Tensor, ...]] = None,
) -> tuple:
    """Run the training backward kernel for FlexSN.

    Chinese:
        执行 FlexSN 训练反向 kernel。

    English:
        Execute the FlexSN training backward kernel.

    :param f: EN: Triton kernel callable. Chinese: Triton kernel 可调用对象。
    :param info: EN: FlexSN metadata. Chinese: FlexSN 元信息。
    :param args: EN: Gradients followed by any saved tensors accepted by the kernel.
        The leading ``info.num_outputs + info.num_states`` entries correspond to
        output/state-sequence gradients, and the remaining entries are saved tensors
        from the forward pass. Chinese: kernel 接收的梯度与保存张量。前
        ``info.num_outputs + info.num_states`` 个参数对应输出/状态序列梯度, 其余参数
        为前向保存张量。
    :type args: tuple
    :param input_templates: EN: Per-input-sequence templates used to allocate input
        gradients. They are required when all incoming gradients are ``None`` and
        ensure that state-only cores still return correctly shaped input-sequence
        gradients. When omitted, the fallback allocation path is only valid for
        single-input cores whose output-sequence gradients are present, because
        it infers the input gradient shape from the first non-``None``
        output-sequence gradient. Chinese: 每个输入序列的模板, 用于分配输入梯度。
        当所有传入梯度都为 ``None`` 时必须提供, 并确保仅有状态输出的 core 仍返回
        形状正确的输入梯度。若省略该参数, 当前仅支持单输入且存在输出序列梯度的 core,
        因为回退路径会从第一个非 ``None`` 的输出序列梯度推断输入梯度形状。
    :type input_templates: Optional[Tuple[torch.Tensor, ...]]
    :param state_templates: EN: Per-initial-state templates used to allocate initial
        state gradients. When provided, the returned state gradients preserve the
        original initial-state shapes instead of inferring them from state-sequence
        gradients. Chinese: 每个初始状态的模板, 用于分配初始状态梯度。提供后, 返回的
        状态梯度会保持初始状态原始形状, 而不是从状态序列梯度中推断。
    :type state_templates: Optional[Tuple[torch.Tensor, ...]]
    :return: EN: Gradients for inputs and initial states. When ``T == 0`` or all
        incoming gradients are ``None``, returns zero-filled gradients that follow
        the provided templates. Chinese: 输入与初始状态的梯度；当 ``T == 0`` 或所有传入
        梯度都为 ``None`` 时, 返回符合模板的零梯度。
    :rtype: tuple
    """
    required_grad_count = info.num_outputs + info.num_states
    grad_output_args = args[:required_grad_count]
    grad_output_example = _first_non_none_tensor(grad_output_args[: info.num_outputs])
    grad_example = _first_non_none_tensor(grad_output_args)
    if input_templates is None:
        if grad_example is None:
            raise ValueError(
                "input_templates are required when all incoming FlexSN gradients are None"
            )
        if info.num_inputs != 1:
            raise ValueError(
                "input_templates are required when FlexSN has multiple input sequences"
            )
        if grad_output_example is None:
            raise ValueError(
                "input_templates are required when FlexSN output-sequence gradients "
                "are all None"
            )
        input_templates = tuple(grad_output_example for _ in range(info.num_inputs))
    if len(input_templates) != info.num_inputs:
        raise ValueError(
            "input_templates must provide one template per FlexSN input sequence"
        )
    if state_templates is not None and len(state_templates) != info.num_states:
        raise ValueError(
            "state_templates must provide one template per FlexSN initial state"
        )
    if grad_example is None:
        if state_templates is None and info.num_states > 0:
            raise ValueError(
                "state_templates are required when all incoming FlexSN gradients are None"
            )
        grad_inputs = [torch.zeros_like(template) for template in input_templates]
        if state_templates is not None:
            grad_inputs.extend(torch.zeros_like(template) for template in state_templates)
        return tuple(grad_inputs)
    T = grad_example.shape[0]
    NCL = _num_elements_per_step(grad_example)
    grad_inputs = [
        (
            torch.zeros_like(input_templates[i])
            if T == 0
            else torch.empty_like(input_templates[i])
        )
        for i in range(info.num_inputs)
    ]
    grad_state_seq_examples = grad_output_args[
        info.num_outputs : info.num_outputs + info.num_states
    ]
    if state_templates is None and any(
        grad is None for grad in grad_state_seq_examples
    ):
        raise ValueError(
            "state_templates are required when any incoming FlexSN "
            "state-sequence gradient is None"
        )
    grad_kernel_args = [
        grad if grad is not None else torch.zeros_like(grad_example)
        for grad in grad_output_args
    ]
    # State-sequence gradients include the leading time dimension. The wrapper
    # returns gradients for the initial states, so their templates are shape[1:].
    grad_inputs += [
        _allocate_state_grad(
            i,
            T,
            state_templates,
            grad_state_seq_examples,
            grad_example,
        )
        for i in range(info.num_states)
    ]
    dtype = grad_example.dtype
    if T == 0:
        return tuple(grad_inputs)
    grid = _make_grid(NCL)

    f[grid](
        *grad_kernel_args,
        *args[required_grad_count:],
        *grad_inputs,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple(grad_inputs)


class FlexSNFunction(autograd.Function):
    """Autograd bridge between FlexSN Python code and Triton kernels.

    Chinese:
        连接 FlexSN Python 逻辑与 Triton kernel 的 autograd 桥接类。

    English:
        Autograd bridge between FlexSN Python logic and Triton kernels.
    """
    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_fwd
    def forward(
        ctx,
        fn_inf,
        fn_fwd,
        fn_bwd,
        info: FlexSNInfo,
        *args,  # len = num_inputs; including initial states
    ):
        if any(ctx.needs_input_grad):
            results = flexsn_forward(fn_fwd, info, *args)
            outputs_states = results[: info.num_outputs + info.num_states]
            to_save = []
            for i in info.c2k_return_mapping:
                to_save.append(results[i])
            ctx.save_for_backward(*to_save)
            ctx.fn_bwd = fn_bwd
            ctx.info = info
            ctx.input_template_specs = tuple(
                (tuple(arg.shape), arg.dtype, arg.device)
                for arg in args[: info.num_inputs]
            )
            ctx.state_template_specs = tuple(
                (tuple(arg.shape), arg.dtype, arg.device)
                for arg in args[info.num_inputs : info.num_inputs + info.num_states]
            )
        else:
            outputs_states = flexsn_inference(fn_inf, info, *args)
        if len(outputs_states) == 1:
            return outputs_states[0]
        else:
            return outputs_states

    @staticmethod
    @contiguous_and_device_guard
    @amp_custom_bwd
    def backward(ctx, *args):  # len(args) = num_outputs + num_states
        required_results = ctx.saved_tensors
        fn_bwd = ctx.fn_bwd
        input_templates = tuple(
            torch.empty(shape, dtype=dtype, device=device)
            for shape, dtype, device in getattr(ctx, "input_template_specs", ())
        )
        state_templates = tuple(
            torch.empty(shape, dtype=dtype, device=device)
            for shape, dtype, device in getattr(ctx, "state_template_specs", ())
        )
        grads = flexsn_backward(
            fn_bwd,
            ctx.info,
            *args,
            *required_results,
            input_templates=input_templates,
            state_templates=state_templates,
        )
        return None, None, None, None, *grads
