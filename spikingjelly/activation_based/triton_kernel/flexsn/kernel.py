"""Build Triton scan kernels for FlexSN's triton / inductor backend.

Three entry points:
* build_inference_kernel  — no-grad fast path (make_fx, no PYTORCH_JIT=0 needed)
* build_inference_final_state_kernel — inference path that returns final states only
* build_training_kernels  — forward + backward kernels for full BPTT training
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch

__all__ = [
    "build_inference_kernel",
    "build_inference_final_state_kernel",
    "build_training_kernels",
]


def _example_build_device(example_inputs: Optional[Tuple[torch.Tensor, ...]]):
    if example_inputs is not None:
        for tensor in example_inputs:
            if tensor.device.type == "cuda":
                return tensor.device
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA available for building FlexSN Triton kernel; provide "
            "example_inputs on CUDA or enable CUDA."
        )
    return torch.device("cuda", torch.cuda.current_device())


def _prepare_example_inputs(
    example_inputs: Optional[Tuple[torch.Tensor, ...]],
    num_inputs: int,
    num_states: int,
) -> Tuple[torch.Tensor, ...]:
    expected = num_inputs + num_states
    if example_inputs is not None and len(example_inputs) != expected:
        raise ValueError(
            "example_inputs must have the same length as num_inputs + num_states "
            f"({expected}), but got {len(example_inputs)}."
        )
    build_device = _example_build_device(example_inputs)
    if example_inputs is None:
        return tuple(torch.zeros(1, device=build_device) for _ in range(expected))
    # .clone() breaks aliasing: in-place ops inside core_fn during tracing
    # must not silently mutate the caller's original buffers.
    return tuple(x.detach().to(build_device).clone() for x in example_inputs)


def build_inference_kernel(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    example_inputs: Optional[Tuple[torch.Tensor, ...]] = None,
):
    """
    **API Language:**
    :ref:`中文 <build_inference_kernel-cn>` | :ref:`English <build_inference_kernel-en>`

    ----

    .. _build_inference_kernel-cn:

    * **中文**

    为 ``core_fn`` 构建单次 scan 的推理 Triton kernel。
    构建出的 kernel 会把 ``core_fn`` 的单步计算包裹在 ``tl.static_range(T)``
    时间循环中，因此无论 ``T`` 多大，单次推理都只触发一次 kernel launch。

    :param core_fn: 单步动力学函数，签名应为
        ``(*inputs, *states) -> (*outputs, *updated_states)``
    :type core_fn: Callable
    :param num_inputs: 每个时间步输入张量的数量
    :type num_inputs: int
    :param num_states: 状态张量的数量
    :type num_states: int
    :param num_outputs: 每个时间步输出张量的数量
    :type num_outputs: int
    :param example_inputs: 可选的示例张量 ``[*inputs, *states]`` 。若为
        ``None`` ，则自动构造单位大小的 CUDA ``float32`` 张量
    :type example_inputs: Optional[Tuple[torch.Tensor, ...]]
    :return: ``(kernel, info)`` ，其中 ``kernel`` 为编译后的 Triton
        kernel， ``info`` 为调用
        :func:`spikingjelly.activation_based.triton_kernel.flexsn.wrapper.flexsn_inference`
        所需的 :class:`FlexSNInfo`
    :rtype: Tuple[object, FlexSNInfo]

    ----

    .. _build_inference_kernel-en:

    * **English**

    Build a single-pass scan inference Triton kernel for ``core_fn``.
    The generated kernel wraps ``core_fn``'s per-step computation in a
    ``tl.static_range(T)`` loop, so one inference call launches exactly one
    kernel regardless of ``T``.

    :param core_fn: Single-step dynamics callable with signature
        ``(*inputs, *states) -> (*outputs, *updated_states)``
    :type core_fn: Callable
    :param num_inputs: Number of per-step input tensors
    :type num_inputs: int
    :param num_states: Number of state tensors
    :type num_states: int
    :param num_outputs: Number of per-step output tensors
    :type num_outputs: int
    :param example_inputs: Optional example tensors ``[*inputs, *states]``.
        If ``None``, unit-sized CUDA ``float32`` tensors are created
    :type example_inputs: Optional[Tuple[torch.Tensor, ...]]
    :return: ``(kernel, info)`` where ``kernel`` is the compiled Triton
        kernel and ``info`` is the :class:`FlexSNInfo` metadata required by
        :func:`spikingjelly.activation_based.triton_kernel.flexsn.wrapper.flexsn_inference`
    :rtype: Tuple[object, FlexSNInfo]
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    from ..torch2triton import generate_triton_code_str
    from .info import extract_info
    from .template import get_flexsn_inference_kernel

    example_inputs = _prepare_example_inputs(example_inputs, num_inputs, num_states)

    # Trace core_fn to an aten-level FX graph (no PYTORCH_JIT=0 needed).
    traced = make_fx(core_fn)(*example_inputs)
    graph = traced.graph

    # Generate Triton function source for the per-step body.
    # Sanitize name: lambdas → "<lambda>", functools.partial → no __name__.
    # Replace every non-alphanumeric character (including < > space) with "_".
    raw_name = getattr(core_fn, "__name__", type(core_fn).__name__)
    safe_name = "".join(c if c.isalnum() else "_" for c in raw_name)
    core_name = f"{safe_name}_inductor_scan"
    core_str, core_name = generate_triton_code_str(graph, core_name)

    # Extract metadata: arg/return names, output/state counts.
    info = extract_info(graph, num_inputs, num_states, num_outputs)

    # Build the scan kernel: single tl.static_range(T) loop.
    kernel = get_flexsn_inference_kernel(core_str, core_name, info=info)

    return kernel, info


def build_inference_final_state_kernel(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    example_inputs: Optional[Tuple[torch.Tensor, ...]] = None,
):
    """
    **API Language:**
    :ref:`中文 <build_inference_final_state_kernel-cn>` | :ref:`English <build_inference_final_state_kernel-en>`

    ----

    .. _build_inference_final_state_kernel-cn:

    * **中文**

    为 ``core_fn`` 构建返回输出序列与最终状态的推理 Triton kernel。
    该变体与 :func:`build_inference_kernel` 一样会追踪 ``core_fn`` 并生成
    scan kernel，但它只返回最终状态张量，而不是完整状态序列。

    :param core_fn: 单步动力学函数，签名应为
        ``(*inputs, *states) -> (*outputs, *updated_states)``
    :type core_fn: Callable
    :param num_inputs: 每个时间步输入张量的数量
    :type num_inputs: int
    :param num_states: 状态张量的数量
    :type num_states: int
    :param num_outputs: 每个时间步输出张量的数量
    :type num_outputs: int
    :param example_inputs: 可选的示例张量 ``[*inputs, *states]``
    :type example_inputs: Optional[Tuple[torch.Tensor, ...]]
    :return: ``(kernel, info)`` ，分别来自
        ``get_flexsn_inference_final_state_kernel`` 和 ``extract_info``
    :rtype: Tuple[object, FlexSNInfo]

    ----

    .. _build_inference_final_state_kernel-en:

    * **English**

    Build an inference Triton kernel that returns output sequences and final
    states for ``core_fn``. This variant traces ``core_fn`` like
    :func:`build_inference_kernel`, but materializes final state tensors
    instead of full state sequences.

    :param core_fn: Single-step dynamics callable with signature
        ``(*inputs, *states) -> (*outputs, *updated_states)``
    :type core_fn: Callable
    :param num_inputs: Number of per-step input tensors
    :type num_inputs: int
    :param num_states: Number of state tensors
    :type num_states: int
    :param num_outputs: Number of per-step output tensors
    :type num_outputs: int
    :param example_inputs: Optional example tensors ``[*inputs, *states]``
    :type example_inputs: Optional[Tuple[torch.Tensor, ...]]
    :return: ``(kernel, info)`` produced by
        ``get_flexsn_inference_final_state_kernel`` and ``extract_info``
    :rtype: Tuple[object, FlexSNInfo]
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    from ..torch2triton import generate_triton_code_str
    from .info import extract_info
    from .template import get_flexsn_inference_final_state_kernel

    example_inputs = _prepare_example_inputs(example_inputs, num_inputs, num_states)

    traced = make_fx(core_fn)(*example_inputs)
    graph = traced.graph

    raw_name = getattr(core_fn, "__name__", type(core_fn).__name__)
    safe_name = "".join(c if c.isalnum() else "_" for c in raw_name)
    core_name = f"{safe_name}_inductor_scan_final_state"
    core_str, core_name = generate_triton_code_str(graph, core_name)
    info = extract_info(graph, num_inputs, num_states, num_outputs)
    kernel = get_flexsn_inference_final_state_kernel(core_str, core_name, info=info)
    return kernel, info


def _diff_mask(
    core_fn: Callable,
    num_outputs: int,
    num_states: int,
    example_inputs: Tuple[torch.Tensor, ...],
    requires_grad_mask: Tuple[bool, ...],
) -> List[bool]:
    """Return which of core_fn's outputs are differentiable."""
    # Only float/complex tensors support requires_grad; skip int/bool inputs.
    ex = []
    for i, t in enumerate(example_inputs):
        probe = t.clone().detach()
        if requires_grad_mask[i] and (probe.is_floating_point() or probe.is_complex()):
            probe.requires_grad_(True)
        ex.append(probe)
    with torch.enable_grad():
        outs = core_fn(*ex)
    if isinstance(outs, torch.Tensor):
        outs = (outs,)
    return [
        isinstance(o, torch.Tensor) and (o.requires_grad or o.grad_fn is not None)
        for o in outs
    ]


def _make_bwd_shim(
    bwd_fn_name: str,
    n_saved: int,
    num_outputs: int,
    num_states: int,
    diff_mask: List[bool],
) -> Tuple[str, str]:
    """Wrap the AOT backward function to match the template's calling convention.

    The template calls bwd_fn(saved..., grad_out0..., grad_state0...) for ALL
    outputs and states.  AOT autograd drops gradient arguments for non-
    differentiable outputs (e.g. spike signals from hard threshold), so the
    actual backward function may accept fewer arguments.  This shim accepts
    the full template signature and forwards only the used arguments.
    """
    all_grads = ["gs_" + str(i) for i in range(num_outputs)] + [
        "gv_" + str(i) for i in range(num_states)
    ]
    if len(diff_mask) != len(all_grads):
        raise ValueError(
            f"diff_mask length {len(diff_mask)} != "
            f"num_outputs+num_states {len(all_grads)}"
        )
    shim_args = ["sv_" + str(i) for i in range(n_saved)] + all_grads
    fwd_call = ["sv_" + str(i) for i in range(n_saved)] + [
        name for name, d in zip(all_grads, diff_mask) if d
    ]
    shim_name = bwd_fn_name + "_shim"
    shim_code = (
        "\n@triton.jit\ndef " + shim_name + "(" + ", ".join(shim_args) + "):\n"
        "    return " + bwd_fn_name + "(" + ", ".join(fwd_call) + ")\n"
    )
    return shim_code, shim_name


def build_training_kernels(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    example_inputs: Optional[Tuple[torch.Tensor, ...]] = None,
    requires_grad: Optional[Tuple[bool, ...]] = None,
):
    """
    **API Language:**
    :ref:`中文 <build_training_kernels-cn>` | :ref:`English <build_training_kernels-en>`

    ----

    .. _build_training_kernels-cn:

    * **中文**

    为 BPTT 训练构建 FlexSN 的前向与反向 Triton scan kernel。
    该函数会使用 ``aot_function`` 追踪 ``core_fn`` 的正向与反向图，并生成：

    * 保存反向所需中间量的前向 scan kernel
    * 执行逆时间反向传播的 backward scan kernel

    若某些输出（例如硬阈值脉冲）不可微，AOT backward 会省略对应梯度参数，
    此函数会自动生成 shim 以对齐 kernel template 的调用约定。

    :param core_fn: 单步动力学函数，签名应为
        ``(*inputs, *states) -> (*outputs, *updated_states)``
    :type core_fn: Callable
    :param num_inputs: 每个时间步输入张量的数量
    :type num_inputs: int
    :param num_states: 状态张量的数量
    :type num_states: int
    :param num_outputs: 每个时间步输出张量的数量
    :type num_outputs: int
    :param example_inputs: 可选的示例张量 ``[*inputs, *states]``
    :type example_inputs: Optional[Tuple[torch.Tensor, ...]]
    :param requires_grad: 指示 ``example_inputs`` 中每个参数是否需要梯度。
        若为 ``None`` ，则对所有浮点/复数输入启用梯度追踪
    :type requires_grad: Optional[Tuple[bool, ...]]
    :return: ``(fwd_kernel, bwd_kernel, info)`` ，可直接接入 FlexSN
        当前共享的 custom-op 执行路径
    :rtype: Tuple[object, object, FlexSNInfo]

    ----

    .. _build_training_kernels-en:

    * **English**

    Build FlexSN forward and backward Triton scan kernels for BPTT training.
    This function uses ``aot_function`` to trace both the forward and backward
    of ``core_fn`` and then produces:

    * a forward scan kernel that saves intermediates needed by backward
    * a backward scan kernel that runs the reverse-time pass

    When some outputs (for example, hard-threshold spike signals) are
    non-differentiable, AOT backward drops the corresponding gradient inputs.
    This function automatically generates a shim so the kernel template
    calling convention stays aligned.

    :param core_fn: Single-step dynamics callable with signature
        ``(*inputs, *states) -> (*outputs, *updated_states)``
    :type core_fn: Callable
    :param num_inputs: Number of per-step input tensors
    :type num_inputs: int
    :param num_states: Number of state tensors
    :type num_states: int
    :param num_outputs: Number of per-step output tensors
    :type num_outputs: int
    :param example_inputs: Optional example tensors ``[*inputs, *states]``
    :type example_inputs: Optional[Tuple[torch.Tensor, ...]]
    :param requires_grad: Flags indicating whether each example input should
        require gradients. If ``None``, all floating-point and complex inputs
        are traced as differentiable
    :type requires_grad: Optional[Tuple[bool, ...]]
    :return: ``(fwd_kernel, bwd_kernel, info)`` suitable for FlexSN's shared
        custom-op execution path
    :rtype: Tuple[object, object, FlexSNInfo]
    """
    from ..torch2triton import (
        generate_forward_and_backward_graph,
        generate_triton_code_str,
    )
    from .info import extract_info
    from .template import get_flexsn_backward_kernel, get_flexsn_forward_kernel

    example_inputs = _prepare_example_inputs(example_inputs, num_inputs, num_states)

    # Build requires_grad mask: only float/complex tensors support autograd.
    # Passing the mask prevents generate_forward_and_backward_graph from calling
    # requires_grad_(True) on int/bool inputs, which would error during tracing.
    if requires_grad is None:
        requires_grad_mask = tuple(
            t.is_floating_point() or t.is_complex() for t in example_inputs
        )
    else:
        if len(requires_grad) != len(example_inputs):
            raise ValueError(
                "requires_grad must have the same length as example_inputs "
                f"({len(example_inputs)}), but got {len(requires_grad)}."
            )
        requires_grad_mask = tuple(
            bool(flag) and (tensor.is_floating_point() or tensor.is_complex())
            for flag, tensor in zip(requires_grad, example_inputs)
        )

    # Trace forward AND backward (aot_function, no PYTORCH_JIT=0 needed)
    fwd_graph, bwd_graph = generate_forward_and_backward_graph(
        core_fn, example_inputs, requires_grad=requires_grad_mask
    )
    info = extract_info(fwd_graph, num_inputs, num_states, num_outputs)

    # Determine which outputs have gradients in the AOT backward graph
    mask = _diff_mask(
        core_fn,
        num_outputs,
        num_states,
        example_inputs,
        requires_grad_mask,
    )
    expected = num_outputs + num_states
    if len(mask) != expected:
        raise ValueError(
            f"core_fn returned {len(mask)} values but "
            f"num_outputs+num_states={expected}; "
            f"ensure core_fn returns exactly (*outputs, *updated_states)."
        )
    n_saved = len(info.c2k_return_mapping)

    raw_name = getattr(core_fn, "__name__", type(core_fn).__name__)
    safe_name = "".join(c if c.isalnum() else "_" for c in raw_name)
    core_name = safe_name + "_inductor_train"

    # Forward kernel — saves intermediates needed by backward
    fwd_str, fwd_name = generate_triton_code_str(fwd_graph, core_name + "_fwd")
    fwd_kernel = get_flexsn_forward_kernel(fwd_str, fwd_name, info=info)

    # Backward kernel — wrap with shim if non-differentiable outputs exist
    bwd_str, bwd_name = generate_triton_code_str(bwd_graph, core_name + "_bwd")
    if sum(mask) < expected:
        shim_code, bwd_call_name = _make_bwd_shim(
            bwd_name, n_saved, num_outputs, num_states, mask
        )
        bwd_str = bwd_str + shim_code
    else:
        bwd_call_name = bwd_name
    bwd_kernel = get_flexsn_backward_kernel(bwd_str, bwd_call_name, info=info)

    return fwd_kernel, bwd_kernel, info
