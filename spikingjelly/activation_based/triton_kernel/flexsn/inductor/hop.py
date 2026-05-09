"""FlexSN time-step scan as a HigherOrderOperator.

Current progress:

M1:
* HOP definition with an eager Python time-step loop impl.
* Eager autograd works via the natural computation graph (``x[t]`` indexing
  and ``torch.stack`` are differentiable, so the per-step ``core_fn`` graph
  is correctly chained through time). Verified with ``gradcheck``.

M2:
* AOTAutograd tracing (``torch.fx.experimental.proxy_tensor.make_fx`` /
  ``torch._functorch.aot_autograd.aot_function``) works by unrolling the scan
  into T copies of ``core_fn``'s aten ops.

M3:
* ``FlexSN(backend="hop")`` is available as an explicit backend.
* Dynamo recognizes ``flex_sn_scan`` via a compatibility registration and can
  rewrite the call into a HOP node with a traced ``GraphModule`` body.
* ``torch.compile(fullgraph=True)`` for the HOP backend is verified on the
  Linux CI/server environment, including tensor lifted freevars/closures.

M4:
* ``lowerable_scan`` re-expresses the FlexSN step function through PyTorch's
  built-in ``torch.ops.higher_order.scan`` when that API is available.
* It is kept as an explicit experimental helper for investigating a
  single-scan-node forward path instead of fully unrolling the body.
* ``lowerable_while_loop_scan`` provides an alternative experimental forward
  path based on ``torch.ops.higher_order.while_loop``. On the Linux validation
  environment, its ``torch.compile(fullgraph=True) + no_grad`` path is working
  after switching to fixed-shape queue carries instead of ``x[t]`` indexing.
* The experimental while-loop path is wired into ``FlexSN(backend="hop")`` via
  ``SJ_ENABLE_EXPERIMENTAL_LOWERABLE_WHILE_LOOP=1`` for compile-time forward
  evaluation, and has been validated on:
  - a single FlexSN layer,
  - ``Linear -> FlexSN -> Linear``,
  - ``SpikingVGG`` forward inference.

Current limitations:

* The custom Dynamo registration in this file is still a compatibility shim,
  not a true in-tree ``BaseHOP`` integration.
* ``lowerable_scan`` is currently an experimental helper, not the default
  compiled path. In the PyTorch versions we validate against, fake/proxy/export
  handling for this out-of-tree scan shape is not yet stable enough to enable
  it by default.
* Training and autograd still use the existing eager/unrolled path.
* The current while-loop lowering is functionally correct for the validated
  forward ``no_grad`` cases, but it is not yet faster than the current
  ``backend="inductor"`` custom-op compile path on the server benchmark.
* A true first-class Inductor lowering for ``flex_sn_scan`` itself does not
  exist yet; the current "less unrolled" path relies on PyTorch's built-in
  scan / while_loop decomposition.

Usage::

    from spikingjelly.activation_based.triton_kernel.flexsn.inductor import flex_sn_scan

    # inputs_seq: tuple of T-leading tensors, e.g. shape [T, N, ...]
    # init_states: tuple of per-step state tensors, e.g. shape [N, ...]
    # returns: (*output_seqs, *state_seqs) — each with shape [T, ...]
    result = flex_sn_scan(core_fn, num_inputs, num_states, num_outputs,
                         *inputs_seq, *init_states)

Captured tensor freevars from ``core_fn`` are appended after the
``[*inputs_seq, *init_states]`` segment when Dynamo rewrites the HOP call.
"""
from __future__ import annotations
import inspect
import warnings
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator

try:
    from torch._higher_order_ops.scan import scan_op as _torch_scan_op
except (ImportError, AttributeError):
    _torch_scan_op = None

try:
    from torch._higher_order_ops.while_loop import while_loop as _torch_while_loop
except (ImportError, AttributeError):
    _torch_while_loop = None


class FlexSNScan(HigherOrderOperator):
    """HOP that runs a user-defined single-step ``core`` function over the
    leading time dimension of its inputs.

    The HOP is invoked with a flat argument list so that Dynamo / AOTAutograd
    can treat it uniformly. Shapes/semantics:

    * ``core_fn``: callable with signature
      ``(*step_inputs, *states) -> (*step_outputs, *updated_states)``.
    * ``num_inputs`` / ``num_states`` / ``num_outputs``: int literals used to
      partition the flat tensor args.
    * ``flat_args``: first ``num_inputs`` tensors are input sequences with
      leading time dim ``T``; the next ``num_states`` tensors are initial
      states (no time dim); any remaining tensors are lifted freevars that are
      passed through to ``core_fn`` unchanged at every time step.

    Return: ``num_outputs`` output sequences followed by ``num_states`` state
    sequences, all stacked along the leading time dim.
    """

    def __init__(self) -> None:
        super().__init__("flex_sn_scan")

    def __call__(
        self,
        core_fn: Callable,
        num_inputs: int,
        num_states: int,
        num_outputs: int,
        *flat_args: torch.Tensor,
        output_template_specs: Optional[OutputTemplateSpecs] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Invoke the FlexSN scan HigherOrderOperator.

        Chinese:
            调用 FlexSN scan HigherOrderOperator。

        English:
            Invoke the FlexSN scan HigherOrderOperator with flattened
            input-sequence, initial-state, and lifted tensor arguments.

        :param core_fn: EN: Single-step core callable with signature
            ``(*step_inputs, *states, *lifted_args)``.
            Chinese: 单步 ``core`` 可调用对象, 签名为
            ``(*step_inputs, *states, *lifted_args)``。
        :type core_fn: Callable
        :param num_inputs: EN: Number of T-leading input sequences.
            Chinese: 带时间维 ``T`` 的输入序列数量。
        :type num_inputs: int
        :param num_states: EN: Number of initial-state tensors without a time
            dimension. Chinese: 不带时间维的初始状态张量数量。
        :type num_states: int
        :param num_outputs: EN: Number of per-step outputs produced by
            ``core_fn``. Chinese: ``core_fn`` 每个时间步产生的输出数量。
        :type num_outputs: int
        :param flat_args: EN: Flattened tensor arguments: first the
            ``num_inputs`` input sequences ``[T, ...]``, then the
            ``num_states`` initial states, then any lifted tensor freevars.
            Chinese: 展平后的张量参数: 先是 ``num_inputs`` 个输入序列 ``[T, ...]``,
            再是 ``num_states`` 个初始状态, 最后是提升出来的张量自由变量。
        :type flat_args: torch.Tensor
        :param output_template_specs: EN: Optional output templates used when
            ``T == 0`` to materialize empty output sequences without executing
            ``core_fn``. Each item is ``(shape, dtype)`` or
            ``(shape, dtype, device)``; when ``device`` is omitted, the runtime
            device follows the first input sequence. Chinese: 可选输出模板, 在
            ``T == 0`` 时用于在不执行 ``core_fn`` 的情况下构造空输出序列。每个模板
            为 ``(shape, dtype)`` 或 ``(shape, dtype, device)``；省略 ``device``
            时, 运行时设备跟随第一个输入序列。
        :type output_template_specs: Optional[OutputTemplateSpecs]
        :return: EN: ``num_outputs`` output sequences followed by ``num_states``
            state sequences, all stacked along the leading time dimension.
            Chinese: 先返回 ``num_outputs`` 个输出序列, 再返回 ``num_states`` 个
            状态序列, 均沿首个时间维进行堆叠。
        :rtype: Tuple[torch.Tensor, ...]
        """
        return super().__call__(
            core_fn,
            num_inputs,
            num_states,
            num_outputs,
            *flat_args,
            output_template_specs=output_template_specs,
        )


flex_sn_scan = FlexSNScan()
_DYNAMO_HOP_REGISTERED = False
OutputTemplateSpec = Union[
    Tuple[Tuple[int, ...], torch.dtype],
    Tuple[Tuple[int, ...], torch.dtype, torch.device],
]
OutputTemplateSpecs = Tuple[OutputTemplateSpec, ...]


def _as_tuple(outputs):
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    return tuple(outputs)


def _empty_outputs_from_template(
    input_seqs: Tuple[torch.Tensor, ...],
    num_outputs: int,
    output_template_specs: Optional[OutputTemplateSpecs],
) -> Tuple[torch.Tensor, ...]:
    if num_outputs == 0:
        return ()
    if output_template_specs is None:
        raise ValueError(
            "FlexSN HOP empty scans require output_template_specs so output "
            "shapes and dtypes can be built without executing core_fn."
        )
    if len(output_template_specs) != num_outputs:
        raise ValueError(
            f"expected {num_outputs} output template specs, got "
            f"{len(output_template_specs)}"
        )
    outputs = []
    for spec in output_template_specs:
        if len(spec) == 2:
            shape, dtype = spec
            device = input_seqs[0].device
        else:
            shape, dtype, device = spec
        if device == input_seqs[0].device:
            outputs.append(input_seqs[0].new_empty((0, *shape), dtype=dtype))
        else:
            outputs.append(torch.empty((0, *shape), dtype=dtype, device=device))
    return tuple(outputs)


def _flatten_dynamo_body_result(value) -> Tuple[object, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(
            leaf for item in value for leaf in _flatten_dynamo_body_result(item)
        )
    variable_items = getattr(value, "items", None)
    if isinstance(variable_items, (tuple, list)):
        return tuple(
            leaf
            for item in variable_items
            for leaf in _flatten_dynamo_body_result(item)
        )
    return (value,)


def _dynamo_leaf_example_value(value):
    if isinstance(value, torch.Tensor):
        return value
    as_proxy = getattr(value, "as_proxy", None)
    if callable(as_proxy):
        try:
            proxy = as_proxy()
        except Exception:
            return None
        node = getattr(proxy, "node", None)
        meta = getattr(node, "meta", None)
        if isinstance(meta, dict):
            return meta.get("example_value")
    return None


def _output_template_specs_from_dynamo_body_result(
    body_result,
    num_outputs: int,
) -> Optional[OutputTemplateSpecs]:
    leaves = _flatten_dynamo_body_result(body_result)
    if len(leaves) < num_outputs:
        return None
    specs = []
    for leaf in leaves[:num_outputs]:
        example_value = _dynamo_leaf_example_value(leaf)
        if not isinstance(example_value, torch.Tensor):
            return None
        specs.append((tuple(example_value.shape), example_value.dtype))
    return tuple(specs)


def lowerable_scan_available() -> bool:
    """Report whether PyTorch's built-in ``scan`` HOP is available.

    Chinese:
        返回当前环境是否提供 PyTorch 内置 ``scan`` HOP。

    English:
        Return whether the current environment exposes PyTorch's built-in
        ``scan`` higher-order operator.

    :return: EN: ``True`` when ``torch.ops.higher_order.scan`` is available;
        otherwise ``False``. Chinese: 若 ``torch.ops.higher_order.scan`` 可用则
        返回 ``True``，否则返回 ``False``。
    :rtype: bool
    """
    return _torch_scan_op is not None


def dynamo_hop_available() -> bool:
    """Report whether the FlexSN Dynamo HOP registration succeeded.

    Chinese:
        返回 FlexSN 的 Dynamo HOP 注册是否成功。

    English:
        Return whether the FlexSN-specific Dynamo HigherOrderOperator
        registration has been installed successfully.

    :return: EN: ``True`` when the Dynamo compatibility shim for
        ``flex_sn_scan`` is registered; otherwise ``False``. Chinese:
        当 ``flex_sn_scan`` 的 Dynamo 兼容注册已完成时返回 ``True``，否则返回
        ``False``。
    :rtype: bool
    """
    return _DYNAMO_HOP_REGISTERED


def lowerable_while_loop_available() -> bool:
    """Report whether PyTorch's built-in ``while_loop`` HOP is available.

    Chinese:
        返回当前环境是否提供 PyTorch 内置 ``while_loop`` HOP。

    English:
        Return whether the current environment exposes PyTorch's built-in
        ``while_loop`` higher-order operator.

    :return: EN: ``True`` when ``torch.ops.higher_order.while_loop`` is
        available; otherwise ``False``. Chinese: 若
        ``torch.ops.higher_order.while_loop`` 可用则返回 ``True``，否则返回
        ``False``。
    :rtype: bool
    """
    return _torch_while_loop is not None


def _callable_positional_arg_range(
    fn: Callable,
) -> Optional[Tuple[int, Optional[int]]]:
    target = fn.forward if isinstance(fn, torch.nn.Module) else fn
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    for parameter in signature.parameters.values():
        if (
            parameter.kind == inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty
        ):
            return None
    min_required = 0
    positional_capacity = 0
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return min_required, None
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_capacity += 1
            if parameter.default is inspect.Parameter.empty:
                min_required += 1
    return min_required, positional_capacity


def _callable_accepts_positional_args(fn: Callable, n_args: int) -> bool | None:
    arg_range = _callable_positional_arg_range(fn)
    if arg_range is None:
        target = fn.forward if isinstance(fn, torch.nn.Module) else fn
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return None
        if any(
            parameter.kind == inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty
            for parameter in signature.parameters.values()
        ):
            return False
        return None
    min_required, capacity = arg_range
    if n_args < min_required:
        return False
    if capacity is None:
        return True
    return n_args <= capacity


def _reorder_placeholders_to_canonical_args(
    graph: torch.fx.Graph, canonical_arg_names: Tuple[str, ...]
) -> Tuple[torch.fx.Node, ...]:
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    if not placeholders:
        return ()

    by_name = {node.name: node for node in placeholders}
    ordered = [by_name[name] for name in canonical_arg_names if name in by_name]
    ordered.extend(node for node in placeholders if node not in ordered)

    if ordered != placeholders:
        first_non_placeholder = next(
            (node for node in graph.nodes if node.op != "placeholder"), None
        )
        if first_non_placeholder is not None:
            for node in ordered:
                first_non_placeholder.prepend(node)

    return tuple(node for node in graph.nodes if node.op == "placeholder")


def _check_lifted_arg_arity(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    lifted_args: Tuple[torch.Tensor, ...],
    *,
    skip_check: bool = False,
) -> None:
    if skip_check:
        return
    expected = num_inputs + num_states
    total = expected + len(lifted_args)
    accepts = _callable_accepts_positional_args(core_fn, total)
    if accepts is False:
        raise ValueError(
            f"flex_sn_scan expected {expected} tensor args "
            f"(num_inputs={num_inputs} + num_states={num_states}), "
            f"got {total}"
        )


def eager_scan(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    *flat_args: torch.Tensor,
    output_template_specs: Optional[OutputTemplateSpecs] = None,
) -> Tuple[torch.Tensor, ...]:
    """Run the FlexSN scan with an eager Python time-step loop.

    Chinese:
        通过 Python 时间步循环执行 FlexSN scan。

    English:
        Run the FlexSN scan with an eager Python loop. This helper is reused by
        both the HOP eager implementation and the Dynamo-friendly
        ``backend="inductor"`` path, so ``torch.compile`` can trace the
        unrolled loop into a standard FX graph.

        When ``T == 0``, ``output_template_specs`` must describe the output
        sequence shapes/dtypes so empty outputs can be materialized without
        executing ``core_fn``.

    :param core_fn: EN: Single-step core callable with signature
        ``(*step_inputs, *states, *lifted_args)``.
        Chinese: 单步 ``core`` 可调用对象, 签名为
        ``(*step_inputs, *states, *lifted_args)``。
    :type core_fn: Callable
    :param num_inputs: EN: Number of T-leading input sequences.
        Chinese: 带时间维 ``T`` 的输入序列数量。
    :type num_inputs: int
    :param num_states: EN: Number of initial-state tensors.
        Chinese: 初始状态张量数量。
    :type num_states: int
    :param num_outputs: EN: Number of per-step outputs.
        Chinese: 每个时间步输出数量。
    :type num_outputs: int
    :param flat_args: EN: Flattened input sequences, initial states, then lifted
        tensor freevars. Chinese: 展平后的输入序列、初始状态以及提升出来的张量自由变量。
    :type flat_args: torch.Tensor
    :param output_template_specs: EN: Optional ``(shape, dtype)`` or
        ``(shape, dtype, device)`` templates used to build empty output
        sequences when ``T == 0``. Omitted devices follow the first input
        sequence at runtime. Chinese: 在 ``T == 0`` 时用于构造空输出序列的可选模板,
        每项为 ``(shape, dtype)`` 或 ``(shape, dtype, device)``；省略 ``device``
        时运行时设备跟随第一个输入序列。
    :type output_template_specs: Optional[OutputTemplateSpecs]
    :return: EN: ``num_outputs`` output sequences followed by ``num_states``
        state sequences. Chinese: 先返回 ``num_outputs`` 个输出序列, 再返回
        ``num_states`` 个状态序列。
    :rtype: Tuple[torch.Tensor, ...]
    """
    expected = num_inputs + num_states
    if len(flat_args) < expected:
        raise ValueError(
            f"flex_sn_scan expected at least {expected} tensor args "
            f"(num_inputs={num_inputs} + num_states={num_states}), "
            f"got {len(flat_args)}"
        )

    inputs_seq = flat_args[:num_inputs]
    states = list(flat_args[num_inputs:expected])
    lifted_args = tuple(flat_args[expected:])
    _check_lifted_arg_arity(
        core_fn,
        num_inputs,
        num_states,
        lifted_args,
        skip_check=(
            num_inputs > 0
            and inputs_seq[0].shape[0] == 0
            and isinstance(core_fn, torch.fx.GraphModule)
        ),
    )

    if num_inputs == 0:
        raise ValueError("flex_sn_scan requires at least one input sequence")

    T = inputs_seq[0].shape[0]
    for i, x in enumerate(inputs_seq):
        if x.shape[0] != T:
            raise ValueError(
                f"input {i} has leading dim {x.shape[0]}, expected {T}"
            )

    if T == 0:
        empty_outputs = _empty_outputs_from_template(
            inputs_seq, num_outputs, output_template_specs
        )
        empty_states = tuple(
            state.new_empty((0, *state.shape))
            for state in states
        )
        return (*empty_outputs, *empty_states)

    output_buffers = [[] for _ in range(num_outputs)]
    state_buffers = [[] for _ in range(num_states)]

    for t in range(T):
        step_inputs = tuple(x[t] for x in inputs_seq)
        results = core_fn(*step_inputs, *states, *lifted_args)
        if not isinstance(results, (tuple, list)):
            results = (results,)
        if len(results) != num_outputs + num_states:
            raise ValueError(
                f"core returned {len(results)} values, "
                f"expected num_outputs + num_states "
                f"= {num_outputs + num_states}"
            )
        outputs = results[:num_outputs]
        states = list(results[num_outputs:])
        for i, y in enumerate(outputs):
            output_buffers[i].append(y)
        for i, s in enumerate(states):
            state_buffers[i].append(s)

    output_seqs = tuple(torch.stack(buf, dim=0) for buf in output_buffers)
    state_seqs = tuple(torch.stack(buf, dim=0) for buf in state_buffers)
    return (*output_seqs, *state_seqs)


flex_sn_scan.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)(eager_scan)
# HOPs route every tensor call through the Autograd dispatch key even when
# ``requires_grad=False``. Re-entering ``eager_scan`` from Autograd is
# correct: the inner ``core_fn`` invocations build a standard per-timestep
# autograd graph which is chained via ``torch.stack``/indexing, giving a
# full BPTT graph. AOTAutograd (``aot_function`` / ``make_fx``) traces this
# graph natively by unrolling; see module docstring.
flex_sn_scan.py_impl(torch._C.DispatchKey.Autograd)(eager_scan)


def eager_scan_final_state(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    *flat_args: torch.Tensor,
    output_template_specs: Optional[OutputTemplateSpecs] = None,
) -> Tuple[torch.Tensor, ...]:
    """Run the eager scan and return output sequences plus final states.

    Chinese:
        执行 eager scan, 返回输出序列以及最终状态。

    English:
        Variant of :func:`eager_scan` used when ``store_state_seqs=False`` so
        the HOP backend does not materialize full state sequences only to
        discard them. When ``T == 0``, ``output_template_specs`` is used to
        build empty output sequences and the provided initial states are cloned
        into the returned final states.

    :param core_fn: EN: Single-step core callable. Chinese: 单步 ``core`` 可调用对象。
    :type core_fn: Callable
    :param num_inputs: EN: Number of T-leading input sequences.
        Chinese: 带时间维 ``T`` 的输入序列数量。
    :type num_inputs: int
    :param num_states: EN: Number of initial-state tensors.
        Chinese: 初始状态张量数量。
    :type num_states: int
    :param num_outputs: EN: Number of per-step outputs.
        Chinese: 每个时间步输出数量。
    :type num_outputs: int
    :param flat_args: EN: Flattened input sequences, initial states, then lifted
        tensor freevars. Chinese: 展平后的输入序列、初始状态以及提升出来的张量自由变量。
    :type flat_args: torch.Tensor
    :param output_template_specs: EN: Optional ``(shape, dtype)`` or
        ``(shape, dtype, device)`` templates used to materialize empty output
        sequences when ``T == 0``. Omitted devices follow the first input
        sequence at runtime. Chinese: 在 ``T == 0`` 时用于构造空输出序列的可选模板,
        每项为 ``(shape, dtype)`` 或 ``(shape, dtype, device)``；省略 ``device``
        时运行时设备跟随第一个输入序列。
    :type output_template_specs: Optional[OutputTemplateSpecs]
    :return: EN: ``num_outputs`` output sequences followed by the final states.
        Chinese: 先返回 ``num_outputs`` 个输出序列, 再返回最终状态。
    :rtype: Tuple[torch.Tensor, ...]
    """
    expected = num_inputs + num_states
    if len(flat_args) < expected:
        raise ValueError(
            f"flex_sn_scan expected at least {expected} tensor args "
            f"(num_inputs={num_inputs} + num_states={num_states}), "
            f"got {len(flat_args)}"
        )

    inputs_seq = flat_args[:num_inputs]
    states = list(flat_args[num_inputs:expected])
    lifted_args = tuple(flat_args[expected:])
    _check_lifted_arg_arity(
        core_fn,
        num_inputs,
        num_states,
        lifted_args,
        skip_check=(
            num_inputs > 0
            and inputs_seq[0].shape[0] == 0
            and isinstance(core_fn, torch.fx.GraphModule)
        ),
    )

    if num_inputs == 0:
        raise ValueError("flex_sn_scan requires at least one input sequence")

    T = inputs_seq[0].shape[0]
    for i, x in enumerate(inputs_seq):
        if x.shape[0] != T:
            raise ValueError(
                f"input {i} has leading dim {x.shape[0]}, expected {T}"
            )

    if T == 0:
        empty_outputs = _empty_outputs_from_template(
            inputs_seq, num_outputs, output_template_specs
        )
        final_states = tuple(s.clone() for s in states)
        return (*empty_outputs, *final_states)

    output_buffers = [[] for _ in range(num_outputs)]

    for t in range(T):
        step_inputs = tuple(x[t] for x in inputs_seq)
        results = core_fn(*step_inputs, *states, *lifted_args)
        if not isinstance(results, (tuple, list)):
            results = (results,)
        if len(results) != num_outputs + num_states:
            raise ValueError(
                f"core returned {len(results)} values, "
                f"expected num_outputs + num_states "
                f"= {num_outputs + num_states}"
            )
        outputs = results[:num_outputs]
        states = list(results[num_outputs:])
        for i, y in enumerate(outputs):
            output_buffers[i].append(y)

    output_seqs = tuple(torch.stack(buf, dim=0) for buf in output_buffers)
    return (*output_seqs, *tuple(states))


def lowerable_scan(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    *flat_args: torch.Tensor,
    output_template_specs: Optional[OutputTemplateSpecs] = None,
) -> Tuple[torch.Tensor, ...]:
    """Run FlexSN scan through PyTorch's built-in ``scan`` HOP.

    Chinese:
        通过 PyTorch 内置 ``scan`` HOP 执行 FlexSN scan。

    English:
        Keep the FlexSN scan as a single higher-order op under tracing so
        downstream compilers can lower it as a loop instead of fully unrolling
        the body ``T`` times. This remains an experimental helper rather than
        the default compiled path.

        When ``T == 0``, ``output_template_specs`` must contain
        ``num_outputs`` items shaped as ``(shape, dtype)`` or
        ``(shape, dtype, device)``. Runtime devices default to the first input
        sequence when omitted.

    :param core_fn: EN: Single-step core callable. Chinese: 单步 ``core`` 可调用对象。
    :type core_fn: Callable
    :param num_inputs: EN: Number of T-leading input sequences.
        Chinese: 带时间维 ``T`` 的输入序列数量。
    :type num_inputs: int
    :param num_states: EN: Number of initial-state tensors.
        Chinese: 初始状态张量数量。
    :type num_states: int
    :param num_outputs: EN: Number of per-step outputs.
        Chinese: 每个时间步输出数量。
    :type num_outputs: int
    :param flat_args: EN: Flattened input sequences, initial states, then lifted
        tensor freevars. Chinese: 展平后的输入序列、初始状态以及提升出来的张量自由变量。
    :type flat_args: torch.Tensor
    :param output_template_specs: EN: Optional ``(shape, dtype)`` or
        ``(shape, dtype, device)`` templates used to materialize empty output
        sequences when ``T == 0``. Omitted devices follow the first input
        sequence at runtime. Chinese: 在 ``T == 0`` 时用于构造空输出序列的可选模板,
        每项为 ``(shape, dtype)`` 或 ``(shape, dtype, device)``；省略 ``device``
        时运行时设备跟随第一个输入序列。
    :type output_template_specs: Optional[OutputTemplateSpecs]
    :return: EN: ``num_outputs`` output sequences followed by ``num_states``
        state sequences. Chinese: 先返回 ``num_outputs`` 个输出序列, 再返回
        ``num_states`` 个状态序列。
    :rtype: Tuple[torch.Tensor, ...]
    """
    if _torch_scan_op is None:
        raise RuntimeError("PyTorch scan HOP is unavailable in this environment")

    expected = num_inputs + num_states
    if len(flat_args) < expected:
        raise ValueError(
            f"flex_sn_scan expected at least {expected} tensor args "
            f"(num_inputs={num_inputs} + num_states={num_states}), "
            f"got {len(flat_args)}"
        )
    if num_inputs == 0:
        raise ValueError("flex_sn_scan requires at least one input sequence")

    input_seqs = flat_args[:num_inputs]
    init_states = flat_args[num_inputs:expected]
    lifted_args = tuple(flat_args[expected:])
    _check_lifted_arg_arity(
        core_fn,
        num_inputs,
        num_states,
        lifted_args,
        skip_check=(
            num_inputs > 0
            and input_seqs[0].shape[0] == 0
            and isinstance(core_fn, torch.fx.GraphModule)
        ),
    )

    T = input_seqs[0].shape[0]
    for i, x in enumerate(input_seqs):
        if x.shape[0] != T:
            raise ValueError(
                f"input {i} has leading dim {x.shape[0]}, expected {T}"
            )

    if T == 0:
        empty_outputs = _empty_outputs_from_template(
            input_seqs, num_outputs, output_template_specs
        )
        empty_states = tuple(
            state.new_empty((0, *state.shape))
            for state in init_states
        )
        return (*empty_outputs, *empty_states)

    def combine_fn(carry, step_inputs, additional_inputs):
        carry = tuple(carry)
        step_inputs = tuple(step_inputs)
        additional_inputs = tuple(additional_inputs)
        results = core_fn(*step_inputs, *carry, *additional_inputs)
        results = tuple(results) if not isinstance(results, torch.Tensor) else (results,)
        if len(results) != num_outputs + num_states:
            raise ValueError(
                f"core returned {len(results)} values, "
                f"expected num_outputs + num_states = {num_outputs + num_states}"
            )

        outputs = list(results[:num_outputs])
        next_states = list(results[num_outputs:])
        output_states = [state.clone() for state in next_states]
        return next_states, [*outputs, *output_states]

    leaves_init = list(init_states)
    leaves_xs = list(input_seqs)
    _, spec_init = pytree.tree_flatten(leaves_init)
    _, spec_xs = pytree.tree_flatten(leaves_xs)

    def wrapped_combine_fn(*args):
        expected_args = len(leaves_init) + len(leaves_xs) + len(lifted_args)
        if len(args) != expected_args:
            raise ValueError(
                f"scan combine_fn expected {expected_args} flattened args, got {len(args)}"
            )
        carry = pytree.tree_unflatten(args[: len(leaves_init)], spec_init)
        xs = pytree.tree_unflatten(
            args[len(leaves_init) : len(leaves_init) + len(leaves_xs)],
            spec_xs,
        )
        additional_inputs = tuple(args[len(leaves_init) + len(leaves_xs) :])
        return combine_fn(carry, xs, additional_inputs)

    result = _torch_scan_op(
        wrapped_combine_fn,
        leaves_init,
        leaves_xs,
        additional_inputs=lifted_args,
    )
    result = tuple(result)
    # PyTorch scan returns final carry first, followed by the stacked outputs.
    return result[num_states:]


def lowerable_scan_final_state(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    *flat_args: torch.Tensor,
    output_template_specs: Optional[OutputTemplateSpecs] = None,
) -> Tuple[torch.Tensor, ...]:
    """Run the built-in ``scan`` HOP and return final states only.

    Chinese:
        通过内置 ``scan`` HOP 执行 FlexSN, 返回输出序列与最终状态。

    English:
        Final-state variant of :func:`lowerable_scan`. When ``T == 0``,
        ``output_template_specs`` materializes empty output sequences and the
        initial states are cloned into the returned final states.

    :param core_fn: EN: Single-step core callable. Chinese: 单步 ``core`` 可调用对象。
    :type core_fn: Callable
    :param num_inputs: EN: Number of T-leading input sequences.
        Chinese: 带时间维 ``T`` 的输入序列数量。
    :type num_inputs: int
    :param num_states: EN: Number of initial-state tensors.
        Chinese: 初始状态张量数量。
    :type num_states: int
    :param num_outputs: EN: Number of per-step outputs.
        Chinese: 每个时间步输出数量。
    :type num_outputs: int
    :param flat_args: EN: Flattened input sequences, initial states, then lifted
        tensor freevars. Chinese: 展平后的输入序列、初始状态以及提升出来的张量自由变量。
    :type flat_args: torch.Tensor
    :param output_template_specs: EN: Optional ``(shape, dtype)`` or
        ``(shape, dtype, device)`` templates used to materialize empty output
        sequences when ``T == 0``. Omitted devices follow the first input
        sequence at runtime. Chinese: 在 ``T == 0`` 时用于构造空输出序列的可选模板,
        每项为 ``(shape, dtype)`` 或 ``(shape, dtype, device)``；省略 ``device``
        时运行时设备跟随第一个输入序列。
    :type output_template_specs: Optional[OutputTemplateSpecs]
    :return: EN: ``num_outputs`` output sequences followed by the final states.
        Chinese: 先返回 ``num_outputs`` 个输出序列, 再返回最终状态。
    :rtype: Tuple[torch.Tensor, ...]
    """
    if _torch_scan_op is None:
        raise RuntimeError("PyTorch scan HOP is unavailable in this environment")

    expected = num_inputs + num_states
    if len(flat_args) < expected:
        raise ValueError(
            f"flex_sn_scan expected at least {expected} tensor args "
            f"(num_inputs={num_inputs} + num_states={num_states}), "
            f"got {len(flat_args)}"
        )
    if num_inputs == 0:
        raise ValueError("flex_sn_scan requires at least one input sequence")

    input_seqs = flat_args[:num_inputs]
    init_states = flat_args[num_inputs:expected]
    lifted_args = tuple(flat_args[expected:])
    _check_lifted_arg_arity(
        core_fn,
        num_inputs,
        num_states,
        lifted_args,
        skip_check=(
            num_inputs > 0
            and input_seqs[0].shape[0] == 0
            and isinstance(core_fn, torch.fx.GraphModule)
        ),
    )

    T = input_seqs[0].shape[0]
    for i, x in enumerate(input_seqs):
        if x.shape[0] != T:
            raise ValueError(
                f"input {i} has leading dim {x.shape[0]}, expected {T}"
            )

    if T == 0:
        empty_outputs = _empty_outputs_from_template(
            input_seqs, num_outputs, output_template_specs
        )
        final_states = tuple(s.clone() for s in init_states)
        return (*empty_outputs, *final_states)

    def combine_fn(carry, step_inputs, additional_inputs):
        carry = tuple(carry)
        step_inputs = tuple(step_inputs)
        additional_inputs = tuple(additional_inputs)
        results = core_fn(*step_inputs, *carry, *additional_inputs)
        results = tuple(results) if not isinstance(results, torch.Tensor) else (results,)
        if len(results) != num_outputs + num_states:
            raise ValueError(
                f"core returned {len(results)} values, "
                f"expected num_outputs + num_states = {num_outputs + num_states}"
            )
        outputs = list(results[:num_outputs])
        next_states = list(results[num_outputs:])
        return next_states, outputs

    leaves_init = list(init_states)
    leaves_xs = list(input_seqs)
    _, spec_init = pytree.tree_flatten(leaves_init)
    _, spec_xs = pytree.tree_flatten(leaves_xs)

    def wrapped_combine_fn(*args):
        expected_args = len(leaves_init) + len(leaves_xs) + len(lifted_args)
        if len(args) != expected_args:
            raise ValueError(
                f"scan combine_fn expected {expected_args} flattened args, got {len(args)}"
            )
        carry = pytree.tree_unflatten(args[: len(leaves_init)], spec_init)
        xs = pytree.tree_unflatten(
            args[len(leaves_init) : len(leaves_init) + len(leaves_xs)],
            spec_xs,
        )
        additional_inputs = tuple(args[len(leaves_init) + len(leaves_xs) :])
        return combine_fn(carry, xs, additional_inputs)

    result = _torch_scan_op(
        wrapped_combine_fn,
        leaves_init,
        leaves_xs,
        additional_inputs=lifted_args,
    )
    result = tuple(result)
    # PyTorch scan returns final carry first; keep that as the final states.
    final_states = result[:num_states]
    output_seqs = result[num_states:]
    return (*output_seqs, *final_states)


def _ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() >= 4:
        return tensor.contiguous(memory_format=torch.contiguous_format)
    return tensor.contiguous()


def _carry_device(*tensor_groups) -> torch.device:
    for group in tensor_groups:
        for tensor in group:
            return tensor.device
    return torch.device("cpu")


def _append_to_tail(buffer: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (buffer[1:], _ensure_contiguous(value).unsqueeze(0)),
        dim=0,
    )


def _shift_input_queue(queue: torch.Tensor) -> torch.Tensor:
    return torch.cat((queue[1:], queue[-1:]), dim=0)


def lowerable_while_loop_scan(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    *flat_args: torch.Tensor,
    output_template_specs: Optional[OutputTemplateSpecs] = None,
) -> Tuple[torch.Tensor, ...]:
    """Run FlexSN scan through PyTorch's built-in ``while_loop`` HOP.

    Chinese:
        通过 PyTorch 内置 ``while_loop`` HOP 执行 FlexSN scan。

    English:
        Experimental helper for studying whether a first-class loop
        representation is a better fit than the current unrolled scan path.
        Current while-loop capture does not support symbolic ``x[t]`` indexing
        here, so this implementation keeps functional queue buffers.

        When ``T == 0``, ``output_template_specs`` materializes empty output
        sequences without running ``core_fn``.

    :param core_fn: EN: Single-step core callable. Chinese: 单步 ``core`` 可调用对象。
    :type core_fn: Callable
    :param num_inputs: EN: Number of T-leading input sequences.
        Chinese: 带时间维 ``T`` 的输入序列数量。
    :type num_inputs: int
    :param num_states: EN: Number of initial-state tensors.
        Chinese: 初始状态张量数量。
    :type num_states: int
    :param num_outputs: EN: Number of per-step outputs.
        Chinese: 每个时间步输出数量。
    :type num_outputs: int
    :param flat_args: EN: Flattened input sequences, initial states, then lifted
        tensor freevars. Chinese: 展平后的输入序列、初始状态以及提升出来的张量自由变量。
    :type flat_args: torch.Tensor
    :param output_template_specs: EN: Optional ``(shape, dtype)`` or
        ``(shape, dtype, device)`` templates used to materialize empty output
        sequences when ``T == 0``. Omitted devices follow the first input
        sequence at runtime. Chinese: 在 ``T == 0`` 时用于构造空输出序列的可选模板,
        每项为 ``(shape, dtype)`` 或 ``(shape, dtype, device)``；省略 ``device``
        时运行时设备跟随第一个输入序列。
    :type output_template_specs: Optional[OutputTemplateSpecs]
    :return: EN: ``num_outputs`` output sequences followed by ``num_states``
        state sequences. Chinese: 先返回 ``num_outputs`` 个输出序列, 再返回
        ``num_states`` 个状态序列。
    :rtype: Tuple[torch.Tensor, ...]
    """
    if _torch_while_loop is None:
        raise RuntimeError("PyTorch while_loop HOP is unavailable in this environment")

    expected = num_inputs + num_states
    if len(flat_args) < expected:
        raise ValueError(
            f"flex_sn_scan expected at least {expected} tensor args "
            f"(num_inputs={num_inputs} + num_states={num_states}), "
            f"got {len(flat_args)}"
        )
    if num_inputs == 0:
        raise ValueError("flex_sn_scan requires at least one input sequence")

    input_seqs = tuple(flat_args[:num_inputs])
    init_states = tuple(flat_args[num_inputs:expected])
    lifted_args = tuple(flat_args[expected:])
    _check_lifted_arg_arity(
        core_fn,
        num_inputs,
        num_states,
        lifted_args,
        skip_check=(
            num_inputs > 0
            and input_seqs[0].shape[0] == 0
            and isinstance(core_fn, torch.fx.GraphModule)
        ),
    )
    lifted_args = tuple(_ensure_contiguous(arg) for arg in lifted_args)

    T = input_seqs[0].shape[0]
    for i, x in enumerate(input_seqs):
        if x.shape[0] != T:
            raise ValueError(f"input {i} has leading dim {x.shape[0]}, expected {T}")

    if T == 0:
        empty_outputs = _empty_outputs_from_template(
            input_seqs, num_outputs, output_template_specs
        )
        empty_states = tuple(
            state.new_empty((0, *state.shape))
            for state in init_states
        )
        return (*empty_outputs, *empty_states)

    input_seqs = tuple(_ensure_contiguous(seq) for seq in input_seqs)
    init_states = tuple(_ensure_contiguous(state) for state in init_states)

    first_step_inputs = tuple(_ensure_contiguous(x[0]) for x in input_seqs)
    first_results = core_fn(*first_step_inputs, *init_states, *lifted_args)
    first_results = (
        tuple(first_results)
        if not isinstance(first_results, torch.Tensor)
        else (first_results,)
    )
    if len(first_results) != num_outputs + num_states:
        raise ValueError(
            f"core returned {len(first_results)} values, "
            f"expected num_outputs + num_states = {num_outputs + num_states}"
        )

    first_outputs = tuple(_ensure_contiguous(x) for x in first_results[:num_outputs])
    first_states = tuple(_ensure_contiguous(x) for x in first_results[num_outputs:])
    output_buffers = tuple(
        _append_to_tail(out.new_zeros((T, *out.shape)), out) for out in first_outputs
    )
    state_buffers = tuple(
        _append_to_tail(state.new_zeros((T, *state.shape)), state)
        for state in first_states
    )
    pending_inputs = tuple(_shift_input_queue(seq) for seq in input_seqs)

    t0 = torch.tensor(
        1,
        dtype=torch.int64,
        device=_carry_device(first_states, input_seqs, first_outputs),
    )

    def cond_fn(t, *carry):
        return t < T

    def body_fn(t, *carry):
        pending_seq_end = num_inputs
        states_end = pending_seq_end + num_states
        outputs_end = states_end + num_outputs
        lifted_end = outputs_end + len(lifted_args)

        step_input_queues = carry[:pending_seq_end]
        states = carry[pending_seq_end:states_end]
        outputs_acc = carry[states_end:outputs_end]
        lifted = carry[outputs_end:lifted_end]
        states_acc = carry[lifted_end:]

        step_inputs = tuple(_ensure_contiguous(queue[0]) for queue in step_input_queues)
        results = core_fn(*step_inputs, *states, *lifted)
        results = tuple(results) if not isinstance(results, torch.Tensor) else (results,)
        if len(results) != len(first_results):
            raise ValueError(
                f"core returned {len(results)} values at runtime, "
                f"expected {len(first_results)}"
            )
        outputs = tuple(_ensure_contiguous(x) for x in results[:num_outputs])
        next_states = tuple(_ensure_contiguous(x) for x in results[num_outputs:])
        next_pending_inputs = tuple(
            _shift_input_queue(queue) for queue in step_input_queues
        )
        if len(outputs_acc) != len(outputs):
            raise ValueError(
                f"core returned {len(outputs)} outputs at runtime, "
                f"expected {len(outputs_acc)}"
            )
        next_output_acc = tuple(
            _append_to_tail(outputs_acc[i], outputs[i])
            for i in range(len(outputs_acc))
        )
        if len(states_acc) != len(next_states):
            raise ValueError(
                f"core returned {len(next_states)} states at runtime, "
                f"expected {len(states_acc)}"
            )
        next_state_acc = tuple(
            _append_to_tail(states_acc[i], next_states[i])
            for i in range(len(states_acc))
        )
        return (
            t + 1,
            *next_pending_inputs,
            *next_states,
            *next_output_acc,
            *lifted,
            *next_state_acc,
        )

    final = _torch_while_loop(
        cond_fn,
        body_fn,
        (
            t0,
            *pending_inputs,
            *first_states,
            *output_buffers,
            *lifted_args,
            *state_buffers,
        ),
    )
    final = tuple(final)
    pending_seq_end = 1 + num_inputs
    states_end = pending_seq_end + num_states
    outputs_end = states_end + num_outputs
    lifted_end = outputs_end + len(lifted_args)
    final_output_buffers = final[states_end:outputs_end]
    final_state_buffers = final[lifted_end:]
    return (*final_output_buffers, *final_state_buffers)


def lowerable_while_loop_scan_final_state(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    *flat_args: torch.Tensor,
    output_template_specs: Optional[OutputTemplateSpecs] = None,
) -> Tuple[torch.Tensor, ...]:
    """Run the while-loop HOP and return output sequences plus final states.

    Chinese:
        执行 while-loop HOP, 返回输出序列以及最终状态。

    English:
        Final-state variant of :func:`lowerable_while_loop_scan`. When
        ``T == 0``, ``output_template_specs`` is used to build empty output
        sequences and the provided initial states are cloned into the returned
        final states.

    :param core_fn: EN: Single-step core callable. Chinese: 单步 ``core`` 可调用对象。
    :type core_fn: Callable
    :param num_inputs: EN: Number of T-leading input sequences.
        Chinese: 带时间维 ``T`` 的输入序列数量。
    :type num_inputs: int
    :param num_states: EN: Number of initial-state tensors.
        Chinese: 初始状态张量数量。
    :type num_states: int
    :param num_outputs: EN: Number of per-step outputs.
        Chinese: 每个时间步输出数量。
    :type num_outputs: int
    :param flat_args: EN: Flattened input sequences, initial states, then lifted
        tensor freevars. Chinese: 展平后的输入序列、初始状态以及提升出来的张量自由变量。
    :type flat_args: torch.Tensor
    :param output_template_specs: EN: Optional ``(shape, dtype)`` or
        ``(shape, dtype, device)`` templates used to materialize empty output
        sequences when ``T == 0``. Omitted devices follow the first input
        sequence at runtime. Chinese: 在 ``T == 0`` 时用于构造空输出序列的可选模板,
        每项为 ``(shape, dtype)`` 或 ``(shape, dtype, device)``；省略 ``device``
        时运行时设备跟随第一个输入序列。
    :type output_template_specs: Optional[OutputTemplateSpecs]
    :return: EN: ``num_outputs`` output sequences followed by the final states.
        Chinese: 先返回 ``num_outputs`` 个输出序列, 再返回最终状态。
    :rtype: Tuple[torch.Tensor, ...]
    """
    if _torch_while_loop is None:
        raise RuntimeError("PyTorch while_loop HOP is unavailable in this environment")

    expected = num_inputs + num_states
    if len(flat_args) < expected:
        raise ValueError(
            f"flex_sn_scan expected at least {expected} tensor args "
            f"(num_inputs={num_inputs} + num_states={num_states}), "
            f"got {len(flat_args)}"
        )
    if num_inputs == 0:
        raise ValueError("flex_sn_scan requires at least one input sequence")

    input_seqs = tuple(flat_args[:num_inputs])
    init_states = tuple(flat_args[num_inputs:expected])
    lifted_args = tuple(flat_args[expected:])
    _check_lifted_arg_arity(
        core_fn,
        num_inputs,
        num_states,
        lifted_args,
        skip_check=(
            num_inputs > 0
            and input_seqs[0].shape[0] == 0
            and isinstance(core_fn, torch.fx.GraphModule)
        ),
    )
    lifted_args = tuple(_ensure_contiguous(arg) for arg in lifted_args)

    T = input_seqs[0].shape[0]
    for i, x in enumerate(input_seqs):
        if x.shape[0] != T:
            raise ValueError(f"input {i} has leading dim {x.shape[0]}, expected {T}")

    if T == 0:
        empty_outputs = _empty_outputs_from_template(
            input_seqs, num_outputs, output_template_specs
        )
        return (*empty_outputs, *(s.clone() for s in init_states))

    input_seqs = tuple(_ensure_contiguous(seq) for seq in input_seqs)
    init_states = tuple(_ensure_contiguous(state) for state in init_states)

    first_step_inputs = tuple(_ensure_contiguous(x[0]) for x in input_seqs)
    first_results = core_fn(*first_step_inputs, *init_states, *lifted_args)
    first_results = (
        tuple(first_results)
        if not isinstance(first_results, torch.Tensor)
        else (first_results,)
    )
    if len(first_results) != num_outputs + num_states:
        raise ValueError(
            f"core returned {len(first_results)} values, "
            f"expected num_outputs + num_states = {num_outputs + num_states}"
        )

    first_outputs = tuple(_ensure_contiguous(x) for x in first_results[:num_outputs])
    first_states = tuple(_ensure_contiguous(x) for x in first_results[num_outputs:])

    output_buffers = tuple(
        _append_to_tail(out.new_zeros((T, *out.shape)), out) for out in first_outputs
    )
    pending_inputs = tuple(_shift_input_queue(seq) for seq in input_seqs)

    t0 = torch.tensor(
        1,
        dtype=torch.int64,
        device=_carry_device(first_states, input_seqs, first_outputs),
    )

    def cond_fn(t, *carry):
        return t < T

    def body_fn(t, *carry):
        pending_seq_end = num_inputs
        states_end = pending_seq_end + num_states

        step_input_queues = carry[:pending_seq_end]
        outputs_end = states_end + num_outputs
        lifted_end = outputs_end + len(lifted_args)
        states = carry[pending_seq_end:states_end]
        outputs_acc = carry[states_end:outputs_end]
        lifted = carry[outputs_end:lifted_end]

        step_inputs = tuple(_ensure_contiguous(queue[0]) for queue in step_input_queues)
        results = core_fn(*step_inputs, *states, *lifted)
        results = tuple(results) if not isinstance(results, torch.Tensor) else (results,)
        if len(results) != len(first_results):
            raise ValueError(
                f"core returned {len(results)} values at runtime, "
                f"expected {len(first_results)}"
            )
        outputs = tuple(_ensure_contiguous(x) for x in results[:num_outputs])
        next_states = tuple(_ensure_contiguous(x) for x in results[num_outputs:])
        next_pending_inputs = tuple(
            _shift_input_queue(queue) for queue in step_input_queues
        )
        if len(outputs_acc) != len(outputs):
            raise ValueError(
                f"core returned {len(outputs)} outputs at runtime, "
                f"expected {len(outputs_acc)}"
            )
        next_output_acc = tuple(
            _append_to_tail(outputs_acc[i], outputs[i])
            for i in range(len(outputs_acc))
        )
        return (
            t + 1,
            *next_pending_inputs,
            *next_states,
            *next_output_acc,
            *lifted,
        )

    final = _torch_while_loop(
        cond_fn,
        body_fn,
        (
            t0,
            *pending_inputs,
            *first_states,
            *output_buffers,
            *lifted_args,
        ),
    )
    final = tuple(final)
    pending_seq_end = 1 + num_inputs
    states_end = pending_seq_end + num_states
    outputs_end = states_end + num_outputs
    final_states = final[pending_seq_end:states_end]
    final_output_buffers = final[states_end:outputs_end]
    return (*final_output_buffers, *final_states)


def _register_dynamo_hop() -> None:
    global _DYNAMO_HOP_REGISTERED
    try:
        from torch._dynamo.variables import higher_order_ops as hop_vars
        from torch._dynamo.variables.builder import wrap_fx_proxy
        from torch._dynamo.variables.constant import ConstantVariable
        from torch._dynamo.variables.functions import (
            NestedUserFunctionVariable,
            UserFunctionVariable,
        )
        from torch._dynamo.variables.higher_order_ops import (
            TorchHigherOrderOperatorVariable,
            make_attr,
            speculate_subgraph,
        )
        from torch._dynamo.variables.tensor import TensorVariable
    except (ImportError, ModuleNotFoundError, AttributeError):
        return
    except Exception as e:
        # Import-time registration must never break package import on
        # unsupported or drifting Torch internals; warn and leave the HOP
        # available through its eager fallback instead.
        warnings.warn(
            f"FlexSN HOP Dynamo registration failed unexpectedly: {type(e).__name__}: {e}",
            stacklevel=2,
        )
        return

    make_descriptor = TorchHigherOrderOperatorVariable.__dict__.get("make")
    original_make_is_bound = make_descriptor is None
    if make_descriptor is None:
        make_descriptor = TorchHigherOrderOperatorVariable.make
    make_func = (
        make_descriptor.__func__
        if isinstance(make_descriptor, (classmethod, staticmethod))
        else make_descriptor
    )
    if getattr(make_func, "_spikingjelly_flexsn_hop", False):
        _DYNAMO_HOP_REGISTERED = True
        return

    original_make = make_descriptor

    install_subgraph = getattr(hop_vars, "add_subgraph", None)
    if install_subgraph is None:
        def install_subgraph(tx, source, name, gm):
            return tx.output.install_subgraph(name, gm)

    class FlexSNScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
        _HOP_NAME = "spikingjelly.flex_sn_scan"
        _ALLOW_FALLBACK_TO_EAGER = False

        def call_function(self, tx, args, kwargs):
            output_template_specs_arg = kwargs.pop("output_template_specs", None)
            if kwargs:
                raise hop_vars.unimplemented(
                    "flex_sn_scan only supports output_template_specs as a kwarg"
                )
            explicit_output_template_specs = None
            if output_template_specs_arg is not None:
                try:
                    explicit_output_template_specs = (
                        output_template_specs_arg.as_python_constant()
                    )
                except Exception as e:
                    raise hop_vars.unimplemented(
                        "flex_sn_scan output_template_specs must be a Python constant"
                    ) from e

            if len(args) < 4:
                raise hop_vars.unimplemented(
                    "flex_sn_scan expects body_fn, num_inputs, num_states, "
                    "num_outputs, and tensor arguments"
                )

            body_fn = args[0]
            if not isinstance(body_fn, (UserFunctionVariable, NestedUserFunctionVariable)):
                raise hop_vars.unimplemented(
                    "flex_sn_scan expects a user-defined Python function body"
                )

            const_args = args[1:4]
            if not all(isinstance(arg, ConstantVariable) for arg in const_args):
                raise hop_vars.unimplemented(
                    "flex_sn_scan expects num_inputs/num_states/num_outputs to be constants"
                )

            num_inputs, num_states, num_outputs = [
                arg.as_python_constant() for arg in const_args
            ]
            flat_args = args[4:]
            expected = num_inputs + num_states
            if len(flat_args) < expected:
                raise hop_vars.unimplemented(
                    f"flex_sn_scan expected at least {expected} tensor args, got {len(flat_args)}"
                )
            if num_inputs == 0:
                raise hop_vars.unimplemented(
                    "flex_sn_scan requires at least one input sequence"
                )
            if not all(isinstance(arg, TensorVariable) for arg in flat_args):
                raise hop_vars.unimplemented(
                    "flex_sn_scan only supports tensor inputs and states"
                )

            def _make_step_template(arg: TensorVariable):
                example_value = arg.as_proxy().node.meta["example_value"]
                if example_value.shape[0] > 0:
                    return arg.call_method(tx, "__getitem__", [ConstantVariable(0)], {})

                shape_without_t = tuple(example_value.shape[1:])
                proxy = tx.output.create_proxy(
                    "call_function",
                    torch.ops.aten.new_empty.default,
                    args=(arg.as_proxy(), shape_without_t),
                    kwargs={},
                )
                return wrap_fx_proxy(
                    tx=tx,
                    proxy=proxy,
                    example_value=example_value.new_empty(shape_without_t),
                )

            step_inputs = [_make_step_template(arg) for arg in flat_args[:num_inputs]]
            body_args = [*step_inputs, *flat_args[num_inputs:]]
            canonical_body_arg_names = tuple(
                arg.as_proxy().node.name for arg in body_args
            )

            speculated = speculate_subgraph(
                tx,
                body_fn,
                body_args,
                {},
                "flex_sn_scan",
                source_target=self.value,
            )
            if len(speculated) == 4:
                (
                    _body_r,
                    body_graph,
                    body_lifted_freevars,
                    _parent_proxy_map,
                ) = speculated
            elif len(speculated) == 3:
                _body_r, body_graph, body_lifted_freevars = speculated
            else:
                raise hop_vars.unimplemented(
                    "flex_sn_scan received an unsupported speculate_subgraph result"
                )

            if hasattr(body_lifted_freevars, "keys"):
                lifted_freevars = tuple(body_lifted_freevars.keys())
            else:
                lifted_freevars = tuple(body_lifted_freevars)
            if lifted_freevars and not all(
                isinstance(freevar, torch.fx.Proxy) for freevar in lifted_freevars
            ):
                raise hop_vars.unimplemented(
                    "flex_sn_scan only supports tensor lifted freevars"
                )
            for freevar in lifted_freevars:
                example_value = freevar.node.meta.get("example_value")
                if not isinstance(example_value, torch.Tensor):
                    raise hop_vars.unimplemented(
                        "flex_sn_scan only supports tensor lifted freevars"
                    )

            placeholders = _reorder_placeholders_to_canonical_args(
                body_graph, canonical_body_arg_names
            )
            placeholder_freevar_names = tuple(
                node.name for node in placeholders[len(body_args) :]
            )
            if placeholder_freevar_names:
                freevars_by_name = {
                    freevar.node.name: freevar for freevar in lifted_freevars
                }
                missing = [
                    name
                    for name in placeholder_freevar_names
                    if name not in freevars_by_name
                ]
                if missing:
                    raise hop_vars.unimplemented(
                        "flex_sn_scan could not map lifted tensor freevars"
                    )
                lifted_freevars = tuple(
                    freevars_by_name[name] for name in placeholder_freevar_names
                )
            else:
                lifted_freevars = ()

            body_gm = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
            body_name = install_subgraph(tx, self.source, "flex_sn_scan_body", body_gm)
            body_node = make_attr(tx, body_name)
            output_template_specs = _output_template_specs_from_dynamo_body_result(
                _body_r,
                num_outputs,
            )
            if explicit_output_template_specs is not None:
                output_template_specs = explicit_output_template_specs
            proxy_kwargs = (
                {}
                if output_template_specs is None
                else {"output_template_specs": output_template_specs}
            )

            proxy = tx.output.create_proxy(
                "call_function",
                self.value,
                args=(
                    body_node,
                    num_inputs,
                    num_states,
                    num_outputs,
                    *(arg.as_proxy() for arg in flat_args),
                    *lifted_freevars,
                ),
                kwargs=proxy_kwargs,
            )
            body_leaves = _flatten_dynamo_body_result(_body_r)
            example_value = []
            T = flat_args[0].as_proxy().node.meta["example_value"].shape[0]
            for i in range(num_outputs + num_states):
                if i >= len(body_leaves):
                    example_value = None
                    break
                leaf_ev = _dynamo_leaf_example_value(body_leaves[i])
                if not isinstance(leaf_ev, torch.Tensor):
                    example_value = None
                    break
                example_value.append(leaf_ev.new_empty((T, *leaf_ev.shape)))
            if example_value is None:
                example_value = eager_scan(
                    body_gm,
                    num_inputs,
                    num_states,
                    num_outputs,
                    *(
                        arg.as_proxy().node.meta["example_value"]
                        for arg in flat_args
                    ),
                    *(
                        freevar.node.meta["example_value"]
                        for freevar in lifted_freevars
                    ),
                    output_template_specs=output_template_specs,
                )
            else:
                example_value = tuple(example_value)
            return wrap_fx_proxy(tx=tx, proxy=proxy, example_value=example_value)

    def patched_make(cls, value, source=None, **kwargs):
        if value is flex_sn_scan:
            return FlexSNScanHigherOrderVariable(value, source, **kwargs)
        if isinstance(original_make, classmethod):
            return original_make.__func__(cls, value, source=source, **kwargs)
        if isinstance(original_make, staticmethod):
            return original_make.__func__(value, source=source, **kwargs)
        if original_make_is_bound:
            original_make_func = getattr(original_make, "__func__", None)
            if original_make_func is not None:
                return original_make_func(cls, value, source=source, **kwargs)
            return original_make(value, source=source, **kwargs)
        return original_make(cls, value, source=source, **kwargs)

    patched_make._spikingjelly_flexsn_hop = True
    # Dynamo does not currently expose a public registry for this HOP hook.
    # Patch only the flex_sn_scan dispatch and delegate every other operator
    # back to PyTorch's original implementation.
    TorchHigherOrderOperatorVariable.make = classmethod(patched_make)
    _DYNAMO_HOP_REGISTERED = True


_register_dynamo_hop()
