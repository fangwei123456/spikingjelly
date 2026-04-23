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

    from spikingjelly.activation_based.triton_kernel.flex_sn_inductor import flex_sn_scan

    # inputs_seq: tuple of T-leading tensors, e.g. shape [T, N, ...]
    # init_states: tuple of per-step state tensors, e.g. shape [N, ...]
    # returns: (*output_seqs, *state_seqs) — each with shape [T, ...]
    result = flex_sn_scan(core_fn, num_inputs, num_states, num_outputs,
                         *inputs_seq, *init_states)

Captured tensor freevars from ``core_fn`` are appended after the
``[*inputs_seq, *init_states]`` segment when Dynamo rewrites the HOP call.
"""
from __future__ import annotations
import warnings
from typing import Callable, Tuple

import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator

try:
    from torch._higher_order_ops.scan import scan_op as _torch_scan_op
    from torch._higher_order_ops.scan import wrap_combine_fn_flat as _wrap_scan_combine_fn_flat
except (ImportError, AttributeError):
    _torch_scan_op = None
    _wrap_scan_combine_fn_flat = None

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
    ) -> Tuple[torch.Tensor, ...]:
        return super().__call__(
            core_fn, num_inputs, num_states, num_outputs, *flat_args
        )


flex_sn_scan = FlexSNScan()


def _as_tuple(outputs):
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    return tuple(outputs)


def lowerable_scan_available() -> bool:
    return _torch_scan_op is not None and _wrap_scan_combine_fn_flat is not None


def lowerable_while_loop_available() -> bool:
    return _torch_while_loop is not None


def eager_scan(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    *flat_args: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Plain-Python scan loop reused by both the HOP eager impl and the
    Dynamo-friendly path in :class:`FlexSN` (see ``backend="inductor"``).

    Dynamo cannot enter a :class:`HigherOrderOperator` today; calling this
    helper directly under ``torch.compile`` lets Dynamo trace the unrolled
    loop into an FX graph that Inductor can lower normally.
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

    if num_inputs == 0:
        raise ValueError("flex_sn_scan requires at least one input sequence")

    T = inputs_seq[0].shape[0]
    for i, x in enumerate(inputs_seq):
        if x.shape[0] != T:
            raise ValueError(
                f"input {i} has leading dim {x.shape[0]}, expected {T}"
            )

    if T == 0:
        step_inputs = tuple(x.new_empty(x.shape[1:]) for x in inputs_seq)
        with torch.no_grad():
            template_results = _as_tuple(core_fn(*step_inputs, *states, *lifted_args))
        if len(template_results) != num_outputs + num_states:
            raise ValueError(
                f"core returned {len(template_results)} values, "
                f"expected num_outputs + num_states "
                f"= {num_outputs + num_states}"
            )

        def _empty_output(i: int) -> torch.Tensor:
            ref = template_results[i]
            return ref.new_empty((0, *ref.shape))

        empty_outputs = tuple(_empty_output(i) for i in range(num_outputs))
        empty_states = tuple(
            state.new_empty((0, *state.shape))
            for state in template_results[num_outputs:]
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


def eager_scan_final_state(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    *flat_args: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Variant of :func:`eager_scan` that returns output sequences followed by
    final states only.

    This is used by :class:`FlexSN` when ``store_state_seqs=False`` so the HOP
    backend does not materialize full state sequences only to discard them.
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

    if num_inputs == 0:
        raise ValueError("flex_sn_scan requires at least one input sequence")

    T = inputs_seq[0].shape[0]
    for i, x in enumerate(inputs_seq):
        if x.shape[0] != T:
            raise ValueError(
                f"input {i} has leading dim {x.shape[0]}, expected {T}"
            )

    if T == 0:
        step_inputs = tuple(x.new_empty(x.shape[1:]) for x in inputs_seq)
        with torch.no_grad():
            template_results = _as_tuple(core_fn(*step_inputs, *states, *lifted_args))
        if len(template_results) != num_outputs + num_states:
            raise ValueError(
                f"core returned {len(template_results)} values, "
                f"expected num_outputs + num_states "
                f"= {num_outputs + num_states}"
            )

        def _empty_output(i: int) -> torch.Tensor:
            ref = template_results[i]
            return ref.new_empty((0, *ref.shape))

        empty_outputs = tuple(_empty_output(i) for i in range(num_outputs))
        final_states = tuple(states)
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
) -> Tuple[torch.Tensor, ...]:
    """Run FlexSN scan through PyTorch's built-in ``scan`` HOP.

    This path keeps the scan as a single higher-order op under tracing so
    downstream compilers can decompose it to a loop instead of unrolling the
    body T times in the traced graph. It is currently intended as an
    experimental helper for investigation rather than a production default.

    Notes:
    * On the PyTorch versions we validate against, fake/proxy/export handling
      for this out-of-tree scan pattern is still not stable enough to make this
      the default compiled path.
    * We mirror the internal ``scan`` frontend contract by routing through
      ``wrap_combine_fn_flat`` with explicit tree specs for the flattened
      inputs/states.
    """
    if _torch_scan_op is None or _wrap_scan_combine_fn_flat is None:
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
        return next_states, [*outputs, *next_states]

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
    return result[num_states:]


def lowerable_scan_final_state(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    *flat_args: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    if _torch_scan_op is None or _wrap_scan_combine_fn_flat is None:
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
    final_states = result[:num_states]
    output_seqs = result[num_states:]
    return (*output_seqs, *final_states)


def lowerable_while_loop_scan(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    *flat_args: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Run FlexSN scan through PyTorch's built-in ``while_loop`` HOP.

    This is an explicit research helper for probing whether a first-class loop
    representation is a better fit than the current unrolled scan path.
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

    def _ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() >= 4:
            return tensor.clone(memory_format=torch.contiguous_format)
        return tensor.contiguous()

    def _carry_device(*tensor_groups) -> torch.device:
        for group in tensor_groups:
            for tensor in group:
                return tensor.device
        return torch.device("cpu")

    T = input_seqs[0].shape[0]
    for i, x in enumerate(input_seqs):
        if x.shape[0] != T:
            raise ValueError(f"input {i} has leading dim {x.shape[0]}, expected {T}")

    if T == 0:
        step_inputs = tuple(x.new_empty(x.shape[1:]) for x in input_seqs)
        with torch.no_grad():
            template_results = _as_tuple(
                core_fn(*step_inputs, *init_states, *lifted_args)
            )
        if len(template_results) != num_outputs + num_states:
            raise ValueError(
                f"core returned {len(template_results)} values, "
                f"expected num_outputs + num_states "
                f"= {num_outputs + num_states}"
            )

        def _empty_output(i: int) -> torch.Tensor:
            ref = template_results[i]
            return ref.new_empty((0, *ref.shape))

        empty_outputs = tuple(
            _empty_output(i) for i in range(num_outputs)
        )
        empty_states = tuple(
            state.new_empty((0, *state.shape))
            for state in template_results[num_outputs:]
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
    def _append_to_tail(buffer: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.cat((buffer[1:], _ensure_contiguous(value).unsqueeze(0)), dim=0).contiguous()

    def _shift_input_queue(queue: torch.Tensor) -> torch.Tensor:
        return torch.cat((queue[1:], queue[-1:].clone()), dim=0).contiguous()

    output_buffers = tuple(
        _append_to_tail(out.new_zeros((T, *out.shape)), out) for out in first_outputs
    )
    state_buffers = tuple(
        _append_to_tail(state.new_zeros((T, *state.shape)), state)
        for state in first_states
    )
    pending_inputs = tuple(
        _shift_input_queue(seq) for seq in input_seqs
    )

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
        next_output_acc = tuple(
            _append_to_tail(buf, out)
            for buf, out in zip(outputs_acc, outputs, strict=True)
        )
        next_state_acc = tuple(
            _append_to_tail(buf, state)
            for buf, state in zip(states_acc, next_states, strict=True)
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
) -> Tuple[torch.Tensor, ...]:
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

    def _ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() >= 4:
            return tensor.clone(memory_format=torch.contiguous_format)
        return tensor.contiguous()

    def _carry_device(*tensor_groups) -> torch.device:
        for group in tensor_groups:
            for tensor in group:
                return tensor.device
        return torch.device("cpu")

    T = input_seqs[0].shape[0]
    for i, x in enumerate(input_seqs):
        if x.shape[0] != T:
            raise ValueError(f"input {i} has leading dim {x.shape[0]}, expected {T}")

    if T == 0:
        step_inputs = tuple(x.new_empty(x.shape[1:]) for x in input_seqs)
        with torch.no_grad():
            template_results = _as_tuple(
                core_fn(*step_inputs, *init_states, *lifted_args)
            )
        if len(template_results) != num_outputs + num_states:
            raise ValueError(
                f"core returned {len(template_results)} values, "
                f"expected num_outputs + num_states "
                f"= {num_outputs + num_states}"
            )

        def _empty_output(i: int) -> torch.Tensor:
            ref = template_results[i]
            return ref.new_empty((0, *ref.shape))

        empty_outputs = tuple(
            _empty_output(i) for i in range(num_outputs)
        )
        return (*empty_outputs, *init_states)

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

    def _append_to_tail(buffer: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.cat((buffer[1:], _ensure_contiguous(value).unsqueeze(0)), dim=0).contiguous()

    def _shift_input_queue(queue: torch.Tensor) -> torch.Tensor:
        return torch.cat((queue[1:], queue[-1:].clone()), dim=0).contiguous()

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
        next_output_acc = tuple(
            _append_to_tail(buf, out)
            for buf, out in zip(outputs_acc, outputs, strict=True)
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


flex_sn_scan.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)(eager_scan)
# HOPs route every tensor call through the Autograd dispatch key even when
# ``requires_grad=False``. Re-entering ``eager_scan`` from Autograd is
# correct: the inner ``core_fn`` invocations build a standard per-timestep
# autograd graph which is chained via ``torch.stack``/indexing, giving a
# full BPTT graph. AOTAutograd (``aot_function`` / ``make_fx``) traces this
# graph natively by unrolling; see module docstring.
flex_sn_scan.py_impl(torch._C.DispatchKey.Autograd)(eager_scan)


def _register_dynamo_hop() -> None:
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
        warnings.warn(
            f"FlexSN HOP Dynamo registration failed unexpectedly: {e}",
            stacklevel=2,
        )
        return

    if getattr(TorchHigherOrderOperatorVariable.make, "_spikingjelly_flexsn_hop", False):
        return

    original_make = TorchHigherOrderOperatorVariable.make

    install_subgraph = getattr(hop_vars, "add_subgraph", None)
    if install_subgraph is None:
        def install_subgraph(tx, source, name, gm):
            return tx.output.install_subgraph(name, gm)

    class FlexSNScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
        def call_function(self, tx, args, kwargs):
            if kwargs:
                raise hop_vars.unimplemented("flex_sn_scan does not support kwargs")

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

            from torch._dynamo.variables.constant import ConstantVariable as _ConstantVariable

            def _make_step_template(arg: TensorVariable):
                example_value = arg.as_proxy().node.meta["example_value"]
                if example_value.shape[0] > 0:
                    return arg.call_method(tx, "__getitem__", [_ConstantVariable(0)], {})

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

            _body_r, body_graph, body_lifted_freevars, _parent_proxy_map = speculate_subgraph(
                tx,
                body_fn,
                body_args,
                {},
                "flex_sn_scan",
                source_target=self.value,
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

            body_gm = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
            body_name = install_subgraph(tx, self.source, "flex_sn_scan_body", body_gm)
            body_node = make_attr(tx, body_name)

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
                kwargs={},
            )
            example_value = eager_scan(
                body_gm,
                num_inputs,
                num_states,
                num_outputs,
                *(arg.as_proxy().node.meta["example_value"] for arg in flat_args),
                *(freevar.node.meta["example_value"] for freevar in lifted_freevars),
            )
            return wrap_fx_proxy(tx=tx, proxy=proxy, example_value=example_value)

    def patched_make(value, source=None, **kwargs):
        if value is flex_sn_scan:
            return FlexSNScanHigherOrderVariable(value, source, **kwargs)
        return original_make(value, source=source, **kwargs)

    patched_make._spikingjelly_flexsn_hop = True
    TorchHigherOrderOperatorVariable.make = staticmethod(patched_make)


_register_dynamo_hop()
