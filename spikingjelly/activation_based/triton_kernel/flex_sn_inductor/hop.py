"""FlexSN time-step scan as a HigherOrderOperator.

Current scope (M1 + M2):

* HOP definition with an eager Python time-step loop impl.
* Eager autograd works via the natural computation graph (``x[t]`` indexing
  and ``torch.stack`` are differentiable, so the per-step ``core_fn`` graph
  is correctly chained through time). Verified with ``gradcheck``.
* AOTAutograd tracing (``torch.fx.experimental.proxy_tensor.make_fx`` /
  ``torch._functorch.aot_autograd.aot_function``) works out of the box by
  unrolling the scan into T copies of ``core_fn``'s aten ops. This is the
  input format Inductor expects.

Deferred to M3:

* A Dynamo ``VariableBuilder`` so ``torch.compile(model, fullgraph=True)``
  can lift the HOP directly. Today the HOP is unsupported by Dynamo and
  users must either wrap the scan in ``torch.compiler.disable`` or use
  the lower-level ``aot_function`` API.
* An Inductor lowering that preserves the scan as a single node and emits
  a ``tl.static_range`` time loop (vs. the current unrolled-aten path).

Usage::

    from spikingjelly.activation_based.triton_kernel.flex_sn_inductor import flex_sn_scan

    # inputs_seq: tuple of T-leading tensors, e.g. shape [T, N, ...]
    # init_states: tuple of per-step state tensors, e.g. shape [N, ...]
    # returns: (*output_seqs, *state_seqs) — each with shape [T, ...]
    result = flex_sn_scan(core_fn, num_inputs, num_states, num_outputs,
                         *inputs_seq, *init_states)
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch._ops import HigherOrderOperator


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
      states (no time dim).

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
    if len(flat_args) != expected:
        raise ValueError(
            f"flex_sn_scan expected {expected} tensor args "
            f"(num_inputs={num_inputs} + num_states={num_states}), "
            f"got {len(flat_args)}"
        )

    inputs_seq = flat_args[:num_inputs]
    states = list(flat_args[num_inputs:])

    if num_inputs == 0:
        raise ValueError("flex_sn_scan requires at least one input sequence")

    T = inputs_seq[0].shape[0]
    for i, x in enumerate(inputs_seq):
        if x.shape[0] != T:
            raise ValueError(
                f"input {i} has leading dim {x.shape[0]}, expected {T}"
            )

    output_buffers = [[] for _ in range(num_outputs)]
    state_buffers = [[] for _ in range(num_states)]

    for t in range(T):
        step_inputs = tuple(x[t] for x in inputs_seq)
        results = core_fn(*step_inputs, *states)
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
) -> Tuple[torch.Tensor, ...]:
    """Eager scan variant that returns final states instead of full state seqs."""
    results = eager_scan(core_fn, num_inputs, num_states, num_outputs, *flat_args)
    output_seqs = results[:num_outputs]
    state_seqs = results[num_outputs:]
    final_states = tuple(state_seq[-1] for state_seq in state_seqs)
    return (*output_seqs, *final_states)


lowerable_scan = None
lowerable_scan_final_state = None
lowerable_while_loop_scan = None
lowerable_while_loop_scan_final_state = None


def lowerable_scan_available() -> bool:
    return False


def lowerable_while_loop_available() -> bool:
    return False
