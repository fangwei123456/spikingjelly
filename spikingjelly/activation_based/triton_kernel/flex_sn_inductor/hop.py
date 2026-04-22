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

* A true Inductor lowering that keeps FlexSN as its own first-class HOP and
  emits a time loop without relying on PyTorch's built-in scan decomposition.
* Training/autograd support for the lowerable scan path. The built-in scan HOP
  used here to avoid full unrolling is currently inference-oriented.

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

from typing import Callable, Tuple

import torch
from torch._ops import HigherOrderOperator

try:
    from torch._higher_order_ops.scan import scan_op as _torch_scan_op
except Exception:
    _torch_scan_op = None


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


def lowerable_scan_available() -> bool:
    return _torch_scan_op is not None


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

    output_buffers = [[] for _ in range(num_outputs)]
    state_buffers = [[] for _ in range(num_states)]

    for t in range(T):
        step_inputs = tuple(x[t] for x in inputs_seq)
        results = core_fn(*step_inputs, *states, *lifted_args)
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
    body T times in the traced graph. It is currently intended for inference /
    no-grad usage.
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

    def combine_fn(*args):
        carry = args[:num_states]
        step_inputs = args[num_states : num_states + num_inputs]
        extra_inputs = args[num_states + num_inputs :]
        results = core_fn(*step_inputs, *carry, *extra_inputs)
        results = tuple(results) if not isinstance(results, torch.Tensor) else (results,)
        if len(results) != num_outputs + num_states:
            raise ValueError(
                f"core returned {len(results)} values, "
                f"expected num_outputs + num_states = {num_outputs + num_states}"
            )
        outputs = results[:num_outputs]
        next_states = results[num_outputs:]
        return [*next_states, *outputs, *next_states]

    result = _torch_scan_op(
        combine_fn,
        list(init_states),
        list(input_seqs),
        additional_inputs=lifted_args,
    )
    result = tuple(result)
    return result[num_states:]


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
    except Exception:
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
            if len(flat_args) != expected:
                raise hop_vars.unimplemented(
                    f"flex_sn_scan expected {expected} tensor args, got {len(flat_args)}"
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

            step_inputs = [
                arg.call_method(tx, "__getitem__", [_ConstantVariable(0)], {})
                for arg in flat_args[:num_inputs]
            ]
            body_args = [*step_inputs, *flat_args[num_inputs:]]

            if "source_target" in getattr(speculate_subgraph, "__code__").co_varnames:
                body_r, body_graph, body_lifted_freevars = speculate_subgraph(
                    tx,
                    body_fn,
                    body_args,
                    {},
                    "flex_sn_scan",
                    source_target=self.value,
                )
            else:
                graph_checkpoint, checkpoint = tx.output.graph, tx.copy_graphstate()
                body_r, body_graph, body_lifted_freevars = speculate_subgraph(
                    tx,
                    body_fn,
                    body_args,
                    {},
                    graph_checkpoint,
                    checkpoint,
                    "flex_sn_scan",
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
