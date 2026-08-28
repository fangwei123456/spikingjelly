"""The single white-box HigherOrderOperator used by FlexSN."""

from __future__ import annotations

from typing import Callable

import torch
from torch._ops import HigherOrderOperator

__all__: list[str] = []


class _FlexSNScan(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flex_sn_scan")

    def __call__(
        self,
        core: Callable,
        num_inputs: int,
        num_states: int,
        num_outputs: int,
        num_static: int,
        return_state_sequences: bool,
        *flat_args: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return super().__call__(
            core,
            num_inputs,
            num_states,
            num_outputs,
            num_static,
            return_state_sequences,
            *flat_args,
        )


_hop_scan = _FlexSNScan()


def _eager_scan(
    core: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    num_static: int,
    return_state_sequences: bool,
    *flat_args: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    expected = num_inputs + num_states + num_static
    if num_inputs <= 0 or len(flat_args) < expected:
        raise ValueError(
            f"FlexSN HOP expected {expected} tensors, got {len(flat_args)}."
        )
    input_sequences = flat_args[:num_inputs]
    states = tuple(flat_args[num_inputs : num_inputs + num_states])
    static_end = num_inputs + num_states + num_static
    static_inputs = tuple(flat_args[num_inputs + num_states : static_end])
    lifted_inputs = tuple(flat_args[static_end:])
    T = input_sequences[0].shape[0]
    if T == 0:
        raise ValueError("FlexSN HOP does not support empty sequences.")
    if any(sequence.shape[0] != T for sequence in input_sequences):
        raise ValueError("FlexSN HOP input sequences must share the same T.")

    output_steps = [[] for _ in range(num_outputs)]
    state_steps = [[] for _ in range(num_states)] if return_state_sequences else None
    for t in range(T):
        returns = core(
            *(sequence[t] for sequence in input_sequences),
            *states,
            *static_inputs,
            *lifted_inputs,
        )
        returns = returns if isinstance(returns, tuple) else (returns,)
        if len(returns) != num_outputs + num_states:
            raise ValueError(
                f"FlexSN core returned {len(returns)} tensors, expected "
                f"{num_outputs + num_states}."
            )
        outputs = returns[:num_outputs]
        states = returns[num_outputs:]
        for steps, output in zip(output_steps, outputs, strict=True):
            steps.append(output)
        if state_steps is not None:
            for steps, state in zip(state_steps, states, strict=True):
                steps.append(state)

    outputs = tuple(torch.stack(steps) for steps in output_steps)
    if state_steps is None:
        return (*outputs, *states)
    return (*outputs, *(torch.stack(steps) for steps in state_steps))


_hop_scan.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)(_eager_scan)
_hop_scan.py_impl(torch._C.DispatchKey.Autograd)(_eager_scan)


def _flatten_result(value) -> tuple:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(leaf for item in value for leaf in _flatten_result(item))
    items = getattr(value, "items", None)
    if isinstance(items, (tuple, list)):
        return tuple(leaf for item in items for leaf in _flatten_result(item))
    return (value,)


def _example_value(value):
    if isinstance(value, torch.Tensor):
        return value
    as_proxy = getattr(value, "as_proxy", None)
    if not callable(as_proxy):
        return None
    try:
        return as_proxy().node.meta.get("example_value")
    except (AttributeError, KeyError):
        return None


def _reorder_placeholders(graph: torch.fx.Graph, argument_names: tuple[str, ...]):
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    by_name = {node.name: node for node in placeholders}
    ordered = [by_name[name] for name in argument_names if name in by_name]
    ordered.extend(node for node in placeholders if node not in ordered)
    if ordered != placeholders:
        first_body_node = next(
            (node for node in graph.nodes if node.op != "placeholder"), None
        )
        if first_body_node is not None:
            for node in ordered:
                first_body_node.prepend(node)
    return tuple(node for node in graph.nodes if node.op == "placeholder")


def _register_dynamo_hop() -> None:
    try:
        from torch._dynamo.variables import higher_order_ops as hop_variables
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
    except (ImportError, AttributeError):
        return

    descriptor = TorchHigherOrderOperatorVariable.__dict__.get("make")
    original_is_bound = descriptor is None
    if descriptor is None:
        descriptor = TorchHigherOrderOperatorVariable.make
    function = (
        descriptor.__func__
        if isinstance(descriptor, (classmethod, staticmethod))
        else descriptor
    )
    if getattr(function, "_spikingjelly_flexsn_hop", False):
        return
    original_make = descriptor

    install_subgraph = getattr(hop_variables, "add_subgraph", None)
    if install_subgraph is None:

        def install_subgraph(tx, source, name, graph_module):
            return tx.output.install_subgraph(name, graph_module)

    class _FlexSNScanVariable(TorchHigherOrderOperatorVariable):
        _HOP_NAME = "spikingjelly.flex_sn_scan"
        _ALLOW_FALLBACK_TO_EAGER = False

        def call_function(self, tx, args, kwargs):
            if kwargs or len(args) < 6:
                raise hop_variables.unimplemented(
                    "FlexSN HOP expects a body, five constants, and tensors."
                )
            body = args[0]
            if not isinstance(body, (UserFunctionVariable, NestedUserFunctionVariable)):
                raise hop_variables.unimplemented(
                    "FlexSN HOP body must be a Python function."
                )
            constants = args[1:6]
            if not all(isinstance(value, ConstantVariable) for value in constants):
                raise hop_variables.unimplemented(
                    "FlexSN HOP metadata must be Python constants."
                )
            (
                num_inputs,
                num_states,
                num_outputs,
                num_static,
                return_state_sequences,
            ) = (value.as_python_constant() for value in constants)
            flat_args = args[6:]
            expected = num_inputs + num_states + num_static
            if len(flat_args) != expected or not all(
                isinstance(value, TensorVariable) for value in flat_args
            ):
                raise hop_variables.unimplemented(
                    f"FlexSN HOP expected {expected} tensor operands."
                )

            step_inputs = [
                value.call_method(tx, "__getitem__", [ConstantVariable(0)], {})
                for value in flat_args[:num_inputs]
            ]
            body_args = [*step_inputs, *flat_args[num_inputs:]]
            argument_names = tuple(value.as_proxy().node.name for value in body_args)
            speculated = speculate_subgraph(
                tx,
                body,
                body_args,
                {},
                "flex_sn_scan",
                source_target=self.value,
            )
            if len(speculated) == 4:
                body_result, body_graph, lifted, _ = speculated
            elif len(speculated) == 3:
                body_result, body_graph, lifted = speculated
            else:
                raise hop_variables.unimplemented(
                    "Unsupported FlexSN HOP speculate_subgraph result."
                )
            lifted = tuple(lifted.keys()) if hasattr(lifted, "keys") else tuple(lifted)
            placeholders = _reorder_placeholders(body_graph, argument_names)
            lifted_names = tuple(node.name for node in placeholders[len(body_args) :])
            if lifted_names:
                lifted_by_name = {value.node.name: value for value in lifted}
                try:
                    lifted = tuple(lifted_by_name[name] for name in lifted_names)
                except KeyError as error:
                    raise RuntimeError(
                        "FlexSN HOP could not map Dynamo lifted operands."
                    ) from error
            else:
                lifted = ()

            graph_module = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
            body_name = install_subgraph(
                tx, self.source, "flex_sn_scan_body", graph_module
            )
            body_node = make_attr(tx, body_name)
            proxy = tx.output.create_proxy(
                "call_function",
                self.value,
                args=(
                    body_node,
                    num_inputs,
                    num_states,
                    num_outputs,
                    num_static,
                    return_state_sequences,
                    *(value.as_proxy() for value in flat_args),
                    *lifted,
                ),
                kwargs={},
            )

            leaves = _flatten_result(body_result)
            if len(leaves) < num_outputs + num_states:
                raise hop_variables.unimplemented(
                    "FlexSN HOP could not infer body outputs."
                )
            T = flat_args[0].as_proxy().node.meta["example_value"].shape[0]
            examples = []
            for index, leaf in enumerate(leaves[: num_outputs + num_states]):
                example = _example_value(leaf)
                if not isinstance(example, torch.Tensor):
                    raise hop_variables.unimplemented(
                        "FlexSN HOP body must return tensors."
                    )
                is_sequence = index < num_outputs or return_state_sequences
                shape = (T, *example.shape) if is_sequence else example.shape
                examples.append(example.new_empty(shape))
            return wrap_fx_proxy(tx=tx, proxy=proxy, example_value=tuple(examples))

    def patched_make(cls, value, source=None, **kwargs):
        if value is _hop_scan:
            return _FlexSNScanVariable(value, source, **kwargs)
        if isinstance(original_make, classmethod):
            return original_make.__func__(cls, value, source=source, **kwargs)
        if isinstance(original_make, staticmethod):
            return original_make.__func__(value, source=source, **kwargs)
        if original_is_bound:
            original_function = getattr(original_make, "__func__", None)
            if original_function is not None:
                return original_function(cls, value, source=source, **kwargs)
            return original_make(value, source=source, **kwargs)
        return original_make(cls, value, source=source, **kwargs)

    patched_make._spikingjelly_flexsn_hop = True
    TorchHigherOrderOperatorVariable.make = classmethod(patched_make)


_register_dynamo_hop()
