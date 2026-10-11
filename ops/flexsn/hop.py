"""The single white-box HigherOrderOperator used by FlexSN."""

from __future__ import annotations

from typing import Callable

import torch
from torch._ops import HigherOrderOperator

__all__: list[str] = []


class _FlexSNScan(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("sj_flexsn_scan")

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
