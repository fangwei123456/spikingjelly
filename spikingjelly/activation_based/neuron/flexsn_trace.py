"""Capture a FlexSN callable as inference, forward and backward FX graphs."""

from typing import Callable

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from ..._ops.torch2triton.torch2graph import generate_forward_and_backward_graph

__all__: list[str] = []


def _differentiable_returns(
    core_fn: Callable,
    examples: tuple[torch.Tensor, ...],
) -> list[bool]:
    probes = []
    for tensor in examples:
        probe = tensor.detach().clone()
        if probe.is_floating_point() or probe.is_complex():
            probe.requires_grad_(True)
        probes.append(probe)
    with torch.enable_grad():
        returns = core_fn(*probes)
    return [
        isinstance(value, torch.Tensor)
        and (value.requires_grad or value.grad_fn is not None)
        for value in (returns if isinstance(returns, tuple) else (returns,))
    ]


def _trace_core(
    core: Callable,
    examples: tuple[torch.Tensor, ...],
    num_outputs: int,
    num_states: int,
) -> tuple[torch.fx.Graph, torch.fx.Graph, torch.fx.Graph, list[bool]]:
    examples = tuple(tensor.detach().clone() for tensor in examples)
    with torch.enable_grad():
        inference_graph = make_fx(core)(*examples).graph
        forward_graph, backward_graph = generate_forward_and_backward_graph(
            core,
            examples,
            requires_grad=tuple(
                tensor.is_floating_point() or tensor.is_complex() for tensor in examples
            ),
        )
        differentiable = _differentiable_returns(core, examples)
    expected = num_outputs + num_states
    if len(differentiable) != expected:
        raise ValueError(
            f"FlexSN core returned {len(differentiable)} values, expected {expected}."
        )
    return inference_graph, forward_graph, backward_graph, differentiable
