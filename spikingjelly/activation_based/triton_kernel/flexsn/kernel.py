"""Build the private Triton kernels used by FlexSN."""

from __future__ import annotations

from typing import Callable

import torch

__all__: list[str] = []


def _prepare_example_inputs(
    example_inputs: tuple[torch.Tensor, ...], expected: int
) -> tuple[torch.Tensor, ...]:
    if len(example_inputs) != expected:
        raise ValueError(
            f"FlexSN expected {expected} example tensors, got {len(example_inputs)}."
        )
    device = example_inputs[0].device
    if device.type != "cuda":
        raise RuntimeError("FlexSN Triton examples must be CUDA tensors.")
    return tuple(tensor.detach().to(device).clone() for tensor in example_inputs)


def _core_name(core_fn: Callable, suffix: str) -> str:
    raw_name = getattr(core_fn, "__name__", type(core_fn).__name__)
    safe_name = "".join(
        character if character.isalnum() else "_" for character in raw_name
    )
    return f"{safe_name}_{suffix}"


def build_inference_kernels(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    example_inputs: tuple[torch.Tensor, ...],
):
    """Trace ``core_fn`` once and build both inference return variants."""
    from torch.fx.experimental.proxy_tensor import make_fx

    from ..torch2triton import generate_triton_code_str
    from .info import extract_info
    from .template import (
        get_flexsn_inference_final_state_kernel,
        get_flexsn_inference_kernel,
    )

    examples = _prepare_example_inputs(example_inputs, num_inputs + num_states)
    graph = make_fx(core_fn)(*examples).graph
    core_str, core_name = generate_triton_code_str(
        graph, _core_name(core_fn, "triton_inference")
    )
    info = extract_info(graph, num_inputs, num_states, num_outputs)
    return (
        get_flexsn_inference_kernel(core_str, core_name, info),
        get_flexsn_inference_final_state_kernel(core_str, core_name, info),
        info,
    )


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


def _backward_shim(
    backward_name: str,
    num_saved: int,
    num_outputs: int,
    num_states: int,
    differentiable: list[bool],
) -> tuple[str, str]:
    gradients = [f"gs_{i}" for i in range(num_outputs)] + [
        f"gv_{i}" for i in range(num_states)
    ]
    if len(differentiable) != len(gradients):
        raise ValueError("FlexSN core return count changed while tracing backward.")
    saved = [f"sv_{i}" for i in range(num_saved)]
    shim_name = f"{backward_name}_shim"
    signature = ", ".join([*saved, *gradients])
    forwarded = ", ".join(
        [
            *saved,
            *(
                name
                for name, used in zip(gradients, differentiable, strict=True)
                if used
            ),
        ]
    )
    return (
        f"\n@triton.jit\ndef {shim_name}({signature}):\n"
        f"    return {backward_name}({forwarded})\n",
        shim_name,
    )


def build_training_kernels(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    example_inputs: tuple[torch.Tensor, ...],
):
    from ..torch2triton import (
        generate_forward_and_backward_graph,
        generate_triton_code_str,
    )
    from .info import extract_info
    from .template import get_flexsn_backward_kernel, get_flexsn_forward_kernel

    examples = _prepare_example_inputs(example_inputs, num_inputs + num_states)
    requires_grad = tuple(
        tensor.is_floating_point() or tensor.is_complex() for tensor in examples
    )
    forward_graph, backward_graph = generate_forward_and_backward_graph(
        core_fn, examples, requires_grad=requires_grad
    )
    info = extract_info(forward_graph, num_inputs, num_states, num_outputs)
    differentiable = _differentiable_returns(core_fn, examples)
    expected = num_outputs + num_states
    if len(differentiable) != expected:
        raise ValueError(
            f"FlexSN core returned {len(differentiable)} values, expected {expected}."
        )

    stem = _core_name(core_fn, "triton_training")
    forward_str, forward_name = generate_triton_code_str(
        forward_graph, f"{stem}_forward"
    )
    forward_kernel = get_flexsn_forward_kernel(forward_str, forward_name, info)

    backward_str, backward_name = generate_triton_code_str(
        backward_graph, f"{stem}_backward"
    )
    if not all(differentiable):
        shim, backward_name = _backward_shim(
            backward_name,
            len(info.c2k_return_mapping),
            num_outputs,
            num_states,
            differentiable,
        )
        backward_str += shim
    backward_kernel = get_flexsn_backward_kernel(backward_str, backward_name, info)
    return forward_kernel, backward_kernel, info
