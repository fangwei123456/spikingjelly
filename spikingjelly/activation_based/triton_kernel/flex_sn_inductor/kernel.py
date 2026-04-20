"""Build a single-pass Triton scan kernel from a user-defined core_fn.

Uses make_fx (no PYTORCH_JIT=0 required) to trace core_fn, then reuses the
existing flexsn template infrastructure to emit a tl.static_range(T) loop.

The resulting kernel is suited for the inference (no-grad) fast path of
FlexSN backend="inductor".
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


def build_inference_kernel(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    example_inputs: Optional[Tuple[torch.Tensor, ...]] = None,
):
    """Build a single-pass scan inference kernel for *core_fn*.

    The kernel wraps ``core_fn``'s per-step computation in a
    ``tl.static_range(T)`` loop, producing a single Triton kernel
    launch regardless of T.

    Args:
        core_fn: single-step dynamics callable with signature
            ``(*inputs, *states) -> (*outputs, *updated_states)``.
        num_inputs: number of per-step input tensors.
        num_states: number of state tensors.
        num_outputs: number of per-step output tensors.
        example_inputs: optional example tensors ``[*inputs, *states]``
            on CUDA.  If *None*, unit-sized float32 CUDA tensors are used.

    Returns:
        ``(kernel, info)`` — the compiled Triton kernel and the
        :class:`FlexSNInfo` metadata needed to call it via
        :func:`spikingjelly.activation_based.triton_kernel.flexsn.wrapper.flexsn_inference`.
    """
    from torch.fx.experimental.proxy_tensor import make_fx
    from ..torch2triton import generate_triton_code_str
    from ..flexsn import extract_info, get_flexsn_inference_kernel

    if example_inputs is None:
        example_inputs = tuple(
            torch.zeros(1, device="cuda") for _ in range(num_inputs + num_states)
        )
    else:
        example_inputs = tuple(x.detach().to("cuda") for x in example_inputs)

    # Trace core_fn to an aten-level FX graph (no PYTORCH_JIT=0 needed).
    traced = make_fx(core_fn)(*example_inputs)
    graph = traced.graph

    # Generate Triton function source for the per-step body.
    core_name = f"{core_fn.__name__}_inductor_scan"
    core_str, core_name = generate_triton_code_str(graph, core_name)

    # Extract metadata: arg/return names, output/state counts.
    info = extract_info(graph, num_inputs, num_states, num_outputs)

    # Build the scan kernel: single tl.static_range(T) loop.
    kernel = get_flexsn_inference_kernel(core_str, core_name, info=info)

    return kernel, info
