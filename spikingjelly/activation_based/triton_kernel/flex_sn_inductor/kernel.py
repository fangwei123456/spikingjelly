"""Build Triton scan kernels from a user-defined core_fn for FlexSN inductor backend.

Two entry points:
* build_inference_kernel  — no-grad fast path (make_fx, no PYTORCH_JIT=0 needed)
* build_inference_final_state_kernel — inference path that returns final states only
* build_training_kernels  — forward + backward kernels for full BPTT training
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch


def _training_final_state_specialized_wins(info) -> bool:
    saved_non_output_indices = []
    seen = set()
    for idx in info.c2k_return_mapping:
        if idx < info.num_outputs or idx in seen:
            continue
        saved_non_output_indices.append(idx)
        seen.add(idx)
    specialized_seq_count = info.num_outputs + len(saved_non_output_indices)
    return specialized_seq_count < info.num_fwd_kernel_returns


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
        # .clone() breaks aliasing: in-place ops inside core_fn during tracing
        # must not silently mutate the caller's original buffers.
        example_inputs = tuple(x.detach().to("cuda").clone() for x in example_inputs)

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
    from torch.fx.experimental.proxy_tensor import make_fx
    from ..torch2triton import generate_triton_code_str
    from ..flexsn import extract_info, get_flexsn_inference_final_state_kernel

    if example_inputs is None:
        example_inputs = tuple(
            torch.zeros(1, device="cuda") for _ in range(num_inputs + num_states)
        )
    else:
        example_inputs = tuple(x.detach().to("cuda").clone() for x in example_inputs)

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
) -> List[bool]:
    """Return which of core_fn's outputs are differentiable."""
    # Only float/complex tensors support requires_grad; skip int/bool inputs.
    ex = []
    for t in example_inputs:
        probe = t.clone().detach()
        if probe.is_floating_point() or probe.is_complex():
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


def _make_bwd_shim(bwd_fn_name: str, n_saved: int, num_outputs: int, num_states: int,
                   diff_mask: List[bool]) -> Tuple[str, str]:
    """Wrap the AOT backward function to match the template's calling convention.

    The template calls bwd_fn(saved..., grad_out0..., grad_state0...) for ALL
    outputs and states.  AOT autograd drops gradient arguments for non-
    differentiable outputs (e.g. spike signals from hard threshold), so the
    actual backward function may accept fewer arguments.  This shim accepts
    the full template signature and forwards only the used arguments.
    """
    all_grads = (["gs_" + str(i) for i in range(num_outputs)] +
                 ["gv_" + str(i) for i in range(num_states)])
    if len(diff_mask) != len(all_grads):
        raise ValueError(
            f"diff_mask length {len(diff_mask)} != "
            f"num_outputs+num_states {len(all_grads)}"
        )
    shim_args = ["sv_" + str(i) for i in range(n_saved)] + all_grads
    fwd_call  = (["sv_" + str(i) for i in range(n_saved)] +
                 [name for name, d in zip(all_grads, diff_mask) if d])
    shim_name = bwd_fn_name + "_shim"
    shim_code = (
        "\n@triton.jit\ndef " + shim_name +
        "(" + ", ".join(shim_args) + "):\n"
        "    return " + bwd_fn_name + "(" + ", ".join(fwd_call) + ")\n"
    )
    return shim_code, shim_name


def build_training_kernels(
    core_fn: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    example_inputs: Optional[Tuple[torch.Tensor, ...]] = None,
):
    """Build forward + backward Triton scan kernels for BPTT training.

    Uses ``aot_function`` (no ``PYTORCH_JIT=0`` required) to trace both the
    forward and backward of ``core_fn``, then builds:

    * a forward scan kernel that saves the intermediates needed for backward
    * a backward scan kernel that runs the reverse-time pass

    A shim is generated automatically when some outputs (e.g. spike signals)
    are non-differentiable, since AOT drops their gradient from the backward
    graph but the kernel template still passes them.

    Returns:
        ``(fwd_kernel, fwd_final_state_kernel, bwd_kernel, bwd_final_state_kernel, info)`` — compatible with
        :class:`spikingjelly.activation_based.triton_kernel.flexsn.wrapper.FlexSNFunction`.
    """
    from ..torch2triton import generate_forward_and_backward_graph, generate_triton_code_str
    from ..flexsn import (
        extract_info,
        get_flexsn_backward_kernel,
        get_flexsn_backward_final_state_kernel,
        get_flexsn_forward_final_state_kernel,
        get_flexsn_forward_kernel,
    )

    if example_inputs is None:
        example_inputs = tuple(
            torch.zeros(1, device="cuda") for _ in range(num_inputs + num_states)
        )
    else:
        # .clone() breaks aliasing so tracing cannot mutate the caller's buffers.
        example_inputs = tuple(x.detach().to("cuda").clone() for x in example_inputs)

    # Build requires_grad mask: only float/complex tensors support autograd.
    # Passing the mask prevents generate_forward_and_backward_graph from calling
    # requires_grad_(True) on int/bool inputs, which would error during tracing.
    requires_grad = tuple(
        t.is_floating_point() or t.is_complex() for t in example_inputs
    )

    # Trace forward AND backward (aot_function, no PYTORCH_JIT=0 needed)
    fwd_graph, bwd_graph = generate_forward_and_backward_graph(
        core_fn, example_inputs, requires_grad=requires_grad
    )
    info = extract_info(fwd_graph, num_inputs, num_states, num_outputs)

    # Determine which outputs have gradients in the AOT backward graph
    mask = _diff_mask(core_fn, num_outputs, num_states, example_inputs)
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
    if _training_final_state_specialized_wins(info):
        fwd_final_state_kernel = get_flexsn_forward_final_state_kernel(
            fwd_str, fwd_name, info=info
        )
    else:
        fwd_final_state_kernel = None

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
    bwd_final_state_kernel = get_flexsn_backward_final_state_kernel(
        bwd_str, bwd_call_name, info=info
    )

    return fwd_kernel, fwd_final_state_kernel, bwd_kernel, bwd_final_state_kernel, info
