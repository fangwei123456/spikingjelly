"""Lower captured FX graphs into FlexSN Triton scan kernels."""

import torch.fx as fx

from ..torch2triton.graph2triton import generate_triton_code_str
from .info import _extract_info
from .template import (
    _get_backward_kernel,
    _get_forward_kernel,
    _get_inference_final_state_kernel,
    _get_inference_kernel,
)

__all__: list[str] = []


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


def _build_kernels(
    core_name: str,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    inference_graph: fx.Graph,
    forward_graph: fx.Graph,
    backward_graph: fx.Graph,
    differentiable: list[bool],
) -> dict[str, object]:
    name = "".join(c if c.isalnum() else "_" for c in core_name)
    inference_info = _extract_info(inference_graph, num_inputs, num_states, num_outputs)
    core_str, core_name = generate_triton_code_str(
        inference_graph, f"{name}_triton_inference"
    )
    inference_kernel = _get_inference_kernel(core_str, core_name, inference_info)
    final_kernel = _get_inference_final_state_kernel(
        core_str, core_name, inference_info
    )
    training_info = _extract_info(forward_graph, num_inputs, num_states, num_outputs)
    stem = f"{name}_triton_training"
    forward_str, forward_name = generate_triton_code_str(
        forward_graph, f"{stem}_forward"
    )
    forward_kernel = _get_forward_kernel(forward_str, forward_name, training_info)
    backward_str, backward_name = generate_triton_code_str(
        backward_graph, f"{stem}_backward"
    )
    if not all(differentiable):
        shim, backward_name = _backward_shim(
            backward_name,
            len(training_info.c2k_return_mapping),
            num_outputs,
            num_states,
            differentiable,
        )
        backward_str += shim
    backward_kernel = _get_backward_kernel(backward_str, backward_name, training_info)
    return dict(
        inference_kernel=inference_kernel,
        inference_final_state_kernel=final_kernel,
        inference_info=inference_info,
        forward_kernel=forward_kernel,
        backward_kernel=backward_kernel,
        training_info=training_info,
    )
