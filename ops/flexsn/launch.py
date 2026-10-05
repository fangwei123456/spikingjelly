from typing import Optional, Tuple

import torch

from spikingjelly.logger import logger

try:
    import triton
except (ImportError, OSError) as e:
    from .. import triton_missing as dummy

    logger.info("Optional Triton dependency unavailable: {}", e)
    triton = dummy.DummyImport()

from ..triton_runtime import type_dict
from .info import _FlexSNInfo

__all__: list[str] = []


def _num_elements_per_step(x: torch.Tensor) -> int:
    n = 1
    for dim in x.shape[1:]:
        n *= dim
    return n


def _make_grid(ncl: int):
    def grid(meta):
        return (triton.cdiv(ncl, meta["BLOCK_NCL"]),)

    return grid


def _first_non_none_tensor(tensors):
    for tensor in tensors:
        if tensor is not None:
            return tensor
    return None


def _allocate_state_grad(
    i: int,
    T: int,
    state_templates: Optional[Tuple[torch.Tensor, ...]],
    grad_state_seq_examples,
    grad_example: torch.Tensor,
) -> torch.Tensor:
    if state_templates is not None:
        return (
            torch.zeros_like(state_templates[i])
            if T == 0
            else torch.empty_like(state_templates[i])
        )

    if i < len(grad_state_seq_examples) and grad_state_seq_examples[i] is not None:
        example = grad_state_seq_examples[i]
        return (
            example.new_zeros(example.shape[1:])
            if T == 0
            else example.new_empty(example.shape[1:])
        )

    return (
        grad_example.new_zeros(grad_example.shape[1:])
        if T == 0
        else grad_example.new_empty(grad_example.shape[1:])
    )


def _inference(f, info: _FlexSNInfo, *args) -> tuple:
    x_example = args[0]
    T = x_example.shape[0]
    NCL = _num_elements_per_step(x_example)
    dtype = x_example.dtype
    outputs = [
        torch.empty_like(x_example) for _ in range(info.num_outputs + info.num_states)
    ]
    if T == 0:
        return tuple(outputs)
    grid = _make_grid(NCL)

    f[grid](
        *args,
        *outputs,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple(outputs)


def _inference_final_state(f, info: _FlexSNInfo, *args) -> tuple:
    x_example = args[0]
    T = x_example.shape[0]
    NCL = _num_elements_per_step(x_example)
    dtype = x_example.dtype
    output_seqs = [torch.empty_like(x_example) for _ in range(info.num_outputs)]
    init_states = args[info.num_inputs : info.num_inputs + info.num_states]
    final_states = [
        init_states[i].new_empty(init_states[i].shape)
        if i < len(init_states)
        else x_example.new_empty(x_example.shape[1:])
        for i in range(info.num_states)
    ]
    if T == 0:
        final_states = [
            (
                init_states[i].clone()
                if i < len(init_states)
                else x_example.new_zeros(x_example.shape[1:])
            )
            for i in range(info.num_states)
        ]
        return (*output_seqs, *final_states)
    grid = _make_grid(NCL)

    f[grid](
        *args,
        *output_seqs,
        *final_states,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return (*output_seqs, *final_states)


def _forward(f, info: _FlexSNInfo, *args) -> tuple:
    x_example = args[0]
    T = x_example.shape[0]
    NCL = _num_elements_per_step(x_example)
    returns = [torch.empty_like(x_example) for _ in range(info.num_fwd_kernel_returns)]
    dtype = x_example.dtype
    if T == 0:
        return tuple(returns)
    grid = _make_grid(NCL)

    f[grid](
        *args,
        *returns,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple(returns)


def _backward(
    f,
    info: _FlexSNInfo,
    *args,
    input_templates: Optional[Tuple[torch.Tensor, ...]] = None,
    state_templates: Optional[Tuple[torch.Tensor, ...]] = None,
) -> tuple:
    required_grad_count = info.num_outputs + info.num_states
    grad_output_args = args[:required_grad_count]
    grad_output_example = _first_non_none_tensor(grad_output_args[: info.num_outputs])
    grad_example = _first_non_none_tensor(grad_output_args)
    if input_templates is None:
        if grad_example is None:
            raise ValueError(
                "input_templates are required when all incoming FlexSN gradients are None"
            )
        if info.num_inputs != 1:
            raise ValueError(
                "input_templates are required when FlexSN has multiple input sequences"
            )
        if grad_output_example is None:
            raise ValueError(
                "input_templates are required when FlexSN output-sequence gradients "
                "are all None"
            )
        input_templates = tuple(grad_output_example for _ in range(info.num_inputs))
    if len(input_templates) != info.num_inputs:
        raise ValueError(
            "input_templates must provide one template per FlexSN input sequence"
        )
    if state_templates is not None and len(state_templates) != info.num_states:
        raise ValueError(
            "state_templates must provide one template per FlexSN initial state"
        )
    if grad_example is None:
        if state_templates is None and info.num_states > 0:
            raise ValueError(
                "state_templates are required when all incoming FlexSN gradients are None"
            )
        grad_inputs = [torch.zeros_like(template) for template in input_templates]
        if state_templates is not None:
            grad_inputs.extend(
                torch.zeros_like(template) for template in state_templates
            )
        return tuple(grad_inputs)
    T = grad_example.shape[0]
    NCL = _num_elements_per_step(grad_example)
    grad_inputs = [
        (
            torch.zeros_like(input_templates[i])
            if T == 0
            else torch.empty_like(input_templates[i])
        )
        for i in range(info.num_inputs)
    ]
    grad_state_seq_examples = grad_output_args[
        info.num_outputs : info.num_outputs + info.num_states
    ]
    if state_templates is None and any(
        grad is None for grad in grad_state_seq_examples
    ):
        raise ValueError(
            "state_templates are required when any incoming FlexSN "
            "state-sequence gradient is None"
        )
    grad_kernel_args = [
        grad if grad is not None else torch.zeros_like(grad_example)
        for grad in grad_output_args
    ]
    # State-sequence gradients include the leading time dimension. The wrapper
    # returns gradients for the initial states, so their templates are shape[1:].
    grad_inputs += [
        _allocate_state_grad(
            i,
            T,
            state_templates,
            grad_state_seq_examples,
            grad_example,
        )
        for i in range(info.num_states)
    ]
    dtype = grad_example.dtype
    if T == 0:
        return tuple(grad_inputs)
    grid = _make_grid(NCL)

    f[grid](
        *grad_kernel_args,
        *args[required_grad_count:],
        *grad_inputs,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple(grad_inputs)
