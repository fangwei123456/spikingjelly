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
    if T == 0:
        return (*output_seqs, *(state.clone() for state in init_states))
    final_states = [state.new_empty(state.shape) for state in init_states]
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
    input_templates: tuple[torch.Tensor, ...],
    state_templates: tuple[torch.Tensor, ...],
) -> tuple:
    required_grad_count = info.num_outputs + info.num_states
    grad_output_args = args[:required_grad_count]
    if len(input_templates) != info.num_inputs:
        raise ValueError("input_templates must match the FlexSN input count")
    if len(state_templates) != info.num_states:
        raise ValueError("state_templates must match the FlexSN state count")
    grad_example = _first_non_none_tensor(grad_output_args)
    templates = (*input_templates, *state_templates)
    if grad_example is None:
        return tuple(torch.zeros_like(template) for template in templates)
    T = grad_example.shape[0]
    NCL = _num_elements_per_step(grad_example)
    grad_inputs = [
        torch.zeros_like(template) if T == 0 else torch.empty_like(template)
        for template in templates
    ]
    grad_kernel_args = [
        grad if grad is not None else torch.zeros_like(grad_example)
        for grad in grad_output_args
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
