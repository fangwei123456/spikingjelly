"""Private registered operators for FlexSN's generated Triton kernels."""

from __future__ import annotations

import contextlib
import weakref
from dataclasses import dataclass
from itertools import count
from threading import Lock

import torch

from ..triton_utils import register_op, wrap_triton
from .info import FlexSNInfo
from .wrapper import (
    flexsn_backward,
    flexsn_forward,
    flexsn_inference,
    flexsn_inference_final_state,
)

__all__: list[str] = []


@dataclass
class _KernelBundle:
    inference_kernel: object
    inference_final_state_kernel: object
    forward_kernel: object
    backward_kernel: object
    inference_info: FlexSNInfo
    training_info: FlexSNInfo
    owner_refs: int = 1
    active_refs: int = 0


_REGISTRY: dict[int, _KernelBundle] = {}
_REGISTRY_LOCK = Lock()
_HANDLE_IDS = count(1)


def _handle_id(handle: int) -> int:
    return int(handle)


def _bundle(handle: int) -> _KernelBundle:
    handle = _handle_id(handle)
    try:
        return _REGISTRY[handle]
    except KeyError as error:
        raise RuntimeError(f"Unknown FlexSN kernel handle: {handle}") from error


def register_flexsn_kernel_handle(
    *,
    inference_kernel,
    inference_info,
    inference_final_state_kernel,
    forward_kernel,
    backward_kernel,
    training_info,
) -> int:
    with _REGISTRY_LOCK:
        handle = next(_HANDLE_IDS)
        _REGISTRY[handle] = _KernelBundle(
            inference_kernel,
            inference_final_state_kernel,
            forward_kernel,
            backward_kernel,
            inference_info,
            training_info,
        )
    return handle


def _cleanup(bundle: _KernelBundle) -> None:
    for kernel in (
        bundle.inference_kernel,
        bundle.inference_final_state_kernel,
        bundle.forward_kernel,
        bundle.backward_kernel,
    ):
        close = getattr(kernel, "close", None)
        if callable(close):
            close()


def _release(handle: int, *, active: bool) -> None:
    handle = _handle_id(handle)
    with _REGISTRY_LOCK:
        bundle = _REGISTRY.get(handle)
        if bundle is None:
            return
        name = "active_refs" if active else "owner_refs"
        setattr(bundle, name, max(0, getattr(bundle, name) - 1))
        should_cleanup = bundle.owner_refs == 0 and bundle.active_refs == 0
        if should_cleanup:
            _REGISTRY.pop(handle)
    if should_cleanup:
        _cleanup(bundle)


def _retain_active(handle: int) -> None:
    with _REGISTRY_LOCK:
        _bundle(handle).active_refs += 1


def attach_flexsn_handle_finalizer(owner, handle: int):
    return weakref.finalize(owner, _release, handle, active=False)


def _device_guard(tensors):
    for tensor in tensors:
        if tensor.is_cuda:
            return torch.cuda.device(tensor.device)
    return contextlib.nullcontext()


def _seq_outputs(args: list[torch.Tensor], count: int):
    template = args[0]
    return [template.new_empty(template.shape) for _ in range(count)]


def _state_outputs(info: FlexSNInfo, args: list[torch.Tensor]):
    states = args[info.num_inputs : info.num_inputs + info.num_states]
    return [state.new_empty(state.shape) for state in states]


def _saved_final_state_indices(info: FlexSNInfo) -> list[int]:
    return [index for index in info.c2k_return_mapping if index >= info.num_outputs]


def _inference_impl(
    bundle: _KernelBundle,
    args: list[torch.Tensor],
    return_state_sequences: bool,
):
    args = [tensor.contiguous() for tensor in args]
    with _device_guard(args):
        if return_state_sequences:
            return list(
                flexsn_inference(
                    wrap_triton(bundle.inference_kernel),
                    bundle.inference_info,
                    *args,
                )
            )
        return list(
            flexsn_inference_final_state(
                wrap_triton(bundle.inference_final_state_kernel),
                bundle.inference_info,
                *args,
            )
        )


def _training_impl(
    bundle: _KernelBundle,
    args: list[torch.Tensor],
    return_state_sequences: bool,
):
    args = [tensor.contiguous() for tensor in args]
    with _device_guard(args):
        full_returns = list(
            flexsn_forward(
                wrap_triton(bundle.forward_kernel), bundle.training_info, *args
            )
        )
    if return_state_sequences:
        return full_returns

    info = bundle.training_info
    outputs = full_returns[: info.num_outputs]
    state_sequences = full_returns[
        info.num_outputs : info.num_outputs + info.num_states
    ]
    final_states = [state_sequence[-1].clone() for state_sequence in state_sequences]
    saved = [full_returns[index] for index in _saved_final_state_indices(info)]
    return [*outputs, *final_states, *saved]


def _backward_impl(
    bundle: _KernelBundle,
    grad_outputs: list[torch.Tensor],
    saved_tensors: list[torch.Tensor],
    templates: list[torch.Tensor],
):
    grads = [tensor.contiguous() for tensor in grad_outputs]
    saved = [tensor.contiguous() for tensor in saved_tensors]
    templates = [tensor.contiguous() for tensor in templates]
    info = bundle.training_info
    input_templates = tuple(templates[: info.num_inputs])
    state_templates = tuple(
        templates[info.num_inputs : info.num_inputs + info.num_states]
    )
    with _device_guard([*grads, *saved, *templates]):
        return list(
            flexsn_backward(
                wrap_triton(bundle.backward_kernel),
                info,
                *grads,
                *saved,
                input_templates=input_templates,
                state_templates=state_templates,
            )
        )


@register_op("sj::flexsn_triton_inference", mutates_args=())
def flexsn_triton_inference(
    handle: int,
    flat_args: list[torch.Tensor],
    return_state_sequences: bool,
) -> list[torch.Tensor]:
    return _inference_impl(_bundle(handle), flat_args, return_state_sequences)


@torch.library.register_fake("sj::flexsn_triton_inference")
def _inference_fake(
    handle: int,
    flat_args: list[torch.Tensor],
    return_state_sequences: bool,
) -> list[torch.Tensor]:
    info = _bundle(handle).inference_info
    outputs = _seq_outputs(flat_args, info.num_outputs)
    states = (
        _seq_outputs(flat_args, info.num_states)
        if return_state_sequences
        else _state_outputs(info, flat_args)
    )
    return [*outputs, *states]


@register_op("sj::flexsn_triton_training", mutates_args=())
def flexsn_triton_training(
    handle: int,
    flat_args: list[torch.Tensor],
    return_state_sequences: bool,
) -> list[torch.Tensor]:
    return _training_impl(_bundle(handle), flat_args, return_state_sequences)


@torch.library.register_fake("sj::flexsn_triton_training")
def _training_fake(
    handle: int,
    flat_args: list[torch.Tensor],
    return_state_sequences: bool,
) -> list[torch.Tensor]:
    info = _bundle(handle).training_info
    if return_state_sequences:
        return _seq_outputs(flat_args, info.num_fwd_kernel_returns)
    return [
        *_seq_outputs(flat_args, info.num_outputs),
        *_state_outputs(info, flat_args),
        *_seq_outputs(flat_args, len(_saved_final_state_indices(info))),
    ]


@register_op("sj::flexsn_triton_backward", mutates_args=())
def flexsn_triton_backward(
    handle: int,
    grad_outputs: list[torch.Tensor],
    saved_tensors: list[torch.Tensor],
    input_templates: list[torch.Tensor],
) -> list[torch.Tensor]:
    return _backward_impl(_bundle(handle), grad_outputs, saved_tensors, input_templates)


@torch.library.register_fake("sj::flexsn_triton_backward")
def _backward_fake(
    handle: int,
    grad_outputs: list[torch.Tensor],
    saved_tensors: list[torch.Tensor],
    input_templates: list[torch.Tensor],
) -> list[torch.Tensor]:
    return [tensor.new_empty(tensor.shape) for tensor in input_templates]


def _spec(tensor: torch.Tensor):
    return tuple(tensor.shape), tensor.dtype, tensor.device


def _materialize(spec):
    shape, dtype, device = spec
    return torch.empty((), dtype=dtype, device=device).expand(shape)


def _zero_grad(grad_outputs, index, spec):
    if index < len(grad_outputs) and grad_outputs[index] is not None:
        return grad_outputs[index]
    shape, dtype, device = spec
    return torch.zeros(shape, dtype=dtype, device=device)


def _training_setup(ctx, inputs, output) -> None:
    handle = _handle_id(inputs[0])
    info = _bundle(handle).training_info
    return_state_sequences = inputs[2]
    _retain_active(handle)
    ctx._active_finalizer = weakref.finalize(ctx, _release, handle, active=True)
    ctx.handle = handle
    ctx.return_state_sequences = return_state_sequences
    ctx.input_specs = [_spec(tensor) for tensor in inputs[1]]

    if return_state_sequences:
        visible = info.num_outputs + info.num_states
        ctx.output_specs = [_spec(tensor) for tensor in output[:visible]]
        saved = [output[index] for index in info.c2k_return_mapping]
    else:
        ctx.output_specs = [_spec(tensor) for tensor in output[: info.num_outputs]]
        sequence_length = ctx.input_specs[0][0][0]
        state_start = info.num_outputs
        state_end = state_start + info.num_states
        ctx.state_sequence_specs = [
            ((sequence_length, *shape), dtype, device)
            for shape, dtype, device in (
                _spec(tensor) for tensor in output[state_start:state_end]
            )
        ]
        extras = iter(output[info.num_outputs + info.num_states :])
        saved = [
            output[index] if index < info.num_outputs else next(extras)
            for index in info.c2k_return_mapping
        ]
    ctx.save_for_backward(*saved)


def _training_backward(ctx, grad_outputs):
    info = _bundle(ctx.handle).training_info
    if ctx.return_state_sequences:
        required = info.num_outputs + info.num_states
        grads = [
            _zero_grad(grad_outputs, index, ctx.output_specs[index])
            for index in range(required)
        ]
    else:
        grads = [
            _zero_grad(grad_outputs, index, ctx.output_specs[index])
            for index in range(info.num_outputs)
        ]
        for index, spec in enumerate(ctx.state_sequence_specs):
            shape, dtype, device = spec
            sequence_grad = torch.zeros(shape, dtype=dtype, device=device)
            final_index = info.num_outputs + index
            final_grad = (
                grad_outputs[final_index] if final_index < len(grad_outputs) else None
            )
            if final_grad is not None:
                sequence_grad[-1].copy_(final_grad)
            grads.append(sequence_grad)

    templates = [_materialize(spec) for spec in ctx.input_specs]
    try:
        if ctx._active_finalizer.alive:
            ctx._active_finalizer.detach()
        input_grads = flexsn_triton_backward(
            ctx.handle, grads, list(ctx.saved_tensors), templates
        )
    finally:
        _release(ctx.handle, active=True)
    return None, input_grads, None


torch.library.register_autograd(
    "sj::flexsn_triton_training",
    _training_backward,
    setup_context=_training_setup,
)
