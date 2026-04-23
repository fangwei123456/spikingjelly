"""Opaque custom ops for FlexSN inductor kernels.

This module is the bridge between FlexSN's per-layer Triton kernels and
``torch.compile``:

* Triton kernels and metadata are stored in a Python-side registry and
  referenced from graphs by a lightweight integer handle.
* Forward and backward are exposed as opaque ``torch.library`` custom ops so
  Dynamo / AOTAutograd no longer trace into Python kernel objects or Triton
  launcher internals.
* Tensor inputs/outputs are passed as ``list[Tensor]`` because FlexSN has a
  variable number of inputs, outputs, and states while ``custom_op`` does not
  support ``*args`` schemas.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from itertools import count
from threading import Lock
import weakref

import torch

from ..flexsn.wrapper import (
    flexsn_backward,
    flexsn_forward,
    flexsn_inference,
    flexsn_inference_final_state,
)
from ..flexsn.info import FlexSNInfo


@dataclass
class FlexSNKernelHandle:
    inference_kernel: object | None
    inference_info: FlexSNInfo | None
    inference_final_state_kernel: object | None
    inference_final_state_info: FlexSNInfo | None
    forward_kernel: object | None
    backward_kernel: object | None
    training_info: FlexSNInfo | None
    owner_refs: int = 1
    active_refs: int = 0


_KERNEL_REGISTRY: dict[int, FlexSNKernelHandle] = {}
_KERNEL_REGISTRY_LOCK = Lock()
_KERNEL_ID_GEN = count(1)


def _normalize_kernel_handle(handle: int) -> int:
    if isinstance(handle, int):
        return handle
    try:
        return int(handle)
    except Exception as exc:
        raise TypeError(
            f"Unsupported FlexSN kernel handle type: {type(handle)!r}"
        ) from exc


def register_flexsn_kernel_handle(
    *,
    inference_kernel,
    inference_info,
    inference_final_state_kernel,
    inference_final_state_info,
    forward_kernel,
    backward_kernel,
    training_info,
) -> int:
    with _KERNEL_REGISTRY_LOCK:
        handle = next(_KERNEL_ID_GEN)
        _KERNEL_REGISTRY[handle] = FlexSNKernelHandle(
            inference_kernel=inference_kernel,
            inference_info=inference_info,
            inference_final_state_kernel=inference_final_state_kernel,
            inference_final_state_info=inference_final_state_info,
            forward_kernel=forward_kernel,
            backward_kernel=backward_kernel,
            training_info=training_info,
        )
    return handle


def unregister_flexsn_kernel_handle(handle: int) -> None:
    with _KERNEL_REGISTRY_LOCK:
        bundle = _KERNEL_REGISTRY.pop(handle, None)
    if bundle is None:
        return
    _cleanup_kernel_handle(bundle)


def _cleanup_kernel_handle(bundle: FlexSNKernelHandle) -> None:
    for obj in (
        bundle.inference_kernel,
        bundle.inference_final_state_kernel,
        bundle.forward_kernel,
        bundle.backward_kernel,
    ):
        closer = getattr(obj, "close", None)
        if callable(closer):
            closer()


def _lookup_kernel_handle(handle: int) -> FlexSNKernelHandle:
    handle = _normalize_kernel_handle(handle)
    try:
        return _KERNEL_REGISTRY[handle]
    except KeyError as e:
        raise RuntimeError(f"Unknown FlexSN kernel handle: {handle}") from e


def retain_flexsn_kernel_handle(handle: int) -> None:
    handle = _normalize_kernel_handle(handle)
    with _KERNEL_REGISTRY_LOCK:
        bundle = _lookup_kernel_handle(handle)
        bundle.active_refs += 1


def retain_owner_flexsn_kernel_handle(handle: int) -> None:
    handle = _normalize_kernel_handle(handle)
    with _KERNEL_REGISTRY_LOCK:
        bundle = _lookup_kernel_handle(handle)
        bundle.owner_refs += 1


def release_flexsn_kernel_handle(handle: int) -> None:
    handle = _normalize_kernel_handle(handle)
    with _KERNEL_REGISTRY_LOCK:
        bundle = _KERNEL_REGISTRY.get(handle)
        if bundle is None:
            return
        bundle.owner_refs = max(0, bundle.owner_refs - 1)
        should_cleanup = bundle.owner_refs == 0 and bundle.active_refs == 0
        if should_cleanup:
            _KERNEL_REGISTRY.pop(handle, None)
    if should_cleanup:
        _cleanup_kernel_handle(bundle)


def release_active_flexsn_kernel_handle(handle: int) -> None:
    handle = _normalize_kernel_handle(handle)
    with _KERNEL_REGISTRY_LOCK:
        bundle = _KERNEL_REGISTRY.get(handle)
        if bundle is None:
            return
        bundle.active_refs = max(0, bundle.active_refs - 1)
        should_cleanup = bundle.owner_refs == 0 and bundle.active_refs == 0
        if should_cleanup:
            _KERNEL_REGISTRY.pop(handle, None)
    if should_cleanup:
        _cleanup_kernel_handle(bundle)


def _make_seq_outputs_like(
    info: FlexSNInfo, flat_args: list[torch.Tensor], n: int
) -> list[torch.Tensor]:
    if not flat_args:
        raise ValueError("Expected at least one FlexSN argument tensor.")
    # The underlying FlexSN Triton wrappers allocate all sequence outputs with
    # ``empty_like(flat_args[0])``. Keep fake tensors aligned with that runtime
    # contract so Dynamo/AOTAutograd sees the same shapes during tracing.
    seq_template = flat_args[0]
    return [seq_template.new_empty(seq_template.shape) for _ in range(n)]


def _template_spec(tensor: torch.Tensor):
    return tuple(tensor.shape), tensor.dtype, tensor.device


def _materialize_template(spec):
    shape, dtype, device = spec
    return torch.empty((), dtype=dtype, device=device).expand(shape)


def _device_guard(tensors: list[torch.Tensor]):
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            return torch.cuda.device(tensor.device)
    return contextlib.nullcontext()


def _materialize_zero_state_args(
    info: FlexSNInfo, flat_args: list[torch.Tensor]
) -> list[torch.Tensor]:
    if len(flat_args) == info.num_inputs:
        # In the common reset-before-forward path, FlexSN initial states are
        # zero tensors matching the per-step input shape. Materialize them
        # inside the opaque custom op so compile graphs do not need explicit
        # ``zeros_like`` nodes in front of every neuron layer.
        if info.num_inputs == 0:
            raise ValueError("FlexSN custom ops require at least one input sequence.")
        seq0 = flat_args[0]
        state_template = seq0[0]
        zero_states = [torch.zeros_like(state_template) for _ in range(info.num_states)]
        return [*flat_args, *zero_states]
    return flat_args


def _flexsn_inductor_inference_impl(
    bundle: FlexSNKernelHandle, flat_args: list[torch.Tensor]
) -> list[torch.Tensor]:
    args = _materialize_zero_state_args(bundle.inference_info, flat_args)
    args = [arg.contiguous() for arg in args]
    with _device_guard(args):
        return list(
            flexsn_inference(
                bundle.inference_kernel,
                bundle.inference_info,
                *args,
            )
        )


def _flexsn_inductor_inference_final_state_impl(
    bundle: FlexSNKernelHandle, flat_args: list[torch.Tensor]
) -> list[torch.Tensor]:
    args = _materialize_zero_state_args(bundle.inference_final_state_info, flat_args)
    args = [arg.contiguous() for arg in args]
    with _device_guard(args):
        return list(
            flexsn_inference_final_state(
                bundle.inference_final_state_kernel,
                bundle.inference_final_state_info,
                *args,
            )
        )


def _flexsn_inductor_training_impl(
    bundle: FlexSNKernelHandle, flat_args: list[torch.Tensor]
) -> list[torch.Tensor]:
    args = _materialize_zero_state_args(bundle.training_info, flat_args)
    args = [arg.contiguous() for arg in args]
    with _device_guard(args):
        return list(
            flexsn_forward(
                bundle.forward_kernel,
                bundle.training_info,
                *args,
            )
        )


def _flexsn_inductor_training_final_state_impl(
    bundle: FlexSNKernelHandle, flat_args: list[torch.Tensor]
) -> list[torch.Tensor]:
    full_returns = _flexsn_inductor_training_impl(bundle, flat_args)
    info = bundle.training_info
    assert info is not None
    visible_outputs = list(full_returns[: info.num_outputs])
    state_seqs = list(
        full_returns[info.num_outputs : info.num_outputs + info.num_states]
    )
    final_states = [state_seq[-1] for state_seq in state_seqs]
    saved_tensors = [full_returns[i] for i in info.c2k_return_mapping]
    return [*visible_outputs, *final_states, *saved_tensors]


def _flexsn_inductor_backward_impl(
    bundle: FlexSNKernelHandle,
    grad_outputs: list[torch.Tensor],
    saved_tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    grads = [grad.contiguous() for grad in grad_outputs]
    saved = [tensor.contiguous() for tensor in saved_tensors]
    with _device_guard([*grads, *saved]):
        return list(
            flexsn_backward(
                bundle.backward_kernel,
                bundle.training_info,
                *grads,
                *saved,
            )
        )


@torch.library.custom_op("sj::flexsn_inductor_inference", mutates_args=())
def flexsn_inductor_inference(handle: int, flat_args: list[torch.Tensor]) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.inference_kernel is None or bundle.inference_info is None:
        raise RuntimeError("FlexSN inference kernel is unavailable for this handle.")
    return _flexsn_inductor_inference_impl(bundle, flat_args)


@torch.library.custom_op("sj::flexsn_inductor_inference_final_state", mutates_args=())
def flexsn_inductor_inference_final_state(handle: int, flat_args: list[torch.Tensor]) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if (
        bundle.inference_final_state_kernel is None
        or bundle.inference_final_state_info is None
    ):
        raise RuntimeError("FlexSN inference-final-state kernel is unavailable for this handle.")
    return _flexsn_inductor_inference_final_state_impl(bundle, flat_args)


@torch.library.register_fake("sj::flexsn_inductor_inference")
def _flexsn_inductor_inference_fake(
    handle: int, flat_args: list[torch.Tensor]
) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.inference_info is None:
        raise RuntimeError("FlexSN inference metadata is unavailable for this handle.")
    return _make_seq_outputs_like(
        bundle.inference_info,
        flat_args,
        bundle.inference_info.num_outputs + bundle.inference_info.num_states,
    )


@torch.library.register_fake("sj::flexsn_inductor_inference_final_state")
def _flexsn_inductor_inference_final_state_fake(
    handle: int, flat_args: list[torch.Tensor]
) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.inference_final_state_info is None:
        raise RuntimeError(
            "FlexSN inference-final-state metadata is unavailable for this handle."
        )
    seq_outputs = _make_seq_outputs_like(
        bundle.inference_final_state_info,
        flat_args,
        bundle.inference_final_state_info.num_outputs,
    )
    if flat_args:
        state_template = flat_args[0][0]
    else:
        raise ValueError("Expected at least one input tensor for FlexSN fake inference.")
    final_states = [
        state_template.new_empty(state_template.shape)
        for _ in range(bundle.inference_final_state_info.num_states)
    ]
    return [*seq_outputs, *final_states]


@torch.library.custom_op("sj::flexsn_inductor_training", mutates_args=())
def flexsn_inductor_training(handle: int, flat_args: list[torch.Tensor]) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if (
        bundle.forward_kernel is None
        or bundle.backward_kernel is None
        or bundle.training_info is None
    ):
        raise RuntimeError("FlexSN training kernels are unavailable for this handle.")
    return _flexsn_inductor_training_impl(bundle, flat_args)


@torch.library.custom_op("sj::flexsn_inductor_training_final_state", mutates_args=())
def flexsn_inductor_training_final_state(
    handle: int, flat_args: list[torch.Tensor]
) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if (
        bundle.forward_kernel is None
        or bundle.backward_kernel is None
        or bundle.training_info is None
    ):
        raise RuntimeError("FlexSN training kernels are unavailable for this handle.")
    return _flexsn_inductor_training_final_state_impl(bundle, flat_args)


@torch.library.register_fake("sj::flexsn_inductor_training")
def _flexsn_inductor_training_fake(
    handle: int, flat_args: list[torch.Tensor]
) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    return _make_seq_outputs_like(
        bundle.training_info,
        flat_args,
        bundle.training_info.num_fwd_kernel_returns,
    )


@torch.library.register_fake("sj::flexsn_inductor_training_final_state")
def _flexsn_inductor_training_final_state_fake(
    handle: int, flat_args: list[torch.Tensor]
) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    seq_outputs = _make_seq_outputs_like(
        bundle.training_info,
        flat_args,
        bundle.training_info.num_outputs,
    )
    if flat_args:
        state_template = flat_args[0][0]
    else:
        raise ValueError("Expected at least one input tensor for FlexSN fake training.")
    final_states = [
        state_template.new_empty(state_template.shape)
        for _ in range(bundle.training_info.num_states)
    ]
    saved_tensors = _make_seq_outputs_like(
        bundle.training_info,
        flat_args,
        len(bundle.training_info.c2k_return_mapping),
    )
    return [*seq_outputs, *final_states, *saved_tensors]


@torch.library.custom_op("sj::flexsn_inductor_backward", mutates_args=())
def flexsn_inductor_backward(
    handle: int,
    grad_outputs: list[torch.Tensor],
    saved_tensors: list[torch.Tensor],
    input_templates: list[torch.Tensor],
) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.backward_kernel is None or bundle.training_info is None:
        raise RuntimeError("FlexSN backward kernel is unavailable for this handle.")
    return _flexsn_inductor_backward_impl(bundle, grad_outputs, saved_tensors)


@torch.library.register_fake("sj::flexsn_inductor_backward")
def _flexsn_inductor_backward_fake(
    handle: int,
    grad_outputs: list[torch.Tensor],
    saved_tensors: list[torch.Tensor],
    input_templates: list[torch.Tensor],
) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    if len(input_templates) == bundle.training_info.num_inputs:
        return [
            input_templates[i].new_empty(input_templates[i].shape)
            for i in range(bundle.training_info.num_inputs)
        ]
    seq_grads = [
        input_templates[i].new_empty(input_templates[i].shape)
        for i in range(bundle.training_info.num_inputs)
    ]
    state_offset = bundle.training_info.num_inputs
    state_grads = [
        input_templates[state_offset + i].new_empty(input_templates[state_offset + i].shape)
        for i in range(bundle.training_info.num_states)
    ]
    return [*seq_grads, *state_grads]


def _flexsn_training_setup_context(ctx, inputs, output):
    handle = _normalize_kernel_handle(inputs[0])
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    retain_flexsn_kernel_handle(handle)
    ctx._active_ref_finalizer = weakref.finalize(
        ctx, release_active_flexsn_kernel_handle, handle
    )
    ctx.handle = handle
    ctx.input_template_specs = [_template_spec(t) for t in inputs[1]]
    ctx.output_template_specs = [
        _template_spec(t)
        for t in output[: bundle.training_info.num_outputs + bundle.training_info.num_states]
    ]
    saved = [output[i] for i in bundle.training_info.c2k_return_mapping]
    ctx.save_for_backward(*saved)


def _flexsn_training_final_state_setup_context(ctx, inputs, output):
    handle = _normalize_kernel_handle(inputs[0])
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    retain_flexsn_kernel_handle(handle)
    ctx._active_ref_finalizer = weakref.finalize(
        ctx, release_active_flexsn_kernel_handle, handle
    )
    ctx.handle = handle
    ctx.input_template_specs = [_template_spec(t) for t in inputs[1]]
    ctx.output_template_specs = [
        _template_spec(t) for t in output[: bundle.training_info.num_outputs]
    ]
    if bundle.training_info.num_inputs == 0:
        raise RuntimeError("FlexSN training requires at least one input sequence.")
    seq_shape, seq_dtype, seq_device = ctx.input_template_specs[0]
    ctx.state_seq_template_spec = (seq_shape, seq_dtype, seq_device)
    visible = bundle.training_info.num_outputs + bundle.training_info.num_states
    ctx.save_for_backward(*list(output[visible:]))


def _flexsn_training_backward(ctx, grad_out: list[torch.Tensor | None]):
    bundle = _lookup_kernel_handle(ctx.handle)
    if bundle.backward_kernel is None or bundle.training_info is None:
        raise RuntimeError("FlexSN backward kernel is unavailable for this handle.")

    required_grads = bundle.training_info.num_outputs + bundle.training_info.num_states
    grad_inputs = [
        grad_out[i]
        if grad_out[i] is not None
        else torch.zeros(
            ctx.output_template_specs[i][0],
            dtype=ctx.output_template_specs[i][1],
            device=ctx.output_template_specs[i][2],
        )
        for i in range(required_grads)
    ]
    input_templates = [_materialize_template(spec) for spec in ctx.input_template_specs]
    try:
        if ctx._active_ref_finalizer.alive:
            ctx._active_ref_finalizer.detach()
        grads = list(
            flexsn_inductor_backward(
                ctx.handle,
                grad_inputs,
                list(ctx.saved_tensors),
                input_templates,
            )
        )
        if len(grads) != len(ctx.input_template_specs):
            grads = grads[: len(ctx.input_template_specs)]
    finally:
        release_active_flexsn_kernel_handle(ctx.handle)
    return None, grads


def _flexsn_training_final_state_backward(ctx, grad_out: list[torch.Tensor | None]):
    bundle = _lookup_kernel_handle(ctx.handle)
    if bundle.backward_kernel is None or bundle.training_info is None:
        raise RuntimeError("FlexSN backward kernel is unavailable for this handle.")

    output_grads = []
    for i in range(bundle.training_info.num_outputs):
        if grad_out[i] is not None:
            output_grads.append(grad_out[i])
        else:
            output_grads.append(
                torch.zeros(
                    ctx.output_template_specs[i][0],
                    dtype=ctx.output_template_specs[i][1],
                    device=ctx.output_template_specs[i][2],
                )
            )

    state_seq_shape, state_seq_dtype, state_seq_device = ctx.state_seq_template_spec
    state_grads = []
    for i in range(bundle.training_info.num_states):
        final_grad = grad_out[bundle.training_info.num_outputs + i]
        seq_grad = torch.zeros(
            state_seq_shape,
            dtype=state_seq_dtype,
            device=state_seq_device,
        )
        if final_grad is not None:
            seq_grad[-1].copy_(final_grad)
        state_grads.append(seq_grad)

    input_templates = [_materialize_template(spec) for spec in ctx.input_template_specs]
    try:
        if ctx._active_ref_finalizer.alive:
            ctx._active_ref_finalizer.detach()
        grads = list(
            flexsn_inductor_backward(
                ctx.handle,
                [*output_grads, *state_grads],
                list(ctx.saved_tensors),
                input_templates,
            )
        )
        if len(grads) != len(ctx.input_template_specs):
            grads = grads[: len(ctx.input_template_specs)]
    finally:
        release_active_flexsn_kernel_handle(ctx.handle)
    return None, grads


torch.library.register_autograd(
    "sj::flexsn_inductor_training",
    _flexsn_training_backward,
    setup_context=_flexsn_training_setup_context,
)

torch.library.register_autograd(
    "sj::flexsn_inductor_training_final_state",
    _flexsn_training_final_state_backward,
    setup_context=_flexsn_training_final_state_setup_context,
)


def attach_flexsn_handle_finalizer(owner, handle: int):
    return weakref.finalize(owner, release_flexsn_kernel_handle, handle)
