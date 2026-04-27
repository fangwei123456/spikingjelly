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
import os
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
from ..triton_utils import type_dict


@dataclass
class FlexSNKernelHandle:
    inference_kernel: object | None
    inference_info: FlexSNInfo | None
    inference_final_state_kernel: object | None
    inference_final_state_info: FlexSNInfo | None
    forward_kernel: object | None
    forward_final_state_kernel: object | None
    backward_kernel: object | None
    backward_final_state_kernel: object | None
    training_info: FlexSNInfo | None
    state_template_specs: tuple[tuple[tuple[int, ...], torch.dtype, torch.device], ...] | None = None
    owner_refs: int = 1
    active_refs: int = 0


_KERNEL_REGISTRY: dict[int, FlexSNKernelHandle] = {}
_KERNEL_REGISTRY_LOCK = Lock()
_KERNEL_ID_GEN = count(1)


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_TRAINING_FINAL_STATE_SPECIALIZED_MIN_TOKENS = _get_env_int(
    "SJ_FLEXSN_INDUCTOR_TRAINING_FINAL_STATE_MIN_TOKENS", 1 << 20
)
_BACKWARD_FINAL_STATE_SPECIALIZED_MIN_STEPS = _get_env_int(
    "SJ_FLEXSN_INDUCTOR_BACKWARD_FINAL_STATE_MIN_STEPS", 24
)
_BACKWARD_FINAL_STATE_SPECIALIZED_MIN_TOKENS = _get_env_int(
    "SJ_FLEXSN_INDUCTOR_BACKWARD_FINAL_STATE_MIN_TOKENS", 1 << 18
)


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
    forward_final_state_kernel,
    backward_kernel,
    backward_final_state_kernel,
    training_info,
    state_template_specs=None,
) -> int:
    with _KERNEL_REGISTRY_LOCK:
        handle = next(_KERNEL_ID_GEN)
        _KERNEL_REGISTRY[handle] = FlexSNKernelHandle(
            inference_kernel=inference_kernel,
            inference_info=inference_info,
            inference_final_state_kernel=inference_final_state_kernel,
            inference_final_state_info=inference_final_state_info,
            forward_kernel=forward_kernel,
            forward_final_state_kernel=forward_final_state_kernel,
            backward_kernel=backward_kernel,
            backward_final_state_kernel=backward_final_state_kernel,
            training_info=training_info,
            state_template_specs=state_template_specs,
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
        bundle.forward_final_state_kernel,
        bundle.backward_kernel,
        bundle.backward_final_state_kernel,
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
    return torch.empty(shape, dtype=dtype, device=device)


def _materialize_runtime_template_arg(spec):
    shape, dtype, device = spec
    return torch.empty((), dtype=dtype, device=device).expand(shape)


def _resolve_state_template_specs(
    info: FlexSNInfo,
    flat_args: list[torch.Tensor],
    bundle: FlexSNKernelHandle,
):
    if info.num_states == 0:
        return []
    if len(flat_args) >= info.num_inputs + info.num_states:
        return [_template_spec(flat_args[info.num_inputs + i]) for i in range(info.num_states)]
    if bundle.state_template_specs is not None:
        runtime_device = flat_args[0].device if flat_args else None
        specs = []
        for shape, dtype, device in bundle.state_template_specs:
            specs.append((shape, dtype, runtime_device if runtime_device is not None else device))
        return specs
    if not flat_args:
        raise ValueError("Expected at least one FlexSN argument tensor.")
    runtime_template = flat_args[0][0]
    state_template = (tuple(runtime_template.shape), runtime_template.dtype, runtime_template.device)
    return [state_template for _ in range(info.num_states)]


def _materialize_empty_from_spec(spec):
    shape, dtype, device = spec
    return torch.empty(shape, dtype=dtype, device=device)


def _materialize_zeros_from_spec(spec):
    shape, dtype, device = spec
    return torch.zeros(shape, dtype=dtype, device=device)


def _device_guard(tensors: list[torch.Tensor]):
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            return torch.cuda.device(tensor.device)
    return contextlib.nullcontext()


def _visible_fwd_return_count(info: FlexSNInfo) -> int:
    return info.num_outputs + info.num_states


def _extra_saved_return_indices(info: FlexSNInfo) -> list[int]:
    visible = _visible_fwd_return_count(info)
    return [idx for idx in info.c2k_return_mapping if idx >= visible]


def _saved_non_output_indices(info: FlexSNInfo) -> list[int]:
    saved = []
    seen = set()
    for idx in info.c2k_return_mapping:
        if idx < info.num_outputs or idx in seen:
            continue
        saved.append(idx)
        seen.add(idx)
    return saved


def _training_final_state_specialized_wins(info: FlexSNInfo) -> bool:
    specialized_seq_count = info.num_outputs + len(_saved_non_output_indices(info))
    return specialized_seq_count < info.num_fwd_kernel_returns


def _materialize_zero_state_args(
    bundle: FlexSNKernelHandle,
    info: FlexSNInfo,
    flat_args: list[torch.Tensor],
) -> list[torch.Tensor]:
    if len(flat_args) == info.num_inputs:
        # In the common reset-before-forward path, FlexSN initial states are
        # materialized inside the opaque custom op so compile graphs do not
        # need explicit ``zeros_like`` nodes in front of every neuron layer.
        if info.num_inputs == 0:
            raise ValueError("FlexSN custom ops require at least one input sequence.")
        zero_states = [
            _materialize_zeros_from_spec(spec)
            for spec in _resolve_state_template_specs(info, flat_args, bundle)
        ]
        return [*flat_args, *zero_states]
    return flat_args


def _flexsn_inductor_inference_impl(
    bundle: FlexSNKernelHandle, flat_args: list[torch.Tensor]
) -> list[torch.Tensor]:
    args = _materialize_zero_state_args(bundle, bundle.inference_info, flat_args)
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
    args = _materialize_zero_state_args(bundle, bundle.inference_final_state_info, flat_args)
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
    args = _materialize_zero_state_args(bundle, bundle.training_info, flat_args)
    args = [arg.contiguous() for arg in args]
    with _device_guard(args):
        return list(
            flexsn_forward(
                bundle.forward_kernel,
                bundle.training_info,
                *args,
            )
        )


def _flexsn_forward_final_state(
    kernel,
    info: FlexSNInfo,
    state_template_specs,
    *args: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    x_example = args[0]
    T = x_example.shape[0]
    NCL = x_example[0].numel()
    dtype = x_example.dtype
    output_seqs = [torch.empty_like(x_example) for _ in range(info.num_outputs)]
    final_states = [_materialize_empty_from_spec(spec) for spec in state_template_specs]
    saved_tensors = [
        torch.empty_like(x_example) for _ in _saved_non_output_indices(info)
    ]
    grid = lambda meta: ((NCL + meta["BLOCK_NCL"] - 1) // meta["BLOCK_NCL"],)

    kernel[grid](
        *args,
        *output_seqs,
        *final_states,
        *saved_tensors,
        T=T,
        NCL=NCL,
        dtype=type_dict[dtype],
    )
    return tuple([*output_seqs, *final_states, *saved_tensors])


def _should_use_training_final_state_kernel(args: list[torch.Tensor]) -> bool:
    if _TRAINING_FINAL_STATE_SPECIALIZED_MIN_TOKENS <= 0:
        return True
    if not args:
        return False
    return args[0].numel() >= _TRAINING_FINAL_STATE_SPECIALIZED_MIN_TOKENS


def _should_use_backward_final_state_kernel(grad_outputs: list[torch.Tensor]) -> bool:
    if not grad_outputs:
        return False
    grad0 = grad_outputs[0]
    steps_enabled = _BACKWARD_FINAL_STATE_SPECIALIZED_MIN_STEPS > 0
    tokens_enabled = _BACKWARD_FINAL_STATE_SPECIALIZED_MIN_TOKENS > 0
    if steps_enabled and grad0.shape[0] >= _BACKWARD_FINAL_STATE_SPECIALIZED_MIN_STEPS:
        return True
    if tokens_enabled and grad0.numel() >= _BACKWARD_FINAL_STATE_SPECIALIZED_MIN_TOKENS:
        return True
    return not steps_enabled and not tokens_enabled


def _flexsn_inductor_training_final_state_impl(
    bundle: FlexSNKernelHandle, flat_args: list[torch.Tensor]
) -> list[torch.Tensor]:
    args = _materialize_zero_state_args(bundle, bundle.training_info, flat_args)
    args = [arg.contiguous() for arg in args]
    if (
        bundle.forward_final_state_kernel is not None
        and _training_final_state_specialized_wins(bundle.training_info)
        and _should_use_training_final_state_kernel(args)
    ):
        with _device_guard(args):
            return list(
                _flexsn_forward_final_state(
                    bundle.forward_final_state_kernel,
                    bundle.training_info,
                    _resolve_state_template_specs(bundle.training_info, args, bundle),
                    *args,
                )
            )

    full_returns = _flexsn_inductor_training_impl(bundle, flat_args)
    info = bundle.training_info
    assert info is not None
    visible_outputs = list(full_returns[: info.num_outputs])
    state_seqs = list(full_returns[info.num_outputs : info.num_outputs + info.num_states])
    final_states = [state_seq[-1].clone() for state_seq in state_seqs]
    saved_non_output_tensors = [full_returns[i] for i in _saved_non_output_indices(info)]
    return [*visible_outputs, *final_states, *saved_non_output_tensors]


def _flexsn_inductor_backward_impl(
    bundle: FlexSNKernelHandle,
    grad_outputs: list[torch.Tensor],
    saved_tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    return _flexsn_inductor_backward_with_kernel(
        bundle.backward_kernel,
        bundle.training_info,
        grad_outputs,
        saved_tensors,
    )


def _flexsn_inductor_backward_with_kernel(
    kernel,
    info: FlexSNInfo,
    grad_outputs: list[torch.Tensor],
    saved_tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    grads = [grad.contiguous() for grad in grad_outputs]
    saved = [tensor.contiguous() for tensor in saved_tensors]
    with _device_guard([*grads, *saved]):
        return list(
            flexsn_backward(
                kernel,
                info,
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
    state_specs = _resolve_state_template_specs(
        bundle.inference_final_state_info, flat_args, bundle
    )
    final_states = [_materialize_empty_from_spec(spec) for spec in state_specs]
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
    state_specs = _resolve_state_template_specs(bundle.training_info, flat_args, bundle)
    final_states = [_materialize_empty_from_spec(spec) for spec in state_specs]
    extra_saved_tensors = _make_seq_outputs_like(
        bundle.training_info,
        flat_args,
        len(_saved_non_output_indices(bundle.training_info)),
    )
    return [*seq_outputs, *final_states, *extra_saved_tensors]


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
    seq_grads = []
    for i in range(bundle.training_info.num_inputs):
        seq_grads.append(input_templates[i].new_empty(input_templates[i].shape))
    state_specs = _resolve_state_template_specs(
        bundle.training_info,
        input_templates,
        bundle,
    )
    state_grads = [_materialize_empty_from_spec(spec) for spec in state_specs]
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
    T = ctx.input_template_specs[0][0][0]
    ctx.final_state_template_specs = [
        _template_spec(t)
        for t in output[
            bundle.training_info.num_outputs : bundle.training_info.num_outputs + bundle.training_info.num_states
        ]
    ]
    ctx.state_seq_template_specs = [
        ((T, *shape), dtype, device)
        for shape, dtype, device in ctx.final_state_template_specs
    ]
    extra_saved_offset = bundle.training_info.num_outputs + bundle.training_info.num_states
    extra_saved = {
        idx: output[extra_saved_offset + pos]
        for pos, idx in enumerate(_saved_non_output_indices(bundle.training_info))
    }
    saved = []
    for idx in bundle.training_info.c2k_return_mapping:
        if idx < bundle.training_info.num_outputs:
            saved.append(output[idx])
        else:
            saved.append(extra_saved[idx])
    ctx.save_for_backward(*saved)


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
    input_templates = [
        _materialize_runtime_template_arg(spec) for spec in ctx.input_template_specs
    ]
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

    try:
        if ctx._active_ref_finalizer.alive:
            ctx._active_ref_finalizer.detach()
        if (
            bundle.backward_final_state_kernel is not None
            and _should_use_backward_final_state_kernel(output_grads)
        ):
            final_state_grads = []
            for i, final_state_spec in enumerate(ctx.final_state_template_specs):
                final_grad = grad_out[bundle.training_info.num_outputs + i]
                if final_grad is not None:
                    final_state_grads.append(final_grad)
                else:
                    final_state_grads.append(
                        _materialize_zeros_from_spec(final_state_spec)
                    )
            grads = list(
                _flexsn_inductor_backward_with_kernel(
                    bundle.backward_final_state_kernel,
                    bundle.training_info,
                    [*output_grads, *final_state_grads],
                    list(ctx.saved_tensors),
                )
            )
        else:
            input_templates = [
                _materialize_runtime_template_arg(spec)
                for spec in ctx.input_template_specs
            ]
            state_grads = []
            for i, state_seq_spec in enumerate(ctx.state_seq_template_specs):
                final_grad = grad_out[bundle.training_info.num_outputs + i]
                seq_grad = _materialize_zeros_from_spec(state_seq_spec)
                if final_grad is not None:
                    seq_grad[-1].copy_(final_grad)
                state_grads.append(seq_grad)
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
