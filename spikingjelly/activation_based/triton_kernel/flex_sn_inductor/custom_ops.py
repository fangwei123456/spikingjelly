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

from dataclasses import dataclass
from itertools import count
from threading import Lock
import weakref

import torch

from ..flexsn.wrapper import flexsn_backward, flexsn_forward, flexsn_inference
from ..flexsn.info import FlexSNInfo


@dataclass
class FlexSNKernelHandle:
    inference_kernel: object | None
    inference_info: FlexSNInfo | None
    forward_kernel: object | None
    backward_kernel: object | None
    training_info: FlexSNInfo | None


_KERNEL_REGISTRY: dict[int, FlexSNKernelHandle] = {}
_KERNEL_REGISTRY_LOCK = Lock()
_KERNEL_ID_GEN = count(1)


def register_flexsn_kernel_handle(
    *,
    inference_kernel,
    inference_info,
    forward_kernel,
    backward_kernel,
    training_info,
) -> int:
    with _KERNEL_REGISTRY_LOCK:
        handle = next(_KERNEL_ID_GEN)
        _KERNEL_REGISTRY[handle] = FlexSNKernelHandle(
            inference_kernel=inference_kernel,
            inference_info=inference_info,
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
    for obj in (
        bundle.inference_kernel,
        bundle.forward_kernel,
        bundle.backward_kernel,
    ):
        closer = getattr(obj, "close", None)
        if callable(closer):
            closer()


def _lookup_kernel_handle(handle: int) -> FlexSNKernelHandle:
    try:
        return _KERNEL_REGISTRY[handle]
    except KeyError as e:
        raise RuntimeError(f"Unknown FlexSN kernel handle: {handle}") from e


def _make_seq_tensor_list_like(xs: list[torch.Tensor], n: int) -> list[torch.Tensor]:
    if not xs:
        raise ValueError("Expected at least one template tensor.")
    return [xs[min(i, len(xs) - 1)].new_empty(xs[min(i, len(xs) - 1)].shape) for i in range(n)]


def _make_seq_outputs_like(
    info: FlexSNInfo, flat_args: list[torch.Tensor], n: int
) -> list[torch.Tensor]:
    if not flat_args:
        raise ValueError("Expected at least one FlexSN argument tensor.")
    seq_template = flat_args[0]
    return [seq_template.new_empty(seq_template.shape) for _ in range(n)]


@torch.library.custom_op("sj::flexsn_inductor_inference", mutates_args=())
def flexsn_inductor_inference(handle: int, flat_args: list[torch.Tensor]) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.inference_kernel is None or bundle.inference_info is None:
        raise RuntimeError("FlexSN inference kernel is unavailable for this handle.")
    return list(
        flexsn_inference(
            bundle.inference_kernel,
            bundle.inference_info,
            *(arg.contiguous() for arg in flat_args),
        )
    )


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


@torch.library.custom_op("sj::flexsn_inductor_training", mutates_args=())
def flexsn_inductor_training(handle: int, flat_args: list[torch.Tensor]) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if (
        bundle.forward_kernel is None
        or bundle.backward_kernel is None
        or bundle.training_info is None
    ):
        raise RuntimeError("FlexSN training kernels are unavailable for this handle.")
    return list(
        flexsn_forward(
            bundle.forward_kernel,
            bundle.training_info,
            *(arg.contiguous() for arg in flat_args),
        )
    )


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


@torch.library.custom_op("sj::flexsn_inductor_backward", mutates_args=())
def flexsn_inductor_backward(
    handle: int,
    grad_outputs: list[torch.Tensor],
    saved_tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.backward_kernel is None or bundle.training_info is None:
        raise RuntimeError("FlexSN backward kernel is unavailable for this handle.")
    return list(
        flexsn_backward(
            bundle.backward_kernel,
            bundle.training_info,
            *(grad.contiguous() for grad in grad_outputs),
            *(tensor.contiguous() for tensor in saved_tensors),
        )
    )


@torch.library.register_fake("sj::flexsn_inductor_backward")
def _flexsn_inductor_backward_fake(
    handle: int,
    grad_outputs: list[torch.Tensor],
    saved_tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    seq_grads = _make_seq_tensor_list_like(grad_outputs, bundle.training_info.num_inputs)
    state_offset = bundle.training_info.num_outputs
    state_grads = [
        grad_outputs[state_offset + i][0].new_empty(grad_outputs[state_offset + i][0].shape)
        for i in range(bundle.training_info.num_states)
    ]
    return [*seq_grads, *state_grads]


def _flexsn_training_setup_context(ctx, inputs, output):
    handle = inputs[0]
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    ctx.handle = handle
    ctx.output_templates = output[: bundle.training_info.num_outputs + bundle.training_info.num_states]
    saved = [output[i] for i in bundle.training_info.c2k_return_mapping]
    ctx.save_for_backward(*saved)


def _flexsn_training_backward(ctx, grad_out: list[torch.Tensor | None]):
    bundle = _lookup_kernel_handle(ctx.handle)
    if bundle.backward_kernel is None or bundle.training_info is None:
        raise RuntimeError("FlexSN backward kernel is unavailable for this handle.")

    required_grads = bundle.training_info.num_outputs + bundle.training_info.num_states
    grad_inputs = [
        grad_out[i].contiguous()
        if grad_out[i] is not None
        else torch.zeros_like(ctx.output_templates[i])
        for i in range(required_grads)
    ]
    saved = [tensor.contiguous() for tensor in ctx.saved_tensors]
    grads = list(
        flexsn_inductor_backward(
            ctx.handle,
            grad_inputs,
            saved,
        )
    )
    return None, grads


torch.library.register_autograd(
    "sj::flexsn_inductor_training",
    _flexsn_training_backward,
    setup_context=_flexsn_training_setup_context,
)


def attach_flexsn_handle_finalizer(owner, handle: int):
    return weakref.finalize(owner, unregister_flexsn_kernel_handle, handle)
