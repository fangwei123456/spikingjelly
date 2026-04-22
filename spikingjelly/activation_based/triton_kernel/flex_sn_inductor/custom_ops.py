from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from threading import Lock

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


def _lookup_kernel_handle(handle: int) -> FlexSNKernelHandle:
    try:
        return _KERNEL_REGISTRY[handle]
    except KeyError as e:
        raise RuntimeError(f"Unknown FlexSN kernel handle: {handle}") from e


def _make_seq_tensor_list_like(xs: list[torch.Tensor], n: int) -> list[torch.Tensor]:
    x_example = xs[0]
    return [x_example.new_empty(x_example.shape) for _ in range(n)]


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
    return _make_seq_tensor_list_like(
        flat_args, bundle.inference_info.num_outputs + bundle.inference_info.num_states
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
    return _make_seq_tensor_list_like(
        flat_args, bundle.training_info.num_fwd_kernel_returns
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
    seq_grads = _make_seq_tensor_list_like(
        grad_outputs, bundle.training_info.num_inputs
    )
    state_example = grad_outputs[0][0]
    state_grads = [
        state_example.new_empty(state_example.shape)
        for _ in range(bundle.training_info.num_states)
    ]
    return [*seq_grads, *state_grads]


def _flexsn_training_setup_context(ctx, inputs, output):
    handle = inputs[0]
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    ctx.handle = handle
    ctx.output_template = output[0]
    saved = [output[i] for i in bundle.training_info.c2k_return_mapping]
    ctx.save_for_backward(*saved)


def _flexsn_training_backward(ctx, grad_out: list[torch.Tensor | None]):
    bundle = _lookup_kernel_handle(ctx.handle)
    if bundle.backward_kernel is None or bundle.training_info is None:
        raise RuntimeError("FlexSN backward kernel is unavailable for this handle.")

    required_grads = bundle.training_info.num_outputs + bundle.training_info.num_states
    zero = torch.zeros_like(ctx.output_template)
    grad_inputs = [
        grad_out[i].contiguous() if grad_out[i] is not None else zero
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
