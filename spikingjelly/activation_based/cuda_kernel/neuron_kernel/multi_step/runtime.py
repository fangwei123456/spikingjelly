"""Runtime context helpers for handwritten multi-step custom operators."""

import math
import threading

import torch
from torch._subclasses.fake_tensor import is_fake

from spikingjelly.logger import logger

try:
    import cupy
except (ImportError, OSError) as e:
    logger.debug(
        "spikingjelly.activation_based.cuda_kernel.neuron_kernel.multi_step: %s", e
    )
    cupy = None


def _decode_v_reset(v_reset_value: float):
    # Custom-op schemas use NaN because Optional[float] is unsupported here.
    return None if math.isnan(v_reset_value) else v_reset_value


class _CapturedAutogradCtx:
    saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


_CAPTURE_CTX_LOCK = threading.Lock()
_CAPTURE_CTX_BY_ID: dict[int, _CapturedAutogradCtx] = {}
_CAPTURE_CTX_NEXT_ID = 0


def _stash_capture_ctx(captured_ctx: _CapturedAutogradCtx) -> int:
    global _CAPTURE_CTX_NEXT_ID
    with _CAPTURE_CTX_LOCK:
        _CAPTURE_CTX_NEXT_ID += 1
        capture_id = _CAPTURE_CTX_NEXT_ID
        _CAPTURE_CTX_BY_ID[capture_id] = captured_ctx
    return capture_id


def _should_stash_capture_ctx(inputs, capture_context: bool | None = None) -> bool:
    if capture_context is None:
        capture_context = torch.is_grad_enabled()
    if not capture_context:
        return False
    return any(isinstance(item, torch.Tensor) and item.requires_grad for item in inputs)


def _capture_token(captured_ctx, inputs, device, capture_context: bool | None = None):
    del device
    capture_id = (
        _stash_capture_ctx(captured_ctx)
        if _should_stash_capture_ctx(inputs, capture_context)
        else -1
    )
    return torch.tensor(capture_id, dtype=torch.int64)


def _take_capture_ctx(capture_id: int) -> _CapturedAutogradCtx:
    with _CAPTURE_CTX_LOCK:
        captured_ctx = _CAPTURE_CTX_BY_ID.pop(capture_id, None)
    if captured_ctx is None:
        raise RuntimeError(f"Unknown capture context id={capture_id}")
    return captured_ctx


def _setup_capture_ctx(ctx, inputs, output):
    del inputs
    capture_token = output[-1]
    if capture_token.is_meta or is_fake(capture_token):
        ctx.captured = None
        return
    capture_id = int(capture_token.item())
    ctx.captured = None if capture_id < 0 else _take_capture_ctx(capture_id)


def _surrogate_cuda_dtype(dtype) -> str:
    if dtype == "fp32" or dtype == torch.float32:
        return "float"
    if dtype == "fp16" or dtype == torch.float16:
        return "half2"
    raise NotImplementedError(dtype)
