from typing import Optional

import torch

from ..validation import _check_inputs


def _check(
    x: torch.Tensor,
    v: torch.Tensor,
    threshold: torch.Tensor,
    offset: torch.Tensor,
    channels: int,
    inner: int,
    reset: Optional[float],
    store_v_seq: bool,
):
    _check_inputs(x, v, 1.0, reset, 4.0)
    for tensor in (x, v, threshold, offset):
        torch._check(
            not tensor.requires_grad,
            lambda: "registered inference transitions do not support autograd",
        )
    if channels <= 0 or inner <= 0:
        raise ValueError("channel dimensions must be positive")
    torch._check(
        v.numel() % (channels * inner) == 0,
        lambda: "channel dimensions must divide the state size",
    )
    for tensor in (threshold, offset):
        torch._check(
            tensor.numel() in (1, channels),
            lambda: "parameters must be scalar or match channel count",
        )
    for tensor in (threshold, offset):
        torch._check(
            tensor.device == x.device
            and tensor.dtype == torch.float32
            and tensor.layout == torch.strided,
            lambda: "parameters must be strided FP32 on the input device",
        )


def _forward_fake(
    x: torch.Tensor,
    v: torch.Tensor,
    threshold: torch.Tensor,
    offset: torch.Tensor,
    channels: int,
    inner: int,
    reset: Optional[float],
    store_v_seq: bool,
):
    _check(x, v, threshold, offset, channels, inner, reset, store_v_seq)
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(
            x if store_v_seq else v,
            dtype=torch.float32,
            memory_format=torch.contiguous_format,
        ),
    )
