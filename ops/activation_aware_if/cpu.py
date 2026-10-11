from typing import Optional

import torch

from .validation import _check


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    threshold: torch.Tensor,
    offset: torch.Tensor,
    channels: int,
    inner: int,
    reset: Optional[float],
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check(x, v, threshold, offset, channels, inner, reset, store_v_seq)
    outputs, voltages = ([], [])
    if threshold.numel() > 1 or offset.numel() > 1:
        indices = (
            torch.arange(v.numel(), device=x.device) // inner % channels
        ).reshape(v.shape)
    threshold = (
        threshold.reshape(-1)[indices]
        if threshold.numel() > 1
        else threshold.reshape(())
    )
    offset = offset.reshape(-1)[indices] if offset.numel() > 1 else offset.reshape(())
    for current in x.float():
        h = v + current
        cur = (h + offset >= threshold).float()
        v = h - cur * threshold if reset is None else cur * reset + (1 - cur) * h
        output = cur.to(x.dtype)
        outputs.append(output)
        if store_v_seq:
            voltages.append(v)
    return (torch.stack(outputs), torch.stack(voltages) if voltages else v.contiguous())
