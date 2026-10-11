from typing import Tuple

import torch

from ..tempotron_roots.reference import _phi, _voltage


def _forward_impl(
    times: torch.Tensor,
    incoming: torch.Tensor,
    voltage: torch.Tensor,
    current: torch.Tensor,
    origin: torch.Tensor,
    slow: float,
    fast: float,
    delta: float,
    normalizer: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    delay = (times[:, :, None] - times[:, None, :]).clamp_min(0)
    kernel = torch.exp(-delay / slow) * _phi(delay, delta) / normalizer
    elapsed = (times - origin[:, None])[:, None, :]
    block_voltage = _voltage(
        voltage[..., None], current[..., None], elapsed, slow, delta, normalizer
    ) + incoming @ kernel.transpose(-1, -2)
    decay = torch.exp(-delay / fast).tril()
    block_current = current[..., None] * torch.exp(
        -elapsed / fast
    ) + incoming @ decay.transpose(-1, -2)
    return block_voltage, block_current
