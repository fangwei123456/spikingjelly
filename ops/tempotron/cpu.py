from typing import Tuple

import torch

from .reference import _phi, _roots, _voltage


def _roots_impl(
    voltage: torch.Tensor,
    current: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    mask: torch.Tensor,
    slow: float,
    fast: float,
    delta: float,
    normalizer: torch.Tensor,
    threshold: float,
    duration: float,
) -> torch.Tensor:
    return _roots(
        voltage,
        current,
        lower,
        upper,
        mask,
        slow,
        fast,
        delta,
        normalizer,
        threshold,
        duration,
    )


def _states_impl(
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


def _weight_gradient_impl(
    times: torch.Tensor,
    indices: torch.Tensor,
    weight: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
    peak: torch.Tensor,
    accepted: torch.Tensor,
    grad_max: torch.Tensor,
    slow: float,
    delta: float,
    normalizer: torch.Tensor,
) -> torch.Tensor:
    gradient = torch.zeros_like(weight, dtype=torch.float64)
    batches = torch.repeat_interleave(
        torch.arange(peak.shape[0], device=times.device),
        offsets.diff(),
        output_size=order.numel(),
    )
    ranks = torch.arange(order.numel(), device=times.device) - offsets[batches]
    for start in range(0, order.numel(), 16384):
        pos, batch = order[start : start + 16384], batches[start : start + 16384]
        event_times = times[pos].double()[None, :]
        delay = (peak[batch].T - event_times).clamp_min(0)
        valid = (ranks[start : start + 16384][None, :] < accepted[batch].T) & (
            event_times < peak[batch].T
        )
        value = torch.exp(-delay / slow) * _phi(delay, delta) / normalizer
        gradient.index_add_(
            1, indices[pos], torch.where(valid, grad_max[batch].T * value, 0)
        )
    return gradient
