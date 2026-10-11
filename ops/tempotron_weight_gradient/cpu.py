import torch

from ..tempotron_roots.reference import _phi


def _forward_impl(
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
