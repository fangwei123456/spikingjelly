import torch

from .reference import _roots


def _forward_impl(
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
