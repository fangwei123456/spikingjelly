import torch


def _forward_fake(
    times,
    indices,
    weight,
    order,
    offsets,
    peak,
    accepted,
    grad_max,
    slow,
    delta,
    normalizer,
    *,
    _strided=False,
):
    for tensor in (
        indices,
        weight,
        order,
        offsets,
        peak,
        accepted,
        grad_max,
        normalizer,
    ):
        torch._check(tensor.device == times.device)
    torch._check(weight.ndim == 2 and peak.ndim == 2)
    torch._check(peak.shape == accepted.shape and peak.shape == grad_max.shape)
    torch._check(
        peak.shape[1] == weight.shape[0] and offsets.numel() == peak.shape[0] + 1
    )
    torch._check(times.ndim == 1 and indices.shape == times.shape)
    torch._check(
        indices.dtype == torch.int64
        and order.dtype == torch.int64
        and offsets.dtype == torch.int64
        and accepted.dtype == torch.int64
    )
    torch._check(
        peak.dtype == torch.float64
        and grad_max.dtype == torch.float64
        and normalizer.numel() == 1
    )
    return torch.empty(weight.shape, device=weight.device, dtype=torch.float64)
