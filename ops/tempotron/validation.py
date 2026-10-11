import torch


def _roots_fake(
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
    *,
    _strided=False,
):
    for tensor in (current, lower, upper, mask, normalizer):
        torch._check(tensor.device == voltage.device)
    torch._check(voltage.dtype == torch.float64)
    torch._check(
        current.shape == voltage.shape
        and lower.shape == voltage.shape
        and upper.shape == voltage.shape
        and mask.shape == voltage.shape
    )
    torch._check(mask.dtype == torch.bool and normalizer.numel() == 1)
    return torch.empty_like(upper)


def _states_fake(
    times,
    incoming,
    voltage,
    current,
    origin,
    slow,
    fast,
    delta,
    normalizer,
    *,
    _strided=False,
):
    for tensor in (incoming, voltage, current, origin, normalizer):
        torch._check(tensor.device == times.device and tensor.dtype == torch.float64)
    torch._check(times.ndim == 2 and times.dtype == torch.float64)
    torch._check(
        incoming.ndim == 3
        and incoming.shape[0] == times.shape[0]
        and incoming.shape[2] == times.shape[1]
    )
    torch._check(voltage.shape == incoming.shape[:2] and current.shape == voltage.shape)
    torch._check(origin.shape == times.shape[:1] and normalizer.numel() == 1)
    return torch.empty_like(incoming), torch.empty_like(incoming)


def _weight_gradient_fake(
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
