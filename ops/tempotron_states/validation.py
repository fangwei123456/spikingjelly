import torch


def _forward_fake(
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
