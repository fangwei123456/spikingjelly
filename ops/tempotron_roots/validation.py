import torch


def _forward_fake(
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
