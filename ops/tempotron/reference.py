import math

import torch


def _phi(delay, delta):
    z = delta * delay
    series = delay * (1 - z / 2 + z.square() / 6)
    return torch.where(
        z.abs() < math.sqrt(torch.finfo(torch.float64).eps),
        series,
        -torch.expm1(-z) / delta,
    )


def _voltage(voltage, current, delay, slow, delta, normalizer):
    decay = torch.exp(-delay / slow)
    return voltage * decay / normalizer + current * (
        decay * _phi(delay, delta) / normalizer
    )


def _stationary(voltage, current, slow, delta):
    nonzero = current != 0
    ratio = voltage / torch.where(nonzero, current, 1)
    denominator = 1 + delta * ratio
    valid = (
        nonzero
        & torch.isfinite(ratio)
        & torch.isfinite(denominator)
        & (denominator > 0)
    )
    z = (
        delta
        * (slow - torch.where(valid, ratio, 0))
        / torch.where(valid, denominator, 1)
    )
    valid = valid & torch.isfinite(z) & (z > -1)
    delay = torch.log1p(torch.where(valid, z, 0)) / delta
    return torch.where(valid, delay, torch.inf)


def _roots(
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
):
    eps = torch.finfo(torch.float64).eps
    width = upper - lower
    absolute = 8 * eps * torch.minimum(width, width.new_tensor(fast))
    absolute = absolute.clamp_min(math.ulp(0.0))
    lower = torch.where(mask, lower, 0)
    upper = torch.where(mask, upper, 0)
    iterations = (
        math.ceil(max(0.0, math.log2(duration) - math.log2(fast)) - math.log2(8 * eps))
        + 2
    )
    for _ in range(iterations):
        middle = lower + (upper - lower) / 2
        tolerance = absolute + 8 * eps * torch.maximum(lower.abs(), upper.abs())
        moving = (
            mask & ((upper - lower) > tolerance) & (middle > lower) & (middle < upper)
        )
        value = _voltage(voltage, current, middle, slow, delta, normalizer)
        below = value < threshold
        lower = torch.where(moving & below, middle, lower)
        upper = torch.where(moving & ~below, middle, upper)
    tolerance = absolute + 8 * eps * torch.maximum(lower.abs(), upper.abs())
    if bool(
        (
            mask
            & ((upper - lower) > tolerance)
            & (torch.nextafter(lower, upper) < upper)
        ).any()
    ):
        raise RuntimeError("Tempotron threshold root did not converge.")
    root = upper
    value = _voltage(voltage, current, root, slow, delta, normalizer)
    raw = value * normalizer
    q = current * torch.exp(-root / fast)
    slope = (q - raw / slow) / normalizer
    scale = (
        1
        + threshold
        + (voltage.abs() + (current * _phi(root, delta)).abs()) / normalizer
    )
    residual_tolerance = 64 * eps * scale + slope.abs() * tolerance
    if bool(
        (
            mask
            & (
                ~torch.isfinite(value)
                | ((value - threshold).abs() > residual_tolerance)
            )
        ).any()
    ):
        raise RuntimeError("Tempotron threshold root has an invalid voltage residual.")
    return root


def _psp(delay, slow, fast, delta, normalizer):
    decay = torch.exp(-delay / slow)
    fast_decay = torch.exp(-delta * delay)
    phi = _phi(delay, delta)
    value = decay * phi / normalizer
    first = decay * (fast_decay - phi / slow) / normalizer
    second = decay * (phi / slow / slow - (2 / slow + delta) * fast_decay) / normalizer
    return value, first, second
