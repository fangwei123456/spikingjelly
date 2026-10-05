import math

import torch

_SURROGATE_IDS = {
    "Sigmoid": 0,
    "ATan": 1,
    "PiecewiseQuadratic": 2,
    "PiecewiseExp": 3,
    "SoftSign": 4,
    "SuperSpike": 5,
    "Erf": 6,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _surrogate_gradient(x, alpha, surrogate_id):
    if surrogate_id == 0:
        s = torch.sigmoid(alpha * x)
        return (1 - s) * s * alpha
    if surrogate_id == 1:
        z = x * (math.pi / 2 * alpha)
        return (alpha / 2) / (1 + z * z)
    if surrogate_id == 2:
        return (alpha - alpha * alpha * x.abs()).clamp_min(0)
    if surrogate_id == 3:
        return (alpha / 2) * torch.exp(-alpha * x.abs())
    if surrogate_id == 4:
        z = 1 / alpha + x.abs()
        return 1 / (2 * alpha * z * z)
    if surrogate_id == 5:
        z = 1 + x.abs()
        return alpha / (z * z)
    if surrogate_id == 6:
        z = alpha * x
        return (alpha / math.sqrt(math.pi)) * torch.exp(-z * z)
    raise ValueError("surrogate_id must be in [0, 6]")
