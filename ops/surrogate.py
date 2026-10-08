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


def _surrogate_spec(function):
    if function is None:
        return 0, 4.0
    from spikingjelly.activation_based import surrogate as surrogate_module

    if type(function).__module__ != surrogate_module.__name__:
        return None
    surrogate_id = _SURROGATE_IDS.get(type(function).__name__)
    if surrogate_id is None or not getattr(function, "spiking", True):
        return None
    alpha = function.alpha
    if isinstance(alpha, torch.Tensor) or not isinstance(alpha, (int, float)):
        return None
    return surrogate_id, float(alpha)


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


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, surrogate_id):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.surrogate_id = surrogate_id
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return (
            grad_output * _surrogate_gradient(x, ctx.alpha, ctx.surrogate_id),
            None,
            None,
        )


def _surrogate_spike(x, alpha, surrogate_id):
    return _SurrogateSpike.apply(x, alpha, surrogate_id)
