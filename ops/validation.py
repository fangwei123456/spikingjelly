import math

import torch

from .surrogate import _DTYPES


def _check_inputs(x, v, threshold, reset, alpha, surrogate_id: int = 0):
    torch._check(x.ndim >= 2, lambda: "x must have shape [T, ...]")
    torch._check(x.shape[0] > 0, lambda: "T must be positive")
    torch._check(v.numel() > 0, lambda: "neuron dimensions must be nonempty")
    torch._check(x.shape[1:] == v.shape, lambda: "state shape must match x[0]")
    torch._check(x.device == v.device, lambda: "input/state devices must match")
    torch._check(
        x.dtype in _DTYPES and v.dtype == torch.float32,
        lambda: "input must be float32, float16 or bfloat16; state must be float32",
    )
    torch._check(
        x.layout == torch.strided and v.layout == torch.strided,
        lambda: "only strided tensors are supported",
    )
    if surrogate_id not in range(7):
        raise ValueError("surrogate_id must be in [0, 6]")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(threshold) or (reset is not None and not math.isfinite(reset)):
        raise ValueError("threshold and reset must be finite")


def _check_gradients(
    gs, gv, h, threshold, reset, alpha, store_v_seq: bool = True, surrogate_id: int = 0
):
    _check_inputs(h, h[0], threshold, reset, alpha, surrogate_id)
    for index, grad in enumerate((gs, gv)):
        torch._check(grad.device == h.device, lambda: "gradient devices must match")
        torch._check(
            grad.dtype in _DTYPES if index == 0 else grad.dtype == torch.float32,
            lambda: (
                "spike gradient must be floating and voltage gradient must be float32"
            ),
        )
        expected = h.shape if index == 0 or store_v_seq else h.shape[1:]
        torch._check(grad.shape == expected, lambda: "gradient shapes must match")
        torch._check(grad.layout == torch.strided, lambda: "gradients must be strided")
