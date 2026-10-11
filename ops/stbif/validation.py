import torch

from ..layout import _fake_empty_like
from ..validation import _check_inputs


def _check_reference(x, q, acc_q, q_threshold, pos_max, neg_min):
    tensors = (x, q, acc_q, q_threshold, pos_max, neg_min)
    if any(tensor.layout != torch.strided for tensor in tensors):
        raise ValueError("STBIF tensors must use strided layout")
    if (
        x.ndim < 2
        or x.shape[0] == 0
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    ):
        raise ValueError("STBIF input must be a nonempty floating-point sequence")
    if (
        q.shape != x.shape[1:]
        or acc_q.shape != q.shape
        or q.device != x.device
        or acc_q.device != x.device
        or q.dtype != x.dtype
        or acc_q.dtype != x.dtype
    ):
        raise ValueError("state shape, dtype, and device must match")
    if any(parameter.numel() != 1 for parameter in (q_threshold, pos_max, neg_min)):
        raise ValueError("parameters must be scalar tensors")
    if any(
        not parameter.is_floating_point()
        for parameter in (q_threshold, pos_max, neg_min)
    ):
        raise ValueError("STBIF parameters must be floating-point tensors")


def _check(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
):
    if any(tensor.numel() != 1 for tensor in (q_threshold, pos_max, neg_min)):
        raise ValueError("parameters must be scalar tensors")
    if (
        q.shape != x.shape[1:]
        or q.dtype != torch.float32
        or q.device != x.device
        or acc_q.shape != q.shape
        or acc_q.dtype != torch.float32
        or acc_q.device != q.device
    ):
        raise ValueError("state shape, dtype, and device must match")
    _check_inputs(x, q, 1.0, None, 4.0)
    for tensor in (x, q, acc_q, q_threshold, pos_max, neg_min):
        torch._check(
            not tensor.requires_grad,
            lambda: "registered inference transitions do not support autograd",
        )
    torch._check(
        acc_q.shape == q.shape
        and acc_q.device == q.device
        and acc_q.dtype == torch.float32
        and acc_q.layout == torch.strided,
        lambda: "acc_q must match q shape/device and be strided FP32",
    )
    for tensor in (q_threshold, pos_max, neg_min):
        torch._check(tensor.numel() == 1, lambda: "parameters must be scalar tensors")
    for tensor in (q_threshold, pos_max, neg_min):
        torch._check(
            tensor.device == x.device
            and tensor.dtype == torch.float32
            and tensor.layout == torch.strided,
            lambda: "parameters must be strided FP32 on the input device",
        )


def _forward_fake(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
    *,
    _strided=False,
):
    _check(x, q, acc_q, q_threshold, pos_max, neg_min)
    return (
        _fake_empty_like(x, strided=_strided),
        _fake_empty_like(q, strided=_strided),
        _fake_empty_like(q, strided=_strided),
        _fake_empty_like(q, strided=_strided),
    )
