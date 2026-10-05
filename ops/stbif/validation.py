import torch

from ..validation import _check_inputs


def _check(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
):
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
):
    _check(x, q, acc_q, q_threshold, pos_max, neg_min)
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(q, memory_format=torch.contiguous_format),
    )
