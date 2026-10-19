import torch

from .validation import _check


def _forward_impl(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, q, acc_q, q_threshold, pos_max, neg_min)
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    # Subsequent updates are integers, so only the incoming count needs rounding.
    acc_q = acc_q.round()
    neg_min = neg_min.reshape(())
    pos_max = pos_max.reshape(())
    q_threshold = q_threshold.reshape(())
    for t, current in enumerate(x):
        q = q + current.float() / q_threshold
        pos = ((q >= 1) & (acc_q < pos_max)).float()
        neg = ((q < 0) & (acc_q > neg_min)).float()
        cur = pos - neg
        acc_q.add_(cur)
        q.sub_(pos).add_(neg)
        torch.mul(cur, q_threshold, out=out[t])
    return (out, q.contiguous(), acc_q.contiguous())
