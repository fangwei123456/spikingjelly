import torch

from .validation import _check


def _forward_impl(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, q, acc_q, q_threshold, pos_max, neg_min)
    outputs = []
    neg_min = neg_min.reshape(())
    pos_max = pos_max.reshape(())
    q_threshold = q_threshold.reshape(())
    for current in x.float():
        q = q + current / q_threshold
        acc_q = acc_q.round()
        pos = (q >= 1) & (acc_q < pos_max)
        neg = (q < 0) & (acc_q > neg_min.reshape(()))
        cur = pos.float() - neg.float()
        acc_q = acc_q + cur
        q = q - pos.float() + neg.float()
        output = (cur * q_threshold).to(x.dtype)
        outputs.append(output)
    return (torch.stack(outputs), q.contiguous(), acc_q.contiguous(), cur.contiguous())
