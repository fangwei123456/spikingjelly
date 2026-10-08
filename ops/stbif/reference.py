import torch

from .validation import _check


def step(x, q, acc_q, q_threshold, pos_max, neg_min):
    q_threshold = q_threshold.to(device=x.device, dtype=x.dtype)
    pos_max = pos_max.to(device=x.device, dtype=x.dtype)
    neg_min = neg_min.to(device=x.device, dtype=x.dtype)
    q = q + x / q_threshold
    acc_q = acc_q.round()
    pos = (q >= 1) & (acc_q < pos_max)
    neg = (q < 0) & (acc_q > neg_min)
    current = pos.to(x.dtype) - neg.to(x.dtype)
    acc_q = acc_q + current
    q = q - pos.float() + neg.float()
    return current * q_threshold, q, acc_q, current


def _forward_impl(x_seq, q, acc_q, q_threshold, pos_max, neg_min):
    _check(x_seq, q, acc_q, q_threshold, pos_max, neg_min)
    outputs = []
    cur_output = torch.zeros_like(q)
    for current in x_seq.float():
        output, q, acc_q, cur_output = step(
            current, q, acc_q, q_threshold, pos_max, neg_min
        )
        outputs.append(output.to(x_seq.dtype))
    return (
        torch.stack(outputs),
        q.contiguous(),
        acc_q.contiguous(),
        cur_output.contiguous(),
    )
