def step(x, q, acc_q, q_threshold, pos_max, neg_min):
    q_threshold = q_threshold.to(device=x.device, dtype=x.dtype)
    pos_max = pos_max.to(device=x.device, dtype=x.dtype)
    neg_min = neg_min.to(device=x.device, dtype=x.dtype)
    q = q + x / q_threshold
    pos = (q >= 1) & (acc_q < pos_max)
    neg = (q < 0) & (acc_q > neg_min)
    current = pos.to(x.dtype) - neg.to(x.dtype)
    acc_q = acc_q + current
    q = q - pos.float() + neg.float()
    return current * q_threshold, q, acc_q
