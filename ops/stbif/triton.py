import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .validation import _check


@triton.jit
def _kernel(X, V, W, TH, OFF, NEG, OUT, VO, WO, CUR, T, N, BLOCK: tl.constexpr):
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    th = tl.load(TH)
    off = tl.load(OFF)
    neg_bound = tl.load(NEG)
    v = tl.load(V + n, mask, 0)
    w = tl.load(W + n, mask, 0)
    cur = tl.full((BLOCK,), 0.0, tl.float32)
    for t in range(T):
        i = t.to(tl.int64) * N + n
        x = tl.load(X + i, mask, 0).to(tl.float32)
        v = v + x / th
        w = libdevice.rint(w)
        pos = (v >= 1.0) & (w < off)
        neg = (v < 0.0) & (w > neg_bound)
        cur = pos.to(tl.float32) - neg.to(tl.float32)
        w = w + cur
        v = v - pos.to(tl.float32) + neg.to(tl.float32)
        tl.store(OUT + i, cur * th, mask)
    tl.store(VO + n, v, mask)
    tl.store(WO + n, w, mask)
    tl.store(CUR + n, cur, mask)


def _forward_impl(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
    *,
    _kernel_wrapper=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, q, acc_q, q_threshold, pos_max, neg_min)
    torch._check(x.is_cuda, lambda: "Triton stbif requires CUDA")
    x, q, acc_q, q_threshold, pos_max, neg_min = (
        x.contiguous(),
        q.contiguous(),
        acc_q.contiguous(),
        q_threshold.contiguous(),
        pos_max.contiguous(),
        neg_min.contiguous(),
    )
    out = torch.empty_like(x)
    vo = torch.empty_like(q)
    wo = torch.empty_like(q)
    cur = torch.empty_like(q)
    with torch.cuda.device(x.device):
        (_kernel if _kernel_wrapper is None else _kernel_wrapper(_kernel))[
            (triton.cdiv(q.numel(), 256),)
        ](
            x,
            q,
            acc_q,
            q_threshold,
            pos_max,
            neg_min,
            out,
            vo,
            wo,
            cur,
            x.shape[0],
            q.numel(),
            256,
            enable_fp_fusion=False,
        )
    return out, vo, wo, cur
