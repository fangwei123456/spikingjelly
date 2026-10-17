import torch

from .autograd import _check
from .reference import _SURROGATES, _forward_impl as _forward_impl


def _backward_impl(
    gs: torch.Tensor,
    gv: torch.Tensor,
    gm: torch.Tensor,
    h: torch.Tensor,
    m: torch.Tensor,
    m_next: torch.Tensor,
    tau: float,
    threshold: float,
    alpha: float,
    surrogate_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(h, (gs, gv, gm, m, m_next), tau, threshold, alpha, surrogate_id)
    z = h - threshold
    spike = (z >= 0).to(h)
    b = torch.sigmoid(m_next)
    gs_reset = -gv * (threshold + b)
    gm = gm + torch.ops.aten.sigmoid_backward(-gv * spike, b)
    gs = gs + gs_reset + gm
    a = torch.sigmoid(h / tau)
    if surrogate_id == 8:
        gh_spike = torch.zeros_like(gs)
    else:
        gh_spike = _SURROGATES[surrogate_id].backward(gs, z, alpha)
    gh = gv + gh_spike + torch.ops.aten.sigmoid_backward(gm * m, a) / tau
    return gh.contiguous(), (gh / -tau).contiguous(), (gm * a).contiguous()
