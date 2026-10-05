from pathlib import Path

import cupy
import numpy as np
import torch

from ..cupy_loader import _kernel
from .validation import _check, _forward_fake

_SOURCE = str(Path(__file__).with_name("kernels.cuh"))


def _forward_impl(
    x: torch.Tensor,
    q: torch.Tensor,
    acc_q: torch.Tensor,
    q_threshold: torch.Tensor,
    pos_max: torch.Tensor,
    neg_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, q, acc_q, q_threshold, pos_max, neg_min)
    with torch.cuda.device(x.device), cupy.cuda.Device(x.device.index):
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
        _kernel(_SOURCE, "stbif_forward", x.dtype)(
            (min((q.numel() + 255) // 256, 65535),),
            (256,),
            (
                *(
                    np.uint64(t.data_ptr())
                    for t in (
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
                    )
                ),
                np.int64(x.shape[0]),
                np.int64(q.numel()),
            ),
            stream=cupy.cuda.ExternalStream(
                torch.cuda.current_stream(x.device).cuda_stream
            ),
        )
    return out, vo, wo, cur


_forward = torch.library.custom_op(
    "sj_stbif::cupy_forward", mutates_args=(), device_types="cuda"
)(_forward_impl)


torch.library.register_fake("sj_stbif::cupy_forward", _forward_fake)
