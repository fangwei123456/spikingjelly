from pathlib import Path

import cupy
import numpy as np
import torch

from ..cupy_loader import _kernel
from .autograd import _check, _check_backward, _register_ops

_SOURCE = str(Path(__file__).with_name("kernels.cuh"))


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    count: float,
    lower: float,
    upper: float,
    threshold: float,
    detach_reset: bool,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, v, tau, count, lower, upper, threshold, detach_reset, store_v_seq)
    with torch.cuda.device(x.device), cupy.cuda.Device(x.device.index):
        x, v = (x.contiguous(), v.contiguous())
        s = torch.empty_like(x)
        vo = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
        h = torch.empty_like(x, dtype=torch.float32)
        _kernel(_SOURCE, "ilif_forward", x.dtype)(
            (min((v.numel() + 255) // 256, 65535),),
            (256,),
            (
                *(np.uint64(t.data_ptr()) for t in (x, v, s, vo, h)),
                np.int64(x.shape[0]),
                np.int64(v.numel()),
                np.float32(tau),
                np.float32(count),
                np.float32(threshold),
                np.int32(store_v_seq),
            ),
            stream=cupy.cuda.ExternalStream(
                torch.cuda.current_stream(x.device).cuda_stream
            ),
        )
    return s, vo, h


_forward = torch.library.custom_op(
    "sj_ilif::cupy_forward", mutates_args=(), device_types="cuda"
)(_forward_impl)


def _backward_impl(
    gs: torch.Tensor,
    gv: torch.Tensor,
    h: torch.Tensor,
    tau: float,
    count: float,
    lower: float,
    upper: float,
    threshold: float,
    detach_reset: bool,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_backward(
        gs, gv, h, tau, count, lower, upper, threshold, detach_reset, store_v_seq
    )
    with torch.cuda.device(h.device), cupy.cuda.Device(h.device.index):
        gs, gv, h = (gs.contiguous(), gv.contiguous(), h.contiguous())
        gx = torch.empty_like(gs)
        v0 = torch.empty_like(h[0])
        _kernel(_SOURCE, "ilif_backward", gs.dtype)(
            (min((v0.numel() + 255) // 256, 65535),),
            (256,),
            (
                *(np.uint64(t.data_ptr()) for t in (gs, gv, h, gx, v0)),
                np.int64(h.shape[0]),
                np.int64(v0.numel()),
                np.float32(tau),
                np.float32(lower),
                np.float32(upper),
                np.float32(threshold),
                np.int32(detach_reset),
                np.int32(store_v_seq),
            ),
            stream=cupy.cuda.ExternalStream(
                torch.cuda.current_stream(h.device).cuda_stream
            ),
        )
    return gx, v0


_backward = torch.library.custom_op(
    "sj_ilif::cupy_backward", mutates_args=(), device_types="cuda"
)(_backward_impl)


_register_ops("sj_ilif::cupy_forward", "sj_ilif::cupy_backward")
