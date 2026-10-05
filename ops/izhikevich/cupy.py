from pathlib import Path
from typing import Optional

import cupy
import numpy as np
import torch

from ..cupy_loader import _kernel
from .autograd import _check, _check_backward, _register_ops

_SOURCE = str(Path(__file__).with_name("kernels.cuh"))


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    tau: float,
    rest: float,
    critical: float,
    a0: float,
    a: float,
    b: float,
    tau_w: float,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool,
    surrogate_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(
        x,
        v,
        w,
        tau,
        rest,
        critical,
        a0,
        a,
        b,
        tau_w,
        threshold,
        reset,
        detach_reset,
        alpha,
        store_v_seq,
        surrogate_id,
    )
    with torch.cuda.device(x.device), cupy.cuda.Device(x.device.index):
        x, v, w = (x.contiguous(), v.contiguous(), w.contiguous())
        s = torch.empty_like(x)
        vo = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
        wo = torch.empty_like(vo)
        h = torch.empty_like(x, dtype=torch.float32)
        previous = torch.empty_like(h)
        _kernel(_SOURCE, "izhikevich_forward", x.dtype)(
            (min((v.numel() + 255) // 256, 65535),),
            (256,),
            (
                *(np.uint64(t.data_ptr()) for t in (x, v, w, s, vo, wo, h, previous)),
                np.int64(x.shape[0]),
                np.int64(v.numel()),
                np.float32(tau),
                np.float32(rest),
                np.float32(critical),
                np.float32(a0),
                np.float32(a),
                np.float32(b),
                np.float32(tau_w),
                np.float32(threshold),
                np.float32(reset or 0.0),
                np.int32(reset is None),
                np.int32(store_v_seq),
            ),
            stream=cupy.cuda.ExternalStream(
                torch.cuda.current_stream(x.device).cuda_stream
            ),
        )
    return s, vo, wo, h, previous


_forward = torch.library.custom_op(
    "sj_izhikevich::cupy_forward", mutates_args=(), device_types="cuda"
)(_forward_impl)


def _backward_impl(
    gs: torch.Tensor,
    gv: torch.Tensor,
    gw: torch.Tensor,
    h: torch.Tensor,
    previous: torch.Tensor,
    tau: float,
    rest: float,
    critical: float,
    a0: float,
    a: float,
    b: float,
    tau_w: float,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool,
    surrogate_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_backward(
        gs,
        gv,
        gw,
        h,
        previous,
        tau,
        rest,
        critical,
        a0,
        a,
        b,
        tau_w,
        threshold,
        reset,
        detach_reset,
        alpha,
        store_v_seq,
        surrogate_id,
    )
    with torch.cuda.device(h.device), cupy.cuda.Device(h.device.index):
        gs, gv, gw, h, previous = (
            gs.contiguous(),
            gv.contiguous(),
            gw.contiguous(),
            h.contiguous(),
            previous.contiguous(),
        )
        gx = torch.empty_like(gs)
        v0 = torch.empty_like(h[0])
        w0 = torch.empty_like(v0)
        _kernel(_SOURCE, "izhikevich_backward", gs.dtype, surrogate_id)(
            (min((v0.numel() + 255) // 256, 65535),),
            (256,),
            (
                *(
                    np.uint64(t.data_ptr())
                    for t in (gs, gv, gw, h, previous, gx, v0, w0)
                ),
                np.int64(h.shape[0]),
                np.int64(v0.numel()),
                np.float32(tau),
                np.float32(rest),
                np.float32(critical),
                np.float32(a0),
                np.float32(a),
                np.float32(b),
                np.float32(tau_w),
                np.float32(threshold),
                np.float32(reset or 0.0),
                np.int32(reset is None),
                np.int32(detach_reset),
                np.float32(alpha),
                np.int32(store_v_seq),
            ),
            stream=cupy.cuda.ExternalStream(
                torch.cuda.current_stream(h.device).cuda_stream
            ),
        )
    return gx, v0, w0


_backward = torch.library.custom_op(
    "sj_izhikevich::cupy_backward", mutates_args=(), device_types="cuda"
)(_backward_impl)


_register_ops("sj_izhikevich::cupy_forward", "sj_izhikevich::cupy_backward")
