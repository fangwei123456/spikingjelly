from pathlib import Path
from typing import Optional

import cupy
import numpy as np
import torch

from ..cupy_loader import _kernel
from .autograd import _check_backward, _check_forward, _register_ops

_SOURCE = str(Path(__file__).with_name("kernels.cuh"))


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_forward(x, v, w, threshold, reset, alpha, surrogate_id)
    with torch.cuda.device(x.device), cupy.cuda.Device(x.device.index):
        x, v = x.contiguous(), v.contiguous()
        q = w.float().sigmoid()
        spikes = torch.empty_like(x)
        voltages = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
        charged = torch.empty_like(x, dtype=torch.float32)
        # Torch owns all memory; launch on its current stream, including copies.
        stream = cupy.cuda.ExternalStream(
            torch.cuda.current_stream(x.device).cuda_stream
        )
        _kernel(_SOURCE, "plif_forward", x.dtype)(
            (min((v.numel() + 255) // 256, 65535),),
            (256,),
            (
                *(
                    np.uint64(t.data_ptr())
                    for t in (x, v, q, spikes, voltages, charged)
                ),
                np.int64(x.shape[0]),
                np.int64(v.numel()),
                np.int32(decay_input),
                np.float32(threshold),
                np.float32(0.0 if reset is None else reset),
                np.int32(reset is None),
                np.int32(store_v_seq),
            ),
            stream=stream,
        )
    return spikes, voltages, charged


_forward = torch.library.custom_op(
    "sj_plif::cupy_forward", mutates_args=(), device_types="cuda"
)(_forward_impl)


def _backward_impl(
    gs: torch.Tensor,
    gv: torch.Tensor,
    x: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_backward(
        gs, gv, x, v, w, h, threshold, reset, alpha, store_v_seq, surrogate_id
    )
    with torch.cuda.device(h.device), cupy.cuda.Device(h.device.index):
        gs, gv, h = gs.contiguous(), gv.contiguous(), h.contiguous()
        x, v = x.contiguous(), v.contiguous()
        q = w.float().sigmoid()
        gx = torch.empty_like(h, dtype=gs.dtype)
        gv_init = torch.empty_like(h[0])
        gq = torch.empty_like(v)
        stream = cupy.cuda.ExternalStream(
            torch.cuda.current_stream(h.device).cuda_stream
        )
        _kernel(_SOURCE, "plif_backward", gs.dtype, surrogate_id)(
            (min((gv_init.numel() + 255) // 256, 65535),),
            (256,),
            (
                *(
                    np.uint64(t.data_ptr())
                    for t in (gs, gv, x, v, q, h, gx, gv_init, gq)
                ),
                np.int64(h.shape[0]),
                np.int64(gv_init.numel()),
                np.int32(decay_input),
                np.float32(threshold),
                np.float32(0.0 if reset is None else reset),
                np.int32(reset is None),
                np.int32(detach_reset),
                np.float32(alpha),
                np.int32(store_v_seq),
            ),
            stream=stream,
        )
        gw = (gq.sum() * q * (1 - q)).to(w.dtype)
    return gx, gv_init, gw


_backward = torch.library.custom_op(
    "sj_plif::cupy_backward", mutates_args=(), device_types="cuda"
)(_backward_impl)


_register_ops("sj_plif::cupy_forward", "sj_plif::cupy_backward")
