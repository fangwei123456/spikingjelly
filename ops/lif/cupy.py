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
    tau: float,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_forward(x, v, tau, threshold, reset, alpha, surrogate_id)
    with torch.cuda.device(x.device), cupy.cuda.Device(x.device.index):
        x, v = x.contiguous(), v.contiguous()
        spikes = torch.empty_like(x)
        voltages = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
        charged = torch.empty_like(x, dtype=torch.float32)
        # Torch owns all memory; launch on its current stream, including copies.
        stream = cupy.cuda.ExternalStream(
            torch.cuda.current_stream(x.device).cuda_stream
        )
        _kernel(_SOURCE, "lif_forward", x.dtype)(
            (min((v.numel() + 255) // 256, 65535),),
            (256,),
            (
                *(np.uint64(t.data_ptr()) for t in (x, v, spikes, voltages, charged)),
                np.int64(x.shape[0]),
                np.int64(v.numel()),
                np.float32(tau),
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
    "sj_lif::cupy_forward", mutates_args=(), device_types="cuda"
)(_forward_impl)


def _backward_impl(
    gs: torch.Tensor,
    gv: torch.Tensor,
    h: torch.Tensor,
    tau: float,
    decay_input: bool,
    threshold: float,
    reset: Optional[float],
    detach_reset: bool,
    alpha: float,
    store_v_seq: bool = True,
    surrogate_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_backward(gs, gv, h, tau, threshold, reset, alpha, store_v_seq, surrogate_id)
    with torch.cuda.device(h.device), cupy.cuda.Device(h.device.index):
        gs, gv, h = gs.contiguous(), gv.contiguous(), h.contiguous()
        gx = torch.empty_like(h, dtype=gs.dtype)
        gv_init = torch.empty_like(h[0])
        stream = cupy.cuda.ExternalStream(
            torch.cuda.current_stream(h.device).cuda_stream
        )
        elements = (
            2
            if gv_init.numel() >= 2097152 and (reset is not None or detach_reset)
            else 1
        )
        _kernel(_SOURCE, "lif_backward", gs.dtype, surrogate_id, elements)(
            (min((gv_init.numel() + 255) // 256, 65535),),
            (256 // elements,),
            (
                *(np.uint64(t.data_ptr()) for t in (gs, gv, h, gx, gv_init)),
                np.int64(h.shape[0]),
                np.int64(gv_init.numel()),
                np.float32(tau),
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
    return gx, gv_init


_backward = torch.library.custom_op(
    "sj_lif::cupy_backward", mutates_args=(), device_types="cuda"
)(_backward_impl)


_register_ops("sj_lif::cupy_forward", "sj_lif::cupy_backward")
