from pathlib import Path
from typing import Optional

import cupy
import numpy as np
import torch

from ..cupy_loader import _kernel
from .validation import _check, _forward_fake

_SOURCE = str(Path(__file__).with_name("kernels.cuh"))


def _forward_impl(
    x: torch.Tensor,
    v: torch.Tensor,
    threshold: torch.Tensor,
    offset: torch.Tensor,
    channels: int,
    inner: int,
    reset: Optional[float],
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check(x, v, threshold, offset, channels, inner, reset, store_v_seq)
    with torch.cuda.device(x.device), cupy.cuda.Device(x.device.index):
        x, v, threshold, offset = (
            x.contiguous(),
            v.contiguous(),
            threshold.contiguous(),
            offset.contiguous(),
        )
        out = torch.empty_like(x)
        vo = torch.empty_like(x if store_v_seq else v, dtype=torch.float32)
        _kernel(_SOURCE, "activation_aware_if_forward", x.dtype)(
            (min((v.numel() + 255) // 256, 65535),),
            (256,),
            (
                *(np.uint64(t.data_ptr()) for t in (x, v, threshold, offset, out, vo)),
                np.int64(x.shape[0]),
                np.int64(v.numel()),
                np.int64(channels),
                np.int64(inner),
                np.int32(threshold.numel() == 1),
                np.int32(offset.numel() == 1),
                np.float32(reset or 0),
                np.int32(reset is None),
                np.int32(store_v_seq),
            ),
            stream=cupy.cuda.ExternalStream(
                torch.cuda.current_stream(x.device).cuda_stream
            ),
        )
    return out, vo


_forward = torch.library.custom_op(
    "sj_activation_aware_if::cupy_forward", mutates_args=(), device_types="cuda"
)(_forward_impl)


torch.library.register_fake("sj_activation_aware_if::cupy_forward", _forward_fake)
