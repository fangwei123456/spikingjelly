from __future__ import annotations

from typing import Optional, Sequence

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

try:
    from torch.distributed._tensor import DTensor
except ImportError:
    DTensor = None


def prepare_snn_data_parallel(
    module: nn.Module,
    process_group=None,
    device_ids: Optional[Sequence[int]] = None,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
) -> DistributedDataParallel:
    return DistributedDataParallel(
        module,
        device_ids=list(device_ids) if device_ids is not None else None,
        process_group=process_group,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
    )


def unwrap_parallel_module(module: nn.Module) -> nn.Module:
    if isinstance(module, DistributedDataParallel):
        return module.module
    return module


def materialize_dtensor_output(output):
    if DTensor is not None and isinstance(output, DTensor):
        return output.full_tensor()
    full_tensor = getattr(output, "full_tensor", None)
    if callable(full_tensor):
        return full_tensor()
    if isinstance(output, tuple):
        if hasattr(output, "_fields"):
            return output.__class__(
                *(materialize_dtensor_output(item) for item in output)
            )
        return tuple(materialize_dtensor_output(item) for item in output)
    if isinstance(output, list):
        return [materialize_dtensor_output(item) for item in output]
    if isinstance(output, dict):
        return {key: materialize_dtensor_output(value) for key, value in output.items()}
    return output
