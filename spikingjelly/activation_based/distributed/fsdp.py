from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch
import torch.nn as nn

from .config import SNNDistributedConfig
from .mesh import _resolve_mesh_submesh

if TYPE_CHECKING:
    from torch.distributed._tensor import DeviceMesh

try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    FSDP2_AVAILABLE = True
except ImportError:
    MixedPrecisionPolicy = None
    fully_shard = None
    FSDP2_AVAILABLE = False


def _build_fsdp_mp_policy(config: SNNDistributedConfig):
    if MixedPrecisionPolicy is None:
        return None
    if (
        config.fsdp_param_dtype is None
        and config.fsdp_reduce_dtype is None
        and config.fsdp_output_dtype is None
    ):
        return None
    return MixedPrecisionPolicy(
        param_dtype=config.fsdp_param_dtype,
        reduce_dtype=config.fsdp_reduce_dtype,
        output_dtype=config.fsdp_output_dtype,
    )


def fully_shard_snn_module(
    module: nn.Module,
    device_mesh: "DeviceMesh",
    shard_roots: Optional[Sequence[str]] = None,
    shard_module_root: bool = True,
    root_reshard_after_forward: Optional[bool] = False,
    mp_policy=None,
) -> nn.Module:
    if not FSDP2_AVAILABLE:
        raise RuntimeError(
            "FSDP2 fully_shard is unavailable in the current PyTorch build."
        )

    named_modules = dict(module.named_modules())
    shard_roots = list(shard_roots or [])
    for name in shard_roots:
        if name not in named_modules:
            raise KeyError(f"Unknown FSDP shard root '{name}'.")
        submodule = named_modules[name]
        if mp_policy is None:
            fully_shard(submodule, mesh=device_mesh)
        else:
            fully_shard(submodule, mesh=device_mesh, mp_policy=mp_policy)

    if shard_module_root:
        if mp_policy is None:
            fully_shard(
                module,
                mesh=device_mesh,
                reshard_after_forward=root_reshard_after_forward,
            )
        else:
            fully_shard(
                module,
                mesh=device_mesh,
                reshard_after_forward=root_reshard_after_forward,
                mp_policy=mp_policy,
            )
    return module


def apply_snn_fsdp2(
    module: nn.Module,
    device_mesh: "DeviceMesh",
    dp_mesh_dim: Optional[int] = None,
    shard_roots: Optional[Sequence[str]] = None,
    shard_module_root: bool = True,
    root_reshard_after_forward: Optional[bool] = False,
    param_dtype: Optional[torch.dtype] = None,
    reduce_dtype: Optional[torch.dtype] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    config = SNNDistributedConfig(
        enable_fsdp2=True,
        dp_mesh_dim=dp_mesh_dim,
        fsdp_shard_roots=shard_roots,
        fsdp_shard_module_root=shard_module_root,
        fsdp_root_reshard_after_forward=root_reshard_after_forward,
        fsdp_param_dtype=param_dtype,
        fsdp_reduce_dtype=reduce_dtype,
        fsdp_output_dtype=output_dtype,
    )
    fsdp_mesh_dim = dp_mesh_dim if dp_mesh_dim is not None else 0
    fsdp_mesh = _resolve_mesh_submesh(device_mesh, fsdp_mesh_dim)
    mp_policy = _build_fsdp_mp_policy(config)
    return fully_shard_snn_module(
        module=module,
        device_mesh=fsdp_mesh,
        shard_roots=shard_roots,
        shard_module_root=shard_module_root,
        root_reshard_after_forward=root_reshard_after_forward,
        mp_policy=mp_policy,
    )
