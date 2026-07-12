from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch
import torch.nn as nn

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


def _build_fsdp_mp_policy(
    param_dtype: Optional[torch.dtype],
    reduce_dtype: Optional[torch.dtype],
    output_dtype: Optional[torch.dtype],
):
    if MixedPrecisionPolicy is None:
        return None
    if param_dtype is None and reduce_dtype is None and output_dtype is None:
        return None
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
    )


def fully_shard_snn_module(
    module: nn.Module,
    device_mesh: "DeviceMesh",
    shard_roots: Optional[Sequence[str]] = None,
    shard_module_root: bool = True,
    root_reshard_after_forward: Optional[bool] = False,
    mp_policy: Optional["MixedPrecisionPolicy"] = None,
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
    fsdp_mesh_dim = dp_mesh_dim if dp_mesh_dim is not None else 0
    fsdp_mesh = _resolve_mesh_submesh(device_mesh, fsdp_mesh_dim)
    mp_policy = _build_fsdp_mp_policy(param_dtype, reduce_dtype, output_dtype)
    return fully_shard_snn_module(
        module=module,
        device_mesh=fsdp_mesh,
        shard_roots=shard_roots,
        shard_module_root=shard_module_root,
        root_reshard_after_forward=root_reshard_after_forward,
        mp_policy=mp_policy,
    )
