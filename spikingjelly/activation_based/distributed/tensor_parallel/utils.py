from typing import Optional

import torch.nn as nn

from spikingjelly.activation_based import base
from spikingjelly.activation_based.distributed.tensor_parallel.state import (
    TensorShardMemoryModule,
)


def _replace_child_module(parent: nn.Module, child_name: str, new_child: nn.Module):
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child_name.isdigit():
        parent[int(child_name)] = new_child
    else:
        setattr(parent, child_name, new_child)


def _overwrite_sequential_children(target: nn.Module, source: nn.Module):
    target_children = list(target.named_children())
    source_children = list(source.named_children())
    if len(target_children) != len(source_children):
        raise ValueError(
            f"Cannot overwrite {type(target)} with {type(source)} because child counts differ: "
            f"{len(target_children)} vs {len(source_children)}."
        )
    for (target_name, _), (_, source_child) in zip(
        target_children, source_children, strict=True
    ):
        _replace_child_module(target, target_name, source_child)


def _wrap_tensor_shard_memory_module(
    module: nn.Module,
    process_group,
    shard_dim: int,
    logical_dim_size: Optional[int] = None,
) -> Optional[nn.Module]:
    if isinstance(module, TensorShardMemoryModule):
        return module
    if isinstance(module, base.MemoryModule):
        return TensorShardMemoryModule(
            module,
            shard_dim=shard_dim,
            logical_dim_size=logical_dim_size,
            process_group=process_group,
        )
    return None
