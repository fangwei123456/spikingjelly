from typing import Optional, Sequence

import torch.nn as nn

from spikingjelly.activation_based import base, layer
from spikingjelly.activation_based.distributed.mesh import _resolve_mesh_dim_group
from spikingjelly.activation_based.distributed.tensor_parallel.channel import (
    ChannelShardBatchNorm2d,
    ChannelShardConv2d,
)
from spikingjelly.activation_based.distributed.tensor_parallel.state import (
    make_tensor_shard_memory_module,
)
from spikingjelly.activation_based.distributed.tensor_parallel.utils import (
    _overwrite_sequential_children,
    _replace_child_module,
    _wrap_tensor_shard_memory_module,
)

Conv2dLike = (nn.Conv2d, layer.Conv2d)
BatchNorm2dLike = (nn.BatchNorm2d, layer.BatchNorm2d)


def _try_convert_vgg_like_block(
    block: nn.Module, process_group, mode: str
) -> Optional[nn.Module]:
    if not (hasattr(block, "proj_bn") and hasattr(block, "neuron")):
        return None
    if not isinstance(block.proj_bn, nn.Sequential):
        return None

    modules = list(block.proj_bn.children())
    if len(modules) < 2:
        return None
    conv = modules[-2]
    bn = modules[-1]
    if not isinstance(conv, Conv2dLike) or not isinstance(bn, BatchNorm2dLike):
        return None

    converted = []
    converted.extend(modules[:-2])
    converted.append(ChannelShardConv2d(conv, process_group, mode=mode))
    if mode == "colwise":
        converted.append(ChannelShardBatchNorm2d(bn, process_group))
        if isinstance(block.neuron, base.MemoryModule):
            block.neuron = make_tensor_shard_memory_module(
                block.neuron,
                shard_dim=2,
                logical_dim_size=conv.out_channels,
                process_group=process_group,
            )
    else:
        converted.append(bn)
    block.proj_bn = nn.Sequential(*converted)
    return block


def _convert_trailing_conv2d_bn(
    container: nn.Module, process_group, mode: str
) -> Optional[nn.Module]:
    if not isinstance(container, nn.Sequential):
        return None
    modules = list(container.children())
    if len(modules) < 2:
        return None
    conv = modules[-2]
    bn = modules[-1]
    if not isinstance(conv, Conv2dLike) or not isinstance(bn, BatchNorm2dLike):
        return None
    converted = list(modules[:-2])
    converted.append(ChannelShardConv2d(conv, process_group, mode=mode))
    converted.append(
        ChannelShardBatchNorm2d(bn, process_group) if mode == "colwise" else bn
    )
    return nn.Sequential(*converted)


def _convert_vgg_like_tree(
    module: nn.Module, process_group, mode: str, state: Optional[dict] = None
) -> bool:
    if state is None:
        state = {"projection_converted": False, "memory_wrapped": mode != "colwise"}

    converted = _try_convert_vgg_like_block(module, process_group, mode)
    if converted is not None:
        state["projection_converted"] = True
        state["memory_wrapped"] = True
        return True

    if not state["projection_converted"]:
        converted_container = _convert_trailing_conv2d_bn(module, process_group, mode)
        if converted_container is not None:
            if converted_container is not module:
                _overwrite_sequential_children(module, converted_container)
            state["projection_converted"] = True
            return True

    if state["projection_converted"] and not state["memory_wrapped"]:
        wrapped = _wrap_tensor_shard_memory_module(module, process_group, shard_dim=2)
        if wrapped is not None and wrapped is not module:
            state["memory_wrapped"] = True
            return True

    changed = False
    for child_name, child in list(module.named_children()):
        replacement = child
        if state["projection_converted"] and not state["memory_wrapped"]:
            wrapped = _wrap_tensor_shard_memory_module(
                child, process_group, shard_dim=2
            )
            if wrapped is not None and wrapped is not child:
                replacement = wrapped
                state["memory_wrapped"] = True
                changed = True
        if _convert_vgg_like_tree(replacement, process_group, mode, state):
            changed = True
        if replacement is not child:
            _replace_child_module(module, child_name, replacement)
    return changed


def parallelize_snn_conv_blocks(
    module: nn.Module,
    device_mesh,
    roots: Sequence[str],
    tp_mesh_dim: int = 0,
) -> nn.Module:
    process_group = _resolve_mesh_dim_group(device_mesh, tp_mesh_dim)
    named_modules = dict(module.named_modules())

    for root in roots:
        if root not in named_modules:
            raise KeyError(f"Unknown conv tensor parallel root '{root}'.")
        root_module = named_modules[root]
        if not isinstance(root_module, nn.Sequential):
            raise TypeError(
                f"Conv tensor parallel root '{root}' must be an nn.Sequential, but got {type(root_module)}."
            )

        block_index = 0
        for child_name, child in list(root_module.named_children()):
            mode = "colwise" if block_index % 2 == 0 else "rowwise"
            replacement = child
            changed = _convert_vgg_like_tree(
                replacement, process_group=process_group, mode=mode
            )
            if changed:
                if replacement is not child:
                    root_module[int(child_name)] = replacement
                block_index += 1

    return module
