from typing import Sequence

import torch.nn as nn

from spikingjelly.activation_based import layer
from spikingjelly.activation_based.distributed.mesh import _resolve_mesh_dim_group
from spikingjelly.activation_based.distributed.tensor_parallel.channel import (
    ChannelShardBatchNorm1d,
    ChannelShardBatchNorm2d,
    ChannelShardConv1d,
    ChannelShardConv2d,
)
from spikingjelly.activation_based.distributed.tensor_parallel.state import (
    make_tensor_shard_memory_module,
)
from spikingjelly.activation_based.distributed.tensor_parallel.utils import (
    _overwrite_sequential_children,
)
from spikingjelly.activation_based.layer.attention import SpikingSelfAttention
from spikingjelly.activation_based.model.spikformer import (
    SpikformerBlock,
    SpikformerConv2dBNLIF,
    SpikformerPatchStem,
)


def _find_descendant(module: nn.Module, module_type):
    return next(
        (child for child in module.modules() if isinstance(child, module_type)),
        None,
    )


def _convert_conv1d_bn(container, process_group, mode):
    if not isinstance(container, layer.SeqToANNContainer):
        return None
    modules = list(container.children())
    if (
        len(modules) != 2
        or not isinstance(modules[0], nn.Conv1d)
        or not isinstance(modules[1], nn.BatchNorm1d)
    ):
        return None

    conv, bn = modules
    if mode == "colwise":
        bn = ChannelShardBatchNorm1d(bn, process_group)
    return layer.SeqToANNContainer(
        ChannelShardConv1d(conv, process_group, mode=mode),
        bn,
    )


def _convert_attention(attn: SpikingSelfAttention, process_group):
    qkv = _convert_conv1d_bn(attn.qkv_conv_bn, process_group, "colwise")
    if qkv is not None:
        attn.qkv_conv_bn = qkv
        attn.qkv_lif = make_tensor_shard_memory_module(
            attn.qkv_lif,
            shard_dim=2,
            logical_dim_size=attn.dim * 3,
            process_group=process_group,
        )
        attn.attn_lif = make_tensor_shard_memory_module(
            attn.attn_lif,
            shard_dim=2,
            process_group=process_group,
        )

    projection = _convert_conv1d_bn(attn.proj_conv_bn, process_group, "rowwise")
    if projection is not None:
        attn.proj_conv_bn = projection


def _convert_block(block: SpikformerBlock, process_group):
    _convert_attention(block.attn, process_group)

    mlp = block.mlp
    fc1 = _convert_conv1d_bn(mlp.fc1, process_group, "colwise")
    if fc1 is not None:
        mlp.fc1 = fc1
        conv = next(iter(fc1.children()))
        mlp.neuron1 = make_tensor_shard_memory_module(
            mlp.neuron1,
            shard_dim=2,
            logical_dim_size=conv.out_channels,
            process_group=process_group,
        )

    fc2 = _convert_conv1d_bn(mlp.fc2, process_group, "rowwise")
    if fc2 is not None:
        mlp.fc2 = fc2


def _convert_leading_conv2d_bn(container, process_group, mode):
    if not isinstance(container, layer.SeqToANNContainer):
        return None
    modules = list(container.children())
    if (
        len(modules) < 2
        or not isinstance(modules[0], (nn.Conv2d, layer.Conv2d))
        or not isinstance(modules[1], (nn.BatchNorm2d, layer.BatchNorm2d))
    ):
        return None

    conv, bn = modules[:2]
    converted = [ChannelShardConv2d(conv, process_group, mode=mode)]
    converted.append(
        ChannelShardBatchNorm2d(bn, process_group) if mode == "colwise" else bn
    )
    return layer.SeqToANNContainer(*converted, *modules[2:])


def _convert_stem(module: nn.Module, process_group, mode: str):
    stage = _find_descendant(module, SpikformerConv2dBNLIF)
    if stage is not None:
        conv = next(iter(stage.conv_bn.block.children()))
        converted = _convert_leading_conv2d_bn(stage.conv_bn.block, process_group, mode)
        if converted is None:
            return False
        stage.conv_bn.block = converted
        if mode == "colwise":
            stage.neuron = make_tensor_shard_memory_module(
                stage.neuron,
                shard_dim=2,
                logical_dim_size=conv.out_channels,
                process_group=process_group,
            )
        return True

    container = _find_descendant(module, layer.SeqToANNContainer)
    converted = _convert_leading_conv2d_bn(container, process_group, mode)
    if converted is None:
        return False
    _overwrite_sequential_children(container, converted)
    return True


def parallelize_spikformer_blocks(
    module: nn.Module,
    device_mesh,
    roots: Sequence[str],
    tp_mesh_dim: int = 0,
) -> nn.Module:
    process_group = _resolve_mesh_dim_group(device_mesh, tp_mesh_dim)
    named_modules = dict(module.named_modules())

    for root in roots:
        root_module = named_modules.get(root)
        if root_module is None:
            raise KeyError(f"Unknown Spikformer tensor parallel root '{root}'.")
        if not isinstance(root_module, (nn.Sequential, nn.ModuleList)):
            raise TypeError(
                f"Spikformer tensor parallel root '{root}' must be an nn.Sequential or nn.ModuleList, "
                f"but got {type(root_module)}."
            )

        for child in root_module.children():
            block = _find_descendant(child, SpikformerBlock)
            if block is not None:
                _convert_block(block, process_group)

    return module


def _sequential_root(module, root, named_modules):
    root_module = named_modules.get(root)
    if root_module is None:
        raise KeyError(f"Unknown Spikformer patch stem root '{root}'.")
    if isinstance(root_module, SpikformerPatchStem):
        return list(root_module.stages.children())
    if isinstance(root_module, (nn.Sequential, nn.ModuleList)) and not isinstance(
        root_module, layer.SeqToANNContainer
    ):
        return list(root_module.children())

    parent_name, _, child_name = root.rpartition(".")
    parent = named_modules.get(parent_name) if parent_name else module
    if (
        not isinstance(parent, (nn.Sequential, nn.ModuleList))
        or not child_name.isdigit()
    ):
        return None

    children = list(parent.children())
    start = int(child_name)
    remaining = len(children) - start
    count = remaining - remaining % 2
    if count < 2:
        raise ValueError(
            "An isolated Spikformer patch-stem root must include at least two consecutive stem "
            "blocks so TP can restore full channels before rejoining unsharded modules."
        )
    return children[start : start + count]


def parallelize_spikformer_patch_stem(
    module: nn.Module,
    device_mesh,
    roots: Sequence[str],
    tp_mesh_dim: int = 0,
) -> nn.Module:
    process_group = _resolve_mesh_dim_group(device_mesh, tp_mesh_dim)
    named_modules = dict(module.named_modules())

    for root in roots:
        stages = _sequential_root(module, root, named_modules)
        if stages is None:
            raise ValueError(
                f"Unsupported isolated Spikformer patch stem root '{root}'. Use 'patch_embed' or a root that "
                "belongs to a sequential stem with at least two consecutive blocks."
            )
        for index, stage in enumerate(stages):
            _convert_stem(
                stage,
                process_group,
                mode="colwise" if index % 2 == 0 else "rowwise",
            )

    return module
