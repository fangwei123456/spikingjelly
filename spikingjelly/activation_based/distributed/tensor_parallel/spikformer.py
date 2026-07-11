from typing import Optional, Sequence

import torch.nn as nn

from spikingjelly.activation_based import base, layer
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
    _replace_child_module,
    _wrap_tensor_shard_memory_module,
)

Conv1dLike = (nn.Conv1d,)
Conv2dLike = (nn.Conv2d, layer.Conv2d)
BatchNorm1dLike = (nn.BatchNorm1d,)
BatchNorm2dLike = (nn.BatchNorm2d, layer.BatchNorm2d)


def _convert_leading_conv2d_bn(
    container: nn.Module, process_group, mode: str
) -> Optional[nn.Module]:
    if not isinstance(container, layer.SeqToANNContainer):
        return None
    modules = list(container.children())
    if len(modules) < 2:
        return None
    conv = modules[0]
    bn = modules[1]
    if not isinstance(conv, Conv2dLike) or not isinstance(bn, BatchNorm2dLike):
        return None
    converted = [ChannelShardConv2d(conv, process_group, mode=mode)]
    converted.append(
        ChannelShardBatchNorm2d(bn, process_group) if mode == "colwise" else bn
    )
    converted.extend(modules[2:])
    return layer.SeqToANNContainer(*converted)


def _convert_seq_to_ann_conv1d_bn(
    container: nn.Module, process_group, mode: str
) -> Optional[nn.Module]:
    if not isinstance(container, layer.SeqToANNContainer):
        return None
    modules = list(container.children())
    if len(modules) != 2:
        return None
    conv, bn = modules
    if not isinstance(conv, Conv1dLike) or not isinstance(bn, BatchNorm1dLike):
        return None
    if mode == "colwise":
        return layer.SeqToANNContainer(
            ChannelShardConv1d(conv, process_group, mode=mode),
            ChannelShardBatchNorm1d(bn, process_group),
        )
    return layer.SeqToANNContainer(
        ChannelShardConv1d(conv, process_group, mode=mode),
        bn,
    )


def _try_convert_spiking_self_attention(
    attn: nn.Module, process_group
) -> Optional[nn.Module]:
    if not hasattr(attn, "qkv_conv_bn"):
        return None

    converted = _convert_seq_to_ann_conv1d_bn(
        attn.qkv_conv_bn, process_group, mode="colwise"
    )
    if converted is not None:
        attn.qkv_conv_bn = converted
        if isinstance(attn.qkv_lif, base.MemoryModule):
            attn.qkv_lif = make_tensor_shard_memory_module(
                attn.qkv_lif,
                shard_dim=2,
                logical_dim_size=getattr(attn, "dim", None) * 3
                if getattr(attn, "dim", None) is not None
                else None,
                process_group=process_group,
            )
        if isinstance(attn.attn_lif, base.MemoryModule):
            attn.attn_lif = make_tensor_shard_memory_module(
                attn.attn_lif,
                shard_dim=2,
                logical_dim_size=None,
                process_group=process_group,
            )
    converted = _convert_seq_to_ann_conv1d_bn(
        attn.proj_conv_bn, process_group, mode="rowwise"
    )
    if converted is not None:
        attn.proj_conv_bn = converted
    return attn


def _try_convert_spikformer_mlp(mlp: nn.Module, process_group) -> Optional[nn.Module]:
    if not (hasattr(mlp, "fc1") or hasattr(mlp, "fc2")):
        return None
    if hasattr(mlp, "fc1"):
        converted = _convert_seq_to_ann_conv1d_bn(
            mlp.fc1, process_group, mode="colwise"
        )
        if converted is not None:
            mlp.fc1 = converted
            if isinstance(mlp.neuron1, base.MemoryModule):
                logical_dim = None
                conv = next(iter(mlp.fc1.children()))
                if hasattr(conv, "out_channels"):
                    logical_dim = conv.out_channels
                mlp.neuron1 = make_tensor_shard_memory_module(
                    mlp.neuron1,
                    shard_dim=2,
                    logical_dim_size=logical_dim,
                    process_group=process_group,
                )
    if hasattr(mlp, "fc2"):
        converted = _convert_seq_to_ann_conv1d_bn(
            mlp.fc2, process_group, mode="rowwise"
        )
        if converted is not None:
            mlp.fc2 = converted
    return mlp


def _convert_spiking_self_attention_tree(module: nn.Module, process_group) -> bool:
    converted = _try_convert_spiking_self_attention(module, process_group)
    if converted is not None:
        return True
    changed = False
    for child_name, child in list(module.named_children()):
        if _convert_spiking_self_attention_tree(child, process_group):
            changed = True
            _replace_child_module(module, child_name, child)
    return changed


def _convert_spikformer_mlp_tree(
    module: nn.Module, process_group, state: Optional[dict] = None
) -> bool:
    if state is None:
        state = {
            "fc1_converted": False,
            "neuron1_wrapped": False,
            "fc2_converted": False,
        }
    converted = _try_convert_spikformer_mlp(module, process_group)
    if converted is not None:
        state["fc1_converted"] = True
        state["neuron1_wrapped"] = True
        state["fc2_converted"] = True
        return True

    if not state["fc1_converted"]:
        converted_fc1 = _convert_seq_to_ann_conv1d_bn(
            module, process_group, mode="colwise"
        )
        if converted_fc1 is not None:
            if converted_fc1 is not module and isinstance(
                module, layer.SeqToANNContainer
            ):
                _overwrite_sequential_children(module, converted_fc1)
            state["fc1_converted"] = True
            return True

    if state["fc1_converted"] and not state["neuron1_wrapped"]:
        wrapped = _wrap_tensor_shard_memory_module(module, process_group, shard_dim=2)
        if wrapped is not None and wrapped is not module:
            state["neuron1_wrapped"] = True
            return True

    if state["neuron1_wrapped"] and not state["fc2_converted"]:
        converted_fc2 = _convert_seq_to_ann_conv1d_bn(
            module, process_group, mode="rowwise"
        )
        if converted_fc2 is not None:
            if converted_fc2 is not module and isinstance(
                module, layer.SeqToANNContainer
            ):
                _overwrite_sequential_children(module, converted_fc2)
            state["fc2_converted"] = True
            return True

    changed = False
    for child_name, child in list(module.named_children()):
        replacement = child
        if state["fc1_converted"] and not state["neuron1_wrapped"]:
            wrapped = _wrap_tensor_shard_memory_module(
                child, process_group, shard_dim=2
            )
            if wrapped is not None and wrapped is not child:
                replacement = wrapped
                state["neuron1_wrapped"] = True
                changed = True
        if _convert_spikformer_mlp_tree(replacement, process_group, state):
            changed = True
        if replacement is not child:
            _replace_child_module(module, child_name, replacement)
    return changed


def _try_convert_spikformer_block(
    block: nn.Module, process_group
) -> Optional[nn.Module]:
    if not (hasattr(block, "attn") and hasattr(block, "mlp")):
        return None

    _convert_spiking_self_attention_tree(block.attn, process_group=process_group)
    _convert_spikformer_mlp_tree(block.mlp, process_group=process_group)
    return block


def _try_convert_spikformer_stem_block(
    block: nn.Module, process_group, mode: str
) -> Optional[nn.Module]:
    if not (hasattr(block, "conv_bn") and hasattr(block, "neuron")):
        return None
    conv_bn = getattr(block, "conv_bn")
    if not hasattr(conv_bn, "block"):
        return None
    modules = list(conv_bn.block.children())
    if len(modules) < 2:
        return None
    conv = modules[0]
    bn = modules[1]
    if not isinstance(conv, Conv2dLike) or not isinstance(bn, BatchNorm2dLike):
        return None

    converted = [ChannelShardConv2d(conv, process_group, mode=mode)]
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
    converted.extend(modules[2:])
    conv_bn.block = layer.SeqToANNContainer(*converted)
    return block


def _convert_spikformer_stem_tree(
    module: nn.Module, process_group, mode: str, state: Optional[dict] = None
) -> bool:
    if state is None:
        state = {"projection_converted": False, "memory_wrapped": mode != "colwise"}

    converted = _try_convert_spikformer_stem_block(module, process_group, mode)
    if converted is not None:
        state["projection_converted"] = True
        state["memory_wrapped"] = True
        return True

    if not state["projection_converted"]:
        if hasattr(module, "block"):
            converted_container = _convert_leading_conv2d_bn(
                module.block, process_group, mode
            )
            if converted_container is not None:
                module.block = converted_container
                state["projection_converted"] = True
                return True
        converted_container = _convert_leading_conv2d_bn(module, process_group, mode)
        if converted_container is not None:
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
        if _convert_spikformer_stem_tree(replacement, process_group, mode, state):
            changed = True
        if replacement is not child:
            _replace_child_module(module, child_name, replacement)
    return changed


def parallelize_spikformer_blocks(
    module: nn.Module,
    device_mesh,
    roots: Sequence[str],
    tp_mesh_dim: int = 0,
) -> nn.Module:
    process_group = _resolve_mesh_dim_group(device_mesh, tp_mesh_dim)
    named_modules = dict(module.named_modules())

    for root in roots:
        if root not in named_modules:
            raise KeyError(f"Unknown Spikformer tensor parallel root '{root}'.")
        root_module = named_modules[root]
        if not isinstance(root_module, (nn.Sequential, nn.ModuleList)):
            raise TypeError(
                f"Spikformer tensor parallel root '{root}' must be an nn.Sequential or nn.ModuleList, "
                f"but got {type(root_module)}."
            )

        for child_name, child in list(root_module.named_children()):
            replacement = child
            changed = False
            changed = (
                _convert_spiking_self_attention_tree(
                    replacement, process_group=process_group
                )
                or changed
            )
            changed = (
                _convert_spikformer_mlp_tree(replacement, process_group=process_group)
                or changed
            )
            if changed and replacement is not child:
                root_module[int(child_name)] = replacement

    return module


def parallelize_spikformer_patch_stem(
    module: nn.Module,
    device_mesh,
    roots: Sequence[str],
    tp_mesh_dim: int = 0,
) -> nn.Module:
    process_group = _resolve_mesh_dim_group(device_mesh, tp_mesh_dim)
    named_modules = dict(module.named_modules())

    for root in roots:
        if root not in named_modules:
            raise KeyError(f"Unknown Spikformer patch stem root '{root}'.")
        root_module = named_modules[root]
        if hasattr(root_module, "stages") and hasattr(
            root_module, "positional_encoding"
        ):
            block_index = 0
            stage_sequence = getattr(root_module, "stages")
            for child_name, child in list(stage_sequence.named_children()):
                mode = "colwise" if block_index % 2 == 0 else "rowwise"
                replacement = child
                changed = _convert_spikformer_stem_tree(
                    replacement,
                    process_group=process_group,
                    mode=mode,
                )
                if changed:
                    if replacement is not child:
                        stage_sequence[int(child_name)] = replacement
                    block_index += 1
            continue
        if isinstance(root_module, (nn.Sequential, nn.ModuleList)) and not isinstance(
            root_module, layer.SeqToANNContainer
        ):
            block_index = 0
            for child_name, child in list(root_module.named_children()):
                mode = "colwise" if block_index % 2 == 0 else "rowwise"
                replacement = child
                changed = _convert_spikformer_stem_tree(
                    replacement, process_group=process_group, mode=mode
                )
                if changed:
                    if replacement is not child:
                        root_module[int(child_name)] = replacement
                    block_index += 1
            continue

        parent_name, _, child_name = root.rpartition(".")
        parent_module = named_modules.get(parent_name) if parent_name else module
        if (
            isinstance(parent_module, (nn.Sequential, nn.ModuleList))
            and child_name.isdigit()
        ):
            child_index = int(child_name)
            child_items = list(parent_module.named_children())
            remaining = len(child_items) - child_index
            convertible_count = remaining if remaining % 2 == 0 else remaining - 1
            if convertible_count < 2:
                raise ValueError(
                    "An isolated Spikformer patch-stem root must include at least two consecutive stem "
                    "blocks so TP can restore full channels before rejoining unsharded modules."
                )
            block_index = 0
            for local_offset in range(convertible_count):
                current_name, child = child_items[child_index + local_offset]
                mode = "colwise" if block_index % 2 == 0 else "rowwise"
                replacement = child
                changed = _convert_spikformer_stem_tree(
                    replacement,
                    process_group=process_group,
                    mode=mode,
                )
                if changed:
                    if replacement is not child:
                        parent_module[int(current_name)] = replacement
                    block_index += 1
            continue

        raise ValueError(
            f"Unsupported isolated Spikformer patch stem root '{root}'. Use 'patch_embed' or a root that "
            "belongs to a sequential stem with at least two consecutive blocks."
        )

    return module
