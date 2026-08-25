from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import torch.nn as nn

from spikingjelly.activation_based import layer, memopt
from spikingjelly.activation_based.distributed.tensor_parallel import (
    ChannelShardBatchNorm1d,
    ChannelShardBatchNorm2d,
    ChannelShardConv1d,
    ChannelShardConv2d,
)
from spikingjelly.activation_based.distributed.tensor_parallel.channel import (
    _ColwiseBackwardAllReduce,
)
from spikingjelly.activation_based.model.spikformer import (
    SpikformerBlock,
    SpikformerPatchStem,
    spikformer_cifar10,
    spikformer_s,
)

from .config import ModelBuilder, ModelConfig


class _SpikformerHead(nn.Module):
    def __init__(self, block: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.block = block
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(x).flatten(3).mean(dim=-1))


def _pipeline_stage(model: nn.Module, rank: int, size: int) -> nn.Module:
    if size == 1:
        return model
    if size not in {2, 4}:
        raise ValueError("Spikformer PP size must be 1, 2, or 4.")
    if len(model.blocks) == 6:
        chunks = (
            model.patch_embed,
            nn.Sequential(model.blocks[0], model.blocks[1]),
            nn.Sequential(model.blocks[2], model.blocks[3]),
            _SpikformerHead(
                nn.Sequential(model.blocks[4], model.blocks[5]), model.head
            ),
        )
    elif len(model.blocks) == 4:
        if size == 4:
            raise ValueError("The 4-block Spikformer supports PP size 1 or 2.")
        chunks = (
            model.patch_embed,
            nn.Sequential(model.blocks[0], model.blocks[1]),
            nn.Sequential(model.blocks[2], model.blocks[3]),
            _SpikformerHead(nn.Identity(), model.head),
        )
    else:
        raise ValueError("Spikformer PP requires 4 or 6 transformer blocks.")
    chunks_per_stage = len(chunks) // size
    start = rank * chunks_per_stage
    return nn.Sequential(*chunks[start : start + chunks_per_stage])


def _copy_batch_norm(
    source: nn.modules.batchnorm._BatchNorm,
    indices: Optional[torch.Tensor] = None,
) -> nn.Module:
    if not isinstance(source, (nn.BatchNorm1d, nn.BatchNorm2d)):
        raise TypeError(f"Unsupported batch norm type {type(source).__name__}.")
    size = source.num_features if indices is None else indices.numel()
    batch_norm = (
        layer.BatchNorm1d if isinstance(source, nn.BatchNorm1d) else layer.BatchNorm2d
    )
    target = batch_norm(
        size,
        eps=source.eps,
        momentum=source.momentum,
        affine=source.affine,
        track_running_stats=source.track_running_stats,
    )

    def select(value: torch.Tensor) -> torch.Tensor:
        return value if indices is None else value[indices]

    with torch.no_grad():
        if source.affine:
            target.weight.copy_(select(source.weight))
            target.bias.copy_(select(source.bias))
            target.weight.requires_grad_(source.weight.requires_grad)
            target.bias.requires_grad_(source.bias.requires_grad)
        if source.track_running_stats:
            target.running_mean.copy_(select(source.running_mean))
            target.running_var.copy_(select(source.running_var))
            target.num_batches_tracked.copy_(source.num_batches_tracked)
    target.train(source.training)
    return target


def _convert_batch_norms(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)) and not isinstance(
            child, (layer.BatchNorm1d, layer.BatchNorm2d)
        ):
            setattr(module, name, _copy_batch_norm(child))
        else:
            _convert_batch_norms(child)


class _HeadShardQKVConv1d(nn.Conv1d):
    def __init__(
        self,
        source: nn.Conv1d,
        process_group: ProcessGroup,
        num_heads: int,
    ) -> None:
        if source.kernel_size != (1,) or source.groups != 1:
            raise ValueError("Spikformer QKV TP requires an ungrouped 1x1 Conv1d.")
        if source.out_channels != 3 * source.in_channels:
            raise ValueError("Spikformer QKV projection must have 3 * dim outputs.")

        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)
        if num_heads % world_size:
            raise ValueError("Spikformer num_heads must be divisible by TP size.")
        dim = source.in_channels
        if dim % num_heads:
            raise ValueError("Spikformer channels must be divisible by num_heads.")
        heads_per_rank = num_heads // world_size
        head_dim = dim // num_heads
        start = rank * heads_per_rank * head_dim
        end = start + heads_per_rank * head_dim
        self.output_indices = torch.cat(
            [torch.arange(offset + start, offset + end) for offset in (0, dim, 2 * dim)]
        )
        local_dim = end - start
        super().__init__(
            dim,
            3 * local_dim,
            source.kernel_size,
            source.stride,
            source.padding,
            source.dilation,
            source.groups,
            source.bias is not None,
            source.padding_mode,
        )
        with torch.no_grad():
            self.weight.copy_(source.weight[self.output_indices])
            if source.bias is not None:
                self.bias.copy_(source.bias[self.output_indices])
        self.weight.requires_grad_(source.weight.requires_grad)
        if source.bias is not None:
            self.bias.requires_grad_(source.bias.requires_grad)
        self.process_group = process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ColwiseBackwardAllReduce.apply(x, self.process_group)
        return super().forward(x)


@dataclass(frozen=True)
class SpikformerConfig(ModelConfig):
    builder: ClassVar[str] = (
        "spikingjelly.activation_based.distributed.vision.spikformer.SpikformerBuilder"
    )
    image_height: int = 224
    image_width: int = 224
    in_channels: int = 3
    neuron_backend: str = "torch"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.image_height <= 0 or self.image_width <= 0 or self.in_channels <= 0:
            raise ValueError("image dimensions and in_channels must be positive.")
        if self.step_mode != "m":
            raise ValueError("The built-in Spikformer requires step_mode='m'.")


SpikformerConfig.__init__.__doc__ = r"""Configure the built-in Spikformer-S.

**API Language** - 中文 | English

**中文：** 声明 Spikformer-S 的时间步、图像尺寸、输入通道、类别数和神经元后端。
TP 按 attention head 分片，并对 Q、K、V 分别取本地 head 后重新拼接。

**English:** Declare time steps, image dimensions, input channels, classes, and
neuron backend for Spikformer-S. TP shards attention heads and reconstructs local
QKV from separately selected Q, K, and V heads.

:param time_steps: SNN 时间步。 / SNN time steps.
:type time_steps: int
:param num_classes: 分类类别数。 / Number of classes.
:type num_classes: int
:param step_mode: 固定为 ``"m"``。 / Must be ``"m"``.
:type step_mode: str
:param image_height: 输入图像高度。 / Input image height.
:type image_height: int
:param image_width: 输入图像宽度。 / Input image width.
:type image_width: int
:param in_channels: 输入通道数。 / Input channels.
:type in_channels: int
:param neuron_backend: 神经元 backend。 / Neuron backend.
:type neuron_backend: str
:raises ValueError: 图像尺寸、通道或 step mode 无效。 / If image dimensions,
    channels, or the step mode are invalid.
"""


@dataclass(frozen=True)
class SpikformerCIFAR10Config(ModelConfig):
    builder: ClassVar[str] = (
        "spikingjelly.activation_based.distributed.vision.spikformer.SpikformerBuilder"
    )
    num_classes: int = 10
    neuron_backend: str = "torch"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.step_mode != "m":
            raise ValueError("The built-in Spikformer requires step_mode='m'.")


SpikformerCIFAR10Config.__init__.__doc__ = r"""Configure CIFAR-10 Spikformer.

**API Language** - 中文 | English

**中文：** 声明固定 32×32 输入、4×4 patch、384 维、12 heads、4 blocks 的
CIFAR-10 Spikformer。TP 按 attention head 分片；PP 支持 1 或 2 stages。

**English:** Declare the CIFAR-10 Spikformer with fixed 32×32 input, 4×4 patches,
384 channels, 12 heads, and 4 blocks. TP shards attention heads; PP supports one
or two stages.

:param time_steps: SNN 时间步。 / SNN time steps.
:type time_steps: int
:param num_classes: 分类类别数。 / Number of classes.
:type num_classes: int
:param step_mode: 固定为 ``"m"``。 / Must be ``"m"``.
:type step_mode: str
:param neuron_backend: 神经元 backend。 / Neuron backend.
:type neuron_backend: str
:raises ValueError: 时间步、类别数或 step mode 无效。 / If time steps, class count,
    or the step mode are invalid.
"""


class SpikformerBuilder(ModelBuilder):
    def _build_canonical_model(self) -> nn.Module:
        config = self.config
        if isinstance(config, SpikformerCIFAR10Config):
            model = spikformer_cifar10(
                T=config.time_steps,
                num_classes=config.num_classes,
                backend=config.neuron_backend,
            )
        elif isinstance(config, SpikformerConfig):
            model = spikformer_s(
                T=config.time_steps,
                in_channels=config.in_channels,
                img_size_h=config.image_height,
                img_size_w=config.image_width,
                num_classes=config.num_classes,
                backend=config.neuron_backend,
            )
        else:
            raise TypeError("SpikformerBuilder requires a Spikformer config.")
        _convert_batch_norms(model)
        return model

    def _pipeline_stage(self, model: nn.Module, rank: int, size: int) -> nn.Module:
        return _pipeline_stage(model, rank, size)

    @staticmethod
    def _qkv_indices(total: int, rank: int, size: int) -> torch.Tensor:
        dim = total // 3
        local_dim = dim // size
        start = rank * local_dim
        end = start + local_dim
        return torch.cat(
            [torch.arange(offset + start, offset + end) for offset in (0, dim, 2 * dim)]
        )

    def _merge_tensor_parallel_shards(
        self,
        name: str,
        shards: Sequence[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if (
            "attn.qkv_conv_bn." not in name
            or reference.ndim == 0
            or shards[0].shape == reference.shape
        ):
            return super()._merge_tensor_parallel_shards(name, shards, reference)
        result = torch.empty_like(reference)
        for rank, shard in enumerate(shards):
            result[self._qkv_indices(reference.shape[0], rank, len(shards))] = shard
        return result

    def _shard_tensor_parallel_tensor(
        self,
        name: str,
        value: torch.Tensor,
        target: torch.Tensor,
        tensor_rank: int,
        tensor_size: int,
    ) -> torch.Tensor:
        if (
            "attn.qkv_conv_bn." in name
            and value.ndim > 0
            and value.shape != target.shape
        ):
            return value[
                self._qkv_indices(value.shape[0], tensor_rank, tensor_size)
            ].contiguous()
        return super()._shard_tensor_parallel_tensor(
            name, value, target, tensor_rank, tensor_size
        )

    @staticmethod
    def _parallelize_stem(model: nn.Module, process_group: ProcessGroup) -> None:
        for index, stage in enumerate(model.patch_embed.stages):
            container = stage.conv_bn.block
            conv = container[0]
            batch_norm = container[1]
            if index % 2 == 0:
                container[0] = ChannelShardConv2d(conv, process_group, "colwise")
                container[1] = ChannelShardBatchNorm2d(batch_norm, process_group)
            else:
                container[0] = ChannelShardConv2d(conv, process_group, "rowwise")
                container[1] = _copy_batch_norm(batch_norm)
        positional = model.patch_embed.positional_encoding.conv_bn.block
        positional[1] = _copy_batch_norm(positional[1])

    @staticmethod
    def _parallelize_block(block: SpikformerBlock, process_group: ProcessGroup) -> None:
        attention = block.attn
        qkv_source = attention.qkv_conv_bn[0]
        qkv = _HeadShardQKVConv1d(
            qkv_source,
            process_group,
            attention.num_heads,
        )
        attention.qkv_conv_bn[0] = qkv
        attention.qkv_conv_bn[1] = _copy_batch_norm(
            attention.qkv_conv_bn[1], qkv.output_indices
        )
        attention.num_heads //= dist.get_world_size(process_group)
        attention.proj_conv_bn[0] = ChannelShardConv1d(
            attention.proj_conv_bn[0], process_group, "rowwise"
        )
        attention.proj_conv_bn[1] = _copy_batch_norm(attention.proj_conv_bn[1])

        mlp = block.mlp
        fc1 = mlp.fc1[0]
        mlp.fc1[0] = ChannelShardConv1d(fc1, process_group, "colwise")
        mlp.fc1[1] = ChannelShardBatchNorm1d(mlp.fc1[1], process_group)
        mlp.fc2[0] = ChannelShardConv1d(mlp.fc2[0], process_group, "rowwise")
        mlp.fc2[1] = _copy_batch_norm(mlp.fc2[1])

    def build(
        self,
        *,
        process_group: Optional[ProcessGroup],
        pipeline_rank: int,
        pipeline_size: int,
        pipeline_microbatches: int,
        device: torch.device,
        micro_batch_size: int,
        memopt_level: int,
        memopt_compress_inputs: bool,
    ) -> tuple[
        nn.Module,
        tuple[str, ...],
        Optional[tuple[int, ...]],
        Optional[tuple[int, ...]],
    ]:
        config = self.config
        if isinstance(config, SpikformerCIFAR10Config):
            image_height = image_width = 32
            in_channels = 3
            patch_size = 4
        elif isinstance(config, SpikformerConfig):
            image_height = config.image_height
            image_width = config.image_width
            in_channels = config.in_channels
            patch_size = 16
        else:
            raise TypeError("SpikformerBuilder requires a Spikformer config.")
        model = self._build_canonical_model()
        if pipeline_size > 1 and (
            image_height % patch_size or image_width % patch_size
        ):
            raise ValueError(
                f"Spikformer image dimensions must be divisible by {patch_size} with PP."
            )
        world_size = dist.get_world_size(process_group) if process_group else 1
        if world_size > 1:
            self._parallelize_stem(model, process_group)
            for block in model.blocks:
                self._parallelize_block(block, process_group)

        if pipeline_size > 1 and memopt_level:
            raise ValueError("Spikformer-S memopt is not supported together with PP.")
        model = self._pipeline_stage(model, pipeline_rank, pipeline_size)
        fsdp_roots = tuple(
            name
            for name, module in model.named_modules()
            if isinstance(module, (SpikformerPatchStem, SpikformerBlock))
        )
        if memopt_level:
            dummy = torch.zeros(
                micro_batch_size,
                in_channels,
                image_height,
                image_width,
                device=device,
            )
            model = memopt.memory_optimization(
                model,
                SpikformerBlock,
                dummy_input=(dummy,),
                compress_x=memopt_compress_inputs,
                level=memopt_level,
            )
        model.to(device)
        if pipeline_size == 1:
            return model, fsdp_roots, None, None
        micro_batch = micro_batch_size // pipeline_microbatches
        activation_shape = (
            config.time_steps,
            micro_batch,
            384,
            image_height // patch_size,
            image_width // patch_size,
        )
        stage_shapes = (
            activation_shape,
            activation_shape,
            activation_shape,
            (config.time_steps, micro_batch, config.num_classes),
        )
        chunks_per_stage = len(stage_shapes) // pipeline_size
        start = pipeline_rank * chunks_per_stage
        input_shape = (
            (
                micro_batch,
                config.time_steps,
                in_channels,
                image_height,
                image_width,
            )
            if start == 0
            else stage_shapes[start - 1]
        )
        return (
            model,
            fsdp_roots,
            input_shape,
            stage_shapes[start + chunks_per_stage - 1],
        )
