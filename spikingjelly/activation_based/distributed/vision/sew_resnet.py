from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from spikingjelly.activation_based import functional, memopt, neuron
from spikingjelly.activation_based.distributed.tensor_parallel import (
    ChannelShardBatchNorm2d,
    ChannelShardConv2d,
)
from spikingjelly.activation_based.model.sew_resnet import (
    BasicBlock,
    sew_resnet34,
)

from .config import ModelBuilder, ModelConfig


class _SEWHead(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(self.layer4(x))
        return self.fc(torch.flatten(x, 2))


def _pipeline_stage(model: nn.Module, rank: int, size: int) -> nn.Module:
    if size == 1:
        return model
    if size not in {2, 4}:
        raise ValueError("SEW-ResNet34 PP size must be 1, 2, or 4.")
    chunks = (
        nn.Sequential(model.conv1, model.bn1, model.sn1, model.maxpool, model.layer1),
        model.layer2,
        model.layer3,
        _SEWHead(model),
    )
    chunks_per_stage = len(chunks) // size
    start = rank * chunks_per_stage
    return nn.Sequential(*chunks[start : start + chunks_per_stage])


@dataclass(frozen=True)
class SEWResNet34Config(ModelConfig):
    builder: ClassVar[str] = (
        "spikingjelly.activation_based.distributed.vision.sew_resnet.SEWResNet34Builder"
    )
    image_size: int = 224
    in_channels: int = 3
    connection: str = "ADD"
    neuron_backend: str = "torch"
    tau: float = 2.0
    detach_reset: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.in_channels != 3:
            raise ValueError("The built-in SEW-ResNet34 requires in_channels=3.")
        if self.connection not in {"ADD", "AND", "IAND"}:
            raise ValueError("connection must be 'ADD', 'AND', or 'IAND'.")
        if self.tau <= 1.0:
            raise ValueError("tau must be greater than 1.0.")
        if self.step_mode == "s" and self.neuron_backend == "triton":
            raise ValueError("The Triton neuron backend requires step_mode='m'.")


SEWResNet34Config.__init__.__doc__ = r"""Configure the built-in SEW-ResNet34.

**API Language** - 中文 | English

**中文：** 声明 ImageNet 型 SEW-ResNet34 的时间步、输入、连接方式和 LIF 参数。
模型使用 BasicBlock 内显式的 colwise/rowwise 通道并行策略。

**English:** Declare time-step, input, residual-connection, and LIF parameters
for the built-in ImageNet-style SEW-ResNet34. BasicBlocks use an explicit
colwise/rowwise channel-parallel strategy.

:param time_steps: SNN 时间步。 / SNN time steps.
:type time_steps: int
:param num_classes: 分类类别数。 / Number of classes.
:type num_classes: int
:param step_mode: ``"s"``（单步）或 ``"m"``（多步）。 / ``"s"``
    (single-step) or ``"m"`` (multi-step).
:type step_mode: str
:param image_size: 输入图像边长。 / Input image side length.
:type image_size: int
:param in_channels: 固定为 ``3``。 / Must be ``3``.
:type in_channels: int
:param connection: ``"ADD"``、``"AND"`` 或 ``"IAND"`` 残差连接。 / Residual
    connection type.
:type connection: str
:param neuron_backend: LIFNode backend。 / LIFNode backend.
:type neuron_backend: str
:param tau: LIF 膜时间常数。 / LIF membrane time constant.
:type tau: float
:param detach_reset: 是否分离 reset 梯度。 / Whether to detach reset gradients.
:type detach_reset: bool
:raises ValueError: 图像、通道、连接方式或神经元参数无效。 / If image, channel,
    connection, or neuron values are invalid.
"""


class SEWResNet34Builder(ModelBuilder):
    def build(
        self,
        *,
        process_group: Optional[Any],
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
        if not isinstance(config, SEWResNet34Config):
            raise TypeError("SEWResNet34Builder requires SEWResNet34Config.")

        model = sew_resnet34(
            pretrained=False,
            cnf=config.connection,
            spiking_neuron=neuron.LIFNode,
            num_classes=config.num_classes,
            tau=config.tau,
            detach_reset=config.detach_reset,
            backend=config.neuron_backend,
        )
        functional.set_step_mode(model, config.step_mode)

        world_size = dist.get_world_size(process_group) if process_group else 1
        if world_size > 1:
            for block in model.modules():
                if not isinstance(block, BasicBlock):
                    continue
                block.conv1 = ChannelShardConv2d(block.conv1, process_group, "colwise")
                block.bn1 = ChannelShardBatchNorm2d(block.bn1, process_group)
                block.conv2 = ChannelShardConv2d(block.conv2, process_group, "rowwise")

        if pipeline_size > 1 and memopt_level:
            raise ValueError("SEW-ResNet34 memopt is not supported together with PP.")
        model = _pipeline_stage(model, pipeline_rank, pipeline_size)
        fsdp_roots = tuple(
            name
            for name, module in model.named_modules()
            if isinstance(module, BasicBlock)
        )
        if memopt_level:
            dummy = torch.zeros(
                config.time_steps,
                micro_batch_size,
                config.in_channels,
                config.image_size,
                config.image_size,
                device=device,
            )
            model = memopt.memory_optimization(
                model,
                BasicBlock,
                dummy_input=(dummy,),
                compress_x=memopt_compress_inputs,
                level=memopt_level,
            )
        model.to(device)
        if pipeline_size == 1:
            return model, fsdp_roots, None, None
        micro_batch = micro_batch_size // pipeline_microbatches
        stem_size = (config.image_size + 3) // 4
        stage_shapes = (
            (config.time_steps, micro_batch, 64, stem_size, stem_size),
            (
                config.time_steps,
                micro_batch,
                128,
                (stem_size + 1) // 2,
                (stem_size + 1) // 2,
            ),
            (
                config.time_steps,
                micro_batch,
                256,
                (stem_size + 3) // 4,
                (stem_size + 3) // 4,
            ),
            (config.time_steps, micro_batch, config.num_classes),
        )
        chunks_per_stage = len(stage_shapes) // pipeline_size
        start = pipeline_rank * chunks_per_stage
        input_shape = (
            (
                micro_batch,
                config.time_steps,
                config.in_channels,
                config.image_size,
                config.image_size,
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
