r"""
**API Language** - 中文 | English

**中文：** 基于 PyTorch DDP/FSDP2 与 architecture-specific 通道 TP 的 SNN
视觉分类训练。

**English:** SNN vision classification with native PyTorch DDP/FSDP2 and
architecture-specific channel tensor parallelism.
"""

from .config import ModelBuilder, ModelConfig, TrainingConfig
from .sew_resnet import SEWResNet34Config
from .spikformer import SpikformerCIFAR10Config, SpikformerConfig
from .training import build_imagefolder_datasets, train_classification

__all__ = [
    "ModelBuilder",
    "ModelConfig",
    "SEWResNet34Config",
    "SpikformerCIFAR10Config",
    "SpikformerConfig",
    "TrainingConfig",
    "build_imagefolder_datasets",
    "train_classification",
]
