r"""
**API Language** - 中文 | English

**中文：** 基于 Megatron Core 的大规模 SNN 语言模型规划、训练、checkpoint 与推理。

**English:** Megatron Core based planning, training, checkpointing, and inference
for large SNN language models.
"""

from .config import ModelBuilder, ModelConfig, TrainingConfig
from .planning import plan_training
from .training import train

__all__ = [
    "ModelBuilder",
    "ModelConfig",
    "TrainingConfig",
    "plan_training",
    "train",
]
