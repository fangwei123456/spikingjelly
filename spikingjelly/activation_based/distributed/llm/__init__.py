r"""
**API Language** - 中文 | English

**中文：** 基于 Megatron Core 的大规模 SNN 语言模型规划、训练、checkpoint
与训练一致推理，以及独立环境中的 SGLang 离线推理。

**English:** Megatron Core based planning, training, checkpointing, and
training-consistent inference for large SNN language models, plus SGLang offline
inference from a separate environment.
"""

from .config import (
    EvaluationConfig,
    MCoreGenerationConfig,
    ModelBuilder,
    ModelConfig,
    SGLangEngineConfig,
    TrainingConfig,
)
from .inference import evaluate, generate, generate_mcore, load_for_inference
from .planning import plan_training
from .sglang import open_sglang_engine
from .sglang_export import SGLangExportStage, export_sglang_artifact
from .training import train

__all__ = [
    "EvaluationConfig",
    "MCoreGenerationConfig",
    "ModelBuilder",
    "ModelConfig",
    "SGLangExportStage",
    "SGLangEngineConfig",
    "TrainingConfig",
    "evaluate",
    "export_sglang_artifact",
    "generate",
    "generate_mcore",
    "load_for_inference",
    "open_sglang_engine",
    "plan_training",
    "train",
]
