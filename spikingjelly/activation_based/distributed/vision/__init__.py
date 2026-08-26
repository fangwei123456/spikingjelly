r"""
**API Language** - 中文 | English

**中文：** 基于 PyTorch DDP/FSDP2 与 architecture-specific 通道 TP/PP 的
SNN 视觉分类训练、评测与预测。

**English:** SNN vision classification training, evaluation, and prediction
with native PyTorch DDP/FSDP2 and architecture-specific channel TP/PP.
"""

from .config import (
    EvaluationConfig,
    ModelBuilder,
    ModelConfig,
    PredictionConfig,
    TrainingConfig,
)
from .inference import (
    evaluate_classification,
    export_inference_artifact,
    load_inference_artifact,
    predict_classification,
)
from .training import build_imagefolder_datasets, train_classification

__all__ = [
    "EvaluationConfig",
    "ModelBuilder",
    "ModelConfig",
    "PredictionConfig",
    "TrainingConfig",
    "build_imagefolder_datasets",
    "evaluate_classification",
    "export_inference_artifact",
    "load_inference_artifact",
    "predict_classification",
    "train_classification",
]
