from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from spikingjelly.activation_based.distributed.llm import (
    ModelConfig,
    TrainingConfig,
    plan_training,
)


@dataclass
class _Transformer:
    num_layers: int = 4
    hidden_size: int = 128
    num_attention_heads: int = 4
    num_query_groups: int | None = 2
    kv_channels: int = 32
    ffn_hidden_size: int = 512
    gated_linear_unit: bool = True
    normalization: str = "RMSNorm"
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    microbatch_group_size_per_vp_stage: int = 1
    calculate_per_token_loss: bool = True
    fp16: bool = False
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    pipeline_dtype: torch.dtype = torch.bfloat16
    fp8: str | None = None
    recompute_granularity: str | None = None
    recompute_method: str | None = None
    recompute_num_layers: int | None = None
    recompute_modules: list[str] | None = None


@dataclass(frozen=True, kw_only=True)
class _ModelConfig(ModelConfig):
    builder: ClassVar[str] = "package.ModelBuilder"


def _config() -> TrainingConfig:
    return TrainingConfig(
        model=_ModelConfig(
            transformer=_Transformer(),
            vocab_size=1024,
            max_sequence_length=512,
            time_steps=4,
        ),
        optimizer=SimpleNamespace(
            use_distributed_optimizer=True,
            fp16=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            lr=1e-4,
            min_lr=1e-5,
        ),
        dataset_builder="package.build_datasets",
        sequence_length=128,
        micro_batch_size=1,
        global_batch_size=4,
        train_steps=10,
    )


def test_plan_training_returns_train_config_without_mutating_input():
    config = _config()

    planned = plan_training(
        config,
        world_size=2,
        device_memory_bytes=8 * 1024**3,
    )

    assert planned is not config
    assert planned.model is not config.model
    assert planned.model.transformer is not config.model.transformer
    assert config.model.transformer.tensor_model_parallel_size == 1
    assert planned.model.transformer.recompute_granularity is None
    assert planned.memopt_level == 0


def test_memory_objective_prefers_snn_memopt_without_mcore_recompute():
    planned = plan_training(
        _config(),
        world_size=2,
        device_memory_bytes=8 * 1024**3,
        objective="memory",
    )

    assert planned.memopt_level == 1
    assert planned.model.transformer.recompute_granularity is None
    assert planned.model.transformer.recompute_modules is None


def test_planner_preserves_requested_snn_memopt():
    config = _config()
    config.memopt_level = 2
    planned = plan_training(
        config,
        world_size=1,
        device_memory_bytes=8 * 1024**3,
    )

    assert planned.memopt_level == 2


def test_planner_uses_only_non_overlapping_selective_recompute_as_fallback():
    config = _config()
    config.sequence_length = 127
    config.model.transformer.hidden_size = 120
    config.model.transformer.ffn_hidden_size = 480
    config.model.transformer.num_attention_heads = 3
    config.model.transformer.num_query_groups = 3
    config.model.transformer.kv_channels = 40
    planned = plan_training(
        config,
        world_size=2,
        device_memory_bytes=int(18.25 * 1024**2),
        memory_fraction=1.0,
    )

    assert planned.memopt_level == 1
    assert planned.model.transformer.recompute_granularity == "selective"
    assert planned.model.transformer.recompute_modules == ["core_attn"]


def test_planner_does_not_assume_fp8_hybrid_memory_savings():
    config = _config()
    config.sequence_length = 127
    config.model.transformer.hidden_size = 120
    config.model.transformer.ffn_hidden_size = 480
    config.model.transformer.num_attention_heads = 3
    config.model.transformer.num_query_groups = 3
    config.model.transformer.kv_channels = 40
    config.model.transformer.fp8 = "hybrid"

    with pytest.raises(ValueError, match="does not fit|No topology fits"):
        plan_training(
            config,
            world_size=2,
            device_memory_bytes=int(18.25 * 1024**2),
            memory_fraction=1.0,
        )
