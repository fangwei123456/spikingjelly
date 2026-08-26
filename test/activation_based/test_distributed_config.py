from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from spikingjelly.activation_based.distributed.llm.config import (
    ModelConfig,
    TrainingConfig,
)


@dataclass(frozen=True, kw_only=True)
class _ModelConfig(ModelConfig):
    builder: ClassVar[str] = "package.ModelBuilder"


def _config(**kwargs):
    transformer = SimpleNamespace(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        sequence_parallel=True,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        calculate_per_token_loss=True,
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        recompute_granularity=None,
        recompute_modules=None,
        recompute_method=None,
        recompute_num_layers=None,
    )
    optimizer = SimpleNamespace(
        use_distributed_optimizer=True,
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        lr=1e-4,
        min_lr=1e-5,
    )
    values = dict(
        model=_ModelConfig(
            transformer=transformer,
            vocab_size=128,
            max_sequence_length=256,
            time_steps=4,
        ),
        optimizer=optimizer,
        dataset_builder="package.build_datasets",
        sequence_length=128,
        micro_batch_size=2,
        global_batch_size=16,
        train_steps=100,
    )
    values.update(kwargs)
    return TrainingConfig(**values)


def test_snn_training_config_uses_mcore_as_topology_source():
    config = _config()

    assert config.lr_decay_steps == config.train_steps
    assert config.model.transformer.tensor_model_parallel_size == 2
    assert not hasattr(config, "transformer")
    assert not hasattr(config, "model_builder")
    assert not hasattr(config, "model_builder_kwargs")
    assert not hasattr(config, "tensor_parallel_size")
    assert not hasattr(config, "pipeline_parallel_size")
    assert not hasattr(config, "sequence_parallel")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"global_batch_size": 15}, "divisible"),
        ({"eval_interval": 10}, "both be zero or positive"),
        ({"checkpoint_interval": 10}, "checkpoint_dir"),
        ({"lr_warmup_steps": 100}, "greater than"),
        ({"timing_warmup_steps": 100}, "timing_warmup_steps"),
    ],
)
def test_snn_training_config_rejects_inconsistent_values(changes, message):
    with pytest.raises(ValueError, match=message):
        _config(**changes)


def test_snn_training_config_accepts_cp_without_duplicating_topology():
    config = _config()
    config.model.transformer.context_parallel_size = 2

    configured = TrainingConfig(
        model=config.model,
        optimizer=config.optimizer,
        dataset_builder=config.dataset_builder,
        sequence_length=config.sequence_length,
        micro_batch_size=config.micro_batch_size,
        global_batch_size=config.global_batch_size,
        train_steps=config.train_steps,
    )

    assert configured.model.transformer.context_parallel_size == 2
    assert not hasattr(configured, "context_parallel_size")


def test_snn_training_config_allows_odd_sequence_without_cp():
    assert _config(sequence_length=127).sequence_length == 127


def test_snn_training_config_rejects_unshardable_cp_sequence():
    config = _config()
    config.model.transformer.context_parallel_size = 2

    with pytest.raises(ValueError, match="context_parallel_size"):
        TrainingConfig(
            model=config.model,
            optimizer=config.optimizer,
            dataset_builder=config.dataset_builder,
            sequence_length=130,
            micro_batch_size=config.micro_batch_size,
            global_batch_size=config.global_batch_size,
            train_steps=config.train_steps,
        )


def test_snn_training_config_rejects_precision_dtype_mismatch():
    config = _config()
    config.optimizer.params_dtype = torch.float32

    with pytest.raises(ValueError, match="params_dtype"):
        TrainingConfig(
            model=config.model,
            optimizer=config.optimizer,
            dataset_builder=config.dataset_builder,
            sequence_length=config.sequence_length,
            micro_batch_size=config.micro_batch_size,
            global_batch_size=config.global_batch_size,
            train_steps=config.train_steps,
        )


def test_model_config_rejects_invalid_position_embedding():
    with pytest.raises(ValueError, match="position_embedding_type"):
        _ModelConfig(
            transformer=_config().model.transformer,
            vocab_size=128,
            max_sequence_length=256,
            time_steps=4,
            position_embedding_type="alibi",
        )


def test_snn_memopt_rejects_overlapping_mcore_recompute():
    config = _config()
    config.model.transformer.recompute_granularity = "full"

    with pytest.raises(ValueError, match="cannot overlap"):
        _config(model=config.model, memopt_level=1)
