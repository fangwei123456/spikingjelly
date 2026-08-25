import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from benchmark.plot_distributed_inference import _pareto_frontier
from benchmark.snn_llm.spikelm import SpikeLMConfig
from spikingjelly.activation_based.distributed.llm import (
    EvaluationConfig,
    SGLangGenerationConfig,
    generate_sglang,
)
from spikingjelly.activation_based.distributed.llm.inference import _EvaluationDataset
from spikingjelly.activation_based.distributed.llm.temporal import _reduce_time_batch


def test_inference_temporal_logit_reduction():
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    assert torch.equal(
        _reduce_time_batch(logits, 2, "sum"),
        torch.tensor([[6.0, 8.0], [10.0, 12.0]]),
    )
    assert torch.equal(
        _reduce_time_batch(logits, 2, "mean"),
        torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
    )
    with pytest.raises(ValueError, match="reduction"):
        _reduce_time_batch(logits, 2, "max")


def test_inference_plot_uses_pareto_frontier():
    points = [
        {"peak_memory_gib_median": 2.0, "throughput_median": 20.0},
        {"peak_memory_gib_median": 2.0, "throughput_median": 30.0},
        {"peak_memory_gib_median": 2.01, "throughput_median": 25.0},
        {"peak_memory_gib_median": 3.0, "throughput_median": 25.0},
        {"peak_memory_gib_median": 4.0, "throughput_median": 40.0},
    ]

    assert _pareto_frontier(points) == [points[1], points[4]]


def test_evaluation_config_and_padding_mask():
    model = SpikeLMConfig(
        transformer=object(), vocab_size=16, max_sequence_length=4, time_steps=2
    )
    config = EvaluationConfig(
        model=model,
        checkpoint=Path("checkpoint"),
        dataset_builder="package.dataset.build",
        sequence_length=4,
        micro_batch_size=2,
    )
    assert config.model is model

    with pytest.raises(ValueError, match="timing_warmup_batches"):
        EvaluationConfig(
            model=model,
            checkpoint=Path("checkpoint"),
            dataset_builder="package.dataset.build",
            sequence_length=4,
            micro_batch_size=2,
            timing_warmup_batches=-1,
        )
    with pytest.raises(ValueError, match="pipeline_microbatches"):
        EvaluationConfig(
            model=model,
            checkpoint=Path("checkpoint"),
            dataset_builder="package.dataset.build",
            sequence_length=4,
            micro_batch_size=2,
            pipeline_microbatches=0,
        )

    class Tokens(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {
                "input_ids": torch.arange(4),
                "labels": torch.arange(4),
            }

    padded = _EvaluationDataset(Tokens(), padded_size=2, sequence_length=4)

    assert torch.equal(padded[0]["loss_mask"], torch.ones(4))
    assert torch.equal(padded[1]["loss_mask"], torch.zeros(4))


def test_sglang_generation_config_rejects_invalid_dcp_artifact(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "spikingjelly_artifact_schema": 1,
                "num_key_value_heads": 8,
            }
        ),
        encoding="utf-8",
    )
    config = SGLangGenerationConfig(
        artifact=artifact,
        max_new_tokens=2,
        tensor_parallel_size=2,
        decode_context_parallel_size=2,
    )

    with pytest.raises(ValueError, match="TP-replicated KV heads"):
        generate_sglang(config, torch.ones((1, 2), dtype=torch.long))
