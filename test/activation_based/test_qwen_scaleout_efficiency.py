from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from benchmark.snn_llm.qwen_conversion import scaleout_efficiency as runner


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.collect_statistics = True
        self.grad_enabled = []

    def set_collect_statistics(self, enabled):
        self.collect_statistics = enabled

    def reset(self):
        pass

    def forward(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        self.grad_enabled.append(torch.is_grad_enabled())
        return SimpleNamespace(
            logits=input_ids.float().unsqueeze(-1), past_key_values={}
        )


def test_statistics_are_disabled_before_timing():
    model = _FakeModel()
    runner._prepare_benchmark_model(model)

    assert model.collect_statistics is False
    assert model.training is False


def test_gpu_snapshot_classification_requires_idle_utilization_and_memory():
    clean = {
        "utilization_gpu_percent_median": 2,
        "memory_used_mb_max": 512,
        "external_compute_processes": [],
        "sample_count": 60,
    }
    assert runner._measurement_label(clean) == "clean_gpu_measurement"
    busy = dict(clean, utilization_gpu_percent_median=10)
    assert runner._measurement_label(busy) == "shared_gpu_smoke_measurement"
    busy = dict(clean, memory_used_mb_max=4096)
    assert runner._measurement_label(busy) == "shared_gpu_smoke_measurement"
    busy = dict(clean, external_compute_processes=[{"pid": 123}])
    assert runner._measurement_label(busy) == "shared_gpu_smoke_measurement"
    short = dict(clean, sample_count=59)
    assert runner._measurement_label(short) == "shared_gpu_smoke_measurement"


def test_gpu_snapshot_uses_median_and_records_processes(monkeypatch):
    outputs = iter(
        [
            "0, A100, 1, 100, 81920\n",
            "111, python, 500\n",
            "0, A100, 9, 200, 81920\n",
            "\n",
        ]
    )

    def fake_run(*_args, **_kwargs):
        assert _kwargs["timeout"] == 30
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner.os, "getpid", lambda: 999)
    snapshot = runner._gpu_snapshot(sample_count=2, sample_interval_seconds=0)

    assert snapshot["utilization_gpu_percent_median"] == 5
    assert snapshot["memory_used_mb_max"] == 200
    assert snapshot["external_compute_processes"][0]["pid"] == 111


@pytest.mark.parametrize("value", (0.0, float("inf"), float("nan")))
def test_benchmark_median_must_be_positive_and_finite(value):
    with pytest.raises(RuntimeError, match="non-positive or non-finite"):
        runner._positive_median([value], "benchmark")

    assert runner._positive_median([1.0, 3.0], "benchmark") == 2.0


def test_backend_parity_requires_equal_tokens_and_bounded_logits():
    reference = {"logits": torch.ones(2, 3), "token_id": 7}
    candidate = {"logits": torch.ones(2, 3), "token_id": 7}
    assert runner._backend_parity(candidate, reference)["relative_l2"] == 0.0

    candidate["token_id"] = 8
    with pytest.raises(ValueError, match="token"):
        runner._backend_parity(candidate, reference)


def test_cli_rejects_non_cuda_and_exposes_fixed_benchmark_controls():
    with pytest.raises(SystemExit):
        runner._parse_args(["--device", "cpu"])

    with pytest.raises(SystemExit) as info:
        runner._parse_args(["--help"])
    assert info.value.code == 0
    assert 512 in runner.TIME_STEP_CHOICES
