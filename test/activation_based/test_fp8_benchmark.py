from argparse import Namespace
from contextlib import contextmanager

import pytest
import torch

from benchmark.benchmark_fp8_training_inference import (
    _aggregate_precision_trials,
    _commit,
    _cuda_time_ms,
    _samples_per_second,
    validate_args,
)
import benchmark.benchmark_triton_neuron_kernels as triton_neuron_benchmark
from benchmark.benchmark_triton_neuron_kernels import (
    _higher_precision_variants,
    _write_markdown,
)
from benchmark.fp8_efficiency import (
    assess_model_efficiency,
    assess_triton_efficiency,
    require_efficiency,
)


def _patch_cuda_timing(monkeypatch):
    active_devices: list[torch.device] = []
    entered_devices: list[torch.device] = []
    event_synchronize_calls: list[None] = []
    created_events: list[None] = []

    @contextmanager
    def cuda_device(device: torch.device):
        active_devices.append(device)
        entered_devices.append(device)
        try:
            yield
        finally:
            active_devices.pop()

    def require_active_device(*args, **kwargs):
        del args, kwargs
        assert active_devices
        return 1.0

    class Event:
        def __init__(self, *, enable_timing: bool):
            assert enable_timing
            assert active_devices
            created_events.append(None)

        def record(self):
            assert active_devices

        def synchronize(self):
            assert active_devices
            event_synchronize_calls.append(None)

        def elapsed_time(self, other):
            assert isinstance(other, Event)
            assert active_devices
            return 2.0

    monkeypatch.setattr(torch.cuda, "device", cuda_device)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", require_active_device)
    monkeypatch.setattr(torch.cuda, "empty_cache", require_active_device)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", require_active_device)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", require_active_device)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", require_active_device)
    return entered_devices, event_synchronize_calls, created_events


def test_model_benchmark_timing_uses_requested_cuda_device(monkeypatch):
    entered_devices, event_synchronize_calls, _ = _patch_cuda_timing(monkeypatch)
    device = torch.device("cuda", 1)

    elapsed_ms = _cuda_time_ms(device, 2, lambda: None)

    assert elapsed_ms == pytest.approx(1.0)
    assert entered_devices == [device]
    assert len(event_synchronize_calls) == 1


def test_triton_benchmark_timing_uses_requested_cuda_device(monkeypatch):
    entered_devices, event_synchronize_calls, created_events = _patch_cuda_timing(
        monkeypatch
    )
    monkeypatch.setattr(triton_neuron_benchmark, "_cuda_sync", lambda device: None)
    device = torch.device("cuda", 1)
    calls = 0

    def benchmark_fn():
        nonlocal calls
        calls += 1
        if calls > 1:
            assert len(created_events) == 4

    result = triton_neuron_benchmark._measure(
        device=device, repeats=2, warmup=1, fn=benchmark_fn
    )

    assert result["median_ms"] == pytest.approx(2.0)
    assert entered_devices == [device]
    assert not event_synchronize_calls


def test_aggregate_precision_trials_uses_medians_and_preserves_raw_trials():
    trials = [
        {
            "precision": "fp8-torchao",
            "trial": trial,
            "training_ms": training_ms,
            "training_samples_per_sec": samples_per_sec,
            "training_peak_allocated_mb": 800.0 + trial,
            "training_peak_reserved_mb": 900.0 + trial,
            "inference_ms": training_ms / 2,
            "inference_samples_per_sec": samples_per_sec * 2,
            "inference_peak_allocated_mb": 400.0 + trial,
            "inference_peak_reserved_mb": 500.0 + trial,
            "preparation_ms": 10.0 + trial,
            "parameter_update_max_abs": 0.01,
            "output_checksum": 0.1,
            "capability_report": {"can_execute": True},
            "conversion_report": {"converted_modules": ["layers.0"]},
        }
        for trial, training_ms, samples_per_sec in (
            (0, 3.0, 100.0),
            (1, 1.0, 300.0),
            (2, 2.0, 200.0),
        )
    ]

    result = _aggregate_precision_trials("fp8-torchao", trials)

    assert result["training_ms"] == pytest.approx(2.0)
    assert result["training_samples_per_sec"] == pytest.approx(200.0)
    assert result["training_ms_p25"] == pytest.approx(1.5)
    assert result["training_ms_p75"] == pytest.approx(2.5)
    assert result["inference_ms"] == pytest.approx(1.0)
    assert result["training_peak_reserved_mb"] == pytest.approx(901.0)
    assert result["inference_peak_reserved_mb"] == pytest.approx(501.0)
    assert result["preparation_ms"] == pytest.approx(11.0)
    assert result["trial_count"] == 3
    assert result["trials"] == trials


@pytest.mark.parametrize("elapsed_ms", [0.0, -1.0, float("nan"), float("inf")])
def test_samples_per_second_rejects_invalid_cuda_timing(elapsed_ms):
    with pytest.raises(RuntimeError, match="timing must be finite and positive"):
        _samples_per_second(16, elapsed_ms)


def test_samples_per_second_converts_milliseconds():
    assert _samples_per_second(16, 2.0) == pytest.approx(8000.0)


def test_triton_benchmark_includes_fp16_and_bf16_comparisons():
    variants = _higher_precision_variants()

    assert [(name, compute) for name, _, compute in variants] == [
        ("float16", "fp16"),
        ("bfloat16", "bf16"),
    ]


def test_model_benchmark_commit_prefers_explicit_value(monkeypatch):
    monkeypatch.delenv("SJ_COMMIT", raising=False)
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.setenv("CODEX_COMMIT", "environment-commit")

    assert _commit("explicit-commit") == "explicit-commit"
    assert _commit() == "environment-commit"


@pytest.mark.parametrize(
    ("speedup_name", "invalid_value"),
    [
        ("min_training_speedup", 0.0),
        ("min_training_speedup", float("nan")),
        ("min_training_speedup", float("inf")),
        ("min_inference_speedup", 0.0),
        ("min_inference_speedup", float("nan")),
        ("min_inference_speedup", float("inf")),
    ],
)
def test_model_benchmark_rejects_invalid_min_speedup(speedup_name, invalid_value):
    args = Namespace(
        baseline_precision="bf16",
        precisions=["bf16", "fp8-torchao"],
        batch_size=16,
        width=16,
        num_classes=16,
        training_steps=1,
        inference_steps=1,
        trials=2,
        warmup=1,
        depth=2,
        min_training_speedup=1.0,
        min_inference_speedup=1.0,
    )
    setattr(args, speedup_name, invalid_value)

    with pytest.raises(ValueError, match="must be finite and > 0"):
        validate_args(args)


def test_triton_markdown_flattens_multiline_failure_reason(tmp_path):
    output = tmp_path / "benchmark.md"
    rows = [
        {
            "T": 1,
            "N": 1,
            "neuron_type": "if",
            "variant": "mp_plan_float8_e4m3fn_fp8",
            "process": "inference_forward",
            "success": False,
            "failure_reason": "compile failed:\nline one\r\nline two",
        }
    ]
    metadata = {
        "host": "test-host",
        "gpu": "test-gpu",
        "commit": "test-commit",
        "commit_source": "test",
        "command": "benchmark command",
    }
    efficiency = {"min_speedup": 1.05, "comparisons": [], "failures": []}

    _write_markdown(output, rows, metadata, efficiency)

    rendered = output.read_text(encoding="utf-8")
    assert (
        "- `T=1 N=1 neuron=if variant=mp_plan_float8_e4m3fn_fp8 "
        "process=inference_forward: compile failed: line one  line two`" in rendered
    )
    assert "\nline one" not in rendered


def test_assess_model_efficiency_compares_training_inference_and_memory():
    results = [
        {
            "precision": "bf16",
            "training_samples_per_sec": 100.0,
            "inference_samples_per_sec": 200.0,
            "training_peak_allocated_mb": 1000.0,
            "inference_peak_allocated_mb": 600.0,
        },
        {
            "precision": "fp8-torchao",
            "training_samples_per_sec": 125.0,
            "inference_samples_per_sec": 250.0,
            "training_peak_allocated_mb": 800.0,
            "inference_peak_allocated_mb": 450.0,
        },
    ]

    report = assess_model_efficiency(
        results,
        baseline_precision="bf16",
        min_training_speedup=1.05,
        min_inference_speedup=1.05,
    )

    assert report["passed"] is True
    comparison = report["comparisons"][0]
    assert comparison["training_speedup"] == pytest.approx(1.25)
    assert comparison["inference_speedup"] == pytest.approx(1.25)
    assert comparison["training_memory_ratio"] == pytest.approx(0.8)
    assert comparison["inference_memory_ratio"] == pytest.approx(0.75)


def test_assess_model_efficiency_reports_each_failed_process():
    results = [
        {
            "precision": "fp32",
            "training_samples_per_sec": 100.0,
            "inference_samples_per_sec": 200.0,
            "training_peak_allocated_mb": 1000.0,
            "inference_peak_allocated_mb": 600.0,
        },
        {
            "precision": "fp8-te",
            "training_samples_per_sec": 104.0,
            "inference_samples_per_sec": 180.0,
            "training_peak_allocated_mb": 900.0,
            "inference_peak_allocated_mb": 500.0,
        },
    ]

    report = assess_model_efficiency(
        results,
        baseline_precision="fp32",
        min_training_speedup=1.05,
        min_inference_speedup=1.05,
    )

    assert report["passed"] is False
    assert report["failures"] == [
        "fp8-te training speedup 1.0400x < 1.0500x",
        "fp8-te inference speedup 0.9000x < 1.0500x",
    ]
    with pytest.raises(RuntimeError, match="fp8-te training speedup"):
        require_efficiency(report)


def test_assess_model_efficiency_reports_missing_metric():
    results = [
        {
            "precision": "bf16",
            "inference_samples_per_sec": 200.0,
            "training_peak_allocated_mb": 1000.0,
            "inference_peak_allocated_mb": 600.0,
        },
        {
            "precision": "fp8-te",
            "training_samples_per_sec": 120.0,
            "inference_samples_per_sec": 220.0,
            "training_peak_allocated_mb": 900.0,
            "inference_peak_allocated_mb": 500.0,
        },
    ]

    with pytest.raises(ValueError, match="training_samples_per_sec is required"):
        assess_model_efficiency(
            results,
            baseline_precision="bf16",
            min_training_speedup=1.0,
            min_inference_speedup=1.0,
        )


def test_assess_model_efficiency_reports_missing_precision():
    with pytest.raises(ValueError, match="precision is required"):
        assess_model_efficiency(
            [
                {
                    "training_samples_per_sec": 100.0,
                    "inference_samples_per_sec": 200.0,
                    "training_peak_allocated_mb": 1000.0,
                    "inference_peak_allocated_mb": 600.0,
                }
            ],
            baseline_precision="bf16",
            min_training_speedup=1.0,
            min_inference_speedup=1.0,
        )


def test_assess_triton_efficiency_reports_fp8_group_without_baseline():
    rows = [
        {
            "T": 4,
            "N": 8,
            "neuron_type": "if",
            "process": "inference_forward",
            "variant": "stable_fp32",
            "success": True,
            "median_ms": 2.0,
            "peak_allocated_mb": 8.0,
        },
        {
            "T": 8,
            "N": 8,
            "neuron_type": "if",
            "process": "inference_forward",
            "variant": "mp_plan_float8_e4m3fn_fp8",
            "success": True,
            "median_ms": 1.0,
            "peak_allocated_mb": 4.0,
        },
    ]

    report = assess_triton_efficiency(rows, min_speedup=1.0)

    assert report["passed"] is False
    assert report["failures"] == [
        "T=4 N=8 neuron=if process=inference_forward has no successful FP8 "
        "prepared-plan result",
        "T=8 N=8 neuron=if process=inference_forward has no successful "
        "stable_fp32 baseline",
    ]


def test_assess_model_efficiency_rejects_fp8_baseline():
    with pytest.raises(ValueError, match="must not be an FP8 variant"):
        assess_model_efficiency(
            [
                {
                    "precision": "fp8-te",
                    "training_samples_per_sec": 100.0,
                    "inference_samples_per_sec": 200.0,
                    "training_peak_allocated_mb": 1000.0,
                    "inference_peak_allocated_mb": 600.0,
                }
            ],
            baseline_precision="fp8-te",
            min_training_speedup=1.0,
            min_inference_speedup=1.0,
        )


@pytest.mark.parametrize("invalid_speedup", [float("nan"), float("inf")])
def test_assess_efficiency_rejects_nonfinite_speedup(invalid_speedup):
    with pytest.raises(ValueError, match="must be finite and positive"):
        assess_model_efficiency(
            [],
            baseline_precision="bf16",
            min_training_speedup=invalid_speedup,
            min_inference_speedup=1.0,
        )
    with pytest.raises(ValueError, match="must be finite and positive"):
        assess_triton_efficiency([], min_speedup=invalid_speedup)


def test_model_benchmark_rejects_zero_warmup():
    args = Namespace(
        baseline_precision="bf16",
        precisions=["bf16", "fp8-torchao"],
        batch_size=16,
        width=16,
        num_classes=16,
        training_steps=1,
        inference_steps=1,
        trials=2,
        warmup=0,
        depth=2,
        min_training_speedup=1.0,
        min_inference_speedup=1.0,
    )

    with pytest.raises(ValueError, match="--warmup must be >= 1"):
        validate_args(args)


@pytest.mark.parametrize("invalid_value", [None, "invalid", float("nan"), float("inf")])
def test_assess_model_efficiency_rejects_invalid_metric(invalid_value):
    results = [
        {
            "precision": "bf16",
            "training_samples_per_sec": invalid_value,
            "inference_samples_per_sec": 200.0,
            "training_peak_allocated_mb": 1000.0,
            "inference_peak_allocated_mb": 600.0,
        },
        {
            "precision": "fp8-te",
            "training_samples_per_sec": 120.0,
            "inference_samples_per_sec": 220.0,
            "training_peak_allocated_mb": 900.0,
            "inference_peak_allocated_mb": 500.0,
        },
    ]

    with pytest.raises(ValueError, match="must be a finite positive number"):
        assess_model_efficiency(
            results,
            baseline_precision="bf16",
            min_training_speedup=1.0,
            min_inference_speedup=1.0,
        )


def test_assess_triton_efficiency_selects_best_prepared_fp8_variant():
    rows = [
        {
            "T": 32,
            "N": 4096,
            "neuron_type": "lif",
            "process": "inference_forward",
            "variant": "stable_fp32",
            "success": True,
            "median_ms": 2.0,
            "peak_allocated_mb": 100.0,
        },
        {
            "T": 32,
            "N": 4096,
            "neuron_type": "lif",
            "process": "inference_forward",
            "variant": "mp_safe_float8_e4m3fn_fp32",
            "success": True,
            "median_ms": 3.0,
            "peak_allocated_mb": 80.0,
        },
        {
            "T": 32,
            "N": 4096,
            "neuron_type": "lif",
            "process": "inference_forward",
            "variant": "mp_plan_float8_e4m3fn_fp32",
            "compute_dtype": "fp32",
            "success": True,
            "median_ms": 1.5,
            "peak_allocated_mb": 75.0,
        },
        {
            "T": 32,
            "N": 4096,
            "neuron_type": "lif",
            "process": "inference_forward",
            "variant": "mp_plan_float8_e5m2_fp16",
            "compute_dtype": "fp16",
            "success": True,
            "median_ms": 1.0,
            "peak_allocated_mb": 70.0,
        },
    ]

    report = assess_triton_efficiency(rows, min_speedup=1.05)

    assert report["passed"] is True
    comparison = report["comparisons"][0]
    assert comparison["best_fp8_variant"] == "mp_plan_float8_e5m2_fp16"
    assert comparison["best_compute_dtype"] == "fp16"
    assert comparison["speedup"] == pytest.approx(2.0)
    assert comparison["memory_ratio"] == pytest.approx(0.7)


def test_assess_triton_efficiency_fails_when_a_process_has_no_usable_fp8_plan():
    rows = [
        {
            "T": 32,
            "N": 4096,
            "neuron_type": "if",
            "process": "training_forward_backward",
            "variant": "stable_fp32",
            "success": True,
            "median_ms": 2.0,
            "peak_allocated_mb": 100.0,
        },
        {
            "T": 32,
            "N": 4096,
            "neuron_type": "if",
            "process": "training_forward_backward",
            "variant": "mp_plan_float8_e4m3fn_fp32",
            "success": False,
            "failure_reason": "unsupported",
        },
    ]

    report = assess_triton_efficiency(rows, min_speedup=1.05)

    assert report["passed"] is False
    assert report["failures"] == [
        "T=32 N=4096 neuron=if process=training_forward_backward has no "
        "successful FP8 prepared-plan result"
    ]


def test_assess_triton_efficiency_reports_failed_baseline():
    rows = [
        {
            "T": 16,
            "N": 2048,
            "neuron_type": "plif",
            "process": "inference_forward",
            "variant": "stable_fp32",
            "success": False,
            "failure_reason": "out of memory",
        }
    ]

    report = assess_triton_efficiency(rows, min_speedup=1.05)

    assert report["passed"] is False
    assert report["failures"] == [
        "T=16 N=2048 neuron=plif process=inference_forward stable_fp32 "
        "baseline failed: out of memory",
    ]


def test_assess_triton_efficiency_reports_absent_baseline():
    report = assess_triton_efficiency([], min_speedup=1.05)

    assert report["passed"] is False
    assert report["failures"] == ["no usable stable_fp32 baseline results"]


def test_assess_triton_efficiency_reports_missing_baseline_group_field():
    rows = [
        {
            "T": 16,
            "N": 2048,
            "neuron_type": "plif",
            "variant": "stable_fp32",
            "success": True,
            "median_ms": 2.0,
            "peak_allocated_mb": 100.0,
        }
    ]

    report = assess_triton_efficiency(rows, min_speedup=1.05)

    assert report["passed"] is False
    assert report["failures"] == [
        "T=16 N=2048 neuron=plif process=None stable_fp32 baseline is missing "
        "required group fields: process"
    ]


def test_assess_triton_efficiency_rejects_duplicate_successful_baseline():
    baseline = {
        "T": 16,
        "N": 2048,
        "neuron_type": "plif",
        "process": "inference_forward",
        "variant": "stable_fp32",
        "success": True,
        "median_ms": 2.0,
        "peak_allocated_mb": 100.0,
    }
    candidate = {
        **baseline,
        "variant": "mp_plan_float8_e4m3fn_fp32",
        "compute_dtype": "fp32",
        "median_ms": 1.0,
        "peak_allocated_mb": 80.0,
    }

    report = assess_triton_efficiency(
        [baseline, dict(baseline), candidate], min_speedup=1.05
    )

    assert report["passed"] is False
    assert report["failures"] == [
        "T=16 N=2048 neuron=plif process=inference_forward has duplicate "
        "successful stable_fp32 baselines"
    ]
    assert len(report["comparisons"]) == 1
