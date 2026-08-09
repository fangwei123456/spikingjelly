from pathlib import Path

import pytest

import benchmark.benchmark_snn_single_gpu as benchmark
import benchmark.probe_snn_compile_boundary as probe


def _record(label: str, round_index: int, latency_ms: float, peak_bytes: int):
    return {
        "source_label": label,
        "round": round_index,
        "case": {
            "model": "sew_resnet18",
            "phase": "training",
            "execution": "compile",
            "T": 4,
            "batch_size": 32,
            "image_size": 224,
        },
        "timing": {"median_ms": latency_ms},
        "memory": {"peak_allocated_bytes": peak_bytes},
    }


def test_case_parser_keeps_required_reproduction_fields(tmp_path: Path):
    args = benchmark.build_parser().parse_args(
        [
            "case",
            "--model",
            "spikformer_ti",
            "--phase",
            "inference",
            "--execution",
            "compile",
            "--batch-size",
            "64",
            "--warmup",
            "100",
            "--steps",
            "500",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert (args.T, args.image_size, args.seed) == (4, 224, 20260808)
    assert (args.model, args.phase, args.execution) == (
        "spikformer_ti",
        "inference",
        "compile",
    )


def test_source_parser_requires_one_baseline_and_one_candidate(tmp_path: Path):
    package = tmp_path / "spikingjelly"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two"):
        benchmark.parse_source_specs([f"baseline={tmp_path}"])
    with pytest.raises(ValueError, match="unique"):
        benchmark.parse_source_specs([f"baseline={tmp_path}", f"baseline={tmp_path}"])


def test_physical_gpu_selector_uses_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,GPU-example")

    assert benchmark._physical_gpu_selector(benchmark.torch.device("cuda", 0)) == "3"
    assert (
        benchmark._physical_gpu_selector(benchmark.torch.device("cuda", 1))
        == "GPU-example"
    )


def test_aggregate_records_reports_paired_latency_and_memory_changes():
    records = [
        _record("baseline", 1, 10.0, 1000),
        _record("candidate", 1, 9.0, 800),
        _record("candidate", 2, 8.0, 800),
        _record("baseline", 2, 10.0, 1000),
        _record("baseline", 3, 11.0, 1000),
        _record("candidate", 3, 9.0, 800),
    ]

    comparison = benchmark.aggregate_records(records, "baseline", "candidate")
    result = comparison["comparisons"][0]

    assert result["rounds"] == 3
    assert result["latency_change_pct"] == pytest.approx(-10.0)
    assert result["peak_allocated_change_pct"] == pytest.approx(-20.0)
    assert result["all_candidate_rounds_faster"] is True
    assert result["candidate_round_spread"] == pytest.approx(9.0 / 8.0)
    acceptance = comparison["acceptance"]
    assert acceptance["qualifying_model_families"] == ["sew_resnet18"]
    assert acceptance["at_least_two_model_families"] is False
    assert acceptance["three_stable_rounds_per_case"] is False
    assert acceptance["no_case_latency_regression_over_3pct"] is True
    assert acceptance["accepted"] is False


def test_probe_marks_unmeasured_physical_metrics_as_null():
    result = probe._unsupported(
        "triton_lif", "training", "cuda_required", "CUDA required"
    )

    assert result["status"] == "unsupported"
    assert result["kernel_launch_count"] is None
    assert result["allocation_count"] is None
    assert result["graph_break_count"] is None


def test_flexsn_probe_core_is_differentiable():
    x = probe.torch.randn(2, 3, requires_grad=True)
    state = probe.torch.zeros_like(x)

    output, next_state = probe._lif_core(x, state)
    (output + next_state).sum().backward()

    assert x.grad is not None


def test_probe_clears_dynamo_counters():
    counters = probe.torch._dynamo.utils.counters
    counters["stats"]["unique_graphs"] = 7

    probe._clear_dynamo_state()

    assert not counters
