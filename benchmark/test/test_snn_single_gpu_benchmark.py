from pathlib import Path
from types import SimpleNamespace

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


def test_matrix_records_child_timeouts(monkeypatch, tmp_path: Path):
    sources = []
    for label in ("baseline", "candidate"):
        root = tmp_path / label
        package = root / "spikingjelly"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        sources.extend(["--source", f"{label}={root}"])
    args = benchmark.build_parser().parse_args(
        [
            "matrix",
            *sources,
            "--models",
            "sew_resnet18",
            "--phases",
            "inference",
            "--executions",
            "eager",
            "--rounds",
            "1",
            "--timeout",
            "1",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    def timeout(command, _env, seconds):
        raise benchmark.subprocess.TimeoutExpired(command, seconds)

    monkeypatch.setattr(benchmark, "_run_isolated_case", timeout)
    payload = benchmark.run_matrix(args)

    assert len(payload["records"]) == 2
    assert len(payload["comparison"]["failures"]) == 2
    assert payload["comparison"]["acceptance"]["accepted"] is False


def test_isolated_case_timeout_kills_process_group(monkeypatch):
    class HungProcess:
        pid = 123
        returncode = None

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise benchmark.subprocess.TimeoutExpired(["case"], timeout)
            return "partial stdout", "partial stderr"

    process = HungProcess()
    killed = []
    monkeypatch.setattr(
        benchmark.subprocess, "Popen", lambda *_args, **_kwargs: process
    )
    monkeypatch.setattr(
        benchmark.os, "killpg", lambda pid, sig: killed.append((pid, sig))
    )

    with pytest.raises(benchmark.subprocess.TimeoutExpired) as raised:
        benchmark._run_isolated_case(["case"], {}, 1)

    assert killed == [(123, benchmark.signal.SIGKILL)]
    assert raised.value.stdout == "partial stdout"
    assert raised.value.stderr == "partial stderr"


def test_physical_gpu_selector_uses_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,GPU-example")

    assert benchmark._physical_gpu_selector(benchmark.torch.device("cuda", 0)) == "3"
    assert (
        benchmark._physical_gpu_selector(benchmark.torch.device("cuda", 1))
        == "GPU-example"
    )


def test_environment_metadata_filters_secrets(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SJ_MODE", "benchmark")
    monkeypatch.setenv("SJ_API_TOKEN", "secret")
    monkeypatch.setattr(
        benchmark.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            name="test-gpu", major=8, minor=0, total_memory=1024
        ),
    )
    monkeypatch.setattr(benchmark, "_git_metadata", lambda _root: {})
    monkeypatch.setattr(benchmark, "_nvidia_snapshot", lambda _selector: None)

    metadata = benchmark._environment_metadata(
        tmp_path,
        benchmark.torch.device("cuda", 0),
        tmp_path / "spikingjelly" / "__init__.py",
        "0",
    )

    assert metadata["environment"]["SJ_MODE"] == "benchmark"
    assert "SJ_API_TOKEN" not in metadata["environment"]


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


def test_aggregate_records_compares_only_matching_successful_rounds():
    records = [
        _record("baseline", 1, 10.0, 1000),
        _record("baseline", 2, 11.0, 1000),
        _record("candidate", 2, 9.0, 800),
        _record("candidate", 3, 8.0, 800),
        {"source_label": "candidate", "round": 1, "status": "error"},
    ]

    result = benchmark.aggregate_records(records, "baseline", "candidate")[
        "comparisons"
    ][0]

    assert result["rounds"] == 1
    assert result["baseline_round_medians_ms"] == [11.0]
    assert result["candidate_round_medians_ms"] == [9.0]


def test_probe_marks_unmeasured_physical_metrics_as_null():
    result = probe._unsupported(
        "triton_lif", "training", "cuda_required", "CUDA required"
    )

    assert result["status"] == "unsupported"
    assert result["kernel_launch_count"] is None
    assert result["allocation_count"] is None
    assert result["graph_break_count"] is None


def test_probe_records_child_timeouts(monkeypatch, tmp_path: Path):
    args = probe.build_parser().parse_args(
        [
            "--cases",
            "torch_lif",
            "--phases",
            "inference",
            "--timeout",
            "1",
            "--output",
            str(tmp_path / "probe.json"),
        ]
    )

    def timeout(command, **kwargs):
        raise probe.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(probe.subprocess, "run", timeout)
    result = probe.run_parent(args)["results"][0]

    assert result["status"] == "error"
    assert result["reason_code"] == "child_process_timeout"
    assert result["registration_environment"] == {
        "SJ_USE_TRITON_OP": "0",
        "SJ_USE_WRAP_TRITON": "0",
    }


@pytest.mark.parametrize(
    ("phase", "message"),
    [
        ("inference", "inference kernel construction failed"),
        ("training", "training kernel construction failed"),
    ],
)
def test_flexsn_probe_rejects_unavailable_phase_kernel(monkeypatch, phase, message):
    from spikingjelly.activation_based.neuron import flexsn

    class UnavailableFlexSN(probe.torch.nn.Module):
        _inductor_inference_available = False
        _inductor_inference_final_state_available = False
        _inductor_training_available = False

        def __init__(self, *_args, **_kwargs):
            super().__init__()

    monkeypatch.setattr(flexsn, "FlexSN", UnavailableFlexSN)

    with pytest.raises(RuntimeError, match=message):
        probe._build_model("triton_flexsn", phase, probe.torch.device("cpu"))


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
