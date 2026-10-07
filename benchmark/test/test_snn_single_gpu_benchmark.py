import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import benchmark.benchmark_snn_single_gpu as benchmark


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
        "dynamo": {"graph_break_count": 0, "recompile_count": 0},
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
            "--fp8-fallback-dtype",
            "bf16",
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
    assert (args.neuron_family, args.precision, args.fp8_recipe) == (
        "lif",
        "fp32",
        "auto",
    )
    assert args.fp8_fallback_dtype == "bf16"


def test_case_parser_rejects_removed_backend_option(tmp_path: Path):
    with pytest.raises(SystemExit):
        benchmark.build_parser().parse_args(
            [
                "case",
                "--model",
                "spikformer_ti",
                "--phase",
                "training",
                "--execution",
                "eager",
                "--batch-size",
                "16",
                "--warmup",
                "30",
                "--steps",
                "25",
                "--neuron-backend",
                "triton",
                "--output",
                str(tmp_path / "removed.json"),
            ]
        )


def test_spikformer_s_uses_image_batch():
    args = SimpleNamespace(
        model="spikformer_s",
        batch_size=2,
        image_size=32,
        T=4,
        num_classes=10,
        channels_last=False,
    )
    x, target = benchmark._make_batch(args, torch.device("cpu"))
    assert x.shape == (2, 3, 32, 32)
    assert target.shape == (2,)


def test_full_model_runner_supports_izhikevich_neurons():
    from spikingjelly.activation_based import neuron

    model = benchmark._build_model(
        "sew_resnet18", T=2, num_classes=3, neuron_family="izhikevich"
    )
    neurons = [
        module for module in model.modules() if isinstance(module, neuron.BaseNode)
    ]
    assert neurons
    assert all(isinstance(module, neuron.IzhikevichNode) for module in neurons)
    assert all(not hasattr(module, "backend") for module in neurons)


def test_case_parser_builds_triton_throughput_compile_options(tmp_path: Path):
    args = benchmark.build_parser().parse_args(
        [
            "case",
            "--model",
            "sew_resnet18",
            "--phase",
            "inference",
            "--execution",
            "compile",
            "--batch-size",
            "32",
            "--warmup",
            "20",
            "--steps",
            "30",
            "--compile-layout-optimization",
            "off",
            "--compile-max-autotune",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert benchmark.compile_options(args) == {
        "triton.cudagraphs": False,
        "triton.cudagraph_trees": False,
        "layout_optimization": False,
        "max_autotune": True,
    }


def test_source_parser_requires_one_baseline_and_one_candidate(tmp_path: Path):
    package = tmp_path / "spikingjelly"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two"):
        benchmark.parse_source_specs([f"baseline={tmp_path}"])
    with pytest.raises(ValueError, match="unique"):
        benchmark.parse_source_specs([f"baseline={tmp_path}", f"baseline={tmp_path}"])


def test_profile_hooks_record_metadata_once(monkeypatch, tmp_path: Path):
    ranges: list[str | None] = []
    monkeypatch.setattr(
        benchmark.torch.cuda.nvtx,
        "range_push",
        lambda name: ranges.append(name),
    )
    monkeypatch.setattr(
        benchmark.torch.cuda.nvtx,
        "range_pop",
        lambda: ranges.append(None),
    )

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with benchmark.nsys.module_ranges(model, tmp_path / "tensors.jsonl"):
        model(torch.randn(1, 2))
        model(torch.randn(1, 2))

    records = [
        json.loads(line)
        for line in (tmp_path / "tensors.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == ["input", "output"]
    assert len(ranges) == 4


@pytest.mark.parametrize(
    ("execution", "profile", "reason"),
    [("compile", True, "eager diagnostic"), ("eager", False, "--profile")],
)
def test_module_detail_requires_separate_profiled_eager_run(
    monkeypatch, tmp_path: Path, execution: str, profile: bool, reason: str
):
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    argv = [
        "case",
        "--model",
        "sew_resnet18",
        "--phase",
        "inference",
        "--execution",
        execution,
        "--batch-size",
        "1",
        "--warmup",
        "1",
        "--steps",
        "1",
        "--tensor-metadata",
        str(tmp_path / "tensors.jsonl"),
        "--output",
        str(tmp_path / "result.json"),
    ]
    if profile:
        argv.append("--profile")
    args = benchmark.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match=reason):
        benchmark.run_case(args)


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
            "--profile",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    commands = []

    def timeout(command, _env, seconds):
        commands.append(command)
        raise benchmark.subprocess.TimeoutExpired(command, seconds)

    monkeypatch.setattr(benchmark, "_run_isolated_case", timeout)
    payload = benchmark.run_matrix(args)

    assert len(payload["records"]) == 2
    assert len(payload["comparison"]["failures"]) == 2
    assert payload["comparison"]["performance_gates"]["met"] is False
    assert all("--profile" in command for command in commands)
    assert all("--tensor-metadata" not in command for command in commands)


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


def test_stop_monitor_tolerates_exit_race():
    class ExitedMonitor:
        def poll(self):
            return None

        def terminate(self):
            raise ProcessLookupError

    benchmark._stop_monitor(ExitedMonitor())


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
    gates = comparison["performance_gates"]
    assert gates["qualifying_model_families"] == ["sew_resnet18"]
    assert gates["at_least_two_model_families"] is False
    assert gates["three_stable_rounds_per_case"] is False
    assert gates["no_case_latency_regression_over_3pct"] is True
    assert gates["met"] is False


def test_aggregate_records_rejects_zero_latency_measurements():
    records = [
        _record("baseline", 1, 0.0, 1000),
        _record("candidate", 1, 0.0, 800),
    ]

    comparison = benchmark.aggregate_records(records, "baseline", "candidate")
    result = comparison["comparisons"][0]

    assert result["latency_change_pct"] is None
    assert result["candidate_round_spread"] is None
    assert comparison["performance_gates"]["met"] is False


def test_aggregate_does_not_pair_different_neuron_families():
    baseline = _record("baseline", 1, 10.0, 1000)
    candidate = _record("candidate", 1, 9.0, 800)
    baseline["case"]["neuron_family"] = "if"
    candidate["case"]["neuron_family"] = "plif"
    result = benchmark.aggregate_records([baseline, candidate], "baseline", "candidate")
    assert result["comparisons"] == []


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


@pytest.mark.parametrize("metric", ["graph_break_count", "recompile_count"])
def test_aggregate_records_rejects_invalid_compile_metrics(metric):
    baseline = _record("baseline", 1, 10.0, 1000)
    candidate = _record("candidate", 1, 9.0, 800)
    candidate["dynamo"][metric] = 1

    comparison = benchmark.aggregate_records(
        [baseline, candidate], "baseline", "candidate"
    )

    assert comparison["failures"] == [candidate]
    assert comparison["comparisons"] == []
    assert comparison["performance_gates"]["met"] is False


@pytest.mark.parametrize("family", ["if", "lif", "plif"])
def test_atan_override_reaches_every_neuron(family):
    from spikingjelly.activation_based import neuron, surrogate

    model = benchmark._build_model("spikformer_ti", 4, 10, family, "ATan")
    nodes = [m for m in model.modules() if isinstance(m, neuron.BaseNode)]
    assert len(nodes) > 0
    for node in nodes:
        assert type(node.surrogate_function) is surrogate.ATan
        assert node.surrogate_function.alpha == 2.0


def test_aggregate_records_separates_surrogates():
    baseline = _record("baseline", 1, 10.0, 100)
    candidate = _record("candidate", 1, 9.0, 100)
    baseline["case"]["surrogate"] = "Sigmoid"
    candidate["case"]["surrogate"] = "ATan"
    result = benchmark.aggregate_records([baseline, candidate], "baseline", "candidate")
    assert not result["comparisons"]
