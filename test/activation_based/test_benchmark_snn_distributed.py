import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2] / "benchmark" / "benchmark_snn_distributed.py"
)
_BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "benchmark_snn_distributed", _BENCHMARK_PATH
)
bench = importlib.util.module_from_spec(_BENCHMARK_SPEC)
assert _BENCHMARK_SPEC.loader is not None
_BENCHMARK_SPEC.loader.exec_module(bench)


def test_comparison_key_normalizes_tuple_values():
    record = {
        "benchmark_regime": "throughput_weak_scaling",
        "model": "cifar10dvs_vgg",
        "mode": "fsdp2_tp",
        "backend": "inductor",
        "world_size": 4,
        "optimizer_sharding": "none",
        "memopt_level": 1,
        "batch_size": 8,
        "global_batch_size": 32,
        "per_rank_batch_size": 8,
        "T": 4,
        "steps": 3,
        "image_size": 224,
        "mesh_shape": (2, 2),
        "pp_microbatches": None,
        "pp_memopt_stage_budget_ratio": 0.5,
        "prefer": "memory",
    }
    key = bench._comparison_key(record)
    assert key["mesh_shape"] == [2, 2]


def test_append_benchmark_record_finds_previous_matching_entry(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    record = {
        "timestamp": "2026-04-29T00:00:00+00:00",
        "benchmark_regime": "throughput_weak_scaling",
        "model": "cifar10dvs_vgg",
        "mode": "fsdp2_tp",
        "backend": "inductor",
        "world_size": 4,
        "optimizer_sharding": "none",
        "memopt_level": 1,
        "batch_size": 8,
        "global_batch_size": 32,
        "per_rank_batch_size": 8,
        "T": 4,
        "steps": 3,
        "warmup": 1,
        "image_size": 224,
        "mesh_shape": (2, 2),
        "pp_microbatches": None,
        "pp_memopt_stage_budget_ratio": 0.5,
        "prefer": "memory",
        "step_latency_ms": 12.0,
        "global_throughput_sps": 100.0,
        "per_device_throughput_sps": 25.0,
        "peak_allocated_mb": 512.0,
        "optimize_ms": 42.0,
        "forward_ms": 4.0,
        "backward_ms": 5.0,
        "optimizer_ms": 1.0,
        "reset_ms": 0.5,
        "materialize_ms": 0.3,
        "tp_all_reduce_calls": 8,
        "tp_all_reduce_mb": 3.5,
        "warning_count": 1,
        "recompile_count": 0,
        "graph_break_count": 0,
    }
    previous = bench._append_benchmark_record(str(path), dict(record))
    assert previous is None
    second_previous = bench._append_benchmark_record(str(path), dict(record))
    assert second_previous is not None
    assert second_previous["comparison_key"]["mesh_shape"] == [2, 2]


def test_summarize_benchmark_comparison_reports_pct_deltas():
    current = {
        "step_latency_ms": 10.0,
        "global_throughput_sps": 120.0,
        "per_device_throughput_sps": 30.0,
        "peak_allocated_mb": 400.0,
        "optimize_ms": 50.0,
        "forward_ms": 3.0,
        "backward_ms": 4.0,
        "optimizer_ms": 1.0,
        "reset_ms": 0.4,
        "materialize_ms": 0.2,
        "tp_all_reduce_calls": 10,
        "tp_all_reduce_mb": 4.0,
        "warning_count": 2,
        "recompile_count": 1,
        "graph_break_count": 0,
    }
    previous = {
        "step_latency_ms": 12.0,
        "global_throughput_sps": 100.0,
        "per_device_throughput_sps": 25.0,
        "peak_allocated_mb": 500.0,
        "optimize_ms": 40.0,
        "forward_ms": 4.0,
        "backward_ms": 5.0,
        "optimizer_ms": 1.0,
        "reset_ms": 0.5,
        "materialize_ms": 0.3,
        "tp_all_reduce_calls": 8,
        "tp_all_reduce_mb": 3.5,
        "warning_count": 4,
        "recompile_count": 2,
        "graph_break_count": 0,
    }
    summary = bench._summarize_benchmark_comparison(current, previous)
    assert summary["global_throughput_sps"]["delta"] == 20.0
    assert summary["peak_allocated_mb"]["delta"] == -100.0
    assert summary["warning_count"]["pct"] == -50.0


def test_line_pattern_counter_tracks_torch_warning_prefix():
    counter = bench._LinePatternCounter()
    counter.feed("W0429 17:23:09.278000 some torch warning\n")
    counter.feed("recompile because shape changed\n")
    counter.feed("graph break detected\n")
    counter.finalize()
    assert counter.warning_count == 1
    assert counter.recompile_count == 1
    assert counter.graph_break_count == 1


def test_capture_benchmark_events_counts_fd_level_warnings():
    with bench.capture_benchmark_events() as counter:
        os.write(2, b"W9999 native warning that bypasses sys.stderr\n")
    assert counter.warning_count == 1


def test_reduce_classification_output_keeps_batch_major_logits():
    logits = torch.randn(4, 10)
    target = torch.tensor([0, 1, 2, 3])
    reduced_logits, reduced_target = bench._reduce_classification_output(logits, target)
    assert reduced_logits.shape == logits.shape
    assert reduced_target.shape == target.shape


def test_reduce_classification_output_preserves_singleton_index_targets():
    logits = torch.randn(4, 10)
    target = torch.tensor([[0], [1], [2], [3]])
    _, reduced_target = bench._reduce_classification_output(logits, target)
    assert torch.equal(reduced_target, torch.tensor([0, 1, 2, 3]))


def test_resolve_benchmark_batch_semantics_weak_scaling():
    global_batch, per_rank_batch = bench._resolve_benchmark_batch_semantics(
        batch_size=8,
        data_replicas=4,
        benchmark_regime="throughput_weak_scaling",
    )
    assert global_batch == 32
    assert per_rank_batch == 8


def test_resolve_benchmark_batch_semantics_strong_scaling():
    global_batch, per_rank_batch = bench._resolve_benchmark_batch_semantics(
        batch_size=8,
        data_replicas=4,
        benchmark_regime="latency_strong_scaling",
    )
    assert global_batch == 8
    assert per_rank_batch == 2


def test_resolve_benchmark_batch_semantics_memory_capacity():
    global_batch, per_rank_batch = bench._resolve_benchmark_batch_semantics(
        batch_size=8,
        data_replicas=4,
        benchmark_regime="memory_capacity",
    )
    assert global_batch == 8
    assert per_rank_batch == 2


def test_resolve_benchmark_batch_semantics_rejects_non_divisible_strong_scaling():
    with pytest.raises(ValueError, match="must be divisible by data_replicas=4"):
        bench._resolve_benchmark_batch_semantics(
            batch_size=10,
            data_replicas=4,
            benchmark_regime="latency_strong_scaling",
        )


def test_throughput_from_regime_matches_semantics():
    weak_global, weak_per_device = bench._throughput_from_regime(
        benchmark_regime="throughput_weak_scaling",
        elapsed=2.0,
        steps=10,
        world_size=4,
        global_batch_size=32,
        per_rank_batch_size=8,
    )
    assert weak_global == 160.0
    assert weak_per_device == 40.0

    strong_global, strong_per_device = bench._throughput_from_regime(
        benchmark_regime="latency_strong_scaling",
        elapsed=2.0,
        steps=10,
        world_size=4,
        global_batch_size=8,
        per_rank_batch_size=2,
    )
    assert strong_global == 40.0
    assert strong_per_device == 10.0

    capacity_global, capacity_per_device = bench._throughput_from_regime(
        benchmark_regime="memory_capacity",
        elapsed=2.0,
        steps=10,
        world_size=4,
        global_batch_size=8,
        per_rank_batch_size=2,
    )
    assert capacity_global == 40.0
    assert capacity_per_device == 10.0


def test_throughput_from_regime_uses_world_size_for_per_device_metric():
    global_tp, per_device_tp = bench._throughput_from_regime(
        benchmark_regime="throughput_weak_scaling",
        elapsed=2.0,
        steps=10,
        world_size=8,
        global_batch_size=16,
        per_rank_batch_size=16,
    )
    assert global_tp == 80.0
    assert per_device_tp == 10.0


def test_aggregate_tp_debug_stats_returns_local_totals_without_process_group(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(bench.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        bench,
        "get_tp_communication_debug_stats",
        lambda: {"all_reduce_calls": 1, "all_reduce_bytes": 16},
    )
    stats = bench._aggregate_tp_debug_stats(torch.device("cpu"))
    assert stats["all_reduce_calls"] == 1
    assert stats["all_reduce_bytes"] == 16


def test_build_model_fsdp2_disables_auto_tensor_parallel(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def _fake_configure(model, config):
        captured["config"] = config
        return model, None, None

    monkeypatch.setattr(bench, "configure_snn_distributed", _fake_configure)
    args = SimpleNamespace(
        model="cifar10dvs_vgg",
        mode="fsdp2",
        memopt_level=0,
        backend="torch",
        T=4,
        image_size=224,
        num_classes=1000,
        mesh_shape=None,
        tp_mesh_dim=0,
        dp_mesh_dim=None,
        pp_microbatches=None,
        pp_schedule="auto",
        pp_virtual_stages=1,
        pp_layout=None,
        pp_delay_wgrad=False,
        memopt_compress_x=False,
    )
    bench.build_model(args, torch.device("cpu"), 1, 2)
    assert captured["config"].auto_tensor_parallel is False


def test_make_synthetic_batch_uses_requested_batch_size():
    args = type(
        "Args",
        (),
        {
            "model": "cifar10dvs_vgg",
            "T": 4,
            "image_size": 224,
            "num_classes": 1000,
        },
    )()
    x, y = bench._make_synthetic_batch(
        args,
        torch.device("cpu"),
        2,
    )
    assert x.shape[0] == 2
    assert y.shape[0] == 2


def test_time_block_synchronizes_cuda_device(monkeypatch: pytest.MonkeyPatch):
    calls = {"count": 0}

    def _fake_sync(device):
        calls["count"] += 1

    monkeypatch.setattr(bench, "_synchronize_timing_device", _fake_sync)
    result, elapsed_ms = bench._time_block(torch.device("cpu"), lambda: 7)
    assert result == 7
    assert elapsed_ms >= 0.0
    assert calls["count"] == 2


def test_benchmark_step_pipeline_uses_cached_reset_modules(
    monkeypatch: pytest.MonkeyPatch,
):
    class _Schedule:
        def step(self, *args, **kwargs):
            return None

    runtime = SimpleNamespace(
        is_first=True,
        is_last=True,
        schedule=_Schedule(),
        stage_module=object(),
    )
    calls = {"reset_net": 0, "reset_collected": 0}
    monkeypatch.setattr(
        bench.functional,
        "reset_net",
        lambda module: calls.__setitem__("reset_net", calls["reset_net"] + 1),
    )
    monkeypatch.setattr(
        bench.functional,
        "reset_collected_modules",
        lambda modules: calls.__setitem__(
            "reset_collected", calls["reset_collected"] + 1
        ),
    )
    optimizer = SimpleNamespace(
        zero_grad=lambda set_to_none=True: None,
        step=lambda: None,
    )

    bench._benchmark_step_pipeline(
        runtime,
        optimizer,
        torch.randn(1),
        torch.tensor([0]),
        torch.device("cpu"),
        reset_modules=(object(),),
    )

    assert calls == {"reset_net": 0, "reset_collected": 1}


def test_parse_args_rejects_non_positive_steps(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_snn_distributed.py", "--steps", "0"])
    with pytest.raises(SystemExit):
        bench.parse_args()
