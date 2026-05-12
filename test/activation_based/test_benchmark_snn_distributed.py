import importlib.util
import os
import sys
from pathlib import Path

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
        "model": "cifar10dvs_vgg",
        "mode": "fsdp2_tp",
        "backend": "inductor",
        "world_size": 4,
        "optimizer_sharding": "none",
        "memopt_level": 1,
        "batch_size": 8,
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
        "model": "cifar10dvs_vgg",
        "mode": "fsdp2_tp",
        "backend": "inductor",
        "world_size": 4,
        "optimizer_sharding": "none",
        "memopt_level": 1,
        "batch_size": 8,
        "T": 4,
        "steps": 3,
        "warmup": 1,
        "image_size": 224,
        "mesh_shape": (2, 2),
        "pp_microbatches": None,
        "pp_memopt_stage_budget_ratio": 0.5,
        "prefer": "memory",
        "global_samples_per_second": 100.0,
        "peak_allocated_mb": 512.0,
        "optimize_ms": 42.0,
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
        "global_samples_per_second": 120.0,
        "peak_allocated_mb": 400.0,
        "optimize_ms": 50.0,
        "warning_count": 2,
        "recompile_count": 1,
        "graph_break_count": 0,
    }
    previous = {
        "global_samples_per_second": 100.0,
        "peak_allocated_mb": 500.0,
        "optimize_ms": 40.0,
        "warning_count": 4,
        "recompile_count": 2,
        "graph_break_count": 0,
    }
    summary = bench._summarize_benchmark_comparison(current, previous)
    assert summary["global_samples_per_second"]["delta"] == 20.0
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


def test_parse_args_rejects_non_positive_steps(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_snn_distributed.py", "--steps", "0"])
    with pytest.raises(SystemExit):
        bench.parse_args()
