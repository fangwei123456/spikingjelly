from contextlib import nullcontext
from datetime import timedelta
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from benchmark.snn_llm.qwen_conversion import quality_eval as runner
from spikingjelly.activation_based.ann2snn import Qwen2SNNCalibration


def test_distributed_task_timeout_allows_imbalanced_long_tasks():
    assert runner.DISTRIBUTED_TASK_TIMEOUT >= timedelta(hours=6)


def test_quality_metric_extraction_uses_declared_primary_metric():
    results = {
        "results": {
            "piqa": {
                "acc,none": 0.5,
                "acc_norm,none": 0.625,
            }
        }
    }

    assert runner._metric_value(results, "piqa", "acc_norm") == 0.625


def test_task_adapters_are_configured_after_both_are_constructed():
    class FakeAdapter:
        def __init__(self):
            self.distributed_enabled = False
            self.reset_before_call = False

        def enable_distributed_requests(self):
            self.distributed_enabled = True

    dense = FakeAdapter()
    snn = FakeAdapter()

    runner._configure_task_adapters(dense, snn)

    assert dense.distributed_enabled is True
    assert snn.distributed_enabled is True
    assert dense.reset_before_call is False
    assert snn.reset_before_call is True


def test_only_distributed_rank_zero_builds_task_report(monkeypatch):
    monkeypatch.setattr(runner.torch_dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runner.torch_dist, "get_rank", lambda: 1)
    assert runner._is_task_report_rank() is False

    monkeypatch.setattr(runner.torch_dist, "get_rank", lambda: 0)
    assert runner._is_task_report_rank() is True


def test_lm_eval_version_is_fail_closed(monkeypatch):
    monkeypatch.setattr(runner.metadata, "version", lambda _name: "0.4.11")
    with pytest.raises(RuntimeError, match="requires lm-eval==0.4.12"):
        runner._require_lm_eval_version()


def test_quality_acceptance_enforces_ppl_mean_and_per_task_gates():
    report = {
        "quality": {
            "wikitext": {"relative_degradation": 0.10},
            "zero_shot": {
                "mean_drop_percentage_points": 2.0,
                "max_drop_percentage_points": 5.0,
            },
        }
    }
    runner._validate_quality(report)

    report["quality"]["zero_shot"]["max_drop_percentage_points"] = 7.1
    with pytest.raises(ValueError, match="task accuracy drop"):
        runner._validate_quality(report)


def test_quality_rejects_tasks_on_validation_before_cuda(tmp_path, capsys):
    exit_code = runner.main(
        [
            "--model-key",
            "0.5b",
            "--model-root",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
            "--hf-cache",
            str(tmp_path / "cache"),
            "--device",
            "cuda",
            "--worktree-revision",
            "revision",
            "--time-steps",
            "32",
            "--calibration-levels",
            "16",
            "--calibration-quantile",
            "1.0",
            "--wikitext-split",
            "validation",
            "--run-tasks",
        ]
    )

    assert exit_code == 2
    assert "only allowed with the test split" in capsys.readouterr().err


@pytest.mark.parametrize("batch_size", ("0", "-1"))
def test_quality_rejects_nonpositive_task_batch_before_cuda(
    tmp_path, capsys, batch_size
):
    exit_code = runner.main(
        [
            "--model-key",
            "0.5b",
            "--model-root",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
            "--hf-cache",
            str(tmp_path / "cache"),
            "--device",
            "cuda",
            "--worktree-revision",
            "revision",
            "--time-steps",
            "32",
            "--calibration-levels",
            "16",
            "--calibration-quantile",
            "1.0",
            "--wikitext-split",
            "test",
            "--task-batch-size",
            batch_size,
        ]
    )

    assert exit_code == 2
    assert "task-batch-size must be positive" in capsys.readouterr().err


def test_distributed_quality_is_task_only_and_requires_gseries_guard(monkeypatch):
    args = SimpleNamespace(
        skip_ppl=False,
        run_tasks=True,
        calibration_artifact=object(),
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")

    with pytest.raises(ValueError, match="task-only"):
        runner._validate_distributed_task_args(args)

    args.skip_ppl = True
    args.calibration_artifact = None
    with pytest.raises(ValueError, match="calibration-artifact"):
        runner._validate_distributed_task_args(args)

    args.calibration_artifact = object()
    monkeypatch.delenv("NCCL_P2P_DISABLE")
    with pytest.raises(ValueError, match="NCCL_P2P_DISABLE=1"):
        runner._validate_distributed_task_args(args)

    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    assert runner._validate_distributed_task_args(args) == 2

    monkeypatch.setenv("WORLD_SIZE", "0")
    with pytest.raises(ValueError, match="WORLD_SIZE must be positive"):
        runner._validate_distributed_task_args(args)


def test_load_calibration_requires_matching_configuration(tmp_path):
    calibration = Qwen2SNNCalibration(
        input_scale=torch.ones(2),
        layer_scales=(
            {
                "query": torch.ones(2),
                "key": torch.ones(2),
                "value": torch.ones(2),
                "mlp": torch.ones(4),
            },
        ),
        time_steps=32,
        calibration_levels=16,
        calibration_quantile=1.0,
        calibration_reservoir_size=4096,
        calibration_seed=20260719,
        valid_token_count=512,
    )
    path = tmp_path / "calibration.pt"
    torch.save(calibration.state_dict(), path)

    loaded, digest = runner._load_calibration(path)
    runner._validate_calibration_config(
        loaded,
        time_steps=32,
        calibration_levels=16,
        calibration_quantile=1.0,
        calibration_reservoir_size=4096,
        calibration_seed=20260719,
    )

    assert loaded.time_steps == 32
    assert len(digest) == 64
    with pytest.raises(ValueError, match="time_steps"):
        runner._validate_calibration_config(
            loaded,
            time_steps=64,
            calibration_levels=16,
            calibration_quantile=1.0,
            calibration_reservoir_size=4096,
            calibration_seed=20260719,
        )
    with pytest.raises(ValueError, match="calibration_seed"):
        runner._validate_calibration_config(
            loaded,
            time_steps=32,
            calibration_levels=16,
            calibration_quantile=1.0,
            calibration_reservoir_size=4096,
            calibration_seed=0,
        )


def test_save_calibration_removes_temporary_file_on_failure(monkeypatch, tmp_path):
    target = tmp_path / "calibration.pt"
    calibration = SimpleNamespace(state_dict=lambda: {"scale": torch.ones(1)})

    def fail_after_write(_state, path):
        path.write_bytes(b"partial")
        raise OSError("save failed")

    monkeypatch.setattr(runner.torch, "save", fail_after_write)

    with pytest.raises(OSError, match="save failed"):
        runner._save_calibration(target, calibration)
    assert not target.exists()
    assert not (tmp_path / ".calibration.pt.tmp").exists()


def test_full_evaluation_requires_full_test_ppl_and_all_tasks():
    assert runner._is_full_evaluation(
        split="test", max_ppl_windows=None, run_tasks=True, task_limit=None
    )
    assert not runner._is_full_evaluation(
        split="validation", max_ppl_windows=None, run_tasks=True, task_limit=None
    )
    assert not runner._is_full_evaluation(
        split="test", max_ppl_windows=1, run_tasks=True, task_limit=None
    )
    assert not runner._is_full_evaluation(
        split="test", max_ppl_windows=None, run_tasks=False, task_limit=None
    )
    assert not runner._is_full_evaluation(
        split="test",
        max_ppl_windows=None,
        run_tasks=True,
        task_limit=None,
        ppl_shard_count=2,
    )
    assert not runner._is_full_evaluation(
        split="test",
        max_ppl_windows=None,
        run_tasks=True,
        task_limit=None,
        tasks=("piqa",),
    )


def test_encoder_summary_reports_worst_layer_and_finite_rates():
    class FakeConverted:
        def encoder_statistics(self):
            return (
                {
                    "name": "input",
                    "local_relative_l2": 0.1,
                    "positive_spike_rate": 0.2,
                    "negative_spike_rate": 0.3,
                    "boundary_correction_fraction": 0.01,
                },
                {
                    "name": "layer.0.query",
                    "local_relative_l2": 0.4,
                    "positive_spike_rate": 0.4,
                    "negative_spike_rate": 0.5,
                    "boundary_correction_fraction": 0.02,
                },
            )

    summary = runner._encoder_summary(FakeConverted())

    assert summary["count"] == 2
    assert summary["worst_local_encoder"] == "layer.0.query"
    assert summary["max_local_relative_l2"] == 0.4
    assert summary["mean_positive_spike_rate"] == pytest.approx(0.3)


def test_statistics_probe_disables_collection_before_quality():
    class FakeConverted(nn.Module):
        def __init__(self):
            super().__init__()
            self.collection_states = []
            self.forward_collection_state = None
            self.collect_statistics = True

        def set_collect_statistics(self, enabled):
            self.collect_statistics = enabled
            self.collection_states.append(enabled)

        def forward(self, input_ids, attention_mask, encoding_mode):
            assert input_ids.shape == attention_mask.shape
            assert encoding_mode == "signed_if"
            self.forward_collection_state = self.collect_statistics
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 2))

        def encoder_statistics(self):
            return (
                {
                    "name": "probe",
                    "local_relative_l2": 0.1,
                    "positive_spike_rate": 0.2,
                    "negative_spike_rate": 0.3,
                    "boundary_correction_fraction": 0.0,
                },
            )

    def tokenizer(_text, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }

    converted = FakeConverted()
    summary = runner._capture_encoder_summary(
        converted,
        tokenizer,
        "cpu",
        nullcontext,
    )

    assert converted.forward_collection_state is True
    assert converted.collection_states == [True, False]
    assert converted.collect_statistics is False
    assert summary["worst_local_encoder"] == "probe"


def test_runner_sha_is_captured_at_module_import():
    assert len(runner.RUNNER_SHA256) == 64
    int(runner.RUNNER_SHA256, 16)


def test_chunked_nll_preserves_cross_chunk_targets(monkeypatch):
    class FakeCachedModel(nn.Module):
        def forward(
            self,
            input_ids,
            attention_mask,
            past_key_values=None,
            use_cache=False,
        ):
            del attention_mask
            assert use_cache
            start = 0 if past_key_values is None else past_key_values
            positions = torch.arange(start, start + input_ids.shape[1]).float()
            logits = torch.stack(
                tuple(torch.sin(positions + offset) for offset in range(7)), dim=-1
            ).unsqueeze(0)
            return SimpleNamespace(
                logits=logits,
                past_key_values=start + input_ids.shape[1],
            )

    monkeypatch.setattr(runner, "PPL_CACHE_CHUNK", 3)
    model = FakeCachedModel()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    full = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
    ).logits
    expected_nll, expected_count = runner._nll(full, labels)

    actual_nll, actual_count = runner._chunked_nll(
        model=model,
        input_ids=input_ids,
        labels=labels,
        autocast_context=nullcontext,
        encoding_mode=None,
    )

    assert actual_count == expected_count
    assert actual_nll == pytest.approx(expected_nll, rel=1e-6)


def test_rolling_ppl_shards_partition_windows_without_changing_nll(monkeypatch):
    monkeypatch.setattr(runner, "PPL_CONTEXT", 6)
    monkeypatch.setattr(runner, "PPL_STRIDE", 3)

    def fake_chunked_nll(**kwargs):
        labels = kwargs["labels"][:, 1:]
        valid = labels.ne(-100)
        return float(labels[valid].sum()), int(valid.sum())

    monkeypatch.setattr(runner, "_chunked_nll", fake_chunked_nll)
    tokens = torch.arange(1, 16)
    common = {
        "dense_model": object(),
        "snn_model": object(),
        "tokens": tokens,
        "device": "cpu",
        "autocast_context": nullcontext,
        "max_windows": None,
        "encoding_mode": "signed_if",
    }

    complete = runner._rolling_ppl(**common)
    shards = [
        runner._rolling_ppl(**common, shard_index=index, shard_count=2)
        for index in range(2)
    ]

    assert sum(value["dense_nll"] for value in shards) == complete["dense_nll"]
    assert sum(value["snn_nll"] for value in shards) == complete["snn_nll"]
    assert sum(value["token_count"] for value in shards) == complete["token_count"]
    indices = [index for value in shards for index in value["processed_window_indices"]]
    assert sorted(indices) == complete["processed_window_indices"]


def test_dataset_revision_wrapper_enforces_lock_and_records_fingerprint(monkeypatch):
    calls = []

    def fake_load_dataset(repository, **kwargs):
        calls.append((repository, kwargs))
        return {"test": SimpleNamespace(_fingerprint="fixed-fingerprint")}

    datasets = SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    lock = {
        "task": {
            "repository": "owner/dataset",
            "revision": "fixed-revision",
        }
    }
    with runner._pin_dataset_revisions(lock) as records:
        datasets.load_dataset("owner/dataset", split="test")

    assert calls[0][1]["revision"] == "fixed-revision"
    assert records == [
        {
            "repository": "owner/dataset",
            "revision": "fixed-revision",
            "fingerprints": {"test": "fixed-fingerprint"},
        }
    ]


def test_dataset_revision_wrapper_rejects_missing_fingerprint(monkeypatch):
    def fake_load_dataset(_repository, **_kwargs):
        return {"test": SimpleNamespace(_fingerprint=None)}

    datasets = SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    lock = {
        "task": {
            "repository": "owner/dataset",
            "revision": "fixed-revision",
        }
    }

    with pytest.raises(RuntimeError, match="did not expose stable fingerprints"):
        with runner._pin_dataset_revisions(lock):
            datasets.load_dataset("owner/dataset", split="test")
