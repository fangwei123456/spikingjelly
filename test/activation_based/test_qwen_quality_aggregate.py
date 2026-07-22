import json

import pytest

from benchmark.snn_llm.qwen_conversion import quality_aggregate as runner
from benchmark.snn_llm.qwen_conversion._quality import TASKS


def _base_report():
    return {
        "kind": "qwen2-snn-paper-quality",
        "source": {
            "worktree_revision": "rev",
            "runner_sha256": "runner",
            "artifact_lock_sha256": "lock",
        },
        "model": {"key": "0.5b", "files": {"model": "sha"}},
        "data": {
            "wikitext": {"repository": "Salesforce/wikitext", "revision": "data"},
            "lambada_openai": {"repository": "lambada", "revision": "data"},
            "piqa": {"repository": "piqa", "revision": "data"},
            "hellaswag": {"repository": "hellaswag", "revision": "data"},
            "winogrande": {"repository": "winogrande", "revision": "data"},
            "arc": {"repository": "arc", "revision": "data"},
        },
        "precision": {
            "requested_config": {"mode": "bf16", "device": "cuda"},
            "effective_config": {"mode": "bf16", "device": "cuda"},
            "policy": {"dtype": "torch.bfloat16"},
            "fallback_reason": None,
        },
        "conversion": {
            "temporal_layout": "[T,B,S,H]",
            "execution_schedule": "layerwise_offline_multistep",
            "online_inference": False,
            "calibration_sha256": "calibration",
        },
        "configuration": {
            "precision": "bf16",
            "time_steps": 160,
            "calibration_levels": 16,
            "calibration_quantile": 0.999,
            "calibration_reservoir_size": 4096,
            "calibration_seed": 20260719,
            "wikitext_split": "test",
            "evaluation_mode": "multistep_signed_if",
            "statistics_enabled_during_quality": False,
            "max_ppl_windows": None,
            "skip_ppl": False,
            "ppl_shard_index": 0,
            "ppl_shard_count": 1,
            "tasks": [],
            "task_batch_size": 1,
            "task_world_size": 1,
        },
        "quality": {"wikitext": None, "zero_shot": None},
    }


def _write_reports(tmp_path):
    paths = []
    for index, indices in enumerate(((0, 2), (1, 3))):
        report = _base_report()
        report["quality"]["wikitext"] = {
            "shard_index": index,
            "shard_count": 2,
            "global_window_count": 4,
            "processed_window_indices": list(indices),
            "token_count": 20,
            "dense_nll": 40.0,
            "snn_nll": 42.0,
            "context_length": 2048,
            "stride": 512,
            "cache_chunk_length": 128,
        }
        report["configuration"]["ppl_shard_index"] = index
        report["configuration"]["ppl_shard_count"] = 2
        path = tmp_path / f"ppl-{index}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    for index, names in enumerate((TASKS[:3], TASKS[3:])):
        report = _base_report()
        report["quality"]["zero_shot"] = {
            "lm_eval_version": "0.4.12",
            "num_fewshot": 0,
            "batch_size": 4,
            "world_size": 1,
            "limit": None,
            "tasks": {
                name: {
                    "metric": "acc",
                    "dense": 0.5,
                    "snn": 0.49,
                    "drop_percentage_points": 1.0,
                }
                for name in names
            },
            "datasets": [{"repository": f"dataset-{index}"}],
        }
        report["configuration"]["skip_ppl"] = True
        report["configuration"]["tasks"] = list(names)
        report["configuration"]["task_batch_size"] = 4
        report["precision"]["requested_config"]["device"] = "cuda:0"
        report["precision"]["effective_config"]["device"] = "cuda:0"
        path = tmp_path / f"tasks-{index}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    return paths


def test_quality_aggregate_requires_complete_disjoint_evidence(tmp_path):
    paths = _write_reports(tmp_path)

    report = runner.aggregate(paths)

    assert report["acceptance"]["formal_phase_gate_passed"] is True
    assert report["quality"]["wikitext"]["window_count"] == 4
    assert report["quality"]["wikitext"]["token_count"] == 40
    assert set(report["quality"]["zero_shot"]["tasks"]) == set(TASKS)
    assert set(report["quality"]["zero_shot"]["batch_sizes"].values()) == {4}
    assert set(report["quality"]["zero_shot"]["world_sizes"].values()) == {1}


def test_quality_aggregate_rejects_overlapping_ppl_windows(tmp_path):
    paths = _write_reports(tmp_path)
    report = json.loads(paths[1].read_text())
    report["quality"]["wikitext"]["processed_window_indices"] = [0, 3]
    paths[1].write_text(json.dumps(report))

    with pytest.raises(ValueError, match="incomplete or overlapping"):
        runner.aggregate(paths)


def test_quality_aggregate_rejects_mixed_calibration(tmp_path):
    paths = _write_reports(tmp_path)
    report = json.loads(paths[-1].read_text())
    report["conversion"]["calibration_sha256"] = "different"
    paths[-1].write_text(json.dumps(report))

    with pytest.raises(ValueError, match="calibration"):
        runner.aggregate(paths)


def test_quality_aggregate_rejects_mixed_dataset_lock(tmp_path):
    paths = _write_reports(tmp_path)
    report = json.loads(paths[-1].read_text())
    report["data"]["hellaswag"]["revision"] = "different"
    paths[-1].write_text(json.dumps(report))

    with pytest.raises(ValueError, match="configuration differs"):
        runner.aggregate(paths)


def test_quality_aggregate_rejects_mixed_device_types(tmp_path):
    paths = _write_reports(tmp_path)
    report = json.loads(paths[-1].read_text())
    report["precision"]["requested_config"]["device"] = "cpu"
    report["precision"]["effective_config"]["device"] = "cpu"
    paths[-1].write_text(json.dumps(report))

    with pytest.raises(ValueError, match="configuration differs"):
        runner.aggregate(paths)


def test_quality_aggregate_rejects_validation_split(tmp_path):
    paths = _write_reports(tmp_path)
    for path in paths:
        report = json.loads(path.read_text())
        report["configuration"]["wikitext_split"] = "validation"
        path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="test split"):
        runner.aggregate(paths)


def test_quality_aggregate_rejects_mixed_ppl_context(tmp_path):
    paths = _write_reports(tmp_path)
    report = json.loads(paths[1].read_text())
    report["quality"]["wikitext"]["context_length"] = 1024
    paths[1].write_text(json.dumps(report))

    with pytest.raises(ValueError, match="context_length"):
        runner.aggregate(paths)


def test_quality_aggregate_records_per_task_batch_size(tmp_path):
    paths = _write_reports(tmp_path)
    report = json.loads(paths[-1].read_text())
    report["quality"]["zero_shot"]["batch_size"] = 2
    report["configuration"]["task_batch_size"] = 2
    paths[-1].write_text(json.dumps(report))

    report = runner.aggregate(paths)

    for task in TASKS[:3]:
        assert report["quality"]["zero_shot"]["batch_sizes"][task] == 4
    for task in TASKS[3:]:
        assert report["quality"]["zero_shot"]["batch_sizes"][task] == 2


def test_quality_aggregate_rejects_task_batch_payload_config_mismatch(tmp_path):
    paths = _write_reports(tmp_path)
    report = json.loads(paths[-1].read_text())
    report["quality"]["zero_shot"]["batch_size"] = 2
    paths[-1].write_text(json.dumps(report))

    with pytest.raises(ValueError, match="payload and configuration"):
        runner.aggregate(paths)


def test_quality_aggregate_rejects_malformed_report_with_clear_error(tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps({"kind": "qwen2-snn-paper-quality"}))

    with pytest.raises(ValueError, match="missing or malformed"):
        runner.aggregate([path])


def test_quality_aggregate_rejects_perplexity_overflow(tmp_path):
    paths = _write_reports(tmp_path)
    for path in paths[:2]:
        report = json.loads(path.read_text())
        report["quality"]["wikitext"]["dense_nll"] = 100_000.0
        report["quality"]["wikitext"]["snn_nll"] = 100_000.0
        path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="perplexity is not finite"):
        runner.aggregate(paths)


def test_quality_aggregate_rejects_negative_nll(tmp_path):
    paths = _write_reports(tmp_path)
    for path in paths[:2]:
        report = json.loads(path.read_text())
        report["quality"]["wikitext"]["dense_nll"] = -1.0
        path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="non-negative"):
        runner.aggregate(paths)
