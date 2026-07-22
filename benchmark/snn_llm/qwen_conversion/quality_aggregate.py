"""Aggregate complete, disjoint Qwen2 SNN paper-quality shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from benchmark.snn_llm._reporting import write_report as _write_report
from benchmark.snn_llm.qwen_conversion._quality import (
    TASKS,
    validate_quality as _validate_quality,
)


SCHEMA_VERSION = 1
CONTRACT_KIND = "qwen2-snn-paper-quality-aggregate"


def _load_report(path: Path) -> tuple[Dict[str, object], str]:
    payload = path.read_bytes()
    report = json.loads(payload)
    if not isinstance(report, dict):
        raise ValueError(f"Input {path} must contain a JSON object.")
    if report.get("kind") != "qwen2-snn-paper-quality":
        raise ValueError(f"Input {path} is not a Qwen paper-quality report.")
    return report, hashlib.sha256(payload).hexdigest()


def _perplexity(nll: float, token_count: int) -> float:
    if token_count <= 0 or not math.isfinite(nll) or nll < 0:
        raise ValueError(
            "Aggregated perplexity inputs must be finite and non-negative."
        )
    try:
        value = math.exp(nll / token_count)
    except OverflowError as exc:
        raise ValueError("Aggregated perplexity is not finite.") from exc
    if not math.isfinite(value):
        raise ValueError("Aggregated perplexity is not finite.")
    return value


def _precision_config_identity(config: Mapping[str, object]) -> Dict[str, object]:
    identity = dict(config)
    device = identity.get("device")
    if isinstance(device, str) and (device == "cuda" or device.startswith("cuda:")):
        identity["device"] = "cuda"
    return identity


def _identity(report: Mapping[str, object]) -> Dict[str, object]:
    config = report["configuration"]
    precision = report["precision"]
    return {
        "worktree_revision": report["source"]["worktree_revision"],
        "runner_sha256": report["source"]["runner_sha256"],
        "artifact_lock_sha256": report["source"]["artifact_lock_sha256"],
        "model": report["model"],
        "data": report["data"],
        "precision": {
            "requested_config": _precision_config_identity(
                precision["requested_config"]
            ),
            "effective_config": _precision_config_identity(
                precision["effective_config"]
            ),
            "policy": precision["policy"],
            "fallback_reason": precision["fallback_reason"],
        },
        "conversion": {
            name: report["conversion"][name]
            for name in (
                "temporal_layout",
                "execution_schedule",
                "online_inference",
                "calibration_sha256",
            )
        },
        "configuration": {
            name: config[name]
            for name in (
                "precision",
                "time_steps",
                "calibration_levels",
                "calibration_quantile",
                "calibration_reservoir_size",
                "calibration_seed",
                "wikitext_split",
                "evaluation_mode",
                "statistics_enabled_during_quality",
            )
        },
    }


def _aggregate_ppl(reports: List[Mapping[str, object]]) -> Dict[str, object]:
    ppl_reports = [
        report for report in reports if report["quality"]["wikitext"] is not None
    ]
    records = [report["quality"]["wikitext"] for report in ppl_reports]
    if not records:
        raise ValueError("Aggregate requires PPL shard reports.")
    shard_count = int(records[0]["shard_count"])
    if len(records) != shard_count:
        raise ValueError("PPL report count does not match shard_count.")
    if {int(record["shard_index"]) for record in records} != set(range(shard_count)):
        raise ValueError("PPL shard indices are incomplete or duplicated.")
    for report, record in zip(ppl_reports, records, strict=True):
        config = report["configuration"]
        if bool(config["skip_ppl"]):
            raise ValueError("PPL report cannot declare skip_ppl.")
        if int(config["ppl_shard_count"]) != shard_count or int(
            config["ppl_shard_index"]
        ) != int(record["shard_index"]):
            raise ValueError("PPL payload and shard configuration disagree.")
    global_counts = {int(record["global_window_count"]) for record in records}
    if len(global_counts) != 1:
        raise ValueError("PPL shards disagree on global window count.")
    indices = [
        int(index) for record in records for index in record["processed_window_indices"]
    ]
    global_count = global_counts.pop()
    if len(indices) != len(set(indices)) or set(indices) != set(range(global_count)):
        raise ValueError("PPL window coverage is incomplete or overlapping.")
    if any(
        report["configuration"]["max_ppl_windows"] is not None for report in reports
    ):
        raise ValueError("Formal PPL shards must not use max_ppl_windows.")
    evaluation_fields = ("context_length", "stride", "cache_chunk_length")
    for name in evaluation_fields:
        if len({record[name] for record in records}) != 1:
            raise ValueError(f"PPL shards disagree on {name}.")
    token_count = sum(int(record["token_count"]) for record in records)
    dense_nll = sum(float(record["dense_nll"]) for record in records)
    snn_nll = sum(float(record["snn_nll"]) for record in records)
    if token_count <= 0 or not all(
        math.isfinite(value) and value >= 0 for value in (dense_nll, snn_nll)
    ):
        raise ValueError(
            "Aggregated PPL inputs must be finite, non-negative, and non-empty."
        )
    dense_ppl = _perplexity(dense_nll, token_count)
    snn_ppl = _perplexity(snn_nll, token_count)
    relative_degradation = snn_ppl / dense_ppl - 1.0
    if not math.isfinite(relative_degradation):
        raise ValueError("Aggregated perplexity degradation is not finite.")
    return {
        "dense_ppl": dense_ppl,
        "snn_ppl": snn_ppl,
        "relative_degradation": relative_degradation,
        "dense_nll": dense_nll,
        "snn_nll": snn_nll,
        "token_count": token_count,
        "window_count": global_count,
        "shard_count": shard_count,
        "context_length": records[0]["context_length"],
        "stride": records[0]["stride"],
        "cache_chunk_length": records[0]["cache_chunk_length"],
    }


def _aggregate_tasks(reports: List[Mapping[str, object]]) -> Dict[str, object]:
    task_records: Dict[str, object] = {}
    task_batch_sizes: Dict[str, int] = {}
    task_world_sizes: Dict[str, int] = {}
    datasets = []
    versions = set()
    for report in reports:
        record = report["quality"]["zero_shot"]
        if record is None:
            continue
        if set(report["configuration"]["tasks"]) != set(record["tasks"]):
            raise ValueError("Task payload and task configuration disagree.")
        if record["limit"] is not None:
            raise ValueError("Formal task shards must not use a task limit.")
        versions.add(record["lm_eval_version"])
        batch_size = int(record["batch_size"])
        if batch_size <= 0:
            raise ValueError("Task report batch size must be positive.")
        if int(report["configuration"]["task_batch_size"]) != batch_size:
            raise ValueError("Task payload and configuration batch size disagree.")
        world_size = int(record["world_size"])
        if world_size <= 0:
            raise ValueError("Task report world size must be positive.")
        if int(report["configuration"]["task_world_size"]) != world_size:
            raise ValueError("Task payload and configuration world size disagree.")
        datasets.extend(record["datasets"])
        for name, metrics in record["tasks"].items():
            if name in task_records:
                raise ValueError(f"Zero-shot task {name!r} is duplicated.")
            task_records[name] = metrics
            task_batch_sizes[name] = batch_size
            task_world_sizes[name] = world_size
    if set(task_records) != set(TASKS):
        raise ValueError("Zero-shot task coverage is incomplete.")
    if len(versions) != 1:
        raise ValueError("Task shards disagree on lm-eval version.")
    drops = [float(value["drop_percentage_points"]) for value in task_records.values()]
    return {
        "lm_eval_version": versions.pop(),
        "num_fewshot": 0,
        "batch_sizes": task_batch_sizes,
        "world_sizes": task_world_sizes,
        "limit": None,
        "tasks": task_records,
        "mean_drop_percentage_points": sum(drops) / len(drops),
        "max_drop_percentage_points": max(drops),
        "snn_encoding_mode": "signed_if",
        "datasets": datasets,
    }


def _aggregate_loaded(
    paths: List[Path], reports: List[Dict[str, object]], digests: List[str]
) -> Dict[str, object]:
    identity = _identity(reports[0])
    if any(_identity(report) != identity for report in reports[1:]):
        raise ValueError(
            "Quality shard model, source, calibration, or configuration differs."
        )
    if identity["configuration"]["wikitext_split"] != "test":
        raise ValueError("Formal quality aggregation requires the WikiText test split.")
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "identity": identity,
        "source_reports": [
            {"path": str(path.resolve()), "sha256": digest}
            for path, digest in zip(paths, digests, strict=True)
        ],
        "quality": {
            "wikitext": _aggregate_ppl(reports),
            "zero_shot": _aggregate_tasks(reports),
        },
        "acceptance": {
            "ppl_gate_passed": True,
            "zero_shot_gate_passed": True,
            "formal_phase_gate_passed": True,
        },
    }
    _validate_quality(report)
    json.dumps(report, allow_nan=False)
    return report


def aggregate(paths: List[Path]) -> Dict[str, object]:
    if not paths:
        raise ValueError("At least one input report is required.")
    loaded = [_load_report(path) for path in paths]
    reports = [report for report, _digest in loaded]
    digests = [digest for _report, digest in loaded]
    try:
        return _aggregate_loaded(paths, reports, digests)
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            "Input report is missing or malformed required fields."
        ) from exc


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    try:
        args = parser.parse_args(argv)
        report = aggregate(args.input_report)
        target = _write_report(args.output_dir, report)
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"report_path={target.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
