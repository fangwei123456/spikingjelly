from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def _positive_float(row: Mapping[str, Any], key: str) -> float:
    if key not in row:
        raise ValueError(f"{key} is required.")
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{key} must be a finite positive number, but got {row[key]!r}."
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{key} must be a finite positive number, but got {row[key]!r}."
        )
    return value


def assess_model_efficiency(
    results: Iterable[Mapping[str, Any]],
    *,
    baseline_precision: str,
    min_training_speedup: float,
    min_inference_speedup: float,
) -> dict[str, Any]:
    results = list(results)
    if (
        not math.isfinite(min_training_speedup)
        or not math.isfinite(min_inference_speedup)
        or min_training_speedup <= 0
        or min_inference_speedup <= 0
    ):
        raise ValueError("Minimum speedups must be finite and positive.")
    if baseline_precision.startswith("fp8-"):
        raise ValueError(
            "baseline_precision must not be an FP8 variant, but got "
            f"{baseline_precision!r}."
        )
    baselines = [row for row in results if row["precision"] == baseline_precision]
    if len(baselines) != 1:
        raise ValueError(
            f"Expected exactly one {baseline_precision!r} baseline, "
            f"but found {len(baselines)}."
        )
    candidates = [row for row in results if str(row["precision"]).startswith("fp8-")]
    if not candidates:
        raise ValueError("At least one FP8 precision result is required.")

    baseline = baselines[0]
    baseline_training = _positive_float(baseline, "training_samples_per_sec")
    baseline_inference = _positive_float(baseline, "inference_samples_per_sec")
    baseline_training_memory = _positive_float(baseline, "training_peak_allocated_mb")
    baseline_inference_memory = _positive_float(baseline, "inference_peak_allocated_mb")
    comparisons = []
    failures = []

    for candidate in candidates:
        precision = str(candidate["precision"])
        training_speedup = (
            _positive_float(candidate, "training_samples_per_sec") / baseline_training
        )
        inference_speedup = (
            _positive_float(candidate, "inference_samples_per_sec") / baseline_inference
        )
        training_memory_ratio = (
            _positive_float(candidate, "training_peak_allocated_mb")
            / baseline_training_memory
        )
        inference_memory_ratio = (
            _positive_float(candidate, "inference_peak_allocated_mb")
            / baseline_inference_memory
        )
        training_passed = training_speedup >= min_training_speedup
        inference_passed = inference_speedup >= min_inference_speedup
        if not training_passed:
            failures.append(
                f"{precision} training speedup {training_speedup:.4f}x "
                f"< {min_training_speedup:.4f}x"
            )
        if not inference_passed:
            failures.append(
                f"{precision} inference speedup {inference_speedup:.4f}x "
                f"< {min_inference_speedup:.4f}x"
            )
        comparisons.append(
            {
                "precision": precision,
                "training_speedup": training_speedup,
                "inference_speedup": inference_speedup,
                "training_memory_ratio": training_memory_ratio,
                "inference_memory_ratio": inference_memory_ratio,
                "training_passed": training_passed,
                "inference_passed": inference_passed,
                "passed": training_passed and inference_passed,
            }
        )

    return {
        "baseline_precision": baseline_precision,
        "min_training_speedup": min_training_speedup,
        "min_inference_speedup": min_inference_speedup,
        "comparisons": comparisons,
        "failures": failures,
        "passed": not failures,
    }


def assess_triton_efficiency(
    rows: Iterable[Mapping[str, Any]], *, min_speedup: float
) -> dict[str, Any]:
    if not math.isfinite(min_speedup) or min_speedup <= 0:
        raise ValueError("min_speedup must be finite and positive.")
    rows = list(rows)
    group_fields = ("T", "N", "neuron_type", "process")
    all_baselines = [row for row in rows if row.get("variant") == "stable_fp32"]
    baselines = [row for row in all_baselines if row.get("success") is True]
    comparisons = []
    failures = []

    if not all_baselines:
        failures.append("no usable stable_fp32 baseline results")

    for baseline in all_baselines:
        if baseline.get("success") is True:
            continue
        group = tuple(baseline.get(field) for field in group_fields)
        label = f"T={group[0]} N={group[1]} neuron={group[2]} process={group[3]}"
        reason = baseline.get("failure_reason") or "unknown failure"
        failures.append(f"{label} stable_fp32 baseline failed: {reason}")

    seen_groups = set()
    for baseline in baselines:
        group = tuple(baseline.get(field) for field in group_fields)
        label = f"T={group[0]} N={group[1]} neuron={group[2]} process={group[3]}"
        missing_fields = [
            field
            for field, value in zip(group_fields, group, strict=True)
            if value is None
        ]
        if missing_fields:
            failures.append(
                f"{label} stable_fp32 baseline is missing required group fields: "
                + ", ".join(missing_fields)
            )
            continue
        if group in seen_groups:
            failures.append(f"{label} has duplicate successful stable_fp32 baselines")
            continue
        seen_groups.add(group)
        candidates = [
            row
            for row in rows
            if row.get("success") is True
            and str(row.get("variant", "")).startswith("mp_plan_float8_")
            and tuple(row.get(field) for field in group_fields) == group
        ]
        if not candidates:
            failures.append(f"{label} has no successful FP8 prepared-plan result")
            continue

        best = min(candidates, key=lambda row: _positive_float(row, "median_ms"))
        baseline_ms = _positive_float(baseline, "median_ms")
        best_ms = _positive_float(best, "median_ms")
        speedup = baseline_ms / best_ms
        baseline_memory = _positive_float(baseline, "peak_allocated_mb")
        best_memory = _positive_float(best, "peak_allocated_mb")
        passed = speedup >= min_speedup
        if not passed:
            failures.append(
                f"{label} best FP8 speedup {speedup:.4f}x < {min_speedup:.4f}x"
            )
        comparisons.append(
            {
                "T": group[0],
                "N": group[1],
                "neuron_type": group[2],
                "process": group[3],
                "baseline_variant": "stable_fp32",
                "best_fp8_variant": best["variant"],
                "best_compute_dtype": best.get("compute_dtype"),
                "speedup": speedup,
                "memory_ratio": best_memory / baseline_memory,
                "passed": passed,
            }
        )

    return {
        "baseline_variant": "stable_fp32",
        "min_speedup": min_speedup,
        "comparisons": comparisons,
        "failures": failures,
        "passed": not failures,
    }


def require_efficiency(report: Mapping[str, Any]) -> None:
    if report.get("passed") is True:
        return
    failures = report.get("failures") or ["efficiency requirements were not met"]
    raise RuntimeError("FP8 efficiency validation failed: " + "; ".join(failures))
