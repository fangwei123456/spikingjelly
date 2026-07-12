"""
Analyze and compare distributed SNN benchmark results.

Reads JSONL records produced by ``benchmark_snn_distributed.py`` and provides
grouping, aggregation, and comparison utilities.

Examples::

    # Show all records grouped by mode
    python benchmark/analyze_distributed_benchmark.py --group-by mode

    # Compare modes for a specific model
    python benchmark/analyze_distributed_benchmark.py --filter model=cifar10dvs_vgg --group-by mode

    # Compare compiled vs eager
    python benchmark/analyze_distributed_benchmark.py --filter model=cifar10dvs_vgg,mode=fsdp2_tp --group-by compile_enabled

    # Export as CSV
    python benchmark/analyze_distributed_benchmark.py --group-by mode --format csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_DEFAULT_METRICS = (
    "step_latency_ms",
    "global_throughput_sps",
    "per_device_throughput_sps",
    "peak_allocated_mb",
    "forward_ms",
    "backward_ms",
    "optimizer_ms",
    "reset_ms",
    "materialize_ms",
    "tp_all_reduce_calls",
    "tp_all_reduce_mb",
    "warning_count",
    "recompile_count",
    "graph_break_count",
)


def load_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def filter_records(
    records: List[Dict[str, Any]], filters: Dict[str, str]
) -> List[Dict[str, Any]]:
    if not filters:
        return records
    result = []
    for record in records:
        match = True
        for key, value in filters.items():
            record_value = record.get(key)
            if record_value is None:
                match = False
                break
            if str(record_value) != value:
                match = False
                break
        if match:
            result.append(record)
    return result


def _try_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def group_and_aggregate(
    records: List[Dict[str, Any]],
    group_keys: Sequence[str],
    metrics: Sequence[str] = _DEFAULT_METRICS,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_key = tuple(str(record.get(k, "")) for k in group_keys)
        label = "|".join(group_key) if len(group_key) > 1 else group_key[0]
        groups[label].append(record)

    result = {}
    for label, group_records in sorted(groups.items()):
        metric_stats = {}
        for metric in metrics:
            values = [
                v for r in group_records if (v := _try_float(r.get(metric))) is not None
            ]
            if not values:
                continue
            stats = {
                "count": len(values),
                "mean": statistics.mean(values),
            }
            if len(values) >= 2:
                stats["stdev"] = statistics.stdev(values)
            stats["min"] = min(values)
            stats["max"] = max(values)
            metric_stats[metric] = stats
        result[label] = metric_stats
    return result


def format_table(
    groups: Dict[str, Dict[str, Dict[str, float]]],
    metrics: Sequence[str] = _DEFAULT_METRICS,
) -> str:
    if not groups:
        return "(no records)"

    lines: List[str] = []
    group_labels = list(groups.keys())
    available_metrics = []
    for m in metrics:
        if any(m in groups[g] for g in group_labels):
            available_metrics.append(m)

    header = f"{'Metric':<35}"
    for label in group_labels:
        header += f"  {label:>14}"
    lines.append(header)
    lines.append("-" * len(header))

    for metric in available_metrics:
        row = f"{metric:<35}"
        for label in group_labels:
            stats = groups[label].get(metric)
            if stats is None:
                row += f"  {'--':>14}"
            else:
                mean = stats["mean"]
                stdev = stats.get("stdev")
                if stdev is not None and stats["count"] >= 2:
                    row += f"  {mean:>8.2f} +/-{stdev:>4.1f}"
                else:
                    row += f"  {mean:>14.2f}"
        lines.append(row)

    return "\n".join(lines)


def format_csv(
    groups: Dict[str, Dict[str, Dict[str, float]]],
    metrics: Sequence[str] = _DEFAULT_METRICS,
) -> str:
    if not groups:
        return ""

    group_labels = list(groups.keys())
    available_metrics = [
        m for m in metrics if any(m in groups[g] for g in group_labels)
    ]

    output = io.StringIO()
    writer = csv.writer(output)
    header = ["metric"]
    for label in group_labels:
        header.extend([f"{label}_mean", f"{label}_stdev"])
    writer.writerow(header)

    for metric in available_metrics:
        row = [metric]
        for label in group_labels:
            stats = groups[label].get(metric)
            if stats is None:
                row.extend(["", ""])
            else:
                row.append(f"{stats['mean']:.6f}")
                row.append(f"{stats.get('stdev', 0.0):.6f}")
        writer.writerow(row)

    return output.getvalue()


def format_json(
    groups: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    return json.dumps(groups, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze distributed SNN benchmark results."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path("benchmark") / "results" / "benchmark_snn_distributed.jsonl"),
        help="Path to JSONL benchmark results file.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="*",
        default=[],
        help="Filter records by key=value pairs (e.g. --filter model=cifar10dvs_vgg mode=tp).",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        nargs="+",
        default=["mode"],
        help="Group records by these keys (e.g. --group-by mode compile_enabled).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Specific metrics to show (default: all standard metrics).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="table",
        choices=("table", "csv", "json"),
        help="Output format.",
    )
    args = parser.parse_args()

    filters: Dict[str, str] = {}
    for f in args.filter:
        if "=" not in f:
            parser.error(f"Filter must be key=value, got: {f}")
        k, v = f.split("=", 1)
        filters[k.strip()] = v.strip()

    records = load_records(args.input)
    if not records:
        print(f"No records found in {args.input}", file=sys.stderr)
        return

    filtered = filter_records(records, filters)
    if not filtered:
        print(
            f"No records match filters: {filters}",
            file=sys.stderr,
        )
        return

    metrics = args.metrics if args.metrics else _DEFAULT_METRICS
    groups = group_and_aggregate(filtered, args.group_by, metrics)

    if args.format == "table":
        print(f"Total records: {len(filtered)}")
        print(f"Grouped by: {', '.join(args.group_by)}")
        if filters:
            print(f"Filtered: {filters}")
        print()
        print(format_table(groups, metrics))
    elif args.format == "csv":
        print(format_csv(groups, metrics))
    elif args.format == "json":
        print(format_json(groups))


if __name__ == "__main__":
    main()
