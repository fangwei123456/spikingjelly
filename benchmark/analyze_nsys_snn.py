"""Analyze bounded SNN Nsight Systems SQLite exports without a GPU."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

if __package__:
    from .plot_nsys_snn import render
else:
    from plot_nsys_snn import render


STEP = re.compile(r"^(?:sj\.step|benchmark_step):(?:(training|inference):)?(\d+)$")
PHASES = {
    "forward",
    "loss",
    "backward",
    "graph_runner",
    "optimizer",
    "zero_grad",
    "reset",
}
NS_PER_MS = 1_000_000


def _table_columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    end = 0
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def _pid(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def _category(name: str) -> str:
    lower = name.lower()
    if any(word in lower for word in ("lif", "plif", "ifnode", "neuron", "flexsn")):
        return "neuron"
    if any(
        word in lower
        for word in (
            "batch_norm",
            "batchnorm",
            "layer_norm",
            "layernorm",
            "group_norm",
            "cudnn::bn_",
        )
    ):
        return "normalization"
    if "pool" in lower:
        return "pool"
    if any(word in lower for word in ("softmax", "attention")):
        return "attention"
    if any(
        word in lower
        for word in (
            "memcpy",
            "copy",
            "transpose",
            "convert",
            "permute",
            "nhwctonchw",
            "nchwtonhwc",
            "nhwc2nchw",
            "nchw2nhwc",
        )
    ):
        return "layout_copy"
    if any(
        word in lower
        for word in (
            "conv",
            "gemm",
            "matmul",
            "sgemm",
            "hgemm",
            "winograd",
            "cutlass::",
            "cutlass__",
            "cudnn::cnn::",
        )
    ) or ("fft2d" in lower and "cudnn::" in lower):
        return "conv_gemm"
    if any(
        word in lower
        for word in (
            "pointwise",
            "elementwise",
            "vectorized_elementwise",
            "triton_poi_",
        )
    ):
        return "elementwise"
    return "unknown"


def analyze(
    sqlite_path: Path,
    benchmark_path: Path | None = None,
    step_index: int = 0,
) -> dict:
    db = sqlite3.connect(f"file:{sqlite_path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        if not _table_columns(db, "NVTX_EVENTS"):
            raise ValueError(
                "capture has no NVTX_EVENTS; enable NVTX tracing and mark complete steps"
            )
        required = {
            "NVTX_EVENTS": {"start", "end", "text", "textId", "globalTid"},
            "CUPTI_ACTIVITY_KIND_RUNTIME": {
                "start",
                "end",
                "correlationId",
                "globalTid",
                "nameId",
            },
            "CUPTI_ACTIVITY_KIND_KERNEL": {
                "start",
                "end",
                "correlationId",
                "demangledName",
                "globalPid",
            },
            "StringIds": {"id", "value"},
        }
        for table, columns in required.items():
            missing = columns - _table_columns(db, table)
            if missing:
                raise ValueError(
                    f"unsupported NSYS SQLite schema: {table} lacks {sorted(missing)}"
                )

        strings = dict(db.execute("SELECT id, value FROM StringIds"))
        ranges = []
        for row in db.execute(
            "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL"
        ):
            name = row["text"] or strings.get(row["textId"], "")
            if STEP.match(name) or name in PHASES or name.startswith("module:"):
                ranges.append((row["start"], row["end"], name, row["globalTid"]))
        graph_stage_by_node = {}
        if {"start", "graphNodeId", "originalGraphNodeId"} <= _table_columns(
            db, "CUDA_GRAPH_NODE_EVENTS"
        ):
            capture_stages = [
                (start, end, name)
                for start, end, name, _ in ranges
                if name in ("forward", "loss", "backward")
            ]
            capture_nodes = {}
            replay_nodes = []
            for row in db.execute(
                "SELECT start, graphNodeId, originalGraphNodeId "
                "FROM CUDA_GRAPH_NODE_EVENTS"
            ):
                if row["originalGraphNodeId"] is None:
                    stage = next(
                        (
                            name
                            for start, end, name in capture_stages
                            if start <= row["start"] < end
                        ),
                        None,
                    )
                    if stage is not None:
                        capture_nodes[row["graphNodeId"]] = stage
                else:
                    replay_nodes.append(
                        (row["graphNodeId"], row["originalGraphNodeId"])
                    )
            graph_stage_by_node = {
                node: capture_nodes[original]
                for node, original in replay_nodes
                if original in capture_nodes
            }
        steps = []
        for start, end, name, tid in ranges:
            match = STEP.match(name)
            if match:
                steps.append(
                    {
                        "name": name,
                        "phase": match.group(1),
                        "index": int(match.group(2)),
                        "start_ns": start,
                        "end_ns": end,
                        "tid": tid,
                        "kernels": [],
                        "cuda_api": [],
                        "cuda_api_events": [],
                        "categories_ns": defaultdict(int),
                        "phases_ns": defaultdict(int),
                        "unknown_kernels": defaultdict(int),
                    }
                )
        if not steps:
            raise ValueError("no sj.step or benchmark_step NVTX ranges in capture")

        gil = {"collected": False, "threads": []}
        gil_events = []
        nvtx_columns = _table_columns(db, "NVTX_EVENTS")
        if {"eventType", "domainId"} <= nvtx_columns:
            gil_domains = {
                row[0]
                for row in db.execute(
                    "SELECT domainId FROM NVTX_EVENTS WHERE eventType = 75 "
                    "AND text = 'GIL Trace'"
                )
            }
            if gil_domains:
                gil["collected"] = True
                placeholders = ",".join("?" for _ in gil_domains)
                for row in db.execute(
                    "SELECT start, end, textId, globalTid FROM NVTX_EVENTS "
                    f"WHERE eventType = 59 AND domainId IN ({placeholders}) "
                    "AND end IS NOT NULL",
                    tuple(gil_domains),
                ):
                    state = strings.get(row["textId"])
                    if state in ("Holding GIL", "Waiting for GIL"):
                        gil_events.append(
                            (
                                row["start"],
                                row["end"],
                                row["globalTid"],
                                state,
                            )
                        )
        gil_by_thread = defaultdict(lambda: {"Holding GIL": [], "Waiting for GIL": []})
        for step in steps:
            step["gil"] = defaultdict(
                lambda: {"Holding GIL": [], "Waiting for GIL": []}
            )
            for start, end, tid, state in gil_events:
                if _pid(tid) != _pid(step["tid"]):
                    continue
                clipped = (max(start, step["start_ns"]), min(end, step["end_ns"]))
                if clipped[0] < clipped[1]:
                    step["gil"][tid][state].append(clipped)
                    gil_by_thread[tid][state].append(clipped)
        for tid, states in sorted(gil_by_thread.items()):
            gil["threads"].append(
                {
                    "global_tid": tid,
                    "holding_ms": _union_ns(states["Holding GIL"]) / NS_PER_MS,
                    "waiting_ms": _union_ns(states["Waiting for GIL"]) / NS_PER_MS,
                }
            )

        launches = {}
        for row in db.execute(
            "SELECT start, end, correlationId, globalTid, nameId "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId IS NOT NULL"
        ):
            if row["globalTid"] is None:
                continue
            launch = dict(row)
            launch["name"] = strings.get(row["nameId"], "unknown CUDA API")
            active_steps = [
                index
                for index, step in enumerate(steps)
                if _pid(step["tid"]) == _pid(row["globalTid"])
                and step["start_ns"] <= row["start"] < step["end_ns"]
            ]
            launch["step_index"] = active_steps[0] if len(active_steps) == 1 else None
            launches[(_pid(row["globalTid"]), row["correlationId"])] = launch
            if launch["step_index"] is not None:
                step = steps[launch["step_index"]]
                step["cuda_api"].append((row["start"], row["end"]))
                step["cuda_api_events"].append(launch)

        def add_gpu_events(table: str, name_column: str, fallback: str) -> None:
            columns = _table_columns(db, table)
            if not columns:
                return
            stream_column = "streamId" if "streamId" in columns else "NULL"
            graph_node_column = "graphNodeId" if "graphNodeId" in columns else "NULL"
            for row in db.execute(
                f"SELECT start, end, correlationId, globalPid, {name_column} AS name, "
                f"{stream_column} AS streamId, {graph_node_column} AS graphNodeId "
                f"FROM {table} "
                "WHERE correlationId IS NOT NULL"
            ):
                if row["globalPid"] is None:
                    continue
                launch = launches.get((_pid(row["globalPid"]), row["correlationId"]))
                if launch is None:
                    continue
                if launch["step_index"] is not None:
                    step = steps[launch["step_index"]]
                    when = launch["start"]
                    name = (
                        strings.get(row["name"], fallback)
                        if row["name"] is not None
                        else fallback
                    ) or fallback
                    category = (
                        _category(name)
                        if fallback == "kernel"
                        else "layout_copy"
                        if fallback == "memcpy"
                        else "memory_fill"
                    )
                    duration = row["end"] - row["start"]
                    matching = [
                        (end - start, label)
                        for start, end, label, tid in ranges
                        if tid == step["tid"]
                        and label in PHASES
                        and start <= when < end
                    ]
                    phase = min(matching)[1] if matching else "other"
                    if category == "unknown" and phase == "optimizer":
                        category = "optimizer"
                    step["kernels"].append(
                        {
                            "start": row["start"],
                            "end": row["end"],
                            "category": category,
                            "phase": phase,
                            "name": name,
                            "kind": fallback,
                            "stream_id": row["streamId"],
                            "correlation_id": row["correlationId"],
                            "launch": launch,
                            "graph_stage": graph_stage_by_node.get(row["graphNodeId"]),
                        }
                    )
                    step["categories_ns"][category] += duration
                    if category == "unknown":
                        step["unknown_kernels"][name] += duration
                    step["phases_ns"][phase] += duration

        add_gpu_events("CUPTI_ACTIVITY_KIND_KERNEL", "demangledName", "kernel")
        add_gpu_events("CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind", "memcpy")
        add_gpu_events("CUPTI_ACTIVITY_KIND_MEMSET", "memKind", "memset")

        output_steps = []
        unknown = defaultdict(int)
        for step in sorted(steps, key=lambda item: item["start_ns"]):
            kernels = step["kernels"]
            intervals = [(event["start"], event["end"]) for event in kernels]
            busy_ns = _union_ns(intervals)
            gpu_span_ns = (
                max(end for _, end in intervals) - min(start for start, _ in intervals)
                if intervals
                else 0
            )
            for name, duration in step["unknown_kernels"].items():
                unknown[name] += duration
            output_steps.append(
                {
                    "name": step["name"],
                    "phase": step["phase"],
                    "index": step["index"],
                    "cpu_range_ms": (step["end_ns"] - step["start_ns"]) / NS_PER_MS,
                    "gpu_span_ms": gpu_span_ns / NS_PER_MS,
                    "gpu_busy_union_ms": busy_ns / NS_PER_MS,
                    "gpu_idle_within_span_ms": (gpu_span_ns - busy_ns) / NS_PER_MS,
                    "cuda_api_union_ms": _union_ns(step["cuda_api"]) / NS_PER_MS,
                    "gpu_event_count": len(kernels),
                    "main_thread_gil_holding_ms": (
                        _union_ns(step["gil"][step["tid"]]["Holding GIL"]) / NS_PER_MS
                        if gil["collected"]
                        else None
                    ),
                    "main_thread_gil_waiting_ms": (
                        _union_ns(step["gil"][step["tid"]]["Waiting for GIL"])
                        / NS_PER_MS
                        if gil["collected"]
                        else None
                    ),
                    "categories_ms": {
                        key: value / NS_PER_MS
                        for key, value in sorted(step["categories_ns"].items())
                    },
                    "phases_ms": {
                        key: value / NS_PER_MS
                        for key, value in sorted(step["phases_ns"].items())
                    },
                }
            )
        ordered_steps = sorted(steps, key=lambda item: item["start_ns"])
        selected = [
            position
            for position, step in enumerate(ordered_steps)
            if step["index"] == step_index
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected one captured step with index {step_index}, found {len(selected)}"
            )
        timeline_step = ordered_steps[selected[0]]
        origin = timeline_step["start_ns"]
        timeline = {
            "step": timeline_step["name"],
            "phases": [
                {
                    "name": name,
                    "start_ms": (max(start, origin) - origin) / NS_PER_MS,
                    "duration_ms": (
                        min(end, timeline_step["end_ns"]) - max(start, origin)
                    )
                    / NS_PER_MS,
                }
                for start, end, name, tid in ranges
                if tid == timeline_step["tid"]
                and name in PHASES
                and start < timeline_step["end_ns"]
                and end > origin
            ],
            "modules": [
                {
                    "name": name.removeprefix("module:"),
                    "start_ms": (max(start, origin) - origin) / NS_PER_MS,
                    "duration_ms": (
                        min(end, timeline_step["end_ns"]) - max(start, origin)
                    )
                    / NS_PER_MS,
                }
                for start, end, name, tid in ranges
                if tid == timeline_step["tid"]
                and name.startswith("module:")
                and start < timeline_step["end_ns"]
                and end > origin
            ],
            "gpu_events": [
                {
                    "category": event["category"],
                    "start_ms": (event["start"] - origin) / NS_PER_MS,
                    "duration_ms": (event["end"] - event["start"]) / NS_PER_MS,
                }
                for event in timeline_step["kernels"]
            ],
            "graph_stages": [
                {
                    "name": event["graph_stage"],
                    "start_ms": (event["start"] - origin) / NS_PER_MS,
                    "duration_ms": (event["end"] - event["start"]) / NS_PER_MS,
                }
                for event in timeline_step["kernels"]
                if event["graph_stage"] is not None
            ],
            "gil_events": [
                {
                    "global_tid": tid,
                    "state": state,
                    "start_ms": (max(start, origin) - origin) / NS_PER_MS,
                    "duration_ms": (
                        min(end, timeline_step["end_ns"]) - max(start, origin)
                    )
                    / NS_PER_MS,
                }
                for start, end, tid, state in gil_events
                if _pid(tid) == _pid(timeline_step["tid"])
                and start < timeline_step["end_ns"]
                and end > origin
            ],
        }
        timeline["async_step"] = {
            "main_thread_tid": timeline_step["tid"],
            "cuda_api_events": [
                {
                    "name": call["name"],
                    "global_tid": call["globalTid"],
                    "correlation_id": call["correlationId"],
                    "start_ms": (call["start"] - origin) / NS_PER_MS,
                    "duration_ms": (call["end"] - call["start"]) / NS_PER_MS,
                }
                for call in timeline_step["cuda_api_events"]
            ],
            "gpu_events": [
                {
                    "name": event["name"],
                    "kind": event["kind"],
                    "stream_id": event["stream_id"],
                    "correlation_id": event["correlation_id"],
                    "launch_thread_tid": event["launch"]["globalTid"],
                    "start_ms": (event["start"] - origin) / NS_PER_MS,
                    "duration_ms": (event["end"] - event["start"]) / NS_PER_MS,
                }
                for event in timeline_step["kernels"]
            ],
        }
        window_steps = ordered_steps[selected[0] : selected[0] + 4]
        window_origin = window_steps[0]["start_ns"]
        timeline["step_window"] = {
            "steps": [
                {
                    "name": step["name"],
                    "start_ms": (step["start_ns"] - window_origin) / NS_PER_MS,
                    "duration_ms": (step["end_ns"] - step["start_ns"]) / NS_PER_MS,
                    "phases": [
                        {
                            "name": name,
                            "start_ms": (start - window_origin) / NS_PER_MS,
                            "duration_ms": (end - start) / NS_PER_MS,
                        }
                        for start, end, name, tid in ranges
                        if tid == step["tid"]
                        and name in PHASES
                        and step["start_ns"] <= start < step["end_ns"]
                    ],
                    "gpu_events": [
                        {
                            "start_ms": (event["start"] - window_origin) / NS_PER_MS,
                            "duration_ms": (event["end"] - event["start"]) / NS_PER_MS,
                            "launch_phase": event["phase"],
                        }
                        for event in step["kernels"]
                    ],
                }
                for step in window_steps
            ]
        }
        benchmark = json.loads(benchmark_path.read_text()) if benchmark_path else None
        return {
            "schema_version": 1,
            "source_sqlite": str(sqlite_path),
            "benchmark": benchmark,
            "peak_allocated_mib": (
                benchmark["memory"]["peak_allocated_bytes"] / 1024**2
                if benchmark and benchmark.get("memory")
                else None
            ),
            "steps": output_steps,
            "gil": gil,
            "timeline": timeline,
            "unknown_kernels_ms": {
                key: value / NS_PER_MS
                for key, value in sorted(unknown.items(), key=lambda item: -item[1])[
                    :30
                ]
            },
        }
    finally:
        db.close()


def _write_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output_dir / "steps.csv").open("w", newline="", encoding="utf-8") as file:
        fields = [
            "name",
            "cpu_range_ms",
            "gpu_span_ms",
            "gpu_busy_union_ms",
            "gpu_idle_within_span_ms",
            "cuda_api_union_ms",
            "gpu_event_count",
            "main_thread_gil_holding_ms",
            "main_thread_gil_waiting_ms",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: step[field] for field in fields} for step in report["steps"]
        )
    if report["gil"]["collected"]:
        with (output_dir / "gil_threads.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=("global_tid", "holding_ms", "waiting_ms")
            )
            writer.writeheader()
            writer.writerows(report["gil"]["threads"])
    if report["timeline"]["modules"]:
        with (output_dir / "module_ranges.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=("name", "start_ms", "duration_ms")
            )
            writer.writeheader()
            writer.writerows(report["timeline"]["modules"])
    render(report, output_dir)


def compare(baseline: dict, candidate: dict) -> dict:
    left = baseline.get("benchmark") or {}
    right = candidate.get("benchmark") or {}
    baseline_case = left.get("case") or {}
    candidate_case = right.get("case") or {}
    workload_keys = (
        "model",
        "phase",
        "T",
        "batch_size",
        "image_size",
        "num_classes",
        "seed",
    )
    if (
        baseline_case
        and candidate_case
        and any(
            key in baseline_case
            and key in candidate_case
            and baseline_case[key] != candidate_case[key]
            for key in workload_keys
        )
    ):
        raise ValueError("benchmark workload metadata differs; cannot compare")
    workload_verified = bool(
        baseline_case
        and candidate_case
        and all(key in baseline_case and key in candidate_case for key in workload_keys)
    )

    def median(report: dict, key: str) -> float:
        return statistics.median(step[key] for step in report["steps"])

    keys = (
        "cpu_range_ms",
        "gpu_span_ms",
        "gpu_busy_union_ms",
        "gpu_idle_within_span_ms",
        "cuda_api_union_ms",
    )
    categories = sorted(
        {
            name
            for report in (baseline, candidate)
            for item in report["steps"]
            for name in item["categories_ms"]
        }
    )

    def category_median(report: dict, name: str) -> float:
        return statistics.median(
            item["categories_ms"].get(name, 0.0) for item in report["steps"]
        )

    return {
        "schema_version": 1,
        "note": "NSYS attribution deltas only; use independent unprofiled runs for speed claims",
        "workload_verified": workload_verified,
        "baseline_case": baseline_case,
        "candidate_case": candidate_case,
        "metrics": {
            key: {
                "baseline": median(baseline, key),
                "candidate": median(candidate, key),
                "delta": median(candidate, key) - median(baseline, key),
            }
            for key in keys
        },
        "categories_ms": {
            name: {
                "baseline": category_median(baseline, name),
                "candidate": category_median(candidate, name),
                "delta": category_median(candidate, name)
                - category_median(baseline, name),
            }
            for name in categories
        },
        "peak_allocated_mib": {
            "baseline": baseline.get("peak_allocated_mib"),
            "candidate": candidate.get("peak_allocated_mib"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SNN Nsight Systems SQLite exports"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("sqlite", type=Path)
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser.add_argument("--benchmark-json", type=Path)
    analyze_parser.add_argument("--step-index", type=int, default=0)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "analyze":
        _write_report(
            analyze(args.sqlite, args.benchmark_json, args.step_index),
            args.output_dir,
        )
    else:
        result = compare(
            json.loads(args.baseline.read_text()),
            json.loads(args.candidate.read_text()),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
