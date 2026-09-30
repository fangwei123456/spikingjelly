"""Analyze one single-host Nsight Systems SQLite export without a GPU."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
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


def _table_columns(db, table):
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _rows(db, table, columns):
    available = _table_columns(db, table)
    if not available:
        return []
    projection = ", ".join(
        name if name in available else f"NULL AS {name}" for name in columns
    )
    return db.execute(f"SELECT {projection} FROM {table}")


def _union_ns(intervals):
    total, end = 0, None
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if end is None:
            total += stop - start
        elif stop > end:
            total += stop - max(start, end)
        end = max(stop, end) if end is not None else stop
    return total


def _pid(global_id):
    return (global_id >> 24) & 0xFFFFFF


def _process(global_id):
    return global_id & ~0xFFFFFF


def _category(name):
    lower = name.lower()
    if "nccl" in lower:
        return "communication"
    groups = (
        ("neuron", ("lif", "plif", "ifnode", "neuron", "flexsn")),
        (
            "normalization",
            (
                "batch_norm",
                "batchnorm",
                "layer_norm",
                "layernorm",
                "group_norm",
                "cudnn::bn_",
            ),
        ),
        ("pool", ("pool",)),
        ("attention", ("softmax", "attention")),
        (
            "layout_copy",
            (
                "memcpy",
                "copy",
                "transpose",
                "convert",
                "permute",
                "nhwctonchw",
                "nchwtonhwc",
                "nhwc2nchw",
                "nchw2nhwc",
            ),
        ),
        (
            "conv_gemm",
            (
                "conv",
                "gemm",
                "matmul",
                "sgemm",
                "hgemm",
                "winograd",
                "cutlass::",
                "cutlass__",
                "cudnn::cnn::",
            ),
        ),
        (
            "elementwise",
            ("pointwise", "elementwise", "vectorized_elementwise", "triton_poi_"),
        ),
    )
    for category, words in groups:
        if any(word in lower for word in words):
            return category
    return "conv_gemm" if "fft2d" in lower and "cudnn::" in lower else "unknown"


def _decode_label(text):
    name, delimiter, metadata = text.rpartition("|sj:")
    if not delimiter:
        return text, {}
    values = json.loads(metadata)
    if not isinstance(values, dict):
        raise ValueError(f"invalid SpikingJelly NVTX metadata: {text}")
    return name, values


def _enclosing(scopes, event):
    matches = [
        r
        for r in scopes
        if r["process_id"] == event["process_id"]
        and r["start_ns"] <= event["start_ns"] < r["end_ns"]
    ]
    local = [r for r in matches if r["global_tid"] == event["global_tid"]]
    if local:
        shortest = min(r["end_ns"] - r["start_ns"] for r in local)
        matches = [r for r in local if r["end_ns"] - r["start_ns"] == shortest]
    elif len({r["global_tid"] for r in matches}) == 1 and matches:
        matches = [min(matches, key=lambda r: r["end_ns"] - r["start_ns"])]
    return matches[0] if len(matches) == 1 else None


def _gpu_metrics(events, start=None, end=None):
    intervals = [(e["start_ns"], e["end_ns"]) for e in events]
    busy = _union_ns(intervals)
    span = (
        max(e for _, e in intervals) - min(s for s, _ in intervals) if intervals else 0
    )
    compute = [
        (e["start_ns"], e["end_ns"])
        for e in events
        if e["kind"] == "kernel" and e["category"] not in ("communication", "unknown")
    ]
    communication = [
        (e["start_ns"], e["end_ns"]) for e in events if e["category"] == "communication"
    ]
    compute_ns, comm_ns = _union_ns(compute), _union_ns(communication)
    return {
        "gpu_event_count": len(events),
        "gpu_span_ms": span / NS_PER_MS,
        "gpu_busy_union_ms": busy / NS_PER_MS,
        "gpu_idle_within_span_ms": (span - busy) / NS_PER_MS,
        "window_idle_ms": (
            (end - start)
            - _union_ns([(max(a, start), min(b, end)) for a, b in intervals])
        )
        / NS_PER_MS
        if start is not None and end is not None
        else None,
        "compute_union_ms": compute_ns / NS_PER_MS,
        "communication_union_ms": comm_ns / NS_PER_MS,
        "compute_communication_overlap_ms": (
            compute_ns + comm_ns - _union_ns(compute + communication)
        )
        / NS_PER_MS,
    }


def _device_summaries(events, start=None, end=None):
    groups = defaultdict(list)
    for event in events:
        groups[event["device_key"]].append(event)
    return [
        {"device_key": key, **_gpu_metrics(values, start, end)}
        for key, values in sorted(groups.items())
    ]


def analyze(
    sqlite_path: Path,
    benchmark_path: Path | None = None,
    step_index: int = 0,
    *,
    pid: int | None = None,
    rank: int | None = None,
    device: str | None = None,
    time_range_ms: tuple[float, float] | None = None,
) -> dict:
    """Return schema-v2 full-capture statistics and a filtered timeline selection."""
    db = sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        strings = (
            dict(db.execute("SELECT id, value FROM StringIds"))
            if _table_columns(db, "StringIds")
            else {}
        )
        processes = {}

        def process_info(key):
            if key not in processes:
                processes[key] = {
                    "process_id": key,
                    "pid": _pid(key),
                    "name": None,
                    "ranks": [],
                }
            return processes[key]

        for row in _rows(db, "PROCESSES", ["globalPid", "name"]):
            if row["globalPid"] is not None:
                process_info(_process(row["globalPid"]))["name"] = row["name"]

        device_map, devices = {}, {}
        for row in _rows(
            db, "TARGET_INFO_CUDA_DEVICE", ["gpuId", "cudaId", "pid", "uuid"]
        ):
            key = row["uuid"] or f"gpu:{row['gpuId']}"
            device_map[(row["pid"], row["gpuId"])] = key
            devices[key] = {
                "device_key": key,
                "gpu_id": row["gpuId"],
                "uuid": row["uuid"],
            }

        ranges, domains = [], {}
        raw_nvtx = list(
            _rows(
                db,
                "NVTX_EVENTS",
                [
                    "start",
                    "end",
                    "text",
                    "textId",
                    "globalTid",
                    "eventType",
                    "domainId",
                ],
            )
        )
        for row in raw_nvtx:
            if row["globalTid"] is not None and row["eventType"] == 75:
                domains[(_process(row["globalTid"]), row["domainId"])] = row[
                    "text"
                ] or strings.get(row["textId"], "")
        for row in raw_nvtx:
            if row["globalTid"] is None or row["eventType"] == 75:
                continue
            name, metadata = _decode_label(
                row["text"] or strings.get(row["textId"], "")
            )
            process_id = _process(row["globalTid"])
            domain = domains.get((process_id, row["domainId"]), "")
            match = STEP.match(name)
            kind = (
                "step"
                if match
                else "gil"
                if domain == "GIL Trace"
                else "communication"
                if "nccl" in domain.lower()
                else "region"
            )
            if row["end"] is None and not match:
                continue
            item = {
                "name": name,
                "kind": kind,
                "process_id": process_id,
                "global_tid": row["globalTid"],
                "start_ns": row["start"],
                "end_ns": row["end"],
                "complete": row["end"] is not None,
                "rank": metadata.get("rank"),
                "world_size": metadata.get("world_size"),
                "stage": metadata.get("stage"),
                "microbatch": metadata.get("microbatch"),
            }
            if match:
                item.update(phase=match.group(1), index=int(match.group(2)))
            info = process_info(process_id)
            if item["rank"] is not None and item["rank"] not in info["ranks"]:
                info["ranks"].append(item["rank"])
            ranges.append(item)

        api_events, launches = [], defaultdict(list)
        for api_kind in ("RUNTIME", "DRIVER"):
            for row in _rows(
                db,
                f"CUPTI_ACTIVITY_KIND_{api_kind}",
                ["start", "end", "correlationId", "globalTid", "nameId"],
            ):
                if row["globalTid"] is None:
                    continue
                process_id = _process(row["globalTid"])
                process_info(process_id)
                call = {
                    "api_id": len(api_events),
                    "api_kind": api_kind,
                    "process_id": process_id,
                    "global_tid": row["globalTid"],
                    "name": strings.get(row["nameId"], "unknown CUDA API"),
                    "start_ns": row["start"],
                    "end_ns": row["end"],
                    "correlation_id": row["correlationId"],
                }
                api_events.append(call)
                if row["correlationId"] is not None:
                    launches[(process_id, row["correlationId"], api_kind)].append(call)

        gpu_events = []
        for kind, table, name_column in (
            ("kernel", "CUPTI_ACTIVITY_KIND_KERNEL", "demangledName"),
            ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind"),
            ("memset", "CUPTI_ACTIVITY_KIND_MEMSET", "memKind"),
        ):
            columns = [
                "start",
                "end",
                "globalPid",
                "deviceId",
                "contextId",
                "streamId",
                "correlationId",
                "graphNodeId",
                "bytes",
                "srcDeviceId",
                "dstDeviceId",
                name_column,
            ]
            for row in _rows(db, table, columns):
                process_id = (
                    _process(row["globalPid"]) if row["globalPid"] is not None else None
                )
                if process_id is not None:
                    process_info(process_id)
                device_key = (
                    device_map.get((_pid(process_id), row["deviceId"]))
                    if process_id is not None
                    else None
                )
                if device_key is None:
                    device_key = (
                        f"gpu:{row['deviceId']}"
                        if row["deviceId"] is not None
                        else f"unknown:{process_id}"
                    )
                    devices.setdefault(
                        device_key,
                        {
                            "device_key": device_key,
                            "gpu_id": row["deviceId"],
                            "uuid": None,
                        },
                    )
                name = (
                    strings.get(row[name_column], "kernel")
                    if kind == "kernel"
                    else kind
                )
                candidates = launches.get(
                    (process_id, row["correlationId"], "RUNTIME"), []
                ) or launches.get((process_id, row["correlationId"], "DRIVER"), [])
                call = candidates[0] if len(candidates) == 1 else None
                gpu_events.append(
                    {
                        "event_id": len(gpu_events),
                        "kind": kind,
                        "name": name,
                        "process_id": process_id,
                        "device_key": device_key,
                        "device_id": row["deviceId"],
                        "context_id": row["contextId"],
                        "stream_id": row["streamId"],
                        "start_ns": row["start"],
                        "end_ns": row["end"],
                        "bytes": row["bytes"],
                        "src_device_id": row["srcDeviceId"],
                        "dst_device_id": row["dstDeviceId"],
                        "correlation_id": row["correlationId"],
                        "graph_node_id": row["graphNodeId"],
                        "api_id": call["api_id"] if call else None,
                        "unassigned_reason": "ambiguous_correlation"
                        if len(candidates) > 1
                        else "missing_correlation"
                        if call is None
                        else None,
                        "category": _category(name)
                        if kind == "kernel"
                        else "layout_copy"
                        if kind == "memcpy"
                        else "memory_fill",
                    }
                )
        if not gpu_events and not api_events and not ranges:
            raise ValueError(
                "capture has no CUDA API or GPU events; enable CUDA tracing"
            )
        times = gpu_events + api_events or [
            r for r in ranges if r["end_ns"] is not None
        ]
        origin = min(e["start_ns"] for e in times)
        end = max(e["end_ns"] for e in times)
        for item in ranges:
            if item["end_ns"] is None:
                item["end_ns"] = end
        steps = sorted(
            (r for r in ranges if r["kind"] == "step"),
            key=lambda r: (r["start_ns"], r["process_id"], r["global_tid"]),
        )
        by_process_steps, by_process_regions = defaultdict(list), defaultdict(list)
        for index, step in enumerate(steps):
            step["step_id"] = index
            by_process_steps[step["process_id"]].append(step)
        for item in ranges:
            if item["kind"] == "region" and (
                item["name"] in PHASES
                or item["stage"] is not None
                or item["microbatch"] is not None
            ):
                by_process_regions[item["process_id"]].append(item)
        for call in api_events:
            step = _enclosing(by_process_steps[call["process_id"]], call)
            phase = _enclosing(by_process_regions[call["process_id"]], call)
            call["step_id"] = step["step_id"] if step else None
            call["phase"] = phase["name"] if phase else "other"
            call["stage"] = phase["stage"] if phase else None
            call["microbatch"] = phase["microbatch"] if phase else None

        graph_stages, clones = {}, []
        for row in _rows(
            db,
            "CUDA_GRAPH_NODE_EVENTS",
            ["start", "globalTid", "graphNodeId", "originalGraphNodeId"],
        ):
            process_id = (
                _process(row["globalTid"])
                if row["globalTid"] is not None
                else next(iter(processes))
                if len(processes) == 1
                else None
            )
            if process_id is None:
                continue
            key = (process_id, row["graphNodeId"])
            if row["originalGraphNodeId"] is not None:
                clones.append((key, (process_id, row["originalGraphNodeId"])))
            else:
                phase = _enclosing(
                    by_process_regions[process_id],
                    {
                        "process_id": process_id,
                        "global_tid": row["globalTid"],
                        "start_ns": row["start"],
                    },
                )
                if phase:
                    graph_stages[key] = phase
        while clones:
            remaining = []
            for key, original in clones:
                if original in graph_stages:
                    graph_stages[key] = graph_stages[original]
                else:
                    remaining.append((key, original))
            if len(remaining) == len(clones):
                break
            clones = remaining
        for event in gpu_events:
            call = api_events[event["api_id"]] if event["api_id"] is not None else None
            event.update(
                step_id=call["step_id"] if call else None,
                phase=call["phase"] if call else "other",
                stage=call["stage"] if call else None,
                microbatch=call["microbatch"] if call else None,
            )
            if event["step_id"] is None and event["unassigned_reason"] is None:
                event["unassigned_reason"] = "no_unique_step"
            phase = graph_stages.get((event["process_id"], event["graph_node_id"]))
            event["graph_stage"] = phase["name"] if phase else None
            if phase:
                event.update(stage=phase["stage"], microbatch=phase["microbatch"])
            if event["category"] == "unknown" and event["phase"] == "optimizer":
                event["category"] = "optimizer"

        gil_ranges = [r for r in ranges if r["kind"] == "gil"]
        gil_collected = "GIL Trace" in domains.values()
        gil_processes = {key[0] for key, name in domains.items() if name == "GIL Trace"}
        for info in processes.values():
            info["gil_collected"] = info["process_id"] in gil_processes
        pipeline_groups = defaultdict(lambda: {"ranges": [], "events": []})
        for item in ranges:
            if item["stage"] is not None or item["microbatch"] is not None:
                step = _enclosing(by_process_steps[item["process_id"]], item)
                key = (
                    step["step_id"] if step else None,
                    item["process_id"],
                    item["stage"],
                    item["microbatch"],
                    item["name"],
                )
                pipeline_groups[key]["ranges"].append(item)
        for event in gpu_events:
            if event["stage"] is not None or event["microbatch"] is not None:
                key = (
                    event["step_id"],
                    event["process_id"],
                    event["stage"],
                    event["microbatch"],
                    event["graph_stage"] or event["phase"],
                )
                pipeline_groups[key]["events"].append(event)
        pipeline = [
            {
                "step_id": key[0],
                "process_id": key[1],
                "stage": key[2],
                "microbatch": key[3],
                "phase": key[4],
                "cpu_range_union_ms": _union_ns(
                    [(r["start_ns"], r["end_ns"]) for r in group["ranges"]]
                )
                / NS_PER_MS,
                "devices": _device_summaries(group["events"]),
            }
            for key, group in pipeline_groups.items()
        ]
        step_summaries = []
        for step in steps:
            events = [e for e in gpu_events if e["step_id"] == step["step_id"]]
            calls = [c for c in api_events if c["step_id"] == step["step_id"]]
            categories, phases = defaultdict(float), defaultdict(float)
            for event in events:
                categories[event["category"]] += (
                    event["end_ns"] - event["start_ns"]
                ) / NS_PER_MS
                phases[event["phase"]] += (
                    event["end_ns"] - event["start_ns"]
                ) / NS_PER_MS
            summary = {
                **step,
                "cpu_range_ms": (step["end_ns"] - step["start_ns"]) / NS_PER_MS,
                "gpu_end_ns": max((e["end_ns"] for e in events), default=None),
                "cuda_api_union_ms": _union_ns(
                    [(c["start_ns"], c["end_ns"]) for c in calls]
                )
                / NS_PER_MS,
                "gpu_event_count": len(events),
                "categories_ms": dict(categories),
                "phases_ms": dict(phases),
                "devices": _device_summaries(events),
            }
            single_device = len(summary["devices"]) <= 1
            for name in ("gpu_span_ms", "gpu_busy_union_ms", "gpu_idle_within_span_ms"):
                summary[name] = _gpu_metrics(events)[name] if single_device else None
            for state, field in (
                ("Holding GIL", "main_thread_gil_holding_ms"),
                ("Waiting for GIL", "main_thread_gil_waiting_ms"),
            ):
                summary[field] = (
                    _union_ns(
                        [
                            (
                                max(r["start_ns"], step["start_ns"]),
                                min(r["end_ns"], step["end_ns"]),
                            )
                            for r in gil_ranges
                            if r["global_tid"] == step["global_tid"]
                            and r["name"] == state
                        ]
                    )
                    / NS_PER_MS
                    if step["process_id"] in gil_processes
                    else None
                )
            step_summaries.append(summary)

        groups = defaultdict(list)
        for step in step_summaries:
            if step["rank"] is not None and step["world_size"] is not None:
                groups[(step["phase"], step["index"])].append(step)
        global_steps = []
        for (phase, index), members in sorted(groups.items()):
            sizes = {s["world_size"] for s in members}
            expected = next(iter(sizes)) if len(sizes) == 1 else None
            ranks = [s["rank"] for s in members]
            complete = (
                expected is not None
                and sorted(ranks) == list(range(expected))
                and all(s["complete"] for s in members)
            )
            start, stop = (
                min(s["start_ns"] for s in members),
                max(s["end_ns"] for s in members),
            )
            global_steps.append(
                {
                    "phase": phase,
                    "index": index,
                    "start_ns": start,
                    "end_ns": stop,
                    "world_size": expected,
                    "ranks": sorted(ranks),
                    "complete": complete,
                    "missing_ranks": sorted(set(range(expected)) - set(ranks))
                    if expected is not None
                    else None,
                    "cpu_envelope_ms": (stop - start) / NS_PER_MS,
                    "gpu_end_ns": max(
                        (
                            s["gpu_end_ns"]
                            for s in members
                            if s["gpu_end_ns"] is not None
                        ),
                        default=None,
                    ),
                    "start_skew_ms": (max(s["start_ns"] for s in members) - start)
                    / NS_PER_MS,
                    "end_skew_ms": (stop - min(s["end_ns"] for s in members))
                    / NS_PER_MS,
                    "step_ids": [s["step_id"] for s in members],
                }
            )

        threads = []
        thread_ids = {c["global_tid"] for c in api_events} | {
            r["global_tid"] for r in gil_ranges
        }
        for tid in sorted(thread_ids):
            calls = [c for c in api_events if c["global_tid"] == tid]
            item = {
                "global_tid": tid,
                "process_id": _process(tid),
                "tid": tid & 0xFFFFFF,
                "cuda_api_union_ms": _union_ns(
                    [(c["start_ns"], c["end_ns"]) for c in calls]
                )
                / NS_PER_MS,
                "synchronize_union_ms": _union_ns(
                    [
                        (c["start_ns"], c["end_ns"])
                        for c in calls
                        if "Synchronize" in c["name"]
                    ]
                )
                / NS_PER_MS,
            }
            for state, field in (
                ("Holding GIL", "holding_ms"),
                ("Waiting for GIL", "waiting_ms"),
            ):
                windows = by_process_steps[_process(tid)]
                intervals = (
                    [
                        (
                            max(r["start_ns"], s["start_ns"]),
                            min(r["end_ns"], s["end_ns"]),
                        )
                        for r in gil_ranges
                        if r["global_tid"] == tid and r["name"] == state
                        for s in windows
                    ]
                    if windows
                    else [
                        (r["start_ns"], r["end_ns"])
                        for r in gil_ranges
                        if r["global_tid"] == tid and r["name"] == state
                    ]
                )
                item[field] = (
                    _union_ns(intervals) / NS_PER_MS
                    if _process(tid) in gil_processes
                    else None
                )
            threads.append(item)

        eligible_processes = {
            key
            for key, info in processes.items()
            if (pid is None or info["pid"] == pid)
            and (rank is None or rank in info["ranks"])
        }
        if (
            pid is None
            and rank is None
            and any(e["process_id"] is None for e in gpu_events)
        ):
            eligible_processes.add(None)
        if not eligible_processes:
            raise ValueError("no process matches the PID/rank selection")
        if device is not None and device not in devices:
            raise ValueError(f"unknown device key: {device}")
        selected_steps = [
            s
            for s in step_summaries
            if s["index"] == step_index
            and s["process_id"] in eligible_processes
            and (rank is None or s["rank"] == rank)
        ]
        if steps and not selected_steps and time_range_ms is None:
            raise ValueError(
                f"no captured step with index {step_index} matches the selection"
            )
        selected_ids = {s["step_id"] for s in selected_steps}
        selected_events = [
            e
            for e in gpu_events
            if e["process_id"] in eligible_processes
            and (not steps or e["step_id"] in selected_ids)
        ]
        start = min((s["start_ns"] for s in selected_steps), default=origin)
        stop = max(
            [s["end_ns"] for s in selected_steps]
            + [e["end_ns"] for e in selected_events],
            default=end,
        )
        if time_range_ms is not None:
            if not 0 <= time_range_ms[0] < time_range_ms[1]:
                raise ValueError("time range must satisfy 0 <= start < end")
            start, stop = (origin + int(v * NS_PER_MS) for v in time_range_ms)
        benchmark = json.loads(benchmark_path.read_text()) if benchmark_path else None
        peak_bytes = (
            (benchmark.get("memory") or {}).get("peak_allocated_bytes")
            if benchmark
            else None
        )
        unassigned = Counter(
            e["unassigned_reason"] for e in gpu_events if e["step_id"] is None
        )
        return {
            "schema_version": 2,
            "source_sqlite": str(sqlite_path),
            "benchmark": benchmark,
            "peak_allocated_mib": peak_bytes / 1024**2
            if peak_bytes is not None
            else None,
            "capture_start_ns": origin,
            "capture_end_ns": end,
            "processes": list(processes.values()),
            "devices": list(devices.values()),
            "threads": threads,
            "steps": step_summaries,
            "global_steps": global_steps,
            "pipeline": pipeline,
            "device_summary": _device_summaries(gpu_events, origin, end),
            "gpu_events": gpu_events,
            "cuda_api_events": api_events,
            "ranges": ranges,
            "gil_collected": gil_collected,
            "coverage": {
                "raw_gpu_events": len(gpu_events),
                "assigned_gpu_events": sum(
                    e["step_id"] is not None for e in gpu_events
                ),
                "unassigned_gpu_events": sum(unassigned.values()),
                "unassigned_reasons": dict(unassigned),
                "incomplete_steps": sum(not s["complete"] for s in steps),
            },
            "selection": {
                "step_index": step_index if selected_steps else None,
                "step_ids": sorted(selected_ids),
                "process_ids": sorted(eligible_processes, key=str),
                "device": device,
                "start_ns": start,
                "end_ns": stop,
                "time_range": time_range_ms is not None,
            },
        }
    finally:
        db.close()


def _write_report(report, output_dir):
    if (output_dir / "summary.json").exists():
        raise FileExistsError(f"report already exists in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for filename, records in (
        ("steps", report["steps"]),
        ("global_steps", report["global_steps"]),
        ("pipeline", report["pipeline"]),
        ("devices", report["device_summary"]),
        ("threads", report["threads"]),
        (
            "unassigned_events",
            [e for e in report["gpu_events"] if e["step_id"] is None],
        ),
    ):
        if not records:
            continue
        fields = [
            key
            for key, value in records[0].items()
            if not isinstance(value, (list, dict))
        ]
        with (output_dir / f"{filename}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    render(report, output_dir)


def compare(baseline: dict, candidate: dict) -> dict:
    for report in (baseline, candidate):
        if report.get("schema_version") != 2:
            raise ValueError(
                "schema v2 required; re-run analyze on the original SQLite export"
            )
    cases = [
        (r.get("benchmark") or {}).get("case") or {} for r in (baseline, candidate)
    ]
    keys = (
        "model",
        "phase",
        "T",
        "batch_size",
        "image_size",
        "num_classes",
        "seed",
        "world_size",
        "parallelism",
        "microbatches",
    )
    if any(
        k in cases[0] and k in cases[1] and cases[0][k] != cases[1][k] for k in keys
    ):
        raise ValueError("benchmark workload metadata differs; cannot compare")
    left_devices, right_devices = (
        baseline["device_summary"],
        candidate["device_summary"],
    )
    if {d["device_key"] for d in left_devices} != {
        d["device_key"] for d in right_devices
    }:
        raise ValueError(
            "device identities differ; compare reports collected on the same GPUs"
        )
    metrics = (
        "gpu_busy_union_ms",
        "gpu_idle_within_span_ms",
        "communication_union_ms",
        "compute_communication_overlap_ms",
    )
    by_device = []
    for left in left_devices:
        right = next(d for d in right_devices if d["device_key"] == left["device_key"])
        by_device.append(
            {
                "device_key": left["device_key"],
                "metrics": {
                    k: {
                        "baseline": left[k],
                        "candidate": right[k],
                        "delta": right[k] - left[k],
                    }
                    for k in metrics
                },
            }
        )
    logical = []
    for report in (baseline, candidate):
        samples = [
            s["cpu_envelope_ms"] for s in report["global_steps"] if s["complete"]
        ]
        logical.append(statistics.median(samples) if samples else None)
    return {
        "schema_version": 2,
        "note": "NSYS attribution only; use independent unprofiled runs for speed claims. GPU totals are per capture, not step latency.",
        "baseline_case": cases[0],
        "candidate_case": cases[1],
        "workload_verified": all(all(k in c for k in keys) for c in cases),
        "devices": by_device,
        "logical_step_cpu_median_ms": {"baseline": logical[0], "candidate": logical[1]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("sqlite", type=Path)
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--benchmark-json", type=Path)
    analysis.add_argument("--step-index", type=int, default=0)
    analysis.add_argument("--pid", type=int)
    analysis.add_argument("--rank", type=int)
    analysis.add_argument(
        "--device", help="GPU UUID or report device key; filters the timeline"
    )
    analysis.add_argument(
        "--time-range-ms", nargs=2, type=float, metavar=("START", "END")
    )
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "analyze":
        _write_report(
            analyze(
                args.sqlite,
                args.benchmark_json,
                args.step_index,
                pid=args.pid,
                rank=args.rank,
                device=args.device,
                time_range_ms=args.time_range_ms,
            ),
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
