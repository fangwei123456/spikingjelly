"""Render schema-v2 NSYS attribution on one shared host clock."""

from collections import defaultdict
from pathlib import Path
from shutil import copyfile


def render(report: dict, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    processes = {p["process_id"]: p for p in report["processes"]}
    selected = report["selection"]
    colors = {
        "forward": "#2563eb",
        "backward": "#9333ea",
        "loss": "#ca8a04",
        "optimizer": "#ea580c",
        "zero_grad": "#64748b",
        "reset": "#dc2626",
        "communication": "#059669",
        "other": "#9ca3af",
    }

    def process_label(key):
        info = processes.get(key)
        if info is None:
            return "unknown process"
        rank = ",".join(map(str, info["ranks"])) if info["ranks"] else "?"
        return f"PID {info['pid']} rank {rank}"

    def visible(event, start, end):
        return (
            event["process_id"] in selected["process_ids"]
            and event["start_ns"] < end
            and event["end_ns"] > start
        )

    def draw(start, end, filename, *, pipeline=False, gil_only=False, modules=False):
        events = [
            e
            for e in report["gpu_events"]
            if visible(e, start, end)
            and (selected["device"] is None or e["device_key"] == selected["device"])
        ]
        calls = [c for c in report["cuda_api_events"] if visible(c, start, end)]
        ranges = [r for r in report["ranges"] if visible(r, start, end)]
        spans = defaultdict(list)
        for event in events:
            key = (
                "GPU",
                event["process_id"],
                event["device_key"],
                event["context_id"],
                event["stream_id"],
            )
            phase = event["graph_stage"] or event["phase"]
            color = (
                colors["communication"]
                if event["category"] == "communication"
                else colors.get(phase, colors["other"])
            )
            spans[key].append((event, color, ""))
        if not gil_only:
            for call in calls:
                key = ("CUDA API", call["process_id"], call["global_tid"] & 0xFFFFFF)
                color = "#dc2626" if "Synchronize" in call["name"] else "#475569"
                spans[key].append((call, color, ""))
        for region in ranges:
            if region["kind"] == "gil":
                key = ("GIL", region["process_id"], region["global_tid"] & 0xFFFFFF)
                spans[key].append(
                    (
                        region,
                        "#16a34a" if region["name"] == "Holding GIL" else "#f97316",
                        "",
                    )
                )
            elif not gil_only and region["kind"] == "step":
                key = (
                    "CPU step",
                    region["process_id"],
                    region["global_tid"] & 0xFFFFFF,
                )
                label = str(region["index"]) + ("" if region["complete"] else " (open)")
                spans[key].append((region, "#cbd5e1", label))
            elif not gil_only and region["kind"] in ("region", "communication"):
                if modules and not region["name"].startswith("module:"):
                    continue
                if pipeline and region["stage"] is None:
                    continue
                if not modules and region["name"].startswith("module:"):
                    continue
                key = (
                    "CPU stage",
                    region["process_id"],
                    region["global_tid"] & 0xFFFFFF,
                    region["stage"],
                    region["name"],
                )
                label = (
                    f"mb {region['microbatch']}"
                    if region["microbatch"] is not None
                    else ""
                )
                spans[key].append(
                    (
                        region,
                        colors.get(
                            region["name"],
                            colors["communication"]
                            if region["kind"] == "communication"
                            else colors["other"],
                        ),
                        label,
                    )
                )
        keys = sorted(spans, key=str)
        if not keys:
            return
        fig, ax = plt.subplots(figsize=(15, max(3, 0.35 * len(keys) + 1.8)))
        for y, key in enumerate(keys):
            grouped = defaultdict(list)
            for event, color, label in spans[key]:
                a, b = max(start, event["start_ns"]), min(end, event["end_ns"])
                grouped[color].append(((a - start) / 1e6, (b - a) / 1e6))
                if label and b - a > (end - start) / 50:
                    ax.text(
                        (a + b - 2 * start) / 2e6,
                        y + 0.45,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
            for color, intervals in grouped.items():
                ax.broken_barh(intervals, (y + 0.1, 0.7), facecolors=color)
        # Correlation links express enqueue relationships, not inferred waits.
        lane_y = {key: i + 0.45 for i, key in enumerate(keys)}
        api_by_id = {c["api_id"]: c for c in calls}
        linked = [e for e in events if e["api_id"] in api_by_id]
        for event in linked[:: max(1, len(linked) // 24)]:
            call = api_by_id[event["api_id"]]
            cpu_lane = ("CUDA API", call["process_id"], call["global_tid"] & 0xFFFFFF)
            gpu_lane = (
                "GPU",
                event["process_id"],
                event["device_key"],
                event["context_id"],
                event["stream_id"],
            )
            if cpu_lane in lane_y and gpu_lane in lane_y:
                ax.annotate(
                    "",
                    xy=((event["start_ns"] - start) / 1e6, lane_y[gpu_lane]),
                    xytext=((call["start_ns"] - start) / 1e6, lane_y[cpu_lane]),
                    arrowprops={"arrowstyle": "->", "lw": 0.35, "alpha": 0.35},
                )
        labels = []
        for key in keys:
            if key[0] == "GPU":
                labels.append(
                    f"{process_label(key[1])} / {key[2]} / ctx {key[3]} stream {key[4]}"
                )
            else:
                extra = (
                    (f" stage {key[3]}" if key[3] is not None else "") + f" {key[4]}"
                    if len(key) > 3
                    else ""
                )
                labels.append(f"{key[0]} {process_label(key[1])} TID {key[2]}{extra}")
        ax.set_yticks([i + 0.45 for i in range(len(keys))], labels=labels, fontsize=7)
        ax.set_xlim(0, max((end - start) / 1e6, 0.001))
        ax.set_xlabel(f"Time from host timestamp {start} ns (ms)")
        ax.set_title(
            "Single-host CPU / CUDA / GPU timeline; arrows are launch correlations"
        )
        ax.grid(axis="x", alpha=0.2)
        fig.legend(
            handles=[Patch(color=color, label=name) for name, color in colors.items()],
            loc="lower center",
            ncol=5,
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    start, end = selected["start_ns"], selected["end_ns"]
    draw(start, end, "event_timeline.png")
    if any(e["graph_stage"] for e in report["gpu_events"]):
        timeline = output_dir / "event_timeline.png"
        if timeline.exists():
            copyfile(timeline, output_dir / "graph_stage_timeline.png")
    if report["gil_collected"]:
        draw(start, end, "gil_timeline.png", gil_only=True)
    if any(r["stage"] is not None for r in report["ranges"]):
        draw(start, end, "pipeline_timeline.png", pipeline=True)
    if any(r["name"].startswith("module:") for r in report["ranges"]):
        draw(start, end, "module_timeline.png", modules=True)
    index = selected["step_index"]
    window_steps = [
        s
        for s in report["steps"]
        if index is not None
        and index <= s["index"] < index + 4
        and s["process_id"] in selected["process_ids"]
    ]
    if len({s["index"] for s in window_steps}) == 4:
        draw(
            min(s["start_ns"] for s in window_steps),
            max(max(s["end_ns"], s["gpu_end_ns"] or s["end_ns"]) for s in window_steps),
            "four_step_timeline.png",
        )
    categories = sorted({e["category"] for e in report["gpu_events"]})
    device_keys = [d["device_key"] for d in report["device_summary"]]
    if device_keys:
        fig, ax = plt.subplots(figsize=(max(8, len(device_keys) * 2), 4))
        bottom = [0.0] * len(device_keys)
        for category in categories:
            values = [
                sum(
                    (e["end_ns"] - e["start_ns"]) / 1e6
                    for e in report["gpu_events"]
                    if e["device_key"] == key and e["category"] == category
                )
                for key in device_keys
            ]
            ax.bar(device_keys, values, bottom=bottom, label=category)
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set_ylabel("Summed GPU event time per device (ms); full capture")
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        fig.savefig(output_dir / "kernel_cost.png", dpi=160)
        plt.close(fig)
