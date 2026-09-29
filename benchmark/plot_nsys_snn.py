"""Render NSYS event attribution; benchmark timing plots live elsewhere."""

from pathlib import Path


def render(report: dict, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    steps = report["steps"]
    labels = [str(step["index"]) for step in steps]
    categories = sorted({key for step in steps for key in step["categories_ms"]})

    fig, ax = plt.subplots(figsize=(max(8, len(steps) * 0.65), 4))
    bottom = [0.0] * len(steps)
    for category in categories:
        values = [step["categories_ms"].get(category, 0.0) for step in steps]
        ax.bar(labels, values, bottom=bottom, label=category)
        bottom = [old + value for old, value in zip(bottom, values)]
    ax.set(xlabel="Captured step", ylabel="Summed GPU event time (ms)")
    if categories:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(output_dir / "kernel_cost.png", dpi=160)
    plt.close(fig)

    timeline = report["timeline"]
    detail = timeline["async_step"]
    api_events = detail["cuda_api_events"]
    gpu_events = detail["gpu_events"]
    threads = sorted({event["global_tid"] for event in api_events})
    streams = sorted({event["stream_id"] for event in gpu_events}, key=str)
    lane_labels = [
        *(f"GPU stream {stream}" for stream in streams),
        *(
            "CUDA API main thread"
            if tid == detail["main_thread_tid"]
            else f"CUDA API worker {tid & 0xFFFFFFFF}"
            for tid in threads
        ),
        "CPU NVTX stage",
    ]
    gpu_y = {stream: index for index, stream in enumerate(streams)}
    api_y = {tid: len(streams) + index for index, tid in enumerate(threads)}
    stage_y = len(lane_labels) - 1
    sync_calls = [event for event in api_events if "Synchronize" in event["name"]]
    focus = (
        max(sync_calls, key=lambda event: event["duration_ms"]) if sync_calls else None
    )
    fig, (full, zoom, reset_zoom) = plt.subplots(
        3, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [2.2, 1.8, 1.2]}
    )
    for ax in (full, zoom, reset_zoom):
        for phase in timeline["phases"]:
            ax.broken_barh(
                [(phase["start_ms"], phase["duration_ms"])],
                (stage_y + 0.1, 0.8),
                facecolors="lightsteelblue",
            )
            if ax is full and phase["duration_ms"] >= 1.5:
                ax.text(
                    phase["start_ms"] + phase["duration_ms"] / 2,
                    stage_y + 0.5,
                    phase["name"],
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        for tid, y in api_y.items():
            for kind, color in (
                ("enqueue", "#3182ce"),
                ("other", "#a0aec0"),
                ("sync", "#e53e3e"),
            ):
                spans = [
                    (event["start_ms"], event["duration_ms"])
                    for event in api_events
                    if event["global_tid"] == tid
                    and (
                        "sync"
                        if "Synchronize" in event["name"]
                        else "enqueue"
                        if "Launch" in event["name"] or "Async" in event["name"]
                        else "other"
                    )
                    == kind
                ]
                if spans:
                    ax.broken_barh(spans, (y + 0.1, 0.8), facecolors=color)
        for stream, y in gpu_y.items():
            for main_thread, color in ((True, "#008b8b"), (False, "#805ad5")):
                spans = [
                    (event["start_ms"], event["duration_ms"])
                    for event in gpu_events
                    if event["stream_id"] == stream
                    and (event["launch_thread_tid"] == detail["main_thread_tid"])
                    == main_thread
                ]
                if spans:
                    ax.broken_barh(spans, (y + 0.1, 0.8), facecolors=color)
        ax.set_yticks(
            [index + 0.5 for index in range(len(lane_labels))], labels=lane_labels
        )
        if focus:
            ax.axvspan(
                focus["start_ms"],
                focus["start_ms"] + focus["duration_ms"],
                color="#e53e3e",
                alpha=0.08,
            )
    full.set_xlim(
        0,
        max(
            [
                timeline["step_window"]["steps"][0]["duration_ms"],
                *(event["start_ms"] + event["duration_ms"] for event in gpu_events),
            ]
        ),
    )
    full.set_title(f"{timeline['step']}: CPU stages, CUDA APIs, and GPU execution")
    if focus:
        focus_end = focus["start_ms"] + focus["duration_ms"]
        zoom.set_xlim(max(0, focus["start_ms"] - 0.6), focus_end + 0.3)
        zoom.set_title(
            f"Blocking {focus['name']}: {focus['start_ms']:.3f}–"
            f"{focus_end:.3f} ms ({focus['duration_ms']:.3f} ms in API)"
        )
        calls_by_correlation = {event["correlation_id"]: event for event in api_events}
        linked = [
            event
            for event in gpu_events
            if event["correlation_id"] in calls_by_correlation
            and event["start_ms"] + event["duration_ms"] <= focus_end
            and calls_by_correlation[event["correlation_id"]]["start_ms"]
            < focus["start_ms"]
        ]
        if linked:
            last_gpu = max(
                linked, key=lambda event: event["start_ms"] + event["duration_ms"]
            )
            call = calls_by_correlation[last_gpu["correlation_id"]]
            gpu_start = last_gpu["start_ms"]
            call_end = call["start_ms"] + call["duration_ms"]
            if call_end <= gpu_start:
                zoom.annotate(
                    f"{call['name']} → GPU {last_gpu['kind']} "
                    f"(corr {last_gpu['correlation_id']})",
                    xy=(gpu_start, gpu_y[last_gpu["stream_id"]] + 0.5),
                    xytext=(call_end, api_y[call["global_tid"]] + 0.5),
                    arrowprops={"arrowstyle": "->", "color": "#2d3748", "lw": 1},
                    fontsize=8,
                )
            zoom.axvline(
                last_gpu["start_ms"] + last_gpu["duration_ms"],
                color="#008b8b",
                linestyle="--",
                linewidth=1,
            )
    else:
        zoom.set_xlim(0, min(5, timeline["step_window"]["steps"][0]["duration_ms"]))
        zoom.set_title("No CUDA synchronization API in this step; early launch detail")
    zoom.set_xlabel("Time from step NVTX start (ms)")
    reset = next(
        (phase for phase in timeline["phases"] if phase["name"] == "reset"), None
    )
    if reset:
        reset_end = reset["start_ms"] + reset["duration_ms"]
        reset_zoom.set_xlim(reset["start_ms"] - 0.25, reset_end + 0.25)
        reset_calls = [
            event
            for event in api_events
            if event["start_ms"] < reset_end
            and event["start_ms"] + event["duration_ms"] > reset["start_ms"]
        ]
        reset_gpu = [
            event
            for event in gpu_events
            if event["start_ms"] < reset_end
            and event["start_ms"] + event["duration_ms"] > reset["start_ms"]
        ]
        reset_zoom.axvspan(reset["start_ms"], reset_end, color="#d69e2e", alpha=0.15)
        reset_zoom.set_title(
            f"reset {reset['start_ms']:.3f}–{reset_end:.3f} ms: "
            f"{len(reset_calls)} overlapping CUDA API calls, "
            f"{len(reset_gpu)} overlapping GPU events"
        )
    else:
        reset_zoom.set_visible(False)
    reset_zoom.set_xlabel("Time from step NVTX start (ms)")
    fig.legend(
        handles=[
            Patch(color="#3182ce", label="CUDA enqueue API"),
            Patch(color="#a0aec0", label="Other CUDA API"),
            Patch(color="#e53e3e", label="CUDA synchronize API"),
            Patch(color="#008b8b", label="GPU: main-thread launch"),
            Patch(color="#805ad5", label="GPU: worker-thread launch"),
        ],
        loc="lower center",
        ncol=5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_dir / "event_timeline.png", dpi=180)
    plt.close(fig)

    graph_stages = timeline["graph_stages"]
    if graph_stages:
        lanes = {
            "graph_runner": (5, "#64748b"),
            "cudaGraphLaunch": (4, "#2563eb"),
            "forward": (3, "#16a34a"),
            "loss": (2, "#d97706"),
            "backward": (1, "#9333ea"),
            "optimizer": (0, "#dc2626"),
        }
        graph_launches = [
            event for event in api_events if "cudaGraphLaunch" in event["name"]
        ]
        fig, (overview, launch_zoom) = plt.subplots(2, 1, figsize=(15, 6))
        for ax in (overview, launch_zoom):
            for phase in timeline["phases"]:
                if phase["name"] in ("graph_runner", "optimizer"):
                    y, color = lanes[phase["name"]]
                    ax.broken_barh(
                        [(phase["start_ms"], phase["duration_ms"])],
                        (y + 0.1, 0.8),
                        facecolors=color,
                    )
            for event in graph_launches:
                y, color = lanes["cudaGraphLaunch"]
                ax.broken_barh(
                    [(event["start_ms"], event["duration_ms"])],
                    (y + 0.1, 0.8),
                    facecolors=color,
                )
            for stage in ("forward", "loss", "backward"):
                spans = [
                    (event["start_ms"], event["duration_ms"])
                    for event in graph_stages
                    if event["name"] == stage
                ]
                if spans:
                    y, color = lanes[stage]
                    ax.broken_barh(spans, (y + 0.1, 0.8), facecolors=color)
            ax.set_yticks(
                [value[0] + 0.5 for value in lanes.values()],
                labels=[
                    "CPU graph_runner",
                    "CPU cudaGraphLaunch API",
                    "GPU forward nodes",
                    "GPU loss nodes",
                    "GPU backward nodes",
                    "CPU optimizer (may wait)",
                ],
            )
            ax.grid(axis="x", alpha=0.2)
        step_span_ms = max(
            timeline["step_window"]["steps"][0]["duration_ms"],
            *(event["start_ms"] + event["duration_ms"] for event in graph_stages),
        )
        overview.set_xlim(0, step_span_ms)
        overview.set_title(
            "CUDA Graph replay: build-time NVTX projected onto GPU nodes"
        )
        launch_zoom.set_xlim(
            0,
            min(
                step_span_ms,
                max(
                    (e["start_ms"] + e["duration_ms"] for e in graph_launches),
                    default=0,
                )
                + 0.5,
            ),
        )
        launch_zoom.set_title("CPU graph launch and first GPU nodes")
        launch_zoom.set_xlabel("Time from captured step start (ms)")
        fig.tight_layout()
        fig.savefig(output_dir / "graph_stage_timeline.png", dpi=180)
        plt.close(fig)

    window = timeline["step_window"]["steps"]
    if len(window) == 4:
        phase_colors = {
            "forward": "#1f77b4",
            "loss": "#17becf",
            "backward": "#9467bd",
            "graph_runner": "#64748b",
            "optimizer": "#ff7f0e",
            "zero_grad": "#8c564b",
            "reset": "#d62728",
            "other": "#7f7f7f",
        }
        fig, (overview, zoom) = plt.subplots(
            2, 1, figsize=(15, 8), gridspec_kw={"height_ratios": [4, 1]}
        )
        for index, step in enumerate(window):
            cpu_y = 2 * (3 - index) + 1
            gpu_y = cpu_y - 1
            for phase in step["phases"]:
                overview.broken_barh(
                    [(phase["start_ms"], phase["duration_ms"])],
                    (cpu_y, 0.75),
                    facecolors=phase_colors[phase["name"]],
                    alpha=0.5,
                )
                if phase["duration_ms"] >= 1.5:
                    overview.text(
                        phase["start_ms"] + phase["duration_ms"] / 2,
                        cpu_y + 0.375,
                        phase["name"],
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                if phase["name"] == "reset":
                    overview.axvline(
                        phase["start_ms"],
                        color=phase_colors["reset"],
                        linewidth=0.8,
                        alpha=0.55,
                    )
            for phase, color in phase_colors.items():
                spans = [
                    (event["start_ms"], event["duration_ms"])
                    for event in step["gpu_events"]
                    if event["launch_phase"] == phase
                ]
                if spans:
                    overview.broken_barh(spans, (gpu_y, 0.75), facecolors=color)
            overview.axvspan(
                step["start_ms"],
                step["start_ms"] + step["duration_ms"],
                facecolor="black",
                alpha=0.025,
            )
        overview.set_yticks(
            [2 * (3 - i) + offset + 0.375 for i in range(4) for offset in (1, 0)],
            labels=[
                label
                for step in window
                for label in (
                    f"step {step['name'].rsplit(':', 1)[-1]} CPU",
                    f"step {step['name'].rsplit(':', 1)[-1]} GPU",
                )
            ],
        )
        overview.set_xlim(
            0,
            max(
                [
                    window[-1]["start_ms"] + window[-1]["duration_ms"],
                    *(
                        event["start_ms"] + event["duration_ms"]
                        for step in window
                        for event in step["gpu_events"]
                    ),
                ]
            ),
        )
        overview.set_xlabel("Time from first captured step (ms)")
        overview.set_title("Four captured steps: CPU NVTX stages and GPU events")

        first_reset = next(
            (phase for phase in window[0]["phases"] if phase["name"] == "reset"),
            None,
        )
        if first_reset:
            reset_start = first_reset["start_ms"]
            reset_end = reset_start + first_reset["duration_ms"]
            zoom.broken_barh(
                [(reset_start, first_reset["duration_ms"])],
                (1, 0.75),
                facecolors=phase_colors["reset"],
                alpha=0.5,
            )
            overlapping = [
                event
                for event in window[0]["gpu_events"]
                if event["start_ms"] < reset_end
                and event["start_ms"] + event["duration_ms"] > reset_start
            ]
            for phase, color in phase_colors.items():
                spans = [
                    (event["start_ms"], event["duration_ms"])
                    for event in window[0]["gpu_events"]
                    if event["launch_phase"] == phase
                    and event["start_ms"] < reset_end + 0.5
                    and event["start_ms"] + event["duration_ms"] > reset_start - 0.5
                ]
                if spans:
                    zoom.broken_barh(spans, (0, 0.75), facecolors=color)
            reset_launched = sum(
                event["launch_phase"] == "reset" for event in window[0]["gpu_events"]
            )
            zoom.set_xlim(reset_start - 0.5, reset_end + 0.5)
            zoom.set_title(
                f"First reset: {first_reset['duration_ms'] * 1000:.1f} µs; "
                f"{reset_launched} reset-launched GPU events, "
                f"{len(overlapping)} GPU events overlapping"
            )
            zoom.set_yticks([1.375, 0.375], labels=["reset CPU", "GPU events"])
            zoom.set_xlabel("Time from first captured step (ms)")
        else:
            zoom.set_visible(False)
        fig.tight_layout()
        fig.savefig(output_dir / "four_step_timeline.png", dpi=180)
        plt.close(fig)

    if report["gil"]["collected"]:
        gil_events = timeline["gil_events"]
        tids = sorted({event["global_tid"] for event in gil_events})
        fig, ax = plt.subplots(figsize=(12, max(3, len(tids) * 0.8 + 1.5)))
        for index, tid in enumerate(tids):
            for state, color in (
                ("Holding GIL", "tab:blue"),
                ("Waiting for GIL", "tab:orange"),
            ):
                spans = [
                    (event["start_ms"], event["duration_ms"])
                    for event in gil_events
                    if event["global_tid"] == tid and event["state"] == state
                ]
                if spans:
                    ax.broken_barh(spans, (index + 0.1, 0.8), facecolors=color)
        gpu_spans = [
            (event["start_ms"], event["duration_ms"])
            for event in timeline["gpu_events"]
        ]
        if gpu_spans:
            ax.broken_barh(gpu_spans, (len(tids) + 0.1, 0.8), facecolors="tab:green")
        ax.set_yticks(
            [index + 0.5 for index in range(len(tids) + 1)],
            labels=[*(f"GIL thread {tid & 0xFFFFFFFF}" for tid in tids), "GPU events"],
        )
        ax.set(xlabel="Time from step NVTX start (ms)", title=timeline["step"])
        ax.plot([], [], color="tab:blue", linewidth=6, label="Holding GIL")
        ax.plot([], [], color="tab:orange", linewidth=6, label="Waiting for GIL")
        ax.plot([], [], color="tab:green", linewidth=6, label="GPU event")
        fig.legend(loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(output_dir / "gil_timeline.png", dpi=160)
        plt.close(fig)

    modules = sorted(timeline["modules"], key=lambda item: -item["duration_ms"])[:20]
    if modules:
        ordered = sorted(modules, key=lambda item: item["start_ms"])
        fig, ax = plt.subplots(figsize=(12, max(4, len(modules) * 0.35)))
        for index, module in enumerate(ordered):
            ax.broken_barh(
                [(module["start_ms"], module["duration_ms"])],
                (index + 0.1, 0.8),
            )
        ax.set_yticks(
            [index + 0.5 for index in range(len(ordered))],
            labels=[
                item["name"] if len(item["name"]) <= 45 else "…" + item["name"][-44:]
                for item in ordered
            ],
            fontsize=7,
        )
        ax.set_xlabel("CPU NVTX time from step start (ms); top 20 module ranges")
        ax.set_title(timeline["step"])
        fig.tight_layout()
        fig.savefig(output_dir / "module_timeline.png", dpi=160)
        plt.close(fig)
