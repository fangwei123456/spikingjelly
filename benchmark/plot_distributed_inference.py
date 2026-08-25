from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

VISION_NAME = re.compile(
    r"(?P<model>sew-resnet34|spikformer)_"
    r"(?P<topology>serial|dp4|fsdp4|tp4|pp4)_"
    r"b(?P<batch>\d+)_r\d+_optimized"
)
VISION_PROBE_NAME = re.compile(
    r"(?P<model>sew-resnet34|spikformer)_"
    r"(?P<topology>serial|dp4|fsdp4|tp4|pp4)_"
    r"b(?P<batch>\d+)_probe"
)
MCORE_NAME = re.compile(r"mcore_(?P<topology>tp1|dp4|tp2|pp2|pp4)_b(?P<batch>\d+)_r\d+")
TOPOLOGY_ORDER = ("serial", "dp4", "fsdp4", "tp4", "pp4")
TOPOLOGY_LABELS = {
    "serial": "Single GPU",
    "dp4": "DP4",
    "fsdp4": "FSDP4",
    "tp4": "TP4",
    "pp4": "PP4",
    "tp1": "TP1",
    "dp2": "DP2",
    "tp2": "TP2",
    "pp2": "PP2",
    "dp2tp2": "DP2 + TP2",
}
CSV_FIELDS = (
    "workload",
    "backend",
    "model",
    "topology",
    "gpus",
    "data_parallel_size",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "per_rank_batch_size",
    "global_batch_size",
    "pipeline_microbatches",
    "pipeline_microbatch_size",
    "repeats",
    "successful_repeats",
    "status",
    "throughput_unit",
    "throughput_median",
    "throughput_min",
    "throughput_max",
    "memory_metric",
    "peak_memory_gib_median",
    "peak_memory_gib_min",
    "peak_memory_gib_max",
    "notes",
)
PLOT_RC = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
}


def _last_json(path: Path) -> dict:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"No JSON result found in {path}.")


def _summary(values: list[float]) -> tuple[float, float, float]:
    return median(values), min(values), max(values)


def _vision_topology(topology: str, batch_size: int) -> tuple[int, int, int, int]:
    data_size = 4 if topology in {"dp4", "fsdp4"} else 1
    tensor_size = 4 if topology == "tp4" else 1
    pipeline_size = 4 if topology == "pp4" else 1
    microbatches = max(4, batch_size // 16) if pipeline_size > 1 else 1
    return data_size, tensor_size, pipeline_size, microbatches


def _load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for name in (
            "gpus",
            "data_parallel_size",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "per_rank_batch_size",
            "global_batch_size",
            "repeats",
            "successful_repeats",
        ):
            row[name] = int(row[name])
        if row.get("pipeline_microbatches"):
            row["pipeline_microbatches"] = int(row["pipeline_microbatches"])
        if row.get("pipeline_microbatch_size"):
            row["pipeline_microbatch_size"] = int(row["pipeline_microbatch_size"])
        for name in (
            "throughput_median",
            "throughput_min",
            "throughput_max",
            "peak_memory_gib_median",
            "peak_memory_gib_min",
            "peak_memory_gib_max",
        ):
            if row[name]:
                row[name] = float(row[name])
    return rows


def _load_vision(results: Path) -> list[dict]:
    groups = defaultdict(list)
    for log in (results / "vision").glob("*.log"):
        match = VISION_NAME.fullmatch(log.stem)
        if match is None:
            continue
        status = int(Path(f"{log}.status").read_text().strip())
        text = log.read_text(encoding="utf-8")
        metrics = (
            _last_json(log)
            if any(line.startswith("{") for line in text.splitlines())
            else None
        )
        groups[(match["model"], match["topology"], int(match["batch"]))].append(
            (status, metrics, text)
        )

    rows = []
    for (model, topology, batch_size), runs in sorted(groups.items()):
        completed = [metrics for status, metrics, _ in runs if status == 0]
        status_codes = [status for status, _, _ in runs]
        if len(completed) == 3:
            status = "completed"
        elif completed:
            status = "unstable"
        elif any(
            "OutOfMemoryError" in text or "CUDA out of memory" in text
            for _, _, text in runs
        ):
            status = "cuda_oom"
        elif 124 in status_codes:
            status = "timeout"
        else:
            status = "failed"
        data_parallel_size, tensor_size, pipeline_size, microbatches = _vision_topology(
            topology, batch_size
        )
        row = {
            "workload": "vision_evaluation",
            "backend": "pytorch",
            "model": "SEW-ResNet34" if model == "sew-resnet34" else "Spikformer-S",
            "topology": topology,
            "gpus": data_parallel_size * tensor_size * pipeline_size,
            "data_parallel_size": data_parallel_size,
            "tensor_parallel_size": tensor_size,
            "pipeline_parallel_size": pipeline_size,
            "per_rank_batch_size": batch_size,
            "global_batch_size": batch_size * data_parallel_size,
            "pipeline_microbatches": microbatches,
            "pipeline_microbatch_size": batch_size // microbatches,
            "repeats": len(runs),
            "successful_repeats": len(completed),
            "status": status,
            "throughput_unit": "images/s",
            "memory_metric": "cuda_peak_allocated",
            "notes": (
                (
                    "one measured batch; no warmup; "
                    if topology == "pp4" and batch_size >= 4096
                    else "five warmup and ten measured batches; "
                )
                + f"exit statuses: {','.join(map(str, sorted(status_codes)))}"
            ),
        }
        if status == "completed":
            throughput = _summary(
                [float(metrics["images_per_second"]) for metrics in completed]
            )
            memory = _summary(
                [float(metrics["peak_memory_bytes"]) / 1024**3 for metrics in completed]
            )
            row.update(
                {
                    "throughput_median": throughput[0],
                    "throughput_min": throughput[1],
                    "throughput_max": throughput[2],
                    "peak_memory_gib_median": memory[0],
                    "peak_memory_gib_min": memory[1],
                    "peak_memory_gib_max": memory[2],
                }
            )
        rows.append(row)
    for log in (results / "vision").glob("*_probe.log"):
        match = VISION_PROBE_NAME.fullmatch(log.stem)
        if match is None:
            continue
        exit_status = int(Path(f"{log}.status").read_text().strip())
        text = log.read_text(encoding="utf-8")
        metrics = (
            _last_json(log)
            if any(line.startswith("{") for line in text.splitlines())
            else None
        )
        if exit_status == 0:
            status = "capacity_probe_completed"
        elif "OutOfMemoryError" in text or "CUDA out of memory" in text:
            status = "cuda_oom"
        elif exit_status == 124:
            status = "timeout"
        else:
            status = "failed"
        topology = match["topology"]
        batch_size = int(match["batch"])
        data_parallel_size, tensor_size, pipeline_size, microbatches = _vision_topology(
            topology, batch_size
        )
        row = {
            "workload": "vision_capacity_probe",
            "backend": "pytorch",
            "model": "SEW-ResNet34"
            if match["model"] == "sew-resnet34"
            else "Spikformer-S",
            "topology": topology,
            "gpus": data_parallel_size * tensor_size * pipeline_size,
            "data_parallel_size": data_parallel_size,
            "tensor_parallel_size": tensor_size,
            "pipeline_parallel_size": pipeline_size,
            "per_rank_batch_size": batch_size,
            "global_batch_size": batch_size * data_parallel_size,
            "pipeline_microbatches": microbatches,
            "pipeline_microbatch_size": batch_size // microbatches,
            "repeats": 1,
            "successful_repeats": int(exit_status == 0),
            "status": status,
            "throughput_unit": "",
            "memory_metric": "cuda_peak_allocated",
            "notes": "single-batch constant-input capacity probe",
        }
        if metrics is not None:
            memory = float(metrics["peak_memory_bytes"]) / 1024**3
            row.update(
                {
                    "peak_memory_gib_median": memory,
                    "peak_memory_gib_min": memory,
                    "peak_memory_gib_max": memory,
                }
            )
        rows.append(row)
    return rows


def _load_sglang(results: Path) -> list[dict]:
    rows = []
    for topology in ("tp1", "dp2", "dp4", "tp2", "pp2", "pp4", "dp2tp2"):
        stable = sorted(results.glob(f"sglang_{topology}_stable*.json"))
        if stable:
            datasets = []
            for path in stable:
                try:
                    datasets.append(_last_json(path))
                except ValueError:
                    continue
        else:
            continue
        measurements = {
            measurement["prompt_count"]: (data, measurement)
            for data in datasets
            for measurement in data["measurements"]
        }
        for data, measurement in (
            measurements[prompt_count] for prompt_count in sorted(measurements)
        ):
            generated_tokens = sum(len(output) for output in measurement["outputs"])
            throughputs = [
                generated_tokens / elapsed
                for elapsed in measurement["inference_seconds_samples"]
            ]
            throughput = _summary(throughputs)
            memory = measurement["peak_device_memory_bytes"] / 1024**3
            status = (
                "completed"
                if len(throughputs) == 3 and throughput[2] / throughput[1] <= 1.3
                else "unstable"
            )
            rows.append(
                {
                    "workload": "llm_generation",
                    "backend": "sglang",
                    "model": "Qwen2.5-0.5B-qcfs",
                    "topology": topology,
                    "gpus": data["tensor_parallel_size"]
                    * data["pipeline_parallel_size"]
                    * data["data_parallel_size"],
                    "data_parallel_size": data["data_parallel_size"],
                    "tensor_parallel_size": data["tensor_parallel_size"],
                    "pipeline_parallel_size": data["pipeline_parallel_size"],
                    "per_rank_batch_size": (
                        measurement["prompt_count"] + data["data_parallel_size"] - 1
                    )
                    // data["data_parallel_size"],
                    "global_batch_size": measurement["prompt_count"],
                    "pipeline_microbatches": "",
                    "pipeline_microbatch_size": "",
                    "repeats": len(throughputs),
                    "successful_repeats": len(throughputs),
                    "status": status,
                    "throughput_unit": "generated_tokens/s",
                    "throughput_median": throughput[0],
                    "throughput_min": throughput[1],
                    "throughput_max": throughput[2],
                    "memory_metric": "nvml_device_used",
                    "peak_memory_gib_median": memory,
                    "peak_memory_gib_min": memory,
                    "peak_memory_gib_max": memory,
                    "notes": "Radix cache disabled; static memory fraction 0.5",
                }
            )
    return rows


def _load_mcore(results: Path) -> list[dict]:
    groups = defaultdict(list)
    for log in (results / "mcore").glob("*.log"):
        match = MCORE_NAME.fullmatch(log.stem)
        if match is None:
            continue
        status_path = Path(f"{log}.status")
        if not status_path.is_file():
            continue
        exit_status = int(status_path.read_text().strip().split(":")[-1])
        text = log.read_text(encoding="utf-8")
        metrics = _last_json(log) if exit_status == 0 else None
        groups[(match["topology"], int(match["batch"]))].append(
            (exit_status, metrics, text)
        )

    rows = []
    for (topology, batch_size), runs in sorted(groups.items()):
        completed = [metrics for exit_status, metrics, _ in runs if exit_status == 0]
        if len(completed) == 3:
            status = "completed"
        elif completed:
            status = "unstable"
        elif any(
            "OutOfMemoryError" in text or "CUDA out of memory" in text
            for _, _, text in runs
        ):
            status = "cuda_oom"
        elif any(exit_status == 124 for exit_status, _, _ in runs):
            status = "timeout"
        else:
            status = "failed"
        data_parallel_size = 4 if topology == "dp4" else 1
        tensor_parallel_size = 2 if topology == "tp2" else 1
        pipeline_parallel_size = {"pp2": 2, "pp4": 4}.get(topology, 1)
        if completed:
            pipeline_microbatches = int(completed[0]["pipeline_microbatches"])
        elif topology in {"pp2", "pp4"} and batch_size > 16:
            pipeline_microbatches = batch_size // 16
        else:
            pipeline_microbatches = 1
        row = {
            "workload": "llm_evaluation",
            "backend": "mcore",
            "model": "Qwen2.5-0.5B-qcfs",
            "topology": topology,
            "gpus": data_parallel_size * tensor_parallel_size * pipeline_parallel_size,
            "data_parallel_size": data_parallel_size,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "per_rank_batch_size": batch_size,
            "global_batch_size": batch_size * data_parallel_size,
            "pipeline_microbatches": pipeline_microbatches,
            "pipeline_microbatch_size": batch_size // pipeline_microbatches,
            "repeats": len(runs),
            "successful_repeats": len(completed),
            "status": status,
            "throughput_unit": "semantic_tokens/s",
            "memory_metric": "cuda_peak_allocated",
            "notes": (
                "BF16; T=2; sequence length 16; "
                + (
                    f"{int(float(completed[0]['valid_tokens']) / 16)} samples; "
                    if completed
                    else "capacity candidate; "
                )
                + f"MCore micro batch {batch_size // pipeline_microbatches}"
            ),
        }
        if completed:
            throughput = _summary(
                [float(metrics["semantic_tokens_per_second"]) for metrics in completed]
            )
            memory = _summary(
                [float(metrics["peak_memory_bytes"]) / 1024**3 for metrics in completed]
            )
            row.update(
                {
                    "throughput_median": throughput[0],
                    "throughput_min": throughput[1],
                    "throughput_max": throughput[2],
                    "peak_memory_gib_median": memory[0],
                    "peak_memory_gib_min": memory[1],
                    "peak_memory_gib_max": memory[2],
                }
            )
        rows.append(row)
    return rows


def _pareto_frontier(
    points: list[dict], memory_resolution_gib: float = 0.05
) -> list[dict]:
    memory_bins = {}
    for point in points:
        memory_bin = round(point["peak_memory_gib_median"] / memory_resolution_gib)
        current = memory_bins.get(memory_bin)
        if current is None or point["throughput_median"] > current["throughput_median"]:
            memory_bins[memory_bin] = point
    frontier = []
    best_throughput = -1.0
    for point in sorted(
        memory_bins.values(),
        key=lambda row: (
            row["peak_memory_gib_median"],
            -row["throughput_median"],
        ),
    ):
        if point["throughput_median"] > best_throughput:
            frontier.append(point)
            best_throughput = point["throughput_median"]
    return frontier


def _error(points: list[dict]) -> tuple[list[float], list[float]]:
    return (
        [point["throughput_median"] - point["throughput_min"] for point in points],
        [point["throughput_max"] - point["throughput_median"] for point in points],
    )


def _plot_vision(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    from matplotlib.ticker import FuncFormatter, LogLocator

    markers = ("o", "s", "^", "D", "P")
    linestyles = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))
    for model, filename in (
        ("SEW-ResNet34", "sew-resnet34-inference-tradeoff.png"),
        ("Spikformer-S", "spikformer-inference-tradeoff.png"),
    ):
        model_rows = [row for row in rows if row["model"] == model]
        with (
            plt.style.context(["science", "no-latex", "bright"]),
            plt.rc_context(PLOT_RC),
        ):
            figure, axis = plt.subplots(figsize=(6.1, 3.6))
            for index, topology in enumerate(TOPOLOGY_ORDER):
                points = sorted(
                    (
                        row
                        for row in model_rows
                        if row["topology"] == topology and row["status"] == "completed"
                    ),
                    key=lambda row: row["global_batch_size"],
                )
                lines = axis.errorbar(
                    [point["peak_memory_gib_median"] for point in points],
                    [point["throughput_median"] for point in points],
                    yerr=_error(points),
                    marker=markers[index],
                    linestyle=linestyles[index],
                    capsize=2,
                    elinewidth=0.7,
                    label=TOPOLOGY_LABELS[topology],
                )
                for point in points[-1:]:
                    label_above = topology in {"dp4", "tp4", "pp4"}
                    axis.annotate(
                        f"G={point['global_batch_size']}",
                        (
                            point["peak_memory_gib_median"],
                            point["throughput_median"],
                        ),
                        xytext=(3, 4 if label_above else -6),
                        textcoords="offset points",
                        fontsize=6.5,
                        color=lines.lines[0].get_color(),
                        verticalalignment="bottom" if label_above else "top",
                    )
            stable_rows = [row for row in model_rows if row["status"] == "completed"]
            min_memory = min(row["peak_memory_gib_median"] for row in stable_rows)
            max_memory = max(row["peak_memory_gib_median"] for row in stable_rows)
            min_throughput = min(row["throughput_median"] for row in stable_rows)
            max_throughput = max(row["throughput_median"] for row in stable_rows)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xlim(min_memory / 1.25, max_memory * 1.3)
            axis.set_ylim(min_throughput / 1.3, max_throughput * 1.35)
            axis.xaxis.set_major_locator(LogLocator(base=2))
            axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            formatter = FuncFormatter(lambda value, _: f"{value:g}")
            axis.xaxis.set_major_formatter(formatter)
            axis.yaxis.set_major_formatter(formatter)
            axis.set_xlabel("Peak allocated memory / GPU (GiB)")
            axis.set_ylabel("Aggregate inference throughput (images/s)")
            axis.set_title(f"{model} · inference", loc="left", fontweight="bold", pad=8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="both", which="both", top=False, right=False)
            axis.grid(which="major", color="0.86", linewidth=0.6, linestyle="--")
            axis.grid(which="minor", color="0.93", linewidth=0.4, linestyle=":")
            axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
            figure.savefig(output / filename, dpi=300, bbox_inches="tight")
            plt.close(figure)


def _plot_llm(
    rows: list[dict],
    output: Path,
    topology_order: tuple[str, ...],
    title: str,
    x_label: str,
    y_label: str,
    filename: str,
) -> None:
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    markers = ("o", "s", "^", "D", "P", "X", "v")
    linestyles = (
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
        (0, (5, 2)),
        (0, (1, 1)),
    )
    with (
        plt.style.context(["science", "no-latex", "bright"]),
        plt.rc_context(PLOT_RC),
    ):
        figure, axis = plt.subplots(figsize=(6.1, 3.6))
        for index, topology in enumerate(topology_order):
            measured = [
                row
                for row in rows
                if row["topology"] == topology and row["status"] == "completed"
            ]
            if not measured:
                continue
            points = _pareto_frontier(measured)
            color = f"C{index}"
            axis.errorbar(
                [point["peak_memory_gib_median"] for point in points],
                [point["throughput_median"] for point in points],
                yerr=_error(points),
                marker=markers[index],
                linestyle=linestyles[index],
                capsize=2,
                color=color,
                label=(
                    f"{TOPOLOGY_LABELS[topology]} · max G="
                    f"{max(point['global_batch_size'] for point in measured)}"
                ),
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_ylim(bottom=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", which="both", top=False, right=False)
        axis.grid(color="0.88", linewidth=0.6, linestyle="--")
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.results.is_file():
        rows = _load_csv(args.results)
        vision_rows = [row for row in rows if row["backend"] == "pytorch"]
        sglang_rows = [row for row in rows if row["backend"] == "sglang"]
        mcore_rows = [row for row in rows if row["backend"] == "mcore"]
    else:
        vision_rows = _load_vision(args.results)
        sglang_rows = _load_sglang(args.results)
        mcore_rows = _load_mcore(args.results)
        rows = vision_rows + sglang_rows + mcore_rows
        for row in rows:
            if row["global_batch_size"] != (
                row["per_rank_batch_size"] * row["data_parallel_size"]
            ):
                raise ValueError(
                    f"Inconsistent L/G batch sizes in {row['backend']} "
                    f"{row['topology']}."
                )
            if row["gpus"] != (
                row["data_parallel_size"]
                * row["tensor_parallel_size"]
                * row["pipeline_parallel_size"]
            ):
                raise ValueError(f"Inconsistent topology in {row['topology']}.")
        with (args.output / "distributed-inference-tradeoff.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    _plot_vision(vision_rows, args.output)
    _plot_llm(
        sglang_rows,
        args.output,
        ("tp1", "dp2", "dp4", "tp2", "pp2", "pp4", "dp2tp2"),
        "Qwen2.5-0.5B QCFS · SGLang frontier",
        "Device memory used / GPU (GiB, NVML)",
        "Aggregate generation throughput (tokens/s)",
        "sglang-inference.png",
    )
    _plot_llm(
        mcore_rows,
        args.output,
        ("tp1", "dp4", "tp2", "pp2", "pp4"),
        "Qwen2.5-0.5B QCFS · MCore evaluation frontier",
        "Peak allocated memory / GPU (GiB)",
        "Aggregate evaluation throughput (tokens/s)",
        "mcore-inference.png",
    )


if __name__ == "__main__":
    main()
