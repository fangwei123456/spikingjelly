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
    r"b(?P<batch>\d+)_r\d+_optimized(?P<protocol>_k4)?"
)
VISION_PROBE_NAME = re.compile(
    r"(?P<model>sew-resnet34|spikformer)_"
    r"(?P<topology>serial|dp4|fsdp4|tp4|pp4)_"
    r"b(?P<batch>\d+)_probe"
)
MCORE_NAME = re.compile(
    r"mcore_(?P<topology>tp1|dp4|tp2|pp2|pp4)_b(?P<batch>\d+)_r\d+"
    r"(?:_k(?P<microbatches>\d+))?"
)
TOPOLOGY_ORDER = ("serial", "dp4", "fsdp4", "tp4", "pp4")
TOPOLOGY_LABELS = {
    "serial": "Single GPU",
    "dp4": "DP4",
    "fsdp4": "FSDP4",
    "tp4": "TP4",
    "pp4": "PP4",
    "tp1": "Single GPU",
    "tp2": "TP2",
    "pp2": "PP2",
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


def _vision_topology(topology: str) -> tuple[int, int, int, int]:
    data_size = 4 if topology in {"dp4", "fsdp4"} else 1
    tensor_size = 4 if topology == "tp4" else 1
    pipeline_size = 4 if topology == "pp4" else 1
    microbatches = 4 if pipeline_size > 1 else 1
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


def _classify_runs(runs):
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
    return status, completed, status_codes


def _load_vision(results: Path) -> list[dict]:
    groups = defaultdict(list)
    for log in (results / "vision").glob("*.log"):
        match = VISION_NAME.fullmatch(log.stem)
        if match is None:
            continue
        if match["topology"] == "pp4" and match["protocol"] is None:
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
        status, completed, status_codes = _classify_runs(runs)
        data_parallel_size, tensor_size, pipeline_size, microbatches = _vision_topology(
            topology
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
                "five warmup and ten measured batches; "
                f"exit statuses: {','.join(map(str, sorted(status_codes)))}"
            ),
        }
        if len(completed) == 3:
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
        if match is None or match["topology"] == "pp4":
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
            topology
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


def _load_mcore(results: Path) -> list[dict]:
    groups = defaultdict(list)
    for log in (results / "mcore").glob("*.log"):
        match = MCORE_NAME.fullmatch(log.stem)
        if match is None:
            continue
        topology = match["topology"]
        if topology in {"tp2", "pp2", "pp4"} and match["microbatches"] is None:
            continue
        status_path = Path(f"{log}.status")
        exit_status = int(status_path.read_text().strip())
        text = log.read_text(encoding="utf-8")
        metrics = _last_json(log) if exit_status == 0 else None
        groups[
            (
                topology,
                int(match["batch"]),
                int(match["microbatches"] or 1),
            )
        ].append((exit_status, metrics, text))

    rows = []
    for (topology, batch_size, pipeline_microbatches), runs in sorted(groups.items()):
        status, completed, _ = _classify_runs(runs)
        data_parallel_size = 4 if topology == "dp4" else 1
        tensor_parallel_size = 2 if topology == "tp2" else 1
        pipeline_parallel_size = {"pp2": 2, "pp4": 4}.get(topology, 1)
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
                + f"MCore micro batch {batch_size // pipeline_microbatches}; "
                + (
                    "one warmup"
                    if topology == "pp4" and batch_size == 3072
                    else "five warmups"
                )
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
        stable_rows = [row for row in model_rows if row["status"] == "completed"]
        if not stable_rows:
            continue
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
            axis.set_title(
                f"{model} · inference batch sweep",
                loc="left",
                fontweight="bold",
                pad=8,
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="both", which="both", top=False, right=False)
            axis.grid(which="major", color="0.86", linewidth=0.6, linestyle="--")
            axis.grid(which="minor", color="0.93", linewidth=0.4, linestyle=":")
            axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
            figure.savefig(output / filename, dpi=300, bbox_inches="tight")
            plt.close(figure)


def _plot_mcore(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    markers = ("o", "s", "^", "D", "P")
    linestyles = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))
    with (
        plt.style.context(["science", "no-latex", "bright"]),
        plt.rc_context(PLOT_RC),
    ):
        figure, axis = plt.subplots(figsize=(6.1, 3.6))
        for index, topology in enumerate(("tp1", "dp4", "tp2", "pp2", "pp4")):
            points = sorted(
                (
                    row
                    for row in rows
                    if row["topology"] == topology and row["status"] == "completed"
                ),
                key=lambda row: row["peak_memory_gib_median"],
            )
            if not points:
                continue
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
                    f"{max(point['global_batch_size'] for point in points)}"
                ),
            )
        axis.set_title(
            "Qwen2.5-0.5B QCFS · MCore evaluation batch sweep",
            loc="left",
            fontweight="bold",
        )
        axis.set_xlabel("Peak allocated memory / GPU (GiB)")
        axis.set_ylabel("Aggregate evaluation throughput (tokens/s)")
        axis.set_ylim(bottom=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", which="both", top=False, right=False)
        axis.grid(color="0.88", linewidth=0.6, linestyle="--")
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        figure.tight_layout()
        figure.savefig(output / "mcore-inference.png", dpi=300, bbox_inches="tight")
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
        mcore_rows = [row for row in rows if row["backend"] == "mcore"]
    else:
        vision_rows = _load_vision(args.results)
        mcore_rows = _load_mcore(args.results)
        rows = vision_rows + mcore_rows
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
    _plot_mcore(mcore_rows, args.output)


if __name__ == "__main__":
    main()
