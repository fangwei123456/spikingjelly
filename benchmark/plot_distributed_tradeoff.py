from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from matplotlib.ticker import FuncFormatter, LogLocator


VISION_NAME = re.compile(
    r"(?P<model>sew-resnet34|spikformer)_"
    r"(?P<topology>serial|dp4|fsdp4|tp4|pp4)_g(?P<global>\d+)_m(?P<micro>\d+)_r\d+"
)
LLM_NAME = re.compile(
    r"(?P<topology>dp2|dp4|tp4|pp4|cp4)_"
    r"g(?P<global>\d+)_m(?P<micro>\d+)_r\d+"
)
GPU_COUNTS = {"serial": 1, "dp2": 2}
GPU_COUNTS.update({name: 4 for name in ("dp4", "fsdp4", "tp4", "pp4", "cp4")})


def _vision_metrics(path: Path) -> dict[str, float]:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"No metrics JSON found in {path}")


def _load_runs(results: Path) -> list[dict[str, float | int | str]]:
    runs = []
    for path in (results / "vision").glob("*.log"):
        match = VISION_NAME.fullmatch(path.stem)
        if match is None:
            continue
        status = Path(f"{path}.status")
        if status.exists() and status.read_text().strip() != "0":
            continue
        metrics = _vision_metrics(path)
        runs.append(
            {
                "model": match["model"],
                "topology": match["topology"],
                "gpus": GPU_COUNTS[match["topology"]],
                "global_batch_size": int(match["global"]),
                "micro_batch_size": int(match["micro"]),
                "throughput": metrics["images_per_second"],
                "peak_memory_gib": metrics["peak_memory_bytes"] / 1024**3,
            }
        )
    for path in (results / "llm").glob("*/metrics.json"):
        match = LLM_NAME.fullmatch(path.parent.name)
        if match is None:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        runs.append(
            {
                "model": "spikelm-1.41b",
                "topology": match["topology"],
                "gpus": GPU_COUNTS[match["topology"]],
                "global_batch_size": int(match["global"]),
                "micro_batch_size": int(match["micro"]),
                "throughput": metrics["semantic_tokens_per_second"],
                "peak_memory_gib": metrics["peak_memory_bytes"] / 1024**3,
            }
        )
    return runs


def _aggregate(runs: list[dict[str, float | int | str]]) -> list[dict]:
    groups = defaultdict(list)
    for run in runs:
        key = tuple(
            run[name]
            for name in (
                "model",
                "topology",
                "gpus",
                "global_batch_size",
                "micro_batch_size",
            )
        )
        groups[key].append(run)

    rows = []
    for key, group in groups.items():
        if len(group) != 3:
            raise ValueError(f"Expected three repeats for {key}, found {len(group)}")
        throughput = [float(run["throughput"]) for run in group]
        memory = [float(run["peak_memory_gib"]) for run in group]
        rows.append(
            dict(
                zip(
                    (
                        "model",
                        "topology",
                        "gpus",
                        "global_batch_size",
                        "micro_batch_size",
                    ),
                    key,
                    strict=True,
                )
            )
            | {
                "repeats": len(group),
                "throughput_median": median(throughput),
                "throughput_min": min(throughput),
                "throughput_max": max(throughput),
                "peak_memory_gib_median": median(memory),
                "peak_memory_gib_min": min(memory),
                "peak_memory_gib_max": max(memory),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["model"],
            row["topology"],
            row["global_batch_size"],
        ),
    )


def _plot(rows: list[dict], model: str, output: Path) -> None:
    topology_order = {
        "sew-resnet34": ("serial", "dp4", "fsdp4", "tp4", "pp4"),
        "spikformer": ("serial", "dp4", "fsdp4", "tp4", "pp4"),
        "spikelm-1.41b": ("dp2", "dp4", "tp4", "pp4", "cp4"),
    }[model]
    labels = {
        "serial": "Single GPU",
        "dp2": "DP2",
        "dp4": "DP4",
        "fsdp4": "FSDP4",
        "tp4": "TP4",
        "pp4": "PP4",
        "cp4": "CP4",
    }
    markers = ("o", "s", "^", "D", "P")
    linestyles = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))

    with (
        plt.style.context(["science", "no-latex", "bright"]),
        plt.rc_context(
            {
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
        ),
    ):
        fig, ax = plt.subplots(figsize=(6.1, 3.6))
        for index, topology in enumerate(topology_order):
            points = sorted(
                (
                    row
                    for row in rows
                    if row["model"] == model and row["topology"] == topology
                ),
                key=lambda row: row["global_batch_size"],
            )
            memory = [row["peak_memory_gib_median"] for row in points]
            throughput = [row["throughput_median"] for row in points]
            lines = ax.errorbar(
                memory,
                throughput,
                yerr=(
                    [
                        row["throughput_median"] - row["throughput_min"]
                        for row in points
                    ],
                    [
                        row["throughput_max"] - row["throughput_median"]
                        for row in points
                    ],
                ),
                label=labels[topology],
                marker=markers[index],
                linestyle=linestyles[index],
                capsize=2,
                elinewidth=0.7,
            )
            for row in points[-1:]:
                label_above = topology in {"dp4", "tp4", "pp4"}
                ax.annotate(
                    f"G={row['global_batch_size']}",
                    (row["peak_memory_gib_median"], row["throughput_median"]),
                    xytext=(3, 4 if label_above else -6),
                    textcoords="offset points",
                    fontsize=6.5,
                    color=lines.lines[0].get_color(),
                    verticalalignment="bottom" if label_above else "top",
                )

        unit = "images/s" if model != "spikelm-1.41b" else "semantic tokens/s"
        titles = {
            "sew-resnet34": "SEW-ResNet34",
            "spikformer": "Spikformer-S",
            "spikelm-1.41b": "SpikeLM 1.41B · no accumulation · memopt off",
        }
        model_rows = [row for row in rows if row["model"] == model]
        min_memory = min(row["peak_memory_gib_median"] for row in model_rows)
        max_memory = max(row["peak_memory_gib_median"] for row in model_rows)
        min_throughput = min(row["throughput_median"] for row in model_rows)
        max_throughput = max(row["throughput_median"] for row in model_rows)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(min_memory / 1.25, max_memory * 1.3)
        ax.set_ylim(min_throughput / 1.3, max_throughput * 1.35)
        ax.xaxis.set_major_locator(LogLocator(base=2))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        formatter = FuncFormatter(lambda value, _: f"{value:g}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel("Peak allocated memory / GPU (GiB)")
        ax.set_ylabel(f"Aggregate training throughput ({unit})")
        ax.set_title(titles[model], loc="left", fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.grid(which="major", color="0.86", linewidth=0.6, linestyle="--")
        ax.grid(which="minor", color="0.93", linewidth=0.4, linestyle=":")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.savefig(output / f"{model}-tradeoff.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = _aggregate(_load_runs(args.results))
    with (args.output / "distributed-tradeoff.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for model in ("sew-resnet34", "spikformer", "spikelm-1.41b"):
        _plot(rows, model, args.output)


if __name__ == "__main__":
    main()
