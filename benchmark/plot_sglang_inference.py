from __future__ import annotations

import argparse
import csv
from pathlib import Path

from plot_distributed_inference import PLOT_RC


def _load(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    numeric = (
        "tp",
        "pp",
        "dp",
        "requests",
        "input_tokens",
        "output_tokens",
        "shared_prefix_tokens",
        "input_tokens_per_second_median",
        "output_tokens_per_second_median",
        "ttft_p99_ms_median",
        "tpot_p99_ms_median",
        "peak_memory_mib",
    )
    for row in rows:
        for name in numeric:
            row[name] = float(row[name])
    if not rows or any(float(row[name]) < 0 for row in rows for name in numeric):
        raise ValueError("SGLang result CSV is empty or contains negative metrics.")
    return rows


def _one(rows: list[dict[str, object]], **values: object) -> dict[str, object]:
    matches = [
        row for row in rows if all(row[name] == value for name, value in values.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one SGLang row for {values}, got {len(matches)}.")
    return matches[0]


def plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    from matplotlib.ticker import FuncFormatter

    qwen = [row for row in rows if row["model"] == "Qwen2.5-0.5B-QCFS"]
    fixed = [
        _one(
            qwen,
            tp=1.0,
            pp=1.0,
            dp=1.0,
            input_tokens=128.0,
            output_tokens=128.0,
        ),
        _one(
            qwen,
            tp=1.0,
            pp=2.0,
            dp=1.0,
            input_tokens=128.0,
            output_tokens=128.0,
        ),
        _one(
            qwen,
            tp=1.0,
            pp=1.0,
            dp=4.0,
            input_tokens=128.0,
            output_tokens=128.0,
        ),
    ]
    prefix = [
        fixed[0],
        _one(
            qwen,
            tp=1.0,
            pp=1.0,
            dp=1.0,
            input_tokens=2176.0,
            shared_prefix_tokens=2048.0,
        ),
    ]

    with (
        plt.style.context(["science", "no-latex", "bright"]),
        plt.rc_context(PLOT_RC),
    ):
        figure, (scale_axis, prefix_axis) = plt.subplots(
            1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": (1.05, 1)}
        )

        throughputs = [float(row["output_tokens_per_second_median"]) for row in fixed]
        colors = ("C0", "C2", "C1")
        bars = scale_axis.bar(range(3), throughputs, color=colors, width=0.68)
        baseline = throughputs[0]
        for index, (bar, row) in enumerate(zip(bars, fixed, strict=True)):
            scale_axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(throughputs) * 0.035,
                f"{bar.get_height():.0f}\n{bar.get_height() / baseline:.2f}×",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
            scale_axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.48,
                f"p99 TPOT\n{float(row['tpot_p99_ms_median']):.1f} ms",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
                fontweight="bold",
            )
        scale_axis.set_xticks(
            range(3),
            ("TP1\n1 GPU · 32 req", "PP2\n2 GPU · 32 req", "DP4\n4 GPU · 128 req"),
        )
        scale_axis.set_ylim(0, max(throughputs) * 1.23)
        scale_axis.set_ylabel("Aggregate output throughput (tokens/s)")
        scale_axis.set_title("(a) Scale-out · fixed 128 input / 128 output", loc="left")

        positions = (0, 1)
        width = 0.32
        input_rates = [float(row["input_tokens_per_second_median"]) for row in prefix]
        output_rates = [float(row["output_tokens_per_second_median"]) for row in prefix]
        input_bars = prefix_axis.bar(
            [position - width / 2 for position in positions],
            input_rates,
            width,
            color="C4",
            label="Input tokens/s",
        )
        output_bars = prefix_axis.bar(
            [position + width / 2 for position in positions],
            output_rates,
            width,
            color="C0",
            label="Output tokens/s",
        )
        for bars_to_label in (input_bars, output_bars):
            for bar in bars_to_label:
                prefix_axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.1,
                    f"{bar.get_height():,.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        for position, row in zip(positions, prefix, strict=True):
            prefix_axis.text(
                position,
                0.04,
                f"p99 TTFT {float(row['ttft_p99_ms_median']):.1f} ms",
                transform=prefix_axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
            )
        prefix_axis.set_yscale("log")
        prefix_axis.set_ylim(500, 30000)
        prefix_axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:,.0f}")
        )
        prefix_axis.set_xticks(
            positions, ("128 input\nno sharing", "2176 input\n2048 shared")
        )
        prefix_axis.set_ylabel("Median throughput (tokens/s, log scale)")
        prefix_axis.set_title("(b) Radix reuse · TP1 shared prefix", loc="left")
        prefix_axis.legend(loc="upper left", frameon=False)
        prefix_axis.text(
            positions[1] - width / 2,
            4000,
            f"{input_rates[1] / input_rates[0]:.1f}× input\nthroughput",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
            fontweight="bold",
        )

        for axis in (scale_axis, prefix_axis):
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="both", which="both", top=False, right=False)
            axis.grid(axis="y", color="0.88", linewidth=0.6, linestyle="--")
            axis.set_axisbelow(True)

        figure.suptitle(
            "Qwen2.5-0.5B QCFS · SGLang offline inference",
            x=0.08,
            ha="left",
            fontweight="bold",
            fontsize=11,
        )
        figure.text(
            0.08,
            0.005,
            "BF16 eager · 4× RTX 4090 PCIe host · median of 3 cache-flushed runs",
            fontsize=7,
            color="0.35",
        )
        figure.tight_layout(rect=(0, 0.04, 1, 0.94), w_pad=2.2)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SGLang inference results")
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(_load(args.results), args.output)


if __name__ == "__main__":
    main()
