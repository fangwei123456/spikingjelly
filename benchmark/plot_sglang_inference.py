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
    spikelm = [row for row in rows if row["model"] == "SpikeLM-2.78B"]
    pipeline = [
        [
            _one(
                spikelm,
                tp=1.0,
                pp=parallel_size,
                dp=1.0,
                requests=requests,
                input_tokens=64.0,
                output_tokens=64.0,
            )
            for parallel_size in (1.0, 4.0)
        ]
        for requests in (32.0, 64.0)
    ]
    data_parallel = [
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
            pp=1.0,
            dp=4.0,
            input_tokens=128.0,
            output_tokens=128.0,
        ),
    ]
    prefix = [
        data_parallel[0],
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
        figure, (pipeline_axis, data_axis, prefix_axis) = plt.subplots(
            1,
            3,
            figsize=(10.5, 3.25),
            gridspec_kw={"width_ratios": (1.15, 0.8, 1)},
        )

        positions = (0, 1)
        width = 0.34
        single_rates = [
            float(group[0]["output_tokens_per_second_median"]) for group in pipeline
        ]
        pipeline_rates = [
            float(group[1]["output_tokens_per_second_median"]) for group in pipeline
        ]
        single_bars = pipeline_axis.bar(
            [position - width / 2 for position in positions],
            single_rates,
            width,
            color="C0",
            label="Single GPU",
        )
        pipeline_bars = pipeline_axis.bar(
            [position + width / 2 for position in positions],
            pipeline_rates,
            width,
            color="C2",
            label="PP4",
        )
        for bars in (single_bars, pipeline_bars):
            for bar in bars:
                pipeline_axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(pipeline_rates) * 0.025,
                    f"{bar.get_height():.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        for position, single_rate, pipeline_rate in zip(
            positions, single_rates, pipeline_rates, strict=True
        ):
            pipeline_axis.text(
                position,
                max(single_rate, pipeline_rate) + max(pipeline_rates) * 0.13,
                f"PP4 {pipeline_rate / single_rate:.2f}×",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )
        pipeline_axis.set_xticks(positions, ("32 requests", "64 requests"))
        pipeline_axis.set_ylim(0, max(pipeline_rates) * 1.35)
        pipeline_axis.set_ylabel("Output throughput (tokens/s)")
        pipeline_axis.set_title(
            "(a) SpikeLM-2.78B · fixed 64 input / 64 output", loc="left"
        )
        pipeline_axis.legend(loc="upper left", frameon=False)

        data_rates = [
            float(row["output_tokens_per_second_median"]) for row in data_parallel
        ]
        bars = data_axis.bar(range(2), data_rates, color=("C0", "C1"), width=0.68)
        baseline = data_rates[0]
        for bar, row in zip(bars, data_parallel, strict=True):
            data_axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(data_rates) * 0.035,
                f"{bar.get_height():.0f}\n{bar.get_height() / baseline:.2f}×",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
            data_axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.48,
                f"p99 TPOT\n{float(row['tpot_p99_ms_median']):.1f} ms",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
                fontweight="bold",
            )
        data_axis.set_xticks(range(2), ("Single GPU\n32 requests", "DP4\n128 requests"))
        data_axis.set_ylim(0, max(data_rates) * 1.23)
        data_axis.set_title("(b) Qwen2.5-0.5B QCFS\n128 input / 128 output", loc="left")

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
        prefix_axis.set_title("(c) Qwen2.5-0.5B · single-GPU Radix reuse", loc="left")
        prefix_axis.legend(loc="upper left", frameon=False)
        prefix_axis.text(
            positions[1] - width / 2,
            4000,
            f"{input_rates[1] / input_rates[0]:.1f}×\ninput",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
            fontweight="bold",
        )

        for axis in (pipeline_axis, data_axis, prefix_axis):
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="both", which="both", top=False, right=False)
            axis.grid(axis="y", color="0.88", linewidth=0.6, linestyle="--")
            axis.set_axisbelow(True)

        figure.suptitle(
            "SGLang offline inference: pipeline concurrency, DP scale-out, and prefix reuse",
            x=0.06,
            ha="left",
            fontweight="bold",
            fontsize=11,
        )
        figure.text(
            0.06,
            0.005,
            "BF16 eager · 4× RTX 4090 PCIe hosts · median of 3 cache-flushed runs",
            fontsize=7,
            color="0.35",
        )
        figure.tight_layout(rect=(0, 0.04, 1, 0.94), w_pad=1.6)
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
