import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-scale training benchmarks for the five-layer FC SNN and "
            "plot speed / memory curves for fp32, bf16, fp8-torchao, and fp8-te."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--time-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--attention-every", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="Sweep hidden/input dimensions. Each value defines one square model scale.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["fp32", "bf16", "fp8-torchao"],
        choices=("fp32", "bf16", "fp8-torchao", "fp8-te"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark") / "outputs" / "train_precision_fc",
    )
    return parser.parse_args()


def ensure_aligned(values: list[int], alignment: int = 16) -> None:
    invalid = [str(v) for v in values if v % alignment != 0]
    if invalid:
        raise ValueError(
            f"All sweep dimensions must be divisible by {alignment}; got {', '.join(invalid)}."
        )


def run_single_scale(
    args: argparse.Namespace,
    hidden_dim: int,
    output_dir: Path,
) -> dict:
    scale_name = f"h{hidden_dim}"
    cmd = [
        "python3",
        "benchmark/benchmark_train_precision_snn_fc.py",
        "--device",
        args.device,
        "--backend",
        args.backend,
        "--time-steps",
        str(args.time_steps),
        "--batch-size",
        str(args.batch_size),
        "--input-dim",
        str(hidden_dim),
        "--hidden-dim",
        str(hidden_dim),
        "--num-classes",
        str(args.num_classes),
        "--depth",
        str(args.depth),
        "--attention-every",
        str(args.attention_every),
        "--num-heads",
        str(args.num_heads),
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--precisions",
        *args.precisions,
        "--json",
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    start = proc.stdout.find("{")
    if start < 0:
        raise RuntimeError(
            f"Failed to locate JSON payload in benchmark output:\n{proc.stdout}"
        )
    payload = json.loads(proc.stdout[start:])
    payload["scale_name"] = scale_name
    payload["hidden_dim"] = hidden_dim
    payload["input_dim"] = hidden_dim
    payload["compute_proxy"] = (
        args.time_steps * args.batch_size * hidden_dim * hidden_dim * args.depth
    )

    json_path = output_dir / f"{scale_name}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    return payload


def plot_metric(
    sweep_results: list[dict],
    output_path: Path,
    metric_key: str,
    ylabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))
    for precision in sweep_results[0]["results_by_precision"]:
        x = [result["hidden_dim"] for result in sweep_results]
        y = [
            result["results_by_precision"][precision][metric_key]
            for result in sweep_results
        ]
        plt.plot(x, y, marker="o", linewidth=2, label=precision)

    plt.xlabel("Hidden / Input Dimension")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_aligned([*args.hidden_dims, args.num_classes])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = []
    for hidden_dim in args.hidden_dims:
        payload = run_single_scale(args, hidden_dim, output_dir)
        by_precision = {item["precision"]: item for item in payload["results"]}
        sweep_results.append(
            {
                "hidden_dim": hidden_dim,
                "compute_proxy": payload["compute_proxy"],
                "results_by_precision": by_precision,
            }
        )

    summary = {
        "device": args.device,
        "backend": args.backend,
        "time_steps": args.time_steps,
        "batch_size": args.batch_size,
        "num_classes": args.num_classes,
        "depth": args.depth,
        "attention_every": args.attention_every,
        "num_heads": args.num_heads,
        "warmup": args.warmup,
        "steps": args.steps,
        "hidden_dims": args.hidden_dims,
        "precisions": args.precisions,
        "scales": sweep_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_metric(
        sweep_results,
        output_dir / "speed_curve.png",
        "samples_per_sec",
        "Samples / s",
        "SpikingJelly Training Throughput vs Model Scale",
    )
    plot_metric(
        sweep_results,
        output_dir / "memory_curve.png",
        "peak_allocated_mb",
        "Peak Allocated Memory (MB)",
        "SpikingJelly Peak GPU Memory vs Model Scale",
    )
    plot_metric(
        sweep_results,
        output_dir / "step_time_curve.png",
        "total_step_ms",
        "Step Time (ms)",
        "SpikingJelly Training Step Time vs Model Scale",
    )

    print(f"Wrote summary to {summary_path}")
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
