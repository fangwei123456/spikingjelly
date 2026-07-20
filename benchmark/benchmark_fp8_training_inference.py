from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn as nn

try:
    from benchmark.fp8_efficiency import assess_model_efficiency, require_efficiency
except ModuleNotFoundError:
    from fp8_efficiency import assess_model_efficiency, require_efficiency
from spikingjelly.activation_based.precision import (
    PrecisionConfig,
    prepare_model_for_precision,
)


FP8_MODES = ("fp8-torchao", "fp8-te")


@dataclass(frozen=True)
class BenchmarkResult:
    precision: str
    trial: int
    preparation_ms: float
    training_ms: float
    training_samples_per_sec: float
    training_peak_allocated_mb: float
    training_peak_reserved_mb: float
    inference_ms: float
    inference_samples_per_sec: float
    inference_peak_allocated_mb: float
    inference_peak_reserved_mb: float
    parameter_update_max_abs: float
    output_checksum: float
    capability_report: dict[str, Any]
    conversion_report: dict[str, Any]


class LinearWorkload(nn.Module):
    def __init__(self, width: int, depth: int, num_classes: int):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2.")
        modules: list[nn.Module] = []
        for _ in range(depth - 1):
            modules.extend((nn.Linear(width, width), nn.GELU()))
        modules.append(nn.Linear(width, num_classes))
        self.layers = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate FP8 training and inference usability and compare throughput "
            "and memory against a higher-precision baseline."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--training-steps", type=int, default=30)
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--commit")
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["fp32", "fp16", "bf16", *FP8_MODES],
        choices=("fp32", "fp16", "bf16", *FP8_MODES),
    )
    parser.add_argument(
        "--baseline-precision",
        default="bf16",
        choices=("fp32", "fp16", "bf16"),
    )
    parser.add_argument("--min-training-speedup", type=float, default=1.05)
    parser.add_argument("--min-inference-speedup", type=float, default=1.05)
    parser.add_argument(
        "--require-efficiency",
        action="store_true",
        help="Exit non-zero unless every requested FP8 mode meets both speedups.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.baseline_precision not in args.precisions:
        raise ValueError("--baseline-precision must be included in --precisions.")
    if not any(precision in FP8_MODES for precision in args.precisions):
        raise ValueError("--precisions must include at least one FP8 mode.")
    positive_values = {
        "batch_size": args.batch_size,
        "width": args.width,
        "num_classes": args.num_classes,
        "training_steps": args.training_steps,
        "inference_steps": args.inference_steps,
        "trials": args.trials,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(
            f"The following values must be positive: {', '.join(invalid)}."
        )
    if args.warmup < 1:
        raise ValueError("--warmup must be >= 1.")
    if (
        not math.isfinite(args.min_training_speedup)
        or not math.isfinite(args.min_inference_speedup)
        or args.min_training_speedup <= 0
        or args.min_inference_speedup <= 0
    ):
        raise ValueError(
            "--min-training-speedup and --min-inference-speedup must be finite and > 0."
        )
    if args.trials < 2:
        raise ValueError("--trials must be >= 2 for a median comparison.")
    if args.depth < 2:
        raise ValueError("--depth must be >= 2.")
    fp8_requested = any(precision in FP8_MODES for precision in args.precisions)
    dimensions = (args.batch_size, args.width, args.num_classes)
    if fp8_requested and any(value % 16 for value in dimensions):
        raise ValueError("FP8 benchmark dimensions must be divisible by 16.")


def _cuda_time_ms(device: torch.device, steps: int, fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / steps


def _samples_per_second(batch_size: int, elapsed_ms: float) -> float:
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise RuntimeError("CUDA timing must be finite and positive.")
    return batch_size / (elapsed_ms / 1000.0)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _commit(explicit_commit: str | None = None) -> str | None:
    if explicit_commit:
        return explicit_commit
    for env_name in ("SJ_COMMIT", "GIT_COMMIT", "CODEX_COMMIT"):
        value = os.environ.get(env_name)
        if value:
            return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def benchmark_precision(
    args: argparse.Namespace,
    precision: str,
    trial: int,
    model_state: dict[str, torch.Tensor],
    x: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> BenchmarkResult:
    model = LinearWorkload(args.width, args.depth, args.num_classes).to(device)
    model.load_state_dict(model_state, strict=True)
    model.train()
    torch.cuda.synchronize(device)
    preparation_start = time.perf_counter()
    artifacts = prepare_model_for_precision(
        model,
        device,
        PrecisionConfig(mode=precision, strictness="strict", device=str(device)),
    )
    torch.cuda.synchronize(device)
    preparation_ms = (time.perf_counter() - preparation_start) * 1000.0
    model = artifacts.model
    if artifacts.effective_config.mode != precision:
        raise RuntimeError(
            f"Requested {precision}, but effective mode is "
            f"{artifacts.effective_config.mode}."
        )
    conversion_report = artifacts.policy.conversion_report()
    if precision in FP8_MODES and not conversion_report["converted_modules"]:
        raise RuntimeError(f"{precision} did not convert any benchmark modules.")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("Benchmark model has no trainable parameters.")
    parameters_before = [parameter.detach().clone() for parameter in parameters]
    last_loss = None

    def training_step() -> None:
        nonlocal last_loss
        optimizer.zero_grad(set_to_none=True)
        with artifacts.autocast_context():
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, target)
        artifacts.backward(loss, optimizer, parameters=model.parameters())
        last_loss = loss.detach()

    for _ in range(args.warmup):
        training_step()
    if not torch.isfinite(last_loss):
        raise RuntimeError(f"{precision} training produced a non-finite loss.")
    parameter_update = max(
        float((parameter.detach() - before).abs().max().float().item())
        for parameter, before in zip(parameters, parameters_before, strict=True)
    )
    del parameters_before, parameters
    if parameter_update == 0.0:
        raise RuntimeError(f"{precision} training did not update model parameters.")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    training_ms = _cuda_time_ms(device, args.training_steps, training_step)
    training_peak_mb = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
    training_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024.0 / 1024.0
    if not torch.isfinite(last_loss):
        raise RuntimeError(
            f"{precision} training produced a non-finite loss during timed steps."
        )

    optimizer.zero_grad(set_to_none=True)
    del last_loss
    model.eval()
    torch.cuda.empty_cache()
    output = None

    def inference_step() -> None:
        nonlocal output
        with torch.inference_mode(), artifacts.autocast_context():
            output = model(x)

    for _ in range(args.warmup):
        inference_step()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    inference_ms = _cuda_time_ms(device, args.inference_steps, inference_step)
    inference_peak_mb = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
    inference_peak_reserved_mb = (
        torch.cuda.max_memory_reserved(device) / 1024.0 / 1024.0
    )
    if output is None or not torch.isfinite(output).all():
        raise RuntimeError(f"{precision} inference produced non-finite output.")

    return BenchmarkResult(
        precision=precision,
        trial=trial,
        preparation_ms=preparation_ms,
        training_ms=training_ms,
        training_samples_per_sec=_samples_per_second(args.batch_size, training_ms),
        training_peak_allocated_mb=training_peak_mb,
        training_peak_reserved_mb=training_peak_reserved_mb,
        inference_ms=inference_ms,
        inference_samples_per_sec=_samples_per_second(args.batch_size, inference_ms),
        inference_peak_allocated_mb=inference_peak_mb,
        inference_peak_reserved_mb=inference_peak_reserved_mb,
        parameter_update_max_abs=parameter_update,
        output_checksum=float(output.float().mean().item()),
        capability_report=artifacts.policy.capability_report(),
        conversion_report=conversion_report,
    )


def _aggregate_precision_trials(
    precision: str, trials: list[dict[str, Any]]
) -> dict[str, Any]:
    if not trials:
        raise ValueError(f"No benchmark trials found for {precision}.")
    metric_names = (
        "preparation_ms",
        "training_ms",
        "training_samples_per_sec",
        "training_peak_allocated_mb",
        "training_peak_reserved_mb",
        "inference_ms",
        "inference_samples_per_sec",
        "inference_peak_allocated_mb",
        "inference_peak_reserved_mb",
        "parameter_update_max_abs",
        "output_checksum",
    )
    result = {
        "precision": precision,
        "trial_count": len(trials),
        "trials": trials,
        "capability_report": trials[-1]["capability_report"],
        "conversion_report": trials[-1]["conversion_report"],
    }
    for metric_name in metric_names:
        values = [float(trial[metric_name]) for trial in trials]
        result[metric_name] = median(values)
        result[f"{metric_name}_p25"] = _percentile(values, 0.25)
        result[f"{metric_name}_p75"] = _percentile(values, 0.75)
    return result


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile from an empty sequence.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1].")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _print_results(results: list[dict[str, Any]], efficiency: dict[str, Any]) -> None:
    print(
        f"{'precision':<14} {'train ms':>10} {'train samples/s':>16} "
        f"{'infer ms':>10} {'infer samples/s':>16} {'prep ms':>10} "
        f"{'train alloc/res MB':>20} {'infer alloc/res MB':>20}"
    )
    for result in results:
        print(
            f"{result['precision']:<14} {result['training_ms']:10.3f} "
            f"{result['training_samples_per_sec']:16.1f} "
            f"{result['inference_ms']:10.3f} "
            f"{result['inference_samples_per_sec']:16.1f} "
            f"{result['preparation_ms']:10.3f} "
            f"{result['training_peak_allocated_mb']:9.1f}/"
            f"{result['training_peak_reserved_mb']:<9.1f} "
            f"{result['inference_peak_allocated_mb']:9.1f}/"
            f"{result['inference_peak_reserved_mb']:<9.1f}"
        )
    for comparison in efficiency["comparisons"]:
        print(
            f"{comparison['precision']} vs {efficiency['baseline_precision']}: "
            f"training {comparison['training_speedup']:.3f}x, "
            f"inference {comparison['inference_speedup']:.3f}x, "
            f"training memory {comparison['training_memory_ratio']:.3f}x, "
            f"inference memory {comparison['inference_memory_ratio']:.3f}x, "
            f"passed={comparison['passed']}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base_model = LinearWorkload(args.width, args.depth, args.num_classes)
    model_state = base_model.state_dict()
    x = torch.randn(args.batch_size, args.width, device=device)
    target = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
    trial_results: list[dict[str, Any]] = []
    for trial in range(args.trials):
        offset = trial % len(args.precisions)
        trial_order = args.precisions[offset:] + args.precisions[:offset]
        for precision in trial_order:
            result = benchmark_precision(
                args,
                precision,
                trial,
                model_state,
                x,
                target,
                device,
            )
            trial_results.append(asdict(result))
    result_dicts = [
        _aggregate_precision_trials(
            precision,
            [row for row in trial_results if row["precision"] == precision],
        )
        for precision in args.precisions
    ]
    efficiency = assess_model_efficiency(
        result_dicts,
        baseline_precision=args.baseline_precision,
        min_training_speedup=args.min_training_speedup,
        min_inference_speedup=args.min_inference_speedup,
    )
    payload = {
        "metadata": {
            "host": platform.node(),
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
            "commit": _commit(args.commit),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": torch.cuda.get_device_capability(device),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "torchao_version": _package_version("torchao"),
            "transformer_engine_version": _package_version("transformer-engine"),
            "triton_version": _package_version("triton"),
        },
        "config": {
            "batch_size": args.batch_size,
            "width": args.width,
            "depth": args.depth,
            "num_classes": args.num_classes,
            "warmup": args.warmup,
            "training_steps": args.training_steps,
            "inference_steps": args.inference_steps,
            "trials": args.trials,
            "trial_order": "rotated",
            "timing_scope": "steady_state_excludes_precision_conversion",
            "precisions": args.precisions,
        },
        "results": result_dicts,
        "efficiency": efficiency,
    }

    _print_results(result_dicts, efficiency)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    if args.json:
        print(json.dumps(payload, indent=2))
    if args.require_efficiency:
        require_efficiency(efficiency)


if __name__ == "__main__":
    main()
