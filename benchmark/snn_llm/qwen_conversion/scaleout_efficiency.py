"""Qwen2.5 dense versus public-recipe SNN inference efficiency runner."""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import torch

from benchmark.snn_llm._reporting import write_report as _write_report
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import (
    ModuleConverter,
    Qwen2SNNConfig,
    Qwen2SNNRecipe,
)
from spikingjelly.activation_based.precision import (
    PrecisionConfig,
    prepare_model_for_precision,
)

from benchmark.snn_llm.qwen_conversion._runtime import (
    ARTIFACT_LOCK,
    FIXED_PROMPTS,
    build_environment,
    encode as _encode,
    hash_files as _hash_files,
    load_calibration as _load_calibration,
    load_lock as _load_lock,
    load_model as _load_model,
    relative_l2 as _relative_l2,
    validate_calibration_config as _validate_calibration_config,
)


SCHEMA_VERSION = 1
CONTRACT_KIND = "qwen2-snn-scaleout-efficiency"
WARMUP_COUNT = 2
REPEAT_COUNT = 5
PREFILL_LENGTH = 16
TIME_STEP_CHOICES = (16, 32, 64, 128, 160, 192, 256, 512)


def _gpu_snapshot(
    *, sample_count: int = 60, sample_interval_seconds: float = 1.0
) -> Dict[str, object]:
    if sample_count <= 0 or sample_interval_seconds < 0:
        raise ValueError(
            "GPU sampling requires a positive count and non-negative interval."
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    command = (
        "nvidia-smi",
        f"--id={visible}",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )
    process_command = (
        "nvidia-smi",
        f"--id={visible}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    samples = []
    processes = []
    identity = None
    for index in range(sample_count):
        completed = subprocess.run(
            command, capture_output=True, check=True, text=True, timeout=30
        )
        fields = [value.strip() for value in completed.stdout.strip().split(",")]
        if len(fields) != 5:
            raise RuntimeError("nvidia-smi returned an unexpected GPU snapshot.")
        current_identity = (int(fields[0]), fields[1], int(fields[4]))
        if identity is not None and current_identity != identity:
            raise RuntimeError("nvidia-smi GPU identity changed during sampling.")
        identity = current_identity
        samples.append(
            {
                "utilization_gpu_percent": int(fields[2]),
                "memory_used_mb": int(fields[3]),
            }
        )
        processes_result = subprocess.run(
            process_command,
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )
        for line in processes_result.stdout.splitlines():
            process_fields = [value.strip() for value in line.split(",")]
            if len(process_fields) == 3 and process_fields[0].isdigit():
                processes.append(
                    {
                        "sample_index": index,
                        "pid": int(process_fields[0]),
                        "process_name": process_fields[1],
                        "used_memory_mb": int(process_fields[2]),
                    }
                )
        if index + 1 < sample_count:
            time.sleep(sample_interval_seconds)
    if identity is None:
        raise RuntimeError("GPU sampling did not establish a device identity.")
    return {
        "physical_index": identity[0],
        "name": identity[1],
        "memory_total_mb": identity[2],
        "sample_count": sample_count,
        "sample_interval_seconds": sample_interval_seconds,
        "utilization_gpu_percent_median": statistics.median(
            sample["utilization_gpu_percent"] for sample in samples
        ),
        "memory_used_mb_max": max(sample["memory_used_mb"] for sample in samples),
        "samples": samples,
        "compute_processes": processes,
        "external_compute_processes": [
            process for process in processes if process["pid"] != os.getpid()
        ],
    }


def _measurement_label(snapshot: Mapping[str, object]) -> str:
    if (
        float(snapshot["utilization_gpu_percent_median"]) <= 5
        and int(snapshot["memory_used_mb_max"]) <= 2048
        and not snapshot["external_compute_processes"]
        and int(snapshot["sample_count"]) >= 60
    ):
        return "clean_gpu_measurement"
    return "shared_gpu_smoke_measurement"


def _prepare_benchmark_model(model) -> None:
    model.eval()
    setter = getattr(model, "set_collect_statistics", None)
    if callable(setter):
        setter(False)


def _percentile90(values: List[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def _positive_median(values: List[float], label: str) -> float:
    median = float(statistics.median(values))
    if not math.isfinite(median) or median <= 0.0:
        raise RuntimeError(f"{label} produced a non-positive or non-finite latency.")
    return median


def _timed_call(call, device: torch.device) -> tuple[float, object]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = call()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), result


def _benchmark_prefill(
    *, model, input_ids, attention_mask, autocast_context, encoding_mode
) -> tuple[Dict[str, object], Dict[str, object]]:
    _prepare_benchmark_model(model)

    def call():
        functional.reset_net(model)
        with torch.inference_mode(), autocast_context():
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if encoding_mode is not None:
                kwargs["encoding_mode"] = encoding_mode
            return model(**kwargs)

    for _ in range(WARMUP_COUNT):
        call()
    torch.cuda.reset_peak_memory_stats(input_ids.device)
    baseline = torch.cuda.memory_allocated(input_ids.device)
    samples = []
    output = None
    for _ in range(REPEAT_COUNT):
        elapsed, output = _timed_call(call, input_ids.device)
        samples.append(elapsed)
    if output is None:
        raise RuntimeError("Prefill benchmark did not produce output.")
    peak = torch.cuda.max_memory_allocated(input_ids.device)
    median_ms = _positive_median(samples, "Prefill benchmark")
    result = {
        "warmup_count": WARMUP_COUNT,
        "repeat_count": REPEAT_COUNT,
        "latency_samples_ms": samples,
        "median_ms": median_ms,
        "p90_ms": _percentile90(samples),
        "tokens_per_second": input_ids.numel() / (median_ms / 1e3),
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "peak_delta_bytes": peak - baseline,
        "statistics_enabled": False,
    }
    evidence = {
        "logits": output.logits.detach().float().cpu(),
        "token_id": int(output.logits[:, -1].argmax(-1)[0]),
    }
    return result, evidence


def _benchmark_decode(
    *, model, input_ids, attention_mask, autocast_context, encoding_mode
) -> Dict[str, object]:
    _prepare_benchmark_model(model)

    def build_cache():
        functional.reset_net(model)
        with torch.inference_mode(), autocast_context():
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            if encoding_mode is not None:
                kwargs["encoding_mode"] = encoding_mode
            return model(**kwargs).past_key_values

    next_ids = input_ids[:, -1:]
    next_mask = torch.ones(
        (input_ids.shape[0], input_ids.shape[1] + 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    def call(cache):
        functional.reset_net(model)
        with torch.inference_mode(), autocast_context():
            kwargs = {
                "input_ids": next_ids,
                "attention_mask": next_mask,
                "past_key_values": cache,
                "use_cache": True,
            }
            if encoding_mode is not None:
                kwargs["encoding_mode"] = encoding_mode
            return model(**kwargs)

    for _ in range(WARMUP_COUNT):
        call(build_cache())
    torch.cuda.reset_peak_memory_stats(input_ids.device)
    baseline = torch.cuda.memory_allocated(input_ids.device)
    samples = []
    for _ in range(REPEAT_COUNT):
        cache = build_cache()
        elapsed, _ = _timed_call(lambda: call(cache), input_ids.device)
        samples.append(elapsed)
    peak = torch.cuda.max_memory_allocated(input_ids.device)
    median_ms = _positive_median(samples, "Decode benchmark")
    return {
        "warmup_count": WARMUP_COUNT,
        "repeat_count": REPEAT_COUNT,
        "latency_samples_ms": samples,
        "median_ms": median_ms,
        "p90_ms": _percentile90(samples),
        "tokens_per_second": 1e3 / median_ms,
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "peak_delta_bytes": peak - baseline,
        "statistics_enabled": False,
    }


def _backend_parity(
    candidate: Mapping[str, object], reference: Mapping[str, object]
) -> Dict[str, object]:
    if int(candidate["token_id"]) != int(reference["token_id"]):
        raise ValueError("Torch and Triton SNN token IDs differ.")
    value = _relative_l2(
        torch.as_tensor(candidate["logits"]), torch.as_tensor(reference["logits"])
    )
    if value > 0.02:
        raise ValueError(f"Torch/Triton SNN logits relative L2 {value} exceeds 0.02.")
    return {"relative_l2": value, "token_id_equal": True}


def _run(args: argparse.Namespace) -> Dict[str, object]:
    lock = _load_lock()
    record = lock["models"][args.model_key]
    snapshot = _gpu_snapshot()
    tokenizer, source = _load_model(args.model_root, args.device)
    precision = prepare_model_for_precision(
        source,
        args.device,
        PrecisionConfig(
            mode="bf16", strictness="strict", report=True, device=args.device
        ),
    )
    calibration, calibration_sha256 = _load_calibration(args.calibration_artifact)
    _validate_calibration_config(
        calibration,
        time_steps=args.time_steps,
        calibration_levels=args.calibration_levels,
        calibration_quantile=args.calibration_quantile,
        calibration_reservoir_size=args.calibration_reservoir_size,
    )
    input_ids, attention_mask = _encode(tokenizer, [FIXED_PROMPTS[2]], args.device)
    input_ids = input_ids[:, :PREFILL_LENGTH]
    attention_mask = attention_mask[:, :PREFILL_LENGTH]
    dense_prefill, dense_evidence = _benchmark_prefill(
        model=precision.model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        autocast_context=precision.autocast_context,
        encoding_mode=None,
    )
    dense_decode = _benchmark_decode(
        model=precision.model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        autocast_context=precision.autocast_context,
        encoding_mode=None,
    )
    backend_results = {}
    evidence = {}
    for backend in ("torch", "triton"):
        config = Qwen2SNNConfig(
            time_steps=args.time_steps,
            calibration_levels=args.calibration_levels,
            calibration_quantile=args.calibration_quantile,
            calibration_reservoir_size=args.calibration_reservoir_size,
            neuron_backend=backend,
        )
        converted = ModuleConverter(Qwen2SNNRecipe(calibration, config)).convert(
            precision.model
        )
        prefill, backend_evidence = _benchmark_prefill(
            model=converted,
            input_ids=input_ids,
            attention_mask=attention_mask,
            autocast_context=precision.autocast_context,
            encoding_mode="signed_if",
        )
        decode = _benchmark_decode(
            model=converted,
            input_ids=input_ids,
            attention_mask=attention_mask,
            autocast_context=precision.autocast_context,
            encoding_mode="signed_if",
        )
        backend_results[backend] = {"prefill": prefill, "decode": decode}
        evidence[backend] = backend_evidence
        del converted
        gc.collect()
        torch.cuda.empty_cache()
    parity = _backend_parity(evidence["triton"], evidence["torch"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "source": {
            "worktree_revision": args.worktree_revision,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "artifact_lock_sha256": hashlib.sha256(
                ARTIFACT_LOCK.read_bytes()
            ).hexdigest(),
            "model_files": _hash_files(args.model_root),
        },
        "model": {"key": args.model_key, **record},
        "environment": build_environment(args.device),
        "configuration": {
            "precision": "bf16",
            "time_steps": args.time_steps,
            "calibration_levels": args.calibration_levels,
            "calibration_quantile": args.calibration_quantile,
            "prefill_length": PREFILL_LENGTH,
            "measurement": _measurement_label(snapshot),
        },
        "gpu_before": snapshot,
        "precision": precision.describe(),
        "conversion": {
            "temporal_layout": "[T,B,S,H]",
            "execution_schedule": "layerwise_offline_multistep",
            "online_inference": False,
            "calibration_sha256": calibration_sha256,
        },
        "performance": {
            "dense": {"prefill": dense_prefill, "decode": dense_decode},
            "snn": backend_results,
            "triton_vs_torch": {
                "prefill_speedup": backend_results["torch"]["prefill"]["median_ms"]
                / backend_results["triton"]["prefill"]["median_ms"],
                "decode_speedup": backend_results["torch"]["decode"]["median_ms"]
                / backend_results["triton"]["decode"]["median_ms"],
            },
            "triton_vs_dense": {
                "prefill_speed_ratio": dense_prefill["median_ms"]
                / backend_results["triton"]["prefill"]["median_ms"],
                "decode_speed_ratio": dense_decode["median_ms"]
                / backend_results["triton"]["decode"]["median_ms"],
            },
        },
        "correctness": {
            "torch_triton": parity,
            "dense_token_id": dense_evidence["token_id"],
            "snn_token_id": evidence["triton"]["token_id"],
        },
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True, choices=("0.5b", "1.5b", "3b"))
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--calibration-artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda",))
    parser.add_argument("--worktree-revision", required=True)
    parser.add_argument(
        "--time-steps", required=True, type=int, choices=TIME_STEP_CHOICES
    )
    parser.add_argument("--calibration-levels", required=True, type=int)
    parser.add_argument("--calibration-quantile", required=True, type=float)
    parser.add_argument("--calibration-reservoir-size", type=int, default=4096)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = _parse_args(argv)
        if args.calibration_levels > args.time_steps:
            raise ValueError("calibration-levels must not exceed time-steps.")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen efficiency evaluation requires CUDA.")
        report = _run(args)
        path = _write_report(args.output_dir, report)
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"report_path={path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
