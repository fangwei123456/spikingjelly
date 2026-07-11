from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable

import torch

from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.triton_kernel.fp8_capability import (
    triton_fp8_neuron_capability_report,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel.integrate_and_fire import (
    multistep_if,
    multistep_if_mp,
    multistep_if_mp_with_plan,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel.lif import (
    multistep_lif,
    multistep_lif_mp,
    multistep_lif_mp_with_plan,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel.plif import (
    multistep_plif,
    multistep_plif_mp,
    multistep_plif_mp_with_plan,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel.utils import (
    TritonNeuronForwardPlan,
    prepare_triton_neuron_forward_plan,
)
from spikingjelly.activation_based.triton_kernel.triton_utils import (
    normalize_triton_compute_dtype_name,
)


_SURROGATE = surrogate.Sigmoid()
_R_TAU = 0.5
_V_THRESHOLD = 1.0
_V_RESET = 0.0


def _resolve_commit(explicit_commit: str | None) -> tuple[str | None, str]:
    if explicit_commit:
        return explicit_commit, "argument"
    for env_name in ("SJ_COMMIT", "GIT_COMMIT", "CODEX_COMMIT"):
        value = os.environ.get(env_name)
        if value:
            return value, f"env:{env_name}"
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip(),
            "git",
        )
    except Exception:
        return None, "unavailable"


def _parse_sizes(value: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid size {item!r}; expected T x N, e.g. 64x65536.")
        t_text, n_text = item.split("x", 1)
        T = int(t_text)
        N = int(n_text)
        if T <= 0 or N <= 0:
            raise ValueError(f"Invalid size {item!r}; T and N must be positive.")
        sizes.append((T, N))
    if not sizes:
        raise ValueError("--sizes must contain at least one T x N entry.")
    return sizes


def _cuda_sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _make_input(T: int, N: int, device: torch.device) -> torch.Tensor:
    return torch.randn(T, N, device=device, dtype=torch.float32)


def _call_stable(
    x: torch.Tensor,
    *,
    neuron_type: str,
    v_init: torch.Tensor,
    r_tau_tensor: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    if neuron_type == "if":
        return multistep_if(
            x,
            v_init,
            _V_THRESHOLD,
            _V_RESET,
            False,
            _SURROGATE,
        )
    if neuron_type == "lif":
        return multistep_lif(
            x,
            v_init,
            True,
            1.0 / _R_TAU,
            _V_THRESHOLD,
            _V_RESET,
            False,
            _SURROGATE,
        )
    if neuron_type == "plif":
        if r_tau_tensor is None:
            raise ValueError("PLIF requires r_tau_tensor.")
        return multistep_plif(
            x,
            v_init,
            r_tau_tensor,
            True,
            _V_THRESHOLD,
            _V_RESET,
            False,
            _SURROGATE,
        )
    raise ValueError(f"Unsupported neuron type: {neuron_type}.")


def _prepare_mp_plan(
    *,
    neuron_type: str,
    device: torch.device,
    storage_dtype: torch.dtype,
    compute_dtype: str,
    backward_compute_dtype: str,
) -> TritonNeuronForwardPlan:
    return prepare_triton_neuron_forward_plan(
        neuron_type=neuron_type,
        device=device,
        storage_dtype=storage_dtype,
        compute_dtype=compute_dtype,
        backward_compute_dtype=backward_compute_dtype,
        spike_dtype=torch.float32,
        save_intermediates=True,
    )


def _call_mp_with_plan(
    x: torch.Tensor,
    *,
    neuron_type: str,
    v_init: torch.Tensor,
    r_tau_tensor: torch.Tensor | None,
    plan: TritonNeuronForwardPlan,
) -> tuple[torch.Tensor, ...]:
    if neuron_type == "if":
        return multistep_if_mp_with_plan(
            x,
            v_init,
            plan,
            v_threshold=_V_THRESHOLD,
            v_reset=_V_RESET,
        )
    if neuron_type == "lif":
        return multistep_lif_mp_with_plan(
            x,
            v_init,
            plan,
            decay_input=True,
            tau=1.0 / _R_TAU,
            v_threshold=_V_THRESHOLD,
            v_reset=_V_RESET,
        )
    if neuron_type == "plif":
        if r_tau_tensor is None:
            raise ValueError("PLIF requires r_tau_tensor.")
        return multistep_plif_mp_with_plan(
            x,
            v_init,
            r_tau_tensor,
            plan,
            decay_input=True,
            v_threshold=_V_THRESHOLD,
            v_reset=_V_RESET,
        )
    raise ValueError(f"Unsupported neuron type: {neuron_type}.")


def _call_mp_safe(
    x: torch.Tensor,
    *,
    neuron_type: str,
    v_init: torch.Tensor,
    r_tau_tensor: torch.Tensor | None,
    storage_dtype: torch.dtype,
    compute_dtype: str,
    backward_compute_dtype: str,
) -> tuple[torch.Tensor, ...]:
    if neuron_type == "if":
        return multistep_if_mp(
            x,
            v_init,
            v_threshold=_V_THRESHOLD,
            v_reset=_V_RESET,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            backward_compute_dtype=backward_compute_dtype,
            spike_dtype=torch.float32,
            save_intermediates=True,
        )
    if neuron_type == "lif":
        return multistep_lif_mp(
            x,
            v_init,
            decay_input=True,
            tau=1.0 / _R_TAU,
            v_threshold=_V_THRESHOLD,
            v_reset=_V_RESET,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            backward_compute_dtype=backward_compute_dtype,
            spike_dtype=torch.float32,
            save_intermediates=True,
        )
    if neuron_type == "plif":
        if r_tau_tensor is None:
            raise ValueError("PLIF requires r_tau_tensor.")
        return multistep_plif_mp(
            x,
            v_init,
            r_tau_tensor,
            decay_input=True,
            v_threshold=_V_THRESHOLD,
            v_reset=_V_RESET,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            backward_compute_dtype=backward_compute_dtype,
            spike_dtype=torch.float32,
            save_intermediates=True,
        )
    raise ValueError(f"Unsupported neuron type: {neuron_type}.")


def _loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    loss = outputs[0].float().sum() + outputs[1].float().sum() * 0.125
    if len(outputs) > 2 and outputs[2] is not None:
        loss = loss + outputs[2].float().sum() * 0.03125
    return loss


def _measure(
    *,
    device: torch.device,
    repeats: int,
    warmup: int,
    fn: Callable[[], Any],
) -> dict[str, float]:
    timings: list[float] = []
    torch.cuda.empty_cache()
    _cuda_sync(device)
    for _ in range(warmup):
        fn()
    _cuda_sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _cuda_sync(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
    return {
        "avg_ms": sum(timings) / len(timings),
        "median_ms": median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "peak_allocated_mb": peak_allocated_mb,
    }


def _benchmark_inference(
    *,
    x: torch.Tensor,
    neuron_type: str,
    device: torch.device,
    repeats: int,
    warmup: int,
    variant_kind: str,
    plan: TritonNeuronForwardPlan | None,
    storage_dtype: torch.dtype | None,
    compute_dtype: str | None,
    backward_compute_dtype: str,
) -> dict[str, float]:
    v_init = torch.zeros_like(x[0])
    r_tau_tensor = (
        torch.tensor(_R_TAU, device=device, dtype=torch.float32)
        if neuron_type == "plif"
        else None
    )

    def fn() -> None:
        with torch.no_grad():
            if variant_kind == "stable":
                _call_stable(
                    x,
                    neuron_type=neuron_type,
                    v_init=v_init,
                    r_tau_tensor=r_tau_tensor,
                )
            else:
                if plan is not None:
                    _call_mp_with_plan(
                        x,
                        neuron_type=neuron_type,
                        v_init=v_init,
                        r_tau_tensor=r_tau_tensor,
                        plan=plan,
                    )
                else:
                    if storage_dtype is None or compute_dtype is None:
                        raise ValueError("Mixed precision safe path requires dtypes.")
                    _call_mp_safe(
                        x,
                        neuron_type=neuron_type,
                        v_init=v_init,
                        r_tau_tensor=r_tau_tensor,
                        storage_dtype=storage_dtype,
                        compute_dtype=compute_dtype,
                        backward_compute_dtype=backward_compute_dtype,
                    )

    return _measure(device=device, repeats=repeats, warmup=warmup, fn=fn)


def _benchmark_training(
    *,
    x: torch.Tensor,
    neuron_type: str,
    device: torch.device,
    repeats: int,
    warmup: int,
    variant_kind: str,
    plan: TritonNeuronForwardPlan | None,
    storage_dtype: torch.dtype | None,
    compute_dtype: str | None,
    backward_compute_dtype: str,
) -> dict[str, float]:
    x_req = x.detach().clone().requires_grad_()
    v_init = torch.zeros_like(x[0]).requires_grad_()
    r_tau_tensor = None
    if neuron_type == "plif":
        r_tau_tensor = torch.tensor(
            _R_TAU, device=device, dtype=torch.float32, requires_grad=True
        )

    def fn() -> None:
        x_req.grad = None
        v_init.grad = None
        if r_tau_tensor is not None:
            r_tau_tensor.grad = None
        if variant_kind == "stable":
            outputs = _call_stable(
                x_req,
                neuron_type=neuron_type,
                v_init=v_init,
                r_tau_tensor=r_tau_tensor,
            )
        else:
            if plan is not None:
                outputs = _call_mp_with_plan(
                    x_req,
                    neuron_type=neuron_type,
                    v_init=v_init,
                    r_tau_tensor=r_tau_tensor,
                    plan=plan,
                )
            else:
                if storage_dtype is None or compute_dtype is None:
                    raise ValueError("Mixed precision safe path requires dtypes.")
                outputs = _call_mp_safe(
                    x_req,
                    neuron_type=neuron_type,
                    v_init=v_init,
                    r_tau_tensor=r_tau_tensor,
                    storage_dtype=storage_dtype,
                    compute_dtype=compute_dtype,
                    backward_compute_dtype=backward_compute_dtype,
                )
        _loss(outputs).backward()

    return _measure(device=device, repeats=repeats, warmup=warmup, fn=fn)


def _dtype_variants() -> list[tuple[str, torch.dtype]]:
    variants: list[tuple[str, torch.dtype]] = []
    if hasattr(torch, "float8_e4m3fn"):
        variants.append(("float8_e4m3fn", torch.float8_e4m3fn))
    if hasattr(torch, "float8_e5m2"):
        variants.append(("float8_e5m2", torch.float8_e5m2))
    return variants


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "T",
        "N",
        "elements",
        "neuron_type",
        "variant",
        "storage_dtype",
        "compute_dtype",
        "backward_compute_dtype",
        "process",
        "success",
        "avg_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "throughput_melems_s",
        "peak_allocated_mb",
        "plan_prepare_ms",
        "failure_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines = [
        "# Triton Neuron Kernel Benchmark",
        "",
        f"- Host: `{metadata['host']}`",
        f"- GPU: `{metadata['gpu']}`",
        f"- Commit: `{metadata.get('commit')}` ({metadata.get('commit_source')})",
        f"- Command: `{metadata['command']}`",
        "",
        "| T | N | neuron | variant | process | avg ms | median ms | throughput Melem/s | peak MB |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if not row.get("success"):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["T"]),
                    str(row["N"]),
                    str(row["neuron_type"]),
                    str(row["variant"]),
                    str(row["process"]),
                    _format_float(row["avg_ms"]),
                    _format_float(row["median_ms"]),
                    _format_float(row["throughput_melems_s"]),
                    _format_float(row["peak_allocated_mb"]),
                ]
            )
            + " |"
        )
    failures = [row for row in rows if not row.get("success")]
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            lines.append(
                "- "
                f"T={row['T']} N={row['N']} neuron={row['neuron_type']} "
                f"variant={row['variant']} process={row['process']}: "
                f"{row.get('failure_reason')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_outputs(
    output_dir: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    payload = {"metadata": metadata, "rows": rows}
    json_path = output_dir / "triton_neuron_kernel_benchmark.json"
    csv_path = output_dir / "triton_neuron_kernel_benchmark.csv"
    md_path = output_dir / "triton_neuron_kernel_benchmark.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--commit", default=None)
    parser.add_argument(
        "--sizes",
        default="32x4096,64x65536,256x65536,1024x65536",
        help="Comma-separated T x N sizes.",
    )
    parser.add_argument("--neurons", default="if,lif,plif")
    parser.add_argument("--compute-dtypes", default="fp16,bf16,fp32")
    parser.add_argument("--backward-compute-dtype", default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--include-safe-wrapper",
        action="store_true",
        help="Also benchmark mixed precision safe wrappers that prepare a plan per call.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0.")

    sizes = _parse_sizes(args.sizes)
    neurons = [item.strip() for item in args.neurons.split(",") if item.strip()]
    unsupported_neurons = sorted(set(neurons) - {"if", "lif", "plif"})
    if unsupported_neurons:
        raise ValueError(f"Unsupported neuron types: {unsupported_neurons}.")
    compute_dtypes = [
        normalize_triton_compute_dtype_name(item.strip())
        for item in args.compute_dtypes.split(",")
        if item.strip()
    ]
    if not compute_dtypes:
        raise ValueError("--compute-dtypes must contain at least one dtype.")
    backward_compute_dtype = normalize_triton_compute_dtype_name(
        args.backward_compute_dtype
    )

    torch.manual_seed(0)
    commit, commit_source = _resolve_commit(args.commit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "commit": commit,
        "commit_source": commit_source,
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "triton_capability": triton_fp8_neuron_capability_report(device),
        "sizes": sizes,
        "neurons": neurons,
        "compute_dtypes": compute_dtypes,
        "backward_compute_dtype": backward_compute_dtype,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "v_reset": _V_RESET,
        "v_threshold": _V_THRESHOLD,
        "r_tau": _R_TAU,
        "decay_input": True,
    }
    rows: list[dict[str, Any]] = []
    dtype_variants = _dtype_variants()
    _write_outputs(output_dir, metadata, rows)

    for T, N in sizes:
        x = _make_input(T, N, device)
        elements = T * N
        for neuron_type in neurons:
            variants: list[dict[str, Any]] = [
                {
                    "variant": "stable_fp32",
                    "variant_kind": "stable",
                    "storage_dtype": "float32",
                    "compute_dtype": "fp32",
                    "plan": None,
                    "storage_dtype_obj": None,
                    "plan_prepare_ms": None,
                }
            ]
            for storage_name, storage_dtype in dtype_variants:
                for compute_dtype in compute_dtypes:
                    plan = None
                    plan_prepare_ms = None
                    try:
                        _cuda_sync(device)
                        start = time.perf_counter()
                        plan = _prepare_mp_plan(
                            neuron_type=neuron_type,
                            device=device,
                            storage_dtype=storage_dtype,
                            compute_dtype=compute_dtype,
                            backward_compute_dtype=backward_compute_dtype,
                        )
                        _cuda_sync(device)
                        plan_prepare_ms = (time.perf_counter() - start) * 1000.0
                    except Exception as e:
                        variants.append(
                            {
                                "variant": f"mp_plan_{storage_name}_{compute_dtype}",
                                "variant_kind": "mixed_precision",
                                "storage_dtype": storage_name,
                                "compute_dtype": compute_dtype,
                                "plan": None,
                                "storage_dtype_obj": storage_dtype,
                                "plan_prepare_ms": None,
                                "plan_failure_reason": (
                                    f"{type(e).__name__}: {e or repr(e)}"
                                ),
                            }
                        )
                        continue
                    variants.append(
                        {
                            "variant": f"mp_plan_{storage_name}_{compute_dtype}",
                            "variant_kind": "mixed_precision",
                            "storage_dtype": storage_name,
                            "compute_dtype": compute_dtype,
                            "plan": plan,
                            "storage_dtype_obj": storage_dtype,
                            "plan_prepare_ms": plan_prepare_ms,
                        }
                    )
                    if args.include_safe_wrapper:
                        variants.append(
                            {
                                "variant": f"mp_safe_{storage_name}_{compute_dtype}",
                                "variant_kind": "mixed_precision",
                                "storage_dtype": storage_name,
                                "compute_dtype": compute_dtype,
                                "plan": None,
                                "storage_dtype_obj": storage_dtype,
                                "plan_prepare_ms": None,
                            }
                        )

            for variant in variants:
                for process in ("inference_forward", "training_forward_backward"):
                    print(
                        "START "
                        f"T={T} N={N} neuron={neuron_type} "
                        f"variant={variant['variant']} process={process}",
                        flush=True,
                    )
                    row = {
                        "T": T,
                        "N": N,
                        "elements": elements,
                        "neuron_type": neuron_type,
                        "variant": variant["variant"],
                        "storage_dtype": variant["storage_dtype"],
                        "compute_dtype": variant["compute_dtype"],
                        "backward_compute_dtype": backward_compute_dtype,
                        "process": process,
                        "plan_prepare_ms": variant.get("plan_prepare_ms"),
                    }
                    if variant.get("plan_failure_reason") is not None:
                        row.update(
                            {
                                "success": False,
                                "failure_reason": variant["plan_failure_reason"],
                            }
                        )
                        rows.append(row)
                        _write_outputs(output_dir, metadata, rows)
                        print(
                            "DONE "
                            f"T={T} N={N} neuron={neuron_type} "
                            f"variant={variant['variant']} process={process} "
                            f"success={row['success']}",
                            flush=True,
                        )
                        continue
                    try:
                        if process == "inference_forward":
                            metrics = _benchmark_inference(
                                x=x,
                                neuron_type=neuron_type,
                                device=device,
                                repeats=args.repeats,
                                warmup=args.warmup,
                                variant_kind=variant["variant_kind"],
                                plan=variant["plan"],
                                storage_dtype=variant["storage_dtype_obj"],
                                compute_dtype=variant["compute_dtype"],
                                backward_compute_dtype=backward_compute_dtype,
                            )
                        else:
                            metrics = _benchmark_training(
                                x=x,
                                neuron_type=neuron_type,
                                device=device,
                                repeats=args.repeats,
                                warmup=args.warmup,
                                variant_kind=variant["variant_kind"],
                                plan=variant["plan"],
                                storage_dtype=variant["storage_dtype_obj"],
                                compute_dtype=variant["compute_dtype"],
                                backward_compute_dtype=backward_compute_dtype,
                            )
                        row.update(metrics)
                        row["success"] = True
                        row["throughput_melems_s"] = elements / row["avg_ms"] / 1000.0
                    except Exception as e:
                        row.update(
                            {
                                "success": False,
                                "failure_reason": f"{type(e).__name__}: {e or repr(e)}",
                            }
                        )
                    rows.append(row)
                    _write_outputs(output_dir, metadata, rows)
                    print(
                        "DONE "
                        f"T={T} N={N} neuron={neuron_type} "
                        f"variant={variant['variant']} process={process} "
                        f"success={row['success']}",
                        flush=True,
                    )

    _write_outputs(output_dir, metadata, rows)
    json_path = output_dir / "triton_neuron_kernel_benchmark.json"
    csv_path = output_dir / "triton_neuron_kernel_benchmark.csv"
    md_path = output_dir / "triton_neuron_kernel_benchmark.md"
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
