from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.triton_kernel.fp8_capability import (
    triton_fp8_neuron_capability_report,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel.lif import (
    multistep_lif_mixed_precision_forward,
)

VariantOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _torch_lif_reference(
    x_seq: torch.Tensor,
    *,
    tau: float,
    v_threshold: float,
    v_reset: float | None,
    decay_input: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v = torch.zeros_like(x_seq[0], dtype=torch.float32)
    s_seq = torch.empty_like(x_seq, dtype=torch.float32)
    v_seq = torch.empty_like(x_seq, dtype=torch.float32)
    h_seq = torch.empty_like(x_seq, dtype=torch.float32)
    reset_value = 0.0 if v_reset is None else v_reset
    for t in range(x_seq.shape[0]):
        x = x_seq[t].to(torch.float32)
        if decay_input:
            h = v + (reset_value - v + x) / tau
        else:
            h = v + (reset_value - v) / tau + x
        s = (h >= v_threshold).to(torch.float32)
        if v_reset is None:
            v = h - s * v_threshold
        else:
            v = s * reset_value + (1.0 - s) * h
        s_seq[t] = s
        v_seq[t] = v
        h_seq[t] = h
    return s_seq, v_seq, h_seq


def _scenario(name: str, T: int, N: int, device: torch.device) -> torch.Tensor:
    if name == "random_normal":
        return torch.randn(T, N, device=device, dtype=torch.float32)
    if name == "near_threshold":
        return torch.full((T, N), 0.95, device=device, dtype=torch.float32)
    if name == "large_positive":
        return torch.full((T, N), 3.0, device=device, dtype=torch.float32)
    if name == "large_negative":
        return torch.full((T, N), -3.0, device=device, dtype=torch.float32)
    if name == "long_recurrence":
        steps = torch.linspace(0.1, 1.2, T, device=device, dtype=torch.float32)
        return steps[:, None].expand(T, N).contiguous()
    raise ValueError(name)


def _metrics(
    ref_s: torch.Tensor,
    ref_v: torch.Tensor,
    ref_h: torch.Tensor,
    out_s: torch.Tensor,
    out_v: torch.Tensor,
    out_h: torch.Tensor | None,
) -> dict[str, Any]:
    out_s = out_s.to(torch.float32)
    out_v = out_v.to(torch.float32)
    mismatch = out_s.ne(ref_s).to(torch.float32)
    mismatch_indices = mismatch.flatten().nonzero()
    first_mismatch_timestep = None
    if mismatch_indices.numel() > 0:
        first_mismatch_timestep = int(mismatch_indices[0].item() // ref_s[0].numel())
    result = {
        "spike_mismatch_rate": float(mismatch.mean().item()),
        "first_mismatch_timestep": first_mismatch_timestep,
        "v_max_abs_error": float((out_v - ref_v).abs().max().item()),
        "v_mean_abs_error": float((out_v - ref_v).abs().mean().item()),
    }
    if out_h is not None:
        out_h = out_h.to(torch.float32)
        result["h_max_abs_error"] = float((out_h - ref_h).abs().max().item())
    return result


def _run_variant(
    x: torch.Tensor,
    *,
    tau: float,
    v_threshold: float,
    v_reset: float | None,
    decay_input: bool,
    storage_dtype: torch.dtype,
    compute_dtype: str,
) -> tuple[dict[str, Any], VariantOutput | None]:
    try:
        torch.cuda.synchronize(x.device)
        start = time.perf_counter()
        with torch.no_grad():
            out = multistep_lif_mixed_precision_forward(
                x,
                torch.zeros_like(x[0]),
                decay_input=decay_input,
                tau=tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                storage_dtype=storage_dtype,
                compute_dtype=compute_dtype,
                spike_dtype=torch.float32,
                save_intermediates=True,
            )
        torch.cuda.synchronize(x.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {"compile_success": True, "kernel_time_ms": elapsed_ms}, out
    except Exception as e:
        first_line = str(e).splitlines()[0] if str(e) else repr(e)
        return {
            "compile_success": False,
            "failure_reason": f"{type(e).__name__}: {first_line}",
        }, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument(
        "--compute-dtypes",
        default="fp8,fp16,bf16,fp32",
        help="Comma-separated compute dtype modes to probe.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA.")
    torch.manual_seed(0)

    capability = triton_fp8_neuron_capability_report(device)
    dtype_variants = []
    if hasattr(torch, "float8_e4m3fn"):
        dtype_variants.append(("float8_e4m3fn", torch.float8_e4m3fn))
    if hasattr(torch, "float8_e5m2"):
        dtype_variants.append(("float8_e5m2", torch.float8_e5m2))

    results: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "commit": _git_commit(),
        "command": " ".join(sys.argv),
        "torch_version": torch.__version__,
        "triton_capability": capability,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "scenarios": [],
    }

    scenario_names = [
        "random_normal",
        "near_threshold",
        "large_positive",
        "large_negative",
        "long_recurrence",
    ]
    reset_cases = [("hard_reset", 0.0), ("soft_reset", None)]
    decay_cases = [True, False]
    compute_modes = [
        item.strip() for item in args.compute_dtypes.split(",") if item.strip()
    ]

    for scenario_name in scenario_names:
        x = _scenario(scenario_name, args.T, args.N, device)
        for reset_name, v_reset in reset_cases:
            for decay_input in decay_cases:
                ref_s, ref_v, ref_h = _torch_lif_reference(
                    x,
                    tau=2.0,
                    v_threshold=1.0,
                    v_reset=v_reset,
                    decay_input=decay_input,
                )
                case_result: dict[str, Any] = {
                    "scenario": scenario_name,
                    "reset": reset_name,
                    "decay_input": decay_input,
                    "variants": [],
                }
                triton_ref = neuron.LIFNode(
                    tau=2.0,
                    v_threshold=1.0,
                    v_reset=v_reset,
                    decay_input=decay_input,
                    step_mode="m",
                    backend="triton",
                    store_v_seq=True,
                ).to(device).eval()
                with torch.no_grad():
                    out_s = triton_ref(x)
                case_result["triton_fp32_reference"] = _metrics(
                    ref_s, ref_v, ref_v, out_s, triton_ref.v_seq, None
                )

                for dtype_name, storage_dtype in dtype_variants:
                    for compute_dtype in compute_modes:
                        variant, out = _run_variant(
                            x,
                            tau=2.0,
                            v_threshold=1.0,
                            v_reset=v_reset,
                            decay_input=decay_input,
                            storage_dtype=storage_dtype,
                            compute_dtype=compute_dtype,
                        )
                        variant.update(
                            {
                                "storage_dtype": dtype_name,
                                "compute_dtype": compute_dtype,
                            }
                        )
                        if out is not None:
                            variant.update(_metrics(ref_s, ref_v, ref_h, *out))
                        case_result["variants"].append(variant)
                results["scenarios"].append(case_result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
