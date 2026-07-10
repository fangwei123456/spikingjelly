from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import signal
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
from spikingjelly.activation_based.triton_kernel.neuron_kernel.integrate_and_fire import (
    multistep_if_mp,
    multistep_if_mp_with_plan,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel.lif import (
    multistep_lif_mp,
    multistep_lif_mp_with_plan,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel.plif import (
    multistep_plif_mp,
    multistep_plif_mp_with_plan,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel.utils import (
    prepare_triton_neuron_forward_plan,
)

VariantOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]


class _VariantTimeoutError(TimeoutError):
    pass


@contextlib.contextmanager
def _variant_timeout(seconds: float | None):
    if seconds is None or seconds <= 0:
        yield
        return

    def _handle_timeout(signum, frame):
        del signum, frame
        raise _VariantTimeoutError(f"variant exceeded {seconds:g}s timeout")

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _handle_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


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


def _torch_neuron_reference(
    x_seq: torch.Tensor,
    *,
    neuron_type: str,
    r_tau: float,
    v_threshold: float,
    v_reset: float | None,
    decay_input: bool | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v = torch.zeros_like(x_seq[0], dtype=torch.float32)
    s_seq = torch.empty_like(x_seq, dtype=torch.float32)
    v_seq = torch.empty_like(x_seq, dtype=torch.float32)
    h_seq = torch.empty_like(x_seq, dtype=torch.float32)
    reset_value = 0.0 if v_reset is None else v_reset
    for t in range(x_seq.shape[0]):
        x = x_seq[t].to(torch.float32)
        if neuron_type == "if":
            h = v + x
        elif decay_input:
            h = v + r_tau * (reset_value - v + x)
        else:
            h = v + r_tau * (reset_value - v) + x
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
    raise ValueError(
        f"Unknown scenario name: {name!r}. Valid options: 'random_normal', "
        "'near_threshold', 'large_positive', 'large_negative', 'long_recurrence'."
    )


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


def _call_mixed_precision_variant(
    x: torch.Tensor,
    *,
    neuron_type: str,
    r_tau: float,
    v_threshold: float,
    v_reset: float | None,
    decay_input: bool | None,
    storage_dtype: torch.dtype,
    compute_dtype: str,
    backward_compute_dtype: str = "fp32",
    v_init: torch.Tensor | None = None,
    r_tau_tensor: torch.Tensor | None = None,
) -> VariantOutput:
    if v_init is None:
        v_init = torch.zeros_like(x[0])
    if neuron_type == "if":
        return multistep_if_mp(
            x,
            v_init,
            v_threshold=v_threshold,
            v_reset=v_reset,
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
            decay_input=bool(decay_input),
            tau=1.0 / r_tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            backward_compute_dtype=backward_compute_dtype,
            spike_dtype=torch.float32,
            save_intermediates=True,
        )
    if neuron_type == "plif":
        if r_tau_tensor is None:
            r_tau_tensor = torch.tensor(r_tau, device=x.device, dtype=torch.float32)
        return multistep_plif_mp(
            x,
            v_init,
            r_tau_tensor,
            decay_input=bool(decay_input),
            v_threshold=v_threshold,
            v_reset=v_reset,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            backward_compute_dtype=backward_compute_dtype,
            spike_dtype=torch.float32,
            save_intermediates=True,
        )
    raise ValueError(f"Unsupported neuron type: {neuron_type}.")


def _prepare_variant_plan(
    x: torch.Tensor,
    *,
    neuron_type: str,
    storage_dtype: torch.dtype,
    compute_dtype: str,
    backward_compute_dtype: str = "fp32",
):
    return prepare_triton_neuron_forward_plan(
        neuron_type=neuron_type,
        device=x.device,
        storage_dtype=storage_dtype,
        compute_dtype=compute_dtype,
        backward_compute_dtype=backward_compute_dtype,
        spike_dtype=torch.float32,
        save_intermediates=True,
    )


def _call_mixed_precision_variant_with_plan(
    x: torch.Tensor,
    *,
    plan,
    neuron_type: str,
    r_tau: float,
    v_threshold: float,
    v_reset: float | None,
    decay_input: bool | None,
    v_init: torch.Tensor | None = None,
    r_tau_tensor: torch.Tensor | None = None,
) -> VariantOutput:
    if v_init is None:
        v_init = torch.zeros_like(x[0])
    if neuron_type == "if":
        return multistep_if_mp_with_plan(
            x,
            v_init,
            plan,
            v_threshold=v_threshold,
            v_reset=v_reset,
        )
    if neuron_type == "lif":
        return multistep_lif_mp_with_plan(
            x,
            v_init,
            plan,
            decay_input=bool(decay_input),
            tau=1.0 / r_tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
        )
    if neuron_type == "plif":
        if r_tau_tensor is None:
            r_tau_tensor = torch.tensor(r_tau, device=x.device, dtype=torch.float32)
        return multistep_plif_mp_with_plan(
            x,
            v_init,
            r_tau_tensor,
            plan,
            decay_input=bool(decay_input),
            v_threshold=v_threshold,
            v_reset=v_reset,
        )
    raise ValueError(f"Unsupported neuron type: {neuron_type}.")


def _compare_plan_overhead(
    x: torch.Tensor,
    *,
    neuron_type: str,
    r_tau: float,
    v_threshold: float,
    v_reset: float | None,
    decay_input: bool | None,
    storage_dtype: torch.dtype,
    compute_dtype: str,
    backward_compute_dtype: str,
    repeat: int,
) -> dict[str, Any]:
    torch.cuda.synchronize(x.device)
    start = time.perf_counter()
    plan = _prepare_variant_plan(
        x,
        neuron_type=neuron_type,
        storage_dtype=storage_dtype,
        compute_dtype=compute_dtype,
        backward_compute_dtype=backward_compute_dtype,
    )
    torch.cuda.synchronize(x.device)
    plan_prepare_ms = (time.perf_counter() - start) * 1000.0

    with torch.no_grad():
        safe_out = _call_mixed_precision_variant(
            x,
            neuron_type=neuron_type,
            r_tau=r_tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            decay_input=decay_input,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            backward_compute_dtype=backward_compute_dtype,
        )
        plan_out = _call_mixed_precision_variant_with_plan(
            x,
            plan=plan,
            neuron_type=neuron_type,
            r_tau=r_tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            decay_input=decay_input,
        )
    torch.cuda.synchronize(x.device)
    for safe_tensor, plan_tensor in zip(safe_out, plan_out):
        if safe_tensor is None or plan_tensor is None:
            if safe_tensor is not plan_tensor:
                raise RuntimeError("safe wrapper and with-plan outputs differ")
            continue
        torch.testing.assert_close(safe_tensor, plan_tensor, atol=0.0, rtol=0.0)

    torch.cuda.synchronize(x.device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            _call_mixed_precision_variant(
                x,
                neuron_type=neuron_type,
                r_tau=r_tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                decay_input=decay_input,
                storage_dtype=storage_dtype,
                compute_dtype=compute_dtype,
                backward_compute_dtype=backward_compute_dtype,
            )
    torch.cuda.synchronize(x.device)
    safe_wrapper_total_ms = (time.perf_counter() - start) * 1000.0

    torch.cuda.synchronize(x.device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            _call_mixed_precision_variant_with_plan(
                x,
                plan=plan,
                neuron_type=neuron_type,
                r_tau=r_tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                decay_input=decay_input,
            )
    torch.cuda.synchronize(x.device)
    with_plan_total_ms = (time.perf_counter() - start) * 1000.0

    def _run_backward(use_plan: bool):
        x_req = x.detach().clone().requires_grad_()
        v_req = torch.zeros_like(x[0]).requires_grad_()
        r_tau_req = torch.tensor(r_tau, device=x.device, dtype=torch.float32)
        if neuron_type == "plif":
            r_tau_req.requires_grad_()
        if use_plan:
            out = _call_mixed_precision_variant_with_plan(
                x_req,
                plan=plan,
                neuron_type=neuron_type,
                r_tau=r_tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                decay_input=decay_input,
                v_init=v_req,
                r_tau_tensor=r_tau_req,
            )
        else:
            out = _call_mixed_precision_variant(
                x_req,
                neuron_type=neuron_type,
                r_tau=r_tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                decay_input=decay_input,
                storage_dtype=storage_dtype,
                compute_dtype=compute_dtype,
                backward_compute_dtype=backward_compute_dtype,
                v_init=v_req,
                r_tau_tensor=r_tau_req,
            )
        out[0].sum().backward()
        grads = {
            "x": x_req.grad.detach(),
            "v_init": v_req.grad.detach(),
            "r_tau": r_tau_req.grad.detach() if neuron_type == "plif" else None,
        }
        finite = torch.isfinite(grads["x"]).all() and torch.isfinite(
            grads["v_init"]
        ).all()
        if neuron_type == "plif":
            finite = finite and torch.isfinite(grads["r_tau"]).all()
        return grads, bool(finite.item() if isinstance(finite, torch.Tensor) else finite)

    safe_grads, safe_grad_finite = _run_backward(use_plan=False)
    plan_grads, with_plan_grad_finite = _run_backward(use_plan=True)
    grad_x_abs_error = (safe_grads["x"] - plan_grads["x"]).abs()

    # _run_backward includes forward + backward + setup, so these metrics measure
    # end-to-end backward-path latency rather than backward-kernel-only time.
    torch.cuda.synchronize(x.device)
    start = time.perf_counter()
    for _ in range(repeat):
        _run_backward(use_plan=False)
    torch.cuda.synchronize(x.device)
    safe_wrapper_forward_backward_total_ms = (time.perf_counter() - start) * 1000.0

    torch.cuda.synchronize(x.device)
    start = time.perf_counter()
    for _ in range(repeat):
        _run_backward(use_plan=True)
    torch.cuda.synchronize(x.device)
    with_plan_forward_backward_total_ms = (time.perf_counter() - start) * 1000.0

    return {
        "compile_success": True,
        "repeat": repeat,
        "backward_compute_dtype": backward_compute_dtype,
        "plan_prepare_ms": plan_prepare_ms,
        "safe_wrapper_total_ms": safe_wrapper_total_ms,
        "with_plan_total_ms": with_plan_total_ms,
        "safe_wrapper_avg_ms": safe_wrapper_total_ms / repeat,
        "with_plan_avg_ms": with_plan_total_ms / repeat,
        "safe_wrapper_forward_backward_total_ms": (
            safe_wrapper_forward_backward_total_ms
        ),
        "with_plan_forward_backward_total_ms": with_plan_forward_backward_total_ms,
        "safe_wrapper_forward_backward_avg_ms": (
            safe_wrapper_forward_backward_total_ms / repeat
        ),
        "with_plan_forward_backward_avg_ms": (
            with_plan_forward_backward_total_ms / repeat
        ),
        "safe_grad_finite": safe_grad_finite,
        "with_plan_grad_finite": with_plan_grad_finite,
        "grad_x_max_abs_error": float(grad_x_abs_error.max().item()),
        "grad_x_mean_abs_error": float(grad_x_abs_error.mean().item()),
        "preflight_calls": {
            "safe_wrapper_prepares": repeat,
            "with_plan_prepares": 1,
        },
    }


def _run_variant(
    x: torch.Tensor,
    *,
    neuron_type: str,
    r_tau: float,
    v_threshold: float,
    v_reset: float | None,
    decay_input: bool | None,
    storage_dtype: torch.dtype,
    compute_dtype: str,
    backward_compute_dtype: str,
    timeout_s: float | None,
    warmup_iterations: int,
) -> tuple[dict[str, Any], VariantOutput | None]:
    try:
        with _variant_timeout(timeout_s):
            with torch.no_grad():
                for _ in range(warmup_iterations):
                    _call_mixed_precision_variant(
                        x,
                        neuron_type=neuron_type,
                        r_tau=r_tau,
                        v_threshold=v_threshold,
                        v_reset=v_reset,
                        decay_input=decay_input,
                        storage_dtype=storage_dtype,
                        compute_dtype=compute_dtype,
                        backward_compute_dtype=backward_compute_dtype,
                    )
            torch.cuda.synchronize(x.device)
            start = time.perf_counter()
            with torch.no_grad():
                out = _call_mixed_precision_variant(
                    x,
                    neuron_type=neuron_type,
                    r_tau=r_tau,
                    v_threshold=v_threshold,
                    v_reset=v_reset,
                    decay_input=decay_input,
                    storage_dtype=storage_dtype,
                    compute_dtype=compute_dtype,
                    backward_compute_dtype=backward_compute_dtype,
                )
            torch.cuda.synchronize(x.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "compile_success": True,
            "kernel_time_ms": elapsed_ms,
            "warmup_iterations": warmup_iterations,
        }, out
    except Exception as e:
        msg = str(e) if str(e) else repr(e)
        return {
            "compile_success": False,
            "failure_reason": f"{type(e).__name__}: {msg}",
            "warmup_iterations": warmup_iterations,
        }, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--commit",
        default=None,
        help="Commit hash to record when the probe runs outside a git worktree.",
    )
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="Untimed warmup iterations per variant before measuring kernel_time_ms.",
    )
    parser.add_argument(
        "--variant-timeout-s",
        type=float,
        default=300.0,
        help="Per-variant timeout in seconds. Set <=0 to disable.",
    )
    parser.add_argument(
        "--compare-plan-overhead",
        action="store_true",
        help="Compare repeated safe-wrapper calls with prepared-plan calls.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=100,
        help="Repeat count used by --compare-plan-overhead.",
    )
    parser.add_argument(
        "--neurons",
        default="if,lif,plif",
        help="Comma-separated neuron types to probe: if,lif,plif.",
    )
    parser.add_argument(
        "--compute-dtypes",
        default="fp16,bf16,fp32",
        help=(
            "Comma-separated compute dtype modes to probe. Pure fp8 compute is "
            "experimental and must be requested explicitly with --compute-dtypes."
        ),
    )
    parser.add_argument(
        "--backward-compute-dtype",
        default="fp32",
        help=(
            "Backward compute dtype used by --compare-plan-overhead. Pure fp8 "
            "backward is experimental and must be requested explicitly."
        ),
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
    if not dtype_variants:
        raise RuntimeError(
            "No FP8 dtypes (float8_e4m3fn / float8_e5m2) are available in this "
            "PyTorch build. This probe requires FP8 support to produce useful results."
        )
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be >= 0.")
    if args.repeat <= 0:
        raise ValueError("--repeat must be > 0.")
    commit, commit_source = _resolve_commit(args.commit)

    results: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "commit": commit,
        "commit_source": commit_source,
        "command": " ".join(sys.argv),
        "torch_version": torch.__version__,
        "triton_capability": capability,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "variant_timeout_s": args.variant_timeout_s,
        "compare_plan_overhead": args.compare_plan_overhead,
        "repeat": args.repeat,
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
    neuron_types = [item.strip() for item in args.neurons.split(",") if item.strip()]
    unsupported_neurons = sorted(set(neuron_types) - {"if", "lif", "plif"})
    if unsupported_neurons:
        raise ValueError(f"Unsupported neuron types: {unsupported_neurons}.")
    r_tau = 0.5

    for neuron_type in neuron_types:
        for scenario_name in scenario_names:
            x = _scenario(scenario_name, args.T, args.N, device)
            for reset_name, v_reset in reset_cases:
                neuron_decay_cases = [None] if neuron_type == "if" else decay_cases
                for decay_input in neuron_decay_cases:
                    ref_s, ref_v, ref_h = _torch_neuron_reference(
                        x,
                        neuron_type=neuron_type,
                        r_tau=r_tau,
                        v_threshold=1.0,
                        v_reset=v_reset,
                        decay_input=decay_input,
                    )
                    case_result: dict[str, Any] = {
                        "neuron_type": neuron_type,
                        "scenario": scenario_name,
                        "reset": reset_name,
                        "decay_input": decay_input,
                        "variants": [],
                    }
                    if neuron_type == "if":
                        triton_ref = neuron.IFNode(
                            v_threshold=1.0,
                            v_reset=v_reset,
                            step_mode="m",
                            backend="triton",
                            store_v_seq=True,
                        ).to(device).eval()
                    elif neuron_type == "lif":
                        triton_ref = neuron.LIFNode(
                            tau=1.0 / r_tau,
                            v_threshold=1.0,
                            v_reset=v_reset,
                            decay_input=bool(decay_input),
                            step_mode="m",
                            backend="triton",
                            store_v_seq=True,
                        ).to(device).eval()
                    else:
                        triton_ref = neuron.ParametricLIFNode(
                            init_tau=1.0 / r_tau,
                            v_threshold=1.0,
                            v_reset=v_reset,
                            decay_input=bool(decay_input),
                            step_mode="m",
                            backend="triton",
                            store_v_seq=True,
                        ).to(device).eval()
                    with torch.no_grad():
                        out_s = triton_ref(x)
                    case_result["triton_fp32_reference"] = _metrics(
                        ref_s, ref_v, ref_h, out_s, triton_ref.v_seq, None
                    )

                    for dtype_name, storage_dtype in dtype_variants:
                        for compute_dtype in compute_modes:
                            variant, out = _run_variant(
                                x,
                                neuron_type=neuron_type,
                                r_tau=r_tau,
                                v_threshold=1.0,
                                v_reset=v_reset,
                                decay_input=decay_input,
                                storage_dtype=storage_dtype,
                                compute_dtype=compute_dtype,
                                backward_compute_dtype=args.backward_compute_dtype,
                                timeout_s=args.variant_timeout_s,
                                warmup_iterations=args.warmup_iterations,
                            )
                            variant.update(
                                {
                                    "neuron_type": neuron_type,
                                    "storage_dtype": dtype_name,
                                    "compute_dtype": compute_dtype,
                                    "backward_compute_dtype": (
                                        args.backward_compute_dtype
                                    ),
                                }
                            )
                            if out is not None:
                                variant.update(_metrics(ref_s, ref_v, ref_h, *out))
                                if args.compare_plan_overhead:
                                    try:
                                        with _variant_timeout(args.variant_timeout_s):
                                            variant["plan_overhead"] = (
                                                _compare_plan_overhead(
                                                    x,
                                                    neuron_type=neuron_type,
                                                    r_tau=r_tau,
                                                    v_threshold=1.0,
                                                    v_reset=v_reset,
                                                    decay_input=decay_input,
                                                    storage_dtype=storage_dtype,
                                                    compute_dtype=compute_dtype,
                                                    backward_compute_dtype=(
                                                        args.backward_compute_dtype
                                                    ),
                                                    repeat=args.repeat,
                                                )
                                            )
                                    except Exception as e:
                                        msg = str(e) if str(e) else repr(e)
                                        variant["plan_overhead"] = {
                                            "compile_success": False,
                                            "failure_reason": (
                                                f"{type(e).__name__}: {msg}"
                                            ),
                                            "repeat": args.repeat,
                                        }
                            case_result["variants"].append(variant)
                    results["scenarios"].append(case_result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
