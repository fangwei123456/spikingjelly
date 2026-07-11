from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch

from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.triton_kernel.neuron_kernel import (
    integrate_and_fire,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel.lif import (
    multistep_lif,
    multistep_lif_mp_with_plan,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel.plif import (
    multistep_plif,
    multistep_plif_mp_with_plan,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel.utils import (
    TritonNeuronForwardPlan,
    prepare_triton_neuron_forward_plan,
)


_SURROGATE = surrogate.Sigmoid()
_R_TAU = 0.5


def _parse_sizes(value: str) -> list[tuple[int, int]]:
    sizes = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        t_text, n_text = item.split("x", 1)
        sizes.append((int(t_text), int(n_text)))
    return sizes


def _stable_call(
    x: torch.Tensor,
    v: torch.Tensor,
    neuron_type: str,
    r_tau: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if neuron_type == "if":
        return integrate_and_fire.multistep_if(x, v, 1.0, 0.0, False, _SURROGATE)
    if neuron_type == "lif":
        return multistep_lif(x, v, True, 2.0, 1.0, 0.0, False, _SURROGATE)
    return multistep_plif(x, v, r_tau, True, 1.0, 0.0, False, _SURROGATE)


def _mp_call(
    x: torch.Tensor,
    v: torch.Tensor,
    neuron_type: str,
    r_tau: torch.Tensor,
    plan: TritonNeuronForwardPlan,
) -> tuple[torch.Tensor, ...]:
    if neuron_type == "if":
        return integrate_and_fire.multistep_if_mp_with_plan(
            x, v, plan, v_threshold=1.0, v_reset=0.0
        )
    if neuron_type == "lif":
        return multistep_lif_mp_with_plan(
            x,
            v,
            plan,
            decay_input=True,
            tau=2.0,
            v_threshold=1.0,
            v_reset=0.0,
        )
    return multistep_plif_mp_with_plan(
        x,
        v,
        r_tau,
        plan,
        decay_input=True,
        v_threshold=1.0,
        v_reset=0.0,
    )


def _loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return outputs[0].float().sum() + outputs[1].float().sum() * 0.125


def _measure(
    device: torch.device,
    warmup: int,
    repeats: int,
    fn,
) -> dict[str, float]:
    times = []
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "avg_ms": sum(times) / len(times),
        "median_ms": median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_mb": torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0,
    }


def _write_outputs(
    output_dir: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "same_dtype_stable_vs_mp.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "same_dtype_stable_vs_mp.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sizes", default="32x4096,64x65536,256x65536,1024x65536")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtypes = [
        ("fp32", torch.float32, "fp32"),
        ("fp16", torch.float16, "fp16"),
        ("bf16", torch.bfloat16, "bf16"),
    ]
    metadata = {
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(device),
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "sizes": _parse_sizes(args.sizes),
        "dtypes": [item[0] for item in dtypes],
        "note": (
            "MP uses the same storage/compute/backward/spike dtype as stable "
            "input dtype, reuses a prebuilt plan, saves intermediates only for "
            "training, and computes loss from common outputs only."
        ),
    }
    rows: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    _write_outputs(output_dir, metadata, rows)

    for T, N in metadata["sizes"]:
        for neuron_type in ("if", "lif", "plif"):
            for dtype_name, dtype, compute_dtype in dtypes:
                torch.manual_seed(0)
                x = torch.randn(T, N, device=device, dtype=torch.float32).to(dtype)
                r_tau = torch.tensor(_R_TAU, device=device, dtype=dtype)
                plans = {
                    process: prepare_triton_neuron_forward_plan(
                        neuron_type=neuron_type,
                        device=device,
                        storage_dtype=dtype,
                        compute_dtype=compute_dtype,
                        backward_compute_dtype=compute_dtype,
                        spike_dtype=dtype,
                        save_intermediates=(
                            process == "training_forward_backward"
                        ),
                    )
                    for process in (
                        "inference_forward",
                        "training_forward_backward",
                    )
                }
                for process in ("inference_forward", "training_forward_backward"):
                    for variant in ("stable", "mp"):
                        print(
                            "START",
                            T,
                            N,
                            neuron_type,
                            dtype_name,
                            variant,
                            process,
                            flush=True,
                        )
                        row = {
                            "T": T,
                            "N": N,
                            "neuron": neuron_type,
                            "dtype": dtype_name,
                            "variant": variant,
                            "process": process,
                            "mp_plan_reused": variant == "mp",
                            "save_intermediates": (
                                process == "training_forward_backward"
                            ),
                            "loss_outputs": "s_seq,v_seq",
                        }
                        try:
                            if process == "inference_forward":
                                v = torch.zeros(N, device=device, dtype=dtype)

                                def fn():
                                    with torch.no_grad():
                                        if variant == "stable":
                                            _stable_call(x, v, neuron_type, r_tau)
                                        else:
                                            _mp_call(
                                                x,
                                                v,
                                                neuron_type,
                                                r_tau,
                                                plans[process],
                                            )

                            else:
                                x_req = x.detach().clone().requires_grad_()
                                v = torch.zeros(
                                    N, device=device, dtype=dtype, requires_grad=True
                                )
                                r_tau_req = torch.tensor(
                                    _R_TAU,
                                    device=device,
                                    dtype=dtype,
                                    requires_grad=neuron_type == "plif",
                                )

                                def fn():
                                    x_req.grad = None
                                    v.grad = None
                                    if neuron_type == "plif":
                                        r_tau_req.grad = None
                                    if variant == "stable":
                                        outputs = _stable_call(
                                            x_req, v, neuron_type, r_tau_req
                                        )
                                    else:
                                        outputs = _mp_call(
                                            x_req,
                                            v,
                                            neuron_type,
                                            r_tau_req,
                                            plans[process],
                                        )
                                    _loss(outputs).backward()

                            row.update(
                                {
                                    "success": True,
                                    **_measure(device, args.warmup, args.repeats, fn),
                                }
                            )
                        except Exception as e:
                            message = str(e).splitlines()[0] if str(e) else repr(e)
                            row.update(
                                {
                                    "success": False,
                                    "reason": f"{type(e).__name__}: {message}",
                                }
                            )
                        rows.append(row)
                        _write_outputs(output_dir, metadata, rows)
                        print("DONE", row["success"], flush=True)


if __name__ == "__main__":
    main()
