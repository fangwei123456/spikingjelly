"""Benchmark fused IF/LIF-Linear against unfused dense paths."""

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from spikingjelly.activation_based import functional, surrogate
from spikingjelly.activation_based.cuda_kernel.neuron_linear import (
    if_linear,
    lif_linear,
)


def _neuron_torch(x_seq, v, sg, neuron):
    spikes = []
    for x in x_seq:
        if neuron == "if":
            spike, v = functional.if_step(x, v, 1.0, 0.0, sg)
        else:
            spike, v = functional.lif_step(x, v, 2.0, True, 1.0, 0.0, sg)
        spikes.append(spike)
    return torch.stack(spikes), v


def _neuron_cupy(x_seq, v, sg, neuron):
    if neuron == "if":
        if x_seq.shape[0] == 1:
            spike, v = functional.if_step_cupy(x_seq[0], v, 1.0, 0.0, sg)
            return spike.unsqueeze(0), v
        spike, v, _ = functional.if_multi_step_cupy(x_seq, v, 1.0, 0.0, sg)
        return spike, v
    if x_seq.shape[0] == 1:
        spike, v = functional.lif_step_cupy(x_seq[0], v, 2.0, True, 1.0, 0.0, sg)
        return spike.unsqueeze(0), v
    spike, v, _ = functional.lif_multi_step_cupy(x_seq, v, 2.0, True, 1.0, 0.0, sg)
    return spike, v


def _neuron_triton(x_seq, v, sg, neuron):
    if neuron == "if":
        spike, v, _ = functional.if_multi_step_triton(x_seq, v, 1.0, 0.0, sg)
    else:
        spike, v, _ = functional.lif_multi_step_triton(
            x_seq, v, 2.0, True, 1.0, 0.0, sg
        )
    return spike, v


@torch.no_grad()
def _input_for_density(T, M, K, density, device, neuron):
    base = torch.randn(T, M, K, device=device)
    v = torch.zeros(M, K, device=device)
    low, high = -8.0, 8.0
    for _ in range(20):
        mid = (low + high) / 2
        spikes, _ = _neuron_torch(base + mid, v, surrogate.heaviside, neuron)
        if spikes.float().mean().item() < density:
            low = mid
        else:
            high = mid
    x = base + (low + high) / 2
    spikes, _ = _neuron_torch(x, v, surrogate.heaviside, neuron)
    return x, spikes.float().mean().item()


def _time_setup(fn, iters=100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def _run(fn, tensors, grads, backward):
    for tensor in tensors:
        tensor.grad = None
    outputs = fn()
    if backward:
        torch.autograd.backward(outputs, grads)


def _measure_round(fn, tensors, grads, backward, iters):
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _run(fn, tensors, grads, backward)
    end.record()
    end.synchronize()
    return (
        start.elapsed_time(end) / iters,
        torch.cuda.max_memory_allocated() - allocated,
    )


def _measure(fn, tensors, grads, backward, warmup, iters, rounds):
    for _ in range(warmup):
        _run(fn, tensors, grads, backward)
    torch.cuda.synchronize()
    times = []
    peak_bytes = 0
    for _ in range(rounds):
        elapsed_ms, round_peak_bytes = _measure_round(
            fn, tensors, grads, backward, iters
        )
        times.append(elapsed_ms)
        peak_bytes = max(peak_bytes, round_peak_bytes)
    median = statistics.median(times)
    mad = statistics.median(abs(x - median) for x in times)
    return {"rounds_ms": times, "median_ms": median, "mad_ms": mad}, peak_bytes


def _paired_rounds(measure, rounds):
    pairs = []
    for round_index in range(rounds):
        order = (
            ("baseline", "candidate")
            if round_index % 2 == 0
            else ("candidate", "baseline")
        )
        values = {name: measure(name) for name in order}
        pairs.append(
            {
                "order": list(order),
                "baseline_ms": values["baseline"][0],
                "candidate_ms": values["candidate"][0],
                "speedup": values["baseline"][0] / values["candidate"][0],
                "baseline_peak_bytes": values["baseline"][1],
                "candidate_peak_bytes": values["candidate"][1],
            }
        )
    return pairs


def _measure_paired(
    baseline, candidate, tensors, grads, backward, warmup, iters, rounds
):
    methods = {"baseline": baseline, "candidate": candidate}
    for warmup_index in range(warmup):
        order = (
            ("baseline", "candidate")
            if warmup_index % 2 == 0
            else ("candidate", "baseline")
        )
        for name in order:
            _run(methods[name], tensors, grads, backward)
    torch.cuda.synchronize()

    pairs = _paired_rounds(
        lambda name: _measure_round(methods[name], tensors, grads, backward, iters),
        rounds,
    )
    speedups = [pair["speedup"] for pair in pairs]
    median = statistics.median(speedups)
    mad = statistics.median(abs(x - median) for x in speedups)
    return {
        "pairs": pairs,
        "median_speedup": median,
        "speedup_mad": mad,
        "wins": sum(speedup > 1.0 for speedup in speedups),
        "stable_winner": sum(speedup > 1.0 for speedup in speedups) >= 4
        and median - 1.0 > mad,
    }


def _methods(x, v, weight, bias, sg, mode, neuron, thread_counts):
    training = mode == "train"
    weight_t = None if training else weight.t().contiguous()

    def torch_dense():
        spike, v_out = _neuron_torch(x, v, sg, neuron)
        return F.linear(spike, weight, bias), v_out

    def cupy_dense():
        spike, v_out = _neuron_cupy(x, v, sg, neuron)
        return F.linear(spike, weight, bias), v_out

    methods = {
        "torch_dense": torch_dense,
        "cupy_dense": cupy_dense,
    }
    if x.shape[0] > 1 and importlib.util.find_spec("triton") is not None:

        def triton_dense():
            spike, v_out = _neuron_triton(x, v, sg, neuron)
            return F.linear(spike, weight, bias), v_out

        methods["triton_dense"] = triton_dense

    fused_x = x[0] if x.shape[0] == 1 else x
    fused_op = if_linear if neuron == "if" else lif_linear
    for threads in thread_counts:
        name = f"fused_{threads}"

        def fused(threads=threads):
            prepared = weight.t().contiguous() if training else weight_t
            y, v_out = fused_op(
                fused_x,
                v,
                prepared,
                bias,
                threads=threads,
            )
            return (y.unsqueeze(0) if y.dim() == 2 else y), v_out

        methods[name] = fused
    return methods


def _gpu_uuid():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", str(torch.cuda.current_device()))
    device = visible.split(",", 1)[0]
    if device.startswith("GPU-"):
        return device
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        text=True,
    )
    return output.strip()


def main():
    import cupy

    started_at = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--K", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--N", type=int, nargs="+", default=None)
    parser.add_argument("--T", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument(
        "--density", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.10]
    )
    parser.add_argument("--neuron", choices=("if", "lif"), default="lif")
    parser.add_argument("--threads", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument(
        "--mode", choices=("inference", "train", "both"), default="both"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", default="neuron_linear.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 0 or args.iters <= 0 or args.rounds < 4:
        parser.error("warmup must be non-negative, iters positive, and rounds >= 4")
    if any(x not in (128, 256, 512) for x in args.threads):
        parser.error("threads must contain only 128, 256, or 512")

    torch.manual_seed(0)
    device = torch.device("cuda")
    modes = ("inference", "train") if args.mode == "both" else (args.mode,)
    cases = []
    for T in args.T:
        for M in args.M:
            for K in args.K:
                for N in args.N or [K]:
                    for density in args.density:
                        x_data, actual_density = _input_for_density(
                            T, M, K, density, device, args.neuron
                        )
                        for mode in modes:
                            training = mode == "train"
                            x = x_data.detach().requires_grad_(training)
                            v = torch.zeros(M, K, device=device, requires_grad=training)
                            weight = torch.randn(
                                N, K, device=device, requires_grad=training
                            )
                            bias = torch.randn(N, device=device, requires_grad=training)
                            sg = surrogate.Sigmoid()
                            methods = _methods(
                                x,
                                v,
                                weight,
                                bias,
                                sg,
                                mode,
                                args.neuron,
                                args.threads,
                            )
                            with torch.no_grad():
                                expected = methods["torch_dense"]()
                                for name, fn in methods.items():
                                    actual = fn()
                                    torch.testing.assert_close(
                                        actual[0], expected[0], rtol=1e-4, atol=1e-4
                                    )
                                    torch.testing.assert_close(
                                        actual[1], expected[1], rtol=1e-5, atol=1e-6
                                    )

                            grad_y = torch.randn(T, M, N, device=device)
                            grad_v = torch.randn(M, K, device=device)
                            tensors = (x, v, weight, bias)
                            screening = {}
                            with (
                                torch.enable_grad()
                                if training
                                else torch.inference_mode()
                            ):
                                for name, fn in methods.items():
                                    timing, peak_bytes = _measure(
                                        fn,
                                        tensors,
                                        (grad_y, grad_v),
                                        training,
                                        args.warmup,
                                        args.iters,
                                        1,
                                    )
                                    screening[name] = {
                                        "timing": timing,
                                        "peak_memory_bytes": peak_bytes,
                                    }
                                baseline_name = min(
                                    (
                                        name
                                        for name in methods
                                        if not name.startswith("fused_")
                                    ),
                                    key=lambda name: screening[name]["timing"][
                                        "median_ms"
                                    ],
                                )
                                candidate_name = min(
                                    (
                                        name
                                        for name in methods
                                        if name.startswith("fused_")
                                    ),
                                    key=lambda name: screening[name]["timing"][
                                        "median_ms"
                                    ],
                                )
                                paired = _measure_paired(
                                    methods[baseline_name],
                                    methods[candidate_name],
                                    tensors,
                                    (grad_y, grad_v),
                                    training,
                                    args.warmup,
                                    args.iters,
                                    args.rounds,
                                )
                            setup_ms = _time_setup(
                                lambda: weight.t().contiguous(), args.iters
                            )
                            print(
                                f"neuron={args.neuron:3s} mode={mode:9s} "
                                f"T={T:2d} M={M:4d} K={K:4d} "
                                f"N={N:4d} rate={actual_density:.4f} "
                                f"pair={baseline_name}/{candidate_name} "
                                f"speedup={paired['median_speedup']:.3f} "
                                f"stable={paired['stable_winner']}"
                            )
                            cases.append(
                                {
                                    "T": T,
                                    "M": M,
                                    "K": K,
                                    "N": N,
                                    "target_spike_rate": density,
                                    "actual_spike_rate": actual_density,
                                    "mode": mode,
                                    "weight_transpose_setup_ms": setup_ms,
                                    "inference_prepared_weight_bytes": (
                                        0
                                        if training
                                        else weight.numel() * weight.element_size()
                                    ),
                                    "screening": screening,
                                    "selected_baseline": baseline_name,
                                    "selected_candidate": candidate_name,
                                    "paired": paired,
                                }
                            )

    properties = torch.cuda.get_device_properties(device)
    output = {
        "metadata": {
            "source_revision": os.environ.get("SPIKINGJELLY_COMMIT"),
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(lif_linear.__code__.co_filename), Path(__file__))
            },
            "device_name": properties.name,
            "gpu_uuid": _gpu_uuid(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cupy_version": cupy.__version__,
            "dtype": str(x.dtype),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "warmup": args.warmup,
            "iterations": args.iters,
            "rounds": args.rounds,
            "neuron": args.neuron,
            "inference_weight_transpose_excluded": True,
            "training_weight_transpose_included": True,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        "cases": cases,
    }
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
