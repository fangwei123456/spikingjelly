"""Benchmark fused IF/LIF-Linear against unfused dense paths."""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
from pathlib import Path

import cupy
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


def _measure(fn, tensors, grads, backward, warmup, iters, rounds):
    def run():
        for tensor in tensors:
            tensor.grad = None
        outputs = fn()
        if backward:
            torch.autograd.backward(outputs, grads)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    times = []
    peak_bytes = 0
    for _ in range(rounds):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / iters)
        peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())
    median = statistics.median(times)
    mad = statistics.median(abs(x - median) for x in times)
    return {"rounds_ms": times, "median_ms": median, "mad_ms": mad}, peak_bytes


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


def _stable_winners(measurements):
    baselines = [k for k in measurements if not k.startswith("fused_")]
    baseline_rounds = [
        min(measurements[name]["timing"]["rounds_ms"][r] for name in baselines)
        for r in range(len(next(iter(measurements.values()))["timing"]["rounds_ms"]))
    ]
    baseline_median = statistics.median(baseline_rounds)
    baseline_mad = statistics.median(abs(x - baseline_median) for x in baseline_rounds)
    winners = []
    for name, result in measurements.items():
        if not name.startswith("fused_"):
            continue
        rounds = result["timing"]["rounds_ms"]
        wins = sum(x < y for x, y in zip(rounds, baseline_rounds))
        gap = baseline_median - result["timing"]["median_ms"]
        if wins >= 4 and gap > max(baseline_mad, result["timing"]["mad_ms"]):
            winners.append(name)
    return winners


def main():
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
                            measurements = {}
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
                                        args.rounds,
                                    )
                                    measurements[name] = {
                                        "timing": timing,
                                        "peak_memory_bytes": peak_bytes,
                                    }
                            setup_ms = _time_setup(
                                lambda: weight.t().contiguous(), args.iters
                            )
                            winners = _stable_winners(measurements)
                            print(
                                f"neuron={args.neuron:3s} mode={mode:9s} "
                                f"T={T:2d} M={M:4d} K={K:4d} "
                                f"N={N:4d} rate={actual_density:.4f} "
                                f"best={min(measurements, key=lambda n: measurements[n]['timing']['median_ms'])} "
                                f"stable={','.join(winners) or '-'}"
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
                                    "stable_fused_winners": winners,
                                    "measurements": measurements,
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
            "compute_capability": f"{properties.major}.{properties.minor}",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cupy_version": cupy.__version__,
            "warmup": args.warmup,
            "iterations": args.iters,
            "rounds": args.rounds,
            "neuron": args.neuron,
            "inference_weight_transpose_excluded": True,
            "training_weight_transpose_included": True,
        },
        "cases": cases,
    }
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
