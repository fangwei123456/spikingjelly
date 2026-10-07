"""Offline CUDA implementation calibration; production never runs this search."""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch

from .check_neuron_dispatch import _arguments, _check_gpu_processes


FAMILIES = (
    "if",
    "lif",
    "plif",
    "qif",
    "eif",
    "izhikevich",
    "ilif",
    "activation_aware_if",
    "stbif",
)
IMPLEMENTATIONS = ("triton", "cuda", "cupy", "torch")


def _priority(scores):
    remaining, order = set(IMPLEMENTATIONS), []
    while remaining:
        best = min(scores[name] for name in remaining)
        tied = {name for name in remaining if scores[name] <= best * 1.05}
        order.extend(name for name in IMPLEMENTATIONS if name in tied)
        remaining -= tied
    return order


def _rank(records):
    grouped = {}
    for record in records:
        key = (
            tuple(record["capability"]),
            record["neuron"],
            record.get("execution", "eager"),
        )
        grouped.setdefault(key, {}).setdefault(record["implementation"], []).append(
            record
        )
    result = []
    for (capability, neuron, execution), implementations in sorted(grouped.items()):
        if set(implementations) != set(IMPLEMENTATIONS):
            raise ValueError(f"Incomplete calibration for {capability}/{neuron}")
        environments = {
            (
                r["gpu"],
                r["torch"],
                r["cuda"],
                r["warmup"],
                r["samples"],
                r["iterations"],
            )
            for rounds in implementations.values()
            for r in rounds
        }
        if len(environments) != 1:
            raise ValueError(
                "Calibration GPU and software versions must match, including sampling"
            )
        sources = {}
        for rounds in implementations.values():
            for record in rounds:
                for path, digest in record["source_sha256"].items():
                    if "/benchmark/" in path:
                        continue
                    if path in sources and sources[path] != digest:
                        raise ValueError(f"Calibration operator source changed: {path}")
                    sources[path] = digest
        scores, spreads, round_scores_by_impl = {}, {}, {}
        reference_profiles = None
        reference_rounds = None
        for implementation, rounds in implementations.items():
            rounds = sorted(rounds, key=lambda row: row["round"])
            round_ids = [row["round"] for row in rounds]
            if reference_rounds is None:
                reference_rounds = round_ids
            if round_ids != reference_rounds:
                raise ValueError("Calibration process rounds must match")
            if len(rounds) < 3 or len({r["round"] for r in rounds}) != len(rounds):
                raise ValueError("Three distinct process rounds are required")
            profiles = [
                (r["dtype"], r["T"], r["N"], r["mode"])
                for r in rounds[0]["measurements"]
            ]
            if reference_profiles is None:
                reference_profiles = profiles
            if profiles != reference_profiles or any(
                [(r["dtype"], r["T"], r["N"], r["mode"]) for r in row["measurements"]]
                != profiles
                for row in rounds
            ):
                raise ValueError("Calibration profiles must match")
            weighted, weights = [], []
            for index, profile in enumerate(profiles):
                values = [r["measurements"][index]["median_us"] for r in rounds]
                if any(not math.isfinite(v) or v <= 0 for v in values):
                    raise ValueError("Calibration needs finite positive timings")
                weight = 2 if profile[-1] == "training" else 1
                weighted.append(weight * math.log(statistics.median(values)))
                weights.append(weight)
            scores[implementation] = math.exp(sum(weighted) / sum(weights))
            round_scores = [
                math.exp(
                    sum(
                        w * math.log(x["median_us"])
                        for w, x in zip(weights, row["measurements"], strict=True)
                    )
                    / sum(weights)
                )
                for row in rounds
            ]
            round_scores_by_impl[implementation] = round_scores
            spreads[implementation] = (max(round_scores) / min(round_scores) - 1) * 100
        order = _priority(scores)
        round_orders = [
            _priority(
                {name: values[index] for name, values in round_scores_by_impl.items()}
            )
            for index in range(len(next(iter(round_scores_by_impl.values()))))
        ]
        stable = all(round_order == order for round_order in round_orders)
        result.append(
            {
                "capability": capability,
                "neuron": neuron,
                "execution": execution,
                "priority": order if stable else None,
                "status": "stable" if stable else "inconclusive",
                "round_priorities": round_orders,
                "score_us": scores,
                "maximum_round_spread_percent": spreads,
            }
        )
    return result


def _measure(call, iterations, samples):
    timings = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            call()
        torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - start) / iterations / 1000)
    return {"median_us": statistics.median(timings), "samples_us": timings}


def _case(args):
    os.environ[f"SJ_{args.neuron.upper()}_CUDA_IMPLEMENTATION"] = args.implementation
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _check_gpu_processes()
    package = importlib.import_module(
        f"spikingjelly._ops.{'if_' if args.neuron == 'if' else args.neuron}"
    )
    binding_start = time.perf_counter()
    selected = package._selection.diagnostics(
        torch.device(args.device),
        execution="compile" if args.execution == "compile" else "eager",
    )
    binding_ms = (time.perf_counter() - binding_start) * 1000
    if args.implementation != "auto":
        assert selected["implementation"] == args.implementation
    forward = (
        torch.compile(package._forward, fullgraph=True)
        if args.execution == "compile"
        else package._forward
    )
    measurements = []
    for dtype_name in args.dtype:
        dtype = getattr(torch, dtype_name)
        for shape in args.shape:
            T, N = map(int, shape.split(","))
            torch.manual_seed(42)
            values, inputs = _arguments(args.neuron, T, N, args.device, dtype)
            started = time.perf_counter()
            actual = forward(*values)
            torch.cuda.synchronize()
            first_forward_ms = (time.perf_counter() - started) * 1000
            expected = package._selection._cpu_forward(*values)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
            visible = 3 if args.neuron == "izhikevich" else 2
            gradients = tuple(torch.randn_like(t) for t in actual[:visible])
            if inputs:
                actual_grad = torch.autograd.grad(actual[:visible], inputs, gradients)
                expected_grad = torch.autograd.grad(
                    expected[:visible], inputs, gradients
                )
                for got, want in zip(actual_grad, expected_grad, strict=True):
                    if got.dtype == torch.float32:
                        torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-5)
                    else:
                        # Stored FP16/BF16 gradients can differ by one rounded ULP.
                        torch.testing.assert_close(got, want)
            for mode in ("inference", "training") if inputs else ("inference",):

                def call():
                    output = forward(*values)
                    if mode == "training":
                        return torch.autograd.grad(output[:visible], inputs, gradients)
                    return output

                with torch.enable_grad() if mode == "training" else torch.no_grad():
                    for _ in range(args.warmup):
                        call()
                    if args.execution == "cuda_graph":
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(
                            graph, stream=torch.cuda.current_stream()
                        ):
                            captured = call()
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            captured, call(), rtol=2e-4, atol=2e-5
                        )
                        measurement = _measure(
                            graph.replay, args.iterations, args.samples
                        )
                    else:
                        measurement = _measure(call, args.iterations, args.samples)
                measurements.append(
                    {
                        "dtype": dtype_name,
                        "T": T,
                        "N": N,
                        "mode": mode,
                        "first_forward_ms": first_forward_ms,
                        **measurement,
                    }
                )
    _check_gpu_processes()
    root = Path(package.__file__).resolve().parents[1]
    sources = [Path(__file__), root / "selection.py", root / "dispatch.py"]
    sources += sorted(Path(package.__file__).parent.glob("*.py"))
    sources += sorted(Path(package.__file__).parent.glob("*.cu"))
    sources += sorted(Path(package.__file__).parent.glob("*.cuh"))
    sources += sorted(Path(package.__file__).parent.glob("_C*.so"))
    sources += sorted(Path(package.__file__).parent.glob("_native_build.json"))
    sources += [root / "_cuda.cuh", root / "cuda_surrogate.cuh"]
    return {
        "timing_domain": f"synchronized_{args.execution}_wall_time",
        "execution": args.execution,
        "neuron": args.neuron,
        "implementation": args.implementation,
        "selected_implementation": selected["implementation"],
        "round": args.round,
        "capability": torch.cuda.get_device_capability(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "binding_ms": binding_ms,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "samples": args.samples,
        "measurements": measurements,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--neuron", choices=FAMILIES, default="lif")
    parser.add_argument(
        "--implementation", choices=("auto", *IMPLEMENTATIONS), default="triton"
    )
    parser.add_argument(
        "--execution", choices=("eager", "compile", "cuda_graph"), default="eager"
    )
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=("float32", "float16", "bfloat16"),
        default=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--shape", nargs="+", default=["1,512", "4,32768", "16,32768", "4,2097152"]
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.samples) < 1:
        parser.error("warmup, iterations and samples must be positive")
    if args.summarize:
        result = _rank(
            [json.loads(p.read_text()) for p in sorted(args.summarize.glob("*.json"))]
        )
    else:
        with torch.cuda.stream(torch.cuda.Stream(device=args.device)):
            result = _case(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
