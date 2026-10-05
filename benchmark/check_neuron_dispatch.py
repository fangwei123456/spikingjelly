"""Measure cached CUDA selection and unified neuron device dispatch."""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
from functools import wraps
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import torch


def _check_gpu_processes():
    uuid = str(
        torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
    ).removeprefix("GPU-")
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    others = [
        int(pid.strip())
        for gpu, pid in csv.reader(output.splitlines())
        if gpu.strip().removeprefix("GPU-") == uuid and int(pid.strip()) != os.getpid()
    ]
    if others:
        raise RuntimeError(
            f"GPU {uuid} has other compute processes {others}; discard this round "
            "and rerun on an idle GPU."
        )


def _summary(rows, seed=42):
    ratios = [r["unified"] / r["cached"] for r in rows]
    rng = random.Random(seed)
    bootstrap = sorted(
        statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000)
    )
    return {
        "median_us": {key: statistics.median(r[key] for r in rows) for key in rows[0]},
        "unified_vs_cached_percent": (statistics.median(ratios) - 1) * 100,
        "paired_bootstrap_95_percent": [(bootstrap[i] - 1) * 100 for i in (49, 1949)],
    }


def _measure(
    functions, pairs, iterations, rng, *, gpu_events=False, operations_per_call=1
):
    # Boundary checks are outside the timed region; reject persistent co-runners.
    _check_gpu_processes()
    rows = []
    for _ in range(pairs):
        order = list(functions)
        rng.shuffle(order)
        samples = {key: [] for key in order}
        for key in [*order, *reversed(order)]:
            torch.cuda.synchronize()
            if gpu_events:
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
            else:
                start = time.perf_counter_ns()
            for _ in range(iterations):
                functions[key]()
            if gpu_events:
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end) * 1000
            else:
                torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1000
            samples[key].append(elapsed / iterations / operations_per_call)
        rows.append({key: statistics.mean(values) for key, values in samples.items()})
    _check_gpu_processes()
    return {"summary": _summary(rows), "samples_us": rows}


def _arguments(kind, T, N, device):
    x = torch.randn(T, N, device=device, requires_grad=True)
    v = torch.zeros(N, device=device, requires_grad=True)
    if kind == "lif":
        return (x, v, 2.0, True, 1.0, 0.0, True, 2.0, False, 1), (x, v)
    if kind == "if":
        return (x, v, 1.0, 0.0, True, 2.0, False, 1), (x, v)
    if kind == "izhikevich":
        w = torch.zeros_like(v, requires_grad=True)
        return (
            x,
            v,
            w,
            2.3,
            -0.2,
            0.8,
            0.4,
            0.2,
            0.3,
            3.1,
            1.0,
            0.0,
            True,
            2.0,
            False,
            1,
        ), (x, v, w)
    w = torch.tensor(0.0, device=device, requires_grad=True)
    return (x, v, w, True, 1.0, 0.0, True, 2.0, False, 1), (x, v, w)


def _provider_baseline(module, device):
    selection = module._selection
    if selection.diagnostics(device)["implementation"] != "triton":
        return selection.get_trace_forward(device), ()

    # Reconstruct the removed wrapper only for timing the historical call route.
    namespace = f"sj_benchmark_{module.__name__.rsplit('.', 1)[-1]}"
    definitions = []
    for direction, select in (
        ("forward", selection.get_trace_forward),
        ("backward", selection.get_trace_backward),
    ):

        def body(implementation, reference):
            @wraps(reference)
            def call(*args, **kwargs):
                return implementation(*args, **kwargs)

            return call

        definitions.append(
            torch.library.triton_op(
                f"{namespace}::{direction}",
                body(select(device), getattr(module._cpu, f"_{direction}_impl")),
                mutates_args=(),
            )
        )
    autograd = importlib.import_module(f"{module.__name__}.autograd")
    autograd._register_ops(
        f"{namespace}::forward", f"{namespace}::backward", register_fake=False
    )
    return getattr(torch.ops, namespace).forward.default, definitions


def _round(args):
    os.environ[f"SJ_{args.neuron.upper()}_CUDA_IMPLEMENTATION"] = args.implementation
    module = importlib.import_module(
        f"spikingjelly._ops.{'if_' if args.neuron == 'if' else args.neuron}"
    )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    selection = module._selection
    direct, baseline_registrations = _provider_baseline(module, device)
    rng = random.Random(42 + args.round_index)

    def cached(*values):
        selection.get_trace_forward(values[0].device)
        return direct(*values)

    result = {
        "environment": {
            "torch": str(torch.__version__),
            "python": sys.version,
            "cuda": torch.version.cuda,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("triton",)
                if importlib.util.find_spec(name) is not None
            },
            "gpu": torch.cuda.get_device_name(device),
            "implementation": selection.diagnostics(device),
            "neuron": args.neuron,
            "device": str(device),
            "capability": torch.cuda.get_device_capability(device),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "configuration": {
            "seed": 42,
            "dtype": "float32",
            "surrogate": "ATan",
            "alpha": 2.0,
            "pairs": args.pairs,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "round_index": args.round_index,
        },
        "source_sha256": {
            str(
                source.relative_to(Path(__file__).resolve().parents[1])
            ): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in [
                Path(__file__).resolve(),
                *sorted(Path(module.__file__).parent.glob("*.py")),
                *sorted(Path(module.__file__).parent.glob("*.cu")),
                *sorted(Path(module.__file__).parent.glob("*.cuh")),
                Path(module.__file__).parents[1] / "dispatch.py",
                Path(module.__file__).parents[1] / "selection.py",
                Path(module.__file__).parents[1] / "triton_surrogate.py",
            ]
        },
        "workloads": [],
    }
    cache = []
    probe = torch.empty(0, device=device)
    for _ in range(15):
        start = time.perf_counter_ns()
        for _ in range(100000):
            selection.get_cuda_forward(probe.device)
        cache.append((time.perf_counter_ns() - start) / 100000 / 1000)
    result["cached_selection_us"] = {
        "median": statistics.median(cache),
        "samples": cache,
    }

    for T, N in args.shapes:
        values, inputs = _arguments(args.neuron, T, N, device)
        forward = {"direct": direct, "cached": cached, "unified": module._forward}
        expected = direct(*values)
        actual = module._forward(*values)
        torch.testing.assert_close(actual, expected)
        visible = 3 if args.neuron == "izhikevich" else 2
        cotangents = tuple(torch.randn_like(tensor) for tensor in expected[:visible])
        assert all(torch.isfinite(tensor).all() for tensor in expected)
        expected_grad = torch.autograd.grad(expected[:visible], inputs, cotangents)
        actual_grad = torch.autograd.grad(actual[:visible], inputs, cotangents)
        assert all(torch.isfinite(tensor).all() for tensor in expected_grad)
        torch.testing.assert_close(actual_grad, expected_grad)

        def training(fn):
            def call():
                output = fn(*values)
                return torch.autograd.grad(output[:visible], inputs, cotangents)

            return call

        def inference(fn):
            return lambda: fn(*values)

        workload = {"T": T, "N": N, "correctness": "passed"}
        for mode in ("forward", "forward_backward", "compiled_forward_backward"):
            if mode == "compiled_forward_backward":
                compiled = {
                    key: torch.compile(fn, fullgraph=True)
                    for key, fn in forward.items()
                    if key != "direct"
                }
                for fn in compiled.values():
                    output = fn(*values)
                    torch.testing.assert_close(output, expected)
                    torch.testing.assert_close(
                        torch.autograd.grad(output[:visible], inputs, cotangents),
                        expected_grad,
                    )
                functions = {key: training(fn) for key, fn in compiled.items()}
            elif mode == "forward_backward":
                functions = {key: training(fn) for key, fn in forward.items()}
            else:
                functions = {key: inference(fn) for key, fn in forward.items()}
            context = torch.no_grad if mode == "forward" else torch.enable_grad
            with context():
                for fn in functions.values():
                    for _ in range(args.warmup):
                        fn()
                workload[mode] = _measure(functions, args.pairs, args.iterations, rng)

        graphs, functions = [], {}
        with torch.no_grad():
            for key in ("cached", "unified"):
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(10):
                        forward[key](*values)
                torch.cuda.current_stream(device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(32):
                        output = forward[key](*values)
                graphs.append((graph, output))
                functions[key] = graph.replay
            workload["gpu_graph_forward"] = _measure(
                functions,
                args.pairs,
                max(100, args.iterations),
                rng,
                gpu_events=True,
                operations_per_call=32,
            )
        result["workloads"].append(workload)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    # Keep temporary benchmark registrations alive through all compiled calls.
    del baseline_registrations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--neuron", choices=("lif", "if", "plif", "izhikevich"), default="lif"
    )
    parser.add_argument(
        "--implementation", choices=("triton", "cuda", "cupy"), default="triton"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--shape", action="append", help="T,N; repeat for multiple shapes"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--round-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rounds, args.pairs, args.iterations, args.warmup) < 1:
        parser.error("rounds, pairs, iterations and warmup must be positive")
    try:
        args.shapes = [
            tuple(map(int, s.split(",")))
            for s in args.shape or ("1,32", "1,1024", "4,32768", "32,32768")
        ]
        if any(len(s) != 2 or min(s) < 1 for s in args.shapes):
            raise ValueError
    except ValueError:
        parser.error("each shape must be positive T,N")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.round_index is not None:
        _round(args)
        return
    results = []
    for index in range(args.rounds):
        output = args.output.with_name(f"{args.output.stem}.round-{index}.json")
        command = [
            sys.executable,
            "-m",
            "benchmark.check_neuron_dispatch",
            *sys.argv[1:],
            "--round-index",
            str(index),
            "--output",
            str(output),
        ]
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        results.append(json.loads(output.read_text()))
    summaries = []
    for index, (T, N) in enumerate(args.shapes):
        row = {"T": T, "N": N}
        for mode in (
            "forward",
            "forward_backward",
            "compiled_forward_backward",
            "gpu_graph_forward",
        ):
            measurements = [r["workloads"][index][mode] for r in results]
            row[mode] = {
                "rounds": [m["summary"] for m in measurements],
                "median_round_percent": statistics.median(
                    m["summary"]["unified_vs_cached_percent"] for m in measurements
                ),
                "round_range_percent": [
                    min(
                        m["summary"]["unified_vs_cached_percent"] for m in measurements
                    ),
                    max(
                        m["summary"]["unified_vs_cached_percent"] for m in measurements
                    ),
                ],
            }
        summaries.append(row)
    args.output.write_text(
        json.dumps({"rounds": results, "summary": summaries}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
