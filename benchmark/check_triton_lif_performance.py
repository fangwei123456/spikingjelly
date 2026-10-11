"""Check registered Triton LIF backward against production on a real workload."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

from benchmark.benchmark_snn_single_gpu import DEFAULT_SEED, _build_model


_SURROGATES = (
    "Sigmoid",
    "ATan",
    "PiecewiseQuadratic",
    "PiecewiseExp",
    "SoftSign",
    "SuperSpike",
    "Erf",
)


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_percent(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-gpu-name")
    parser.add_argument("--surrogate", choices=_SURROGATES, default="ATan")
    parser.add_argument("--rounds", type=_positive_int, default=3)
    parser.add_argument("--pairs", type=_positive_int, default=20)
    parser.add_argument("--replays", type=_positive_int, default=5)
    parser.add_argument("--warmup-steps", type=_positive_int, default=50)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--T", type=_positive_int, default=4)
    parser.add_argument("--image-size", type=_positive_int, default=224)
    parser.add_argument(
        "--max-regression-percent", type=_nonnegative_percent, default=2.0
    )
    parser.add_argument("--round-index", type=_positive_int, help=argparse.SUPPRESS)
    return parser


def _summarize(rounds, max_regression_percent):
    medians = []
    samples = []
    for result in rounds:
        rows = result["samples"]
        if not rows or any(
            not math.isfinite(row[key]) or row[key] <= 0
            for row in rows
            for key in ("baseline_ms", "candidate_ms")
        ):
            raise ValueError("each round needs finite, positive paired timings")
        ratios = [row["candidate_ms"] / row["baseline_ms"] for row in rows]
        medians.append(statistics.median(ratios))
        samples.extend(rows)
    if not medians:
        raise ValueError("at least one complete round is required")
    limit = 1 + max_regression_percent / 100
    over = [ratio > limit for ratio in medians]
    spreads = [
        max(
            abs(row[key][0] - row[key][1]) / statistics.mean(row[key]) * 100
            for key in ("baseline_samples_ms", "candidate_samples_ms")
        )
        for row in samples
    ]
    unstable_pairs = sum(spread > max_regression_percent for spread in spreads)
    if unstable_pairs:
        status, reason = "inconclusive", "repeat_spread_exceeds_limit"
    elif all(over):
        status, reason = "fail", "regression"
    elif any(over):
        status, reason = "inconclusive", "rounds_disagree"
    else:
        status, reason = "pass", "within_limit"
    return {
        "status": status,
        "reason": reason,
        "max_regression_percent": max_regression_percent,
        "round_median_percent": [(ratio - 1) * 100 for ratio in medians],
        "median_percent": (statistics.median(medians) - 1) * 100,
        "baseline_median_ms": statistics.median(r["baseline_ms"] for r in samples),
        "candidate_median_ms": statistics.median(r["candidate_ms"] for r in samples),
        "pair_count": len(samples),
        "unstable_pair_count": unstable_pairs,
        "max_repeat_spread_percent": max(spreads),
        "worst_pair_percent": max(
            (r["candidate_ms"] / r["baseline_ms"] - 1) * 100 for r in samples
        ),
    }


def _check_gradient(actual, expected):
    scale = expected.abs().max().item()
    if not math.isfinite(scale):
        raise RuntimeError("non-finite reference gradient")
    # Captured training gradients can be far smaller than a fixed 1e-5 tolerance.
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6 * scale)
    return {
        "max_absolute_error": (actual - expected).abs().max().item(),
        "reference_max_absolute_value": scale,
    }


def _capture_workload(args, kernel_module):
    from spikingjelly.activation_based import functional, surrogate
    from spikingjelly._ops.surrogate_dispatch import (
        resolve_sg_triton_id_and_alpha,
    )

    model = (
        _build_model("spikformer_s", args.T, 1000, "lif", args.surrogate)
        .to(device=args.device, dtype=torch.float32)
        .train()
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=args.device,
        dtype=torch.float32,
    )
    y = torch.randint(1000, (args.batch_size,), device=args.device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x).mean(0), y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite training loss during workload preparation")
        loss.backward()
        optimizer.step()
        functional.reset_net(model)

    records = []
    original = kernel_module._backward_impl
    capture_enabled = False

    def capture(
        gs,
        gv,
        h,
        tau,
        decay_input,
        threshold,
        reset,
        detach_reset,
        alpha,
        store_v_seq,
        surrogate_id,
        *,
        _kernel_wrapper=None,
    ):
        if capture_enabled:
            records.append(
                {
                    "inputs": tuple(
                        t.detach().clone(memory_format=torch.contiguous_format)
                        for t in (gs, gv, h)
                    ),
                    "source_shapes": [list(t.shape) for t in (gs, gv, h)],
                    "source_strides": [list(t.stride()) for t in (gs, gv, h)],
                    "parameters": {
                        "tau": tau,
                        "decay_input": decay_input,
                        "v_threshold": threshold,
                        "v_reset": 0.0 if reset is None else reset,
                        "soft_reset": reset is None,
                        "detach_reset": detach_reset,
                        "sg_alpha": alpha,
                        "compute_dtype_id": 0,
                        "storage_dtype_id": 0,
                        "store_v_seq": store_v_seq,
                        "sg_triton_id": surrogate_id,
                    },
                }
            )
        return original(
            gs,
            gv,
            h,
            tau,
            decay_input,
            threshold,
            reset,
            detach_reset,
            alpha,
            store_v_seq,
            surrogate_id,
            _kernel_wrapper=_kernel_wrapper,
        )

    kernel_module._backward_impl = capture
    try:
        for _ in range(args.warmup_steps):
            step()
        torch.cuda.synchronize()
        capture_enabled = True
        step()
    finally:
        capture_enabled = False
        kernel_module._backward_impl = original
    torch.cuda.synchronize()
    if not records:
        raise RuntimeError("no production LIF backward calls were captured")
    sid, alpha = resolve_sg_triton_id_and_alpha(getattr(surrogate, args.surrogate)())
    for record in records:
        parameters = record["parameters"]
        if (parameters["sg_triton_id"], parameters["sg_alpha"]) != (sid, alpha):
            raise RuntimeError("captured surrogate does not match the requested one")
        for tensor in record["inputs"]:
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
                raise RuntimeError("workload needs finite FP32 backward inputs")
    return records


def _make_graphs(records, baseline, candidate):
    def candidate_chain():
        result = []
        for record in records:
            gs, gv, h = record["inputs"]
            p = record["parameters"]
            result.append(
                candidate(
                    gs,
                    gv,
                    h,
                    p["tau"],
                    p["decay_input"],
                    p["v_threshold"],
                    None if p["soft_reset"] else p["v_reset"],
                    p["detach_reset"],
                    p["sg_alpha"],
                    p["store_v_seq"],
                    p["sg_triton_id"],
                )
            )
        return result

    for _ in range(3):
        candidate_chain()
    torch.cuda.synchronize()
    candidate_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(candidate_graph):
        outputs = candidate_chain()

    def baseline_chain():
        for record, (gx, gv0) in zip(records, outputs):
            baseline(*record["inputs"], gx, gv0, **record["parameters"])

    for _ in range(3):
        baseline_chain()
    torch.cuda.synchronize()
    baseline_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(baseline_graph):
        baseline_chain()
    graphs = {"baseline": baseline_graph, "candidate": candidate_graph}
    checks = {}
    for name, graph in graphs.items():
        # Empty/incorrectly captured graphs must not pass using stale outputs.
        for pair in outputs:
            for tensor in pair:
                tensor.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        checks[name] = []
        for record, actual in zip(records, outputs):
            reference = tuple(torch.empty_like(t).fill_(float("nan")) for t in actual)
            baseline(*record["inputs"], *reference, **record["parameters"])
            checks[name].append(
                [_check_gradient(got, want) for got, want in zip(actual, reference)]
            )
    return graphs, outputs, checks


def _measure(graph, replays):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / replays


def _paired_samples(graphs, pairs, replays, round_index):
    samples = []
    for pair in range(pairs):
        order = ("baseline", "candidate", "candidate", "baseline")
        if (pair + round_index) % 2 == 0:
            order = ("candidate", "baseline", "baseline", "candidate")
        values = {key: [] for key in graphs}
        for key in order:
            values[key].append(_measure(graphs[key], replays))
        samples.append(
            {
                "pair": pair,
                "order": list(order),
                "baseline_samples_ms": values["baseline"],
                "candidate_samples_ms": values["candidate"],
                "baseline_ms": statistics.mean(values["baseline"]),
                "candidate_ms": statistics.mean(values["candidate"]),
            }
        )
    return samples


def _run_round(args):
    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("this benchmark requires NVIDIA CUDA and Triton")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device")
    torch.cuda.set_device(device)
    allocator = torch.cuda.get_allocator_backend()
    if allocator != "native":
        raise RuntimeError(
            "shared graph output buffers require PyTorch's native CUDA allocator"
        )
    properties = torch.cuda.get_device_properties(device)
    if args.require_gpu_name and args.require_gpu_name not in properties.name:
        raise RuntimeError(
            f"expected {args.require_gpu_name!r}, found {properties.name!r}"
        )
    import triton
    from spikingjelly._ops import triton_surrogate as surrogate_math
    from spikingjelly._ops.lif import triton as candidate_module
    from spikingjelly._ops import surrogate_dispatch as surrogate_kernel
    from spikingjelly._ops.lif import triton_precision as production

    sources = [Path(__file__), Path(sys.modules[_build_model.__module__].__file__)]
    sources += [
        Path(m.__file__)
        for m in (production, candidate_module, surrogate_math, surrogate_kernel)
    ]
    metadata = {
        "host": platform.node(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "device": str(device),
        "gpu": str(properties),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "allocator": allocator,
        "sources": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    }
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = _capture_workload(args, candidate_module)
    from spikingjelly.activation_based import functional

    selected = functional.neuron_implementation("lif", device)["implementation"]
    if selected != "triton":
        raise RuntimeError(
            "This check measures the production Triton path. Set "
            "SJ_LIF_CUDA_IMPLEMENTATION=triton before starting Python."
        )
    with torch.no_grad():
        graphs, outputs, checks = _make_graphs(
            records,
            production._launch_lif_backward_kernel,
            candidate_module._backward_impl,
        )
        start = time.monotonic()
        while time.monotonic() - start < 3:
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
        samples = _paired_samples(graphs, args.pairs, args.replays, args.round_index)
        # Keep graph-owned output buffers alive through all replays.
        assert len(outputs) == len(records)
    return {
        "status": "complete",
        "round": args.round_index,
        "metadata": metadata,
        "workload": {
            "model": "spikformer_s",
            "dtype": "float32",
            "surrogate": args.surrogate,
            "seed": DEFAULT_SEED,
            "batch_size": args.batch_size,
            "T": args.T,
            "image_size": args.image_size,
            "warmup_steps": args.warmup_steps,
            "optimizer": {"name": "SGD", "lr": 0.1, "momentum": 0.9},
            "backward_calls_per_replay": len(records),
            "replays_per_sample": args.replays,
            "records": [{k: v for k, v in r.items() if k != "inputs"} for r in records],
        },
        "correctness": checks,
        "samples": samples,
    }


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def _main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.pairs < 2:
        parser.error("--pairs must be at least 2 to exercise both measurement orders")
    if args.round_index is not None:
        try:
            report = _run_round(args)
        except Exception as error:
            _write(
                args.output,
                {"status": "error", "error": f"{type(error).__name__}: {error}"},
            )
            raise
        _write(args.output, report)
        return 0

    report = {
        "schema_version": 1,
        "status": "running",
        "protocol": (
            "FP32 production vs registered Triton LIF backward; identical contiguous "
            "inputs and output addresses; CUDA Graph ABBA/BAAB; independent processes"
        ),
        "max_regression_percent": args.max_regression_percent,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "round_index"
        },
        "rounds": [],
    }
    _write(args.output, report)
    for index in range(1, args.rounds + 1):
        path = args.output.with_name(f"{args.output.stem}.round-{index}.json")
        log = path.with_suffix(".log")
        command = [sys.executable, "-m", "benchmark.check_triton_lif_performance"]
        for key in (
            "device",
            "surrogate",
            "pairs",
            "replays",
            "warmup_steps",
            "batch_size",
            "T",
            "image_size",
            "require_gpu_name",
        ):
            value = getattr(args, key)
            if value is not None:
                command.extend(["--" + key.replace("_", "-"), str(value)])
        command.extend(["--round-index", str(index), "--output", str(path)])
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            report.update(status="error", failed_round=index, error_log=str(log))
            _write(args.output, report)
            print(f"ERROR: round {index}; see {log}", file=sys.stderr)
            return 2
        report["rounds"].append(json.loads(path.read_text()))
        _write(args.output, report)
        print(f"Completed round {index}/{args.rounds}", flush=True)
    report["summary"] = _summarize(report["rounds"], args.max_regression_percent)
    report["status"] = report["summary"]["status"]
    _write(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    return {"pass": 0, "fail": 1, "inconclusive": 2}[report["status"]]


if __name__ == "__main__":
    sys.exit(_main())
