import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import triton


source_root = Path(
    os.getenv("SJ_BENCH_SOURCE_ROOT", Path(__file__).resolve().parents[1])
)
sys.path.insert(0, str(source_root))

from spikingjelly.activation_based.neuron.integrate_and_fire import IFNode
from spikingjelly.activation_based.neuron.lif import LIFNode


CASES = tuple(
    (kind, mode, T, N, dtype)
    for kind in ("if", "lif")
    for mode in ("inference", "training")
    for T, N, dtype in (
        (16, 4096, torch.float32),
        (32, 16381, torch.float16),
        (65, 16381, torch.float32),
    )
)


def percentile(values, q):
    """Return an interpolated quantile from a non-empty sequence."""
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def make_node(kind):
    """Create the requested Triton neuron in last-state mode."""
    cls = IFNode if kind == "if" else LIFNode
    return cls(step_mode="m", backend="triton", store_v_seq=False).cuda()


def run_case(
    kind,
    mode,
    T,
    N,
    dtype,
    warmup,
    repetitions,
    iterations_per_repetition,
):
    """Measure latency and peak allocation for one benchmark case."""
    torch.manual_seed(20260717)
    x = torch.randn(
        T,
        N,
        device="cuda",
        dtype=dtype,
        requires_grad=mode == "training",
    )
    node = make_node(kind)

    if mode == "training":

        def step():
            """Run one forward and backward benchmark iteration."""
            node.reset()
            x.grad = None
            spikes = node(x)
            spikes.sum().backward()

    else:

        def step():
            """Run one inference benchmark iteration."""
            node.reset()
            with torch.inference_mode():
                node(x)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    samples_ms = []
    for _ in range(repetitions):
        start_event.record()
        for _ in range(iterations_per_repetition):
            step()
        end_event.record()
        end_event.synchronize()
        samples_ms.append(
            start_event.elapsed_time(end_event) / iterations_per_repetition
        )

    node.reset()
    x.grad = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - allocated_before

    return {
        "neuron": kind,
        "mode": mode,
        "T": T,
        "N": N,
        "dtype": str(dtype).removeprefix("torch."),
        "warmup": warmup,
        "repetitions": repetitions,
        "iterations_per_repetition": iterations_per_repetition,
        "median_ms": statistics.median(samples_ms),
        "p25_ms": percentile(samples_ms, 0.25),
        "p75_ms": percentile(samples_ms, 0.75),
        "peak_delta_bytes": peak_delta,
        "samples_ms": samples_ms,
    }


def main():
    """Run the benchmark matrix and write the results as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--iterations-per-repetition", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for case in CASES:
        result = run_case(
            *case,
            args.warmup,
            args.repetitions,
            args.iterations_per_repetition,
        )
        results.append(result)
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "samples_ms"}
            ),
            flush=True,
        )

    payload = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
