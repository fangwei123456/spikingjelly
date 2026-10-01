"""Isolated neuron/layout-chain timing and CUDA-profiler capture workload.

Launch this script with the NSYS CLI for capture; --profile only controls the
capture window and NVTX ranges. Use separate runs without --profile for timing.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from spikingjelly import nsys
from spikingjelly.activation_based import functional, layer, neuron


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("triton", "cupy"), required=True)
    parser.add_argument("--workload", choices=("neuron", "chain"), default="neuron")
    parser.add_argument(
        "--layout", choices=("contiguous", "channels-last"), required=True
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("--steps must be positive and --warmup must be non-negative")
    torch.manual_seed(20261001)
    model = neuron.LIFNode(step_mode="m", backend=args.backend)
    if args.workload == "chain":
        model = torch.nn.Sequential(
            layer.Conv2d(64, 64, 3, padding=1, step_mode="m"),
            layer.BatchNorm2d(64, step_mode="m"),
            model,
            layer.Conv2d(64, 64, 1, step_mode="m"),
        )
    model = model.cuda().half().train()
    x = torch.randn(4, 16, 64, 28, 28, device="cuda", dtype=torch.float16)
    if args.layout == "channels-last":
        x = x.flatten(0, 1).contiguous(memory_format=torch.channels_last).view_as(x)
        model.to(memory_format=torch.channels_last)
    x.requires_grad_()
    run = torch.compile(model, fullgraph=True) if args.compile else model

    def iteration(index, mark=False):
        with nsys.step(index, "training", mark):
            with nsys.region("forward", mark):
                output = run(x)
            with nsys.region("loss", mark):
                loss = output.float().square().mean()
            with nsys.region("backward", mark):
                loss.backward()
            with nsys.region("reset", mark):
                functional.reset_net(model)
                model.zero_grad(set_to_none=True)
                x.grad = None

    for i in range(args.warmup):
        iteration(i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
    with nsys.capture(args.profile):
        for i in range(args.steps):
            starts[i].record()
            iteration(i, args.profile)
            ends[i].record()
        ends[-1].synchronize()
    samples = [a.elapsed_time(b) for a, b in zip(starts, ends, strict=True)]
    result = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "input_shape": list(x.shape),
        "input_stride": list(x.stride()),
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"median_ms": result["median_ms"]}))


if __name__ == "__main__":
    main()
