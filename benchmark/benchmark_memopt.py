import argparse
import copy
import json
import time

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, neuron
from spikingjelly.activation_based.memopt import memory_optimization


class MemOptToyNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Linear(channels, channels),
            neuron.IFNode(step_mode="m"),
            nn.Linear(channels, channels),
            neuron.IFNode(step_mode="m"),
            nn.Linear(channels, channels),
            neuron.IFNode(step_mode="m"),
        )

    def forward(self, x):
        return self.blocks(x)


class MemOptBlock(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(
            nn.Linear(channels, channels),
            neuron.IFNode(step_mode="m"),
            nn.Linear(channels, channels),
            neuron.IFNode(step_mode="m"),
        )
        self.n_seq_inputs = 1
        self.n_outputs = 1

    def __spatial_split__(self):
        return [
            nn.Sequential(self[0], self[1]),
            nn.Sequential(self[2], self[3]),
        ]


class MemOptBlockNet(nn.Module):
    def __init__(self, channels: int, depth: int = 3):
        super().__init__()
        self.blocks = nn.Sequential(*[MemOptBlock(channels) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)


def build_case(model_kind: str, channels: int):
    if model_kind == "neuron":
        return MemOptToyNet(channels), neuron.IFNode
    if model_kind == "block":
        return MemOptBlockNet(channels), MemOptBlock
    raise ValueError(f"Unsupported model_kind={model_kind!r}")


def train_step(net: nn.Module, x: torch.Tensor):
    net.train()
    net.zero_grad(set_to_none=True)
    y = net(x)
    loss = y.sum()
    loss.backward()
    net.zero_grad(set_to_none=True)
    functional.reset_net(net)


def benchmark_train_step(net: nn.Module, x: torch.Tensor, warmup: int, iters: int):
    for _ in range(warmup):
        train_step(net, x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        train_step(net, x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "step_ms": elapsed * 1000.0 / iters,
        "peak_allocated_mb": peak_allocated / 1024 / 1024,
        "peak_reserved_mb": peak_reserved / 1024 / 1024,
    }


def optimize_model(
    net,
    instance,
    x,
    level: int,
    warmup_in_main_process: bool,
    warmup_in_profile_workers: bool,
):
    t0 = time.perf_counter()
    optimized = memory_optimization(
        net,
        instance,
        dummy_input=(x,),
        compress_x=True,
        level=level,
        warmup_in_main_process=warmup_in_main_process,
        warmup_in_profile_workers=warmup_in_profile_workers,
    )
    optimize_ms = (time.perf_counter() - t0) * 1000.0
    return optimized, optimize_ms


def run_single_variant(
    base,
    instance,
    x,
    level: int,
    warmup_in_main_process: bool,
    warmup_in_profile_workers: bool,
    warmup: int,
    iters: int,
):
    model, optimize_ms = optimize_model(
        copy.deepcopy(base).cpu(),
        instance,
        x.detach().cpu(),
        level=level,
        warmup_in_main_process=warmup_in_main_process,
        warmup_in_profile_workers=warmup_in_profile_workers,
    )
    model = model.to(x.device)
    result = benchmark_train_step(model, x, warmup, iters)
    result["optimize_ms"] = optimize_ms
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=("neuron", "block"), default="neuron")
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--C", type=int, default=512)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_memopt.py")

    device = torch.device("cuda")
    x = torch.randn(args.T, args.N, args.C, device=device)
    base, instance = build_case(args.model_kind, args.C)
    base = base.to(device)

    baseline = benchmark_train_step(copy.deepcopy(base), x, args.warmup, args.iters)
    results = {
        "model_kind": args.model_kind,
        "shape": {"T": args.T, "N": args.N, "C": args.C},
        "baseline": baseline,
        "levels": {},
    }

    for level in args.levels:
        if level == 0:
            continue
        results["levels"][str(level)] = {
            "warm_main": run_single_variant(
                base, instance, x, level, True, True, args.warmup, args.iters
            ),
            "no_main_warmup": run_single_variant(
                base, instance, x, level, False, True, args.warmup, args.iters
            ),
            "no_profile_worker_warmup": run_single_variant(
                base, instance, x, level, False, False, args.warmup, args.iters
            ),
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
