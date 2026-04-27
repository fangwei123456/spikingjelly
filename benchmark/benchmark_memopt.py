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


def optimize_model(net, x, level: int, warmup_in_main_process: bool):
    t0 = time.perf_counter()
    optimized = memory_optimization(
        net,
        neuron.IFNode,
        dummy_input=(x,),
        compress_x=True,
        level=level,
        warmup_in_main_process=warmup_in_main_process,
    )
    optimize_ms = (time.perf_counter() - t0) * 1000.0
    return optimized, optimize_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--C", type=int, default=512)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_memopt.py")

    device = torch.device("cuda")
    x = torch.randn(args.T, args.N, args.C, device=device)
    base = MemOptToyNet(args.C).to(device)

    baseline = benchmark_train_step(copy.deepcopy(base), x, args.warmup, args.iters)

    warm_model, warm_opt_ms = optimize_model(
        copy.deepcopy(base).cpu(),
        x.detach().cpu(),
        level=args.level,
        warmup_in_main_process=True,
    )
    warm_model = warm_model.to(device)
    warm_result = benchmark_train_step(warm_model, x, args.warmup, args.iters)

    cold_model, cold_opt_ms = optimize_model(
        copy.deepcopy(base).cpu(),
        x.detach().cpu(),
        level=args.level,
        warmup_in_main_process=False,
    )
    cold_model = cold_model.to(device)
    cold_result = benchmark_train_step(cold_model, x, args.warmup, args.iters)

    result = {
        "shape": {"T": args.T, "N": args.N, "C": args.C},
        "level": args.level,
        "baseline": baseline,
        "optimized_warm_main": {
            "optimize_ms": warm_opt_ms,
            **warm_result,
        },
        "optimized_no_main_warmup": {
            "optimize_ms": cold_opt_ms,
            **cold_result,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
