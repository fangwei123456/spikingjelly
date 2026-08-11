"""
Benchmark: FlexSN CUDA backend variants vs torch reference.

Tests two scenarios:
  1. Pure single FlexSN layer
  2. Linear -> FlexSN -> Linear fusion

Compared variants:
  - backend="torch"
  - backend="triton"
  - backend="triton" + torch.compile(fullgraph=True)

FlexSN uses ``backend="triton"`` for its dedicated CUDA kernels. This benchmark
compares that path with the Torch backend, with and without outer compilation.

Usage (run from repo root):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python benchmark/flexsn/flex_sn_compile.py
"""

import math
import sys
from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.neuron.flexsn import FlexSN


def lif_core(x: torch.Tensor, v: torch.Tensor):
    tau, v_th = 2.0, 1.0
    h = v + (x - v) / tau
    s = (h >= v_th).to(h.dtype)
    return s, h * (1.0 - s)


def make_flexsn(backend: str) -> FlexSN:
    return FlexSN(
        core=lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend=backend,
    ).cuda()


def cuda_time_ms(
    fn: Callable[[], Any],
    warmup: int = 10,
    iters: int = 200,
    reset_hook: Optional[Callable[[], Any]] = None,
) -> float:
    for _ in range(warmup):
        if reset_hook is not None:
            reset_hook()
        fn()
    torch.cuda.synchronize()
    elapsed = 0.0
    for _ in range(iters):
        if reset_hook is not None:
            reset_hook()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        elapsed += start.elapsed_time(end)
    return elapsed / iters


def speedup(base_ms: float, candidate_ms: float) -> float:
    if candidate_ms <= 0:
        return float("inf")
    return base_ms / candidate_ms


def speedup_flag(ratio: float) -> str:
    if math.isinf(ratio):
        return "FAST"
    if ratio >= 1.5:
        return "FAST"
    if ratio >= 1.1:
        return "GOOD"
    return "CLOSE"


def bench_single_layer():
    print("=" * 92)
    print("Benchmark 1 — single FlexSN layer (forward only, no grad)")
    print(
        f"  {'T':>4} {'B':>5} {'N':>6}  "
        f"{'torch':>10}  {'triton':>10}  {'compile':>10}  "
        f"{'tri/torch':>9}  {'cmp/torch':>9}"
    )
    print("-" * 92)

    configs = [(8, 128, 1024), (32, 128, 1024), (8, 128, 4096)]
    for T, B, N in configs:
        x = torch.randn(T, B, N, device="cuda")

        n_torch = make_flexsn("torch")
        n_triton = make_flexsn("triton")
        n_triton_compiled = make_flexsn("triton")
        compiled = torch.compile(n_triton_compiled, fullgraph=True)

        with torch.no_grad():
            ms_torch = cuda_time_ms(
                lambda n=n_torch, x=x: n(x), reset_hook=(lambda r=n_torch.reset: r())
            )
            ms_triton = cuda_time_ms(
                lambda n=n_triton, x=x: n(x),
                reset_hook=(lambda r=n_triton.reset: r()),
            )
            ms_triton_compiled = cuda_time_ms(
                lambda c=compiled, x=x: c(x),
                reset_hook=(lambda r=n_triton_compiled.reset: r()),
            )

        eager_speedup = speedup(ms_torch, ms_triton)
        compile_speedup = speedup(ms_torch, ms_triton_compiled)
        print(
            f"  {T:>4} {B:>5} {N:>6}  "
            f"{ms_torch:>9.3f}  {ms_triton:>9.3f}  {ms_triton_compiled:>9.3f}  "
            f"{eager_speedup:>8.2f}x  {compile_speedup:>8.2f}x"
        )


class SeqModelFused(nn.Module):
    """Linear -> FlexSN -> Linear with batch-matmul over T for compile coverage."""

    def __init__(self, hidden: int, backend: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.neuron = make_flexsn(backend)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, N = x.shape
        x_flat = x.reshape(T * B, N)
        h = self.fc1(x_flat).reshape(T, B, N)
        spike = self.neuron(h)
        return self.fc2(spike.reshape(T * B, N)).reshape(T, B, N)


def bench_linear_flexsn_linear():
    print()
    print("=" * 92)
    print("Benchmark 2 — Linear -> FlexSN -> Linear (forward only, no grad)")
    print(
        f"  {'T':>4} {'B':>5} {'N':>6}  "
        f"{'torch':>10}  {'triton':>10}  {'compile':>10}  "
        f"{'tri/torch':>9}  {'cmp/torch':>9}"
    )
    print("-" * 92)

    configs = [(8, 128, 1024), (32, 128, 1024)]
    for T, B, N in configs:
        x = torch.randn(T, B, N, device="cuda")

        m_torch = SeqModelFused(N, "torch").cuda()
        m_triton = SeqModelFused(N, "triton").cuda()
        m_triton_compiled = SeqModelFused(N, "triton").cuda()
        compiled = torch.compile(m_triton_compiled, fullgraph=True)

        with torch.no_grad():
            ms_torch = cuda_time_ms(
                lambda m=m_torch, x=x: m(x),
                reset_hook=(lambda m=m_torch: functional.reset_net(m)),
            )
            ms_triton = cuda_time_ms(
                lambda m=m_triton, x=x: m(x),
                reset_hook=(lambda m=m_triton: functional.reset_net(m)),
            )
            ms_triton_compiled = cuda_time_ms(
                lambda c=compiled, x=x: c(x),
                reset_hook=(lambda m=m_triton_compiled: functional.reset_net(m)),
            )

        eager_speedup = speedup(ms_torch, ms_triton)
        compile_speedup = speedup(ms_torch, ms_triton_compiled)
        print(
            f"  {T:>4} {B:>5} {N:>6}  "
            f"{ms_torch:>9.3f}  {ms_triton:>9.3f}  {ms_triton_compiled:>9.3f}  "
            f"{eager_speedup:>8.2f}x  {compile_speedup:>8.2f}x"
        )


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — skipping benchmark", file=sys.stderr)
        sys.exit(1)

    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()
    print("Columns: torch (ms) | triton eager (ms) | triton compile (ms)")
    print(
        "Ratios : speedup over backend=torch; higher is better "
        f"([{speedup_flag(1.5)} >= 1.5x, {speedup_flag(1.1)} >= 1.1x])."
    )
    print()

    bench_single_layer()
    bench_linear_flexsn_linear()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
