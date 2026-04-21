"""
Benchmark: FlexSN inductor backend vs triton baseline.

Tests three scenarios from design doc (feature/flexsn-inductor-backend):
  1. Pure single FlexSN layer                    (G2 criterion)
  2. Linear -> FlexSN -> Linear fusion           (G3 criterion)

Usage (run from repo root):
  PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python benchmark/flexsn/flex_sn_inductor.py
"""

import os
import sys

import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron.flexsn import FlexSN

# ---------------------------------------------------------------------------
# LIF core
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def cuda_time_ms(fn, warmup: int = 10, iters: int = 200,
                 reset_hook=None) -> float:
    for _ in range(warmup):
        if reset_hook is not None:
            reset_hook()
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        if reset_hook is not None:
            reset_hook()
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def ratio_flag(r: float) -> str:
    if r <= 1.1:
        return "OK"
    if r <= 1.5:
        return "CLOSE"
    return "CONSIDER M3.b"


# ---------------------------------------------------------------------------
# Benchmark 1: single FlexSN layer
# ---------------------------------------------------------------------------


def bench_single_layer():
    print("=" * 70)
    print("Benchmark 1 — single FlexSN layer (forward only, no grad)")
    print(f"  {'T':>4} {'B':>5} {'N':>6}  {'triton':>10}  {'inductor':>10}  {'ratio':>7}  flag")
    print("-" * 70)

    configs = [(8, 128, 1024), (32, 128, 1024), (8, 128, 4096)]
    for T, B, N in configs:
        x = torch.randn(T, B, N, device="cuda")

        n_tri = make_flexsn("triton")
        n_ind = make_flexsn("inductor")
        c_ind = torch.compile(n_ind, fullgraph=True)

        with torch.no_grad():
            ms_tri = cuda_time_ms(lambda: n_tri(x), reset_hook=n_tri.reset)
            ms_ind = cuda_time_ms(lambda: c_ind(x), reset_hook=n_ind.reset)

        r = ms_ind / ms_tri
        print(
            f"  {T:>4} {B:>5} {N:>6}  {ms_tri:>9.3f}  {ms_ind:>9.3f}  {r:>6.2f}x  [{ratio_flag(r)}]"
        )


# ---------------------------------------------------------------------------
# Benchmark 2: Linear -> FlexSN -> Linear fusion (G3)
# ---------------------------------------------------------------------------


class SeqModel(nn.Module):
    def __init__(self, hidden: int, backend: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.neuron = make_flexsn(backend)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        out = []
        for t in range(T):
            out.append(self.fc1(x[t]))
        x_proj = torch.stack(out, dim=0)
        spike = self.neuron(x_proj)
        out2 = []
        for t in range(T):
            out2.append(self.fc2(spike[t]))
        return torch.stack(out2, dim=0)


class SeqModelFused(nn.Module):
    """Linear -> FlexSN -> Linear with batch-matmul over T for full compile."""

    def __init__(self, hidden: int, backend: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.neuron = make_flexsn(backend)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, N = x.shape
        x_flat = x.view(T * B, N)
        h = self.fc1(x_flat).view(T, B, N)
        spike = self.neuron(h)
        return self.fc2(spike.view(T * B, N)).view(T, B, N)


def bench_linear_flexsn_linear():
    print()
    print("=" * 70)
    print("Benchmark 2 — Linear -> FlexSN -> Linear (forward only, no grad)")
    print(f"  {'T':>4} {'B':>5} {'N':>6}  {'triton':>10}  {'inductor':>10}  {'ratio':>7}  flag")
    print("-" * 70)

    configs = [(8, 128, 1024), (32, 128, 1024)]
    for T, B, N in configs:
        x = torch.randn(T, B, N, device="cuda")

        m_tri = SeqModelFused(N, "triton").cuda()
        m_ind = SeqModelFused(N, "inductor").cuda()
        c_ind = torch.compile(m_ind, fullgraph=True)

        from spikingjelly.activation_based import functional
        with torch.no_grad():
            ms_tri = cuda_time_ms(lambda: m_tri(x),
                                  reset_hook=lambda: functional.reset_net(m_tri))
            ms_ind = cuda_time_ms(lambda: c_ind(x),
                                  reset_hook=lambda: functional.reset_net(m_ind))

        r = ms_ind / ms_tri
        print(
            f"  {T:>4} {B:>5} {N:>6}  {ms_tri:>9.3f}  {ms_ind:>9.3f}  {r:>6.2f}x  [{ratio_flag(r)}]"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — skipping benchmark", file=sys.stderr)
        sys.exit(1)
    if os.environ.get("PYTORCH_JIT", "1") != "0":
        print(
            "Error: PYTORCH_JIT != 0. "
            "The triton backend requires PYTORCH_JIT=0.\n"
            "Run with:  PYTORCH_JIT=0 PYTHONPATH=$(pwd) python benchmark/flexsn/flex_sn_inductor.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()
    print("Columns: triton (ms) | inductor (ms) | ratio")
    print("G2 criterion: ratio <= 1.1 => OK; <= 1.5 => CLOSE; > 1.5 => CONSIDER M3.b")
    print()

    bench_single_layer()
    bench_linear_flexsn_linear()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
