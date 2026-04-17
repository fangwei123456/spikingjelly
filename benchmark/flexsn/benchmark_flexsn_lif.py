"""
Benchmark: SpikingJelly FlexSN LIF neuron (multi-step mode).

Equivalent neuron config:
  decay_input=True, v_reset=0.0 (hard reset), tau=2.0, v_threshold=1.0

Backends tested:
  - "torch"  : pure PyTorch multi-step loop (CPU and CUDA)
  - "triton" : Triton-compiled kernel       (CUDA only; requires PYTORCH_JIT=0)

Usage:
  PYTORCH_JIT=0 python benchmark_flexsn_lif.py
"""

import os
import time

import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.neuron.flexsn import FlexSN

# ---------------------------------------------------------------------------
# LIF core — matches SJ default: decay_input=True, v_reset=0.0 (hard reset)
# ---------------------------------------------------------------------------


def make_lif_core(tau: float = 2.0, v_threshold: float = 1.0):
    """Return a stateless single-step LIF function for FlexSN.

    Charge : h  = v + (x - v) / tau     [decay_input=True, v_reset=0]
    Fire   : s  = surrogate(h - v_th)
    Reset  : v' = h * (1 - s)           [hard reset to 0]

    Signature required by FlexSN: (x, v) -> (spike, v_new)
    """
    spike_fn = surrogate.Sigmoid()

    def lif_core(x: torch.Tensor, v: torch.Tensor):
        h = v + (x - v) / tau
        s = spike_fn(h - v_threshold)
        v_new = h * (1.0 - s)
        return s, v_new

    return lif_core


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_benchmark(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iterations: int,
    label: str,
    device: torch.device,
) -> None:
    # warmup
    for _ in range(warmup):
        model.reset()
        model(x)
    sync(device)

    # timed loop
    start = time.perf_counter()
    sync(device)
    for _ in range(iterations):
        model.reset()
        model(x)
    sync(device)
    elapsed = time.perf_counter() - start

    T = x.shape[0]
    avg_ms = elapsed / iterations * 1e3
    print(f"  {label:<35s}  avg = {avg_ms:.3f} ms  (over {iterations} iters, T={T})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print()

    T = 32
    batch = 64
    neurons = 224 * 224
    warmup = 20
    iterations = 500

    torch.manual_seed(0)
    x = torch.rand(T, batch, neurons, device=device)

    core = make_lif_core(tau=2.0, v_threshold=1.0)

    print(f"Benchmark  (batch={batch}, neurons={neurons}, T={T})")
    print("-" * 65)

    # --- torch backend (pure PyTorch loop, always available) ---
    model_torch = FlexSN(
        core=core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
    ).to(device)
    with torch.no_grad():
        model_torch.eval()
        run_benchmark(
            model_torch, x, warmup, iterations, "FlexSN backend=torch", device
        )

    # --- triton backend (CUDA + PYTORCH_JIT=0 required) ---
    jit_disabled = os.environ.get("PYTORCH_JIT", "1") == "0"
    if device.type == "cuda" and jit_disabled:
        model_triton = FlexSN(
            core=core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="triton",
        ).to(device)
        with torch.no_grad():
            model_triton.eval()
            run_benchmark(
                model_triton, x, warmup, iterations, "FlexSN backend=triton", device
            )
    elif device.type != "cuda":
        print("  FlexSN backend=triton              [skipped — CUDA not available]")
    else:
        print(
            "  FlexSN backend=triton              [skipped — set PYTORCH_JIT=0 to enable]"
        )

    print("-" * 65)
    print("Done.")
    print()
    print(
        "Tip: run with  PYTORCH_JIT=0 python benchmark_flexsn_lif.py  to include Triton backend."
    )


if __name__ == "__main__":
    main()
