"""
Benchmark: SpikingJelly FlexSN LIF neuron (multi-step mode).

Equivalent neuron config:
  decay_input=True, v_reset=0.0 (hard reset), tau=2.0, v_threshold=1.0

Backends / modes tested:
  - "torch"                : pure PyTorch multi-step loop (CPU and CUDA)
  - "inductor"             : Triton scan kernel path on CUDA
  - "inductor" + compile   : same CUDA path wrapped by torch.compile

In FlexSN, ``backend="triton"`` and ``backend="inductor"`` are equivalent
CUDA backend labels. This benchmark measures the triton / inductor backend once.

Usage:
  python benchmark/flexsn/benchmark_flexsn_lif.py
"""

import time

import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.neuron.flexsn import FlexSN


def make_lif_core(tau: float = 2.0, v_threshold: float = 1.0):
    """Return a stateless single-step LIF function for FlexSN."""
    spike_fn = surrogate.Sigmoid()

    def lif_core(x: torch.Tensor, v: torch.Tensor):
        h = v + (x - v) / tau
        s = spike_fn(h - v_threshold)
        v_new = h * (1.0 - s)
        return s, v_new

    return lif_core


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
    for _ in range(warmup):
        model.reset()
        model(x)
    sync(device)

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

    if device.type == "cuda":
        model_inductor = FlexSN(
            core=core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="inductor",
        ).to(device)
        model_inductor_compiled = FlexSN(
            core=core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="inductor",
        ).to(device)
        compiled = torch.compile(model_inductor_compiled, fullgraph=True)

        with torch.no_grad():
            model_inductor.eval()
            model_inductor_compiled.eval()
            run_benchmark(
                model_inductor,
                x,
                warmup,
                iterations,
                "FlexSN backend=inductor",
                device,
            )
            run_benchmark(
                compiled,
                x,
                warmup,
                iterations,
                "FlexSN inductor + compile",
                device,
            )
    else:
        print("  FlexSN backend=inductor            [skipped — CUDA not available]")
        print("  FlexSN inductor + compile          [skipped — CUDA not available]")

    print("-" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
