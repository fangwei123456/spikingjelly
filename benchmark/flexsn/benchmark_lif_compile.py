"""
Benchmark: pure-PyTorch LIF neuron, with and without torch.compile.
"""

import time

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# LIF neuron (pure PyTorch, no SpikingJelly dependency)
# ---------------------------------------------------------------------------


class LIFNode(nn.Module):
    """Leaky Integrate-and-Fire neuron.

    Matches SpikingJelly's LIFNode default configuration:
      decay_input=True, v_reset=0.0 (hard reset)

    Charge  : v[t] = v[t-1] + (x[t] - v[t-1]) / tau
                   = v[t-1] * (1 - 1/tau) + x[t] / tau
    Fire    : spike[t] = (v[t] >= threshold)
    Reset   : v[t] = v[t] * (1 - spike[t])      # hard reset to 0

    tau (float): membrane time constant (default 2.0).
        Larger tau → slower decay (more memory).
        Equivalent to SpikingJelly's `tau` parameter.
    threshold (float): spike threshold (default 1.0).
        Equivalent to SpikingJelly's `v_threshold` parameter.
    """

    def __init__(self, tau: float = 2.0, threshold: float = 1.0) -> None:
        super().__init__()
        self.tau = tau
        self.threshold = threshold

    def forward(
        self, x: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = v + (x - v) / self.tau  # charge: decay_input=True, v_reset=0
        spike = (v >= self.threshold).to(x.dtype)
        v = v * (1.0 - spike)  # hard reset to 0
        return spike, v


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_benchmark(
    model: nn.Module,
    x: torch.Tensor,
    T: int,
    warmup: int,
    iterations: int,
    label: str,
    device: torch.device,
) -> None:
    v = torch.zeros_like(x[0])

    # warmup
    for t in range(warmup):
        _, v = model(x[t % T], v)
    sync(device)

    # timed loop
    v = torch.zeros_like(x[0])
    start = time.perf_counter()
    sync(device)
    for _ in range(iterations):
        v = torch.zeros_like(x[0])
        for t in range(T):
            _, v = model(x[t], v)
    sync(device)
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / iterations * 1e3
    print(f"  {label:<30s}  avg = {avg_ms:.3f} ms  (over {iterations} iters, T={T})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print()

    # Input shape: (T, batch, neurons)
    T = 32
    batch = 64
    neurons = 224 * 224
    warmup = 20
    iterations = 500

    torch.manual_seed(0)
    x = torch.rand(T, batch, neurons, device=device)

    # --- eager ---
    model_eager = LIFNode().to(device)
    model_eager.eval()

    # --- compiled ---
    model_compiled = torch.compile(LIFNode().to(device))

    print(f"Benchmark  (batch={batch}, neurons={neurons}, T={T})")
    print("-" * 60)

    with torch.no_grad():
        run_benchmark(model_eager, x, T, warmup, iterations, "eager", device)
        run_benchmark(model_compiled, x, T, warmup, iterations, "torch.compile", device)

    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
