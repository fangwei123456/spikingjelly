"""
Benchmark: SpikingVGG16-BN training (forward + backward) speed.

Compares:
  - LIFNode  backend="torch"    — pure PyTorch BPTT baseline
  - LIFNode  backend="triton"   — hand-optimised Triton fwd+bwd
  - FlexSN   backend="inductor" — Triton fwd+bwd via build_training_kernels

All three use Sigmoid surrogate gradient (alpha=4.0) for a fair comparison.

Usage (run from repo root):
  PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) \\
    python benchmark/flexsn/benchmark_vgg_train.py
"""
import gc
import os
import sys

import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, functional, surrogate
from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg16_bn
from spikingjelly.activation_based.neuron.flexsn import FlexSN

# ---------------------------------------------------------------------------
# LIF core with Sigmoid surrogate — matches LIFNode defaults
# ---------------------------------------------------------------------------

_sg = surrogate.Sigmoid(alpha=4.0)

def lif_core_sg(x: torch.Tensor, v: torch.Tensor):
    tau, v_th = 2.0, 1.0
    h = v + (x - v) / tau
    s = _sg(h - v_th)          # Sigmoid surrogate gradient
    return s, h * (1.0 - s)    # spike, updated membrane potential


def make_flexsn(**kwargs):
    return FlexSN(
        core=lif_core_sg,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode=kwargs.get("step_mode", "m"),
        backend="inductor",
    )


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def cuda_time_ms(fn, warmup: int = 5, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available.", file=sys.stderr); sys.exit(1)

    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    T, B = 4, 64
    C, H, W = 3, 32, 32  # CIFAR-10 scale
    torch.manual_seed(0)
    x_base = torch.randn(T, B, C, H, W, device="cuda")
    target = torch.randint(0, 1000, (B,), device="cuda")
    criterion = nn.CrossEntropyLoss()

    configs = [
        ("LIFNode  backend=torch  ",
         lambda: spiking_vgg16_bn(spiking_neuron=neuron.LIFNode,
                                  step_mode="m", backend="torch",
                                  surrogate_function=surrogate.Sigmoid(alpha=4.0)).cuda()),
        ("LIFNode  backend=triton ",
         lambda: spiking_vgg16_bn(spiking_neuron=neuron.LIFNode,
                                  step_mode="m", backend="triton",
                                  surrogate_function=surrogate.Sigmoid(alpha=4.0)).cuda()),
        ("FlexSN   backend=inductor",
         lambda: spiking_vgg16_bn(spiking_neuron=make_flexsn,
                                  step_mode="m").cuda()),
    ]

    print(f"Input : T={T}, B={B}, {C}×{H}×{W}  (CIFAR-10 scale)")
    print(f"Model : SpikingVGG-16-BN  |  Loss: CrossEntropy")
    print(f"Warmup: 5 iters, Timed: 30 iters (forward + backward)")
    print()
    print(f"  {'backend':<30}  {'ms/iter':>9}  {'img/s':>9}  {'vs torch':>10}")
    print("-" * 68)

    baseline_ms = None
    for label, build_fn in configs:
        model = build_fn()
        functional.set_step_mode(model, "m")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def step():
            functional.reset_net(model)
            optimizer.zero_grad(set_to_none=True)
            out = model(x_base)          # [T*B, num_classes] or [B, num_classes]
            # VGG output: average over T in multi-step mode
            if out.dim() == 3:           # [T, B, C]
                out = out.mean(0)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

        # Trigger Triton JIT + autotune before timing
        with torch.no_grad():
            model(x_base)
        step()  # one full train step to compile everything

        ms = cuda_time_ms(step)
        imgs_per_sec = B * 1000 / ms
        rel = f"{ms / baseline_ms:.2f}×" if baseline_ms is not None else "1.00×"
        if baseline_ms is None:
            baseline_ms = ms
        print(f"  {label:<30}  {ms:>9.2f}  {imgs_per_sec:>9.1f}  {rel:>10}")

        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    print("-" * 68)


if __name__ == "__main__":
    main()
