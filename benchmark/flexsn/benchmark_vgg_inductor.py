"""
Benchmark: SpikingVGG16 with different neuron backends.

Compares:
  - LIFNode  backend="torch"    (pure PyTorch loop, baseline)
  - LIFNode  backend="triton"   (existing Triton scan kernel)
  - FlexSN   backend="inductor" (M3.b single-kernel scan, no PYTORCH_JIT=0)

Usage (run from repo root):
  PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) \\
    python benchmark/flexsn/benchmark_vgg_inductor.py
"""

import gc
import os
import sys

import torch
import torch.nn as nn
from copy import deepcopy

from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg16_bn
from spikingjelly.activation_based.neuron.flexsn import FlexSN

# ---------------------------------------------------------------------------
# LIF core for FlexSN — matches LIFNode defaults (decay_input=True, hard reset)
# ---------------------------------------------------------------------------

def _lif_core(x: torch.Tensor, v: torch.Tensor):
    tau, v_th = 2.0, 1.0
    h = v + (x - v) / tau
    s = (h >= v_th).to(h.dtype)
    return s, h * (1.0 - s)


def make_flexsn_factory():
    """Return a neuron factory compatible with SpikingVGG's spiking_neuron kwarg."""
    def factory(**kwargs):
        step_mode = kwargs.get("step_mode", "m")
        return FlexSN(
            core=_lif_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode=step_mode,
            backend="inductor",
        )
    return factory


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def cuda_time_ms(fn, warmup: int = 5, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# Build model helpers
# ---------------------------------------------------------------------------

def build_vgg(neuron_factory, **neuron_kwargs) -> nn.Module:
    model = spiking_vgg16_bn(
        spiking_neuron=neuron_factory,
        step_mode="m",
        **neuron_kwargs,
    ).cuda()
    functional.set_step_mode(model, "m")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — skipping.", file=sys.stderr)
        sys.exit(1)

    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    T   = 4        # timesteps (typical SNN setting)
    B   = 64       # batch size (CIFAR-10 scale)
    C, H, W = 3, 32, 32   # CIFAR-10 resolution (standard spiking-VGG benchmark)

    torch.manual_seed(0)
    x = torch.randn(T, B, C, H, W, device="cuda")

    configs = [
        ("LIFNode  backend=torch  ",
         lambda: build_vgg(neuron.LIFNode, backend="torch")),
        ("LIFNode  backend=triton ",
         lambda: build_vgg(neuron.LIFNode, backend="triton")),
        ("FlexSN   backend=inductor",
         lambda: build_vgg(make_flexsn_factory())),
    ]

    scale_label = "ImageNet-scale" if H >= 224 and W >= 224 else "CIFAR-scale"
    print(f"Input : T={T}, B={B}, {C}x{H}x{W}  ({scale_label})")
    print("Model : SpikingVGG-16-BN")
    print()
    print(f"  {'backend':<30}  {'ms/iter':>9}  {'img/s':>9}  {'vs torch':>10}")
    print("-" * 68)

    baseline_ms = None
    for label, build_fn in configs:
        model = build_fn()
        model.train()

        def step(model=model):
            functional.reset_net(model)
            model.zero_grad(set_to_none=True)
            outputs = model(x)
            outputs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
            loss = sum(out.float().sum() for out in outputs)
            loss.backward()

        # One extra iteration to initialize states and trigger Triton JIT +
        # autotune before timing begins (avoids measuring compile cost).
        step()
        ms = cuda_time_ms(step)

        imgs_per_sec = B * 1000 / ms
        rel = f"{ms / baseline_ms:.2f}×" if baseline_ms is not None else "1.00×"
        if baseline_ms is None:
            baseline_ms = ms
        print(f"  {label:<30}  {ms:>9.2f}  {imgs_per_sec:>9.1f}  {rel:>10}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("-" * 68)
    print("Note: inductor path does NOT require PYTORCH_JIT=0 at runtime.")
    print("      triton path requires PYTORCH_JIT=0 (set in this script).")


if __name__ == "__main__":
    main()
