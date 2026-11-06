---
id: goal
type: goal-and-requirements
status: draft
title: SpikingJelly goal and requirements
---

## Goal

SpikingJelly is a PyTorch-native framework for building, training, converting and
deploying spiking neural networks (SNNs), from single-GPU research up to
large-scale multi-GPU training and inference.

## Problem

Spikes are non-differentiable, every neuron carries state across time steps, and
training unrolled over time multiplies memory and compute. Event-camera data needs
its own preparation, and trained models must reach neuromorphic hardware and other
frameworks. SpikingJelly handles these inside the PyTorch programming model, so an
SNN is defined and trained like any other `nn.Module`.

## Audience

SNN researchers and engineers, including beginners who want a small modeling API.
Researchers reproduce published neurons and models. Engineers run large-scale
training, inference and deployment.

## Capabilities

- **Modeling:** spiking neurons, surrogate gradients, stateful and stateless layers,
  custom neuron kernels (FlexSN), and predefined model families, including spiking
  Transformers.
- **Training:** surrogate gradients, online learning and synaptic plasticity.
- **Conversion:** ANN-to-SNN for CNNs and Transformers. Language-model conversion is
  experimental.
- **Execution:** Torch, Triton or optional native CUDA operators, chosen automatically
  from the tensor's device and execution path.
- **Scaling:** memory-efficient training, mixed precision, and experimental distributed
  training and inference for vision and language models.
- **Data:** neuromorphic datasets as raw events or cached frames.
- **Analysis:** operation counting, energy estimation, profiling and monitoring.
- **Interchange:** NIR exchange, and export to Lava and Lynxi toolchains.
- **Timing-based models:** continuous spike-time neurons and solvers. `SRM`-family
  models and `first_spike_time` are experimental.

## Principles

- **PyTorch is the programming model.** Models are `nn.Module`s, training loops are
  ordinary PyTorch, and new features extend that model.
- **Execution is automatic.** Users do not pick a backend; `ops` holds the selection
  rule and its override.
- **The base install stays light.** Heavy or platform-specific integrations are extras,
  and default wheels are pure Python.
- **Correctness first.** Correctness and public contracts outrank measured
  critical-path performance, which outranks low complexity.
- **Public APIs are bilingual.** Each public API has semantically equivalent Chinese
  and English documentation.
- **Versions state compatibility.** From V2, releases follow SemVer-style
  `MAJOR.MINOR.PATCH` with PEP 440 pre-release spelling.

## Non-goals

- **Biophysically detailed simulation.** The README points to btorch for
  computational-neuroscience modeling.
- **Hardware compilers and deployment runtimes.** Exchange stops at handing a model to
  the NIR, Lava or Lynxi toolchain.
- **Pre-V2 compatibility.** V2 does not keep V1 APIs working. Projects that need them
  pin `spikingjelly<2` or follow the migration guide.
