---
id: timing-based
type: module-design
status: draft
title: Timing-based SNNs
parent: architecture
depends-on: [ops]
---

## Responsibility

`timing_based` models spikes as exact continuous times instead of per-step tensors.
It holds autograd functions (`tempotron`, `srm`, `cubic_srm`, `cubic_conv2d`), module
wrappers, the `first_spike_time` solver and `GaussianTuning` encoding.

## Boundary

- Kernels and Triton acceleration are in `ops`, and this package never selects them.
- Modules are plain `nn.Module`s with no `MemoryModule` state.
- `architecture` covers the relation to `activation_based`.

## Contracts

- Inputs are compact events: spike times, input indices and batch offsets. `duration`
  is positive and floating point.
- Tempotron returns `(v_max, t_peak, t_spike)`. Non-firing times are `inf`, grazing does
  not fire, and output events precede simultaneous inputs.
- Gradients are first-order; `first_spike_time` raises under `create_graph=True`.
- `first_spike_time` callers supply complete monotone partitions of the threshold
  residual.

## Experimental scope

`first_spike_time`, `SRM`, `CubicSRM` and `CubicConv2d` are experimental.

## Known limitations

- `CubicConv2d` excludes bias, groups and dilation.
- Tempotron Triton kernels run without CUDA Graph capture.
