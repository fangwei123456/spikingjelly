---
id: ops
type: module-design
status: draft
title: Private operators (spikingjelly._ops)
parent: architecture
---

## Responsibility

`ops` holds the operators behind neurons, fused neuron-Linear layers, spike
compression, FlexSN and timing-based kernels. For each family it owns the Torch
reference equations, the accelerated providers, and the policy that picks one for a
given tensor and execution path.

It sits at the repository root and installs as `spikingjelly._ops` to mark a private
boundary. Public API and bilingual docstring rules do not apply here, and internals
may change in any release.

## Boundary

- Owns provider selection, `torch.library` registration, cache tags, native loading,
  layouts, and Triton compilation of custom cores.
- Neuron state belongs to the owning module (see `neuron`), and the public
  explicit-state API belongs to `functional`.
- May import `logger`, `configure` and `activation_based.surrogate`. Surrogate access
  exists only to identify built-in surrogates for kernels.
- Must not import `neuron`, `layer` or any capability module.
- Only the importers listed in `architecture` may use this package.

Each family directory has `reference` (Torch equations), `cpu` (registered CPU entry),
`autograd` or `validation`, and the optional `triton` and `native` providers.

## Provider selection

The tensor's device and execution path select the implementation. There is no public
backend argument.

| Path | Order |
| --- | --- |
| CPU | Torch reference |
| CUDA eager | native, Triton, Torch |
| CUDA `torch.compile` (neuron families) | Triton, native, Torch |
| CUDA Graph | no path of its own; capture keeps what warmup bound |

Compile prefers Triton because Inductor can fuse its kernels, while native kernels are
opaque calls.

- Selection is cached per device and path. Automatic mode falls through only on
  `ImportError` or `OSError` during selection. Later kernel errors propagate, and
  providers are not probed.
- `SJ_<NEURON>_CUDA_IMPLEMENTATION` forces one candidate, and failure raises.
- Higher-order gradients run the reference equations.
- Kernels accept the seven built-in surrogates when `spiking=True` and `alpha` is a
  Python scalar. In automatic mode, other surrogates use Torch reference.
- Explicit `neuron_storage`, `neuron_fwd` or `neuron_bwd` precision requires CUDA
  Triton for multi-step IF, LIF and PLIF, and never falls back.
- CUDA providers require NVIDIA CUDA; selection raises on HIP builds.

## Native extensions

Default wheels are pure Python. `SJ_BUILD_NATIVE_CUDA=1` builds extensions, and
missing prerequisites fail the build. The loader accepts an extension only with an
equal `_NATIVE_ABI`, exact Torch and CUDA versions, and a binary target for the
device. Anything else is an `ImportError`, so selection moves on.

Bump `_NATIVE_ABI` whenever a kernel signature or behavior changes. Cache tags hash
`native.py` and `native_layout.cuh` but not `.cu` or `kernels.cuh`. They do hash
`ops/__init__.py`, which holds `_NATIVE_ABI`, so the bump invalidates a kernel-only
change.

## Compile cache tags

`dispatch` adds a source hash to the compiler cache key, because ahead-of-time
compilation otherwise hashes only the shared operator name and not the selected
provider.
