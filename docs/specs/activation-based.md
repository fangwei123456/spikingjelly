---
id: activation-based
type: module-design
status: draft
title: Activation-based SNN stack
parent: architecture
depends-on: [ops]
---

## Responsibility

`activation_based` is the discrete-time SNN stack: spikes are tensors at each time
step, and networks train through surrogate gradients. It owns the state model,
neurons, layers, and every capability built on them.

## Boundary

- The core is `base`, `surrogate`, `functional`, `neuron` and `layer`. Capabilities
  depend on the core. The core does not depend on capabilities, with one exception:
  `functional/neuron.py` lazily imports `lava_exchange`.
- Only `examples/` may import `datasets`.
- `examples/` holds runnable demonstrations and recipes.

## Dependency edges

| Module | Depends on |
| --- | --- |
| `neuron` | `base`, `functional`, `surrogate`, `ops` |
| `layer` | `base`, `functional`, `neuron` |
| `model` | `functional`, `layer`, `neuron`, `surrogate`, `memopt`; `distributed` is debt, see `architecture` |
| `ann2snn` | `base`, `functional`, `layer`, `neuron`, `quantize`, `surrogate` |
| `learning` | `base`, `functional`, `monitor` |
| `memopt` | `base` |
| `precision` | `functional`, `layer`, `neuron`, `ops` |
| `distributed` | `base`, `functional`, `neuron`, `surrogate`, `precision`, `_cuda_graph` |
| `op_counter` | `neuron` |
| exchange modules | `base`, `functional`, `layer`, `neuron`, `quantize`, `surrogate` |

`memopt` and `precision` do not import `distributed`. `distributed` gets memory
optimization through model builders.

## State model

- **Step modes.** `'s'` processes one step, `[N, *]`; `'m'` processes a time-major
  sequence, `[T, N, *]`.
- **Memory.** `MemoryModule` registers state with reset values. `reset` restores those
  values, and `functional.reset_net` and `functional.detach_net` apply this across a
  network.
- **State shape.** `BaseNode` and its built-in subclasses reject materialized
  states with mismatched per-step shapes instead of discarding them. Reset managed
  networks before changing batch or feature shape; explicit-state calls must supply
  matching states. Scalar initialization and dtype/device conversion still apply.
- **Managed and explicit forms.** `base.to_functional_forward` turns a managed network
  into an explicit-state call, which `memopt` relies on.
  `FlexSN.functional_forward` binds its current registered static tensors when
  omitted; its lower-level step methods require them explicitly.
  Checkpoint buffer snapshots preserve gradients and commit updates once.
  Non-leaf buffer commits rebind shared module aliases without modifying autograd
  sources; leaf buffers retain their objects and are updated in-place.
  FlexSN tracing isolates compiler probes from the caller's saved-tensor hooks.
- **Step-mode leaves.** Built-in `StepModule` implementations are atomic leaves whose
  children are normally not `StepModule`s. Networks, blocks and attention stay plain
  `nn.Module`s. Explicit scheduling containers and modules that own independent
  temporal state may be exceptions, and must document why. The rule does not restrict
  user modules.

## Other modules

- **`learning`** holds local plasticity learners and gradient-based online training
  (FPTT, OTTT, SLTT). Learners consume `monitor` records, and OTTT layers stay in
  `layer`.
- **`monitor`** records values through hooks, and `profiler` measures memory and CUDA
  time.
- **`rnn`, `encoding`, `quantize`** hold spiking RNNs, input encoders and quantization
  functions.

## Exchange

- **`nir_exchange`** imports and exports. Imported graphs carry explicit state, so
  `reset_net` does not affect state a caller already holds.
- **`lava_exchange`** exports to Lava-DL slayer blocks.
- **`lynxi_exchange`** builds through Lyngor and runs through lynpy.

NIR, Lava and Lynxi SDK-dependent functions sit behind `try`/`except`, so the rest of
each module imports without the optional dependencies. NIR import/export raises a
clear `ImportError` with the `spikingjelly[nir]` installation command when needed.

## Known limitations

- Lynxi export and slayer-dependent Lava conversion have no tests in `test/`.
