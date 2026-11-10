---
id: neuron
type: submodule-design
status: draft
title: Spiking neurons
parent: activation-based
depends-on: [ops]
---

## Responsibility

`neuron` holds the public neuron modules. A module owns hyperparameters, state
materialization and trajectories. Dynamics are explicit-state transitions in
`functional/neuron.py`, and kernels and their selection are in `ops`.

`neuron/__init__.py` separates core neurons, which are general building blocks, from
research neurons, which reproduce specific papers.

## Boundary

- Depends on `base`, `functional` and `surrogate`.
- Reaches `ops` only for FlexSN, few-spike execution and surrogate eligibility checks.
  Ordinary dynamics go through `functional`.

## Transition contract

Every explicit-state transition has the form
`(Y_t, V_t) = Phi(X_t, V_{t-1}; theta)`.

- A transition never reads or writes `MemoryModule` memory. The owning module reads
  state, calls the transition, and writes state back.
- `V_t` may hold several tensors, and charge, fire and reset need not be separate
  steps.
- Multi-step transitions take `[T, ...]` and return the output sequence and final
  state, plus optional `v_seq`. `v_seq` is derived monitoring data, and the next call
  does not take it as state.

## Neuron families

- **`BaseNode`** uses the transition contract and can reach accelerated providers.
- **`SimpleBaseNode`** exposes mutable `neuronal_charge`, `neuronal_fire` and
  `neuronal_reset` hooks. It is pure PyTorch and exists for readable customization.
- **`FlexSN`** is for custom dynamics. A user writes a core, and FlexSN picks Torch,
  Triton or a higher-order-operator path itself. It does not use `ops` selection or its
  environment overrides.

A new built-in neuron needs reference equations in `ops`, a transition in `functional`,
and a module here. Custom neurons use explicit-state transitions or FlexSN.

## Known limitations

- `flexsn_hop.py` patches PyTorch Dynamo internals, so Torch upgrades can break it.
