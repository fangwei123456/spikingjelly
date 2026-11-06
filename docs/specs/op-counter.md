---
id: op-counter
type: submodule-design
status: draft
title: Operation counting and energy estimation
parent: activation-based
depends-on: [neuron]
---

## Responsibility

`op_counter` measures one real execution along two axes, and never changes the model.

- **GPU performance analysis and optimization.** `FlopCounter` and
  `MemoryAccessCounter` quantify compute and data movement. Their ratio, arithmetic
  intensity in FLOPs per byte, places a workload on a roofline chart, and the caller
  supplies hardware peak FLOPs and bandwidth. The counts use an idealized model.
- **SNN inference operation counts and energy estimation.** `ACCounter`, `MACCounter`,
  `SynOpCounter`, `NeuronStateCounter` and `NeuromorphicMemoryAccessCounter` count the
  operations of spiking inference, and the energy estimators turn those counts into
  cost estimates. Estimators are separate cost models and are not interchangeable.

## Boundary

- Attaches through Torch dispatch and function modes plus module hooks.
- `strict` decides whether unsupported operations fail. Otherwise `get_unsupported`
  reports them.

## Energy-model fidelity

Each estimator reports its provenance through `EnergyModelInfo`, and new models must
do the same.

| Model | Fidelity | Basis |
| --- | --- | --- |
| Simple | SpikingJelly-defined | 45 nm, Horowitz and STEP sources |
| Lemaire | Paper | 45 nm, arXiv 2210.13107 |
| SpikeSim dense | Reference code | 65 nm, author code |
| SpikeSim event | SpikingJelly-defined | 65 nm, separate sparse model |
| NeuroMC | Source-aligned | 32 nm, author code; BN and optimizer costs lack external validation |

## Known limitations

- Lemaire and SpikeSim estimate forward inference only, and SpikeSim covers `Conv2d`.
- NeuroMC fails on unsupported runtime operations and returns no partial result.
