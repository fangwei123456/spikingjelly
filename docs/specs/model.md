---
id: model
type: submodule-design
status: draft
title: Predefined SNN models
parent: activation-based
depends-on: [neuron, memopt, distributed]
---

## Responsibility

`model` holds predefined SNN families and their builders, covering CNNs and spiking
Transformers.

## Boundary

- Models and blocks are plain `nn.Module`s composed of neurons and layers, under the
  step-mode rule in `activation-based`.
- Families accept `spiking_neuron` and neuron keyword arguments, and keep their defaults
  when these are omitted.
- `spikformer` and `sew_resnet` also provide `ModelBuilder`s, which own tensor
  parallelism and `memopt` for `distributed`. That import direction is debt; see
  `architecture`.
- Import implementation blocks from their submodules. The package does not re-export
  them.
