---
id: ann2snn
type: submodule-design
status: draft
title: ANN-to-SNN conversion
parent: activation-based
depends-on: [neuron]
---

## Responsibility

`ann2snn` turns trained ANNs into SNNs. A converter runs a fixed process and holds no
conversion rules; a recipe supplies them.

## Boundary

- Recipes are strategy objects with no `convert` or `run` entry point.
- Conversion-specific modules (`operators`, `qcfs`, `modules`) belong here, and may use
  the core modules and `quantize`.

## Conversion paths

| Path | For | Process |
| --- | --- | --- |
| `FXConverter` + `FXConversionRecipe` | Networks that FX can trace and rewrite | validate, trace, insert observers, calibrate, replace, finalize |
| `ModuleConverter` + `ModuleConversionRecipe` | Conversions that swap submodules without rewriting a graph, such as SpikeZIP and Qwen2 | validate, convert module |

The FX process runs under `no_grad` and restores training flags. A recipe for one path
raises `TypeError` on the other. `Converter` and `ConversionRecipe` are compatibility
names for the FX path and never dispatch to the module-tree path.

## Known limitations

- Rate coding has no max-pooling rule.
- Qwen2 conversion is experimental and establishes no latency or energy gain.
- FX tracing has Torch 2.6 and 2.7 workarounds that patch `torch.reshape`.
