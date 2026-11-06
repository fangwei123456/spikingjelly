---
id: precision
type: submodule-design
status: draft
title: Precision
parent: activation-based
depends-on: [neuron, ops]
---

## Responsibility

`precision` prepares a model for a precision mode: `fp32`, `fp16`, `bf16` or `fp8`. Its
public surface is `PrecisionConfig`, `PrecisionArtifacts` and
`prepare_model_for_precision`. Policy and capability code stays internal.

## Boundary

- Sets neuron precision by marking IF, LIF and PLIF for explicit execution. Provider
  rules for that are in `ops`.
- Language-model precision stays native to Megatron.
- Imports Transformer Engine lazily, and only for `fp8`.

## Experimental scope

Transformer Engine FP8 and Triton-neuron mixed precision are experimental. FP8 did not
consistently outperform BF16 in measured workloads. FP8 release acceptance is deferred,
so release profiles exclude `fp8_hardware` tests.
