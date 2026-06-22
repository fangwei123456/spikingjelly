# Phase 1: Transformer Spike-Equivalent Operator Design

> Created: 2026-06-22
> Scope: Phase 1B implementation contract for the first operator primitive.

## Goal

Phase 1 adds reusable Transformer operator primitives before attempting a full
Transformer, ViT, BERT, or LLM conversion path. The first vertical slice is
`SpikeSoftmax`, implemented as a tensor-level spike-equivalent differential
operator.

This phase does not change `Converter`, `ActivationRule`, `NeuronFactory`, or
`ThresholdOptimizer`.

## Operator Contract

`SpikeSoftmax` follows the SpikeZIP-TF style cumulative-differential operator
definition:

1. Input is a complete time sequence with shape `[T, ...]`.
2. The time dimension is always dimension `0`.
3. The operator computes cumulative inputs:

   ```python
   x_cum = x_seq.cumsum(dim=0)
   ```

4. It applies ANN softmax to each cumulative input:

   ```python
   y_cum = torch.softmax(x_cum, dim=dim)
   ```

5. It returns the time difference of cumulative outputs:

   ```python
   y_seq[0] = y_cum[0]
   y_seq[t] = y_cum[t] - y_cum[t - 1]
   ```

The cumulative equivalence invariant is:

```python
y_seq.cumsum(dim=0) == torch.softmax(x_seq.cumsum(dim=0), dim=dim)
```

up to normal floating-point tolerance.

## API Decisions

- `SpikeSoftmax` lives in `spikingjelly.activation_based.ann2snn.operators`.
- It is exported by `operators.__all__`, but not by
  `spikingjelly.activation_based.ann2snn.__all__`.
- Constructor signature:

  ```python
  SpikeSoftmax(dim: int = -1)
  ```

- `dim` is the softmax normalization dimension, following `torch.softmax`.
- `dim=0`, or any negative dimension resolved to `0`, is invalid because
  dimension `0` is reserved for time.
- Inputs must be tensors with at least two dimensions. A one-dimensional `[T]`
  tensor has only the time dimension and no valid feature dimension for softmax.

## Non-Goals

- No stateful single-step mode.
- No `reset()` semantics.
- No `step_mode='s'/'m'` API.
- No FX rule or `Converter` integration.
- No claim that outputs are binary spikes.
- No claim that this primitive is fully spike-driven.

`SpikeSoftmax` outputs floating-point differential values. These values may be
negative because softmax over cumulative inputs can decrease for a class when
other classes receive larger later increments.

## Follow-Up Path

After `SpikeSoftmax` is tested, Phase 1 should add `SpikeLayerNorm` and
`SpikeGELU` with the same cumulative-differential contract where applicable.
Only after these primitives are stable should an FX toy Transformer block or
attention proof of concept be added.
