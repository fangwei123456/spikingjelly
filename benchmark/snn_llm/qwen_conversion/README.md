# Qwen2.5 ANN2SNN Evaluation

This directory contains the final private runners used to validate the public
SpikingJelly Qwen2 ANN2SNN recipe. The public model, calibration, conversion,
temporal execution, and reset APIs live in
`spikingjelly.activation_based.ann2snn`.

The runners are local-only and never download models. Model and dataset
revisions are pinned in `artifacts.json`. Download Hugging Face artifacts on a
machine with network access, using the configured mirror when needed, and pass
their local paths explicitly.

Install the pinned model and evaluation dependencies into the active project
environment with `uv pip install ".[qwen]" --group llm-benchmark`.

## Correctness

`scaleout_smoke.py` compares dense, exact temporal, and signed-IF execution. It
also checks cached decoding, deterministic generation tokens, reset replay,
the converted module structure, and the explicit `[T, B, S, H]` temporal
layout.

```bash
PYTHONPATH="$PWD" python benchmark/snn_llm/qwen_conversion/scaleout_smoke.py \
  --model-key 0.5b \
  --model-root benchmark/output/qwen2.5-0.5b \
  --output-dir benchmark/output/snn-llm/qwen/0.5b/correctness \
  --device cuda \
  --worktree-revision <REVISION> \
  --time-steps 160 \
  --calibration-levels 16 \
  --calibration-quantile 0.999 \
  --neuron-backend triton
```

## Efficiency

`scaleout_efficiency.py` performs paired dense, Torch-neuron, and
Triton-neuron measurements from a previously validated calibration artifact.
Shared-GPU measurements are labelled and are not treated as absolute
throughput claims.

```bash
PYTHONPATH="$PWD" python benchmark/snn_llm/qwen_conversion/scaleout_efficiency.py \
  --model-key 0.5b \
  --model-root benchmark/output/qwen2.5-0.5b \
  --calibration-artifact benchmark/output/snn-llm/qwen/0.5b/calibration.pt \
  --output-dir benchmark/output/snn-llm/qwen/0.5b/efficiency \
  --device cuda \
  --worktree-revision <REVISION> \
  --time-steps 160 \
  --calibration-levels 16 \
  --calibration-quantile 0.999
```

## Quality

`quality_eval.py` evaluates rolling WikiText perplexity and the fixed zero-shot
task set. It supports bounded development runs and complete, disjoint shards.
`quality_aggregate.py` validates and combines the final shards. Detailed
protocols and accepted measurements belong in the ANN2SNN Qwen tutorial, not
this runner README.

## Tensor Parallelism

`scaleout_tp_smoke.py` validates the explicit TDLinear tensor-parallel plan.
It requires exactly two CUDA ranks and `NCCL_P2P_DISABLE=1` on g-series hosts.
It is a correctness and parameter-capacity check, not a speedup claim.

## Reports

Each successful command atomically writes one non-overwriting `report.json`.
Generated reports, calibration artifacts, logs, checkpoints, profiler traces,
and downloaded model files must remain under the ignored `benchmark/output/`
tree. Command output is transient; accepted reports that need durable local
retention must be copied to the source worktree archive at
`$SOURCE_WORKTREE/benchmark/output/snn-llm/archive/`, where
`$SOURCE_WORKTREE` is the durable source checkout rather than a disposable
worktree.
They must not be committed to Git.

The reports bind results to the runner, model files, artifact lock,
configuration, precision policy, environment, and calibration. Report schemas
are private benchmark contracts rather than public SpikingJelly APIs.

## Boundaries

- The converted model is an offline multistep SNN with explicit time; it is not
  online whole-model T-step inference.
- The conversion model is frozen and inference-only. Direct SNN training is a
  separate model and recipe.
- No runner downloads data or checkpoints, silently changes precision/backend,
  or falls back to CPU or one GPU.
- The final supported conversion evidence covers Qwen2.5 Base 0.5B, 1.5B, and
  3B. It does not claim 7B support or dense-equivalent execution speed.
