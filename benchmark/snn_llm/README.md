# SNN-LLM Reproduction Runners

These private runners validate SNN language-model integration without adding a
second experiment framework to SpikingJelly. Source revisions are pinned in
`sources.json`; runners do not download data, models, or checkpoints.

## Minimal Smoke

`smoke.py` runs a deterministic tiny dense language-model step and writes one
atomic, non-overwriting manifest.

```bash
PYTHONPATH="$PWD" python benchmark/snn_llm/smoke.py \
  --device cpu \
  --output-dir benchmark/output/snn-llm/smoke-cpu
```

CUDA runs require an explicit source revision when Git metadata is unavailable.

## SpikeGPT

The retained SpikeGPT workflows are:

- `spikegpt_checkpoint_compare.py`: fixed 216M checkpoint inference parity.
- `spikegpt_train_pilot.py`: resumable 47M Enwik8 trainability pilot.
- `spikegpt_distributed_precision_smoke.py`: two-rank precision and ownership
  checks.

They load the author model/WKV implementation from the pinned SpikeGPT source
tree and replace its vendored neuron modules with the current SpikingJelly
implementation. On g-series multi-GPU hosts, commands must set
`NCCL_P2P_DISABLE=1`.

## GPT-2 And Qwen2

The `gpt2_conversion/` directory contains the retained dense, MLP ANN2SNN, and
cache-contract examples. The `qwen_conversion/` directory contains the final
public-recipe correctness, quality, efficiency, and tensor-parallel runners.
See their READMEs and the ANN2SNN Transformer tutorial for commands and
interpretation.

## Local Artifacts

All generated reports, logs, checkpoints, profiler traces, calibration files,
datasets, and downloaded model files belong under the ignored
`benchmark/output/` directory. Runner output in a disposable worktree or on a
remote host is transient. After acceptance, copy only the reports that must be
retained to the source worktree archive at
`/Users/allenyolk/CodeRepo/spikingjelly-dev/benchmark/output/snn-llm/archive/`.
They must not be committed to Git. Published documentation contains only
curated, reproducible summaries.

Performance fields are smoke or shared-GPU measurements unless a tutorial
explicitly documents a controlled benchmark protocol. The runners do not claim
SpikeGPT long-run reproduction, online whole-model T-step inference, Qwen 7B
support, or end-to-end speed superiority over dense models.
