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

For pretrained Qwen2.5-0.5B at T=32, the collapsed-QCFS SGLang adapter matched
generated tokens across TP1, TP2, and PP2. Its KV pool held 32 times as many
token slots; three paired whole-batch workloads improved by 1.9%, 11.1%, and
14.0%, without a measured whole-GPU peak-memory reduction. The recipe's
zero-tail attention path matched 16×256 generated tokens and reduced full-run
time by 1.92–2.41× in three paired runs. Neither result applies to other spike
encodings.

## Historical phase-SpikingLLM T=8

`phase_bitpacked_cache.py` is a private, inference-only benchmark adapter for
the pinned historical phase source in `sources.json`. It preserves the FSNeuron
sign and all eight event times in one `int16` per KV scalar. It requires an
eval-mode, calibrated T=8 phase model with fixed neuron parameters, CUDA/Triton,
and Transformers 4.40.1. Use `_phase_bitpacked_cache(model)` only under
`torch.inference_mode()` in a single-threaded benchmark. The context restores
the temporary cache integration and neuron flags. This is not a SpikingJelly
framework cache API, a current T=1 phase optimization, or evidence for an
official Llama-2 checkpoint. Pretrained SmolLM2-135M and DeepSeek-Coder-1.3B
passed lossless full-model comparisons with this same phase encoding; neither
establishes support for other temporal encodings.
Both models stored one eighth as many KV bytes. On DeepSeek-Coder-1.3B, three
paired runs improved decode time by about 14–15% at batch size 1 and 40% at
batch size 16; the latter workload reduced whole-model peak allocated CUDA
memory by 25.3%, not eightfold. These are shared-host, encoding-specific
measurements, not a general serving-throughput claim.

## Local Artifacts

All generated reports, logs, checkpoints, profiler traces, calibration files,
datasets, and downloaded model files belong under the ignored
`benchmark/output/` directory. Runner output in a disposable worktree or on a
remote host is transient. After acceptance, copy only the reports that must be
retained to the source worktree archive at
`$SOURCE_WORKTREE/benchmark/output/snn-llm/archive/`, where
`$SOURCE_WORKTREE` is the durable source checkout rather than a disposable
worktree.
They must not be committed to Git. Published documentation contains only
curated, reproducible summaries.

Performance fields are smoke or shared-GPU measurements unless a tutorial
explicitly documents a controlled benchmark protocol. The runners do not claim
SpikeGPT long-run reproduction, online whole-model T-step inference, Qwen 7B
support, or end-to-end speed superiority over dense models.
