---
id: distributed
type: submodule-design
status: draft
title: Distributed training and inference
parent: activation-based
depends-on: [neuron, precision]
---

## Responsibility

`distributed` runs SNNs across GPUs. It is experimental.

- **`vision`** owns the full lifecycle for vision models: DDP, FSDP2, tensor and
  pipeline parallelism, training, evaluation, prediction and inference artifacts.
- **`tensor_parallel`** holds channel-sharded layers that depend on neither workflow.
- **`llm`** owns Megatron Core planning, training, checkpoints, evaluation and
  generation, plus SGLang export and serving.

## Boundary

- `vision` and `tensor_parallel` must not require Megatron. Megatron, SGLang and
  safetensors load inside functions, and tests stub them.
- Model builders build models. `ModelBuilder` is the seam: a builder owns structure,
  tensor parallelism and memory optimization, and workflows consume it.
- Language architecture adapters (Qwen2, SpikeLM) are in `benchmark/snn_llm`.
- Language configs have no `step_mode`, because the Megatron `T*B` envelope is not the
  multi-step interface.

## Contracts

- **Inference artifact.** Schema v2 holds a canonical CPU `state_dict`, model config
  and source topology. The loader accepts only v2. Re-export v1 artifacts, whose neuron
  config still contains a backend.
- **Topology independence.** Artifacts load under a different topology through
  `ModelBuilder.merge_state_dicts` and `build_for_inference`. Pipeline stages must
  reuse the canonical model's parameter objects.
- **Vision training checkpoint.** `step_{N:08d}/` holds `config.json` and a DCP save.
  Resume rejects a recipe mismatch.
- **Language training checkpoint.** `step_{N:07d}` is a Megatron checkpoint whose state
  holds the recipe, including `temporal_semantics_version`.
- **Config serialization.** Configs encode `_target_` paths, and only
  `activation_based.model.*` targets are auto-imported. External neuron classes must
  be registered in every process.
- **SGLang artifact.** The manifest records schema version, recipe hash, dtype and
  SGLang version.

## Execution modes

`execution_mode` is `eager` (default), `compile` or `cuda_graph`.

- `compile` needs replicated execution without pipeline parallelism.
- `cuda_graph` needs one rank and rejects FSDP2, TP, PP, `memopt`, FP8 and explicit
  neuron precision.
- FP8 and explicit neuron precision need DDP with TP=PP=1 for training, and replicated
  eager execution for inference.

## Known limitations

- Single-step mode supports neither pipeline parallelism nor `memopt`.
- `llm.train` handles only complete, independent fixed-`T` windows.
- SGLang needs a separate Python 3.12 Linux environment.
- Artifact export needs CUDA and NCCL.
