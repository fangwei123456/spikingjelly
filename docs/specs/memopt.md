---
id: memopt
type: submodule-design
status: draft
title: Memory-efficient training
parent: activation-based
---

## Responsibility

`memopt` reduces training memory. `checkpoint` and `checkpoint_module` recompute,
`SpikeCompressor` shrinks stored spikes, and `optimize_memory` applies these to a model.

## Boundary

- Depends only on `base`.
- Does not own neuron execution, and does not wrap models for DDP or FSDP.

## Contracts

- `checkpoint_module` keeps `state_dict` keys unchanged, so checkpoints and weights
  need no conversion. It commits neuron state once despite recomputation.
- `optimize_memory` changes the model in place and must run before DDP or FSDP
  wrapping.
- Levels are cumulative: 1 selective checkpointing, 2 spatial split, 3 temporal split,
  4 greedy unwrapping.
- Spatial splitting needs `split_fn`, and temporal splitting needs `can_chunk`. Without
  them, levels 2 and 3 stop after level 1 with no error.
- A search measures CUDA peak memory, and raises without CUDA.
- With `process_group`, all ranks must call in the same order, and the group covers
  DP×TP of the current pipeline stage.
- Spatial splitting runs before temporal, because temporal splitting conflicts with
  temporal parallelism and breaks fusion across `T`.

Only Vision and Megatron training integrate `memopt`. Inference and export build with
`memopt_level=0`.

## Known limitations

- `checkpointing.py` and `pipeline.py` use PyTorch's private `checkpoint_wrapper`, so
  Torch upgrades can break them.
- The search measures one representative sample.
