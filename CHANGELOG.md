# 变更日志 | Changelog

All notable changes to SpikingJelly are documented in this file.

SpikingJelly starts maintaining this standard changelog from `2.0.0.dev0`.
For older releases, see the historical fatal-bug record in
[bugs.md](https://github.com/fangwei123456/spikingjelly/blob/master/bugs.md)
and the archived documentation linked from the project README.

## Unreleased

### Features

#### Spiking Neurons

Module: `spikingjelly.activation_based.neuron`.

- Added the paper-faithful `ComplementaryLIFNode` with single-step and
  multi-step PyTorch execution and optional trajectories for both neuron states.

#### Memory Optimization

Module: `spikingjelly.activation_based.memopt`.

- Added `checkpoint` and state-dict-transparent `checkpoint_module` building
  blocks for user-defined checkpoint boundaries, stateless input compression
  through `SpikeCompressor`, and explicit temporal chunking.
- Added `optimize_memory` as the high-level paper preset with levels 0-4,
  speed/balanced/memory checkpoint budgets, user-provided spatial and temporal
  rules, and synchronized decisions within a distributed pipeline stage.
- Integrated the new configuration with Vision and MCore training while keeping
  evaluation, generation, prediction, and artifact export free of memopt wrappers.

#### Operation Counters

Module: `spikingjelly.activation_based.op_counter`.

- Added reproducible external validation for the Lemaire, SpikeSim dense, and
  NeuroMC forward-energy estimators using pinned author code or published
  equations, with scale-free ranking metrics and explicit unvalidated scopes.

### Breaking Changes and Notices

#### FlexSN Migration

Module: `spikingjelly.activation_based.neuron.flexsn`.

- Replaced the separate `FlexSN`/`FlexSNKernel` interfaces with one `FlexSN`
  module that provides explicit-state `functional_forward` and managed-state
  `forward` execution.
- Removed the `num_inputs`, `num_outputs`, `example_inputs`, `example_outputs`,
  and `requires_grad` constructor arguments. Input/output arities and Triton
  examples are now inferred automatically.
- Added registered `static_inputs` for tensor parameters reused at every time
  step, and standardized states, state sequences, and multiple outputs on
  tuples.
- FlexSN no longer accepts empty multi-step sequences or silently falls back
  from the explicitly selected Triton backend.
- Consolidated FlexSN's generated Triton execution into one kernel bundle and
  three private registered operators, with automatic support for scalar and
  elementwise static Tensor gradients.
- Reduced the HOP backend to one eager scan and one compiler-visible custom
  HOP, removing the experimental native scan/while-loop alternatives and
  their environment flags.

#### Precision Migration

Module: `spikingjelly.activation_based.precision`.

- Replaced `fp8-te` with `fp8`; no compatibility alias or silent precision
  fallback is retained.
- Removed the TorchAO FP8 backend and consolidated the Transformer Engine
  dependency as `spikingjelly[fp8]`.
- Reduced the public interface to `PrecisionConfig`, `PrecisionArtifacts`, and
  `prepare_model_for_precision()`. Removed public policy, conversion-helper, report
  writer, and Triton execution-plan interfaces.
- Added `triton_storage`, `triton_fwd`, and `triton_bwd` to `PrecisionConfig` for
  training-time configuration of existing multi-step Triton IF, LIF, and PLIF
  nodes. Sensitive surrogate operations continue to compute locally in FP32.
- Transformer Engine conversion leaves Linear and pointwise Conv1d layers whose
  dimensions violate FP8 alignment in high precision and reports them as
  unsupported instead of failing during forward.
- `distributed.vision` now stores `PrecisionConfig` and supports experimental
  model FP8 and Triton mixed precision with DDP.
- Vision checkpoint resume normalizes the former scalar `fp32`, `fp16`, and
  `bf16` recipe values to `PrecisionConfig`; removed FP8 mode names are rejected.
- MCore language-model precision remains configured by the native MCore
  transformer and optimizer configuration.

#### Memory Optimization Migration

Module: `spikingjelly.activation_based.memopt`.

- Replaced `memory_optimization`, `input_compressed_gc`, `GCContainer`,
  `TCGCContainer`, mutable compressor bases, profile presets, summaries, and
  module-side `__spatial_split__` hooks with the smaller public API above. No
  compatibility aliases are provided.
- MCore training checkpoints whose recipe contains the former `use_snn_memopt`
  field cannot be resumed directly with the new configuration.

## 2.0.0.dev2 - 2026-08-23

### Improvements

#### ANN-to-SNN Conversion

Module: `spikingjelly.activation_based.ann2snn`.

- Rate-coding percentile calibration now uses PyTorch's exact
  `torch.quantile` instead of an internal sampled approximation.
- Clarified the ANN2SNN quick-start path, recipe-to-converter mapping, and
  custom recipe extension points in the API and tutorial documentation.

#### Operation Counters

Module: `spikingjelly.activation_based.op_counter`.

- Clarified the basic counting workflow, estimator/module coverage, and the
  boundary between ATen rules and custom `torch.*` function rules.
- Replaced the compute-only `ComputeEnergy*` interface with `SimpleEnergy*`,
  which combines runtime MAC and AC with logical neuromorphic memory accesses.
- Added `NeuromorphicMemoryAccessCounter` for runtime weight/bias reads and
  persistent neuron-state reads and writes, independently of host-device memory
  traffic and the Lemaire analytical model.
- The Simple Energy default uses `24.96 pJ/byte` for memory traffic and exposes
  the memory coefficient for explicit hardware-regime overrides.
- Operation reports now include stable model provenance and the cost
  configuration required to interpret or reproduce an estimate.
- FLOP counting uses the GPU roofline convention of two FLOPs per MAC, supports
  fused scaled-dot-product attention, and exposes skipped ATen operations.
- Added `ModuleCounter` and `ModuleCounterMode` for runtime module
  forward/backward rules with shared scope, ignored-subtree, strict-mode, and
  hook-lifecycle handling.
- Lemaire, Simple Energy's neuromorphic memory counter, and NeuroMC now share
  `ModuleCounterMode`; energy formulas and reports remain owned by each model.
- Lemaire inference now uses runtime spike activity with the paper's IF/LIF
  compute buckets and fixed 32-bit memory-access regime.
- NeuroMC runtime energy now follows the author-code variable direction
  multipliers, includes register traffic, and uses aggregate SRAM capacities.
- SpikeSim dense energy is locked to the released `c2627bc` PE-cycle formula;
  event mode is a separately identified SpikingJelly sparse model.

#### NIR Exchange

Module: `spikingjelly.activation_based.nir_exchange`.

- Added bidirectional NIR conversion for Conv1d and current-based LIF neurons.
- Imported NIR models now expose neuron and graph state through the returned
  state value.
- The NIR optional dependency now supports `nir>=1.0.7,<2` with
  `nirtorch>=2.6,<2.7`.

#### Spiking Model Families

Module: `spikingjelly.activation_based.model`.

- Added QKFormer builders using the existing Q-K attention layer.
- Added membrane-shortcut MS-ResNet and Max-ResNet builders.
- Added MaxFormer builders using max-pooling, depth-wise convolution, and the
  existing SpikingJelly multi-step layers.
- Added Spike-driven Transformer v1 and its reusable spike-driven self-attention
  layer.
- The new model classes and builders are available directly from
  `spikingjelly.activation_based.model`.

### Bug Fixes

#### Lemaire Memory Accounting

Module: `spikingjelly.activation_based.op_counter`.

- Corrected Lemaire memory accesses to follow the per-layer FNN/SNN formulas,
  count output spikes and membrane-potential traffic, and price accesses using
  each layer's local SRAM capacity instead of a global maximum traffic value.
- Lemaire memory accounting now treats only binary tensors as spike traffic and
  reports unsupported transposed convolutions instead of applying a dense fallback.
- Corrected grouped and depthwise spike fanout to use output channels per group.
- Lemaire profiling now rejects neuron types outside its IF/LIF paper scope
  instead of returning incomplete compute energy.

#### Operation Counter Accuracy

Module: `spikingjelly.activation_based.op_counter`.

- Spike-convolution counters now use float64 reduction accumulators so large
  integer event counts are not rounded through float32.
- Ideal fused-attention traffic now includes positional or keyword masks/biases
  and available bias gradients.
- Module-driven energy profilers now accept keyword tensor inputs and reject
  model rebinding while hooks are active.
- Neuromorphic convolution probes no longer expose their padding or helper
  tensor operations to dispatch counters.
- NeuroMC preserves trainable-parameter shape information required to classify
  backward fragments after the profiling context exits.
- Module-driven profilers now reject rebinding throughout an active context,
  including models that install no module hooks.
- NeuroMC now rejects conflicting options for a reused stage name, matches
  repeated module calls in full backward passes, ignores reentrant-checkpoint
  recomputation, and rejects ambiguous selective backward through a repeated module.

#### NIR Exchange

Module: `spikingjelly.activation_based.nir_exchange`.

- Restored neuron export with NIR 1.0.8 and corrected imported convolution
  channel metadata.
- Export shape inference now preserves the source model's neuron memories.

#### Datasets

Module: `spikingjelly.datasets`.

- Dataset preparation now leaves a `.building` marker after interrupted raw or
  frame preprocessing instead of silently reusing partial output directories.
- NPZ loaders now close archives after reading, and SHD event datasets can be
  used by spawned `DataLoader` workers after access in the parent process.

#### Examples

Module: `spikingjelly.activation_based.examples.dsqn`.

- DSQN replay batches now build correctly with NumPy 2.x.

### Breaking Changes and Notices

#### Operation Counter API Migration

Module: `spikingjelly.activation_based.op_counter`.

- Removed `ComputeEnergyCostConfig`, `ComputeEnergyConfig`,
  `ComputeEnergyProfiler`, `ComputeEnergyReport`, and
  `estimate_compute_energy`. Use the corresponding `SimpleEnergy*` names and
  `estimate_simple_energy` instead.
- Renamed `estimate_spikesim_event_energy` to `estimate_spikesim_energy` because
  the same entry point supports both dense and event activity models.
- Lemaire and SpikeSim estimators now default to strict handling of unsupported
  model paths; partial warning-only reports require an explicit `strict=False`.
- Removed ineffective NeuroMC `core_type`, `strict`, and
  `extra_ignore_modules` parameters. Unsupported runtime operations remain
  fail-closed.
- NeuroMC stage semantics now use explicit `phase`, `reuse_weights`, and
  `batch_norm_backward` arguments instead of parsing stage names.
- Removed unused memory-residency and standalone NeuroMC primitive-counter
  interfaces that were not connected to the runtime energy profiler.
- Removed Lemaire's unused non-binary sparsity and custom state-rule settings;
  `NeuronStateCounter` remains available as an independent diagnostic.
- Replaced `get_unsupported_ops(counter)` with the mode-independent
  `get_unsupported(counter)` on all counter modes.
- `NeuromorphicMemoryAccessCounter` is no longer a context manager and no
  longer has `bind_model()`; use it through `ModuleCounterMode`.
- Removed `LemaireAddressingCounter`; Lemaire addressing is now part of the
  single module-driven Lemaire counter.
- Manual NeuroMC profiling now requires `bind_model()` before entering the
  profiler context; module hook registration is owned by `ModuleCounterMode`.
- Flattened the one-file `analytical_energy` package into
  `op_counter/analytical_energy.py`; import its public symbols from
  `op_counter` or `op_counter.analytical_energy` instead of the removed
  `op_counter.analytical_energy.core` path.

#### NIR Exchange

Module: `spikingjelly.activation_based.nir_exchange`.

- NIR conversion now rejects soft-reset neurons, grouped convolutions, and
  incompatible heterogeneous neuron parameters instead of silently changing
  their semantics.
- Imported NIR models no longer advance implicit module memory. Step-by-step
  callers must pass the returned state to continue; `state=None` restarts, and
  `functional.reset_net` does not reset an already returned state value.
- Recurrent NIR graphs require single-step mode.

#### Model and Example Layout

Modules: `spikingjelly.activation_based.model` and
`spikingjelly.activation_based.examples`.

- Training helpers and ImageNet/FlexSN training entry points moved out of the
  model package into `spikingjelly.activation_based.examples`; imports and
  module commands using the old `model.train_*` paths must be updated.
- Examples now use one shallow entry per example; shared training code lives in
  `spikingjelly.activation_based.examples.common`, and example names use
  lowercase `snake_case`.
- Model package and model-module wildcard exports now focus on complete models
  and builders; implementation blocks remain available from their model modules.
- Composite Spikformer and attention modules are plain `nn.Module` objects;
  step-mode configuration remains on their atomic child layers.
- Removed the unused built-in CIFAR10 loader and no-op epoch hooks from the
  example `Trainer`; active customization points remain model, preprocessing,
  output, optimizer, and scheduler hooks.


## 2.0.0.dev1 - 2026-08-14

### Features

#### Package Configuration

Module: `spikingjelly.configure`.

- Package-level options are now configured exclusively through `SJ_*`
  environment variables read when `spikingjelly.configure` is first imported.
  Assigning `spikingjelly.configure` module attributes is no longer a supported
  configuration method.

#### Logging

Module: `spikingjelly`.

- Migrated the package logger from stdlib `logging` to Loguru. SpikingJelly is
  disabled by default and never adds or removes application-owned sinks; applications
  enable the namespace with `logger.enable("spikingjelly")` and configure sinks
  separately.
- Production diagnostics use direct Loguru calls with parameterized formatting;
  applications own sink configuration and any structured context or stdlib logging
  integration.
- Updated the AST policy checker, Loguru performance benchmark, tests, and bilingual
  API documentation. This is a breaking change: stdlib handlers,
  `basicConfig()`, `dictConfig()`, and pytest `caplog` no longer capture SpikingJelly
  records. Tests should attach and remove a temporary Loguru sink instead.

#### Functional State Transitions

Modules: `spikingjelly.activation_based.functional` and
`spikingjelly.activation_based.base`.

- Added explicit-state functional execution across the framework's stateful
  modules while preserving regular stateful forward behavior.
- Regular and functional forward paths now share state transitions, with optimized
  multi-step implementations retained where needed.
- Functional conversion directly composes flat sequential modules and uses the
  registered-memory interface as the fallback for other modules.

#### ANN-to-SNN Conversion

Module: `spikingjelly.activation_based.ann2snn`.

- Added temporal-difference RMSNorm and SiLU operators and conversion support in
  `TransformerTDEquivalentRecipe`.
- Added public `Qwen2SNNConfig`, `Qwen2SNNCalibration`, `Qwen2SNNModel`, and
  `Qwen2SNNRecipe` for calibration-driven, offline layerwise Qwen2 conversion
  with an explicit `[T, B, S, H]` layout and explicit KV-cache continuation.
- Added `SignedQCFSSequenceEncoder` and revision-pinned Qwen2.5 correctness,
  quality, efficiency, and tensor-parallel evaluation tools. These experimental
  workflows do not claim latency or energy improvements.

#### Precision

Module: `spikingjelly.activation_based.precision`.

- Added optional Transformer Engine FP8 support through
  `PrecisionConfig(mode="fp8-te")`, including Linear, pointwise Conv1d,
  LayerNorm, fused LayerNorm patterns, and SDPA adapters.
- Added `fp8-te` and `fp8-torchao` extras, capability diagnostics, and conversion
  reports.

#### Triton Neuron Kernels

Module: `spikingjelly.activation_based.triton_kernel.neuron_kernel`.

- Added a multi-step, inference-only Triton backend for
  `ActivationAwareIFNode` with FP32 and BF16 execution.
- Added experimental mixed-precision and FP8 IF, LIF, and ParametricLIF kernels,
  capability probes, execution plans, and benchmarks. FP8 does not
  consistently outperform BF16 in the measured workloads.
- Added multi-step Triton training and inference for `ILIFNode`. It reuses
  `LIFNode` dynamics and configures integer firing through `MultiLevelSpikeCount`.
- Added an inference-only single-step Triton kernel for the SpikeZIP STBIF
  neuron, exposed through functional and neuron interfaces.

#### Integer-Valued Spike Conversion

Module: `spikingjelly.activation_based.layer`.

- Added `SpikeCountToBinary` and `TemporalBinSum` for multi-step I-LIF networks,
  enabling integer-valued training and binary-event evaluation around bias-free
  weight layers.

#### Layer

Module: `spikingjelly.activation_based.layer`.

- Added `MaxUnpool1d`, `MaxUnpool2d`, and `MaxUnpool3d` step-mode wrappers
  supporting both `'s'` and `'m'` step modes, so the indices returned by
  `MaxPool1d/2d/3d(return_indices=True)` can be consumed in multi-step
  networks (issue #626).

#### Functional Forward

Module: `spikingjelly.activation_based.functional`.

- `seq_to_ann_forward` now also accepts a tuple of tensors sharing the same
  `[T, batch_size]` leading dimensions; each element is flattened and passed
  to the stateless module as a positional argument (issue #626).

### Bug Fixes

#### Activation-Based Layers

Module: `spikingjelly.activation_based.layer`.

- Fixed `DropConnectLinear(p=0)` rarely dropping weights or biases when the
  random sampler returned exactly zero.
- Fixed `TemporalEffectiveBatchNorm3d` rejecting valid multi-step 3D inputs.

#### Activation-Based Surrogate Functions

Module: `spikingjelly.activation_based.surrogate`.

- Fixed non-spiking `LogTailedReLU` returning `NaN` for negative inputs.

#### ANN-to-SNN Conversion

Module: `spikingjelly.activation_based.ann2snn`.

- Fixed TD operators retaining full cumulative sequences instead of compact
  final-step state.
- Fixed FX conversion on PyTorch 2.6 and 2.7 for dynamic `torch.reshape`
  calls and forward signatures containing PEP 604 union annotations.
- Fixed signed QCFS boundary replay and statistics for non-last channel
  dimensions.
- Fixed SpikeZIP STBIF Triton state rounding at exact half-integers to match
  Torch ties-to-even semantics.
- Fixed converted Qwen2 position IDs, cache validation, and calibration
  metadata checks.

#### Activation-Based Sequence Forwarding

Modules: `spikingjelly.activation_based.functional`,
`spikingjelly.activation_based.layer`.

- Fixed time-last containers advancing stateful modules through stateless
  vectorization and preserved tuple outputs from sequence-to-ANN forwarding.

#### Memory Optimization

Module: `spikingjelly.activation_based.memopt`.

- Fixed temporally chunked gradient checkpointing for sequence lengths that are
  not divisible by the requested chunk count and warned when the count exceeds
  the sequence length.
- Restored deepcopy and multiprocessing spawn support for stateful gradient
  checkpointing containers.
- Invalid local-rank environment variables now raise a clear error instead of
  silently falling through to another CUDA device source.

#### Learning

Module: `spikingjelly.activation_based.learning`.

- Fixed STDP learners retaining network or reward autograd graphs across
  iterations.

#### Recurrent Networks

Module: `spikingjelly.activation_based.rnn`.

- Fixed stacked spiking RNN final states and dropout placement.

#### Models

Module: `spikingjelly.activation_based.model.spiking_vggws_ottt`.

- Fixed `OTTTSpikingVGG` ignoring its requested `drop_rate`.

#### Triton Neuron Kernels

Module: `spikingjelly.activation_based.triton_kernel.neuron_kernel`.

- Fixed gradients for non-contiguous upstream tensors in stable multi-step
  IF/LIF/ParametricLIF kernels.

#### Surrogate CUDA Code Generation

Module: `spikingjelly.activation_based.cuda_kernel.auto_cuda`.

- Fixed `LogTailedReLU` `cuda_codes` for `dtype="half2"`: previously produced
  non-compiling CUDA because `if_else_else` mixed scalar and vector half
  intrinsics, `constant` rendered integer inputs as invalid C++ literals,
  and the binary selector formula accumulated rather than selecting.
  The half2 and float paths now compile and produce the documented
  piecewise gradient.

#### Timing-Based Neurons

Module: `spikingjelly.timing_based.neuron`.

- Fixed `Tempotron` outputs when batch or output-feature size is one.

### Improvements

#### CuPy Neuron Backend

Modules: `spikingjelly.activation_based.neuron`,
`spikingjelly.activation_based.functional`,
`spikingjelly.activation_based.cuda_kernel`.

- Consolidated built-in CuPy execution under functional APIs and
  `backend="cupy"` dispatch; removed duplicate `neuron_cupy` and
  `neuron_cupy_lite` paths.
- Moved concrete kernels to `cuda_kernel.neuron_kernel`, organized by
  single-step and multi-step role; `auto_cuda` now contains only code generation.
- Reused generated `RawKernel` objects through a bounded compile-identity cache
  and kept nonlinear custom-op context tokens on CPU, avoiding repeated wrapper
  construction and device-to-host token reads.

#### CUDA SpikeLinear Kernels

Module: `spikingjelly.activation_based.cuda_kernel.spike_linear`.

- Experimental v3 bit-packed and sparse row-index SpikeLinear kernels now accept
  FP32, FP16, and BF16 tensors, accumulate in FP32, and return the input dtype.

#### Timing-Based Models

Modules: `spikingjelly.timing_based.encoding`,
`spikingjelly.timing_based.neuron`.

- Simplified `GaussianTuning` and `Tempotron` to direct PyTorch operations.

#### Distributed Training and Inference

Module: `spikingjelly.activation_based.distributed`.

- Added a Megatron Core training module for large SNN Transformers with DP, TP,
  PP, sequence parallelism, distributed optimization, and sharded checkpoints.
- Added native PyTorch DDP/FSDP2 and pipeline-parallel vision training with
  architecture-specific channel TP/PP for SEW-ResNet34 and head/channel TP/PP
  for Spikformer-S.
- Added shared channel-sharded Conv/BatchNorm primitives for custom SNN parallel
  training loops; stateful neurons consume their local-channel tensors directly.
- Added a functional `[T, B, S, H]` to `[S, T*B, H]` temporal envelope that keeps
  complete SNN time windows rank-local and safe under pipeline recomputation.
- Added repository benchmark workloads for MCore-native SpikeLM BF16 pretraining
  and Qwen2.5 `qcfs_sg` FP8 fine-tuning, including deterministic Hugging Face
  weight import.
- Added Transformer Engine context parallelism that shards only token length `S`
  while keeping complete SNN time windows `T` rank-local.
- Added TP/PP model execution with optimizer-boundary checkpoint loading,
  functional per-call SNN state, and MCore static KV-cache prefill/decode.
- Corrected Qwen2 input-calibration checkpoint metadata so TP ranks restore the
  replicated scale without duplicate main shards.
- Added standalone Vision evaluation and ordered logits-only HDF5 prediction
  with replicated DP, FSDP2, TP, and PP. `EvaluationConfig` requires targets and
  returns aggregate metrics; `PredictionConfig` ignores targets and prediction
  returns no metrics. Canonical inference artifacts can be restored under a
  different TP/PP topology.
- Added a bounded forward-only Vision pipeline schedule for inference, avoiding
  training-schedule state and cross-batch work accumulation. Communication-aware
  SEW-ResNet34 boundaries and balanced six-block Spikformer stages provide stable
  PP throughput above one GPU.
- Changed the SEW-ResNet34 and Spikformer PP stage boundaries for both training
  and inference. Vision PP checkpoints created before this change cannot be
  resumed; checkpoints without PP are unaffected.
- Rejected PP4 for the four-block Spikformer because its fourth stage would
  contain only the classifier head; PP1 and PP2 remain supported.
- Added inference capacity benchmarks that sweep Vision and MCore evaluation
  batches through measured capacity boundaries with a 2x/1.5x search. Three-run
  throughput-memory plots reuse the training-figure style and report per-rank
  and global batches separately.
  Vision and MCore evaluation support untimed warmup batches; Vision excludes
  DataLoader work from reported throughput.
- Added optimizer-free MCore loss/perplexity evaluation with DP/TP/PP/CP and
  explicit forward-only pipeline microbatches, plus DP-sharded cached generation
  with exact prompt ordering.
- Renamed the previous low-level `distributed.llm.generate(transformer_config,
  model_provider, ...)` API to `generate_mcore(...)`. The `generate(...)` name now
  accepts `MCoreGenerationConfig` and `input_ids`; existing low-level callers
  must use `generate_mcore`.
- Added a managed, model-independent SGLang 0.5.17 offline Engine with an
  explicit external-model package. Reference Qwen2 and SpikeLM adapters live in
  `benchmark/snn_llm` rather than the SpikingJelly wheel.
- Added `export_sglang_artifact()`, a distributed MCore-to-sharded-safetensors
  exporter that delegates model weight mapping to a stage callback and never
  requires the complete model to fit on one GPU. Tokenizer assets are copied
  without assuming a tokenizer family.
- Replaced the experimental one-shot SGLang generation interface with the
  managed `SGLangEngineConfig` and `open_sglang_engine()` lifecycle. Sampling,
  variable-length token IDs, asynchronous generation, and streaming use the
  native SGLang Engine interface.
- Excluded dataset indexing and collation from MCore evaluation timing while
  retaining H2D transfer, model execution, communication, and metric reduction.
- Added a reproducible two-panel SGLang result figure showing fixed-workload
  TP/PP/DP scale-out and Radix shared-prefix reuse, backed by the downloadable
  measurement CSV.
- Added typed `distributed.llm.ModelConfig` subclasses that own MCore
  `TransformerConfig` and their model builder, matching the vision declaration
  style; `plan_training()` returns a TP/PP/CP configuration accepted by
  `train(config)` without a global registry or untyped builder kwargs.
- Added serializable `distributed.vision.TrainingConfig`, ImageFolder datasets,
  optimizer-boundary distributed checkpoints, and `train_classification(config)`
  for image classification. SEW-ResNet34 and Spikformer distributed recipes are
  owned by their corresponding `activation_based.model` modules; serialized
  targets from the previous `distributed.vision` model paths are migrated when
  loaded. Direct imports from `distributed.vision` must move to the corresponding
  `activation_based.model` submodule.
- Added importable Vision classification loss functions with serializable keyword
  arguments shared by non-pipeline and pipeline training and validation.
- Added model-owned Vision `step_mode` configuration. Classification training now
  runs an explicit outer time loop for single-step SEW-ResNet34; the intrinsically
  multi-step Spikformer and single-step PP, memopt, and Triton combinations are
  rejected explicitly.
- Added `SpikformerCIFAR10Config` with the official 32×32, 4×4-patch, 384-channel,
  12-head, 4-block architecture while reusing the existing TP, PP, and FSDP2 paths.
- Added serializable batch-level Vision mixup through `TrainingConfig.mixup_alpha`.
- Added explicit Vision `input_layout` handling for static `NCHW` images and
  default-collated `NTCHW` neuromorphic frame sequences.
- Added rank-zero JSON progress output after every Vision training epoch with
  optimizer step, train loss, validation loss, and validation accuracy.
- Fixed `functional.set_step_mode()` changing the single-step modules owned by
  multi-step and sequence-to-ANN containers.
- Fixed single-step DDP buffer synchronization by broadcasting once before each
  complete temporal window instead of once per single-step forward.
- Added SpikingJelly memopt checkpointing for deterministic SNN temporal
  transforms, with non-overlapping MCore selective `core_attn` recomputation as
  a memory fallback. Full MCore recomputation is never selected automatically.
- Kept channel-sharded BatchNorm in full precision under combined TP and FSDP2,
  fixing BF16 running-statistic dtype failures.
- Normalized SpikeLM activations before the spiking transition and returned an
  integer MCore token count, keeping deep BF16 training gradients finite.
- Made the packed `[S, T*B, H]` temporal envelope contiguous for MCore sequence
  parallel all-gathers.
- Reported Vision and LLM throughput after configurable warm-up steps, including
  maximum per-rank and summed allocated/reserved peak memory across ranks.

#### Triton Neuron Runtime

Module: `spikingjelly.activation_based.triton_kernel.neuron_kernel`.

- Reduced memory usage with `store_v_seq=False` by retaining only the final
  membrane potential.
- PLIF and STBIF kernels now load scalar tensor parameters directly on device,
  avoiding device-to-host scalar reads during kernel dispatch.
- Removed the redundant `supports_triton_fp8_e4m3fn()` and
  `supports_triton_fp8_e5m2()` helpers. Use
  `supports_triton_fp8_neuron_forward(dtype, device, compute_dtype)` or the
  corresponding backward capability API.
- Removed the test-oriented `flexsn_kernel_registry_info()` helper; FlexSN
  kernel handles remain managed internally by the registered-op lifecycle.

### Breaking Changes and Notices

#### Dataset Utility API Changes

Module: `spikingjelly.datasets.utils`.

- **Breaking change:** removed `save_every_frame_of_an_entire_DVS_dataset()`,
  `save_frames_to_npz_and_print()`, and `fast_split_to_train_test_set()`. Construct
  datasets directly and use `save_as_pic()`, `numpy.savez`, or
  `split_to_train_test_set()` for the corresponding operations.

#### Timing-Based API Changes

Module: `spikingjelly.timing_based.encoding`.

- **Breaking change:** `GaussianTuning.encode(max_spike_time=0)` now uses the
  empty encoding window directly, marking every neuron inactive, instead of
  silently replacing zero with `100`.

#### Neuron Execution Backends

Modules: `spikingjelly.activation_based.neuron` and
`spikingjelly.activation_based.functional.neuron`.

- **Breaking change:** removed the per-neuron `backend="inductor"` option from
  IF, LIF, and PLIF nodes, together with their dedicated compiled-function cache
  and functional APIs. Use `backend="triton"` for dedicated CUDA kernels, or use
  `backend="torch"` and compile the complete model with `torch.compile`.
- **Breaking change:** removed the equivalent `backend="inductor"` alias from
  FlexSN. Its Triton custom operators and internal kernel state now consistently
  use `triton` in their names; the supported FlexSN backends are `"triton"`,
  `"torch"`, and `"hop"`.

#### Functional Forward API Changes

Module: `spikingjelly.activation_based.base`.

- **Breaking change:** `to_functional_forward()` now returns a function with the
  grouped interface
  `(inputs, states, **kwargs) -> (outputs, updated_states)`. Both `inputs` and
  `outputs` remain tuples when they contain one tensor. Replace calls such as
  `output, state = forward(x, state)` with
  `outputs, states = forward((x,), (state,))` and read the single output from
  `outputs[0]`.

#### Neuron Extension API Changes

Module: `spikingjelly.activation_based.neuron`.

- **Breaking change:** production neuron classes now define their dynamics solely
  through functional state transitions and no longer provide
  `neuronal_charge()`, `neuronal_fire()`, or `neuronal_reset()`. Custom neurons
  that override these hooks must use `SimpleBaseNode` or implement
  `single_step_functional_forward()`.
- **Breaking change:** removed `BaseNode.v_float_to_tensor()`. Custom functional
  modules that require input-dependent state initialization should override
  `materialize_states(inputs, states, step_mode)`.
- **Breaking change:** sequence caches such as `v_seq`, `i_seq`, and `state_seqs`
  are no longer functional states. Functional forward returns only recurrent
  states; regular multi-step forward stores the requested sequences on the module.
- **Breaking change:** `MemoryModule` subclasses without functional or regular
  forward semantics now raise `NotImplementedError` when called. Stateful
  learners such as `STDPLearner`, `MSTDPLearner`, and `MSTDPETLearner` continue
  to expose their computation through `step()`.

#### Logging-Controlled Diagnostics

- Replaced diagnostic `verbose` controls with the package logger; configure
  `spikingjelly.logger.logger` to control output.

#### Surrogate CUDA Code API Changes

Module: `spikingjelly.activation_based.surrogate`.

- Removed the redundant `SurrogateFunctionBase.cuda_code()` interface. Implement and call `cuda_codes(y, x, dtype)` for surrogate-gradient CUDA code generation; `dtype` is `"float"` or `"half2"`.

#### CuPy Neuron Backend API Changes

Modules: `spikingjelly.activation_based.functional` and
`spikingjelly.activation_based.cuda_kernel`.

- Removed `spikingjelly.activation_based.neuron_cupy` and
  `spikingjelly.activation_based.neuron_cupy_lite`; use functional CuPy APIs or
  neuron-node `backend="cupy"` dispatch.
- Moved custom-kernel imports from
  `spikingjelly.activation_based.cuda_kernel.auto_cuda.neuron_kernel` to
  `spikingjelly.activation_based.cuda_kernel.neuron_kernel.multi_step`.
- `if_step_cupy` now takes `(x, v, v_threshold, v_reset, surrogate_function, detach_reset=False)` and
  `lif_step_cupy` takes `(x, v, tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset=False)`;
  caller-created forward/backward kernels are no longer accepted.

#### ANN-to-SNN API Changes

Module: `spikingjelly.activation_based.ann2snn`.

- Removed the generic `ActivationRule`, `HookFactory`, `ReLURule`,
  `ThresholdOptimizer`, and `NeuronFactory` extension points and the
  corresponding `RateCodingRecipe` constructor arguments. Custom graph
  conversion behavior should be implemented as an `FXConversionRecipe`;
  custom neuron construction now uses a scale-aware callable passed as
  `RateCodingRecipe(neuron_factory=...)`.
- Removed the unreleased `"transformer_spike_equivalent"` recipe string;
  use `"transformer_td_equivalent"`.
- Removed the no-op `strict` argument from `SpikeZIPQANNRecipe`; it only
  accepted the default value and did not select another behavior.

#### Distributed API Changes

Module: `spikingjelly.activation_based.distributed`.

- **Breaking change:** replaced the unreleased Analyze/Plan/Apply API,
  `SNNDistributedConfig`, `SNNDistributedRuntime`, generic model registry, and
  separate pipeline runtime with workload namespaces. Use
  `distributed.vision.TrainingConfig` for native PyTorch vision training,
  `distributed.llm.TrainingConfig` for MCore LLM training, or the root
  `tensor_parallel` primitives in a custom loop. No aliases are retained.
- **Breaking change:** replaced the experimental `SGLangGenerationConfig`,
  `create_sglang_engine()`, and one-shot `generate_sglang()` interfaces with
  `SGLangEngineConfig` and the managed `open_sglang_engine()` context. Request
  generation now uses the native SGLang Engine interface.

#### Learning API Changes

Module: `spikingjelly.activation_based.learning`.

- Removed the unused `tau_trace` argument from
  `mstdpet_linear_single_step()`; eligibility traces are updated by
  `MSTDPETLearner`.

#### Operation-Counting API Changes

Module: `spikingjelly.activation_based.op_counter`.

- Removed the redundant `SpikeSimEventEnergyProfiler` and
  `SpikeSimEventEnergyReport` aliases. Use `SpikeSimEnergyProfiler` and
  `SpikeSimEnergyReport`.
- Removed the legacy memory-residency interfaces; they were not connected to
  the current GPU traffic or NeuroMC runtime models.
- Removed `SpikeSimEnergyProfiler.add_warnings()`; warning collection is now
  internal to the SpikeSim estimate entry point.
- `MemoryHierarchyConfig` now represents the single supported NeuroMC v1
  hierarchy directly. Construct it with `MemoryHierarchyConfig()` and use
  `dataclasses.replace()` for modified copies; the redundant
  `neuromc_like_v1()` and `copy()` methods and configurable `preset_name` /
  `technology_nm` constructor fields were removed. Reports retain the
  `"neuromc_like_v1"` preset name.

#### Precision API Changes

Module: `spikingjelly.activation_based.precision`.

- `PrecisionConfig.from_any()` now accepts only `None`, `PrecisionConfig`, a
  precision-mode string, or a dictionary using the current `mode`,
  `strictness`, `fp8_recipe`, and `device` fields. Replace the legacy
  `precision`, `precision_strict`, and `fp8_report` aliases and attribute-style
  configuration objects with these explicit inputs.
- `Float8TorchAOPolicy` no longer accepts `strict` or `fp8_recipe`, and
  `Float8TransformerEnginePolicy` no longer accepts `strict`. Pass these
  settings through `PrecisionConfig` and `prepare_model_for_precision()`;
  `Float8TransformerEnginePolicy` retains its effective `fp8_recipe` argument.

#### Triton Utility Changes

Module: `spikingjelly.activation_based.triton_kernel.triton_utils`.

- Removed the documented `ensure_cleanup_tmp_python_files()` decorator. Callers
  that create temporary Python files should own their lifecycle with
  `tempfile` context managers.
- Removed `torch_dtype_for_triton_compute_dtype()`. Use
  `normalize_triton_compute_dtype_name()` when validating configuration or
  `resolve_triton_compute_dtype()` when constructing a Triton kernel.

#### Triton Neuron Kernel API Changes

Module: `spikingjelly.activation_based.triton_kernel.neuron_kernel`.

- Replaced `TritonNeuronForwardPlan` and
  `prepare_triton_neuron_forward_plan()` with `TritonNeuronExecutionPlan` and
  `prepare_triton_neuron_execution_plan()`. The `compute_dtype` argument and
  related fields are now named `forward_compute_dtype`,
  `forward_compute_dtype_name`, and `forward_compute_tl_dtype` to distinguish
  forward and backward execution.
- Removed `TritonNeuronExecutionPlan.matches()`; prepare a plan for the current
  arguments instead of maintaining a second normalization/comparison path.
- Removed the legacy
  `multistep_{if,lif,plif}_mixed_precision_forward{,_with_plan}` aliases. Use
  `multistep_{if,lif,plif}_mp` and
  `multistep_{if,lif,plif}_mp_with_plan`.

#### Dependencies

- Removed Pydantic from SpikingJelly's runtime and documentation dependencies
  after replacing its internal timing-based validation. Projects that use
  Pydantic directly must declare it as their own dependency.
- Removed the unused `einops` dependency from the runtime, source, and
  documentation dependency lists.

## 2.0.0.dev0 - 2026-07-09

This entry summarizes the user-visible changes since the previous PyPI stable release, `0.0.0.0.14` (`2941330`), through `2.0.0.dev0` (`b4f3b68`).

### Features

#### ANN-to-SNN Conversion

Module: `spikingjelly.activation_based.ann2snn`.

- Added a redesigned conversion subsystem with recipe-based workflows.

- Added conversion recipes and examples for LTB,
  STA-style Transformer conversion, and SpikeZIP QANN/Transformer experiments.

#### Few-Spike and Activation-Aware Neurons

Modules: `spikingjelly.activation_based.neuron`

- Added few-spike neuron for ann2snn research.
- Added activation-aware IF neuron for ann2snn research.

#### Memory Optimization

Module: `spikingjelly.activation_based.memopt`.

- Added the training memory optimization pipeline with gradient checkpointing and spike compression.

#### Precision

Module: `spikingjelly.activation_based.precision`.

- Added a common precision policy interface for configuring precision behavior
  without depending on backend-specific implementation details.

#### Distributed Training

Module: `spikingjelly.activation_based.distributed`.

- Added distributed training and DTensor utilities for larger-scale SNN
  experiments.

#### Profiling and Energy Estimation

Module: `spikingjelly.activation_based.op_counter`.

- Added operation counting tools for profiling SNN models.
- Added inference energy estimation tools.

### Improvements

- Updated the package version scheme from legacy `0.0.0.0.X` numbering to
  PEP 440 compatible V2 versions.

- Raised the runtime baseline to Python `>=3.11` and `torch>=2.6.0`.

- Updated README and documentation pages for the V2 release policy,
  pre-release installation, and pre-V2 dependency pinning.

- Refactored `spikingjelly.visualizing` into focused submodules.

- Refactored the official website.
- Added broader tutorials and API documentation.
- Reworked public API documentation and docstrings across the project.

- Refined datasets, timing-based modules, exchange utilities, backend kernels,
  model helpers, and training utilities across the V2 development line.

- Added broader regression tests for V2.

### Bug Fixes

- Fixed neuron initialization edge cases.

- Fixed reset-state handling edge cases.

- Fixed spiking RNN hidden-state dtype handling.

- Fixed CuPy and Triton backend dispatch issues for neuron evaluation paths.

- Fixed dataset preprocessing edge cases.

- Fixed publication metadata cleanup edge cases.

- Hardened ANN-to-SNN conversion validation and calibration.

- Hardened ANN-to-SNN step-mode, mask-handling, download, and module-replacement
  paths.

- Fixed documentation rendering, tutorial, and API navigation issues.

### Breaking Changes and Notices

- V2 starts a new compatibility generation. Projects that must remain on the
  legacy release line should pin `spikingjelly<2`.

- Some experimental or internal ANN2SNN conversion interfaces were refactored
  around the V2 recipe and operator model.

- Documentation structure and public API pages were reorganized; external links
  to old generated API pages may need to be updated.

- Before upgrading from `0.0.0.0.14`, review this changelog and the V2 README
  installation notes.

- Conservative projects should pin `spikingjelly<2` until they are ready to
  validate V2 behavior.

- To test published V2 pre-releases, install with
  `pip install --pre spikingjelly`.

- For source installs, follow the current README and ensure the selected
  PyTorch build matches the target CPU/CUDA environment.
