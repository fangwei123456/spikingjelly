# 变更日志 | Changelog

Changes since `2.0.0.dev0`. For older releases, see
[bugs.md](https://github.com/fangwei123456/spikingjelly/blob/master/bugs.md) and the
[legacy documentation](https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.14/).

## Unreleased

### Fixes

#### spikingjelly.activation_based.precision

- BF16 capability detection queries non-CUDA accelerator backends instead of rejecting them unconditionally.

#### spikingjelly.datasets

- Time-based fixed-frame integration handles empty intervals without changing
  previously successful results for ordered events.

#### spikingjelly.activation_based.encoding

- `PoissonEncoder` no longer emits spikes for zero-probability FP16/BF16 pixels;
  seeded spike sequences can change when random draws equal input probabilities.
- `LatencyEncoder` evaluates logarithmic firing times without exponential overflow
  for long periods and low-precision inputs, including binary integer inputs.
- `WeightedPhaseEncoder` outputs spikes in the input floating-point dtype instead
  of the global default dtype, allowing direct use with layers of matching dtype.

#### spikingjelly.activation_based.functional

- Convolutional STDP accumulates FP16/BF16 contributions in FP32, avoiding intermediate overflow when finite weight updates combine large spatial sums.
- TET cross-entropy accepts class probability targets, including MixUp and spatial targets.
- `kernel_dot_product` keeps Gaussian squared distances nonnegative, preventing
  rounded self-similarities above one and resulting BCE failures.

- `spike_similar_loss` matches target dtype to similarities for float64 MSE/BCE
  compatibility; with autocast disabled, explicit FP16/BF16 inputs now produce
  matching low-precision MSE/L1 losses instead of FP32.

#### spikingjelly.activation_based.surrogate

- Torch `Sigmoid` surrogate gradients retain representable positive tails in low-precision training.
- `PoissonPass` preserves the input dtype, allowing matching FP16/BF16/FP64 layers.

#### spikingjelly.activation_based.learning

- FPTT training preserves frozen optimizer parameters and their running-average slots instead of failing on missing gradients.

### Changes

- Few-spike, outlier-aware and hierarchically gated neurons add optional native CUDA alongside Triton for eligible FP32 inference, with explicit provider selection and reference fallbacks.

- Neuron dispatch uses version-aware Triton architecture floors in auto mode; explicit Triton attempts retain provider errors.
- Few-spike tracing and incompatible optional Triton imports retain reference execution in automatic mode.

- ST-BIF uses one-third less FP32 state memory and fewer inference allocations.
- Removed `STBIFNode.cur_output`; `functional.stbif_step` and
  `functional.stbif_multi_step` return `(out, q, acc_q)`. Consume returned spikes,
  update tuple unpacking, rebuild native extensions and regenerate compiled/exported graphs.

## 2.0.0 - 2026-10-18

### Features

#### spikingjelly.activation_based.neuron

- Added `TernaryIFNode` and `TernaryLIFNode` with signed spikes, configurable reset,
  surrogate gradients and voltage recording; execution is Torch-only, without
  the paper's learnable spike amplitude or training STE.
- Added `RAFNode` and the explicit two-state `functional.clif_step` transition.
- CLIF supports Triton FP32/FP16/BF16 forward, backward and compilation;
  higher-order gradients and custom/continuous surrogates use Torch.
- Neuron execution follows tensor device placement: CPU uses Torch; CUDA
  automatically selects a compatible native CUDA, Triton or Torch implementation.
- Eager CUDA prefers native kernels; compiled execution prefers Triton;
  higher-order gradients use Torch reference equations.
- FlexSN automatically selects Torch, Triton or Torch/HOP, and retained backward
  graphs remain usable after deleting the node.
- IF/LIF/PLIF support fullgraph compilation with low-precision state and explicit
  precision plans.

#### spikingjelly.activation_based.precision

- Renamed precision settings to `neuron_storage`, `neuron_fwd` and `neuron_bwd`;
  explicit multi-step IF/LIF/PLIF precision uses CUDA Triton.

#### spikingjelly.activation_based.functional

- Added `functional.ternary_if_step` and `functional.ternary_lif_step` for
  explicit-state single-step execution.
- Added fused IF/LIF-Linear, sparse/prepacked Linear and spike packing interfaces,
  with native CUDA acceleration and Torch fallback.
- Fused IF/LIF-Linear supports compiled forward/backward with seven built-in
  surrogates; custom surrogates use eager Torch derivatives.
- Flat spike packing supports CPU, optional Triton CUDA and compilation;
  decompression accepts an output dtype, defaulting to uint8.
- `fuse_conv_bn_eval_modules` supports networks containing atomic multi-step
  modules, including SpikingVGG.

#### spikingjelly.timing_based

- Added experimental FP32 `first_spike_time` solving with first-order implicit gradients;
  callers must supply complete monotone partitions of the threshold residual.
- Added experimental FP32 single-spike `CubicSRM`, `CubicConv2d` and `functional.cubic_conv2d`,
  with compact events and optional Triton acceleration; convolution excludes
  bias, groups and dilation.
- Added experimental fixed-kernel `SRM` with multiple continuous spikes and
  CPU/CUDA FP32/FP64 autograd; overflow and non-finite gradients raise errors.
- Tempotron now solves continuous events with shunting and returns peak voltage,
  earliest peak time and first-spike time, with first-order weight/input-time gradients.
- Optional Tempotron Triton kernels accelerate threshold solving and peak-voltage
  weight gradients; other gradients use Torch, without CUDA Graph capture.
- `GaussianTuning` emits differentiable continuous compact events using fixed
  receptive-field buffers.
- Added two-layer SRM/CubicSRM MNIST examples; Tempotron MNIST uses threshold-scaled
  logits and a firing-state penalty, with 40 epochs by default (`--epoch 100` for reproduction).

#### spikingjelly.activation_based.model

- Added MA-SNN (`MASNN`, `masnn_dvs128_gesture`) and Att-MS-ResNet
  (`AttMSResNet`, `att_ms_resnet18`) with configurable neurons.
- Added `maxformer_cifar` for the official MaxFormer-4-384 CIFAR configuration.
- Model families and shared attention layers accept `spiking_neuron` and neuron
  keyword arguments while retaining their defaults.

#### spikingjelly.activation_based.layer

- `MultiDimensionalAttention` supports `use_temporal`, `use_channel` and
  `use_spatial` subsets; defaults retain all three axes.

#### spikingjelly.activation_based.distributed.vision

- Added serializable `NeuronConfig`; external neuron classes require explicit
  registration in every process.
- Checkpoint uploads and HDF5 prediction writes overlap subsequent work with
  bounded buffering; evaluation/prediction avoid per-batch CUDA synchronization.

#### spikingjelly.nsys

- Added single-host multi-GPU capture, CUDA Graph diagnostics and analysis of
  compute/communication overlap, including uninstrumented CUDA workloads.
- Capture supports profiler windows, named sessions, GIL events and per-node
  CUDA Graph tracing; offline plots accept `--step-index`.
- Analyzer reports use schema v2; recapture old NVTX workloads and use v2
  summaries for comparison.

#### Examples and benchmarks

- Added `examples.plasticity.stdp_mnist`: shallow current-LIF STDP achieves
  90.2233% mean MNIST accuracy over seeds 0, 1 and 2 with the validation-selected recipe.
- Added CIFAR-100 MaxFormer/MaxResNet and Tiny ImageNet CLIF training,
  checkpoint-evaluation and timing workflows in `benchmark/benchmark_image_classification.py`.
- `benchmark.benchmark_snn_single_gpu` supports Spikformer-S ImageNet-size workloads;
  `tools/nsys/nsys_snn.sh` provides model profiling.
- Added installation, numerical/performance and fixed Qwen2.5-0.5B conversion
  checks under `tools/release/`.

### Improvements and fixes

- Native CUDA/Triton neurons accept compact nonoverlapping, channels-last and
  broadcast layouts without layout copies, including states and gradients.
- IF/LIF/PLIF precision preserves spike input dtype; `store_v_seq=False` avoids
  redundant voltage storage, and Triton 3.6 LIF backward compiles correctly.
- Fixed single-step `store_v_seq` recording, including recurrent containers and
  `CubaLIFNode`; `detach_net` also detaches accumulated voltage sequences (#631).
- Single-step voltage recording grows until `reset()` and copies the trajectory
  each step; use it for monitoring/debugging, not performance-critical execution.
- `CubaLIFNode.store_i_seq` remains multi-step-only; `ComplementaryLIFNode` records
  trajectories through `store_state_seqs`, not `v_seq`.
- Fixed Izhikevich detached hard-reset gradients, QIF scalar-decay rounding
  and GatedLIF half-precision state promotion.
- Fixed noncontiguous mixed-precision execution on Triton 3.3.
- Fixed FlexSN fallback for cores without differentiable outputs and restored
  `MaskedPSN` queue updates before single-step overflow errors.
- Fixed `MemoryModule.reset()` after device/dtype conversion and preserved
  overridden sequential `forward` methods in functional conversion (#760).
- `LinearRecurrentContainer` rejects batch/leading-shape changes without
  `functional.reset_net(container)`.
- Fixed OTTT state detachment and consecutive `OTTTLIFNode` spike/trace propagation.
- Fixed `MSTDPLearner` device/dtype handling, activity validation and retained
  reward graphs; learner `close()` releases hooks, records and state.
- Restored `DSpike` construction and DGS search-parameter device/dtype handling.
- Qwen2 `exact_td`/`qcfs_sg` conversion avoids repeated attention; collapsed-QCFS
  SGLang uses base KV heads, requiring re-export of older expanded-head artifacts.
- Restored `WSLinear(gain=True)`, Torch 2.6 spike unpacking and rank-local
  distributed checkpoint loading; fused kernels report shared-memory limits correctly.
- Reduced neuron import-time work and scalar ActivationAwareIF temporary allocations.
- Added `psutil` to the `megatron` extra for distributed checkpoint saving.
- CLIF task benchmarks now use standard SJ `Rect` gradient boundaries;
  earlier inclusive-window training records are not directly comparable.

### Breaking changes and migration

- Removed neuron `backend`, `supported_backends`, `functional.set_backend` and
  duplicate `*_cupy`, `*_triton`, `*_registered` APIs; tensor device selects execution.
- Use `functional.neuron_implementation` for diagnostics;
  `SJ_<NEURON>_CUDA_IMPLEMENTATION` remains a strict advanced override.
- Removed `activation_based/cuda_kernel`, `activation_based/triton_kernel`,
  experimental neuron wrappers and `SJ_USE_TRITON_OP`; custom neurons use
  explicit-state transitions or FlexSN.
- Removed CuPy providers and `cupy11`/`cupy12` extras; optional native CUDA builds
  (`SJ_BUILD_NATIVE_CUDA=1`) require matching Torch/CUDA tooling and fail when prerequisites are missing.
- Rebuild native extensions and regenerate compiled/exported graphs for the
  updated layout support; default wheels remain pure Python.
- Removed `layer.SpikeLinear`/`SpikeConv*`, `functional.spike_linear`/`spike_conv*`
  and `save_bool_spike_level` (`SJ_SAVE_BOOL_SPIKE_LEVEL`); use Torch layers with `memopt`.
- Removed Auto CUDA's translator/DSL, generated-neuron wrappers and
  `surrogate.cuda_codes()`; fixed neurons use explicit CUDA sources.
- Removed `cuda_threads`, `cuda_compiler_options`, `cuda_compiler_backend` and
  `save_spike_as_bool_in_neuron_kernel`, including their `SJ_*` variables.
- Inference artifacts require schema v2; re-export v1 artifacts, whose neuron
  configuration still contains a backend.
- Renamed `STBIFNeuron` to `STBIFNode`; Spikformer positional `tau` and
  `detach_reset` arguments must become keyword arguments.
- `learning` is now a package; existing learner imports still work. Replace
  layer-object step helpers with `functional.learning` tensor rules or learners.
- Import FPTT/OTTT/SLTT functions from `learning.online_learning` and move examples
  to `examples.plasticity.{stdp_trace,mstdp,mstdpet,stdp_mnist}`.
- Remove `MSTDPLearner(batch_size=...)` and reset before batch-size changes;
  reward `step()` consumes one activity pair per forward and modulates retained eligibility.
- STDP time constants must be finite and at least one step; reward-rule constants
  must be finite and positive.
- Tempotron replaces `T`/`ret_type` with `duration`, takes
  `(spike_times, input_indices, batch_offsets)` and returns `(v_max, t_peak, t_spike)`;
  callers supply their loss instead of built-in `mse_loss`.
- Tempotron uses unit-peak PSPs, shunted continuous peaks, weight initialization
  std 0.001; rename checkpoint key `model.summation_layer.weight` to `weight`.
- Tempotron non-firing times are `inf`, grazing does not fire, and output events
  precede simultaneous inputs.
- `GaussianTuning.forward` replaces `encode`, takes positive floating-point
  `duration` and returns compact events; checkpoint buffers are `mu` and `sigma2`.
- Distributed vision diagnostics now use the package logger; enable its namespace
  and configure a sink, or use the CLI's JSON progress output.
- Profiling workloads moved to `examples/profiling/`, NSYS tools to `tools/nsys/`
  and plotting to `tools/viz/`; retired private benchmark runners are replaced
  by current entries in `benchmark/README.md` and `benchmark.snn_llm.cli`.
- The project now uses Apache-2.0 with third-party notices in `LICENSES/NOTICE`;
  existing tags and published distributions retain their original license records.
- Current docs retain the V2 migration guide; older tutorials remain in the
  [legacy documentation](https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.14/).

## 2.0.0rc1 - 2026-08-29

### Features and improvements

#### spikingjelly.activation_based.distributed

- Added opt-in model-level CUDA Graph execution for Common/Vision workflows and
  native MCore integration, with eager fallback for unsupported shapes.
- Added topology-independent Vision inference artifacts and DP/TP/PP
  evaluation, prediction and export.
- Added MCore loss/perplexity evaluation, cached generation and SGLang export
  without assembling the full model on one GPU.

#### spikingjelly.activation_based.neuron

- Added `ComplementaryLIFNode` with single/multi-step Torch execution and
  optional trajectories for both states.
- Fixed unintended Triton surrogate FP64 arithmetic, redundant LIF voltage
  allocation and multi-step backend setup in SEW-ResNet/Spikformer builders.

#### spikingjelly.activation_based.memopt

- Added `checkpoint`, state-dict-transparent `checkpoint_module`, stateless
  `SpikeCompressor` and explicit temporal chunking.
- Added `optimize_memory` levels 0-4 with configurable checkpoint budgets and
  spatial/temporal rules, integrated with Vision/MCore training only.

#### spikingjelly.activation_based.op_counter

- Validated Lemaire, SpikeSim dense and NeuroMC estimates against author
  code/equations; NeuroMC now uses the author's weight-stationary mapping.

#### spikingjelly.activation_based.monitor

- Added `BaseMonitor.close()` and context managers; `InputMonitor` transforms
  inputs before forward, and STDP learners safely consume/validate monitor records.
- `GPUMonitor` stops promptly, reports `nvidia-smi` failures and closes its writer.

### Breaking changes and migration

- Replace `PredictionConfig.compile=False/True` with `execution_mode="eager"/"compile"`;
  use `"cuda_graph"` for manual capture.
- Replace Common Trainer `--compile` with `--execution-mode compile`;
  `--compile-disable-cudagraphs` is removed and compile options require compile mode.
- Vision CUDA Graph is single-rank and excludes FSDP2, FP8, memopt, CuPy neurons
  and stored voltage trajectories in this release.
- `FlexSN` replaces `FlexSNKernel`, with explicit-state `functional_forward`
  and managed-state `forward`; states, trajectories and multiple outputs use tuples.
- Remove FlexSN `num_inputs`, `num_outputs`, `example_inputs`, `example_outputs`
  and `requires_grad`; arities are inferred, and shared tensor parameters use `static_inputs`.
- FlexSN rejects empty sequences and unsupported explicit Triton requests;
  experimental native scan/while-loop flags are removed.
- Replace `fp8-te` with `fp8` and install `spikingjelly[fp8]`;
  the TorchAO FP8 backend is removed.
- Precision's public API is `PrecisionConfig`, `PrecisionArtifacts` and
  `prepare_model_for_precision()`; policy/helper/report-writer/execution-plan APIs are removed.
- `PrecisionConfig` adds `triton_storage`, `triton_fwd`, `triton_bwd` and
  `fp8_fallback_dtype` (default BF16); misaligned Linear/pointwise Conv1d layers
  stay in high precision and are reported as unsupported.
- Vision supports experimental FP8/Triton precision with DDP and migrates old
  scalar FP32/FP16/BF16 recipes; removed FP8 names are rejected, and MCore keeps native precision configuration.
- The new memopt API replaces `memory_optimization`, `input_compressed_gc`,
  `GCContainer`, `TCGCContainer`, mutable compressors, presets and `__spatial_split__` hooks.
- MCore checkpoints containing the old `use_snn_memopt` recipe field cannot resume directly.

## 2.0.0.dev2 - 2026-08-23

### Features and improvements

#### spikingjelly.activation_based.ann2snn

- Rate-coding percentile calibration now uses exact `torch.quantile`.

#### spikingjelly.activation_based.op_counter

- Added `SimpleEnergy*` estimates combining MAC/AC work and logical neuromorphic
  memory traffic, with configurable memory cost (default `24.96 pJ/byte`).
- Added `NeuromorphicMemoryAccessCounter`, runtime module rules through
  `ModuleCounter`/`ModuleCounterMode`, and provenance/cost settings in reports.
- FLOP counting uses two FLOPs per MAC, supports fused SDPA and reports skipped operations.
- Lemaire uses runtime spike activity and its fixed 32-bit IF/LIF memory model;
  NeuroMC includes register traffic and aggregate SRAM capacities.
- SpikeSim dense uses the released `c2627bc` PE-cycle formula;
  event mode remains a separate SpikingJelly sparse model.

#### spikingjelly.activation_based.nir_exchange

- Added bidirectional Conv1d/current-LIF conversion and explicit imported graph
  state; dependencies are `nir>=1.0.7,<2` and `nirtorch>=2.6,<2.7`.

#### spikingjelly.activation_based.model

- Added QKFormer, MS-ResNet, Max-ResNet, MaxFormer and Spike-driven Transformer v1
  models/builders, available directly from the model package.

### Fixes

- Corrected Lemaire per-layer SRAM/addressing costs, membrane/spike traffic and
  grouped/depthwise fanout; unsupported neurons/transposed convolutions are rejected.
- Spike-convolution counting preserves large integer counts; fused-attention
  traffic includes masks/biases and their available gradients.
- Energy profilers accept keyword tensors, exclude probe internals, preserve
  backward shape metadata and reject rebinding during active contexts.
- NeuroMC handles repeated backward calls and checkpoint recomputation, and
  rejects conflicting stage options or ambiguous selective backward.
- Restored NIR 1.0.8 export, corrected convolution metadata and preserved source
  neuron memories during export shape inference.
- Interrupted dataset preparation leaves a `.building` marker;
  NPZ archives close after reads, and SHD supports spawned DataLoader workers.
- Fixed DSQN replay batches with NumPy 2.x.

### Breaking changes and migration

- Replace `ComputeEnergyCostConfig`, `ComputeEnergyConfig`, `ComputeEnergyProfiler`,
  `ComputeEnergyReport` and `estimate_compute_energy` with their `SimpleEnergy*`
  equivalents and `estimate_simple_energy`.
- Rename `estimate_spikesim_event_energy` to `estimate_spikesim_energy`;
  Lemaire/SpikeSim now default to strict unsupported-path handling (`strict=False` for partial reports).
- Remove NeuroMC `core_type`, `strict`, `extra_ignore_modules`, and Lemaire's
  unused non-binary sparsity/custom-state settings; `NeuronStateCounter` remains available.
- Unsupported NeuroMC runtime operations still fail rather than returning partial estimates.
- NeuroMC stages use `phase`, `reuse_weights` and `batch_norm_backward`;
  manual profiling requires `bind_model()` before entering the context.
- Replace `get_unsupported_ops(counter)` with `get_unsupported(counter)`;
  use `NeuromorphicMemoryAccessCounter` through `ModuleCounterMode`, without its old context-manager or `bind_model()` API.
- Removed unused memory-residency/NeuroMC primitive interfaces and
  `LemaireAddressingCounter`; addressing is included in Lemaire's module counter.
- Import analytical energy symbols from `op_counter` or `op_counter.analytical_energy`;
  `op_counter.analytical_energy.core` is removed.
- NIR rejects soft reset, grouped convolutions and incompatible heterogeneous
  neuron parameters; recurrent graphs require single-step mode.
- Continue imported NIR execution with the returned state; `state=None` restarts,
  while `reset_net` does not reset an already returned state value.
- Remap MaxFormer stage3 checkpoints using the former private SSA parameter layout.
- Move `model.train_*` imports/commands to `activation_based.examples`;
  shared training helpers live in `examples.common` and entry names use `snake_case`.
- Model wildcard exports contain models/builders; import implementation blocks
  from their model modules, whose composite networks now use plain `nn.Module`.
- Removed the example Trainer's unused CIFAR10 loader and no-op epoch hooks.

## 2.0.0.dev1 - 2026-08-14

### Features

#### spikingjelly.configure and spikingjelly.logger

- Configure package options through `SJ_*` variables before first import;
  assigning `spikingjelly.configure` attributes is no longer supported.
- SpikingJelly now uses Loguru, disabled by default; applications enable
  `logger.enable("spikingjelly")` and own sink configuration.

#### spikingjelly.activation_based.base

- Added explicit-state functional execution across stateful modules while
  retaining regular stateful forward and optimized multi-step paths.

#### spikingjelly.activation_based.ann2snn

- Added TD RMSNorm/SiLU and `TransformerTDEquivalentRecipe` conversion support.
- Added experimental `Qwen2SNNConfig`, `Qwen2SNNCalibration`, `Qwen2SNNModel` and
  `Qwen2SNNRecipe` for calibrated, offline layerwise conversion with explicit
  `[T, B, S, H]` layout and KV-cache continuation.
- Added `SignedQCFSSequenceEncoder` and revision-pinned Qwen2.5 evaluation tools;
  these experiments do not establish latency or energy gains.

#### spikingjelly.activation_based.precision

- Added optional Transformer Engine FP8 via `PrecisionConfig(mode="fp8-te")`,
  covering Linear, pointwise Conv1d, LayerNorm/fused patterns and SDPA adapters.
- Added `fp8-te`/`fp8-torchao` extras, capability diagnostics and conversion reports.

#### spikingjelly.activation_based.triton_kernel.neuron_kernel

- Added multi-step inference for `ActivationAwareIFNode` (FP32/BF16), multi-step
  training/inference for `ILIFNode`, and single-step inference for SpikeZIP STBIF.
- Added experimental mixed-precision/FP8 IF/LIF/PLIF kernels;
  FP8 did not consistently outperform BF16 in measured workloads.

#### spikingjelly.activation_based.layer and functional

- Added `SpikeCountToBinary`/`TemporalBinSum` for integer-spike training and
  binary-event evaluation around bias-free weight layers.
- Added single/multi-step `MaxUnpool1d`, `MaxUnpool2d` and `MaxUnpool3d`;
  `seq_to_ann_forward` accepts tuple
  inputs sharing `[T, batch_size]` dimensions and preserves tuple outputs (#626).

#### spikingjelly.activation_based.distributed

- Added MCore LLM training with DP/TP/PP, sequence/context parallelism,
  distributed optimization and sharded checkpoints, keeping SNN time windows rank-local.
- Added native DDP/FSDP2 Vision training, SEW-ResNet34/Spikformer TP/PP and
  channel-sharded Conv/BatchNorm primitives for custom loops.
- Added serializable Vision `TrainingConfig`, model-owned recipes, importable
  losses and `mixup_alpha`.
- Vision `input_layout` supports static `NCHW` images and default-collated
  `NTCHW` neuromorphic frames.
- Added `SpikformerCIFAR10Config` for the official 32×32, four-block configuration.
- Added typed LLM `ModelConfig` subclasses and `plan_training()` configurations
  accepted by `train(config)`.
- Added Vision aggregate evaluation and ordered logits-only HDF5 prediction;
  prediction ignores targets and returns no metrics, and artifacts support topology changes.
- Added optimizer-free MCore loss/perplexity evaluation and DP-sharded cached
  generation with exact prompt ordering.
- Added managed SGLang 0.5.17 engines and distributed `export_sglang_artifact()`;
  Qwen2/SpikeLM adapters live in `benchmark/snn_llm`, outside the wheel.
- Added SpikeLM BF16 pretraining, Qwen2.5 `qcfs_sg` FP8 fine-tuning and
  throughput/memory benchmarks with configurable warmup and per-rank reporting.

### Improvements and fixes

- Fixed `DropConnectLinear(p=0)`, valid multi-step `TemporalEffectiveBatchNorm3d`
  inputs and negative-input `LogTailedReLU` NaNs.
- TD operators retain compact final-step state instead of full cumulative sequences.
- Fixed PyTorch 2.6/2.7 FX conversion for dynamic reshape and PEP 604 annotations,
  signed QCFS non-last-channel boundaries, and Qwen2 position/cache/calibration handling.
- Fixed STBIF Triton half-integer state rounding to match Torch ties-to-even.
- Fixed state advancement in time-last containers, stacked spiking RNN final
  states/dropout and `OTTTSpikingVGG.drop_rate`.
- Fixed uneven temporal checkpoint chunks, excessive chunk-count warnings,
  deepcopy/spawn support and invalid local-rank handling.
- Fixed retained autograd graphs in STDP learners and noncontiguous upstream
  gradients in Triton IF/LIF/PLIF.
- Fixed `LogTailedReLU.cuda_codes(dtype="half2")` compilation and piecewise gradients.
- Fixed Tempotron outputs for singleton batches/features; simplified
  GaussianTuning and Tempotron Torch execution.
- Consolidated CuPy execution and cached generated kernels, avoiding repeated
  wrapper construction and device-to-host token reads.
- Experimental v3 packed/sparse SpikeLinear supports FP32/FP16/BF16 inputs,
  FP32 accumulation and input-dtype output.
- Triton `store_v_seq=False` retains final voltage only; PLIF/STBIF load scalar
  parameters on-device without host synchronization.
- Vision inference uses bounded forward-only pipeline scheduling; evaluation
  timing excludes data loading/collation and supports untimed warmup.
- `set_step_mode()` preserves single-step children of temporal containers;
  DDP synchronizes buffers once per complete temporal window.
- Fixed TP+FSDP2 BF16 BatchNorm, Qwen2 scale restoration and SpikeLM BF16 gradients.
- Added SNN temporal-transform checkpointing with MCore selective `core_attn`
  fallback; full MCore recomputation is never selected automatically.

### Breaking changes and migration

- Stdlib logging handlers and pytest `caplog` no longer capture package records;
  use a Loguru sink and replace diagnostic `verbose` controls with logger configuration.
- Replace removed dataset utilities `save_every_frame_of_an_entire_DVS_dataset()`,
  `save_frames_to_npz_and_print()` and `fast_split_to_train_test_set()` with
  `save_as_pic()`, `numpy.savez` and `split_to_train_test_set()`.
- `GaussianTuning.encode(max_spike_time=0)` now yields an empty encoding window,
  rather than substituting 100.
- Removed neuron `backend="inductor"`; use Triton kernels or compile the Torch
  model with `torch.compile` (FlexSN also supports `"hop"` in this release).
- `to_functional_forward()` uses `(inputs, states, **kwargs) -> (outputs, updated_states)`;
  inputs and outputs remain tuples even for one tensor.
- Custom neurons overriding `neuronal_charge()`, `neuronal_fire()` or
  `neuronal_reset()` must use `SimpleBaseNode`
  or `single_step_functional_forward()`; replace `v_float_to_tensor()` with `materialize_states()`.
- Functional state excludes `v_seq`, `i_seq` and `state_seqs`; regular forward
  stores requested trajectories, and learners continue to compute through `step()`.
- `MemoryModule` subclasses without forward semantics now raise `NotImplementedError`.
- Replace `SurrogateFunctionBase.cuda_code()` with `cuda_codes(y, x, dtype)`,
  using `"float"` or `"half2"`.
- Removed `neuron_cupy`/`neuron_cupy_lite`; use functional CuPy APIs or
  `backend="cupy"`, with custom kernels under `cuda_kernel.neuron_kernel.multi_step`.
- `if_step_cupy` and `lif_step_cupy` accept dynamics, surrogate and `detach_reset`
  arguments instead of caller-created forward/backward kernels.
- Replace ANN2SNN `ActivationRule`, `HookFactory`, `ReLURule`, `ThresholdOptimizer`
  and `NeuronFactory` extensions with `FXConversionRecipe` or
  `RateCodingRecipe(neuron_factory=...)` scale-aware callables.
- Replace `"transformer_spike_equivalent"` with `"transformer_td_equivalent"`;
  remove the no-op `SpikeZIPQANNRecipe.strict` argument.
- Distributed workloads replace Analyze/Plan/Apply, `SNNDistributedConfig`,
  `SNNDistributedRuntime` and the generic registry; use Vision/LLM `TrainingConfig`
  or root `tensor_parallel` primitives.
- Replace `SGLangGenerationConfig`, `create_sglang_engine()` and `generate_sglang()`
  with `SGLangEngineConfig`, `open_sglang_engine()` and native Engine requests.
- Low-level `distributed.llm.generate(transformer_config, model_provider, ...)`
  is now `generate_mcore(...)`; `generate(...)` takes `MCoreGenerationConfig` and `input_ids`.
- Vision PP stage boundaries changed: older PP checkpoints cannot resume;
  non-PP checkpoints are unaffected, and four-block Spikformer supports PP1/PP2, not PP4.
- Move distributed Vision model imports to `activation_based.model`;
  serialized old targets are migrated when loaded.
- Single-step Vision training supports SEW-ResNet34; Spikformer and single-step
  PP/memopt/Triton combinations are rejected.
- Removed `mstdpet_linear_single_step(tau_trace=...)`; `MSTDPETLearner` owns eligibility updates.
- Replace `SpikeSimEventEnergyProfiler`/`SpikeSimEventEnergyReport` with
  `SpikeSimEnergyProfiler`/`SpikeSimEnergyReport`; obsolete memory-residency
  interfaces and `SpikeSimEnergyProfiler.add_warnings()` are removed.
- Use `MemoryHierarchyConfig()` and `dataclasses.replace()` instead of
  `neuromc_like_v1()`/`copy()` or `preset_name`/`technology_nm` constructor fields.
- `PrecisionConfig.from_any()` accepts `None`, a config, mode string or dictionary
  with `mode`, `strictness`, `fp8_recipe` and `device`; replace `precision`,
  `precision_strict`, `fp8_report` and attribute-style aliases.
- Move `Float8TorchAOPolicy`'s `strict`/`fp8_recipe` and
  `Float8TransformerEnginePolicy`'s `strict` into `PrecisionConfig` and
  `prepare_model_for_precision()`; Transformer Engine retains `fp8_recipe`.
- Replace `ensure_cleanup_tmp_python_files()` with caller-owned `tempfile` contexts,
  and `torch_dtype_for_triton_compute_dtype()` with
  `normalize_triton_compute_dtype_name()` or `resolve_triton_compute_dtype()`.
- Replace `TritonNeuronForwardPlan`/`prepare_triton_neuron_forward_plan()` with
  `TritonNeuronExecutionPlan`/`prepare_triton_neuron_execution_plan()`;
  fields become `forward_compute_dtype`, `forward_compute_dtype_name` and
  `forward_compute_tl_dtype`.
- Remove `TritonNeuronExecutionPlan.matches()` and use current-argument preparation;
  replace `multistep_{if,lif,plif}_mixed_precision_forward{,_with_plan}` with
  `multistep_{if,lif,plif}_mp{,_with_plan}`.
- Replace `supports_triton_fp8_e4m3fn()`/`supports_triton_fp8_e5m2()` with
  `supports_triton_fp8_neuron_forward(...)` or the backward capability API;
  `flexsn_kernel_registry_info()` is removed.
- Removed Pydantic and `einops` dependencies; declare them directly if your project uses them.

## 2.0.0.dev0 - 2026-07-09

Changes from `0.0.0.0.14` (`2941330`) through `2.0.0.dev0` (`b4f3b68`).

### Features

- `spikingjelly.activation_based.ann2snn`: recipe-based conversion with LTB,
  STA-style Transformer and SpikeZIP QANN/Transformer experiments.
- `spikingjelly.activation_based.neuron`: few-spike and activation-aware IF neurons
  for ANN-to-SNN research.
- `spikingjelly.activation_based.memopt`: training checkpointing and spike compression.
- `spikingjelly.activation_based.precision`: shared precision policy configuration.
- `spikingjelly.activation_based.distributed`: distributed training and DTensor utilities.
- `spikingjelly.activation_based.op_counter`: operation counting and inference energy estimates.

### Improvements and fixes

- Adopted PEP 440 V2 versioning and raised requirements to Python `>=3.11` and `torch>=2.6.0`.
- Reorganized visualization modules, the website, tutorials and API documentation.
- Fixed neuron initialization/reset, spiking RNN state dtypes, CuPy/Triton
  evaluation dispatch and dataset preprocessing.
- Improved ANN2SNN calibration, validation, step modes, masks, downloads and module replacement.

### Breaking changes and migration

- V2 starts a new compatibility generation; pin `spikingjelly<2` for legacy behavior.
- ANN2SNN interfaces and documentation paths changed; review the V2 README before upgrading.
- Install published prereleases with `pip install --pre spikingjelly`;
  source installations require a PyTorch build matching the target CPU/CUDA environment.
