# 变更日志 | Changelog

All notable changes to SpikingJelly are documented in this file.

SpikingJelly starts maintaining this standard changelog from `2.0.0.dev0`.
For older releases, see the historical fatal-bug record in
[bugs.md](https://github.com/fangwei123456/spikingjelly/blob/master/bugs.md)
and the archived documentation linked from the project README.

## Unreleased

### Features

#### Logging

Module: `spikingjelly`.

- Added a package-level `spikingjelly` logger with application-owned handlers,
  structured summaries for operator registration, memory optimization, and
  operation-counter results, plus an AST policy checker and logging benchmark.
- Added lifecycle summaries for distributed setup, precision fallback, ANN2SNN and
  NIR conversion, graph transforms, external downloads, and compiler cache events.
- Production diagnostics now use lazy logging at lifecycle boundaries; normal
  forward, dispatch, and kernel execution paths do not emit default logs.
- Documented the package-level logger, handler configuration, NullHandler behavior,
  and common application-side usage patterns.

#### Functional State Transitions

Modules: `spikingjelly.activation_based.functional` and
`spikingjelly.activation_based.base`.

- Added explicit-state functional execution across the framework's stateful
  modules while preserving regular stateful forward behavior.
- Regular and functional forward paths now share state transitions, with optimized
  multi-step implementations retained where needed.
- Functional conversion now covers native, composite, and user-defined modules,
  while optional execution traces remain separate from recurrent state.

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

#### Learning

Module: `spikingjelly.activation_based.learning`.

- Fixed STDP learners retaining network or reward autograd graphs across
  iterations.

#### Recurrent Networks

Module: `spikingjelly.activation_based.rnn`.

- Fixed stacked spiking RNN final states and dropout placement.

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

#### Neuron Backend Caches

Module: `spikingjelly.activation_based.neuron`.

- Moved standard IF/LIF/ParametricLIF Inductor compiled-graph ownership from
  individual neuron instances to a bounded, PID-aware neuron backend cache.
  Equivalent modules can now share compiled callables without serializing cache
  entries through module deepcopy or pickle.

#### Timing-Based Models

Modules: `spikingjelly.timing_based.encoding`,
`spikingjelly.timing_based.neuron`.

- Simplified `GaussianTuning` and `Tempotron` to direct PyTorch operations.

#### Distributed Training

Module: `spikingjelly.activation_based.distributed`.

- Added Analyze -> Plan -> Apply configuration for data parallel, tensor
  parallel, FSDP2, and FSDP2+TP. Pipeline execution uses dedicated builders.
- Added explicit tensor-parallel plans and replicated-activation DTensor styles
  for `TDLinear`.
- Updated distributed benchmarks, result fields, and tutorials.

#### Triton IF/LIF Memory Optimisation

Module: `spikingjelly.activation_based.triton_kernel.neuron_kernel`.

- Reduced memory usage with `store_v_seq=False` by retaining only the final
  membrane potential.

### Breaking Changes and Notices

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

- Removed the generic `ActivationRule`, `HookFactory`, `ReLURule`, and
  `ThresholdOptimizer` extension points and the corresponding
  `RateCodingRecipe` constructor arguments. Custom graph conversion behavior
  should be implemented as an `FXConversionRecipe`; `NeuronFactory` remains
  available for configuring neuron construction.

#### Distributed API Changes

Module: `spikingjelly.activation_based.distributed`.

- Replaced `TensorShardMemoryModule` with the
  `make_tensor_shard_memory_module()` factory. Tensor-parallel stateful modules
  now keep their concrete module type and original state-dict paths instead of
  adding an `inner` module namespace.
- Removed the `spikingjelly.activation_based.distributed.dtensor` compatibility
  facade. Import the high-level Analyze -> Plan -> Apply APIs from
  `spikingjelly.activation_based.distributed` and low-level utilities from their
  `data_parallel`, `pipeline`, or `tensor_parallel` modules.
- Replaced the `SNNDistributedConfig` enable/experimental flags with the
  explicit `mode` values `none`, `dp`, `tp`, `fsdp2`, and `fsdp2_tp`.
  Model-specific tensor-parallel and FSDP roots are now passed directly through
  the corresponding root fields.
- Removed `build_eager_config()`, `SNNDistributedRuntime.from_legacy()`, and the
  `build_cifar10dvs_vgg_eager_policy()` and
  `build_spikformer_eager_policy()` helpers. Use
  `configure_snn_distributed()` for manual eager configuration or the Analyze
  -> Plan -> Apply workflow.

#### Operation-Counting API Changes

Module: `spikingjelly.activation_based.op_counter`.

- Removed the redundant `SpikeSimEventEnergyProfiler` and
  `SpikeSimEventEnergyReport` aliases. Use `SpikeSimEnergyProfiler` and
  `SpikeSimEnergyReport`.
- Removed stage-level aggregation and
  `MemoryResidencyCounter.get_stage_level_bits()`. Use the level- and
  operation-level residency methods for current measurements.

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

#### Triton Neuron Kernel API Changes

Module: `spikingjelly.activation_based.triton_kernel.neuron_kernel`.

- Replaced `TritonNeuronForwardPlan` and
  `prepare_triton_neuron_forward_plan()` with `TritonNeuronExecutionPlan` and
  `prepare_triton_neuron_execution_plan()`. The `compute_dtype` argument and
  related fields are now named `forward_compute_dtype`,
  `forward_compute_dtype_name`, and `forward_compute_tl_dtype` to distinguish
  forward and backward execution.

#### Dependencies

- Removed Pydantic from SpikingJelly's runtime and documentation dependencies
  after replacing its internal timing-based validation. Projects that use
  Pydantic directly must declare it as their own dependency.

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
