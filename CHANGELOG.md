# Changelog

All notable changes to SpikingJelly are documented in this file.

SpikingJelly starts maintaining this standard changelog from `2.0.0.dev0`.
For older releases, see the historical fatal-bug record in
[bugs.md](https://github.com/fangwei123456/spikingjelly/blob/master/bugs.md)
and the archived documentation linked from the project README.

## Unreleased

### Added

- None.

### Changed

- None.

### Fixed

- None.

### Breaking Changes

- None.

### Migration Notes

- None.

## 2.0.0.dev0 - 2026-07-09

This entry summarizes the user-visible changes since the previous PyPI stable
release, `0.0.0.0.14`
(`294133011f4897756db6d1cd4a617a00bfb8d7f8`), through `2.0.0.dev0`
(`b4f3b68a6260ebd42cf1585a3284cf6bcee1e112`). It is a curated release
summary rather than a commit-by-commit list.

### Added

- Added the new ANN2SNN conversion framework, including rule/factory/threshold
  conversion pipelines and extensible recipe objects.

- Added time-distributed ANN2SNN operators and Transformer-oriented conversion
  support, including TD linear, activation, attention, softmax, and sequence SNN
  building blocks.

- Added STA Transformer conversion support and tutorial coverage for the
  Transformer conversion workflow.

- Added Local Threshold Balancing and SpikeZIP QANN/Transformer conversion
  recipes, examples, and documentation.

- Added few-spike neuron primitives and related toy attention/block tests for
  conversion experiments.

- Added FP8 precision tooling, conversion reports, and precision examples.

- Added memory-optimization and checkpointing utilities.

- Added distributed/dtensor support and distributed training/benchmark tests.

- Added operation counting and energy-estimation utilities, including compute
  energy and SpikeSim-style event energy reporting.

- Added broad regression coverage for ANN2SNN, Triton/CuPy backends,
  distributed execution, precision conversion, checkpointing, op counters, and
  visualization.

### Changed

- Changed the package version scheme from legacy `0.0.0.0.X` development
  numbering to PEP 440 compatible V2 versions such as `2.0.0.dev0`.

- Raised the package dependency baseline to Python `>=3.11` and
  `torch>=2.6.0`; documentation builds target the PyTorch 2.7 line.

- Updated README and documentation version policy to describe V2 SemVer-style
  releases, V2 pre-release installation with `pip install --pre spikingjelly`,
  and the recommended `spikingjelly<2` pin for pre-V2 projects.

- Refactored `spikingjelly.visualizing` into focused submodules and added
  torch-backed visualization support.

- Reworked public API documentation and docstrings toward bilingual Chinese and
  English content with consistent Sphinx/RST fields.

- Refined datasets, timing-based modules, NIR/Lava/Lynxi exchange paths, CUDA
  kernel utilities, Triton kernels, model helpers, and training utilities across
  the V2 development line.

### Fixed

- Fixed an `LIAFNode.__init__` attribute error caused by a dead assertion.

- Fixed CuPy and Triton backend dispatch issues, including strict backend
  handling for IF/LIF neuron evaluation paths.

- Fixed `reset_net` cache-key handling to avoid stale or unsafe reset behavior.

- Fixed dataset and preprocessing edge cases, including frame integration
  boundaries and publisher-field sanitization for publication metadata.

- Fixed spiking RNN hidden-state dtype handling so default hidden states follow
  the input dtype.

- Hardened ANN2SNN calibration, step-mode adapters, mask handling, module
  refresh, neuron replacement, download validation, and conversion input
  validation.

- Fixed docstring, Sphinx rendering, tutorial, and API toctree issues found
  during the V2 documentation cleanup.

### Breaking Changes

- V2 starts a new compatibility generation. Projects that must remain on the
  legacy release line should pin `spikingjelly<2`.

- The minimum supported runtime baseline is higher than `0.0.0.0.14`, including
  Python `>=3.11` and `torch>=2.6.0`.

- Some experimental or internal ANN2SNN conversion interfaces were refactored
  around the V2 recipe and operator model. Code depending on internal
  conversion details should migrate to the V2 documentation.

- Documentation structure and public API pages were reorganized; external links
  to old generated API pages may need to be updated.

### Migration Notes

- Before upgrading from `0.0.0.0.14`, review this changelog and the V2 README
  installation notes.

- Conservative projects should pin `spikingjelly<2` until they are ready to
  validate V2 behavior.

- To test published V2 pre-releases, install with
  `pip install --pre spikingjelly`.

- For source installs, follow the current README and ensure the selected
  PyTorch build matches the target CPU/CUDA environment.
