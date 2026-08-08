# Ponytail audit

This document records the repository-wide audit of `spikingjelly/` for
over-engineering and defensive programming. The triage rule was correctness and
public contracts first, measured hot-path performance second, and fewer concepts,
states, branches, dependencies, and indirections third.

## Scope and method

- Baseline commit: `11a06feb`.
- Inventory at baseline: 452 Python files and 79 test files.
- CodeGraph was used before text searches to trace candidate helpers, state, and
  call paths.
- Repeated passes checked dead private symbols, unused parameters and attributes,
  one-line forwarding wrappers, identical function bodies, broad/silent exception
  handling, speculative compatibility options, duplicate reset traversal,
  handwritten standard-library operations, and commented-out implementations.
- Every candidate was checked against callers, tests, public documentation, and
  backend/lifecycle boundaries before modification.

## Findings and triage

### High: duplicated rules and lifecycle operations

1. ANN2SNN delay measurement, distributed pipeline runtime, and three examples
   each maintained their own module traversal/reset logic. They now use the
   existing `functional.reset_net`, `collect_reset_modules`, and
   `reset_collected_modules` implementations.
2. `ACCounter` and `SynOpCounter` independently maintained identical spike-driven
   matrix and convolution rules. SynOp now reuses the shared AC rules.
3. NeuroMC addition and multiplication counters contained identical convolution
   arithmetic. The rule now has one owner in `neuromc.utils`.
4. Reduction, memory-access, matrix-addressing, and residency rules had separate
   functions with identical bodies only because their registered ATen operation
   names differed. Each behavior now has one rule, and the registries map all
   applicable operations to it.
5. Six operation-counter modules implemented their own product loop. They now use
   `math.prod`.
6. N-Caltech101 and N-MNIST duplicated the same ATIS binary-to-NPZ conversion.
   The dataset utility module now owns that format conversion once.

### High: dead abstractions, state, and code paths

1. Removed the behavior-free `_TrainPackTracer` subclass and the Conv-BN type
   tables that became unused with it.
2. Removed dead module replacement/reset helpers, nested-dictionary utilities,
   optimizer-fragment wrapper, surrogate capability list, model counters, Lava
   threshold copy, FlexSN mirror state, and per-neuron Inductor cache marker.
3. Replaced the local `Identity` module with `torch.nn.Identity` and simplified
   parameter traversal in `spike_dhs`.
4. Removed commented-out training, checkpointing, backend compilation, debug
   printing, and alternative implementation blocks. Version control, not source
   comments, retains those implementations.
5. Removed redundant `pass` statements whose docstrings already form valid
   protocol method/class bodies, and removed direct `.__len__()` calls in favor of
   normal collection operations.

### Medium: speculative API and defensive fallback

1. Removed `SpikeZIPQANNRecipe.strict`; `False` was rejected and `True` selected no
   alternative behavior.
2. Removed the unused `model_family` argument from distributed `analyze()`; model
   family remains on plans where it affects behavior.
3. Removed the unused `tau_trace` argument from
   `mstdpet_linear_single_step()`; the learner owns eligibility-trace decay.
4. `resolve_device()` no longer hides invalid local-rank environment values or
   CUDA/distributed runtime failures behind a generic CUDA fallback. Invalid rank
   configuration now fails with its source named.
5. Learner eligibility is now a registered memory initialized to `None`, rather
   than an attribute whose existence represented initialization state.
6. Optional backend and compiler boundaries still catch the errors they promise
   to translate, while ordinary internal state no longer gains generic fallback
   paths.
7. Full-suite validation exposed a floating-point boundary where the shared
   pipeline partitioner could return more parts than requested. Reconstruction
   now stops splitting after the requested count, with a deterministic regression
   case for the former three-part result.

### Medium: no-op indirection and ignored input

1. Inlined single-use wrappers for ANN2SNN scaling/channel selection, STA
   threshold broadcasting, memory-optimization module paths, and dataset peeking.
2. Removed unused architecture labels from the OTTT VGG factory.
3. Fixed `OTTTSpikingVGG` forwarding `drop_rate=0.0` instead of the caller's value.
4. Removed an empty platform branch and an always-zero reported accuracy value in
   the sequential-MNIST example.
5. Consolidated identical converter device inference, ANN2SNN submodule
   replacement, Triton CUDA-device normalization, model invocation, and DSQN
   state preprocessing at their existing responsibility boundaries.
6. Replaced hand-written reverse indexing, redundant list/tuple construction,
   and explicit ignored-cleanup `try` blocks with direct iteration and standard
   library constructs where those forms reduce state without hiding the flow.

### Low: dependency surface

- Removed the unused `einops` declaration from runtime, source, and documentation
  dependency lists.

## Deliberately retained

- `reset_net` keeps its weak-key cache: the existing benchmark showed cached reset
  at roughly half the direct traversal time, and tests cover cache lifetime and
  invalidation.
- Backend capability probes retain targeted and, where third-party compilers can
  raise arbitrary errors, broad exception handling. These are external trust
  boundaries that report unavailability rather than internal invariants.
- Torch custom-op fake implementations, autograd callbacks, module hooks, recipe
  lifecycle methods, and distributed adapters keep framework-mandated signatures
  even when an argument is unused by one implementation.
- Remaining exact duplicate bodies are framework-shaped callbacks, separate
  runnable examples, independently vendored DSQN wrappers, or parallel model
  implementations. Sharing them would couple ownership or add an indirection
  without removing a rule.
- Public model constructors, documented recipe/base protocols, strict operation
  counting, hardware calibration controls, and cross-version serialization hooks
  remain because they select current behavior or protect a public boundary.
- Short helpers were retained when the name expresses a domain operation, removes
  noise from a main flow, or owns a real framework/lifecycle seam. Line count alone
  was not used as a deletion criterion.

## Result

The implementation pass changes 90 files under `spikingjelly/`: 323 inserted and
992 deleted lines, a net reduction of 669 lines. No dependency, feature flag,
class, or public compatibility layer was added. The inserted lines are mainly
clear error reporting and shared rule ownership; behavior-level regression checks
live outside `spikingjelly/`.

## Verification

- Local non-test checks: formatting, focused Ruff undefined/unused checks,
  compilation, `git diff --check`, duplicate-body scans, private-symbol/state
  scans, defensive-fallback searches, and commented-code searches.
- Remote g2: source import-path check, Python compilation, generated changelog
  check, and targeted tests (`455 passed, 51 skipped` before the final consolidation;
  `465 passed` on the final code).
- The first final full-suite run found one nondeterministic pipeline-partition
  failure (`2138 passed`). After fixing the shared partitioner, its new regression
  passed and the original failing runtime test passed five consecutive runs.
- The final g2 full suite passed: `2140 passed, 61 skipped, 39 warnings` in
  601.37 seconds.
- A clean-output g2 Sphinx HTML build completed successfully. It reported 1450
  existing documentation warnings; the reused remote build directory was not
  treated as evidence because stale static-file paths prevented that build.

The final duplicate-body, private-symbol/state, broad-exception, commented-code,
deferred-marker, direct-dunder, and simplification-lint passes found no remaining
untriaged high-confidence over-engineering or defensive-programming issue under
`spikingjelly/`. Remaining matches are the deliberate framework, backend,
serialization, model, and standalone-example boundaries listed above.
