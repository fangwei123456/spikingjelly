# Neuron Operator Migration Acceptance

Validation completed on 2026-10-06 from worktree base revision
`0e5b72bbf29c5f6c10d32d60e30af15f5fa6384f` plus the uncommitted migration
changes.

## Dispatch and packaging

- CPU neuron calls use the registered Torch reference implementation. CUDA calls
  dispatch automatically and cache a compatible provider per device. Neuron
  constructors and `functional.neuron_implementation` diagnostics do not expose a
  construction-time backend option.
- Ordinary CUDA backward uses the selected fused implementation. When
  `create_graph=True` requests higher-order gradients, the same Torch reference
  equations are recomputed on CPU or CUDA.
- The updated package builds a `py3-none-any` wheel and an sdist. The wheel maps
  `ops/` to `spikingjelly._ops`, contains no CUDA shared libraries or deleted
  `activation_based/cuda_kernel` / `triton_kernel` packages, and imports from
  outside the source tree. The sdist includes the native `.cu` and `.cuh` sources.
- Installed the wheel into a temporary target outside the repository; CPU LIF
  forward/backward, the STBIF FP64 reference path, and STBIF validation errors
  passed.

## G-series

The host has two RTX 3090 GPUs, PyTorch `2.7.1+cu118`, and Triton `3.3.1`. The
available system `nvcc` is CUDA 10.1, which does not match PyTorch CUDA 11.8;
CuPy JIT also lacks `cuda_bf16.h`. Therefore this host could not validate native
CUDA or CuPy. Triton was exercised on the GPUs.

The final migrated CUDA correctness suite passed **632 tests**, with **27 skipped**.
The targeted CUDA double-backward matrix also passed for IF, LIF, PLIF, QIF, EIF,
Izhikevich, and I-LIF (**7 passed**).
The forced-provider test skipped only implementations unavailable with this
toolchain; the compatible Triton paths passed.

The standard paired Triton LIF backward check used CUDA Graph replay, three
independent rounds, and 60 paired samples. Candidate median latency was **1.10%
higher** than the existing precision kernel, within the **2%** limit; the worst
round was 1.20% higher and maximum repeat spread was 0.53%.

A separate eager forward/backward microbenchmark compared the baseline commit
against automatic selection on the same RTX 3090 in five alternating process
rounds, using T=1/N=512 and T=16/N=2048. Each median covers five repeats of 25
steps. The baseline explicitly selected Triton for IF/LIF/PLIF and used Torch for
Izhikevich; the candidate selected Triton for all four. Median wall-time changes
were:

| Neuron | T=1 inference | T=1 forward/backward | T=16 inference | T=16 forward/backward |
| --- | ---: | ---: | ---: | ---: |
| IF | 0.1935→0.1593 ms (-17.7%) | 1.5000→0.9306 ms (-38.0%) | 0.1956→0.1853 ms (-5.2%) | 1.5117→0.9451 ms (-37.5%) |
| LIF | 0.1873→0.1856 ms (-0.9%) | 1.1505→0.9407 ms (-18.2%) | 0.1995→0.1846 ms (-7.5%) | 1.1000→0.9935 ms (-9.7%) |
| PLIF | 0.2968→0.1890 ms (-36.3%) | 1.7132→1.1655 ms (-32.0%) | 0.2827→0.1911 ms (-32.4%) | 1.7464→1.1666 ms (-33.2%) |
| Izhikevich | 0.3797→0.1890 ms (-50.2%) | 0.8946→1.2576 ms (+40.6%) | 3.8852→0.1925 ms (-95.0%) | 16.0955→1.2607 ms (-92.2%) |

Negative values mean lower candidate wall time. The Izhikevich T=1 training case
is a small-workload regression against the old Torch path; the baseline had no
Triton Izhikevich kernel. Longer sequences are substantially faster with the
registered Triton implementation. On the same host, a warmed CUDA provider-cache
lookup measured 0.215–0.220 microseconds; provider binding after CUDA
initialization took 4.6–6.1 ms per family, with the first IF binding taking
71.7 ms including the first Triton import. Thirty paired measurements of the
small IF inference call with the logger disabled/enabled changed the wall median
by 0.31%; selection logging occurs only during binding.

## RTX 5090 on Vast.ai

The reusable private template is `spikingjelly-dev-cuda-devel`, ID `749282`,
hash `0bb7cc6d57cf9be37826fc5f5850aaa3`. It uses
`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`, 60 GiB, SSH/direct access, and a
startup check that preserves the image's Torch/Triton stack and verifies the
CUDA and C++ toolchains. The original `spikingjelly-dev` template was unchanged.

The single-GPU RTX 5090 instance used a 60 GiB on-demand offer quoted at
`$0.4944/hour` (`$0.5726/hour` conservative adjusted rate). It ran PyTorch
`2.11.0+cu128`, CUDA/NVCC `12.8`, Triton `3.6.0`, and CuPy `14.2.0`. All nine
native extensions compiled for `sm_120` in the instance; no binary was uploaded
to PyPI.

The forced-provider correctness matrix passed for:

- Triton: IF, EIF, ActivationAwareIF
- Native CUDA: LIF, QIF, I-LIF
- CuPy: PLIF, Izhikevich, STBIF

The core migrated-neuron suite passed **637 tests**, with **16 skipped**. An
additional FlexSN, functional-neuron, and operator-migration run passed **187
tests**, with **16 skipped**. After enabling CUDA higher-order recomputation,
the final short native-build run passed the QIF, EIF, Izhikevich, I-LIF, IF, LIF,
and PLIF double-backward comparisons. Runtime diagnostics confirmed native CUDA
selection for all seven.

Three sequential on-demand instances were used: `54339736` (offer `48989721`),
`54346506` (offer `44173711`), and `54349476` (offer `45669277`), each quoted at
about `$0.494/hour`. The final Vast.ai invoice recorded **$0.441 total** across
GPU, disk, and 0.9 GB of download. All three instances were destroyed; the
account has no remaining instance or attached SSH key.

## Additional checks

- The complete local `test/activation_based` suite passed **1,487 tests**, with
  **452 skipped**.
- Ruff, `git diff --check`, Changelog generation check, and Sphinx HTML build
  passed; Sphinx reported no warnings.
- `tools/check_logging_policy.py` still reports three existing violations outside
  this migration: one direct print in distributed Vision training and logger
  imports in the two NIR exchange modules. Operator selection and fallback
  messages use `spikingjelly.logger`.

## Auto CUDA removal follow-up

The source translator and CUDA kernel-building DSL are removed. Fixed neuron
providers use the existing explicit `kernels.cuh` implementations. Fused
IF/LIF-Linear owns explicit forward, rematerialization, and backward kernels;
forward and rematerialization share charge/reset functions and compiler options
to preserve threshold decisions. Built-in surrogate formulas share one CUDA
header; custom surrogates provide their PyTorch derivative. The old
`surrogate.cuda_codes()` interface and generated-only cache/stride wrappers are
removed, as are the unused global CUDA thread/compiler/neuron bool-storage
settings. FlexSN remains the custom multi-step neuron generator; it emits Triton,
not CUDA source. Rollback is through the preceding Git revision.

Validation on the RTX 3090 passed 343 registered-neuron and
ActivationAwareIF/fused-Linear tests. After registering the explicit fused
backward entry, the final fused-Linear suite passed 38 tests, including seven
built-in surrogates, LogTailedReLU, nondefault streams, threshold boundaries,
and compiled forward/backward at a pre-bound registered tensor entry. A pure
Python wheel installed outside the source tree passed the same 38 GPU tests.
The full local suite passed 1,487 tests with 481 skipped; configuration tests
also passed (6 tests).
Wheel/sdist checks confirm explicit CUDA headers are shipped and no generated
wrapper or CUDA code-generation framework remains.

At the Auto CUDA cleanup checkpoint, public fused wrappers bound a Python
surrogate handle during eager training; fullgraph training was blocked by the existing
Python registry lock. Registered tensor entries compiled forward/backward when
the handle was bound before capture. The follow-up below removes that limitation
for built-in surrogates without a compiler monkeypatch.

A same-host ATan training microbenchmark used `[M=8, K=256]` for T=1 and
`[T=16, M=8, K=256]` for T=16, N=128, hard reset 0.2, an RTX 3090, and
PyTorch 2.7.1+cu118/CuPy 13.6.0. The exact pre-cleanup source snapshot was the
baseline. Four alternating isolated process pairs pinned the host thread to CPU
64; each case had 10 warmups and five repeats of 50 forward/backward calls.
This exploratory run preceded the final 64-bit addressing cleanup. Median
wall times across process rounds were:

| Fused operator | T | Before | After | Change |
| --- | ---: | ---: | ---: | ---: |
| IF-Linear | 1 | 1.5178 ms | 1.6006 ms | +5.45% |
| IF-Linear | 16 | 1.7577 ms | 1.7007 ms | -3.24% |
| LIF-Linear | 1 | 1.5813 ms | 1.6988 ms | +7.43% |
| LIF-Linear | 16 | 1.7284 ms | 1.7456 ms | +1.00% |

The shared host had substantial variability despite affinity: LIF T=16 process
medians ranged from 1.6871 to 2.0527 ms before and 1.6648 to 2.4334 ms after.
These are observed training-call latencies, not isolated GPU kernel durations;
they do not establish a stable performance improvement or regression. Raw
samples and the benchmark script are retained locally in
`/tmp/sj-auto-cuda-evidence-final/` and `/tmp/sj-auto-cuda-perf-pinned.py`.

Ruff, diff whitespace, and Changelog generation checks passed. The clean Sphinx
build succeeded with eight formatting warnings in other tutorial pages; the
final incremental build had no warnings. No native CUDA rebuild was performed
for this follow-up; its shared surrogate formulas are unchanged and the new
header is included in native build dependencies.

## Dispatcher audit and master synchronization

Audit baseline: `origin/master` at `c7de8326`. Single-step IF/LIF/PLIF now
normalizes its time dimension into the same functional multi-step call. Supported
FP16/BF16 inputs with FP32 state enter the registered device operator; state
dtypes outside that profile preserve their Torch-reference semantics. Reference
fallback respects strict implementation diagnostics. No BaseNode/MemoryModule
state contract was changed.

Removed three unused legacy Triton implementation modules, experimental
convenience entry points, the backend-based compile probe, self-comparison tests,
and finite-gradient-only duplicate smoke tests. The selection cache is keyed by
device; unused profile hooks were removed, fallback logging uses a minimal marker,
and unpack entry points are bound explicitly. Surrogate numbering has one source.
`SJ_USE_TRITON_OP` was retired; optional-dependency failures remain explicit.

Master commits reviewed: `c6cb8e46` and `c7de8326`. SlidingPSN's parameter-dtype
allocation was already present; its deterministic regression and the TD
chunked-gradient rounding tolerance were adopted. Precision Triton layout keys,
CUDA Graph tuning timing, and 64-bit/constexpr indexing fixes were ported.
The old CuPy PLIF hidden-storage problem is avoided by explicit initial state and
charged-voltage buffers in the new operators; a compiled CuPy gradient comparison
was added. The OCR terminal-status fix and protocol test were restored from master.
The removed kernel packages were not reintroduced, and the uncommitted migration
was not merged or rebased over master.

Final local checks passed **1,610 tests**, with **498 skipped**. A further
focused benchmark/logging check passed **81 tests** after retiring the old
registration environment switch. The GPU matrix on g1 passed **588 tests**, with
**43 skipped**, including the three compiled CuPy PLIF gradient comparisons.
g2's GPUs were busy. Only its matching CUDA 11.8 headers were copied to an
isolated g1 scratch directory for NVRTC; no shared environment or system toolkit
was changed. Native CUDA was not rebuilt for this audit.

The standard Triton LIF kernel gate passed: **+0.41%** median against its precision
kernel baseline, three independent rounds and 60 paired CUDA Graph samples,
zero unstable pairs, and 0.41% maximum repeat spread. This does not imply that
all eager host workloads improved.

The separate before/after ATan training-call comparison used the exact
pre-audit worktree snapshot, RTX 3090, T=1/16, N=512, FP32 state, three alternating
process pairs, ten warmups, and five repeats of 25 calls. It records actual
registered-entry usage: previously zero for single-step and FP16 mixed-state
calls, now one for all supported cases. FP16 T=16 median wall time changed:
IF **8.0501→1.3406 ms (-83.35%)**, LIF **9.4966→1.0121 ms (-89.34%)**, PLIF
**11.0541→1.0650 ms (-90.37%)**. The baseline executed Torch reference kernels
for those profiles; the candidate selected Triton.

Other observed wall-time changes ranged from **+0.59% to +67.99%** on this shared
host, including FP32 workloads whose underlying Triton kernels are unchanged.
These eager timings are exploratory and do not establish a universal speedup
or a stable host-overhead conclusion. All twelve cases and repeat samples are
retained in `/tmp/sj-audit-evidence/summary.json` and the associated round files;
the kernel gate measures GPU work independently of Python scheduling.

A clean pure-Python wheel/sdist build and outside-repository installation passed
mixed-input/state dtype and gradient checks plus the SlidingPSN dtype regression.
Ruff, diff whitespace, Changelog generation, and Sphinx checks passed.
Source cleanup removed **2,542 net lines** across Python/CUDA sources and five
obsolete source files; the one restored source file is master's OCR protocol
regression test.


## Fullgraph and AMP follow-up (2026-10-06)

Baseline: commit `55ef89b1`. Tests and measurements used an idle RTX 3090 on g1,
Torch 2.7.1+cu118, Triton 3.3.1, and CuPy 13.6.0. Each task used an isolated
source directory and private compiler caches; shared dependencies were retained.
No Vast.ai instance was rented: additional Vast.ai charges were **$0**. These
measurements are not RTX 5090 results.

Three changes resolve the observed capture failures:

- Fused IF/LIF-Linear passes built-in surrogate IDs and alpha directly through
  the existing forward/backward schemas. Only custom surrogates need Python
  handles. The public wrappers now support fullgraph training with the seven
  built-ins; custom surrogates retain the eager derivative path. No extra
  operator wrapper, `assume_constant_result`, or compiler monkeypatch was added.
- Torch-reference fallback diagnostics bypass capture and return before acquiring
  a lock on eager cache hits. Strict provider errors are still checked first.
  This removes graph breaks without changing eager recurrence or state dtype.
- Precision execution plans use a direct cache lookup instead of `lru_cache`.
  Dynamo previously unwrapped that decorator and repeated device-name queries
  and logging even after initialization. Device and FP8 capability checks still
  run when preparing the plan. Initialize with `prepare_model_for_precision`, or
  warm up a direct functional precision call, before fullgraph capture.

AMP keeps its existing state policy. Ordinary FP16/BF16 state uses Torch
reference arithmetic; `PrecisionConfig(mode="bf16", neuron_storage="fp32")`
explicitly selects fused FP32 neuron state. This is a different numerical policy,
not an interchangeable baseline. IF/LIF/PLIF dtype and first-order gradient
checks passed. Inductor results match directly compiled Torch reference
formulas exactly in the six low-state-dtype probes. Compared with eager, ordinary
Inductor fusion changes low-precision rounding (up to one stored-state ULP in
these probes); no bitwise eager/compile equivalence is claimed.

Unprofiled standard training runner results use ATan, T=4, B=8, three independent
alternating process pairs, one CPU thread, and medians over each run's CUDA-event
step samples. The diagnostic Spikformer-Ti cases use 64px inputs, 30 warmups and
100 samples. Spikformer-S uses 224px inputs, 50 warmups and 100 samples.

| Training case | Before (ms/step) | After (ms/step) | Change | Graph breaks |
| --- | ---: | ---: | ---: | --- |
| Spikformer-Ti, BF16 input/state, compile | 38.600 | 15.536 | -59.75% | 27 → 0 |
| Spikformer-Ti, BF16 autocast + explicit FP32 neuron state, compile | 40.809 | 14.442 | -64.61% | 28 → 0 |
| Spikformer-S, BF16 input/state, compile | 47.302 | 39.514 | -16.47% | 27 → 0 |

The explicit-state row compares the first follow-up snapshot (fallback logging
fixed, precision cache still using `lru_cache`) against the final snapshot.
The other rows compare `55ef89b1` against the fallback fix/final snapshot with
unchanged state policy. After-run median spread is 1.68%, 1.53%, and 0.59%,
respectively. FP32 Spikformer-Ti eager/compile/CUDA Graph changes were +0.53%,
+0.16%, and +0.10%; no stable FP32 improvement is claimed. BF16 eager remains
host-variable (5.51% repeat spread), while CUDA Graph changes are below 0.4%.
The Triton kernel sources and launch configurations were not changed.

The standard NSYS capture/analyzer used CUDA, NVTX, OS runtime and Python GIL
tracing. Every GPU event in the five captures was assigned to a complete step.
Profiled timings are attribution evidence, not the unprofiled latencies above:

- BF16 eager: 2,562 GPU events/step, 7.54 ms GPU busy time, 61.18 ms idle within
  the GPU span. Moving to explicit FP32 state reduces this to 591 events and
  4.65 ms busy time, at the cost of changing precision policy.
- BF16 compile before/after: 542 → 488 GPU events/step, 3.81 → 3.39 ms GPU busy
  time, 38.75 → 20.96 ms idle time. Main-thread GIL holding decreases from
  23.43 → 7.34 ms; measured GIL waiting is zero. The improvement primarily
  removes host submission overhead and graph boundaries, rather than changing
  an individual neuron kernel.

A separate fused CuPy training-call probe (ATan, M=8/K=256/N=128, T=1/16,
three alternating process pairs, seven batches of 50 calls) observed +3.3% to
+4.9% eager wall time from the scalar-metadata interface change; the first IF
candidate process was unstable. This is a compile-support change, not an eager
CuPy speedup. No further CUDA/CuPy tuning was done, as requested.

Validation: 1,524 local tests passed (509 skipped); final nearest local checks
passed 51 tests (141 skipped). GPU fused/dispatch checks passed 69 tests
(1 skipped); final fullgraph dtype/precision checks passed 9 tests; the final
precision matrix passed 93 tests (26 skipped). Scoped Ruff, formatting, logging,
Changelog generation and Sphinx checks passed. The whole-repository logging
checker still reports three pre-existing training/NIR violations outside this
change. Native CUDA and RTX 5090 were not revalidated in this follow-up.

Raw case JSON, monitors, source hashes, NSYS reports, analysis, commands and test
logs are saved on g1 in `/tmp/sj-opt-results` and the task source directories
`/home/allenyolk/CodeRepo/sj-opt-{baseline,candidate,final}-20261006`. A local copy
of the summary, source manifest and evidence archive is at
`/tmp/sj-opt-evidence-20261006/`.


## Offline implementation ranking (2026-10-06)

The existing selector now reads checked-in priorities per neuron family and exact
CUDA compute capability on first per-device binding. Production performs no
profiling, shape-dependent search, persistent user caching, or additional
steady-state logging. Unknown architectures/families retain the previous
availability order; explicit diagnostic overrides remain strict. Unsupported
profiles, explicit precision, FlexSN and CuPy fused Linear retain their own rules.
The priority table in `ops/selection.py` is the only runtime policy source.

`benchmark/benchmark_neuron_implementations.py` calibrates all nine ordinary
registered families through the real shared operator, including Torch as a
candidate. It validates all outputs and first gradients against Torch before
measuring synchronized wall time of warmed complete eager calls. The matrix uses
FP32 state, FP32/FP16/BF16 inputs, ATan where applicable (I-LIF retains its own STE),
and four T/N sizes: 1/512, 4/32768, 16/32768, and 4/2097152. Reset/trajectory
parameters come from the existing dispatch benchmark's `_arguments` helper.
These representative profiles do not establish an optimum for every surrogate,
reset setting, trajectory policy, shape, or compiled model.

Trainable families receive 2/3 forward+first-backward and 1/3 inference weight;
inference-only families use forward only. Dtypes and sizes have equal weights.
Scores are weighted geometric means of per-profile medians, not the latency of
one workload. A 5% tie band prefers Triton, native CUDA, CuPy, then Torch. The
summary requires all four candidates, matching sampling/profiles/GPU/software, consistent
operator source hashes, and at least three alternating process rounds. Only an
order reproduced in every round is published; inconclusive families retain the
previous policy. Benchmark instrumentation hashes are retained but do not block
aggregation when measurement defaults or summary code change.

A100 (g2, sm80) and RTX 3090 (g1, sm86), using Torch 2.7.1+cu118 and CUDA 11.8,
produced stable native CUDA → Triton → CuPy → Torch orders for all nine families.
This confirms the previous default on those devices; no selection-driven speedup
is claimed. Initial runs used 10 warmups and five batches of 20 calls; families
with inconsistent orders were rerun pinned to one CPU core with 50 warmups and
seven batches of 50 calls. Every candidate/round within a published family used
the same sampling configuration. The benchmark now defaults to the latter.

| GPU | Family | Native CUDA score (us) | Triton score (us) | CuPy score (us) | Torch score (us) |
| --- | --- | ---: | ---: | ---: | ---: |
| A100 | IF | 147.33 | 307.67 | 313.35 | 1055.39 |
| A100 | LIF | 158.31 | 326.41 | 328.98 | 1265.74 |
| A100 | PLIF | 206.35 | 408.66 | 423.07 | 1501.29 |
| A100 | Izhikevich | 160.15 | 293.53 | 307.22 | 1565.01 |
| RTX 3090 | IF | 140.74 | 285.28 | 288.28 | 1051.23 |
| RTX 3090 | LIF | 384.55 | 606.31 | 611.50 | 2001.76 |
| RTX 3090 | PLIF | 194.63 | 383.24 | 387.64 | 1489.68 |
| RTX 3090 | Izhikevich | 481.56 | 717.41 | 747.13 | 3150.00 |

The cached selection getter, including `tensor.device`, measured 0.394 us median
on RTX 3090 over 15 batches of 100,000 calls. Its steady-state implementation is
unchanged. This is not a before/after speedup measurement. Rankings optimize
warmed eager execution; extension loading and first forward/JIT costs are recorded
separately. Compile performance remains a separate measurement question.

Calibration found a QIF BF16 threshold case where native CUDA used float division
while Torch scalar division multiplied by an FP32 reciprocal. A voltage rounded
below 1 instead of to 1, changing a spike. Native/CuPy forward and backward now use
the same reciprocal multiplication. The four-value regression and large-input
FP32/FP16/BF16 comparisons pass. Triton equations were unchanged.

The g1/g2 task directories were made distinct after discovering their source
parent is shared NFS. The initial mixed-build g2 data was rejected. Native
extensions were rebuilt separately on each host; g2 binaries require a newer
glibc than g1. Triton caches are also host-local. Final QIF measurements use the
corrected source consistently.

Validation: local relevant suites passed 1,504 tests (510 skipped), and the
final policy/benchmark suite passed 16 tests (20 skipped on macOS). On each g-series host,
dispatch tests passed 23 tests (nine skipped); forced native CUDA/Triton/CuPy QIF
suites each passed 67 tests, and the auto-only fallback check passed separately.
The standard A100 Spikformer-Ti FP32 LIF training runner completed eager and
compile execution (T=4, B=8, 64px, 10 warmups, 20 steps); compile recorded one
graph and zero graph breaks. These are integration checks, not comparative
performance evidence. Scoped Ruff, formatting, logging, Changelog generation and
Sphinx checks pass.

RTX 5090 (sm120), using Torch 2.7.1+cu128, CUDA 12.8, Triton 3.3.1 and CuPy
14.2, completed all 108 candidate processes (nine families × four providers ×
three rounds). Every output and first-gradient comparison passed. All nine
rankings reproduced native CUDA → Triton → CuPy → Torch in all rounds; sm120
priorities are included in the runtime table. All 5090 cases used 50 warmups,
seven batches of 50 calls and one pinned CPU core.

| GPU | Family | Native CUDA score (us) | Triton score (us) | CuPy score (us) | Torch score (us) |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 5090 | IF | 195.47 | 376.42 | 383.62 | 1544.54 |
| RTX 5090 | LIF | 205.39 | 394.16 | 394.67 | 1834.50 |
| RTX 5090 | PLIF | 269.21 | 510.59 | 500.76 | 2100.22 |
| RTX 5090 | Izhikevich | 281.40 | 502.69 | 493.93 | 3107.62 |

CuPy's slightly lower aggregate PLIF/Izhikevich score is within the 5% tie band,
so the recorded order still prefers Triton. Cross-GPU absolute scores also
include each host's Python/submission cost; they are not GPU throughput rankings.

The existing devel template was kept unchanged. The original 2.11 image failed
to pull on two offers; the working instance used the 2.7.1 CUDA 12.8 devel image
with matching instance-only readiness checks, retaining image Torch/Triton/CUDA.
Native extensions were built locally for sm120. No 2.11-on-5090 claim is made.
The first 5090 attempt's summary lacked a complete local raw archive after its
watchdog teardown, so it was excluded. The final run's complete raw archive was
backed up before validation and teardown. Final sm120 dispatch checks passed 23 tests (nine skipped because the ordinary
selection was native CUDA, while those cases require forced Triton). Each forced
native CUDA/Triton/CuPy QIF suite passed 67 tests. CuPy emitted upstream
`ExternalStream` deprecation warnings. A fresh process confirmed all nine families
bind native CUDA using the final sm120 table. A separate final-policy fullgraph
forward/backward comparison passed for IF, LIF, PLIF and Izhikevich against Torch.
All 324 final per-candidate/per-round records across the three architectures are
saved locally. All four task instances (including the failed image pulls and excluded first
5090 attempt) were destroyed and verified absent from the account's instance
list. Charges listed after teardown total **$1.254**, below the $2.5 cap; no task
storage instance remains. The existing template was preserved. The ledger and
cleanup verification are saved with the evidence.


Raw samples, summaries, source hashes, commands and test logs are retained at
`.agents/artifacts/neuron-priorities-20261006` in the primary checkout, with
host-side records in `/tmp/sj-ranking-results`. Runtime priorities are reviewed
and updated with this evidence; no developer calibration runs on user startup.


## Eager call-path attribution (2026-10-07)

The offline scores measure synchronized complete eager calls, not GPU kernel
throughput. Native CUDA's large advantage is reproducible in that metric and
primarily comes from host-side implementation work. This does not invalidate the
eager result, and it does not establish the fastest provider for compile or
CUDA Graph execution.

Follow-up runs used the same isolated g2/A100 source and native binaries, Torch
2.7.1+cu118, Triton 3.3.1, CuPy 13.6.0, one CPU thread pinned to core 2, FP32 state
and inputs, ATan, hard reset, detached reset and final-state output. IF/LIF/PLIF
were compared at all four calibration sizes. Every ordinary forward/first-gradient
comparison against Torch passed. CUDA Graph replay outputs/gradients were also
checked against the ordinary call. No production checks or execution paths were
changed, and no paid instance was used.

Each GPU measurement captures 100 complete calls into a single CUDA Graph,
warms ten replays, then times 20 replays with CUDA events and divides by 100.
This removes per-call Python submission gaps while retaining the provider's
captured GPU operations. Eager CUDA-event spans alone would not remove those
gaps. Inputs, cotangents and warmups were created on the capture stream; the
initial prototype's default-stream training capture failed and was corrected.
The GPU column is graph execution per call, not a sum of profiler kernel events.

Representative T=4/N=32768 timings (us):

| Family | Provider | Eager forward | GPU graph forward | Eager forward+backward | GPU graph forward+backward |
| --- | --- | ---: | ---: | ---: | ---: |
| IF | CUDA | 29.90 | 3.40 | 259.19 | 7.39 |
| IF | Triton | 93.42 | 3.28 | 471.39 | 6.65 |
| IF | CuPy | 98.47 | 3.38 | 507.72 | 7.41 |
| LIF | CUDA | 31.21 | 3.53 | 293.64 | 7.92 |
| LIF | Triton | 95.91 | 3.50 | 504.22 | 6.64 |
| LIF | CuPy | 103.77 | 3.53 | 543.88 | 7.88 |
| PLIF | CUDA | 37.68 | 5.64 | 411.26 | 26.65 |
| PLIF | Triton | 116.58 | 5.40 | 716.56 | 24.84 |
| PLIF | CuPy | 125.07 | 5.69 | 756.86 | 25.20 |

All providers pass through the shared schema/autograd/device entry and device
cache. Native CUDA then performs validation, contiguous conversion, allocation,
stream lookup and kernel submission inside its C++ operator. Triton performs
those frontend steps in Python and enters the JIT launcher's specialization/cache
lookup and argument binding on each call. CuPy performs Python frontend steps,
current-stream wrapping, tensor-pointer/NumPy-scalar packing and RawKernel launch.
A JIT cache hit removes compilation, not all launcher work. Native still has an
additional operator dispatch; its cheaper host implementation outweighs that cost
in these eager cases. No GIL holding-time measurement is claimed.

A separate LIF forward probe (same T/N, 100 warmups, 11 batches of 100 calls)
measured the following synchronized wall times. Skipping checks and reusing
outputs/packed arguments were temporary diagnostic controls with validated fixed
inputs; they are not proposed production contracts.

| Probe (us) | CUDA | Triton | CuPy |
| --- | ---: | ---: | ---: |
| Shared production entry | 30.84 | 96.60 | 102.47 |
| Shared entry with Python GC disabled | 30.56 | 96.07 | 101.95 |
| Direct provider entry | 24.30 | 65.29 | 68.46 |
| Direct provider with Python validation skipped | — | 51.89 | 50.87 |
| Prepared outputs/arguments, raw launcher | — | 25.20 | 17.19 |

Disabling GC barely changes the result. Python validation accounts for part of
the gap, and allocations/context/argument preparation account for more; warmed
Triton launch still has substantial CPU cost. cProfile locates those functions,
but its instrumented timings overlap and distort absolute latency, so they are
not subtracted from the unprofiled numbers. The native direct entry retains its
normal C++ checks and allocations. These probe rows have different work and are
not substitutes for the shared-entry comparison.

The standard `benchmark_snn_single_gpu.py` training runner independently confirmed
an end-to-end eager benefit. Spikformer-Ti, LIF/ATan, FP32, T=4, B=8, 64px,
SGD, three alternating provider rounds, 50 warmups and 100 samples per process:

| Execution | CUDA (ms/step) | Triton (ms/step) | CuPy (ms/step) |
| --- | ---: | ---: | ---: |
| Eager | 21.268 | 28.664 | 29.610 |
| Compile | 11.594 | 9.821 | 22.161 |
| CUDA Graph | 6.531 | 6.475 | 6.537 |

Each manifest confirms the requested actual provider. All losses/outputs remained
finite; compiled runs recorded zero graph breaks. Across the three final rounds,
eager median spread was 0.68%/0.51%/1.22%; compile spread was 0.34%/2.37%/0.53%;
CUDA Graph spread was 0.03%/0.21%/0.06% for CUDA/Triton/CuPy. The earlier 20-warmup,
50-sample exploratory cohort had a substantial native eager outlier and is kept
separately, not pooled with this final cohort. Whole-model graph replay includes
all captured training work and differs from the synthetic operator graph above.

Native CUDA reduced final eager step time by 25.8% vs Triton and 28.2% vs CuPy.
Triton reduced compiled step time by 15.3% vs native CUDA; model CUDA Graph times
were within about 1%. The current offline policy therefore has evidence for its
declared eager objective, not a universal optimum across execution modes. Future
policy changes should use the intended model/execution mix rather than treating
either GPU-only or eager-only timing as universally representative.

All 72 synthetic profile records, host-control measurements, 54 model case JSON
files (27 exploratory and 27 final), commands, logs, cProfile summaries and source
hashes are saved in `.agents/artifacts/neuron-host-dispatch-20261007` in the primary
checkout. Benchmark JSON now explicitly labels its domain as
`synchronized_eager_wall_time`; its timing and ranking algorithms are unchanged.
