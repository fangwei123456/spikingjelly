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

Public fused wrappers still bind a Python surrogate handle during eager
training; fullgraph training through those wrappers is blocked by the existing
Python registry lock. Registered tensor entries compile forward/backward when
the handle is bound before capture. This cleanup does not add a compiler
monkeypatch or claim fullgraph training support for the public fused wrappers.

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
