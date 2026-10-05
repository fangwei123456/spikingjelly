# Benchmarks

Run commands from the repository root in the installed project environment.
Keep reusable workloads and their tests here; keep historical measurements and
source snapshots outside this directory. GPU timings require an otherwise idle
device and representative warmup.

| Purpose | Entry points |
| --- | --- |
| Full-model SNN training/inference | `benchmark_snn_single_gpu.py` (`case`, `matrix`) |
| CPU/CUDA dispatch and cached CUDA provider overhead | `check_neuron_dispatch.py` |
| Registered vs production Triton LIF backward | `check_triton_lif_performance.py` |
| Nsight Systems capture and analysis | `nsys_snn.sh`, `nsys_lif_example.py`, `nsys_multigpu_example.py`; [profiling guide](../docs/source/APIs/spikingjelly.nsys.rst) |
| Neuron kernels, layouts and final-state execution | `benchmark_triton_neuron_kernels.py`, `benchmark_neuron_layout.py`, `benchmark_triton_last_state.py`, `benchmark_ilif.py` |
| FlexSN and compile boundaries | `flexsn/`, `probe_snn_compile_boundary.py` |
| Precision and memory | `benchmark_fp8_training_inference.py`, `probe_lif_fp8_triton.py`, `benchmark_same_dtype_stable_vs_mp.py`, `benchmark_train_precision_snn_fc.py`, `benchmark_memopt.py` |
| ANN-to-SNN conversion | `benchmark_ann2snn_*.py` |
| Distributed vision and language models | `vision_distributed.py`, `vision_inference.py`, [snn_llm/](snn_llm/README.md), `plot_distributed_*.py`, `plot_sglang_inference.py` |
| Logging, energy accounting and binary kernels | `benchmark_logging.py`, `energy_model_validation.py`, `binary_kernel/` |

Shared analysis helpers and CPU-testable checks live alongside these entry points
and in `test/`.

## Experimental IF / LIF / PLIF

Implementations live in `ops/`, installed as `spikingjelly._ops`; correctness tests
live in `test/activation_based/test_experimental_*.py`. Installation and API
constraints are in the [experimental neuron guide](../docs/source/APIs/spikingjelly.activation_based.neuron.experimental.rst).

Use the full-model runner for production/experimental comparisons. For example:

```bash
SJ_LIF_CUDA_IMPLEMENTATION=triton uv run --no-sync python -m benchmark.benchmark_snn_single_gpu case \
  --model spikformer_s --phase training --execution eager --batch-size 32 \
  --T 4 --image-size 224 --warmup 50 --steps 100 --precision fp32 \
  --neuron-backend triton --neuron-family lif --surrogate ATan \
  --experimental-neurons --output /tmp/sj-benchmark/lif-triton.json
```

Omit `--experimental-neurons` for the production baseline. Use `--neuron-family`
and the corresponding `SJ_IF_CUDA_IMPLEMENTATION`, `SJ_LIF_CUDA_IMPLEMENTATION`,
or `SJ_PLIF_CUDA_IMPLEMENTATION` for other experimental cases. Compare independent
rounds in alternating order. FP16/BF16 state rounding differs between production
and experimental neurons; throughput measurements do not establish convergence.

Capture a separate diagnostic run:

```bash
SJ_LIF_CUDA_IMPLEMENTATION=triton uv run --no-sync bash benchmark/nsys_snn.sh capture /tmp/sj-benchmark/lif-profile -- \
  python -m benchmark.benchmark_snn_single_gpu case \
  --model spikformer_s --phase training --execution eager --batch-size 32 \
  --T 4 --image-size 224 --warmup 50 --steps 10 --precision fp32 \
  --neuron-backend triton --neuron-family lif --surrogate ATan \
  --experimental-neurons --profile --output /tmp/sj-benchmark/lif-profile.json
uv run --no-sync bash benchmark/nsys_snn.sh analyze \
  /tmp/sj-benchmark/lif-profile.nsys-rep /tmp/sj-benchmark/lif-analysis /tmp/sj-benchmark/lif-profile.json
```

Use unprofiled runs for training latency and NSYS for attribution. The retired
`native_lif/` migration runners and fixed ConvNet/handwritten-LIF probes are no
longer maintained. CUDA/CuPy latency relative to Triton is not an acceptance gate.

## Triton LIF performance check

```bash
uv run --no-sync python -m benchmark.check_triton_lif_performance \
  --device cuda:0 --output /tmp/sj-benchmark/triton-lif.json
```

This is the FP32 LIF backward check used for the registered-operator migration.
It requires NVIDIA CUDA, Triton and PyTorch's native CUDA allocator; neither CuPy
nor the native extension is needed. It calls the Triton backward implementation function
directly, irrespective of `SJ_LIF_CUDA_IMPLEMENTATION`.

Defaults are Spikformer-S, B32/T4/224, ATan (default alpha=2), 50 training warmups,
three independent processes and 20 ABBA/BAAB pairs per process. Each event sample
replays the complete captured backward chain five times. The two implementations
receive identical contiguous FP32 inputs and write the same output addresses.
Both captured graphs must overwrite NaN-poisoned buffers and match production
gradients using magnitude-scaled tolerances before timing begins.

The default maximum regression is 2%. Each round uses the median of its paired
candidate/baseline ratios. All rounds within the limit mean `pass` (exit 0); all
above mean `fail` (exit 1); disagreement means `inconclusive` (exit 2). If the
two timings for either implementation within a pair differ by more than the
same percentage limit (relative to their mean), the run is also `inconclusive`,
even when the median ratio is good. Inspect the raw samples and rerun on an idle
device. Execution
or correctness errors also exit 2 and retain the failing worker's log. This
checks closeness to production Triton, not exact equality or superiority.

The output JSON contains the verdict, raw timings, numerical checks, captured
shapes/strides/parameters, source hashes and runtime versions. Per-round JSON and
logs are stored beside it. Use an idle GPU; `CUDA_VISIBLE_DEVICES` can bind a GPU
UUID and `--require-gpu-name A100` can reject an unintended model. Clocks are not
locked automatically.

`--surrogate`, `--batch-size`, `--T`, `--image-size`, `--warmup-steps`, `--rounds`,
`--pairs`, `--replays` and `--max-regression-percent` are configurable. Smaller
workloads and fewer rounds are useful for smoke checks but do not certify the
default workload. This script does not cover IF/PLIF or low-precision state
policies, and its graph timings are not full training-step or NSYS timings.

## Neuron dispatch overhead

```bash
uv run --no-sync python -m benchmark.check_neuron_dispatch \
  --neuron lif --implementation triton --output /tmp/sj-benchmark/dispatch.json
```

The default is three independent processes, with 12 balanced paired measurements
per workload. Each pair runs all routes in a randomized order and then reverses
that order. `direct` calls the provider-style operator; `cached` reconstructs
the earlier per-device Python selection route on the same kernels; `unified`
calls the shared CPU/CUDA dispatcher entry. It compares forward, forward/backward,
and fullgraph-compiled forward/backward, checking values and gradients first.
CPU/Triton provider registrations have been removed from the library. For Triton,
this script recreates the historical wrapper in a temporary `sj_benchmark_*`
namespace solely as a timing baseline; the public library never loads it.
Native CUDA/CuPy use their retained provider operators for this comparison.

Wall timings include host launch overhead and synchronize at each sample's
boundaries. GPU-only forward timings replay graphs with 32 kernel calls per
replay to avoid measuring a host-starved stream. The cached-selection microbench
includes Tensor.device access. All values are microseconds per logical operation.
JSON retains raw paired observations, within-round bootstrap intervals and
per-process results. Intervals describe paired sample uncertainty, not a guarantee
across machines; inspect all rounds before claiming a regression or improvement.
GPU graph variants allocate their own output buffers. For tiny multi-kernel
workloads, results can change direction between processes; such observations are
inconclusive and should not be attributed to kernel arithmetic or selection.

The runner requires `nvidia-smi` and checks compute-process ownership on the target
GPU before and after each timed block. Another process on that GPU rejects the
round; discard it and rerun on an idle device. These boundary checks are outside
timing and do not reserve the GPU or detect every short-lived overlapping job.

Use `--neuron if|plif|izhikevich` and repeat `--shape T,N` to change workloads.
Izhikevich includes both membrane and recovery state outputs and initial-state
gradients. Each process uses seed 42 and random output cotangents, verifies finite
eager results and gradients, then checks compiled outputs and gradients before
timing. JSON records the settings, source hashes, Python/Torch/Triton versions,
CUDA version and GPU capability. Use identical settings and an idle GPU when
comparing reports. This isolates
dispatch on FP32/ATan inputs; it is not a model training throughput measurement.
Use the full-model runner above for Spikformer training evidence. Native CUDA
still crosses its registered C++ operator; no claim of equal dispatch overhead
across providers is implied.
