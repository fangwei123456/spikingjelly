# Benchmarks

Run commands from the repository root in the installed project environment.
Use an otherwise idle GPU and warm up before measuring.

| Purpose | Entry points |
| --- | --- |
| Full-model SNN training and inference | `benchmark_snn_single_gpu.py` |
| Neuron dispatch and provider overhead | `check_neuron_dispatch.py` |
| Neuron implementation comparison | `benchmark_neuron_implementations.py` |
| Triton LIF regression check | `check_triton_lif_performance.py` |
| Nsight Systems capture | `nsys_snn.sh`, `nsys_lif_example.py`, `nsys_multigpu_example.py` |
| Neuron layouts and final-state execution | `benchmark_neuron_layout.py` |
| Precision and memory | `benchmark_fp8_training_inference.py`, `benchmark_triton_neuron_kernels.py`, `benchmark_train_precision_snn_fc.py`, `benchmark_memopt.py` |
| ANN-to-SNN conversion | `benchmark_ann2snn_*.py`, `snn_llm/` |
| Distributed vision and language models | `vision_distributed.py`, `vision_inference.py`, `snn_llm/` |

Neuron modules select execution from the tensor device. CPU uses the Torch
reference; CUDA uses the registered operator and caches a compatible
implementation per device and execution path. Eager checks native CUDA → Triton →
Torch on every architecture, skipping incompatible or missing implementations.
Runtime selection never profiles candidates. Normal model construction has no
backend argument.
Inductor expansion has a separate cache and prefers Triton → native CUDA
→ Torch. The existing eager and expansion entries choose the cache; ordinary
calls do not check `is_compiling()`. Raw CUDA Graphs retain the warmed eager
choice, while graphs captured from compiled functions retain the compiled choice.
To inspect the selected implementation, call
`functional.neuron_implementation(neuron_type, device)` for eager or add
`execution="compile"` for compiler expansion. A query reports the path's binding;
reference-only precision profiles can still follow their documented fallback.

For provider diagnostics only, set `SJ_<NEURON>_CUDA_IMPLEMENTATION` before
starting Python. The default is `auto`; a value such as `triton` is strict and
fails if that implementation is unavailable. Selection is logged once through
the SpikingJelly logger. For example:

```bash
SJ_LIF_CUDA_IMPLEMENTATION=triton uv run --no-sync python -m benchmark.benchmark_snn_single_gpu case \
  --model spikformer_s --phase training --execution eager --batch-size 32 \
  --T 4 --image-size 224 --warmup 50 --steps 100 --precision fp32 \
  --neuron-family lif --surrogate ATan \
  --output /tmp/sj-benchmark/lif-triton.json
```

The standard Triton LIF check compares the production Triton provider against
the candidate backward path:

```bash
SJ_LIF_CUDA_IMPLEMENTATION=triton uv run --no-sync python -m benchmark.check_triton_lif_performance \
  --device cuda:0 --output /tmp/sj-benchmark/triton-lif.json
```

For autocast training, ``--precision bf16`` retains input-dtype neuron state.
That profile uses the Torch reference recurrence, including under fullgraph
compilation. ``--precision bf16 --neuron-storage fp32`` explicitly selects FP32
neuron state and fused precision kernels. These are different numerical policies;
compare revisions with the same policy, and validate model accuracy when changing
it. A device's selected provider does not imply that every dtype profile uses it.

Use unprofiled runs for latency; use Nsight Systems for attribution. The full
model runner and the kernel check test different workloads and should not be
treated as interchangeable evidence. Benchmark JSON records the actual selected
implementation where applicable.

To recalibrate priorities, run each of the nine families with each of `cuda`,
`triton`, and `torch` in three separate process rounds. Alternate candidate
order between rounds, pin the same idle CPU core with `taskset`, and use an idle
GPU. Each invocation validates outputs and first gradients against Torch before
measuring warmed, synchronized complete eager calls:

```bash
taskset -c 2 uv run --no-sync python -m benchmark.benchmark_neuron_implementations \
  --neuron lif --implementation triton --round 0 --device cuda:0 \
  --output /tmp/sj-calibration/lif-triton-r0.json
uv run --no-sync python -m benchmark.benchmark_neuron_implementations \
  --summarize /tmp/sj-calibration --output /tmp/sj-priorities.json
```

The default matrix uses FP32 state, FP32/FP16/BF16 inputs, ATan where applicable,
and four T/N sizes: 1/512, 4/32768, 16/32768, and 4/2097152. Training receives
two-thirds of the score, inference one-third; inference-only families use forward
only. Scores are weighted geometric means with equal dtype/size weights. Within
5%, prefer Triton, then native CUDA and Torch. Publish a priority only when
all three rounds give the same order. An inconclusive result leaves the previous
order in place. Preserve raw samples, environment and source hashes with the
summary. This measures steady-state eager execution; first-load/JIT costs are
recorded separately, and compile workloads require separate validation.

These scores include Python validation, tensor allocation, parameter handling,
kernel submission, and autograd. They compare the installed implementation's
complete eager path, not GPU kernels alone. To isolate GPU execution, capture
many complete calls in one CUDA Graph and time its replay with CUDA events;
timing an eager loop with CUDA events can still include gaps while the CPU
submits work. Compare eager, compile, and CUDA Graph model runs separately.

The neuron layout benchmark uses the functional LIF interface with FP32 initial
state and FP16 input, and checks that native CUDA or Triton is selected. Module
FP16 state would select the reference recurrence and cannot measure this kernel
layout contract. Set `SJ_LIF_CUDA_IMPLEMENTATION=cuda` or `triton` before starting
`python -m benchmark.benchmark_neuron_layout`; compare revisions with the same
benchmark and state policy.
