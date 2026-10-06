# Benchmarks

Run commands from the repository root in the installed project environment.
Use an otherwise idle GPU and warm up before measuring.

| Purpose | Entry points |
| --- | --- |
| Full-model SNN training and inference | `benchmark_snn_single_gpu.py` |
| Neuron dispatch and provider overhead | `check_neuron_dispatch.py` |
| Triton LIF regression check | `check_triton_lif_performance.py` |
| Nsight Systems capture | `nsys_snn.sh`, `nsys_lif_example.py`, `nsys_multigpu_example.py` |
| Neuron layouts and final-state execution | `benchmark_neuron_layout.py`, `benchmark_neuron_last_state.py`, `benchmark_ilif.py` |
| Precision and memory | `benchmark_fp8_training_inference.py`, `probe_lif_fp8_triton.py`, `benchmark_same_dtype_stable_vs_mp.py`, `benchmark_train_precision_snn_fc.py`, `benchmark_memopt.py` |
| ANN-to-SNN conversion | `benchmark_ann2snn_*.py`, `snn_llm/` |
| Distributed vision and language models | `vision_distributed.py`, `vision_inference.py`, `snn_llm/` |

Neuron modules select execution from the tensor device. CPU uses the Torch
reference; CUDA uses the registered operator and caches a compatible
implementation per device. Normal model construction has no backend argument.
To inspect the selected implementation, call
`functional.neuron_implementation(neuron_type, device)`.

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

Use unprofiled runs for latency; use Nsight Systems for attribution. The full
model runner and the kernel check test different workloads and should not be
treated as interchangeable evidence. Benchmark JSON records the actual selected
implementation where applicable.
