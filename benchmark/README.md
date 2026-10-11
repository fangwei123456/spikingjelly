# Benchmarks

Run from the repository root in the installed environment, on an idle GPU.

| Purpose | Entry points |
| --- | --- |
| Full-model training and inference | `benchmark_snn_single_gpu.py` |
| Neuron implementations and layouts | `benchmark_neuron_implementations.py`, `benchmark_neuron_layout.py` |
| FP8 and mixed precision | `benchmark_fp8_training_inference.py`, `benchmark_train_precision_snn_fc.py`, `benchmark_triton_neuron_kernels.py` |
| Memory optimization | `benchmark_memopt.py` |
| Fused neuron–Linear and sparse kernels | `binary_kernel/` |
| ANN2SNN examples | `benchmark_ann2snn_spikezip_vit_qann.py` |
| Distributed vision and language models | `vision_distributed.py`, `vision_inference.py`, `snn_llm/` |
| Energy estimates versus reference models | `energy_model_validation.py` |
| Nsight Systems profiling | `tools/nsys/nsys_snn.sh`; [workloads](../spikingjelly/activation_based/examples/profiling/) |

```bash
uv run --no-sync python -m benchmark.benchmark_snn_single_gpu case \
  --model spikformer_s --phase training --execution eager --batch-size 32 \
  --T 4 --image-size 224 --warmup 50 --steps 100 --precision fp32 \
  --neuron-family lif --surrogate ATan --output /tmp/sj-benchmark/lif.json
```

Neuron execution is selected automatically. Use `SJ_LIF_CUDA_IMPLEMENTATION`
(`cuda`, `triton`, or `torch`) only for strict provider comparisons.
`--precision bf16` retains input-dtype neuron state; adding
`--neuron-storage fp32` selects a different numerical policy. Keep that policy
identical across paired runs. The layout benchmark explicitly uses FP32 state.

Use unprofiled runs for latency and NSYS for attribution. Keep cold-start and
steady-state results separate; retain raw measurements and environment metadata.

Result plotting tools: `tools/viz/`.
