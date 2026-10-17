# Benchmark inventory

Run from the repository root in the installed environment. Entry points below
are Git-tracked. Scripts exist; vision quality/performance orchestration is in [tools/release/](../tools/release/).
Previous-release API adapters remain pending; FP8 release projects are deferred.
Qwen quick conversion/PPL is integrated in the release profiles.

Use the [release benchmark report template](../tools/release/benchmark-report-template.md)
to summarize selected tasks, quality, performance, evidence and limitations for
each formal release. It reuses runner outputs and does not add mandatory tasks.

## Core task benchmarks

Select release cases from real CIFAR-scale or larger tasks, or language tasks.
Small operator/toy measurements alone cannot close full-model acceptance.
The release owner selects tasks and budgets; these are not mandatory full
training runs. For vision, include CNN-SNN and SNN Transformer workloads where
supported. Lock real data, architecture, time steps, weights, precision and
training/evaluation budget before comparing releases.

| ID | Task and representative model | Git-tracked runners | Readiness |
| --- | --- | --- | --- |
| B01 | CIFAR-100, 32×32: MaxFormer-4-384, T4, 6.60M | [benchmark_image_classification.py](benchmark_image_classification.py), `--case maxformer-cifar100` | Calls `model.maxformer_cifar`; CPU/CUDA source parity and bounded CLI passed; full training/comparison pending |
| B02 | CIFAR-100, 32×32: MaxResNet-18, T4, 12.55M | Same runner, `--case maxresnet-cifar100` | Calls existing `model.max_resnet18`; CPU/CUDA source parity and bounded CLI passed; full training/comparison pending |
| B03 | Tiny ImageNet, 64×64: CLIF VGG13, T6, 14.43M | Same runner, `--case clif-tinyimagenet` | Existing CLIF/layers; official single-head/average-pooling topology; CPU/CUDA source parity and bounded CLI passed; full training/comparison pending |
| B04 | Language: Qwen2.5 Base 0.5B/1.5B/3B conversion, WikiText PPL and zero-shot tasks | [quality_eval.py](snn_llm/qwen_conversion/quality_eval.py), [quality_aggregate.py](snn_llm/qwen_conversion/quality_aggregate.py), [scaleout_efficiency.py](snn_llm/qwen_conversion/scaleout_efficiency.py), [sglang_benchmark.py](snn_llm/sglang_benchmark.py) | Release quick profile: pinned 0.5B conversion/cache/reset + four-window WikiText PPL, exercised on g3; larger/full runners remain available; conversion/inference, not native SNN training |

For each selected task, report quality (top1/top5, PPL or task scores), training
curves where applicable, training/inference throughput and peak memory. Separate
GPU execution timing from end-to-end data loading. Use the same trained weights
and real inputs for paired performance runs. A fixed-checkpoint full evaluation
can validate task quality; a bounded training run validates only its stated
training budget, not convergence or a published full-training accuracy.

The previous formal release remains the only version baseline. Quality tolerances
and the 2% efficiency gate are separate. New capabilities without a previous
counterpart need an explicit N/A decision and absolute results.

Release quality benchmarks detect drift against our own version baseline;
matching a paper's reported accuracy is not an acceptance requirement. Papers
remain references for model/recipe implementation. Separate fixed-checkpoint
evaluation drift from retraining variability: lock data, preprocessing, weights,
precision and environment for the former; use matched seeds and training budgets
for the latter, adding paired seeds when needed to resolve training noise.
Report accuracy changes in percentage points and predeclare task-specific
tolerances. Stable accuracy does not replace numerical correctness checks.

[benchmark_snn_single_gpu.py](benchmark_snn_single_gpu.py) remains useful for
controlled full-model timing, but its random-input results do not establish real
task performance. Its historical `performance_gates.met` is not the release's
2% gate. Real-data/checkpoint pairing and new gate evaluation remain to be added.

## Vision task runner

One standalone file using installed SpikingJelly, Torch, torchvision, NumPy and
Pillow. No timm or author checkout is needed. MaxFormer/MaxResNet models come from
the package. CLIF VGG13 is assembled with existing layers and
`ComplementaryLIFNode`; the package's ImageNet VGG has a different head/pooling.

References: [MaxFormer, NeurIPS 2025](https://github.com/bic-L/MaxFormer/tree/6ddd4ccde2cc9c4b965059403868e57803b9b6b0/cifar10-100)
and [CLIF, ICML 2024](https://proceedings.mlr.press/v235/huang24n.html),
[CLIF source](https://github.com/HuuYuLong/Complementary-LIF/tree/d76af5918b5797be367150d6761d10946dcab30b).

- Max: official CIFAR100 YAML, batch64/T4, AdamW .0015/decay .06, warmup20,
  cosine400 plus cooldown10 (410 scheduled epochs), AMP, smoothing .1,
  no weight decay on BN/bias,
  Mixup .75/CutMix .5 through epoch249. Raw-pixel mixing is rounded before
  normalization, as in source FastCollateMixup. RandAugment uses the author's
  five-operation subset, not torchvision's general RandAugment. Prefetching is
  simplified; no data loading/augmentation is included in GPU timing.
- CLIF `--recipe paper` (default): Appendix I batch256/T6/300 epochs, SGD .1,
  momentum .9/decay .0005, cosine300, FP16 AMP training/FP32 evaluation,
  ImageNet AutoAugment, seed2022.
  `--recipe source`: README defaults batch128 without AutoAugment. Both retain
  `.95*CE(all time-step logits)+.05*MSE(logits, ones)`, per-step BN and reset
  after each batch. A small surrogate subclass preserves the author's inclusive
  gradient endpoints; the paper and standard SJ Rect use strict endpoints.
  CLIF arithmetic is numerically checked, not claimed bitwise identical.

```bash
uv run --no-sync python benchmark/benchmark_image_classification.py \
  --case maxformer-cifar100 --data-root /data/cifar100 --download \
  --output-dir benchmark/output/maxformer-cifar100
uv run --no-sync python benchmark/benchmark_image_classification.py \
  --case maxresnet-cifar100 --data-root /data/cifar100 \
  --output-dir benchmark/output/maxresnet-cifar100
uv run --no-sync python benchmark/benchmark_image_classification.py \
  --case clif-tinyimagenet --data-root /data/tiny-imagenet-200 \
  --output-dir benchmark/output/clif-tiny
```

Extract Tiny ImageNet first. Raw `val/images/` plus `val_annotations.txt` and
class-folder validation are accepted against the training class map.
Use a new output directory for each attempt. `--checkpoint` resumes training;
`--mode evaluate` evaluates weights. `--mode benchmark --checkpoint PATH`
with `--phase inference` or `training` measures a fixed real evaluation batch,
including model/reset and, for training, optimizer work. Correctness checks run
outside timing. Training timing is a fixed-batch diagnostic; task convergence
comes from training/evaluation. Repeat paired performance processes three times.

Retain `manifest.json`, `report.json`, `epochs.jsonl` and last/best checkpoints.
Timing also stores `correctness.pt`: full-batch outputs/input/parameter gradients
and two samples of final neuron states, plus real-input and packet SHA256.
CLIF correctness packets use a deterministic window-mean reference for the
2×2-to-7×7 adaptive pooling backward: the CUDA operator uses atomic additions,
so even repeated Torch runs can miss strict gradient comparisons. The reference
checks its forward output bitwise against the original pooling. Timed and
quality runs keep the original model; the packet oracle is recorded and must
match between compared reports. CLIF uses the standard SJ `Rect(alpha=1.0)`
surrogate, whose gradient window excludes endpoints. Older benchmark runs used
an author-specific inclusive window; do not silently mix their gradient packets
or training histories with this configuration. Auto mode can use fused Triton
for both training and inference. CLIF reports record provider requests, selection
and correctness/timed state dtypes.
Inference packets include the evaluated output. GPU UUID/driver are recorded;
CUDA runs need `nvidia-smi`.
Bounded runs, recipe overrides and AMP-skipped updates are recorded. A scale
change during measured steps invalidates that timing run. A shorter `--epochs` stops the
original schedule early; it does not compress the schedule into fewer epochs.
For CPU correctness checks use `--device cpu --no-amp`. No ImageNet retraining is
part of this release menu; optional fixed-weight SpikeZIP evaluation remains.

## Diagnostics and conditional measurements

Choose these when affected by changes or needed to substantiate a release claim.

| Subject | Runners | Measurements |
| --- | --- | --- |
| Neuron implementations/layouts/kernels | [benchmark_neuron_implementations.py](benchmark_neuron_implementations.py), [benchmark_neuron_layout.py](benchmark_neuron_layout.py), [benchmark_triton_neuron_kernels.py](benchmark_triton_neuron_kernels.py) | Layout/kernel latency; layout runner uses FP32 state |
| FP8 GEMM and FC-SNN precision | [benchmark_fp8_training_inference.py](benchmark_fp8_training_inference.py), [benchmark_train_precision_snn_fc.py](benchmark_train_precision_snn_fc.py) | Optional precision diagnostics; FP8 is outside the current pre-release acceptance scope |
| Memory optimization | [benchmark_memopt.py](benchmark_memopt.py) | Training step latency and peak memory |
| Fused and sparse operators | [binary_kernel/benchmark_neuron_linear.py](binary_kernel/benchmark_neuron_linear.py), [binary_kernel/benchmark_sparse_kernel_only.py](binary_kernel/benchmark_sparse_kernel_only.py) | Operator latency and speedup |
| Distributed vision | [vision_distributed.py](vision_distributed.py), [vision_inference.py](vision_inference.py) | Training/inference throughput and memory; real-data mode also reports task quality |
| Distributed language | [qwen_distributed_inference.py](snn_llm/qwen_distributed_inference.py), [sglang_benchmark.py](snn_llm/sglang_benchmark.py), [scaleout_efficiency.py](snn_llm/qwen_conversion/scaleout_efficiency.py) | Inference timing, throughput and memory |
| SpikeZIP ViT conversion | [benchmark_ann2snn_spikezip_vit_qann.py](benchmark_ann2snn_spikezip_vit_qann.py) | Optional real-data evaluation with fixed pretrained weights; no ImageNet retraining |
| Tempotron | [benchmark_tempotron.py](benchmark_tempotron.py) | Event-model timing and memory; optional, no independent release performance/retraining gate |
| Energy estimates | [energy_model_validation.py](energy_model_validation.py) | Agreement with reference energy models; not measured GPU energy |

## Supporting workflow checks

| Task | Runners | Purpose |
| --- | --- | --- |
| Qwen conversion correctness | [scaleout_smoke.py](snn_llm/qwen_conversion/scaleout_smoke.py) | Output, generation, cache and reset comparisons |
| MCore workflow | [mcore_smoke.py](snn_llm/mcore_smoke.py) | Synthetic forward/backward workflow check, not training convergence |
| SpikeLM training | [train_spikelm.py](snn_llm/train_spikelm.py) | Real training entry; release quality protocol/baseline not yet fixed |

Model definitions, export adapters, `snn_llm/cli.py`, `snn_llm/inference.py`,
`fp8_coverage.py` and `fp8_efficiency.py` support these workflows; they are not
additional release cases. See [SNN-LLM](snn_llm/README.md) and
[Qwen runner commands](snn_llm/qwen_conversion/README.md).

Use an idle GPU and unprofiled runs for performance; preserve raw samples,
environment and source revisions. Profiling and plotting tools are listed in
[tools/](../tools/README.md); framework correctness tests in
[test/](../test/README.md).
