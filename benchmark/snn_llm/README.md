# SNN-LLM Benchmarks

## Qwen2

The `qwen_conversion/` directory contains the
public-recipe correctness, quality, efficiency, and tensor-parallel runners.
See its README and the ANN2SNN Transformer tutorial for commands and
interpretation.

For pretrained Qwen2.5-0.5B at T=32, the collapsed-QCFS SGLang adapter matched
generated tokens across TP1, TP2, and PP2. Its KV pool held 32 times as many
token slots; three paired whole-batch workloads improved by 1.9%, 11.1%, and
14.0%, without a measured whole-GPU peak-memory reduction. The recipe's
zero-tail attention path matched 16×256 generated tokens and reduced full-run
time by 1.92–2.41× in three paired runs. Neither result applies to other spike
encodings.

## Outputs

Keep generated reports, checkpoints and traces under ignored `benchmark/output/`.
Historical SpikeGPT and T=8 phase-cache experiments remain recoverable from Git
at `a230b4d053a600f4bca4ac679080eda31e8b0ee0`; their raw artifacts are retained.
