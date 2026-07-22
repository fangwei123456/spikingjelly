# GPT-2 Conversion Checks

These private runners provide small causal-language-model checks for the
SpikingJelly ANN2SNN Transformer components. They load a pinned local GPT-2
snapshot only; no command downloads models or data.

## Dense Baseline

```bash
PYTHONPATH="$PWD" python benchmark/snn_llm/gpt2_conversion/dense_baseline.py \
  --model-root benchmark/output/gpt2-607a30d \
  --output-dir benchmark/output/snn-llm/gpt2/dense \
  --device cpu \
  --source-revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --max-samples 4 --max-length 64
```

## MLP ANN2SNN

```bash
PYTHONPATH="$PWD" python benchmark/snn_llm/gpt2_conversion/mlp_ann2snn_slice.py \
  --model-root benchmark/output/gpt2-607a30d \
  --output-dir benchmark/output/snn-llm/gpt2/mlp \
  --device cuda \
  --source-revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --block-index 0 --time-steps 16 --max-samples 4
```

## Cache Contract

```bash
PYTHONPATH="$PWD" python benchmark/snn_llm/gpt2_conversion/cache_equivalence.py \
  --model-root benchmark/output/gpt2-607a30d \
  --output-dir benchmark/output/snn-llm/gpt2/cache \
  --device cuda \
  --source-revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --max-samples 4 --max-length 64 --prefill-length 16 --decode-steps 8
```

Each runner atomically writes a non-overwriting `report.json` under its selected
output directory. Generated reports and model files remain in the ignored
`benchmark/output/` tree. These checks do not fine-tune GPT-2, run full
perplexity or generation-quality evaluation, or claim full-model ANN2SNN
conversion.
