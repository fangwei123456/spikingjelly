# Framework test inventory

Run from the repository root in the installed project environment. Test files
are Git-tracked runners; use pytest rather than executing them directly.

## Release test sets

| ID | Set | Runner | Requirements / remaining setup |
| --- | --- | --- | --- |
| T01 | CPU correctness and minimum-version compatibility | Entire `test/` | Separate minimum and primary Python/Torch environments; use the candidate's dependency bounds |
| T02 | CUDA correctness | Entire `test/` | CUDA Torch, Triton and locally built native extensions; verify actual providers |
| T03 | Optional capabilities | Relevant modules below, in their dependency environments | NIR and language integrations require their dependencies; FP8 hardware acceptance is deferred |
| T04 | Distributed contracts | `activation_based/test_distributed_*.py`, `activation_based/test_pipeline.py` | These tests do not replace real multi-rank workflow and lifecycle acceptance |

```bash
uv run --no-sync pytest test/ -v -o faulthandler_timeout=60 \
  --junitxml=/absolute/path/to/evidence/pytest.xml
```

T01 and T02 use the same suite in different environments. Preserve skips and
warnings; a missing dependency does not validate its capability. Release profiles and skip rules are in
[tools/release/profiles.toml](../tools/release/profiles.toml); CUDA skips block
until explicitly reviewed for the selected support scope.

## Module map

Patterns below refer to files in `activation_based/`, unless a directory is given.

| Subject | Test modules |
| --- | --- |
| Neurons, operators and execution | `test_neuron_*.py`, `test_triton_neuron.py`, `test_registered_neuron_kernels.py`, `test_cuda_*.py`, `test_ops_migration.py`, `test_*node.py`, `test_ternary_neuron.py`, `test_few_spike_neuron.py`, `test_noisy_and_misc.py` |
| State, scheduling and checkpoints | [memory](activation_based/test_memory.py), `test_functional*.py`, `test_net_config.py`, `test_checkpointing.py`, `test_rnn.py` |
| Models and attention | `test_model_families.py`, `test_spikformer_model.py`, `test_encoding_attention.py`, `test_few_spike_sequence_*.py`, `test_flexsn.py` |
| Learning and monitoring | [learning](activation_based/test_learning.py), `test_online_learning.py`, `test_monitor.py` |
| ANN2SNN | `test_ann2snn*.py`, `test_fuse_conv_bn.py` |
| Precision and FP8 | `test_precision_*.py`; [optional hardware checks](activation_based/test_precision_fp8_hardware.py), excluded by release profiles |
| Distributed execution | `test_distributed_*.py`, `test_pipeline.py` |
| Energy and profiling contracts | `test_op_counter_*.py`, `test_ac.py`, `test_mac.py`, `test_synops.py`, `test_profiler.py`, `test_roofline.py` |
| Timing models and encoding | [timing_based/](timing_based/) |
| Datasets | [datasets/](datasets/) |
| Configuration, logging and plotting API | `test_configure.py`, `test_logger.py`, `test_visualizing.py` at this directory's root |

Package installation, documentation and release reporting belong in
[tools/](../tools/README.md). Performance and task-quality runners are listed in
[benchmark/](../benchmark/README.md).
