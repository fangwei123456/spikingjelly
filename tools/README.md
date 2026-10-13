# Validation and analysis tools

Run in a uv-managed project environment. Release entry points live in
[release/](release/); they never publish packages, create tags or push Git changes.

## Release validation

| Entry | Purpose | Status |
| --- | --- | --- |
| [validate.py](release/validate.py) + [profiles.toml](release/profiles.toml) | Execute profiles; retain commands/logs/JUnit; enforce time budget; alternate three candidate/baseline benchmark pairs; aggregate attempts | Implemented; CI self-check covers startup failure, timeout and skip gating; formal runs require Linux, clean tracked source/config and fixed inputs |
| [package_check.py](release/package_check.py) | Distribution contents; import provenance; installed IF/LIF/PLIF training, gradients, reset and checkpoint; native prerequisites/provider checks | CI uses the same artifact/import checks; wheel/sdist outside-source checks exercised |
| [evidence.py](release/evidence.py) | Numerical/performance comparison, Markdown local paths, JSON/Markdown report, source/evidence ZIP and independent integrity verification | Implemented; self-check also covers archive round-trip and tampering |

### Seven-step readiness

Code readiness is separate from acceptance of the final release commit.

| Step | Runnable code | Remaining code/configuration or acceptance |
| --- | --- | --- |
| 1. Fix inputs | `validate.py`, `profiles.toml`: source/config hashes, environment, budget | Select previous formal release/artifacts; finalize supported extras/topologies and skip rules |
| 2. Build/install | `package_check.py`, `validate.py`: wheel/sdist, isolated installation, native build/load | Run positive native checks on Vast and the minimum-version environment |
| 3. Framework correctness | `test/`, CPU/minimum/CUDA profiles; JUnit and skip gating | Final-candidate Linux CPU/CUDA runs and declared provider/extras matrix |
| 4. Task quality | Visual classification runner and three quality profiles; Qwen quick profile | Select quality budget/threshold; run real CIFAR/Tiny validation or training; Qwen quick has development evidence |
| 5. Performance/resources | `validate.py paired`, `evidence.py compare`; visual correctness packets and three paired rounds | Previous-release API adapter; language performance adapter/profile if selected; formal Vast measurements |
| 6. Documentation/compatibility | Format/Ruff, changelog, logging policy, Sphinx, local Markdown paths | External links/anchors and examples lack a unified entry; bilingual/migration review stays manual |
| 7. Signoff/archive | JSON/Markdown aggregation, `archive`, `verify-archive` | Human signoff and actual clean freeze; durable storage/backup location |

### Organization

```text
tools/release/          validate.py, profiles.toml, package_check.py, evidence.py
tools/                 generate_changelog_rst.py, check_logging_policy.py
tools/nsys/            capture, analysis and plotting tools
tools/viz/             plots of saved benchmark reports
test/                  activation_based/, timing_based/, datasets/, shared tests
benchmark/             benchmark_image_classification.py
benchmark/snn_llm/
  qwen_conversion/     quality_eval.py, quality_aggregate.py, scaleout_smoke.py,
                       scaleout_efficiency.py, artifacts.json, runtime/quality helpers
.github/workflows/     ci.yml calls tracked checks; changelog.yml
.agents/               scope/process, reports and development evidence (Git-ignored)
```

Other benchmark scripts remain available for optional diagnosis; they are not
automatically part of the mandatory release set.

Profiles: `cpu`, `cpu-minimum` (Python3.11/Torch2.6), `cuda`, `package`, `native`,
`docs`, `links`, `maxformer-quality`, `maxresnet-quality`, `clif-quality`,
`qwen`. The Qwen quick profile uses the existing conversion/PPL runners; FP8
is deferred from release acceptance by the maintainer (2026-10-13). CUDA skips require review and an explicit profile
allow-list; missing providers do not pass. Framework tests remain in `test/`.

```bash
uv run --no-sync python tools/release/validate.py run \
  --cases package cpu docs links --output-dir /evidence/run-001
uv run --no-sync python tools/release/validate.py report \
  --runs /evidence/run-001 /evidence/cuda-001 /evidence/perf-001 \
  --output-dir /evidence/summary-001
uv run --no-sync python tools/release/evidence.py archive \
  --repo . --ref FINAL_COMMIT --run /evidence/summary-001 \
  --output-dir /archives/SJ-VERSION
uv run --no-sync python tools/release/evidence.py verify-archive \
  --archive-dir /archives/SJ-VERSION
```

Archive verification works after relocation without the original source/evidence
directories. Keep `archive.json` independently: hashes detect corruption against
that record, not a replacement of the entire archive and record. Verification
does not establish release signoff or backup durability.

Qwen quick conversion/evaluation:

```bash
uv run --no-sync python tools/release/validate.py run --cases qwen \
  --qwen-model-root /models/qwen2.5-0.5b-060db649 \
  --qwen-hf-cache /data/wikitext-cache --output-dir /evidence/qwen-001
```

The fixed profile is Qwen2.5 Base 0.5B, BF16, T160, 16 calibration levels,
quantile .999; 128×512 training tokens for calibration, four WikiText test
windows (2047 scored tokens) for quick PPL. The pinned local model is checked by
`hf cache verify`; provision/cache models and data before running. Evaluation is
offline. Conversion/cache/reset checks reuse that calibration. Existing tolerances
and <=15% PPL degradation apply; this is quick acceptance, not full PPL,
six-task zero-shot, native SNN training or a speed claim. Model/data revisions,
actual dataset fingerprints/token hashes and calibration are retained. Run in
the pinned qwen environment (`transformers==5.13.0`, `datasets>=5,<6`).

Release profiles deselect `fp8_hardware`: the dedicated hardware file and FP8
device numerical/training cases in the neuron suite. Normal precision/API
regressions stay enabled. Plain `pytest test/` still includes all FP8 tests.

Choose quality threshold and training budget before running. Quality profiles
require `--data-root`, `--minimum-top1` and either `--checkpoint` (evaluation) or
`--vision-mode train`; `--epochs` selects an explicit training budget. Full real
validation data is required. A short training budget does not establish paper
convergence. ImageNet retraining and a web dashboard are outside this workflow.

`paired` requires `--candidate-python`, `--baseline-python`, both fixed
`--candidate-wheel`/`--baseline-wheel`, `--baseline-ref`, `--baseline-version`,
`--task`, `--data-root`, `--checkpoint`, and an output directory. The release owner
must select the previous formal release. The tool checks installed wheel bytes,
candidate source, versions, hardware UUID/driver, weights, actual inputs,
precision, output/input/parameter gradients and representative neuron states
before comparing raw wall timing. All three throughput ratios must be >=0.98;
all below fail; mixed results block. New APIs absent from the previous release
need a tracked adapter or an explicit reviewed N/A decision; they never pass
by fallback. Qwen speed report formats are not adapted to the visual 2% comparator; Qwen
quick conversion/quality is a separate release group.

Defaults: <=24h per invocation, <=$5 reported cumulative cost. Timeouts terminate
this run's process group. `--cost-usd` records operator-supplied cost; there is no
GPU rental or live billing integration. Later invocations use `--campaign-manifest FIRST_RUN/manifest.json` to retain
the original deadline, and report cumulative cost. Aggregation also rejects a
combined campaign exceeding 24h/$5, even if each individual run was shorter. Use an idle Vast on-demand GPU for formal
performance/native compilation; shared g-series runs only establish functional
or task-quality evidence.

Fresh package environments are outside the checkout. Dependency versions are
pinned from the invoking environment; `--constraints` adds reviewed constraints,
and `--torch-index` selects the official index for that existing Torch build.
Native builds use the isolated environment's Torch, Toolkit/compiler and sdist.
Installation environments are retained and recorded, not deleted automatically.

`--development` permits dirty/non-Linux checks, marks them ineligible for release,
and allows `--reuse-dependencies` for quick installed-code checks. It cannot close
formal release groups. Every run needs a fresh output directory. Aggregate only
matching candidate/profile revisions and predeclared inputs; latest matching
attempts determine status, while failed attempts remain in history and archives.
Human scope, support-matrix, migration/doc review and release signoff remain
required even when automated groups pass. Durable storage/backup is an owner
operation; ZIP creation alone does not provide it.

## Existing diagnostics

| Purpose | Entry |
| --- | --- |
| Changelog consistency | [generate_changelog_rst.py](generate_changelog_rst.py) |
| Production logging policy | [check_logging_policy.py](check_logging_policy.py) |
| NSYS capture/analysis/comparison | [nsys/nsys_snn.sh](nsys/nsys_snn.sh), [analyzer](nsys/analyze_nsys_snn.py), [plots](nsys/plot_nsys_snn.py) |
| Saved distributed reports | [inference plots](viz/plot_distributed_inference.py), [tradeoff plots](viz/plot_distributed_tradeoff.py), [SGLang plots](viz/plot_sglang_inference.py) |

NSYS workloads: [examples/profiling/](../spikingjelly/activation_based/examples/profiling/).
[Tests](../test/README.md) and [benchmarks](../benchmark/README.md) own their inventories.
`generate_publications.py` is unrelated to release validation.
