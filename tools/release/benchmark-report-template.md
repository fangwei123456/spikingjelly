# SpikingJelly {version} release benchmark report

Copy this template for each release. Select tasks from the
[benchmark inventory](../../benchmark/README.md) before running. Link existing
manifests, JSON reports and raw artifacts; do not copy all logs into this report.
This report records benchmark acceptance; overall release signoff also requires
installation, framework correctness and documentation checks.

## Decision and scope

- Status: {pass / fail / blocked / not-run}; updated: {timestamp and timezone}.
- Candidate: {full commit SHA}; runner/config: {full SHA and config hash}.
- Release scope and agreed criteria: {document and approval reference}.
- Baseline: {previous formal version and full SHA, or approved N/A with reason}.
- Selected tasks: {IDs}; deferred tasks: {IDs and reasons}.
- Blockers, accepted limitations and reviewer: {details}.

Never mark an omitted comparison as passed. Record criteria before running;
report-only quality results need an explicit agreement rather than a threshold
chosen after seeing the result. A running experiment has status `not-run` for
acceptance, with its progress recorded separately.

## Configuration and environment

Use one row per distinct workload/environment. Include the full manifests for
exact commands, dependencies and machine-specific paths.

| Run ID / manifest | Task, model, parameters, T | Data/model revision and splits | Seed, batch, recipe, precision | Python / Torch / Triton / provider | GPU / count / driver / topology |
| --- | --- | --- | --- | --- | --- |
| {fill} | {fill} | {train/calibration/validation/test counts and hashes} | {fill} | {actual selected implementation; native build provenance} | {UUID and sharing/isolation} |

Record checkpoint hashes/epochs and any resume or recipe overrides. State how
checkpoints are selected; keep test data out of training, calibration and model
selection. Separate complete training, fixed-weight evaluation and bounded runs.

## Task quality

Use our previous formal release as the quality reference when comparison is
required. Paper accuracy is context, not a release target. Separate evaluation
of the same checkpoint from independent retraining; an isolated seed does not
establish the range of training noise.

Repeat rows for every seed; include all attempts and the chosen aggregation.
For ongoing training, label metrics interim and record completed/planned epochs,
latest checkpoint and recent time per epoch. Interim metrics do not close a run.

| Task / seed | Budget completed / planned | Evaluation split and count | Final quality | Best validation quality / selection rule | Criterion / result | Curves, checkpoint and report |
| --- | --- | --- | --- | --- | --- | --- |
| {fill} | {epochs/steps/tokens} | {fill} | {top1/top5 %, PPL or task score} | {N/A if not used} | {threshold or approved report-only; status} | {paths and SHA256} |

For version comparisons, record the following alongside the absolute results.
Lock data/splits, preprocessing, precision and environment. For evaluation drift,
also lock checkpoint bytes; for retraining drift, match initial weights/seeds,
recipe and budget, and report paired variability when needed. Separate intended
behavior changes from unexplained drift; a quality score alone cannot establish
numerical equivalence.

| Task / comparison type | Baseline quality | Candidate quality | Change / variability | Predeclared tolerance / decision |
| --- | --- | --- | --- | --- |
| {same-checkpoint evaluation / retraining} | {fill} | {fill} | {accuracy: candidate minus baseline in percentage points; PPL: ratio/change; paired seeds if used} | {criterion or approved N/A} |

For conversion, include original-model quality, converted quality and the change
on the same inputs/calibration. State the evaluation budget: quick PPL is not a
full language evaluation; synthetic loss or generated tokens do not establish
task quality.

## Performance and memory

Use fixed weights and real inputs for task performance. Verify outputs, states,
gradients and reset as applicable before timing, using the existing tolerances;
link the correctness packets/results. Record warmup, measured steps, independent
process count, order, clocks/TF32/CUDA Graph settings and GPU isolation.

| Task / phase / precision / batch | Checkpoint / input hashes | Repetitions and timing boundary | GPU latency median [p10, p90] ms | Wall throughput samples/s or tokens/s | Peak allocated / reserved bytes | Cold start / compilation s | Result / raw samples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| {fill} | {fill} | {forward/backward/optimizer/reset; loader/augmentation included?} | {fill} | {fill} | {fill} | {fill} | {fill} |

Keep per-process results and variation; identify the aggregation above. When a
runtime exposes only device-wide memory polling, report that metric in MiB and
its scope (including pools/other processes), not as Torch peak allocated memory.
Keep invalid measurements, skipped AMP updates and reruns with their reasons.

If a historical comparison is required, add per-case baseline/candidate ratios,
the predeclared metric/threshold and a pass/fail/blocked decision. If waived,
report absolute values and the waiver. Do not infer speedups from absolute data.

## Additional diagnostics and limitations

| Check / environment | Workload and correctness oracle | Result / memory | Supported conclusion | Evidence / remaining gap |
| --- | --- | --- | --- | --- |
| {eager/compile, distributed topology, optional dependency, etc.} | {fill} | {fill} | {fill} | {fill} |

Label random-input timings and functional smoke tests as diagnostics. List
unmeasured hardware, dtypes/topologies, known regressions and missing evaluation
budgets alongside the affected claims. Link framework test results, unexpected
skip resolutions and installation checks without adding their overlapping
counts into a single benchmark total.

## Evidence and handoff

- Exact commands, exit codes, start/end times and all failed attempts: {index}.
- Reports, raw samples, curves, correctness packets and checkpoints: {index}.
- Source/package/evidence archives, SHA256 and verified durable copies: {index}.
- Provider instance IDs, elapsed time, estimated versus billed costs and resource
  cleanup verification: {record}; separate compute, storage and network costs.
- Required follow-up, owner and final candidate/review signoff: {record}.

Changes after measurement require a new source identity and an assessment of
which results need rerunning. A generated report or valid archive hash alone
does not establish final release readiness.
