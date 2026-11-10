---
id: tools
type: module-design
status: draft
title: Validation and release tooling
parent: architecture
---

## Responsibility

`tools/` holds `release/` validation, the changelog and logging-policy checks, and
`nsys/` and `viz/` analysis. `test/` holds the correctness suites, and `benchmark/` the
task and performance runners.

## Boundary

- Release entry points never publish, tag or push. Publishing is the manually
  triggered `publish-pypi.yml` workflow.
- Examples in `activation_based/examples` demonstrate use. `benchmark/` measures
  against tasks, and only cases selected for a release are mandatory.
- `CHANGELOG.md` is edited by hand, and `docs/source/changelog.rst` is generated.

## Release contract

- Formal runs need Linux, clean tracked source and fixed inputs. `--development` runs
  are marked ineligible.
- Skips pass only when they match a profile's `allow_skip_reasons`; the `cuda` profile
  allows none. A missing dependency does not validate its capability.
- The previous formal release is the only version baseline.
- The efficiency gate compares three paired runs. It passes if all throughput ratios
  are at least 0.98, fails if none are, and is blocked otherwise. Quality tolerances
  are separate.
- Release quality detects drift against that baseline. Matching a paper's accuracy is
  not an acceptance requirement.

## Verification layers

| Layer | Runs | Does not run |
| --- | --- | --- |
| CI | Format, Ruff, CPU tests, release self-checks, Sphinx with `-W`, package install | CUDA, Triton, native builds, `test_distributed_*` |
| Release cases | CPU, minimum-version, CUDA, package, native, docs and links checks, plus selected task quality | `fp8_hardware`, which is deferred |

Distributed tests check contracts and do not replace real multi-rank acceptance.
