---
id: datasets
type: module-design
status: draft
title: Neuromorphic datasets
parent: architecture
---

## Responsibility

`datasets` prepares event datasets on disk and loads them as raw events or frames.
`NeuromorphicDatasetFolder` runs one pipeline: validate config, prepare raw data,
select a builder, build, then hand the result to torchvision's `DatasetFolder`.

## Boundary

- Within the package, depends only on `configure` and `logger`. It also uses
  torchvision and the scientific stack.
- New datasets subclass `NeuromorphicDatasetFolder` and supply download, extraction and
  raw-format steps. `SpeechCommands` is a plain `Dataset` outside this pipeline.

## On-disk contract

Under `root/`, `download/` and `extract/` lead to `events_np/` (`events_h5/` for SHD and
SSC), which is the raw root. Frame caches are siblings, named from their parameters:
`frames_number_{N}_split_by_{time|number}/`, `duration_{D}/`, or a custom directory
name.

- An existing processed directory is reused with no content check, so distinct custom
  integrate functions need distinct directory names.
- When integration behavior changes, users must delete cached directories, and the
  changelog must tell them to.
- `<dir>.building` exists during a build and is left behind on failure. The next run
  raises instead of loading partial results, and the user removes the partial output.
- SSC and SHD need separate roots.

`configure` sets preprocessing thread count and NPZ compression; `architecture` covers
how it is read.

## Known limitations

- ASL-DVS, DVS128 Gesture, DVS-Lip, HARDVS, N-Caltech101, N-MNIST and Nav Gesture
  require manual download.
