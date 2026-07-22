"""SpikeGPT source loading with current SpikingJelly neuron injection."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from unittest import mock

from spikingjelly.activation_based import neuron, surrogate


SPIKEGPT_REVISION = "029f86f0536f2b2451524038fc9890cc76c2429e"
_GIT_TIMEOUT_SECONDS = 30


def make_current_lif(backend: str) -> neuron.LIFNode:
    return neuron.LIFNode(
        tau=2.0,
        decay_input=True,
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function=surrogate.ATan(alpha=2.0),
        detach_reset=False,
        step_mode="m",
        backend=backend,
        store_v_seq=True,
    )


def _git_head(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _verify_source(root: Path, declared_revision: str | None) -> str:
    model_path = root / "src" / "model.py"
    if not model_path.is_file():
        raise RuntimeError(f"SpikeGPT model source not found: {model_path}")
    detected_revision = _git_head(root)
    revision = declared_revision or detected_revision
    if revision is None:
        raise RuntimeError(
            "SpikeGPT source has no Git metadata; pass --source-revision."
        )
    if revision != SPIKEGPT_REVISION:
        raise RuntimeError(
            f"Expected SpikeGPT revision {SPIKEGPT_REVISION}, got {revision}."
        )
    if detected_revision is not None and detected_revision != revision:
        raise RuntimeError(
            f"Declared revision {revision} does not match Git HEAD {detected_revision}."
        )
    return revision


def _instrument_wkv(author_model, require_wkv: bool) -> None:
    original = getattr(
        author_model, "_spikingjelly_original_run_cuda", author_model.RUN_CUDA
    )
    author_model._spikingjelly_original_run_cuda = original
    author_model._comparison_wkv_calls = 0
    if require_wkv:

        def _tracked_wkv(*args, **kwargs):
            author_model._comparison_wkv_calls += 1
            return original(*args, **kwargs)

        author_model.RUN_CUDA = _tracked_wkv
    else:

        def _unexpected_wkv(*args, **kwargs):
            raise RuntimeError("The layer-0 comparison unexpectedly called WKV.")

        author_model.RUN_CUDA = _unexpected_wkv


def _load_author_model(root: Path, require_wkv: bool):
    existing_src = sys.modules.get("src")
    if existing_src is not None:
        src_paths = [
            Path(path).resolve() for path in getattr(existing_src, "__path__", [])
        ]
        if (root / "src").resolve() not in src_paths:
            raise RuntimeError(
                "A different top-level 'src' package is already imported."
            )

    root_string = str(root.resolve())
    sys.path.insert(0, root_string)
    try:
        importlib.invalidate_caches()
        if require_wkv:
            from torch.utils.cpp_extension import load as extension_load

            def load_from_spikegpt(*args, **kwargs):
                positional = list(args)
                if "sources" in kwargs:
                    sources = kwargs["sources"]
                elif len(positional) >= 2:
                    sources = positional[1]
                else:
                    raise TypeError("SpikeGPT extension load requires sources.")
                resolved_sources = [
                    str((root / source).resolve())
                    if not Path(source).is_absolute()
                    else source
                    for source in sources
                ]
                if "sources" in kwargs:
                    kwargs["sources"] = resolved_sources
                else:
                    positional[1] = resolved_sources
                return extension_load(*positional, **kwargs)

            extension_patch = mock.patch(
                "torch.utils.cpp_extension.load", side_effect=load_from_spikegpt
            )
        else:
            extension_patch = mock.patch(
                "torch.utils.cpp_extension.load", return_value=object()
            )
        with extension_patch:
            author_model = importlib.import_module("src.model")
    finally:
        sys.path.remove(root_string)

    if Path(author_model.__file__).resolve() != (root / "src" / "model.py").resolve():
        raise RuntimeError(
            f"Imported unexpected SpikeGPT model: {author_model.__file__}"
        )

    _instrument_wkv(author_model, require_wkv)
    return author_model
