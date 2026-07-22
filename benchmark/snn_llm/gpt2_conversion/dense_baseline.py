"""Phase 5.0 GPT-2 dense baseline runner.

Establishes the Hugging Face ``openai-community/gpt2`` dense baseline and
writes a deterministic ``report.json`` describing the run. **No SNN / FAS /
QCFS / IF conversion is performed;** the structural conversion contract
lives in ``conversion_contract.py``.

The module deliberately hides Hugging Face behind a guarded import so the
focused tests can run inside the default virtualenv. CLI callers must run

    uv pip install transformers

once; the runner itself never downloads anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch

try:
    from .conversion_contract import (  # pragma: no cover - module path
        DEFAULT_MODEL_NAME,
        EXPECTED_REVISION,
        GPT2ModelPaths,
        REQUIRED_MODEL_FILES,
        REQUIRED_TOKENIZER_FILES,
        build_model_paths,
        find_missing_files,
    )
except ImportError:  # script entry path; sibling module on sys.path.
    from conversion_contract import (
        DEFAULT_MODEL_NAME,
        EXPECTED_REVISION,
        GPT2ModelPaths,
        REQUIRED_MODEL_FILES,
        REQUIRED_TOKENIZER_FILES,
        build_model_paths,
        find_missing_files,
    )


BASELINE_SCHEMA_VERSION = 1
CONTRACT_KIND = "gpt2-dense-baseline"
MAX_LENGTH = 64
DEFAULT_MAX_SAMPLES = 4
REPORT_FILENAME = "report.json"
VALID_DEVICES = ("cpu", "cuda")

# A short, deterministic prompt bank. The runner truncates this list when
# ``--max-samples`` is below ``DEFAULT_MAX_SAMPLES``; it never extends,
# shuffles or synthesises the bank.
FIXED_PROMPTS: Tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It was the best of times, it was the worst of times.",
    "To be, or not to be, that is the question.",
    "Call me Ishmael. Some years ago\u2014never mind how long precisely\u2014",
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife.",
    "The only way to do great work is to love what you do.",
)


def fixed_prompts() -> Tuple[str, ...]:
    """Return the fixed prompt bank.

    Exposed as a function (in addition to the tuple) so tests and the CLI
    share a single source of truth.
    """

    return FIXED_PROMPTS


def hash_files(root: Path) -> Dict[str, str]:
    """Compute SHA-256 digests for the contract's required artifacts.

    Missing files are silently skipped so the same helper is reusable for
    incremental updates and partial fixture trees.
    """

    candidates: Tuple[str, ...] = REQUIRED_MODEL_FILES + REQUIRED_TOKENIZER_FILES
    digest: Dict[str, str] = {}
    for filename in candidates:
        path = root / filename
        if not path.exists():
            continue
        digest[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def validate_model_root(
    model_root: Path,
    *,
    require_all: bool = False,
) -> Optional[GPT2ModelPaths]:
    """Return a populated :class:`GPT2ModelPaths` or raise.

    When ``require_all`` is true the function raises :class:`FileNotFoundError`
    listing every missing file so the CLI can refuse clearly.
    """

    if not model_root.exists():
        raise FileNotFoundError(
            f"GPT-2 model root does not exist: {model_root}. "
            f"Run the documented `hf download openai-community/gpt2 "
            f"--revision {EXPECTED_REVISION}` command first."
        )
    if not model_root.is_dir():
        raise FileNotFoundError(f"GPT-2 model root is not a directory: {model_root}.")

    missing = find_missing_files(model_root)
    if missing:
        if not require_all:
            return None
        bullet = "\n  ".join(missing)
        raise FileNotFoundError(
            "Required GPT-2 artifacts are missing from "
            f"{model_root}:\n  {bullet}\n"
            "Re-run the documented `hf download openai-community/gpt2 "
            f"--revision {EXPECTED_REVISION} --local-dir {model_root}` "
            "command to populate them."
        )

    return build_model_paths(model_root)


def _import_huggingface(loader: Callable[[], object]) -> object:
    """Return Hugging Face classes or raise a clear ImportError.

    ``loader`` is a thunk the caller supplies so tests can swap in a fake.
    Production callers should pass ``lambda: importlib.import_module(
    "transformers")``.
    """

    try:
        return loader()
    except ImportError as exc:
        raise ImportError(
            "This runner requires the optional Hugging Face dependency. "
            "Install it with: uv pip install transformers"
        ) from exc


def _host_slug() -> str:
    """Stable short slug identifying the host (matches the smoke convention)."""

    host = (socket.gethostname() or "unknown").lower()
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in host).strip("-")
    return cleaned or "unknown"


def build_environment(device: str) -> Dict[str, object]:
    """Collect a deterministic environment description for the report."""

    return {
        "device": device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host": socket.gethostname() or "unknown",
        "slug": _host_slug(),
    }


def compute_baseline(
    *,
    paths: GPT2ModelPaths,
    device: str,
    max_samples: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Run the deterministic dense forward and return ``(metrics, model_meta)``.

    The runner is single-pass: it loads the model once, runs the prompt set,
    and returns finite floating-point metrics. It never trains, never
    decodes, and never touches KV cache.
    """

    transformers = _import_huggingface(lambda: __import__("transformers"))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(paths.root),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer requires a pad token or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(paths.root),
        local_files_only=True,
    )
    model.eval()
    model.to(device)

    prompts = list(FIXED_PROMPTS[:max_samples])
    encoded = tokenizer(
        prompts,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    pad_token_id = tokenizer.pad_token_id
    labels = input_ids.clone()
    if pad_token_id is not None:
        labels[attention_mask == 0] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_tensor = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss_value = loss_tensor.detach().float().cpu().item()
    if not math.isfinite(loss_value):
        raise ValueError(f"Dense baseline loss is not finite: {loss_value!r}.")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    perplexity = math.exp(loss_value)
    if not math.isfinite(perplexity):
        raise ValueError(f"Dense baseline perplexity is not finite: {perplexity!r}.")

    config = model.config
    metrics: Dict[str, object] = {
        "loss": loss_value,
        "perplexity": perplexity,
        "logits_shape": list(logits.shape),
        "parameter_count": parameter_count,
        "vocabulary_size": int(config.vocab_size),
        "context_length": int(getattr(config, "n_positions", MAX_LENGTH)),
        "embedding_dim": int(config.hidden_size),
        "layer_count": int(config.num_hidden_layers),
        "config_revision": EXPECTED_REVISION,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "max_samples": max_samples,
        "fixed_prompts": prompts,
        "file_sha256": hash_files(paths.root),
        "layer_template": {
            "transformer.h.*.attn": "attention_block",
            "transformer.h.*.mlp": "mlp_block",
            "transformer.h.*.ln_1": "layer_norm",
            "transformer.h.*.ln_2": "layer_norm",
        },
        "device": device,
    }
    model_meta: Dict[str, object] = {
        "vocabulary_size": int(config.vocab_size),
        "layer_count": int(config.num_hidden_layers),
        "embedding_dim": int(config.hidden_size),
    }
    return metrics, model_meta


def build_baseline_report(
    *,
    slugs: Mapping[str, object],
    environment: Mapping[str, object],
) -> Dict[str, object]:
    """Compose the top-level report from the metrics gathered upstream."""

    return {
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "model": {
            "name": DEFAULT_MODEL_NAME,
            "revision": slugs["config_revision"],
        },
        "environment": dict(environment),
        "input": {
            "max_samples": slugs["max_samples"],
            "max_length": MAX_LENGTH,
            "prompts": list(slugs["fixed_prompts"]),
        },
        "metrics": {
            "loss": slugs["loss"],
            "perplexity": slugs["perplexity"],
            "logits_shape": list(slugs["logits_shape"]),
            "parameter_count": slugs["parameter_count"],
            "vocabulary_size": slugs["vocabulary_size"],
            "context_length": slugs["context_length"],
        },
        "structure": {
            "embedding_dim": slugs["embedding_dim"],
            "layer_count": slugs["layer_count"],
            "layer_template": dict(slugs["layer_template"]),
        },
        "files": dict(slugs["file_sha256"]),
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 5.0 GPT-2 dense baseline and write "
            "report.json. No SNN / FAS / conversion is performed."
        ),
    )
    parser.add_argument(
        "--model-root",
        required=True,
        type=Path,
        help="Local path containing the GPT-2 artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which to write report.json (refuses to overwrite).",
    )
    parser.add_argument(
        "--device",
        choices=VALID_DEVICES,
        required=True,
        help="Device used for the deterministic forward pass.",
    )
    parser.add_argument(
        "--source-revision",
        required=True,
        help=(
            "Expected GPT-2 revision. Phase 5.0 only accepts "
            f"{EXPECTED_REVISION} so a downstream Phase 5.1 conversion "
            "can rely on the locked weights."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=(
            "Number of fixed prompts to use. Defaults to "
            f"{DEFAULT_MAX_SAMPLES}; can only narrow the bank."
        ),
    )
    return parser.parse_args(argv)


def _emit(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _check_revision(declared: str) -> None:
    if declared != EXPECTED_REVISION:
        raise SystemExit(
            f"--source-revision must equal {EXPECTED_REVISION}; got {declared!r}."
        )


def _write_report(output_dir: Path, report: Mapping[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / REPORT_FILENAME
    if target.exists():
        raise SystemExit(
            f"Refusing to overwrite existing report at {target}. "
            "Choose a new --output-dir or delete the previous result."
        )
    target.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return target


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    _check_revision(args.source_revision)

    if args.max_samples <= 0 or args.max_samples > len(FIXED_PROMPTS):
        _emit(
            f"--max-samples must be between 1 and {len(FIXED_PROMPTS)}; "
            f"got {args.max_samples}."
        )
        return 2

    try:
        paths = validate_model_root(args.model_root, require_all=True)
    except FileNotFoundError as exc:
        _emit(str(exc))
        return 3

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        _emit("CUDA was requested, but torch.cuda.is_available() is false.")
        return 4

    try:
        metrics, _ = compute_baseline(
            paths=paths,
            device=device,
            max_samples=args.max_samples,
        )
    except ImportError as exc:
        _emit(str(exc))
        return 5
    except OSError as exc:
        _emit(f"Failed to load GPT-2 from {paths.root}: {exc}")
        return 6
    except ValueError as exc:
        _emit(str(exc))
        return 7

    environment = build_environment(device)
    report = build_baseline_report(slugs=metrics, environment=environment)
    target = _write_report(args.output_dir, report)

    sys.stdout.write(f"report_path={target.resolve()}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
