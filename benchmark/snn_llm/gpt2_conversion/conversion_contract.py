"""Phase 5.0 GPT-2 conversion contract.

This module only inspects the structural shape of a Hugging Face
``GPT2LMHeadModel``. It **does not** perform FAS / QCFS / IF conversion,
nor does it change the model in any way. The returned contract is a static
manifest that Phase 5.0 stages will consume; the actual conversion work
lives in Phase 5.1.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, Tuple


CONTRACT_SCHEMA_VERSION = 1
DEFAULT_MODEL_NAME = "openai-community/gpt2"
EXPECTED_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
CONTRACT_KIND = "gpt2-dense-baseline-contract"

# Required artifacts. The runner refuses to start if any of these files is
# missing locally; it never reaches out to the network on its own. Both
# lists are part of the contract so Phase 5.1 conversions can reason about
# the same on-disk expectations.
REQUIRED_MODEL_FILES: Tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
)
REQUIRED_TOKENIZER_FILES: Tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


# Order matters for serialization determinism. The keys use the same
# dotted-path conventions the Hugging Face ``nn.Module`` would expose at
# runtime, so a Phase 5.1 scanner that walks ``model.named_modules()`` can
# join against this table.
STRUCTURAL_LAYERS: Tuple[Tuple[str, str], ...] = (
    ("transformer.wte", "token_embedding"),
    ("transformer.wpe", "position_embedding"),
    ("transformer.h.*.attn", "attention_block"),
    ("transformer.h.*.mlp", "mlp_block"),
    ("transformer.h.*.ln_1", "layer_norm"),
    ("transformer.h.*.ln_2", "layer_norm"),
    ("transformer.ln_f", "final_layer_norm"),
    ("lm_head", "output_head"),
)


_PHASE5_CANDIDATE_CONVERSION: Tuple[str, ...] = (
    "Q: Is the dense GPT-2 forward numerically reproduced under FP32 / AMP?",
    "Q: Which ANN modules map onto QCFS / IF / soft-reset candidates?",
    "Q: Can calibration data fit through the same fixed prompt set?",
    "Q: Does the dense forward expose enough hooks for Phase 5.1 conversion?",
)


_EXPLICITLY_UNSUPPORTED_NOW: Tuple[str, ...] = (
    "KV cache / cached decoding and Hugging Face generation APIs (deferred to Phase 6)",
    "Qwen, RoPE, GQA and dynamic-cache layouts (deferred to Phase 7)",
    "Multi-GPU / tensor-parallel inference (out of Phase 5.0 scope)",
    "FP8 / Transformer Engine backed matmuls (Phase 4 scope, not 5.0)",
    "Full PPL benchmark on WikiText / OpenWebText and broader corpora "
    "(out of Phase 5.0 scope)",
)


@dataclasses.dataclass(frozen=True)
class GPT2ModelPaths:
    """Resolved on-disk locations of the locally prepared GPT-2 artifacts."""

    root: Path
    config: Path
    generation_config: Path
    model_safetensors: Path
    tokenizer_json: Path
    tokenizer_config: Path
    vocab: Path
    merges: Path


def scan_structure() -> Dict[str, str]:
    """Return the static GPT-2 layer table consumed by Phase 5.1.

    The scanner is intentionally a pure function returning a fresh dict so
    callers cannot mutate the canonical table by accident.
    """

    return dict(STRUCTURAL_LAYERS)


def resolve_paths(model_root: Path) -> Dict[str, Path]:
    """Return the per-file paths expected inside ``model_root``."""

    return {
        "config": model_root / "config.json",
        "generation_config": model_root / "generation_config.json",
        "model_safetensors": model_root / "model.safetensors",
        "tokenizer_json": model_root / "tokenizer.json",
        "tokenizer_config": model_root / "tokenizer_config.json",
        "vocab": model_root / "vocab.json",
        "merges": model_root / "merges.txt",
    }


def find_missing_files(model_root: Path) -> Tuple[str, ...]:
    """Return the sorted tuple of contract-required files that are missing.

    Pure helper so the dense baseline runner and the contract report can
    agree on what counts as ``ready``.
    """

    missing = sorted(
        filename
        for filename in REQUIRED_MODEL_FILES + REQUIRED_TOKENIZER_FILES
        if not (model_root / filename).is_file()
    )
    return tuple(missing)


def build_model_paths(model_root: Path) -> GPT2ModelPaths:
    """Construct a fully-populated :class:`GPT2ModelPaths`.

    Callers should validate first via :func:`find_missing_files`; this
    function trusts the contract has already been enforced.
    """

    paths = resolve_paths(model_root)
    return GPT2ModelPaths(root=model_root, **paths)


def build_contract_report() -> Dict[str, object]:
    """Assemble the top-level contract report consumed by Phase 5.0 callers.

    The top-level keys are pinned: tests rely on this exact set to detect
    shape regressions.
    """

    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "supported_dense_baseline": {
            "model": DEFAULT_MODEL_NAME,
            "revision": EXPECTED_REVISION,
            "kind": CONTRACT_KIND,
            "structural_layers": scan_structure(),
            "required_model_files": list(REQUIRED_MODEL_FILES),
            "required_tokenizer_files": list(REQUIRED_TOKENIZER_FILES),
        },
        "phase5_candidate_conversion": list(_PHASE5_CANDIDATE_CONVERSION),
        "explicitly_unsupported_now": list(_EXPLICITLY_UNSUPPORTED_NOW),
    }


__all__ = [
    "CONTRACT_KIND",
    "CONTRACT_SCHEMA_VERSION",
    "DEFAULT_MODEL_NAME",
    "EXPECTED_REVISION",
    "GPT2ModelPaths",
    "REQUIRED_MODEL_FILES",
    "REQUIRED_TOKENIZER_FILES",
    "STRUCTURAL_LAYERS",
    "build_contract_report",
    "build_model_paths",
    "find_missing_files",
    "resolve_paths",
    "scan_structure",
]
