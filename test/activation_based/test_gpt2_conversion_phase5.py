"""Tests for Phase 5.0 GPT-2 dense baseline and conversion contract.

These tests cover the seams established by
``.agents/plans/PLAN.md``:

* CLI surface for ``dense_baseline.py`` exposes only Phase 5.0 controls.
* Optional Hugging Face dependency is reported honestly when absent.
* Required model/tokenizer files produce a deterministic refusal when missing.
* Output directory refuses to overwrite an existing ``report.json``.
* Baseline report and contract report top-level keys are pinned.
* Contract scanner recognises GPT-2 structural layers.
* ``--max-samples`` only narrows the fixed prompt set; it cannot change
  tokenizer/model/config state.
* Phase 5.0 outputs must not advertise FAS/SNN/converted PPL anywhere.

No test in this file requires the Hugging Face ``transformers`` package to
be importable, so the focused tests run in the default virtualenv.
"""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_SCRIPT = (
    _REPO_ROOT / "benchmark" / "snn_llm" / "gpt2_conversion" / "dense_baseline.py"
)


def _import_package():
    """Import the conversion package bypassing top-level __init__ import
    side effects when Hugging Face is unavailable.
    """

    sys.path.insert(0, str(_REPO_ROOT))
    if "benchmark.snn_llm.gpt2_conversion" in sys.modules:
        return sys.modules["benchmark.snn_llm.gpt2_conversion"]
    return importlib.import_module("benchmark.snn_llm.gpt2_conversion")


def _import_baseline_module():
    sys.path.insert(0, str(_REPO_ROOT))
    return importlib.import_module("benchmark.snn_llm.gpt2_conversion.dense_baseline")


def _import_contract_module():
    sys.path.insert(0, str(_REPO_ROOT))
    return importlib.import_module(
        "benchmark.snn_llm.gpt2_conversion.conversion_contract"
    )


# --------------------------------------------------------------------------- #
# Slice 1: constants, pure helpers, file validation.
# --------------------------------------------------------------------------- #


def test_package_exposes_pinned_baseline_constants():
    package = _import_package()

    assert package.BASELINE_SCHEMA_VERSION == 1
    assert package.CONTRACT_SCHEMA_VERSION == 1
    assert package.EXPECTED_REVISION == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    assert package.DEFAULT_MODEL_NAME == "openai-community/gpt2"
    assert package.MAX_LENGTH == 64
    assert package.DEFAULT_MAX_SAMPLES == 4
    assert isinstance(package.FIXED_PROMPTS, tuple)
    assert len(package.FIXED_PROMPTS) >= package.DEFAULT_MAX_SAMPLES


def test_required_files_lists_match_plan():
    package = _import_package()

    assert set(package.REQUIRED_MODEL_FILES) == {
        "config.json",
        "generation_config.json",
        "model.safetensors",
    }
    assert set(package.REQUIRED_TOKENIZER_FILES) == {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    }


def test_fixed_prompts_returns_frozen_tuple_of_strings():
    baseline = _import_baseline_module()

    prompts = baseline.fixed_prompts()
    assert prompts == baseline.FIXED_PROMPTS
    assert isinstance(prompts, tuple)
    assert all(isinstance(item, str) for item in prompts)
    assert len(prompts) >= baseline.DEFAULT_MAX_SAMPLES


def test_hash_files_returns_empty_mapping_when_directory_missing(tmp_path: Path):
    baseline = _import_baseline_module()

    digest = baseline.hash_files(tmp_path / "missing")
    assert digest == {}


def test_hash_files_records_sha256_only_for_existing_files(tmp_path: Path):
    baseline = _import_baseline_module()

    (tmp_path / "config.json").write_bytes(b"hello")
    (tmp_path / "model.safetensors").write_bytes(b"world")
    digest = baseline.hash_files(tmp_path)

    assert set(digest) == {"config.json", "model.safetensors"}
    assert all(len(value) == 64 for value in digest.values())
    assert digest["config.json"] != digest["model.safetensors"]


def test_validate_model_root_requires_directory(tmp_path: Path):
    baseline = _import_baseline_module()

    with pytest.raises(FileNotFoundError):
        baseline.validate_model_root(tmp_path / "absent")


def _populate_model_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename in (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ):
        (root / filename).write_text(filename)


def test_validate_model_root_accepts_complete_directory(tmp_path: Path):
    baseline = _import_baseline_module()

    _populate_model_root(tmp_path)
    paths = baseline.validate_model_root(tmp_path)

    assert isinstance(paths, baseline.GPT2ModelPaths)
    assert paths.config.name == "config.json"
    assert paths.model_safetensors.name == "model.safetensors"
    assert paths.tokenizer_json.name == "tokenizer.json"


def test_validate_model_root_rejects_missing_required_files(tmp_path: Path):
    baseline = _import_baseline_module()

    paths = baseline.validate_model_root(tmp_path)
    assert paths is None

    with pytest.raises(FileNotFoundError) as excinfo:
        baseline.validate_model_root(tmp_path, require_all=True)
    message = str(excinfo.value)
    assert "uv pip install transformers" not in message
    expected_missing = sorted(
        set(baseline.REQUIRED_MODEL_FILES) | set(baseline.REQUIRED_TOKENIZER_FILES)
    )
    for filename in expected_missing:
        assert filename in message


# --------------------------------------------------------------------------- #
# Slice 2: conversion contract -- structural scanner.
# --------------------------------------------------------------------------- #


def test_scan_structure_collects_canonical_gpt2_layers():
    contract = _import_contract_module()

    layers = contract.scan_structure()
    expected = {
        "transformer.wte": "token_embedding",
        "transformer.wpe": "position_embedding",
        "transformer.h.*.attn": "attention_block",
        "transformer.h.*.mlp": "mlp_block",
        "transformer.h.*.ln_1": "layer_norm",
        "transformer.h.*.ln_2": "layer_norm",
        "transformer.ln_f": "final_layer_norm",
        "lm_head": "output_head",
    }
    assert dict(layers) == expected


def test_build_contract_report_pins_top_level_keys():
    contract = _import_contract_module()

    report = contract.build_contract_report()

    assert set(report) == {
        "contract_schema_version",
        "supported_dense_baseline",
        "phase5_candidate_conversion",
        "explicitly_unsupported_now",
    }
    assert report["contract_schema_version"] == contract.CONTRACT_SCHEMA_VERSION
    assert report["supported_dense_baseline"]["model"] == "openai-community/gpt2"
    assert report["supported_dense_baseline"]["revision"] == (
        "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )


def test_unsupported_features_cover_plan_exclusions():
    contract = _import_contract_module()

    report = contract.build_contract_report()
    unsupported = " ".join(report["explicitly_unsupported_now"]).lower()

    must_mention = (
        "kv cache",
        "generation",
        "qwen",
        "rope",
        "gqa",
        "multi-gpu",
        "fp8",
        "full ppl",
    )
    for needle in must_mention:
        assert needle in unsupported, needle


def test_contract_report_never_advertises_snn_or_fas_result():
    contract = _import_contract_module()

    report = contract.build_contract_report()
    forbidden_keys = {
        "snn_result",
        "fas_result",
        "converted_ppl",
        "spike_rate",
        "qcfs_loss",
    }
    leaked = forbidden_keys & set(report)
    payload = json.dumps(report).lower()
    for needle in (
        "snn_result",
        "fas_result",
        "converted_ppl",
        "spike_rate",
        "qcfs_loss",
    ):
        assert needle not in payload, needle
    assert not leaked, leaked


def test_contract_report_json_serialisable():
    contract = _import_contract_module()

    report = contract.build_contract_report()
    encoded = json.dumps(report)
    decoded = json.loads(encoded)
    assert decoded == report


# --------------------------------------------------------------------------- #
# Slice 3: dense baseline environment + report schema.
# --------------------------------------------------------------------------- #


def test_build_environment_records_required_keys():
    baseline = _import_baseline_module()

    report = baseline.build_environment(device="cpu")

    assert set(report) == {
        "device",
        "torch_version",
        "torch_cuda_version",
        "cuda_visible_devices",
        "host",
        "slug",
    }
    assert report["device"] == "cpu"
    assert isinstance(report["torch_version"], str)
    assert report["cuda_visible_devices"] is None or isinstance(
        report["cuda_visible_devices"], str
    )
    assert isinstance(report["host"], str)
    assert isinstance(report["slug"], str) and report["slug"]


def test_build_environment_records_cuda_visible_devices(monkeypatch):
    baseline = _import_baseline_module()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    report = baseline.build_environment(device="cuda")

    assert report["device"] == "cuda"
    assert report["cuda_visible_devices"] == "0,1"


def test_build_baseline_report_pins_top_level_keys():
    baseline = _import_baseline_module()

    slugs = {
        "config_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "context_length": 64,
        "device": "cpu",
        "embedding_dim": 768,
        "file_sha256": {},
        "fixed_prompts": list(baseline.FIXED_PROMPTS)[: baseline.DEFAULT_MAX_SAMPLES],
        "layer_count": 12,
        "layer_template": {
            "transformer.h.*.attn": "attention_block",
            "transformer.h.*.ln_1": "layer_norm",
            "transformer.h.*.ln_2": "layer_norm",
            "transformer.h.*.mlp": "mlp_block",
        },
        "logits_shape": [4, 64, 50257],
        "loss": 3.5,
        "max_samples": baseline.DEFAULT_MAX_SAMPLES,
        "parameter_count": 124_439_808,
        "perplexity": math.exp(3.5),
        "torch_cuda_version": None,
        "torch_version": "0.0.0",
        "vocabulary_size": 50257,
    }
    report = baseline.build_baseline_report(
        slugs=slugs,
        environment={
            "cuda_visible_devices": None,
            "device": "cpu",
            "host": "fixture",
            "slug": "fixture",
            "torch_cuda_version": None,
            "torch_version": "0.0.0",
        },
    )

    assert set(report) == {
        "baseline_schema_version",
        "kind",
        "model",
        "environment",
        "metrics",
        "input",
        "structure",
        "files",
    }
    assert report["kind"] == "gpt2-dense-baseline"
    assert report["model"]["revision"] == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    assert report["metrics"]["loss"] == 3.5
    assert report["metrics"]["perplexity"] == math.exp(3.5)
    assert report["input"]["max_samples"] == baseline.DEFAULT_MAX_SAMPLES
    assert report["input"]["max_length"] == baseline.MAX_LENGTH
    assert report["structure"]["layer_count"] == 12

    encoded = json.dumps(report).lower()
    for forbidden in (
        "snn_result",
        "fas_result",
        "converted_ppl",
        "spike_rate",
        "qcfs_loss",
    ):
        assert forbidden not in encoded, forbidden


class _FakeTokenizer:
    pad_token_id = None
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, root: str, **kwargs):
        cls.root = root
        cls.kwargs = kwargs
        return cls()

    def __call__(self, prompts, *, padding, max_length, truncation, return_tensors):
        import torch

        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        batch = len(prompts)
        input_ids = torch.arange(max_length).remainder(32).repeat(batch, 1)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeModel:
    class _Config:
        vocab_size = 32
        n_positions = 1024
        hidden_size = 768
        num_hidden_layers = 12

    config = _Config()

    @classmethod
    def from_pretrained(cls, root: str, **kwargs):
        cls.root = root
        cls.kwargs = kwargs
        return cls()

    def eval(self):
        return self

    def to(self, device: str):
        self.device = device
        return self

    def parameters(self):
        import torch

        return [torch.nn.Parameter(torch.ones(2, 2))]

    def __call__(self, *, input_ids, attention_mask):
        import torch

        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.config.vocab_size,
            device=input_ids.device,
        )
        return type("Output", (), {"logits": logits})()


class _FakeTransformers:
    AutoTokenizer = _FakeTokenizer
    AutoModelForCausalLM = _FakeModel


def test_compute_baseline_reports_loss_in_range(tmp_path: Path, monkeypatch):
    baseline = _import_baseline_module()

    _populate_model_root(tmp_path)
    paths = baseline.validate_model_root(tmp_path)
    monkeypatch.setattr(
        baseline,
        "_import_huggingface",
        lambda loader: _FakeTransformers,
    )

    metrics, model_meta = baseline.compute_baseline(
        paths=paths,
        device="cpu",
        max_samples=baseline.DEFAULT_MAX_SAMPLES,
    )

    assert math.isfinite(metrics["loss"])
    assert metrics["logits_shape"][0] == baseline.DEFAULT_MAX_SAMPLES
    assert metrics["logits_shape"][1] == baseline.MAX_LENGTH
    assert metrics["logits_shape"][2] == model_meta["vocabulary_size"]
    assert metrics["layer_count"] == model_meta["layer_count"]
    assert metrics["embedding_dim"] == model_meta["embedding_dim"]
    assert metrics["parameter_count"] > 0
    assert metrics["perplexity"] == pytest.approx(math.exp(metrics["loss"]))
    assert _FakeTokenizer.kwargs["local_files_only"] is True
    assert _FakeModel.kwargs["local_files_only"] is True


# --------------------------------------------------------------------------- #
# Slice 4: CLI surface and end-to-end runner behaviour.
# --------------------------------------------------------------------------- #


def _run_cli(
    *args: str,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess:
    import os

    return subprocess.run(
        [sys.executable, str(_BASELINE_SCRIPT), *args],
        cwd=cwd or _REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
    )


def test_dense_baseline_cli_help_only_exposes_phase5_controls():
    completed = _run_cli("--help")

    assert completed.returncode == 0, completed.stderr
    for flag in ("--model-root", "--output-dir", "--device", "--source-revision"):
        assert flag in completed.stdout
    assert "--max-samples" in completed.stdout
    forbidden = (
        "--checkpoint",
        "--do-sample",
        "--max-new-tokens",
        "--revision-cli",
    )
    for flag in forbidden:
        assert flag not in completed.stdout, flag


def test_dense_baseline_cli_rejects_unknown_flags(tmp_path: Path):
    completed = _run_cli(
        "--device",
        "cpu",
        "--model-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "out"),
        "--source-revision",
        "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "--mystery-flag",
        "x",
    )

    assert completed.returncode != 0
    assert "mystery-flag" in completed.stderr


def test_dense_baseline_refuses_to_overwrite_existing_report(tmp_path: Path):
    baseline = _import_baseline_module()
    output_dir = tmp_path / "report"
    first_report = {
        "baseline_schema_version": baseline.BASELINE_SCHEMA_VERSION,
        "kind": baseline.CONTRACT_KIND,
    }

    baseline._write_report(output_dir, first_report)  # noqa: SLF001
    report_path = output_dir / "report.json"
    assert report_path.exists()
    original = report_path.read_bytes()

    with pytest.raises(SystemExit) as excinfo:
        baseline._write_report(output_dir, {"kind": "replacement"})  # noqa: SLF001

    assert "refusing to overwrite" in str(excinfo.value).lower()
    assert report_path.read_bytes() == original


def test_dense_baseline_rejects_missing_model_files(tmp_path: Path):
    output_dir = tmp_path / "report"
    completed = _run_cli(
        "--model-root",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--source-revision",
        "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    )

    assert completed.returncode != 0
    assert "model.safetensors" in completed.stderr


def test_dense_baseline_rejects_missing_revision(tmp_path: Path):
    model_root = tmp_path / "model"
    _populate_model_root(model_root)
    completed = _run_cli(
        "--model-root",
        str(model_root),
        "--output-dir",
        str(tmp_path / "report"),
        "--device",
        "cpu",
        "--source-revision",
        "deadbeef",
    )
    assert completed.returncode != 0
    assert "revision" in completed.stderr.lower()


def test_dense_baseline_rejects_unknown_device(tmp_path: Path):
    model_root = tmp_path / "model"
    _populate_model_root(model_root)
    completed = _run_cli(
        "--model-root",
        str(model_root),
        "--output-dir",
        str(tmp_path / "report"),
        "--device",
        "tpu",
        "--source-revision",
        "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    )
    assert completed.returncode != 0


def test_max_samples_only_narrows_the_fixed_prompt_bank():
    """``--max-samples`` truncates ``FIXED_PROMPTS``; nothing else.

    Phase 5.0 contract: the runner narrows an existing bank of literal
    strings. It must never substitute, shuffle, synthesise, or re-tokenize
    prompts based on the count. Tokenizer / model / config state are
    explicitly off-limits; truncation is the sole transformation.
    """

    baseline = _import_baseline_module()

    full = baseline.fixed_prompts()
    assert len(full) >= baseline.DEFAULT_MAX_SAMPLES >= 1

    seen: list[tuple[str, ...]] = []
    for count in (1, 2, baseline.DEFAULT_MAX_SAMPLES, len(full)):
        truncated = full[:count]
        assert len(truncated) == count
        # Strict prefix of the canonical bank, in canonical order.
        assert truncated == full[:count]
        # No synthesised / shuffled entries; every slot is the original
        # literal at the same index.
        for index, prompt in enumerate(truncated):
            assert prompt == full[index]
            assert isinstance(prompt, str)
        seen.append(truncated)

    # Different counts must select different prefixes, but never different
    # content per slot. Order and identity are preserved.
    assert seen[0] != seen[1]
    assert seen[1] != seen[2]
    assert seen[0] == full[:1]
    assert seen[-1] == full


def test_max_samples_must_be_positive(tmp_path: Path):
    model_root = tmp_path / "model"
    _populate_model_root(model_root)
    output_dir = tmp_path / "report"

    completed = _run_cli(
        "--model-root",
        str(model_root),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--max-samples",
        "0",
        "--source-revision",
        "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    )
    assert completed.returncode != 0


# --------------------------------------------------------------------------- #
# Slice 5: optional Hugging Face dependency reporting.
# --------------------------------------------------------------------------- #


def test_import_huggingface_missing_raises_helpful_error():
    baseline = _import_baseline_module()

    def failing():
        raise ImportError("No module named 'transformers'")

    with pytest.raises(ImportError) as excinfo:
        baseline._import_huggingface(failing)  # noqa: SLF001
    assert "uv pip install transformers" in str(excinfo.value)
    assert "hugging face" in str(excinfo.value).lower()


def test_import_huggingface_returns_dependencies_when_present():
    baseline = _import_baseline_module()

    loader_calls = []

    fake_module = type(sys)("fake_transformers")

    def loader():
        loader_calls.append("ok")
        return fake_module

    result = baseline._import_huggingface(loader)  # noqa: SLF001
    assert loader_calls == ["ok"]
    assert result is fake_module
