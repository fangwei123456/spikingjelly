"""Focused tests for the private Phase 5.1 GPT-2 MLP conversion runner."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = (
    _REPO_ROOT / "benchmark" / "snn_llm" / "gpt2_conversion" / "mlp_ann2snn_slice.py"
)
_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_RUNNER), *args],
        cwd=_REPO_ROOT,
        env={"PYTHONPATH": str(_REPO_ROOT), **os.environ},
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_help_only_exposes_phase51_controls():
    completed = _run_cli("--help")

    assert completed.returncode == 0
    for option in (
        "--model-root",
        "--output-dir",
        "--device",
        "--source-revision",
        "--block-index",
        "--time-steps",
        "--max-samples",
    ):
        assert option in completed.stdout
    for forbidden in (
        "generation",
        "cache",
        "checkpoint",
        "fine-tune",
        "fine_tune",
        "ppl",
    ):
        assert forbidden not in completed.stdout.lower()


def test_signed_if_proxy_reconstructs_qcfs_and_uses_public_neuron():
    from benchmark.snn_llm.gpt2_conversion.mlp_ann2snn_slice import (
        build_signed_if_proxy,
        calibrate_signed_qcfs,
    )
    from spikingjelly.activation_based import neuron

    hidden_dense = torch.tensor(
        [
            [[-0.20, 0.25, 0.80, -1.20], [0.55, -0.74, 1.49, -1.51]],
            [[0.10, -0.49, -0.51, 1.01], [-0.99, 0.99, -1.49, 1.49]],
        ],
        dtype=torch.float32,
    )
    scale, hidden_qcfs = calibrate_signed_qcfs(hidden_dense, time_steps=16)
    hidden_if, pos_node, neg_node = build_signed_if_proxy(
        hidden_dense, scale, time_steps=16
    )

    assert scale.shape == (4,)
    assert torch.isfinite(scale).all()
    assert (scale > 0).all()
    assert torch.allclose(hidden_if, hidden_qcfs, atol=1e-5, rtol=1e-5)
    assert isinstance(pos_node, neuron.ActivationAwareIFNode)
    assert isinstance(neg_node, neuron.ActivationAwareIFNode)


def test_signed_if_proxy_matches_qcfs_at_rounding_boundaries():
    from benchmark.snn_llm.gpt2_conversion.mlp_ann2snn_slice import (
        build_signed_if_proxy,
    )

    scale = torch.tensor([0.25])
    hidden_dense = torch.tensor([[[-0.125], [0.125], [-0.625], [0.625]]])

    hidden_if, _, _ = build_signed_if_proxy(hidden_dense, scale, time_steps=16)
    expected = torch.round(hidden_dense / scale).clamp(-16, 16) * scale

    assert torch.equal(hidden_if, expected)


def test_positive_only_proxy_is_not_supported():
    from benchmark.snn_llm.gpt2_conversion.mlp_ann2snn_slice import (
        SIGNED_ENCODING,
    )

    assert SIGNED_ENCODING != "positive_only"
    assert "signed" in SIGNED_ENCODING


def test_recipe_uses_existing_module_converter_seam():
    from benchmark.snn_llm.gpt2_conversion import mlp_ann2snn_slice as runner
    from spikingjelly.activation_based.ann2snn import ModuleConverter
    from spikingjelly.activation_based.ann2snn.recipes import ModuleConversionRecipe

    assert issubclass(runner.GPT2MLPAnn2SNNRecipe, ModuleConversionRecipe)
    assert runner.ModuleConverter is ModuleConverter


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
        (root / filename).write_bytes(filename.encode())


def test_revision_mismatch_is_rejected_before_model_validation(tmp_path: Path):
    completed = _run_cli(
        "--model-root",
        str(tmp_path / "missing-model"),
        "--output-dir",
        str(tmp_path / "report"),
        "--device",
        "cpu",
        "--source-revision",
        "wrong-revision",
    )

    assert completed.returncode != 0
    assert "source-revision" in completed.stderr
    assert "607a30d783dfa663caf39e06633721c8d4cfcd7e" in completed.stderr
    assert "does not exist" not in completed.stderr


def test_non_positive_time_steps_fail_before_model_loading(tmp_path: Path):
    _populate_model_root(tmp_path / "model")
    completed = _run_cli(
        "--model-root",
        str(tmp_path / "model"),
        "--output-dir",
        str(tmp_path / "report"),
        "--device",
        "cpu",
        "--source-revision",
        _REVISION,
        "--time-steps",
        "0",
    )

    assert completed.returncode != 0
    assert "time-steps" in completed.stderr
    assert not (tmp_path / "report" / "report.json").exists()


def test_missing_model_files_are_listed(tmp_path: Path):
    model_root = tmp_path / "model"
    model_root.mkdir()
    completed = _run_cli(
        "--model-root",
        str(model_root),
        "--output-dir",
        str(tmp_path / "report"),
        "--device",
        "cpu",
        "--source-revision",
        _REVISION,
    )

    assert completed.returncode != 0
    for filename in (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ):
        assert filename in completed.stderr


def test_missing_transformers_has_uv_install_message():
    from benchmark.snn_llm.gpt2_conversion import mlp_ann2snn_slice as runner

    def failing_loader():
        raise ImportError("No module named transformers")

    with pytest.raises(ImportError, match="uv pip install transformers"):
        runner._import_huggingface(failing_loader)


def test_block_index_error_is_clear():
    from benchmark.snn_llm.gpt2_conversion import mlp_ann2snn_slice as runner

    class Model:
        transformer = type("Transformer", (), {"h": [object()]})()

    with pytest.raises(ValueError, match="block-index"):
        runner._get_block(Model(), 2)
    with pytest.raises(ValueError, match="block-index"):
        runner._get_block(Model(), -1)


def test_hugging_face_loader_is_local_only(monkeypatch, tmp_path: Path):
    from benchmark.snn_llm.gpt2_conversion import mlp_ann2snn_slice as runner
    from benchmark.snn_llm.gpt2_conversion.conversion_contract import build_model_paths

    calls = []

    class Tokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

    class Model(torch.nn.Module):
        def eval(self):
            return self

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("tokenizer", path, kwargs))
            return Tokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("model", path, kwargs))
            return Model()

    fake_transformers = type(
        "Transformers",
        (),
        {
            "AutoTokenizer": AutoTokenizer,
            "AutoModelForCausalLM": AutoModelForCausalLM,
        },
    )
    monkeypatch.setattr(runner, "_import_huggingface", lambda loader: fake_transformers)
    root = tmp_path / "model"
    _populate_model_root(root)

    runner._load_gpt2(build_model_paths(root), device="cpu")

    assert [call[0] for call in calls] == ["tokenizer", "model"]
    assert all(call[2]["local_files_only"] is True for call in calls)


def test_report_refuses_overwrite_and_preserves_original(tmp_path: Path):
    from benchmark.snn_llm.gpt2_conversion import mlp_ann2snn_slice as runner

    output_dir = tmp_path / "report"
    output_dir.mkdir()
    target = output_dir / "report.json"
    original = b"original report\n"
    target.write_bytes(original)

    with pytest.raises(SystemExit, match="overwrite"):
        runner._write_report(output_dir, {"kind": "replacement"})

    assert target.read_bytes() == original


def test_report_contract_is_exact_and_json_rejects_non_finite(tmp_path: Path):
    from benchmark.snn_llm.gpt2_conversion import mlp_ann2snn_slice as runner
    from benchmark.snn_llm.gpt2_conversion.conversion_contract import build_model_paths

    root = tmp_path / "model"
    _populate_model_root(root)
    report = runner.build_slice_report(
        paths=build_model_paths(root),
        block_index=0,
        time_steps=16,
        max_samples=4,
        metrics={
            "dense_loss": 1.25,
            "hidden_qcfs_relative_l2": 0.1,
            "mlp_qcfs_relative_l2": 0.1,
            "hidden_if_vs_qcfs_relative_l2": 0.0,
            "mlp_if_relative_l2": 0.1,
            "positive_spike_rate": 0.2,
            "negative_spike_rate": 0.2,
        },
        metadata={
            "prompts": ["prompt"],
            "hidden_shape": [4, 64, 3072],
            "mlp_shape": [4, 64, 768],
            "scale_shape": [3072],
        },
        environment={"host": "test", "device": "cpu"},
    )

    assert set(report) == {
        "slice_schema_version",
        "kind",
        "model",
        "environment",
        "input",
        "slice",
        "calibration",
        "spikingjelly_components",
        "metrics",
        "files",
        "unsupported",
    }
    assert report["kind"] == "gpt2-mlp-ann2snn-signed-if-slice"
    assert report["slice"]["target"] == "transformer.h.0.mlp"
    assert report["slice"]["training"] == "none"
    assert report["calibration"]["time_steps"] == 16
    assert report["calibration"]["scale_shape"] == [3072]
    assert report["spikingjelly_components"]["neuron"] == "ActivationAwareIFNode"
    assert report["spikingjelly_components"]["converter_style"] is True
    assert report["spikingjelly_components"]["public_api_added"] is False
    assert report["spikingjelly_components"]["if_reconstruction"] == (
        "qcfs_count_replay_with_activation_aware_if"
    )
    assert report["metrics"]["dense_loss"] == 1.25
    assert set(report["files"]) == {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    }
    assert all(
        forbidden not in str(report).lower()
        for forbidden in (
            "fine_tuned_gpt2",
            "full_fas_result",
            "full_snn_gpt2",
            "converted_ppl",
            "generation_result",
            "cache_result",
        )
    )
    with pytest.raises(ValueError):
        runner.build_slice_report(
            paths=build_model_paths(root),
            block_index=0,
            time_steps=16,
            max_samples=4,
            metrics={"bad": float("nan")},
            metadata={
                "prompts": ["prompt"],
                "hidden_shape": [1, 1, 1],
                "mlp_shape": [1, 1, 1],
                "scale_shape": [1],
            },
            environment={"host": "test", "device": "cpu"},
        )


def test_cuda_request_without_cuda_fails_without_report(
    monkeypatch, tmp_path: Path, capsys
):
    from benchmark.snn_llm.gpt2_conversion import mlp_ann2snn_slice as runner

    root = tmp_path / "model"
    _populate_model_root(root)
    output_dir = tmp_path / "report"
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)

    result = runner.main(
        [
            "--model-root",
            str(root),
            "--output-dir",
            str(output_dir),
            "--device",
            "cuda",
            "--source-revision",
            _REVISION,
        ]
    )

    assert result != 0
    assert "CUDA" in capsys.readouterr().err
    assert not (output_dir / "report.json").exists()


def test_compute_slice_uses_ln2_activation_and_produces_finite_metrics(
    monkeypatch, tmp_path: Path
):
    from benchmark.snn_llm.gpt2_conversion import mlp_ann2snn_slice as runner
    from benchmark.snn_llm.gpt2_conversion.conversion_contract import build_model_paths

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

        def __call__(self, prompts, **kwargs):
            assert kwargs["padding"] == "max_length"
            assert kwargs["max_length"] == 64
            return {
                "input_ids": torch.tensor(
                    [[0, 1, 2], [1, 2, 3]][: len(prompts)], dtype=torch.long
                ),
                "attention_mask": torch.ones(len(prompts), 3, dtype=torch.long),
            }

    class FakeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc = torch.nn.Linear(4, 8)
            self.act = torch.nn.GELU()
            self.c_proj = torch.nn.Linear(8, 4)

    class FakeBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_2 = torch.nn.LayerNorm(4)
            self.mlp = FakeMLP()

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.h = torch.nn.ModuleList([FakeBlock()])
            self.config = type(
                "Config",
                (),
                {"vocab_size": 4, "n_positions": 64, "hidden_size": 4},
            )()

        def forward(self, input_ids, attention_mask=None):
            del attention_mask
            one_hot = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
            self.transformer.h[0].ln_2(one_hot)
            logits = torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                4,
                dtype=torch.float32,
            )
            logits[..., 0] = 0.5
            return type("Output", (), {"logits": logits})()

    fake_transformers = type(
        "Transformers",
        (),
        {
            "AutoTokenizer": type(
                "AutoTokenizer",
                (),
                {
                    "from_pretrained": staticmethod(
                        lambda *args, **kwargs: FakeTokenizer()
                    )
                },
            ),
            "AutoModelForCausalLM": type(
                "AutoModelForCausalLM",
                (),
                {"from_pretrained": staticmethod(lambda *args, **kwargs: FakeModel())},
            ),
        },
    )
    monkeypatch.setattr(runner, "_import_huggingface", lambda loader: fake_transformers)
    root = tmp_path / "model"
    _populate_model_root(root)

    metrics, metadata = runner.compute_mlp_slice(
        paths=build_model_paths(root),
        device="cpu",
        block_index=0,
        time_steps=16,
        max_samples=2,
    )

    assert metadata["hidden_shape"] == [2, 3, 8]
    assert metadata["scale_shape"] == [8]
    assert math.isfinite(metrics["dense_loss"])
    assert metrics["hidden_if_vs_qcfs_relative_l2"] <= 1e-5
    assert 0.0 < metrics["positive_spike_rate"] < 1.0
    assert 0.0 < metrics["negative_spike_rate"] < 1.0
    assert all(
        isinstance(value, (float, int)) and math.isfinite(float(value))
        for value in metrics.values()
    )
