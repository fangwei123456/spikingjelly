"""Focused tests for the private Phase 6.0 GPT-2 cache equivalence runner.

These tests verify both fake-cache seams and the CLI surface. They never
import Hugging Face, so they run inside the default virtualenv.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.snn_llm.gpt2_conversion.conversion_contract import (
    EXPECTED_REVISION as _REVISION,
)

_RUNNER = (
    _REPO_ROOT / "benchmark" / "snn_llm" / "gpt2_conversion" / "cache_equivalence.py"
)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_RUNNER), *args],
        cwd=_REPO_ROOT,
        env={"PYTHONPATH": str(_REPO_ROOT), **os.environ},
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_help_exposes_only_phase60_controls():
    completed = _run_cli("--help")

    assert completed.returncode == 0
    for option in (
        "--model-root",
        "--output-dir",
        "--device",
        "--source-revision",
        "--max-samples",
        "--max-length",
        "--prefill-length",
        "--decode-steps",
    ):
        assert option in completed.stdout, option
    for forbidden in (
        "--do-sample",
        "--max-new-tokens",
        "fine-tune",
        "ppl",
    ):
        assert forbidden not in completed.stdout.lower(), forbidden


def test_cache_helpers_read_dynamic_cache_layer_summaries():
    from benchmark.snn_llm.gpt2_conversion.cache_equivalence import (
        _cache_class_name,
        _cache_layer_summaries,
        _cache_seq_length,
    )

    class _Layer:
        def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
            self.keys = keys
            self.values = values

    class _DynamicCache:
        def __init__(self) -> None:
            self.layers = [
                _Layer(torch.zeros(2, 4, 3, 5), torch.ones(2, 4, 3, 5)),
            ]

        def get_seq_length(self) -> int:
            return 3

    cache = _DynamicCache()
    assert _cache_seq_length(cache) == 3
    assert _cache_class_name(cache) == "_DynamicCache"
    summaries = _cache_layer_summaries(cache)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["key_shape"] == [2, 4, 3, 5]
    assert summary["value_shape"] == [2, 4, 3, 5]
    assert summary["dtype"] == "torch.float32"


def test_cache_clone_uses_deepcopy_and_does_not_alias_tensors():
    from benchmark.snn_llm.gpt2_conversion.cache_equivalence import _clone_cache

    cache = {"tensors": [torch.arange(4.0)], "metadata": {"tag": "orig"}}
    clone = _clone_cache(cache)

    assert clone is not cache
    assert clone["tensors"][0].data_ptr() != cache["tensors"][0].data_ptr()
    assert torch.equal(clone["tensors"][0], cache["tensors"][0])

    cache["tensors"][0][0] = 99.0
    assert clone["tensors"][0][0].item() == 0.0


def test_cache_snapshot_detects_same_shape_tensor_mutation():
    from benchmark.snn_llm.gpt2_conversion.cache_equivalence import (
        _cache_matches_snapshot,
        _snapshot_cache_tensors,
    )

    class _Layer:
        def __init__(self) -> None:
            self.keys = torch.zeros(2, 1, 3, 2)
            self.values = torch.ones(2, 1, 3, 2)

    cache = type("_Cache", (), {"layers": [_Layer()]})()
    snapshot = _snapshot_cache_tensors(cache)

    cache.layers[0].keys.add_(1.0)

    assert _cache_matches_snapshot(cache, snapshot) is False


def test_cache_reorder_supports_dynamic_cache_and_falls_back_to_batch_select():
    from benchmark.snn_llm.gpt2_conversion.cache_equivalence import _reorder_cache

    class _Layer:
        def __init__(self) -> None:
            self.keys = torch.arange(2 * 1 * 3 * 2, dtype=torch.float32).view(
                2, 1, 3, 2
            )
            self.values = self.keys + 100.0

    class _DynamicCache:
        def __init__(self) -> None:
            self.layers = [_Layer()]

        def get_seq_length(self) -> int:
            return 3

        def reorder_cache(self, indices: torch.Tensor) -> None:
            # transformers 5.x ``DynamicCache.reorder_cache`` mutates the
            # cache in place and returns ``None``; the helper therefore
            # returns the same cache object after invoking reorder.
            for layer in self.layers:
                layer.keys = layer.keys[indices]
                layer.values = layer.values[indices]

        def batch_select_indices(self, indices: torch.Tensor) -> None:
            for layer in self.layers:
                layer.keys = layer.keys[indices]
                layer.values = layer.values[indices]

    cache = _DynamicCache()
    indices = torch.tensor([1, 0])
    reordered = _reorder_cache(cache, indices)

    # The helper returns the cache object whose ``reorder_cache`` mutated
    # the batch axis in place.
    assert reordered is cache
    assert cache.layers[0].keys.shape == (2, 1, 3, 2)


def test_cache_reorder_accepts_returned_cache_object():
    from benchmark.snn_llm.gpt2_conversion.cache_equivalence import _reorder_cache

    class _Layer:
        def __init__(self) -> None:
            self.keys = torch.arange(2 * 1 * 3 * 2, dtype=torch.float32).view(
                2, 1, 3, 2
            )
            self.values = self.keys + 100.0

    class _ReturningCache:
        def __init__(self) -> None:
            self.layers = [_Layer()]

        def reorder_cache(self, indices: torch.Tensor) -> "_ReturningCache":
            reordered = _ReturningCache()
            for source, target in zip(self.layers, reordered.layers):
                target.keys = source.keys[indices]
                target.values = source.values[indices]
            return reordered

    cache = _ReturningCache()
    reordered = _reorder_cache(cache, torch.tensor([1, 0]))

    assert reordered is not cache
    assert torch.allclose(reordered.layers[0].keys[0, 0, 0, 0], torch.tensor(6.0))
    assert torch.allclose(cache.layers[0].keys[0, 0, 0, 0], torch.tensor(0.0))


def test_cache_reorder_falls_back_to_batch_select_when_reorder_missing():
    from benchmark.snn_llm.gpt2_conversion.cache_equivalence import _reorder_cache

    class _Layer:
        def __init__(self) -> None:
            self.keys = torch.arange(2 * 1 * 3 * 2, dtype=torch.float32).view(
                2, 1, 3, 2
            )
            self.values = self.keys + 100.0

    class _CacheNoReorder:
        def __init__(self) -> None:
            self.layers = [_Layer()]

        def get_seq_length(self) -> int:
            return 3

        def batch_select_indices(self, indices: torch.Tensor) -> None:
            for layer in self.layers:
                layer.keys = layer.keys[indices]
                layer.values = layer.values[indices]

    cache = _CacheNoReorder()
    reordered = _reorder_cache(cache, torch.tensor([1, 0]))
    # batch_select_indices reordered the batch axis: row 0 of the reordered
    # tensor is the original row 1, so its first scalar equals 6.0.
    assert torch.allclose(cache.layers[0].keys[0, 0, 0, 0], torch.tensor(6.0))
    assert torch.allclose(cache.layers[0].keys[1, 0, 0, 0], torch.tensor(0.0))
    assert reordered is cache


def test_compute_shifted_ce_loss_matches_phase5_convention():
    from benchmark.snn_llm.gpt2_conversion.cache_equivalence import (
        _compute_shifted_ce_loss,
    )

    logits = torch.zeros(1, 4, 3)
    logits[..., 0] = 5.0
    labels = torch.tensor([[0, 1, 2, 0]])
    loss = _compute_shifted_ce_loss(logits, labels, attention_mask=torch.ones(1, 4))
    # 3 of 4 prediction targets are deterministic, 1 is tied.
    assert math.isfinite(loss)


def test_phase60_does_not_claim_snn_or_qwen_or_generation(tmp_path: Path):
    from benchmark.snn_llm.gpt2_conversion.cache_equivalence import (
        build_cache_report,
    )
    from benchmark.snn_llm.gpt2_conversion.conversion_contract import (
        build_model_paths,
    )

    report_keys = {
        "cache_schema_version",
        "kind",
        "model",
        "environment",
        "input",
        "cache_contract",
        "scenarios",
        "metrics",
        "files",
        "unsupported",
    }
    model_root = tmp_path / "model"
    _populate_model_root(model_root)
    paths = build_model_paths(model_root)
    report = build_cache_report(
        paths=paths,
        max_samples=1,
        max_length=8,
        prefill_length=2,
        decode_steps=2,
        cache_contract={"cache_class": "DynamicCache"},
        metrics={},
        environment={"host": "test", "device": "cpu"},
    )

    assert set(report) == report_keys
    rendered = str(report).lower()
    for forbidden in (
        "fine_tuned_gpt2",
        "full_fas_result",
        "full_snn_gpt2",
        "converted_ppl",
        "generation_result",
        "cache_result",
    ):
        assert forbidden not in rendered, forbidden


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


def test_missing_transformers_raises_helpful_error():
    from benchmark.snn_llm.gpt2_conversion import cache_equivalence as runner

    def failing_loader():
        raise ImportError("No module named transformers")

    with pytest.raises(ImportError, match="uv pip install transformers"):
        runner._import_huggingface(failing_loader)


def test_hugging_face_loader_uses_local_files_only(monkeypatch, tmp_path: Path):
    from benchmark.snn_llm.gpt2_conversion import cache_equivalence as runner
    from benchmark.snn_llm.gpt2_conversion.conversion_contract import build_model_paths

    calls = []

    class Tokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

    class Model(torch.nn.Module):
        def eval(self_inner):
            return self_inner

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

    fake = type(
        "Transformers",
        (),
        {
            "AutoTokenizer": AutoTokenizer,
            "AutoModelForCausalLM": AutoModelForCausalLM,
        },
    )
    monkeypatch.setattr(runner, "_import_huggingface", lambda loader: fake)
    root = tmp_path / "model"
    _populate_model_root(root)
    runner._load_gpt2(build_model_paths(root), device="cpu")
    assert [call[0] for call in calls] == ["tokenizer", "model"]
    assert all(call[2]["local_files_only"] is True for call in calls)


def test_invalid_args_fail_before_model_loading(tmp_path: Path):
    _populate_model_root(tmp_path / "model")
    base = [
        "--model-root",
        str(tmp_path / "model"),
        "--output-dir",
        str(tmp_path / "report"),
        "--device",
        "cpu",
        "--source-revision",
        _REVISION,
    ]
    cases = [
        ("--max-samples", "0", "max-samples"),
        ("--max-samples", "1", "max-samples"),
        ("--max-length", "1", "max-length"),
        ("--prefill-length", "0", "prefill-length"),
        ("--prefill-length", "1", "prefill-length"),
        ("--decode-steps", "0", "decode-steps"),
        ("--prefill-length", "100", "prefill-length"),
    ]
    for flag, value, fragment in cases:
        completed = _run_cli(*base, flag, value)
        assert completed.returncode != 0, (flag, value)
        assert fragment in completed.stderr, (flag, value, completed.stderr)
        assert not (tmp_path / "report" / "report.json").exists(), (flag, value)


def test_report_overwrite_is_rejected(tmp_path: Path):
    from benchmark.snn_llm.gpt2_conversion import cache_equivalence as runner

    output_dir = tmp_path / "report"
    output_dir.mkdir()
    target = output_dir / "report.json"
    original = b"original cache report\n"
    target.write_bytes(original)

    with pytest.raises(SystemExit, match="overwrite"):
        runner._write_report(output_dir, {"kind": "replacement"})

    assert target.read_bytes() == original


def test_cuda_request_without_cuda_fails_without_report(
    monkeypatch, tmp_path: Path, capsys
):
    from benchmark.snn_llm.gpt2_conversion import cache_equivalence as runner

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


def _build_fake_model(
    scale: float = 0.5, *, clone_aliases: bool = False
) -> torch.nn.Module:
    """Tiny linear model that emulates a single-step GPT-2 layer.

    The fake caches ``past_key_values`` as a list of ``(K, V)`` tuples and
    reuses them on subsequent calls, mirroring Hugging Face's pre/post
    transformers 4.x cache behavior.  It is intentionally minimal so tests
    can compare against exact closed-form expectations.
    """

    class _FakeLayer:
        __slots__ = ("keys", "values")

        def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
            self.keys = keys
            self.values = values

    class FakeDynamicCache:
        def __init__(self) -> None:
            self.layers: List = []

        def get_seq_length(self) -> int:
            if not self.layers:
                return 0
            return int(self.layers[0].keys.shape[2])

        def reorder_cache(self, indices: torch.Tensor) -> None:
            for layer in self.layers:
                layer.keys = layer.keys[indices]
                layer.values = layer.values[indices]

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(16, 4)
            self.proj = torch.nn.Linear(4, 3, bias=False)
            self.clone_aliases = clone_aliases
            with torch.no_grad():
                self.proj.weight.copy_(torch.ones(3, 4))

        def forward(
            self,
            input_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=None,
            return_dict=None,
        ):
            del attention_mask
            cache = past_key_values
            if cache is None:
                cache = FakeDynamicCache()
            if self.clone_aliases and past_key_values is not None:
                for layer in cache.layers:
                    layer.keys = torch.cat([layer.keys, layer.keys[:, :, -1:]], dim=2)
                    layer.values = torch.cat(
                        [layer.values, layer.values[:, :, -1:]], dim=2
                    )
            prev_seq = cache.get_seq_length()
            new_ids = input_ids
            appended = []
            for _ in new_ids[0]:
                head_dim = 1
                batch_size = new_ids.shape[0]
                key = torch.full(
                    (batch_size, 1, prev_seq + len(appended) + 1, head_dim),
                    scale,
                )
                value = torch.full(
                    (batch_size, 1, prev_seq + len(appended) + 1, head_dim),
                    0.25 * scale,
                )
                appended.append((key, value))
            # Each new token already produces a K/V tensor that spans the
            # full history (cache + new token). The last layer's tensors
            # simply become that updated tensor; no concatenation needed.
            if cache.layers:
                last_layer = cache.layers[-1]
                last_layer.keys = appended[-1][0]
                last_layer.values = appended[-1][1]
            else:
                cache.layers.append(_FakeLayer(appended[-1][0], appended[-1][1]))
            hidden = self.embed(new_ids)
            logits = self.proj(hidden)
            return type("Output", (), {"logits": logits, "past_key_values": cache})()

    return FakeModel()


def test_compute_scenarios_records_required_metrics(tmp_path: Path):
    from benchmark.snn_llm.gpt2_conversion import cache_equivalence as runner

    root = tmp_path / "model"
    _populate_model_root(root)
    model = _build_fake_model()

    input_ids = torch.tensor([[0, 1, 1, 0], [2, 1, 1, 2]])
    attention_mask = torch.ones_like(input_ids)
    contract, metrics = runner.compute_scenarios(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefill_length=2,
        decode_steps=2,
    )

    assert contract["cache_class"] == "FakeDynamicCache"
    assert contract["mutation"] == "decode_mutates_cache_in_place"
    assert contract["clone_required_for_branching"] is True
    assert contract["snn_state_separate"] is True
    assert contract["reset_supported"] is True
    assert contract["clone_supported"] is True
    assert contract["reorder_supported"] is True
    assert contract["serialized_cache"] is False
    assert contract["layer_summaries"] == [
        {
            "key_shape": [2, 1, 4, 1],
            "value_shape": [2, 1, 4, 1],
            "dtype": "torch.float32",
            "device": "cpu",
        }
    ]

    assert metrics["cache_layer_count"] == 1
    assert metrics["cache_seq_length_after_prefill"] == 2
    assert metrics["decode_token_count"] == 4
    assert metrics["segmented_cache_seq_length_after_decode"] == 4
    assert metrics["reorder_max_abs_error"] <= 1e-6
    assert metrics["reset_prefill_max_abs_error"] <= 1e-6
    assert metrics["reset_prefill_mean_abs_error"] <= 1e-6
    assert math.isfinite(metrics["dense_loss"])
    for name, value in metrics.items():
        if name == "clone_independence_preserved":
            continue
        assert isinstance(value, (float, int)), name
        assert math.isfinite(value), name
    assert metrics["clone_independence_preserved"] is True


def test_compute_scenarios_rejects_non_independent_clone(monkeypatch):
    from benchmark.snn_llm.gpt2_conversion import cache_equivalence as runner

    model = _build_fake_model(clone_aliases=True)
    input_ids = torch.tensor([[0, 1, 1, 0], [2, 1, 1, 2]])
    attention_mask = torch.ones_like(input_ids)

    monkeypatch.setattr(runner, "_clone_cache", lambda cache: cache)

    with pytest.raises(ValueError, match="clone is not independent"):
        runner.compute_scenarios(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefill_length=2,
            decode_steps=2,
        )
