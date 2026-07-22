import math
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import torch

from benchmark.snn_llm import _spikegpt_pilot as spikegpt_train_pilot
from benchmark.snn_llm import _spikegpt_training as spikegpt_train_smoke


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "benchmark" / "snn_llm" / "spikegpt_train_pilot.py"


def test_bpc_converts_cross_entropy_from_nats():
    assert spikegpt_train_pilot._bpc_from_nll(math.log(2.0)) == pytest.approx(1.0)


def test_decode_splits_returns_all_three_byte_segments():
    raw = b"abcdeXYZW123"
    training, validation, test = spikegpt_train_pilot._decode_splits(
        raw, training_bytes=5, validation_bytes=4
    )

    assert training == "abcde"
    assert validation == "XYZW"
    assert test == "123"


def test_decode_splits_keeps_raw_byte_boundaries_before_utf8_decode():
    raw = "abcé𝄞".encode("utf-8")
    training, validation, test = spikegpt_train_pilot._decode_splits(
        raw,
        training_bytes=len("abc".encode("utf-8")),
        validation_bytes=len("é".encode("utf-8")),
    )

    assert training == "abc"
    assert validation == "é"
    assert test == "𝄞"
    assert len(test) == 1
    assert len("𝄞".encode("utf-8")) == 4


def test_build_full_vocabulary_unions_all_inputs_deterministically():
    vocab = spikegpt_train_pilot._build_full_vocabulary("abc", "XY", "z")

    assert vocab == ("X", "Y", "a", "b", "c", "z")


def test_checkpoint_accepts_only_the_declared_pilot_config(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    config = {"kind": "spikegpt-enwik8-pilot-v1", "context_length": 1024}
    spikegpt_train_smoke._save_checkpoint(
        checkpoint,
        model,
        optimizer,
        completed_steps=3,
        next_batch_index=3,
        source_revision="revision",
        vocabulary=("a", "b"),
        training_config=config,
        extra_state={"training_wall_seconds": 12.5},
    )

    restored = spikegpt_train_smoke._load_checkpoint(
        checkpoint,
        model,
        optimizer,
        torch.device("cpu"),
        training_config=config,
    )

    assert restored["completed_steps"] == 3
    assert restored["extra_state"] == {"training_wall_seconds": 12.5}
    with pytest.raises(RuntimeError, match="configuration does not match"):
        spikegpt_train_smoke._load_checkpoint(
            checkpoint,
            model,
            optimizer,
            torch.device("cpu"),
            training_config={**config, "context_length": 32},
        )


def test_numpy_rng_checkpoint_state_is_portable_and_exact():
    np.random.seed(123)
    state = spikegpt_train_smoke._rng_state()
    expected = np.random.randint(0, 2**31, size=8)

    assert state["numpy"]["keys"].dtype == torch.int64
    spikegpt_train_smoke._restore_rng_state(state)
    actual = np.random.randint(0, 2**31, size=8)

    np.testing.assert_array_equal(actual, expected)


def test_help_exposes_only_pilot_resource_and_resume_controls():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--spikegpt-root" in completed.stdout
    assert "--data" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--resume-checkpoint" in completed.stdout
    assert "--max-steps" in completed.stdout
    assert "--max-minutes" in completed.stdout
    assert "--checkpoint-every" in completed.stdout
    assert "RWKV_HEAD_QK_DIM" not in completed.stdout


def test_cli_rejects_nonpositive_resource_limits_before_loading_cuda(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--spikegpt-root",
            str(tmp_path),
            "--data",
            str(tmp_path / "enwik8"),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-steps",
            "0",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "must be positive" in completed.stderr
    assert "CUDA" not in completed.stderr

    nonfinite = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--spikegpt-root",
            str(tmp_path),
            "--data",
            str(tmp_path / "enwik8"),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-minutes",
            "nan",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert nonfinite.returncode == 1
    assert "must be finite" in nonfinite.stderr
    assert "CUDA" not in nonfinite.stderr


def test_training_direction_compares_equal_start_and_end_windows():
    improved, start_bpc, end_bpc = spikegpt_train_pilot._training_direction(
        [4.0, 3.0, 2.0, 1.0], window_size=2
    )
    flat, _, _ = spikegpt_train_pilot._training_direction(
        [2.0, 2.0, 2.0, 2.0], window_size=2
    )

    assert improved is True
    assert start_bpc == pytest.approx(3.5 / math.log(2.0))
    assert end_bpc == pytest.approx(1.5 / math.log(2.0))
    assert flat is False


def test_pilot_config_fingerprints_full_validation_protocol():
    config = spikegpt_train_pilot._training_config()

    assert config["tokenizer"] == "utf8-character-sorted-full-enwik8-v1"
    assert config["vocab_size"] == 6064
    assert config["parameter_count"] == 47_173_632
    assert config["validation_target_tokens"] == 4_982_412
    assert config["validation_batches"] == 4866
    assert "validation_offsets" not in config


def test_validation_windows_cover_all_shifted_targets(monkeypatch):
    monkeypatch.setattr(spikegpt_train_pilot, "CONTEXT_LENGTH", 4)
    windows = spikegpt_train_pilot._validation_windows(11)

    assert windows == ((0, 4), (4, 4), (8, 2))
    assert sum(length for _, length in windows) == 10
    assert windows[-1][1] < 4


class _CountingLossModel(torch.nn.Module):
    """Stand-in pilot model that returns deterministic per-call losses."""

    def __init__(self, losses):
        super().__init__()
        self._iterator = iter(losses)

    def forward(self, inputs, targets):
        return torch.tensor(next(self._iterator), dtype=torch.float32)


def test_weighted_bpc_uses_target_token_weighting(monkeypatch):
    monkeypatch.setattr(spikegpt_train_pilot, "CONTEXT_LENGTH", 4)
    text = "abcdef"
    vocab = ("a", "b", "c", "d", "e", "f")
    windows = spikegpt_train_pilot._validation_windows(len(text))

    assert windows == ((0, 4), (4, 1))

    model = _CountingLossModel([1.0, 3.0])
    bpc = spikegpt_train_pilot._evaluate_bpc(model, text, vocab, torch.device("cpu"))

    expected = (1.0 * 4 + 3.0 * 1) / 5 / math.log(2.0)
    assert bpc == pytest.approx(expected)


def test_full_validation_encodes_train_unseen_characters():
    raw = b"abcXYz"
    training, validation, test_split = spikegpt_train_pilot._decode_splits(
        raw, training_bytes=3, validation_bytes=2
    )
    vocab = spikegpt_train_pilot._build_full_vocabulary(
        training, validation, test_split
    )

    assert "X" in vocab and "Y" in vocab

    model = _CountingLossModel([2.0])
    bpc = spikegpt_train_pilot._evaluate_bpc(
        model, validation, vocab, torch.device("cpu")
    )

    expected = 2.0 / 1 / math.log(2.0)
    assert bpc == pytest.approx(expected)


def test_old_train_vocab_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())

    old_config = {
        "kind": "spikegpt-enwik8-context1024-pilot-v1",
        "tokenizer": "utf8-character-sorted-train-v1",
        "validation_batches": 4,
        "validation_offsets": (0, 1245347, 2490694, 3736041),
        "vocab_size": 5458,
        "parameter_count": 46_553_088,
    }
    spikegpt_train_smoke._save_checkpoint(
        checkpoint,
        model,
        optimizer,
        completed_steps=20,
        next_batch_index=20,
        source_revision="revision",
        vocabulary=("a", "b"),
        training_config=old_config,
    )

    new_config = spikegpt_train_pilot._training_config()
    with pytest.raises(RuntimeError, match="configuration does not match"):
        spikegpt_train_smoke._load_checkpoint(
            checkpoint,
            model,
            optimizer,
            torch.device("cpu"),
            training_config=new_config,
        )


def test_training_metrics_reject_nonfinite_membrane_state():
    assert spikegpt_train_smoke._training_metrics_are_finite(
        loss=1.0,
        gradient_l2=2.0,
        spike_rate=0.5,
        membrane_abs_mean=3.0,
        membrane_abs_max=4.0,
    )
    assert not spikegpt_train_smoke._training_metrics_are_finite(
        loss=1.0,
        gradient_l2=2.0,
        spike_rate=0.5,
        membrane_abs_mean=float("nan"),
        membrane_abs_max=4.0,
    )


def test_checkpoint_rejects_nonmapping_extra_state(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    spikegpt_train_smoke._save_checkpoint(
        checkpoint,
        model,
        optimizer,
        completed_steps=0,
        next_batch_index=0,
        source_revision="revision",
        vocabulary=("a", "b"),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["extra_state"] = []
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="extra state must be a mapping"):
        spikegpt_train_smoke._load_checkpoint(
            checkpoint, model, optimizer, torch.device("cpu")
        )
