import subprocess
import sys
from pathlib import Path

import pytest
import torch

from benchmark.snn_llm import spikegpt_checkpoint_compare
from benchmark.snn_llm._spikegpt_author import SPIKEGPT_REVISION
from spikingjelly.activation_based import neuron


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "benchmark" / "snn_llm" / "spikegpt_checkpoint_compare.py"


def test_compare_runs_rejects_logit_or_token_mismatch():
    reference = {
        "logits": [torch.tensor([0.0, 1.0])],
        "tokens": [1],
    }

    logit_mismatch = spikegpt_checkpoint_compare._compare_runs(
        reference,
        {
            "logits": [torch.tensor([0.0, 1.1])],
            "tokens": [1],
        },
    )
    token_mismatch = spikegpt_checkpoint_compare._compare_runs(
        reference,
        {
            "logits": [torch.tensor([0.0, 1.0])],
            "tokens": [0],
        },
    )

    assert logit_mismatch["logits_close"] is False
    assert logit_mismatch["tokens_match"] is True
    assert token_mismatch["logits_close"] is True
    assert token_mismatch["tokens_match"] is False


def test_current_lif_matches_author_inference_defaults():
    lif = spikegpt_checkpoint_compare._make_current_lif()

    assert isinstance(lif, neuron.LIFNode)
    assert lif.tau == 2.0
    assert lif.decay_input is True
    assert lif.v_threshold == 1.0
    assert lif.v_reset == 0.0
    assert lif.detach_reset is False
    assert lif.step_mode == "s"
    assert lif.backend == "torch"
    assert lif.training is False


def test_help_does_not_load_checkpoint_or_author_code():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--spikegpt-root" in completed.stdout
    assert "--checkpoint" in completed.stdout
    assert "RWKV_HEAD_QK_DIM" not in completed.stdout


def test_artifact_hashes_are_enforced(monkeypatch, tmp_path):
    source_root = tmp_path / "SpikeGPT"
    source_dir = source_root / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "model.py").write_text("", encoding="utf-8")
    (source_dir / "model_run.py").write_text("", encoding="utf-8")
    tokenizer = source_root / "20B_tokenizer.json"
    tokenizer.write_bytes(b"tokenizer")
    checkpoint = tmp_path / "SpikeGPT-216M.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        spikegpt_checkpoint_compare,
        "TOKENIZER_SHA256",
        spikegpt_checkpoint_compare._sha256(tokenizer),
    )
    monkeypatch.setattr(
        spikegpt_checkpoint_compare,
        "CHECKPOINT_SHA256",
        spikegpt_checkpoint_compare._sha256(checkpoint),
    )

    root, verified_tokenizer, revision = spikegpt_checkpoint_compare._verify_artifacts(
        source_root,
        checkpoint,
        SPIKEGPT_REVISION,
    )

    assert root == source_root.resolve()
    assert verified_tokenizer == tokenizer
    assert revision == SPIKEGPT_REVISION

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Checkpoint SHA-256 mismatch"):
        spikegpt_checkpoint_compare._verify_artifacts(
            source_root,
            checkpoint,
            SPIKEGPT_REVISION,
        )


def test_current_inference_states_are_independent():
    first_state, first_attention_lifs, first_ffn_lifs = (
        spikegpt_checkpoint_compare._make_state(
            spikegpt_checkpoint_compare._make_current_lif,
            torch.device("cpu"),
        )
    )
    second_state, second_attention_lifs, second_ffn_lifs = (
        spikegpt_checkpoint_compare._make_state(
            spikegpt_checkpoint_compare._make_current_lif,
            torch.device("cpu"),
        )
    )

    first_state[0, 0] = 1.0

    assert second_state[0, 0] == 0.0
    assert all(
        left is not right
        for left, right in zip(first_attention_lifs, second_attention_lifs)
    )
    assert all(
        left is not right for left, right in zip(first_ffn_lifs, second_ffn_lifs)
    )


def test_spike_rates_reject_missing_hook_observations(monkeypatch):
    monkeypatch.setattr(spikegpt_checkpoint_compare, "N_LAYER", 1)

    with pytest.raises(RuntimeError, match="missing spike observations"):
        spikegpt_checkpoint_compare._spike_rates([[None, torch.tensor(1.0)]], [[0, 1]])
