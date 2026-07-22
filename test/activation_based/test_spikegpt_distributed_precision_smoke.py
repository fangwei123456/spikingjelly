import subprocess
import sys
from pathlib import Path

import pytest
import torch

from benchmark.snn_llm import spikegpt_distributed_precision_smoke as phase4
from benchmark.snn_llm import _spikegpt_training as spikegpt_train_smoke


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = (
    _REPO_ROOT / "benchmark" / "snn_llm" / "spikegpt_distributed_precision_smoke.py"
)


def test_help_exposes_only_phase4_controls():
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
    assert "--source-revision" in completed.stdout
    assert "--precision" in completed.stdout
    assert "--max-steps" in completed.stdout
    assert "--checkpoint-every" in completed.stdout
    assert "--resume-checkpoint" in completed.stdout
    assert "--max-minutes" not in completed.stdout


def test_invalid_precision_is_rejected_before_cuda(tmp_path):
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
            "--source-revision",
            "revision",
            "--precision",
            "bad",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "invalid choice" in completed.stderr
    assert "CUDA" not in completed.stderr


def test_distributed_training_requires_world_size_two():
    with pytest.raises(RuntimeError, match="world size >= 2"):
        phase4._require_distributed_training_world(world_size=1, precision="fp32")
    phase4._require_distributed_training_world(world_size=1, precision="fp8-probe")


def test_training_config_fingerprints_phase4_protocol():
    config = phase4._training_config("bf16", world_size=2)

    assert config["kind"] == "spikegpt-enwik8-phase4-distributed-precision-v1"
    assert config["precision"] == "bf16"
    assert config["world_size"] == 2
    assert config["tokenizer"] == "utf8-character-sorted-full-enwik8-v1"
    assert config["vocab_size"] == 6064
    assert config["parameter_count"] == 47_173_632
    assert config["validation_metric"] == "validation_sample_bpc"


def test_report_contains_required_phase4_sections():
    report = phase4._build_report(
        source_revision="revision",
        precision_requested="fp32",
        precision_effective="fp32",
        precision_report={"policy": {"name": "fp32"}},
        distributed_report={"world_size": 2, "mode": "dp"},
        experiment={"vocab_size": 6064},
        ranks=[{"rank": 0}, {"rank": 1}],
        metrics={"training_improved": True},
        checkpoint={"sha256": "digest"},
        gpu_before=["0,A100,90 %,100 MiB,81920 MiB"],
    )

    assert set(report) == {
        "source",
        "environment",
        "distributed",
        "precision",
        "experiment",
        "ranks",
        "metrics",
        "checkpoint",
        "gpu_before",
    }
    assert report["metrics"]["performance"] == "shared_gpu_smoke_measurement"


def test_rank_zero_only_writes_report_and_checkpoint(tmp_path):
    report = {"ok": True}
    report_payload = {"path": "checkpoint.pt"}

    assert phase4._rank_zero_paths(tmp_path, rank=0) == (
        tmp_path / "checkpoint.pt",
        tmp_path / "report.json",
    )
    assert phase4._rank_zero_paths(tmp_path, rank=1) == (None, None)
    phase4._write_rank_zero_json(tmp_path / "report.json", report, rank=1)
    phase4._write_rank_zero_json(tmp_path / "report.json", report_payload, rank=0)

    assert (tmp_path / "report.json").read_text().startswith("{")


def test_rank_zero_refuses_existing_outputs(tmp_path):
    (tmp_path / "report.json").write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        phase4._rank_zero_paths(tmp_path, rank=0)


def test_fp8_probe_report_refuses_existing_outputs(tmp_path):
    assert phase4._rank_zero_report_path(tmp_path, rank=1) is None
    assert phase4._rank_zero_report_path(tmp_path, rank=0) == tmp_path / "report.json"

    (tmp_path / "report.json").write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        phase4._rank_zero_report_path(tmp_path, rank=0)


def test_already_ddp_model_is_rejected_before_wrapping():
    class FakeDdp(torch.nn.Module):
        pass

    phase4._ensure_not_ddp(torch.nn.Linear(1, 1), ddp_type=FakeDdp)
    with pytest.raises(RuntimeError, match="Double DDP"):
        phase4._ensure_not_ddp(FakeDdp(), ddp_type=FakeDdp)


def test_external_scaler_and_precision_scaler_are_rejected():
    class Artifacts:
        scaler = object()

    phase4._ensure_single_scaler(Artifacts(), external_scaler=None)
    with pytest.raises(RuntimeError, match="Double scaler"):
        phase4._ensure_single_scaler(Artifacts(), external_scaler=object())


def test_source_checkpoint_identity_detects_changes(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"first")
    identity = phase4._checkpoint_identity(checkpoint)
    phase4._assert_checkpoint_unchanged(checkpoint, identity)

    checkpoint.write_bytes(b"second")
    with pytest.raises(RuntimeError, match="source checkpoint changed"):
        phase4._assert_checkpoint_unchanged(checkpoint, identity)


def test_current_lif_provenance_rejects_vendored_spikingjelly():
    class Lif:
        __module__ = "src.spikingjelly.clock_driven.neuron"

    with pytest.raises(RuntimeError, match="vendored SpikingJelly"):
        phase4._validate_lif_provenance([Lif()])


def test_mixed_precision_lif_wrapper_keeps_lif_math_float32():
    class RecordingLif(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_dtype = None
            self.v = 0.0
            self.v_seq = torch.zeros(1)

        def forward(self, x):
            self.seen_dtype = x.dtype
            return torch.ones_like(x)

    lif = RecordingLif()
    wrapped = phase4._AutocastFp32LIF(lif)
    x = torch.zeros(2, dtype=torch.bfloat16)

    y = wrapped(x)

    assert lif.seen_dtype == torch.float32
    assert y.dtype == torch.bfloat16
    assert wrapped.v == lif.v
    assert torch.equal(wrapped.v_seq, lif.v_seq)


def test_validation_sample_metric_name_is_not_full_validation():
    assert phase4.VALIDATION_SAMPLE_METRIC == "validation_sample_bpc"
    assert "full" not in phase4.VALIDATION_SAMPLE_METRIC


def test_resume_config_mismatch_is_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    old_config = phase4._training_config("fp32", world_size=2)
    spikegpt_train_smoke._save_checkpoint(
        checkpoint,
        model,
        optimizer,
        completed_steps=1,
        next_batch_index=1,
        source_revision="revision",
        vocabulary=("a", "b"),
        training_config=old_config,
    )

    with pytest.raises(RuntimeError, match="configuration does not match"):
        spikegpt_train_smoke._load_checkpoint(
            checkpoint,
            model,
            optimizer,
            torch.device("cpu"),
            training_config=phase4._training_config("bf16", world_size=2),
        )
