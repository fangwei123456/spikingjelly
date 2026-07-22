import json

import pytest
import torch

from benchmark.snn_llm import _reporting
from benchmark.snn_llm.qwen_conversion import scaleout_smoke as runner
from spikingjelly.activation_based.ann2snn import Qwen2SNNCalibration, Qwen2SNNConfig


def test_scaleout_artifact_lock_contains_complete_model_matrix():
    lock = runner._load_lock()

    assert set(lock["models"]) == {"0.5b", "1.5b", "3b"}
    assert [lock["models"][key]["layer_count"] for key in ("0.5b", "1.5b", "3b")] == [
        24,
        28,
        36,
    ]
    assert lock["models"]["3b"]["license"] == "Qwen-Research"
    assert set(lock["datasets"]) == {
        "arc",
        "hellaswag",
        "lambada_openai",
        "piqa",
        "wikitext",
        "winogrande",
    }
    assert 512 in runner.TIME_STEP_CHOICES


def test_scaleout_invalid_levels_fail_before_cuda_or_model_loading(tmp_path, capsys):
    exit_code = runner.main(
        [
            "--model-key",
            "0.5b",
            "--model-root",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
            "--device",
            "cuda",
            "--worktree-revision",
            "revision",
            "--time-steps",
            "16",
            "--calibration-levels",
            "32",
            "--calibration-quantile",
            "1.0",
            "--neuron-backend",
            "torch",
        ]
    )

    assert exit_code == 2
    assert "must not exceed" in capsys.readouterr().err


def test_scaleout_report_write_is_atomic_and_refuses_overwrite(tmp_path):
    report = {"schema_version": 1, "metric": 1.0}

    target = runner._write_report(tmp_path, report)

    assert json.loads(target.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        runner._write_report(tmp_path, report)


def test_scaleout_report_write_does_not_clobber_competing_target(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    original_link = _reporting.os.link

    def competing_link(source, destination):
        target.write_text("competitor\n", encoding="utf-8")
        return original_link(source, destination)

    monkeypatch.setattr(_reporting.os, "link", competing_link)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        runner._write_report(tmp_path, {"schema_version": 1})

    assert target.read_text(encoding="utf-8") == "competitor\n"
    assert not tuple(tmp_path.glob(".report.json.*.tmp"))


def test_scaleout_report_rejects_nonfinite_top1_agreement():
    report = {
        "metrics": {
            "exact_logits_relative_l2": 0.0,
            "exact_loss_delta": 0.0,
            "signed_logits_relative_l2": 0.0,
            "signed_loss_delta": 0.0,
            "reset_replay_max_abs_error": 0.0,
            "exact_cached_decode_max_relative_l2": 0.0,
            "signed_cached_decode_max_relative_l2": 0.0,
            "signed_top1_agreement": float("nan"),
        },
        "model": {"layer_count": 1},
        "conversion": {
            "structure": runner._expected_structure(1),
            "temporal_layout": "[T,B,S,H]",
            "execution_schedule": "layerwise_offline_multistep",
        },
    }

    with pytest.raises(ValueError, match="top1"):
        runner._validate_report(report)


def test_scaleout_calibration_artifact_must_match_configuration(tmp_path):
    calibration = Qwen2SNNCalibration(
        input_scale=torch.ones(2),
        layer_scales=(
            {
                "query": torch.ones(2),
                "key": torch.ones(2),
                "value": torch.ones(2),
                "mlp": torch.ones(4),
            },
        ),
        time_steps=160,
        calibration_levels=16,
        calibration_quantile=0.999,
        calibration_reservoir_size=4096,
        calibration_seed=20260719,
        valid_token_count=512,
    )
    path = tmp_path / "calibration.pt"
    torch.save(calibration.state_dict(), path)

    loaded, digest = runner._load_calibration_artifact(
        path,
        Qwen2SNNConfig(
            time_steps=160,
            calibration_levels=16,
            calibration_quantile=0.999,
        ),
    )

    assert loaded.valid_token_count == 512
    assert len(digest) == 64
    with pytest.raises(ValueError, match="time_steps"):
        runner._load_calibration_artifact(
            path,
            Qwen2SNNConfig(
                time_steps=128,
                calibration_levels=16,
                calibration_quantile=0.999,
            ),
        )
