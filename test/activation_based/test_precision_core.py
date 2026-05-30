from types import SimpleNamespace

import pytest
import torch

from spikingjelly.activation_based.precision import (
    PrecisionConfig,
    build_capability_report,
    resolve_precision_policy,
    validate_capability,
)


def test_precision_config_from_string():
    cfg = PrecisionConfig.from_any("bf16", default_device="cuda:0")
    assert cfg.mode == "bf16"
    assert cfg.device == "cuda:0"


def test_precision_config_from_dict_aliases_precision_key():
    cfg = PrecisionConfig.from_any({"precision": "fp16", "device": "cuda:0"})
    assert cfg.mode == "fp16"
    assert cfg.device == "cuda:0"


def test_precision_config_from_object_with_precision_fields():
    obj = SimpleNamespace(
        precision="fp8-torchao",
        precision_strict="strict",
        fp8_recipe="auto",
        fp8_report=False,
        device="cuda:0",
    )
    cfg = PrecisionConfig.from_any(obj)
    assert cfg.mode == "fp8-torchao"
    assert cfg.strictness == "strict"
    assert cfg.report is False


def test_precision_config_from_legacy_amp_like_object():
    obj = SimpleNamespace(disable_amp=False, device="cuda:0")
    cfg = PrecisionConfig.from_any(obj)
    assert cfg.mode == "fp16"


def test_precision_config_from_unknown_object_raises():
    with pytest.raises(TypeError):
        PrecisionConfig.from_any(object())


def test_resolve_precision_policy_fp32():
    policy = resolve_precision_policy(PrecisionConfig(mode="fp32"))
    assert policy.describe()["name"] == "fp32"


def test_build_capability_report_cpu_fp32():
    report = build_capability_report(torch.nn.Linear(4, 4), torch.device("cpu"), "fp32")
    assert report["device_type"] == "cpu"
    assert report["can_convert"] is True
    assert report["can_execute"] is True


def test_build_capability_report_mps_classification():
    report = build_capability_report(
        torch.nn.Linear(4, 4),
        torch.device("mps"),
        "fp32",
    )
    assert report["device_type"] == "mps"


def test_validate_capability_rejects_fp16_on_cpu():
    report = build_capability_report(torch.nn.Linear(4, 4), torch.device("cpu"), "fp16")
    with pytest.raises(RuntimeError, match="fp16"):
        validate_capability(report)


def test_build_capability_report_fp8_cpu_cannot_execute():
    report = build_capability_report(
        torch.nn.Linear(4, 4),
        torch.device("cpu"),
        "fp8-torchao",
    )
    assert report["can_execute"] is False

