from types import SimpleNamespace
import sys
import types

import pytest
import torch

from spikingjelly.activation_based.precision import (
    PrecisionArtifacts,
    PrecisionConfig,
    build_capability_report,
    prepare_model_for_precision,
    resolve_precision_policy,
    validate_capability,
)


def test_precision_config_from_string():
    cfg = PrecisionConfig.from_any("bf16", default_device="cuda:0")
    assert cfg.mode == "bf16"
    assert cfg.device == "cuda:0"


def test_precision_config_from_none_uses_defaults():
    cfg = PrecisionConfig.from_any(None, default_device="cpu")
    assert cfg.mode == "fp32"
    assert cfg.device == "cpu"


def test_precision_config_with_none_mode_falls_back_to_fp32():
    cfg = PrecisionConfig(mode=None)
    assert cfg.mode == "fp32"


def test_precision_config_instance_with_none_device_uses_default_device():
    cfg = PrecisionConfig.from_any(
        PrecisionConfig(mode="bf16", device=None),
        default_device="cuda:0",
    )
    assert cfg.mode == "bf16"
    assert cfg.device == "cuda:0"


def test_precision_config_from_dict_aliases_precision_key():
    cfg = PrecisionConfig.from_any({"precision": "fp16", "device": "cuda:0"})
    assert cfg.mode == "fp16"
    assert cfg.device == "cuda:0"


def test_precision_config_from_dict_supports_precision_aliases():
    cfg = PrecisionConfig.from_any(
        {
            "precision": "fp8-torchao",
            "precision_strict": "strict",
            "fp8_report": False,
            "device": "cuda:0",
        }
    )
    assert cfg.mode == "fp8-torchao"
    assert cfg.strictness == "strict"
    assert cfg.report is False


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


def test_precision_config_from_object_with_none_device_uses_default_device():
    obj = SimpleNamespace(
        precision="fp16",
        device=None,
    )
    cfg = PrecisionConfig.from_any(obj, default_device="cuda:0")
    assert cfg.mode == "fp16"
    assert cfg.device == "cuda:0"


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


def test_resolve_precision_policy_fp8_te():
    policy = resolve_precision_policy(PrecisionConfig(mode="fp8-te"))
    assert policy.describe()["name"] == "fp8-te"
    assert policy.describe()["backend"] == "transformer-engine"


def test_build_capability_report_cpu_fp32():
    report = build_capability_report(torch.nn.Linear(4, 4), torch.device("cpu"), "fp32")
    assert report["device_type"] == "cpu"
    assert report["can_convert"] is True
    assert report["can_execute"] is True


def test_build_capability_report_fp32_does_not_probe_broken_te(monkeypatch):
    class BrokenTE(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError("broken te should not be probed")

    fake_root = types.ModuleType("transformer_engine")
    fake_root.pytorch = BrokenTE("transformer_engine.pytorch")
    monkeypatch.setitem(sys.modules, "transformer_engine", fake_root)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", fake_root.pytorch)

    report = build_capability_report(torch.nn.Linear(4, 4), torch.device("cpu"), "fp32")
    assert report["requested_mode"] == "fp32"
    assert report["can_execute"] is True


def test_build_capability_report_cpu_reports_bf16_autocast_support():
    report = build_capability_report(torch.nn.Linear(4, 4), torch.device("cpu"), "bf16")
    assert report["bf16_supported"] == report["cpu_bf16_autocast"]


def test_build_capability_report_accepts_string_device():
    report = build_capability_report(torch.nn.Linear(4, 4), "cpu", "fp32")
    assert report["device"] == "cpu"
    assert report["device_type"] == "cpu"


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


def test_validate_capability_rejects_fp8_torchao_when_torchao_missing():
    report = {
        "requested_mode": "fp8-torchao",
        "device_type": "cuda",
        "cuda_available": True,
        "cuda_device_capability": (9, 0),
        "torchao_installed": False,
    }
    with pytest.raises(RuntimeError, match="torchao"):
        validate_capability(report)


def test_prepare_model_for_precision_warn_falls_back_to_fp32_when_fp8_unavailable():
    model = torch.nn.Linear(4, 4)
    with pytest.warns(RuntimeWarning, match="falling back to fp32"):
        artifacts = prepare_model_for_precision(
            model,
            "cpu",
            PrecisionConfig(mode="fp8-torchao", strictness="warn"),
        )
    assert artifacts.requested_config.mode == "fp8-torchao"
    assert artifacts.effective_config.mode == "fp32"
    assert artifacts.policy.describe()["name"] == "fp32"
    assert artifacts.fallback_reason
    assert artifacts.policy.capability_report()["requested_mode"] == "fp8-torchao"


def test_prepare_model_for_precision_strict_keeps_fp8_capability_error():
    model = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="fp8-torchao|torchao"):
        prepare_model_for_precision(
            model,
            "cpu",
            PrecisionConfig(mode="fp8-torchao", strictness="strict"),
        )


def test_build_capability_report_fp8_te_cpu_reports_te_fields(monkeypatch):
    fake_te = types.ModuleType("transformer_engine.pytorch")

    def is_fp8_available(return_reason=False):
        return (True, None) if return_reason else True

    fake_te.is_fp8_available = is_fp8_available
    fake_root = types.ModuleType("transformer_engine")
    fake_root.pytorch = fake_te
    monkeypatch.setitem(sys.modules, "transformer_engine", fake_root)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", fake_te)

    report = build_capability_report(
        torch.nn.Linear(4, 4),
        torch.device("cpu"),
        "fp8-te",
    )
    assert report["transformer_engine_installed"] is True
    assert report["te_fp8_available"] is True
    assert report["can_convert"] is True
    assert report["can_execute"] is False
    assert report["execution_note"] == "fp8-te requires a CUDA device"


def test_prepare_model_for_precision_warn_falls_back_to_fp32_when_fp8_te_unavailable():
    model = torch.nn.Linear(4, 4)
    with pytest.warns(RuntimeWarning, match="falling back to fp32"):
        artifacts = prepare_model_for_precision(
            model,
            "cpu",
            PrecisionConfig(mode="fp8-te", strictness="warn"),
        )
    assert artifacts.requested_config.mode == "fp8-te"
    assert artifacts.effective_config.mode == "fp32"
    assert artifacts.policy.describe()["name"] == "fp32"
    assert artifacts.fallback_reason
    assert artifacts.policy.capability_report()["requested_mode"] == "fp8-te"


def test_prepare_model_for_precision_strict_keeps_fp8_te_capability_error():
    model = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="fp8-te|transformer-engine|CUDA"):
        prepare_model_for_precision(
            model,
            "cpu",
            PrecisionConfig(mode="fp8-te", strictness="strict"),
        )


def test_precision_artifacts_autocast_context_delegates_to_fake_te(monkeypatch):
    fake_te = types.ModuleType("transformer_engine.pytorch")
    state = {"entered": False, "exited": False, "enabled": None}

    class FakeContext:
        def __enter__(self):
            state["entered"] = True

        def __exit__(self, exc_type, exc, tb):
            state["exited"] = True

    def fp8_autocast(enabled=True):
        state["enabled"] = enabled
        return FakeContext()

    fake_te.fp8_autocast = fp8_autocast
    fake_root = types.ModuleType("transformer_engine")
    fake_root.pytorch = fake_te
    monkeypatch.setitem(sys.modules, "transformer_engine", fake_root)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", fake_te)

    policy = resolve_precision_policy(PrecisionConfig(mode="fp8-te"))
    artifacts = PrecisionArtifacts(
        requested_config=PrecisionConfig(mode="fp8-te"),
        effective_config=PrecisionConfig(mode="fp8-te"),
        policy=policy,
        model=torch.nn.Linear(4, 4),
    )
    with artifacts.autocast_context():
        assert state["entered"] is True
    assert state["enabled"] is True
    assert state["exited"] is True
