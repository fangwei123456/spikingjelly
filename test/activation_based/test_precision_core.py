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

    def autocast(enabled=True, recipe=None):
        return None

    fake_te.is_fp8_available = is_fp8_available
    fake_te.autocast = autocast
    fake_te.is_fp8_block_scaling_available = lambda: False
    fake_te.is_mxfp8_available = lambda: False
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
    assert report["te_autocast_api"] == "autocast"
    assert report["te_recipe_availability"]["auto"] is True
    assert report["te_recipe_availability"]["mxfp8"] is False
    assert report["can_convert"] is True
    assert report["can_execute"] is False
    assert report["execution_note"] == "fp8-te requires a CUDA device"


def test_build_capability_report_fp8_te_catches_legacy_probe_failure(monkeypatch):
    fake_te = types.ModuleType("transformer_engine.pytorch")

    def is_fp8_available(*args, **kwargs):
        if kwargs:
            raise TypeError("return_reason is unsupported")
        raise RuntimeError("driver probe failed")

    fake_te.is_fp8_available = is_fp8_available
    fake_te.autocast = lambda enabled=True, recipe=None: None
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
    assert report["te_fp8_available"] is False
    assert report["te_fp8_unavailable_reason"] == "driver probe failed"


def test_validate_capability_rejects_fp8_te_without_autocast_api():
    report = {
        "requested_mode": "fp8-te",
        "device_type": "cuda",
        "cuda_available": True,
        "transformer_engine_installed": True,
        "te_fp8_available": True,
        "te_autocast_api": None,
    }
    with pytest.raises(RuntimeError, match="autocast"):
        validate_capability(report)


def test_validate_capability_preserves_transformer_engine_import_error():
    report = {
        "requested_mode": "fp8-te",
        "transformer_engine_installed": False,
        "te_fp8_unavailable_reason": (
            "transformer-engine import failed: missing CUDA symbol"
        ),
    }

    with pytest.raises(RuntimeError, match="missing CUDA symbol"):
        validate_capability(report)


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


def test_fp8_te_warn_fallback_does_not_resolve_current_cuda_device(monkeypatch):
    def fail_current_device():
        raise AssertionError("current_device must not be called without CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", fail_current_device)

    with pytest.warns(RuntimeWarning, match="falling back to fp32"):
        artifacts = prepare_model_for_precision(
            torch.nn.Linear(4, 4),
            "cuda",
            PrecisionConfig(mode="fp8-te", strictness="warn"),
        )

    assert artifacts.effective_config.mode == "fp32"
    assert artifacts.fallback_reason


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
    state = {"entered": False, "exited": False, "enabled": None, "recipe": "unset"}

    class FakeContext:
        def __enter__(self):
            state["entered"] = True

        def __exit__(self, exc_type, exc, tb):
            state["exited"] = True

    def autocast(enabled=True, recipe=None):
        state["enabled"] = enabled
        state["recipe"] = recipe
        return FakeContext()

    fake_te.autocast = autocast
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
    assert state["recipe"] is None
    assert state["exited"] is True


def test_precision_artifacts_autocast_context_resolves_fake_te_recipe(monkeypatch):
    fake_te = types.ModuleType("transformer_engine.pytorch")
    fake_recipe_module = types.ModuleType("transformer_engine.common.recipe")
    state = {"recipe": None}

    class FakeDelayedScaling:
        pass

    class FakeContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return None

    def autocast(enabled=True, recipe=None):
        state["recipe"] = recipe
        return FakeContext()

    fake_recipe_module.DelayedScaling = FakeDelayedScaling
    fake_te.autocast = autocast
    fake_root = types.ModuleType("transformer_engine")
    fake_common = types.ModuleType("transformer_engine.common")
    fake_common.recipe = fake_recipe_module
    fake_root.pytorch = fake_te
    fake_root.common = fake_common
    monkeypatch.setitem(sys.modules, "transformer_engine", fake_root)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", fake_te)
    monkeypatch.setitem(sys.modules, "transformer_engine.common", fake_common)
    monkeypatch.setitem(
        sys.modules,
        "transformer_engine.common.recipe",
        fake_recipe_module,
    )

    policy = resolve_precision_policy(
        PrecisionConfig(mode="fp8-te", fp8_recipe="delayed")
    )
    artifacts = PrecisionArtifacts(
        requested_config=PrecisionConfig(mode="fp8-te", fp8_recipe="delayed"),
        effective_config=PrecisionConfig(mode="fp8-te", fp8_recipe="delayed"),
        policy=policy,
        model=torch.nn.Linear(4, 4),
    )
    with artifacts.autocast_context():
        pass
    assert isinstance(state["recipe"], FakeDelayedScaling)


def test_precision_artifacts_fp8_autocast_context_passes_legacy_recipe(monkeypatch):
    fake_te = types.ModuleType("transformer_engine.pytorch")
    fake_recipe_module = types.ModuleType("transformer_engine.common.recipe")
    state = {"recipe": None}

    class FakeDelayedScaling:
        pass

    class FakeContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return None

    def fp8_autocast(enabled=True, fp8_recipe=None):
        state["recipe"] = fp8_recipe
        return FakeContext()

    fake_recipe_module.DelayedScaling = FakeDelayedScaling
    fake_te.fp8_autocast = fp8_autocast
    fake_root = types.ModuleType("transformer_engine")
    fake_common = types.ModuleType("transformer_engine.common")
    fake_common.recipe = fake_recipe_module
    fake_root.pytorch = fake_te
    fake_root.common = fake_common
    monkeypatch.setitem(sys.modules, "transformer_engine", fake_root)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", fake_te)
    monkeypatch.setitem(sys.modules, "transformer_engine.common", fake_common)
    monkeypatch.setitem(
        sys.modules,
        "transformer_engine.common.recipe",
        fake_recipe_module,
    )

    policy = resolve_precision_policy(
        PrecisionConfig(mode="fp8-te", fp8_recipe="delayed")
    )
    artifacts = PrecisionArtifacts(
        requested_config=PrecisionConfig(mode="fp8-te", fp8_recipe="delayed"),
        effective_config=PrecisionConfig(mode="fp8-te", fp8_recipe="delayed"),
        policy=policy,
        model=torch.nn.Linear(4, 4),
    )
    with pytest.warns(RuntimeWarning, match="fp8_autocast"):
        with artifacts.autocast_context():
            pass
    assert isinstance(state["recipe"], FakeDelayedScaling)


def test_fp8_te_autocast_uses_configured_cuda_device(monkeypatch):
    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    events = []
    fake_te = types.ModuleType("transformer_engine.pytorch")

    class RecordingContext:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            events.append(f"enter:{self.name}")

        def __exit__(self, exc_type, exc, tb):
            events.append(f"exit:{self.name}")

    fake_te.autocast = lambda **kwargs: RecordingContext("te")
    fake_root = types.ModuleType("transformer_engine")
    fake_root.pytorch = fake_te
    monkeypatch.setitem(sys.modules, "transformer_engine", fake_root)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", fake_te)
    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda device: RecordingContext(str(device)),
    )
    policy = Float8TransformerEnginePolicy()
    policy._target_device = torch.device("cuda:1")

    with policy.autocast_context():
        events.append("body")

    assert events == [
        "enter:cuda:1",
        "enter:te",
        "body",
        "exit:te",
        "exit:cuda:1",
    ]


def test_fp8_te_device_check_includes_buffers():
    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    class BufferOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.empty(1, device="meta"))

    policy = Float8TransformerEnginePolicy()
    with pytest.raises(RuntimeError, match="parameters and buffers"):
        policy._ensure_model_on_device(BufferOnly(), torch.device("cpu"))


def test_fp8_te_unknown_recipe_warn_falls_back_to_fp32():
    model = torch.nn.Linear(4, 4)
    with pytest.warns(RuntimeWarning, match="falling back to fp32"):
        artifacts = prepare_model_for_precision(
            model,
            "cpu",
            PrecisionConfig(
                mode="fp8-te",
                strictness="warn",
                fp8_recipe="unknown-recipe",
            ),
        )
    assert artifacts.effective_config.mode == "fp32"
    assert "unsupported Transformer Engine FP8 recipe" in artifacts.fallback_reason
    report = artifacts.policy.capability_report()
    assert report["te_recipe_requested"] == "unknown-recipe"
    assert report["te_recipe_fallback_reason"]


def test_fp8_te_unknown_recipe_strict_raises():
    model = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="unsupported Transformer Engine FP8 recipe"):
        prepare_model_for_precision(
            model,
            "cpu",
            PrecisionConfig(
                mode="fp8-te",
                strictness="strict",
                fp8_recipe="unknown-recipe",
            ),
        )


def _install_fake_te_without_delayed_recipe(monkeypatch):
    fake_te = types.ModuleType("transformer_engine.pytorch")
    fake_recipe_module = types.ModuleType("transformer_engine.common.recipe")
    fake_common = types.ModuleType("transformer_engine.common")
    fake_root = types.ModuleType("transformer_engine")

    fake_te.autocast = lambda enabled=True, recipe=None, device=None: None
    fake_common.recipe = fake_recipe_module
    fake_root.pytorch = fake_te
    fake_root.common = fake_common
    monkeypatch.setitem(sys.modules, "transformer_engine", fake_root)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", fake_te)
    monkeypatch.setitem(sys.modules, "transformer_engine.common", fake_common)
    monkeypatch.setitem(
        sys.modules,
        "transformer_engine.common.recipe",
        fake_recipe_module,
    )


def _patch_fp8_te_executable_report(monkeypatch):
    def fake_report(model, device, mode):
        assert mode == "fp8-te"
        return {
            "requested_mode": "fp8-te",
            "device": "cuda:0",
            "device_type": "cuda",
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_capability": (9, 0),
            "torchao_installed": False,
            "transformer_engine_installed": True,
            "te_fp8_available": True,
            "te_fp8_unavailable_reason": None,
            "te_autocast_api": "autocast",
            "te_recipe_availability": {
                "auto": True,
                "delayed": True,
                "current": True,
                "block": False,
                "mxfp8": False,
            },
            "te_recipe_requested": None,
            "te_recipe_resolved": None,
            "te_recipe_fallback_reason": None,
            "bf16_supported": True,
            "cpu_bf16_autocast": False,
            "mps_available": False,
            "model_class": type(model).__name__,
            "can_convert": True,
            "can_execute": True,
            "runtime_validation_required": True,
            "execution_note": None,
        }

    monkeypatch.setattr(
        "spikingjelly.activation_based.precision.float8_te.build_capability_report",
        fake_report,
    )


def test_fp8_te_recipe_resolution_failure_warn_falls_back_to_fp32(monkeypatch):
    _install_fake_te_without_delayed_recipe(monkeypatch)
    _patch_fp8_te_executable_report(monkeypatch)
    model = torch.nn.Linear(4, 4)
    with pytest.warns(RuntimeWarning, match="falling back to fp32"):
        artifacts = prepare_model_for_precision(
            model,
            "cuda:0",
            PrecisionConfig(
                mode="fp8-te",
                strictness="warn",
                fp8_recipe="delayed",
            ),
        )
    assert artifacts.effective_config.mode == "fp32"
    assert "Failed to resolve FP8 recipe" in artifacts.fallback_reason
    report = artifacts.policy.capability_report()
    assert report["te_recipe_requested"] == "delayed"
    assert report["te_recipe_resolved"] is None
    assert "does not expose a recipe class" in report["te_recipe_fallback_reason"]


def test_fp8_te_recipe_resolution_failure_strict_raises(monkeypatch):
    _install_fake_te_without_delayed_recipe(monkeypatch)
    _patch_fp8_te_executable_report(monkeypatch)
    model = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="Failed to resolve FP8 recipe"):
        prepare_model_for_precision(
            model,
            "cuda:0",
            PrecisionConfig(
                mode="fp8-te",
                strictness="strict",
                fp8_recipe="delayed",
            ),
        )
