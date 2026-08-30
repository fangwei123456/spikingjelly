import pytest
import torch

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.precision import (
    PrecisionArtifacts,
    PrecisionConfig,
    prepare_model_for_precision,
)
from spikingjelly.activation_based.precision import convert as precision_convert
from spikingjelly.activation_based.triton_kernel.neuron_kernel import (
    utils as triton_neuron_utils,
)


def test_public_precision_surface():
    import spikingjelly.activation_based.precision as precision

    assert precision.__all__ == [
        "PrecisionArtifacts",
        "PrecisionConfig",
        "prepare_model_for_precision",
    ]


@pytest.mark.parametrize("mode", ("fp32", "fp16", "bf16", "fp8"))
def test_precision_config_modes(mode):
    assert PrecisionConfig.from_any(mode).mode == mode


def test_precision_config_normalizes_triton_fields():
    config = PrecisionConfig(
        triton_storage="torch.float8_e4m3fn",
        triton_fwd="BF16",
        triton_bwd="FP16",
    )
    assert config.triton_storage == "float8_e4m3fn"
    assert config.triton_fwd == "bf16"
    assert config.triton_bwd == "fp16"


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"mode": None}, "mode"),
        ({"mode": "fp8-te"}, "mode"),
        ({"mode": "fp8-torchao"}, "mode"),
        ({"mode": "bf16", "fp8_recipe": "delayed"}, "fp8_recipe"),
        (
            {"mode": "bf16", "fp8_fallback_dtype": "fp16"},
            "mode='fp8'",
        ),
        ({"triton_fwd": "bf16"}, "triton_storage"),
        (
            {"triton_storage": "bf16", "triton_bwd": "fp8"},
            "FP8 Triton compute",
        ),
    ),
)
def test_precision_config_rejects_invalid_combinations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PrecisionConfig(**kwargs)


def test_precision_config_from_dict_rejects_removed_fields():
    with pytest.raises(TypeError, match="unexpected keyword"):
        PrecisionConfig.from_any({"mode": "fp32", "strictness": "warn"})
    with pytest.raises(TypeError, match="unexpected keyword"):
        PrecisionConfig.from_any({"mode": "fp32", "device": "cpu"})
    with pytest.raises(TypeError, match="unexpected keyword"):
        PrecisionConfig.from_any({"mode": "fp8", "fp8_autocast_dtype": "bf16"})


def test_precision_config_defaults_to_bf16_fallback():
    config = PrecisionConfig(mode="fp8")

    assert config.fp8_fallback_dtype == "bf16"


def test_precision_config_accepts_fp8_fallback_override():
    config = PrecisionConfig(mode="fp8", fp8_fallback_dtype="fp16")

    assert config.fp8_fallback_dtype == "fp16"


def test_prepare_fp32_returns_public_artifacts():
    model = torch.nn.Linear(4, 4)
    artifacts = prepare_model_for_precision(model, "cpu", "fp32")
    assert isinstance(artifacts, PrecisionArtifacts)
    assert artifacts.model is model
    assert artifacts.config.mode == "fp32"
    assert artifacts.scaler is None
    report = artifacts.describe()
    assert report["policy"]["name"] == "fp32"
    assert report["triton_neurons"]["converted_modules"] == []


def test_prepare_bf16_uses_cpu_autocast_without_scaler():
    artifacts = prepare_model_for_precision(torch.nn.Linear(4, 4), "cpu", "bf16")
    assert artifacts.scaler is None
    with artifacts.autocast_context():
        output = artifacts.model(torch.randn(2, 4))
    assert output.dtype == torch.bfloat16


def test_prepare_fp8_fails_instead_of_falling_back():
    with pytest.raises(RuntimeError, match="fp8"):
        prepare_model_for_precision(torch.nn.Linear(4, 4), "cpu", "fp8")


def test_triton_precision_requires_convertible_nodes():
    config = PrecisionConfig(triton_storage="float8_e4m3fn")
    with pytest.raises(RuntimeError, match="no multi-step IF/LIF/PLIF"):
        prepare_model_for_precision(torch.nn.Linear(4, 4), "cpu", config)


def test_triton_precision_applies_atomically_and_clears(monkeypatch):
    first = neuron.IFNode(step_mode="m")
    second = neuron.LIFNode(step_mode="s")
    first._backend = second._backend = "triton"
    model = torch.nn.Sequential(first, second)
    monkeypatch.setattr(
        triton_neuron_utils,
        "_prepare_triton_neuron_execution_plan",
        lambda **_kwargs: None,
    )
    config = PrecisionConfig(
        triton_storage="bf16",
        triton_fwd="bf16",
        triton_bwd="fp32",
    )

    with pytest.raises(RuntimeError, match="requires multi-step"):
        precision_convert._configure_triton_neurons(model, config, "cpu")
    assert first._triton_precision is None

    second.step_mode = "m"
    precision_convert._configure_triton_neurons(model, config, "cpu")
    assert first._triton_precision == (torch.bfloat16, "bf16", "fp32")
    assert second._triton_precision == first._triton_precision

    precision_convert._configure_triton_neurons(
        model, PrecisionConfig(mode="fp32"), "cpu"
    )
    assert first._triton_precision is None
    assert second._triton_precision is None


def test_precision_artifacts_backward_steps_optimizer():
    model = torch.nn.Linear(4, 2)
    artifacts = prepare_model_for_precision(model, "cpu", "fp32")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    loss = model(torch.randn(3, 4)).sum()
    artifacts.backward(loss, optimizer)
    assert not torch.equal(before, model.weight)
