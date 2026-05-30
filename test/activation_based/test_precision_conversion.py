import importlib.util

import pytest
import torch

from spikingjelly.activation_based import layer
from spikingjelly.activation_based.model import Spikformer
from spikingjelly.activation_based.precision import (
    Float8LinearStepModule,
    PrecisionConfig,
    analyze_convertible_modules,
    prepare_model_for_precision,
)


HAS_TORCHAO = importlib.util.find_spec("torchao") is not None


def test_conversion_report_marks_spikformer_linear_and_high_precision_modules():
    model = Spikformer(
        T=2,
        in_channels=3,
        img_size_h=64,
        img_size_w=64,
        num_classes=7,
        embed_dims=64,
        num_heads=4,
        depths=2,
        backend="torch",
    )
    report = analyze_convertible_modules(model).to_dict()
    assert report["convertible_linear"] >= 1
    assert "head" in report["convertible_modules"]
    assert report["high_precision_modules"]


def test_float8_linear_step_module_preserves_multistep_shape():
    base = torch.nn.Linear(8, 4)
    wrapped = Float8LinearStepModule(base, step_mode="m")
    x = torch.randn(3, 2, 8)
    y = wrapped(x)
    assert y.shape == (3, 2, 4)


def test_float8_linear_step_module_delegates_attributes():
    base = torch.nn.Linear(8, 4)
    wrapped = Float8LinearStepModule(base, step_mode="s")
    assert wrapped.in_features == 8
    assert wrapped.out_features == 4
    assert wrapped.weight is base.weight


def test_float8_linear_step_module_load_state_dict():
    base = torch.nn.Linear(8, 4)
    wrapped = Float8LinearStepModule(base, step_mode="s")
    state_dict = wrapped.state_dict()
    wrapped.load_state_dict(state_dict, strict=True)


def test_float8_linear_step_module_load_state_dict_from_parent():
    base = torch.nn.Linear(8, 4)
    parent = torch.nn.Sequential(Float8LinearStepModule(base, step_mode="s"))
    state_dict = parent.state_dict()
    assert all("wrapped" not in k for k in state_dict), state_dict.keys()
    parent.load_state_dict(state_dict, strict=True)


def test_float8_linear_step_module_parent_load_state_dict_has_no_duplicate_errors():
    base = torch.nn.Linear(8, 4)
    parent = torch.nn.Sequential(Float8LinearStepModule(base, step_mode="s"))
    state_dict = parent.state_dict()
    incompatible = parent.load_state_dict(state_dict, strict=False)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


@pytest.mark.skipif(
    not HAS_TORCHAO
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability(0) < (8, 9),
    reason="Static fp8-torchao replacement test requires CUDA compute capability >= 8.9.",
)
def test_prepare_model_for_precision_replaces_spikformer_head_with_float8_wrapper():
    model = Spikformer(
        T=2,
        in_channels=3,
        img_size_h=64,
        img_size_w=64,
        num_classes=7,
        embed_dims=64,
        num_heads=4,
        depths=2,
        backend="torch",
    ).cuda()
    artifacts = prepare_model_for_precision(
        model,
        "cuda:0",
        PrecisionConfig(mode="fp8-torchao", strictness="strict", device="cuda:0"),
    )
    assert isinstance(artifacts.model.head, Float8LinearStepModule)
    report = artifacts.policy.conversion_report()
    assert "head" in report["converted_modules"]


def test_capability_report_splits_can_convert_and_can_execute():
    model = torch.nn.Sequential(layer.Linear(4, 8), torch.nn.ReLU(), layer.Linear(8, 4))
    artifacts = prepare_model_for_precision(model, "cpu", PrecisionConfig(mode="fp32"))
    report = artifacts.policy.capability_report()
    assert report["can_convert"] is True
    assert report["can_execute"] is True


@pytest.mark.skipif(
    not HAS_TORCHAO
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability(0) < (8, 9),
    reason="Root Linear fp8-torchao conversion requires torchao and CUDA compute capability >= 8.9.",
)
def test_prepare_model_for_precision_replaces_root_linear_module():
    model = layer.Linear(16, 32).cuda()
    artifacts = prepare_model_for_precision(
        model,
        "cuda:0",
        PrecisionConfig(mode="fp8-torchao", strictness="strict", device="cuda:0"),
    )
    assert isinstance(artifacts.model, Float8LinearStepModule)
    assert "<root>" in artifacts.policy.conversion_report()["converted_modules"]


def test_convert_model_for_precision_preserves_shared_linear_module_identity():
    shared = torch.nn.Linear(8, 8)
    model = torch.nn.ModuleList([shared, shared])
    policy = type(
        "StubPolicy",
        (),
        {
            "name": "fp32",
        },
    )()
    converted, _ = prepare_model_for_precision(
        model,
        "cpu",
        PrecisionConfig(mode="fp32"),
    ).model, None
    assert converted[0] is converted[1]
