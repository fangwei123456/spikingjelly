import copy
import sys
import types

import pytest
import torch

try:
    import torchao  # noqa: F401

    HAS_TORCHAO = True
except ImportError:
    HAS_TORCHAO = False

from spikingjelly.activation_based import layer, neuron
from spikingjelly.activation_based.layer.attention import SpikingSelfAttention
from spikingjelly.activation_based.model import Spikformer
from spikingjelly.activation_based.precision import (
    Float8PointwiseConv1dStepModule,
    Float8LinearStepModule,
    Float8TELayerNormLinearModule,
    Float8TELayerNormMLPModule,
    PrecisionConfig,
    TransformerEngineDotProductAttentionAdapter,
    analyze_convertible_modules,
    make_linear_from_pointwise_conv1d,
    prepare_model_for_precision,
)
from spikingjelly.activation_based.precision.convert import convert_model_for_precision


def _install_fake_te(monkeypatch):
    fake_te = types.ModuleType("transformer_engine.pytorch")

    class FakeTELinear(torch.nn.Linear):
        def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            params_dtype=torch.float32,
            **kwargs,
        ):
            super().__init__(
                in_features,
                out_features,
                bias=bias,
                dtype=params_dtype,
            )

    class FakeTELayerNorm(torch.nn.LayerNorm):
        def __init__(self, hidden_size, eps=1e-5, params_dtype=torch.float32, **kwargs):
            super().__init__(hidden_size, eps=eps, dtype=params_dtype)

    class FakeTELayerNormLinear(torch.nn.Module):
        def __init__(
            self,
            hidden_size,
            out_features,
            eps=1e-5,
            bias=True,
            params_dtype=torch.float32,
            **kwargs,
        ):
            super().__init__()
            self.layer_norm_weight = torch.nn.Parameter(
                torch.ones(hidden_size, dtype=params_dtype)
            )
            self.layer_norm_bias = torch.nn.Parameter(
                torch.zeros(hidden_size, dtype=params_dtype)
            )
            self.weight = torch.nn.Parameter(
                torch.empty(out_features, hidden_size, dtype=params_dtype)
            )
            self.bias = (
                torch.nn.Parameter(torch.empty(out_features, dtype=params_dtype))
                if bias
                else None
            )
            self._extra_state = {"fp8_scale": torch.ones(1, dtype=params_dtype)}
            self.eps = eps
            torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

        def forward(self, x):
            x = torch.nn.functional.layer_norm(
                x,
                self.layer_norm_weight.shape,
                self.layer_norm_weight,
                self.layer_norm_bias,
                self.eps,
            )
            return torch.nn.functional.linear(x, self.weight, self.bias)

        def get_extra_state(self):
            return self._extra_state

        def set_extra_state(self, state):
            self._extra_state = state

    class FakeTELayerNormMLP(torch.nn.Module):
        def __init__(
            self,
            hidden_size,
            ffn_hidden_size,
            eps=1e-5,
            bias=True,
            params_dtype=torch.float32,
            **kwargs,
        ):
            super().__init__()
            self.layer_norm_weight = torch.nn.Parameter(
                torch.ones(hidden_size, dtype=params_dtype)
            )
            self.layer_norm_bias = torch.nn.Parameter(
                torch.zeros(hidden_size, dtype=params_dtype)
            )
            self.fc1_weight = torch.nn.Parameter(
                torch.empty(ffn_hidden_size, hidden_size, dtype=params_dtype)
            )
            self.fc1_bias = (
                torch.nn.Parameter(torch.empty(ffn_hidden_size, dtype=params_dtype))
                if bias
                else None
            )
            self.fc2_weight = torch.nn.Parameter(
                torch.empty(hidden_size, ffn_hidden_size, dtype=params_dtype)
            )
            self.fc2_bias = (
                torch.nn.Parameter(torch.empty(hidden_size, dtype=params_dtype))
                if bias
                else None
            )
            self._extra_state = {"fp8_scale": torch.ones(1, dtype=params_dtype)}
            self.eps = eps
            torch.nn.init.kaiming_uniform_(self.fc1_weight, a=5**0.5)
            torch.nn.init.kaiming_uniform_(self.fc2_weight, a=5**0.5)
            if self.fc1_bias is not None:
                torch.nn.init.zeros_(self.fc1_bias)
            if self.fc2_bias is not None:
                torch.nn.init.zeros_(self.fc2_bias)

        def forward(self, x):
            x = torch.nn.functional.layer_norm(
                x,
                self.layer_norm_weight.shape,
                self.layer_norm_weight,
                self.layer_norm_bias,
                self.eps,
            )
            x = torch.nn.functional.linear(x, self.fc1_weight, self.fc1_bias)
            x = torch.nn.functional.gelu(x)
            return torch.nn.functional.linear(x, self.fc2_weight, self.fc2_bias)

        def get_extra_state(self):
            return self._extra_state

        def set_extra_state(self, state):
            self._extra_state = state

    class FakeTEDotProductAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, query, key, value, *args, **kwargs):
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return y.transpose(1, 2)

    class FakeContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return None

    def autocast(enabled=True, recipe=None, **kwargs):
        return FakeContext()

    def is_fp8_available(return_reason=False):
        return (True, None) if return_reason else True

    fake_te.Linear = FakeTELinear
    fake_te.LayerNorm = FakeTELayerNorm
    fake_te.LayerNormLinear = FakeTELayerNormLinear
    fake_te.LayerNormMLP = FakeTELayerNormMLP
    fake_te.DotProductAttention = FakeTEDotProductAttention
    fake_te.autocast = autocast
    fake_te.is_fp8_available = is_fp8_available
    fake_root = types.ModuleType("transformer_engine")
    fake_common = types.ModuleType("transformer_engine.common")
    fake_recipe = types.ModuleType("transformer_engine.common.recipe")
    fake_common.recipe = fake_recipe
    fake_root.pytorch = fake_te
    fake_root.common = fake_common
    monkeypatch.setitem(sys.modules, "transformer_engine", fake_root)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", fake_te)
    monkeypatch.setitem(sys.modules, "transformer_engine.common", fake_common)
    monkeypatch.setitem(sys.modules, "transformer_engine.common.recipe", fake_recipe)
    return fake_te


def _clone_state_dict(state_dict):
    cloned = {}
    for key, value in state_dict.items():
        if hasattr(value, "clone"):
            cloned[key] = value.clone()
        elif isinstance(value, dict):
            cloned[key] = copy.deepcopy(value)
        else:
            cloned[key] = value
    return cloned


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


def test_pointwise_conv1d_linear_adapter_matches_conv1d():
    conv = torch.nn.Conv1d(8, 4, kernel_size=1, bias=True)
    linear = make_linear_from_pointwise_conv1d(conv)
    wrapped = Float8PointwiseConv1dStepModule(linear, conv, step_mode="s")
    x = torch.randn(3, 8, 5)
    torch.testing.assert_close(wrapped(x), conv(x))


def test_pointwise_conv1d_step_module_preserves_multistep_shape_and_values():
    conv = layer.Conv1d(8, 4, kernel_size=1, bias=False, step_mode="m")
    linear = make_linear_from_pointwise_conv1d(conv)
    wrapped = Float8PointwiseConv1dStepModule(linear, conv, step_mode="m")
    x = torch.randn(2, 3, 8, 5)
    torch.testing.assert_close(wrapped(x), conv(x))
    assert wrapped.step_mode == "m"


def test_pointwise_conv1d_step_module_passes_contiguous_linear_input():
    class ContiguityCheckingLinear(torch.nn.Linear):
        def forward(self, x):
            assert x.is_contiguous()
            return super().forward(x)

    conv = torch.nn.Conv1d(8, 4, kernel_size=1, bias=False)
    linear = ContiguityCheckingLinear(8, 4, bias=False)
    with torch.no_grad():
        linear.weight.copy_(conv.weight.squeeze(-1))

    wrapped_s = Float8PointwiseConv1dStepModule(linear, conv, step_mode="s")
    x_s = torch.randn(3, 8, 5)
    torch.testing.assert_close(wrapped_s(x_s), conv(x_s))

    wrapped_m = Float8PointwiseConv1dStepModule(linear, conv, step_mode="m")
    x = torch.randn(2, 3, 8, 5)
    assert wrapped_m(x).shape == (2, 3, 4, 5)


def test_pointwise_conv1d_step_module_load_state_dict_from_parent():
    conv = torch.nn.Conv1d(8, 4, kernel_size=1, bias=True)
    linear = make_linear_from_pointwise_conv1d(conv)
    parent = torch.nn.Sequential(
        Float8PointwiseConv1dStepModule(linear, conv, step_mode="s")
    )
    state_dict = parent.state_dict()
    assert state_dict["0.weight"].shape == conv.weight.shape
    assert all("wrapped" not in k for k in state_dict), state_dict.keys()
    parent.load_state_dict(state_dict, strict=True)


def test_pointwise_conv1d_step_module_load_state_dict_ignores_neighbor_prefix():
    conv = torch.nn.Conv1d(8, 4, kernel_size=1, bias=True)
    linear = make_linear_from_pointwise_conv1d(conv)

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = Float8PointwiseConv1dStepModule(linear, conv, step_mode="s")
            self.conv_extra = torch.nn.Linear(8, 4)

    parent = Parent()
    state_dict = parent.state_dict()
    expected_neighbor_weight = state_dict["conv_extra.weight"].clone() + 1
    state_dict["conv_extra.weight"] = expected_neighbor_weight
    parent.load_state_dict(state_dict, strict=True)
    torch.testing.assert_close(parent.conv_extra.weight, expected_neighbor_weight)


def test_conversion_report_marks_pointwise_conv1d_convertible():
    model = torch.nn.Sequential(
        torch.nn.Conv1d(8, 16, kernel_size=1, bias=False),
        torch.nn.Conv1d(16, 16, kernel_size=3, padding=1, bias=False),
    )
    report = analyze_convertible_modules(model).to_dict()
    assert report["convertible_pointwise_conv1d"] == 1
    assert "0" in report["convertible_modules"]
    assert "1" in report["unsupported_modules"]


def test_conversion_report_does_not_mark_layer_norm_convertible_by_default():
    model = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.BatchNorm1d(8))
    report = analyze_convertible_modules(model).to_dict()
    assert report["convertible_layer_norm"] == 0
    assert "0" not in report["convertible_modules"]
    assert "0" in report["high_precision_modules"]
    assert "1" in report["high_precision_modules"]


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
    converted, _ = (
        prepare_model_for_precision(
            model,
            "cpu",
            PrecisionConfig(mode="fp32"),
        ).model,
        None,
    )
    assert converted[0] is converted[1]


@pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
def test_convert_model_for_precision_preserves_shared_linear_module_identity_fp8(
    monkeypatch,
):
    class DummyFloat8Linear(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        @classmethod
        def from_float(cls, base, config):
            return cls(base)

        def forward(self, x):
            return self.base(x)

    monkeypatch.setattr(
        "torchao.float8.float8_linear.Float8Linear",
        DummyFloat8Linear,
        raising=False,
    )

    class SharedLinearModule(torch.nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.first = shared
            self.second = shared

    from spikingjelly.activation_based.precision.float8_torchao import (
        Float8TorchAOPolicy,
    )

    shared = torch.nn.Linear(8, 8)
    model = SharedLinearModule(shared)
    policy = Float8TorchAOPolicy()
    policy.float8_linear_config = object()
    converted, report = convert_model_for_precision(model, policy)
    assert converted.first is converted.second
    assert len(report.converted_modules) == 2


@pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
def test_convert_model_for_precision_skips_revisiting_shared_non_linear_modules(
    monkeypatch,
):
    class DummyFloat8Linear(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        @classmethod
        def from_float(cls, base, config):
            return cls(base)

        def forward(self, x):
            return self.base(x)

    monkeypatch.setattr(
        "torchao.float8.float8_linear.Float8Linear",
        DummyFloat8Linear,
        raising=False,
    )

    class SharedBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

    shared = SharedBlock()

    class Parent(torch.nn.Module):
        def __init__(self, block):
            super().__init__()
            self.first = block
            self.second = block

    from spikingjelly.activation_based.precision.float8_torchao import (
        Float8TorchAOPolicy,
    )

    model = Parent(shared)
    policy = Float8TorchAOPolicy()
    policy.float8_linear_config = object()
    converted, report = convert_model_for_precision(model, policy)
    assert converted.first is converted.second
    assert converted.first.linear is converted.second.linear
    assert report.converted_modules == ["first.linear"]


@pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
def test_convert_model_for_precision_replaces_pointwise_conv1d_fp8(monkeypatch):
    class DummyFloat8Linear(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.weight = torch.nn.Parameter(base.weight.detach().clone())
            self.bias = (
                torch.nn.Parameter(base.bias.detach().clone())
                if base.bias is not None
                else None
            )

        @classmethod
        def from_float(cls, base, config):
            return cls(base)

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight, self.bias)

    monkeypatch.setattr(
        "torchao.float8.float8_linear.Float8Linear",
        DummyFloat8Linear,
        raising=False,
    )

    from spikingjelly.activation_based.precision.float8_torchao import (
        Float8TorchAOPolicy,
    )

    model = torch.nn.Sequential(
        torch.nn.Conv1d(8, 16, kernel_size=1, bias=False),
        torch.nn.Conv1d(16, 16, kernel_size=3, padding=1, bias=False),
    )
    policy = Float8TorchAOPolicy()
    policy.float8_linear_config = object()
    converted, report = convert_model_for_precision(model, policy)
    assert isinstance(converted[0], Float8PointwiseConv1dStepModule)
    assert isinstance(converted[1], torch.nn.Conv1d)
    assert report.converted_modules == ["0"]
    assert "1" in report.unsupported_modules


def test_convert_model_for_precision_replaces_nested_linear_fp8_te(monkeypatch):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    x = torch.randn(3, 8)
    expected = model(x)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    torch.testing.assert_close(converted(x), expected)
    assert isinstance(converted[0], Float8LinearStepModule)
    assert isinstance(converted[2], Float8LinearStepModule)
    assert report.converted_modules == ["0", "2"]


def test_convert_model_for_precision_replaces_root_linear_fp8_te(monkeypatch):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.Linear(8, 16)
    x = torch.randn(3, 8)
    expected = model(x)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    torch.testing.assert_close(converted(x), expected)
    assert isinstance(converted, Float8LinearStepModule)
    assert report.converted_modules == ["<root>"]


def test_convert_model_for_precision_preserves_layer_linear_step_mode_fp8_te(
    monkeypatch,
):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = layer.Linear(8, 16, step_mode="m")
    x = torch.randn(2, 3, 8)
    expected = model(x)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    torch.testing.assert_close(converted(x), expected)
    assert isinstance(converted, Float8LinearStepModule)
    assert converted.step_mode == "m"
    assert report.converted_modules == ["<root>"]


def test_convert_model_for_precision_replaces_pointwise_conv1d_fp8_te(monkeypatch):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.Sequential(
        torch.nn.Conv1d(8, 16, kernel_size=1, bias=False),
        torch.nn.Conv1d(16, 16, kernel_size=3, padding=1, bias=False),
    )
    x = torch.randn(3, 8, 5)
    expected = model(x)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    torch.testing.assert_close(converted(x), expected)
    assert isinstance(converted[0], Float8PointwiseConv1dStepModule)
    assert isinstance(converted[1], torch.nn.Conv1d)
    assert report.converted_modules == ["0"]
    assert "1" in report.unsupported_modules


def test_convert_model_for_precision_reports_only_leaf_skips_fp8_te(monkeypatch):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.Sequential(
        torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(8, 4)),
        torch.nn.Sequential(torch.nn.BatchNorm1d(4)),
    )
    policy = Float8TransformerEnginePolicy()
    _, report = convert_model_for_precision(model, policy)
    assert report.converted_modules == ["0.1"]
    assert "0" not in report.skipped_modules
    assert "1" not in report.skipped_modules
    assert "0.0" in report.skipped_modules
    assert "1.0" in report.skipped_modules


def test_convert_model_for_precision_replaces_root_pointwise_conv1d_fp8_te(
    monkeypatch,
):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = layer.Conv1d(8, 16, kernel_size=1, bias=False, step_mode="m")
    x = torch.randn(2, 3, 8, 5)
    expected = model(x)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    torch.testing.assert_close(converted(x), expected)
    assert isinstance(converted, Float8PointwiseConv1dStepModule)
    assert converted.step_mode == "m"
    assert report.converted_modules == ["<root>"]


def test_convert_model_for_precision_replaces_spikformer_projections_fp8_te(
    monkeypatch,
):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

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
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    converted.eval()

    for block in converted.blocks:
        assert isinstance(block.attn, SpikingSelfAttention)
        qkv_children = list(block.attn.qkv_conv_bn.children())
        assert qkv_children
        assert isinstance(qkv_children[0], Float8PointwiseConv1dStepModule)
        assert isinstance(block.attn.qkv_conv_bn[1], torch.nn.BatchNorm1d)
        assert isinstance(block.attn.qkv_lif, neuron.LIFNode)
        assert isinstance(block.attn.attn_lif, neuron.LIFNode)
        proj_children = list(block.attn.proj_conv_bn.children())
        assert proj_children
        assert isinstance(proj_children[0], Float8PointwiseConv1dStepModule)
        assert isinstance(block.attn.proj_conv_bn[1], torch.nn.BatchNorm1d)
        assert isinstance(block.attn.proj_lif, neuron.LIFNode)
        fc1_children = list(block.mlp.fc1.children())
        assert fc1_children
        assert isinstance(fc1_children[0], Float8PointwiseConv1dStepModule)
        assert isinstance(block.mlp.fc1[1], torch.nn.BatchNorm1d)
        assert isinstance(block.mlp.neuron1, neuron.LIFNode)
        fc2_children = list(block.mlp.fc2.children())
        assert fc2_children
        assert isinstance(fc2_children[0], Float8PointwiseConv1dStepModule)
        assert isinstance(block.mlp.fc2[1], torch.nn.BatchNorm1d)
        assert isinstance(block.mlp.neuron2, neuron.LIFNode)
    assert isinstance(converted.head, Float8LinearStepModule)
    assert isinstance(converted.patch_embed.stages[0].conv_bn.block[0], torch.nn.Conv2d)
    assert isinstance(
        converted.patch_embed.stages[0].conv_bn.block[1], torch.nn.BatchNorm2d
    )
    assert isinstance(converted.patch_embed.stages[0].neuron, neuron.LIFNode)
    assert converted(x).shape == (2, 2, 7)
    assert report.converted_modules == [
        "blocks.0.attn.qkv_conv_bn.0",
        "blocks.0.attn.proj_conv_bn.0",
        "blocks.0.mlp.fc1.0",
        "blocks.0.mlp.fc2.0",
        "blocks.1.attn.qkv_conv_bn.0",
        "blocks.1.attn.proj_conv_bn.0",
        "blocks.1.mlp.fc1.0",
        "blocks.1.mlp.fc2.0",
        "head",
    ]


def test_convert_model_for_precision_replaces_layer_norm_fp8_te(monkeypatch):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.Sequential(
        torch.nn.LayerNorm(8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    )
    x = torch.randn(3, 5, 8)
    expected = model(x)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    torch.testing.assert_close(converted(x), expected)
    assert isinstance(converted[0], torch.nn.LayerNorm)
    assert isinstance(converted[2], Float8LinearStepModule)
    assert report.convertible_layer_norm == 1
    assert report.converted_modules == ["0", "2"]


def test_convert_model_for_precision_replaces_root_layer_norm_fp8_te(monkeypatch):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.LayerNorm(8)
    x = torch.randn(3, 5, 8)
    expected = model(x)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    torch.testing.assert_close(converted(x), expected)
    assert isinstance(converted, torch.nn.LayerNorm)
    assert report.converted_modules == ["<root>"]


def test_convert_model_for_precision_copies_alt_named_layer_norm_fp8_te(monkeypatch):
    fake_te = _install_fake_te(monkeypatch)

    class FakeAltTELayerNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-5, params_dtype=torch.float32, **kwargs):
            super().__init__()
            self.layer_norm_weight = torch.nn.Parameter(
                torch.ones(hidden_size, dtype=params_dtype)
            )
            self.layer_norm_bias = torch.nn.Parameter(
                torch.zeros(hidden_size, dtype=params_dtype)
            )
            self.eps = eps

        def forward(self, x):
            return torch.nn.functional.layer_norm(
                x,
                self.layer_norm_weight.shape,
                self.layer_norm_weight,
                self.layer_norm_bias,
                self.eps,
            )

    fake_te.LayerNorm = FakeAltTELayerNorm

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.LayerNorm(8)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(0.5)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    torch.testing.assert_close(converted.layer_norm_weight, model.weight)
    torch.testing.assert_close(converted.layer_norm_bias, model.bias)
    assert report.converted_modules == ["<root>"]


def test_convert_model_for_precision_reports_root_layer_norm_skip_without_te_layer_norm(
    monkeypatch,
):
    fake_te = _install_fake_te(monkeypatch)
    del fake_te.LayerNorm

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.LayerNorm(8)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    assert converted is model
    assert report.converted_modules == []
    assert report.skipped_modules == ["<root>"]


def test_convert_model_for_precision_fuses_layer_norm_linear_fp8_te(monkeypatch):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.Sequential(
        torch.nn.Sequential(torch.nn.LayerNorm(8), layer.Linear(8, 4, step_mode="m"))
    )
    x = torch.randn(3, 5, 8)
    expected = model(x)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    torch.testing.assert_close(converted(x), expected)
    assert isinstance(converted[0], Float8TELayerNormLinearModule)
    assert converted[0].step_mode == "m"
    converted[0].set_step_mode("s")
    assert converted[0].step_mode == "s"
    assert report.converted_modules == ["0"]
    assert report.converted_patterns == [
        {"module": "0", "pattern": "LayerNormLinear", "backend": "te"}
    ]
    state_dict = converted.state_dict()
    assert "0.0.weight" in state_dict
    assert "0.1.weight" in state_dict
    assert any(key.endswith("_extra_state") for key in state_dict)
    converted.load_state_dict(state_dict, strict=True)
    modified_state_dict = _clone_state_dict(state_dict)
    modified_state_dict["0.0.weight"].add_(1.0)
    expected_weight = modified_state_dict["0.0.weight"].clone()
    converted.load_state_dict(modified_state_dict, strict=True)
    torch.testing.assert_close(converted.state_dict()["0.0.weight"], expected_weight)


def test_convert_model_for_precision_preserves_shared_fused_pattern_fp8_te(
    monkeypatch,
):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    shared = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Linear(8, 4))
    model = torch.nn.ModuleList([shared, shared])
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    assert isinstance(converted[0], Float8TELayerNormLinearModule)
    assert converted[0] is converted[1]
    assert report.converted_modules == ["0", "1"]
    assert report.converted_patterns == [
        {"module": "0", "pattern": "LayerNormLinear", "backend": "te"}
    ]


def test_convert_model_for_precision_fuses_layer_norm_mlp_fp8_te(monkeypatch):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.Sequential(
        torch.nn.Sequential(
            torch.nn.LayerNorm(8),
            layer.Linear(8, 16, step_mode="m"),
            torch.nn.GELU(),
            torch.nn.Linear(16, 8),
        )
    )
    x = torch.randn(3, 5, 8)
    expected = model(x)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    torch.testing.assert_close(converted(x), expected)
    assert isinstance(converted[0], Float8TELayerNormMLPModule)
    assert converted[0].step_mode == "m"
    converted[0].set_step_mode("s")
    assert converted[0].step_mode == "s"
    assert report.converted_modules == ["0"]
    assert report.converted_patterns == [
        {"module": "0", "pattern": "LayerNormMLP", "backend": "te"}
    ]
    state_dict = converted.state_dict()
    assert "0.0.weight" in state_dict
    assert "0.1.weight" in state_dict
    assert "0.3.weight" in state_dict
    assert any(key.endswith("_extra_state") for key in state_dict)
    converted.load_state_dict(state_dict, strict=True)
    modified_state_dict = _clone_state_dict(state_dict)
    modified_state_dict["0.3.bias"].add_(1.0)
    expected_bias = modified_state_dict["0.3.bias"].clone()
    converted.load_state_dict(modified_state_dict, strict=True)
    torch.testing.assert_close(converted.state_dict()["0.3.bias"], expected_bias)


def test_convert_model_for_precision_skips_incompatible_layer_norm_mlp_fp8_te(
    monkeypatch,
):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.Sequential(
        torch.nn.Sequential(
            torch.nn.LayerNorm(8),
            torch.nn.Linear(8, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 4),
        )
    )
    x = torch.randn(3, 5, 8)
    expected = model(x)
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    torch.testing.assert_close(converted(x), expected)
    assert not isinstance(converted[0], Float8TELayerNormMLPModule)
    assert converted(x).shape[-1] == 4
    assert report.converted_patterns == []


def test_convert_model_for_precision_skips_approximate_gelu_layer_norm_mlp_fp8_te(
    monkeypatch,
):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.Sequential(
        torch.nn.Sequential(
            torch.nn.LayerNorm(8),
            torch.nn.Linear(8, 16),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(16, 8),
        )
    )
    policy = Float8TransformerEnginePolicy()
    converted, report = convert_model_for_precision(model, policy)
    assert not isinstance(converted[0], Float8TELayerNormMLPModule)
    assert report.converted_patterns == []


def test_transformer_engine_sdpa_adapter_matches_torch_sdpa(monkeypatch):
    _install_fake_te(monkeypatch)
    adapter = TransformerEngineDotProductAttentionAdapter(
        num_attention_heads=2,
        head_dim=4,
    )
    query = torch.randn(3, 2, 5, 4)
    key = torch.randn(3, 2, 5, 4)
    value = torch.randn(3, 2, 5, 4)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(adapter(query, key, value), expected)


def test_transformer_engine_sdpa_adapter_rejects_dropout_mismatch(monkeypatch):
    _install_fake_te(monkeypatch)
    adapter = TransformerEngineDotProductAttentionAdapter(
        num_attention_heads=2,
        head_dim=4,
        attention_dropout=0.1,
    )
    query = torch.randn(3, 2, 5, 4)
    key = torch.randn(3, 2, 5, 4)
    value = torch.randn(3, 2, 5, 4)
    with pytest.raises(ValueError, match="dropout"):
        adapter(query, key, value, dropout_p=0.0)


def test_transformer_engine_sdpa_adapter_allows_zero_dropout_in_eval(monkeypatch):
    _install_fake_te(monkeypatch)
    adapter = TransformerEngineDotProductAttentionAdapter(
        num_attention_heads=2,
        head_dim=4,
        attention_dropout=0.1,
    )
    adapter.eval()
    query = torch.randn(3, 2, 5, 4)
    key = torch.randn(3, 2, 5, 4)
    value = torch.randn(3, 2, 5, 4)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(adapter(query, key, value, dropout_p=0.0), expected)
    with pytest.raises(ValueError, match="evaluation"):
        adapter(query, key, value, dropout_p=0.1)


def test_transformer_engine_sdpa_adapter_rejects_key_value_shape_mismatch(
    monkeypatch,
):
    _install_fake_te(monkeypatch)
    adapter = TransformerEngineDotProductAttentionAdapter(
        num_attention_heads=2,
        head_dim=4,
    )
    query = torch.randn(3, 2, 5, 4)
    key = torch.randn(3, 2, 5, 4)
    value = torch.randn(3, 2, 6, 4)
    with pytest.raises(ValueError, match="same sequence length"):
        adapter(query, key, value)


def test_float8_te_linear_step_module_load_state_dict_from_parent(monkeypatch):
    _install_fake_te(monkeypatch)

    from spikingjelly.activation_based.precision.float8_te import (
        Float8TransformerEnginePolicy,
    )

    model = torch.nn.Sequential(torch.nn.Linear(8, 16))
    policy = Float8TransformerEnginePolicy()
    converted, _ = convert_model_for_precision(model, policy)
    state_dict = converted.state_dict()
    assert all("wrapped" not in k for k in state_dict), state_dict.keys()
    converted.load_state_dict(state_dict, strict=True)
