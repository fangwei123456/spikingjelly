import inspect
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import ann2snn, neuron, surrogate
from spikingjelly.activation_based.ann2snn import (
    Converter,
    ConversionRecipe,
    HookFactory,
    NeuronFactory,
    RateCodingRecipe,
    ReLURule,
    ThresholdOptimizer,
    TransformerSpikeEquivalentRecipe,
)
from spikingjelly.activation_based.ann2snn.modules import VoltageHook, VoltageScaler
from spikingjelly.activation_based.ann2snn.operators import (
    TDGELU,
    TDLayerNorm,
    TDLinear,
    TDMultiheadAttention,
    TDScaledDotProductAttention,
)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn(self.conv(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CustomConv2d(nn.Conv2d):
    pass


class CustomBatchNorm2d(nn.BatchNorm2d):
    pass


CustomConv2d.__module__ = nn.Conv2d.__module__
CustomBatchNorm2d.__module__ = nn.BatchNorm2d.__module__


class SubclassCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = CustomConv2d(1, 8, 3, padding=1)
        self.bn = CustomBatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn(self.conv(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TwoConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 4, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(4)
        self.conv1 = nn.Conv2d(4, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

    def forward(self, x):
        x = self.bn0(self.conv0(x))
        return self.bn1(self.conv1(x))


class SimpleCNNNoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class UnderscoreModuleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class IdentityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(4, 4)
        self.identity = nn.Identity()
        self.fc1 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc1(self.identity(self.fc0(x)))


class CoreTransformerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.fc0 = nn.Linear(4, 6)
        self.act = nn.GELU(approximate="tanh")
        self.keep = nn.Identity()
        self.fc1 = nn.Linear(6, 4, bias=False)

    def forward(self, x):
        return self.fc1(self.keep(self.act(self.fc0(self.norm(x)))))


class CustomLinear(nn.Linear):
    pass


class CustomLayerNorm(nn.LayerNorm):
    pass


class CustomGELU(nn.GELU):
    pass


CustomLinear.__module__ = nn.Linear.__module__
CustomLayerNorm.__module__ = nn.LayerNorm.__module__
CustomGELU.__module__ = nn.GELU.__module__


class SubclassTransformerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = CustomLayerNorm(4)
        self.fc0 = CustomLinear(4, 6)
        self.act = CustomGELU(approximate="tanh")
        self.fc1 = CustomLinear(6, 4)

    def forward(self, x):
        return self.fc1(self.act(self.fc0(self.norm(x))))


class NoAffineLayerNormMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4, elementwise_affine=False)
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(self.norm(x))


class DropoutCoreMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(self.dropout(x))


class FunctionalDropoutCoreMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc(x)


class SDPABlock(nn.Module):
    def forward(self, query, key, value, attn_mask=None):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )


class SDPAWithExistingTargetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.td_scaled_dot_product_attention_0 = nn.Identity()

    def forward(self, query, key, value):
        query = self.td_scaled_dot_product_attention_0(query)
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
        )


class SDPAPositionalBlock(nn.Module):
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value, None, 0.0, False)


class SDPACausalBlock(nn.Module):
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
        )


class SDPAScaleBlock(nn.Module):
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            scale=0.25,
        )


class SDPANonzeroDropoutBlock(nn.Module):
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value, dropout_p=0.1)


class SDPAEnableGQABlock(nn.Module):
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            enable_gqa=True,
        )


class SelfAttentionBlock(nn.Module):
    def __init__(self, batch_first=True, **kwargs):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            batch_first=batch_first,
            **kwargs,
        )

    def forward(self, x):
        y, _ = self.mha(x, x, x, need_weights=False)
        return y


class CustomMultiheadAttention(nn.MultiheadAttention):
    pass


CustomMultiheadAttention.__module__ = nn.MultiheadAttention.__module__


class SubclassSelfAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = CustomMultiheadAttention(
            embed_dim=8,
            num_heads=2,
            batch_first=True,
        )

    def forward(self, x):
        y, _ = self.mha(x, x, x, need_weights=False)
        return y


class SelfAttentionWithMaskBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            batch_first=True,
        )

    def forward(self, x, attn_mask):
        y, _ = self.mha(x, x, x, need_weights=False, attn_mask=attn_mask)
        return y


class CrossAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            batch_first=True,
        )

    def forward(self, query, key, value, attn_mask=None):
        y, _ = self.mha(
            query,
            key,
            value,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return y


class MHAWithTupleReturn(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            batch_first=True,
        )

    def forward(self, x):
        return self.mha(x, x, x, need_weights=False)


class MHAWithDefaultNeedWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            batch_first=True,
        )

    def forward(self, x):
        y, _ = self.mha(x, x, x)
        return y


class MHAWithKeyPaddingMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            batch_first=True,
        )

    def forward(self, x, key_padding_mask):
        y, _ = self.mha(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return y


class MHAWithAverageWeightsFalse(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            batch_first=True,
        )

    def forward(self, x):
        y, _ = self.mha(
            x,
            x,
            x,
            need_weights=False,
            average_attn_weights=False,
        )
        return y


class MHAWithCausalMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            batch_first=True,
        )

    def forward(self, x, attn_mask):
        y, _ = self.mha(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=attn_mask,
            is_causal=True,
        )
        return y


class MHAWithSeparateKVDim(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            kdim=6,
            vdim=6,
            batch_first=True,
        )

    def forward(self, query, key, value):
        y, _ = self.mha(query, key, value, need_weights=False)
        return y


class MHAWithMissingPackedWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            batch_first=True,
        )
        self.mha.in_proj_weight = None

    def forward(self, x):
        y, _ = self.mha(x, x, x, need_weights=False)
        return y


def _make_loader(batch_size=2, channels=1, h=28, w=28):
    imgs = torch.randn(batch_size, channels, h, w)
    labels = torch.zeros(batch_size, dtype=torch.long)
    return [(imgs, labels)]


def _make_vector_loader(batch_size=2, features=4):
    return [torch.randn(batch_size, features)]


def _rate_converter(
    dataloader=None,
    mode="Max",
    momentum=0.1,
    fuse_flag=False,
    rules=None,
    neuron_factory=None,
    threshold_optimizer=None,
):
    return Converter(
        recipe=RateCodingRecipe(
            dataloader=_make_loader() if dataloader is None else dataloader,
            mode=mode,
            momentum=momentum,
            fuse_flag=fuse_flag,
            rules=rules,
            neuron_factory=neuron_factory,
            threshold_optimizer=threshold_optimizer,
        )
    )


def _td_converter():
    return Converter(recipe=TransformerSpikeEquivalentRecipe())


def _activation_aware_calibration(
    activation: torch.Tensor,
    channel_dim: int = -1,
    threshold_std_scale: float = 3.0,
    eps: float = 1e-6,
):
    if channel_dim < 0:
        channel_dim += activation.dim()
    if channel_dim < 0 or channel_dim >= activation.dim():
        raise ValueError("channel_dim is out of range.")
    reduce_dims = tuple(dim for dim in range(activation.dim()) if dim != channel_dim)
    offset = activation.mean(dim=reduce_dims)
    threshold = activation.std(dim=reduce_dims, unbiased=False) * threshold_std_scale
    threshold = torch.clamp(threshold, min=eps)
    return threshold.detach(), offset.detach()


class _ActivationAwareCalibrationHook(nn.Module):
    def __init__(self, channel_dim: int = -1, eps: float = 1e-6):
        super().__init__()
        self.channel_dim = channel_dim
        self.eps = eps
        self.activations = []

    def forward(self, x):
        self.activations.append(x.detach())
        return x

    def compute_params(self):
        if len(self.activations) == 0:
            raise ValueError("No calibration activations have been recorded.")
        activation = torch.cat(self.activations, dim=0)
        return _activation_aware_calibration(
            activation, channel_dim=self.channel_dim, eps=self.eps
        )


def _apply_cumulative(module, *seq_args, **kwargs):
    cumulative_args = [seq_arg.cumsum(dim=0) for seq_arg in seq_args]
    outputs = [
        module(*(seq_arg[t] for seq_arg in cumulative_args), **kwargs)
        for t in range(cumulative_args[0].shape[0])
    ]
    if isinstance(outputs[0], tuple):
        return tuple(
            (
                torch.stack([output[index] for output in outputs])
                if outputs[0][index] is not None
                else None
            )
            for index in range(len(outputs[0]))
        )
    return torch.stack(outputs)


class TestVoltageHook:
    def test_max_mode(self):
        hook = VoltageHook(mode="Max")
        x = torch.tensor([1.0, 2.0, 3.0])
        hook(x)
        assert hook.scale.item() == pytest.approx(3.0)

    def test_percentile_mode(self):
        hook = VoltageHook(mode="99.9%")
        x = torch.randn(1000)
        hook(x)
        assert hook.scale.item() > 0

    def test_percentile_mode_uses_torch_quantile(self):
        hook = VoltageHook(mode="50%")
        x = torch.arange(101, dtype=torch.float32)

        hook(x)

        assert hook.scale.item() == pytest.approx(torch.quantile(x, 0.5).item())

    def test_out_of_range_percentile_mode_raises(self):
        hook = VoltageHook(mode="101%")

        with pytest.raises(NotImplementedError):
            hook(torch.arange(10, dtype=torch.float32))

    def test_scalar_mode(self):
        hook = VoltageHook(mode=0.5)
        hook(torch.tensor([10.0]))
        assert hook.scale.item() == pytest.approx(5.0)

    def test_ema_momentum(self):
        hook = VoltageHook(momentum=0.5, mode="Max")
        hook(torch.tensor([4.0]))
        hook(torch.tensor([2.0]))
        assert hook.scale.item() == pytest.approx(3.0)

    def test_first_batch_no_ema(self):
        hook = VoltageHook(momentum=0.1, mode="Max")
        hook(torch.tensor([7.0]))
        assert hook.scale.item() == pytest.approx(7.0)

    def test_passthrough(self):
        hook = VoltageHook(mode="Max")
        x = torch.randn(3, 4)
        out = hook(x)
        assert torch.equal(out, x)


class TestVoltageScaler:
    def test_scaling(self):
        scaler = VoltageScaler(2.5)
        x = torch.tensor([1.0, 2.0])
        result = scaler(x)
        assert torch.allclose(result, torch.tensor([2.5, 5.0]))

    def test_identity(self):
        scaler = VoltageScaler(1.0)
        x = torch.randn(3, 4)
        assert torch.allclose(scaler(x), x)

    def test_extra_repr(self):
        scaler = VoltageScaler(3.14)
        assert "3.14" in scaler.extra_repr()


class TestReLURule:
    def test_matches_relu(self):
        rule = ReLURule()
        mod = nn.ReLU()
        modules = {"relu": mod}
        node = SimpleNamespace(op="call_module", target="relu")
        assert rule.match(node, modules)

    def test_rejects_gelu(self):
        rule = ReLURule()
        modules = {"gelu": nn.GELU()}
        node = SimpleNamespace(op="call_module", target="gelu")
        assert not rule.match(node, modules)

    def test_rejects_leaky_relu(self):
        rule = ReLURule()
        modules = {"lrelu": nn.LeakyReLU(0.1)}
        node = SimpleNamespace(op="call_module", target="lrelu")
        assert not rule.match(node, modules)

    def test_rejects_function_node(self):
        rule = ReLURule()
        modules = {}
        node = SimpleNamespace(op="call_function", target="relu")
        assert not rule.match(node, modules)


class TestPublicExports:
    def test_all_contains_new_public_api(self):
        assert set(ann2snn.__all__) == {
            "Converter",
            "ConversionRecipe",
            "RateCodingRecipe",
            "TransformerSpikeEquivalentRecipe",
            "download_url",
            "ReLURule",
            "NeuronFactory",
            "HookFactory",
            "ThresholdOptimizer",
        }

    def test_recipe_api_is_importable(self):
        assert ann2snn.ConversionRecipe is ConversionRecipe
        assert ann2snn.RateCodingRecipe is RateCodingRecipe
        assert (
            ann2snn.TransformerSpikeEquivalentRecipe is TransformerSpikeEquivalentRecipe
        )

    def test_recipe_base_has_no_execution_entrypoint(self):
        assert not hasattr(ConversionRecipe, "convert")
        assert not hasattr(ConversionRecipe, "run")
        assert "__call__" not in ConversionRecipe.__dict__

    def test_recipe_api_has_no_name_metadata(self):
        assert not hasattr(ConversionRecipe, "name")
        assert not hasattr(RateCodingRecipe, "name")
        assert not hasattr(TransformerSpikeEquivalentRecipe, "name")


class TestConverterRecipes:
    def test_converter_signature_is_algorithm_agnostic(self):
        signature = inspect.signature(Converter)
        assert "recipe" in signature.parameters
        assert signature.parameters["recipe"].default is inspect.Signature.empty
        assert "device" in signature.parameters
        algorithm_parameters = {
            "dataloader",
            "mode",
            "momentum",
            "fuse_flag",
            "rules",
            "neuron_factory",
            "threshold_optimizer",
        }
        assert algorithm_parameters.isdisjoint(signature.parameters)

    def test_rate_coding_recipe_object_is_accepted(self):
        converter = Converter(recipe=RateCodingRecipe(dataloader=_make_loader()))
        assert isinstance(converter.recipe, RateCodingRecipe)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dataloader": _make_loader()},
            {"mode": "Max"},
            {"fuse_flag": False},
            {"rules": []},
            {"neuron_factory": NeuronFactory()},
            {"threshold_optimizer": ThresholdOptimizer()},
        ],
    )
    def test_converter_rejects_algorithm_parameters(self, kwargs):
        with pytest.raises(TypeError):
            Converter(recipe=TransformerSpikeEquivalentRecipe(), **kwargs)

    def test_transformer_recipe_does_not_require_dataloader(self):
        converter = Converter(recipe="transformer_spike_equivalent")
        assert isinstance(converter.recipe, TransformerSpikeEquivalentRecipe)

    def test_rate_coding_recipe_name_requires_recipe_object(self):
        with pytest.raises(ValueError, match="rate_coding recipe"):
            Converter(recipe="rate_coding")

    def test_unknown_recipe_raises(self):
        with pytest.raises(ValueError, match="Unknown ann2snn conversion recipe"):
            Converter(recipe="missing_recipe")

    def test_invalid_recipe_object_raises(self):
        with pytest.raises(TypeError, match="recipe must be"):
            Converter(recipe=object())

    def test_none_recipe_raises(self):
        with pytest.raises(TypeError, match="recipe must be"):
            Converter(recipe=None)

    def test_custom_recipe_runs_template_steps_in_order(self):
        class RecordingRecipe(ConversionRecipe):
            def __init__(self):
                self.calls = []

            def validate(self, converter):
                self.calls.append("validate")

            def before_trace(self, converter, ann):
                self.calls.append("before_trace")
                return ann

            def after_trace(self, converter, fx_model):
                self.calls.append("after_trace")
                return fx_model

            def insert_observers(self, converter, fx_model):
                self.calls.append("insert_observers")
                return fx_model

            def calibrate(self, converter, fx_model):
                self.calls.append("calibrate")
                return fx_model

            def replace(self, converter, fx_model):
                self.calls.append("replace")
                return fx_model

            def finalize(self, converter, fx_model):
                self.calls.append("finalize")
                return fx_model

        recipe = RecordingRecipe()
        converter = Converter(recipe=recipe)
        model = nn.Sequential(nn.Linear(2, 2))
        converted = converter.convert(model)

        assert isinstance(converted, torch.fx.GraphModule)
        assert recipe.calls == [
            "validate",
            "before_trace",
            "after_trace",
            "insert_observers",
            "calibrate",
            "replace",
            "finalize",
        ]

    def test_validate_sees_resolved_device(self):
        class DeviceCheckingRecipe(ConversionRecipe):
            def __init__(self):
                self.device = None

            def validate(self, converter):
                self.device = converter.device

        recipe = DeviceCheckingRecipe()
        model = nn.Sequential(nn.Linear(2, 2))

        Converter(recipe=recipe).convert(model)

        assert recipe.device == torch.device("cpu")

    def test_unified_convert_uses_rate_coding_recipe(self):
        model = SimpleCNNNoBN()
        model.eval()
        converter = _rate_converter(fuse_flag=False)

        snn = converter.convert(model)

        assert isinstance(snn, torch.fx.GraphModule)
        assert any(isinstance(m, neuron.IFNode) for m in snn.modules())

    def test_rate_coding_recipe_sets_eval_before_tracing(self):
        model = FunctionalDropoutCoreMLP()
        model.train()
        converter = _rate_converter(
            dataloader=[torch.randn(2, 4)],
            fuse_flag=False,
        )

        converted = converter.convert(model)

        dropout_nodes = [
            node
            for node in converted.graph.nodes
            if node.op == "call_function" and node.target is F.dropout
        ]
        assert len(dropout_nodes) == 1
        assert dropout_nodes[0].kwargs["training"] is False

    def test_unified_convert_uses_transformer_recipe(self):
        model = CoreTransformerMLP()
        model.eval()
        converter = Converter(recipe="transformer_spike_equivalent")

        converted = converter.convert(model)
        modules = dict(converted.named_modules())

        assert isinstance(modules["norm"], TDLayerNorm)
        assert isinstance(modules["fc0"], TDLinear)
        assert isinstance(modules["act"], TDGELU)
        assert isinstance(modules["fc1"], TDLinear)


class TestNeuronFactory:
    def test_default_creates_ifnode(self):
        factory = NeuronFactory()
        n = factory.create(scale=5.0)
        assert isinstance(n, neuron.IFNode)
        assert n.v_threshold == 1.0
        assert n.v_reset is None

    def test_custom_threshold(self):
        factory = NeuronFactory(v_threshold=2.0)
        n = factory.create(scale=5.0)
        assert n.v_threshold == 2.0

    def test_custom_neuron_type(self):
        factory = NeuronFactory(neuron_type=neuron.LIFNode, tau=2.0)
        n = factory.create(scale=1.0)
        assert isinstance(n, neuron.LIFNode)


class TestHookFactory:
    def test_default_mode(self):
        factory = HookFactory()
        hook = factory.create()
        assert hook.mode == "Max"
        assert hook.momentum == 0.1

    def test_custom_mode(self):
        factory = HookFactory(mode="99.9%", momentum=0.5)
        hook = factory.create()
        assert hook.mode == "99.9%"
        assert hook.momentum == 0.5


class TestThresholdOptimizer:
    def test_fixed_returns_scale(self):
        opt = ThresholdOptimizer("fixed")
        hook = VoltageHook(mode="Max")
        hook(torch.tensor([5.0]))
        assert opt.compute_threshold(hook) == pytest.approx(5.0)

    def test_unknown_strategy_raises_when_computed(self):
        opt = ThresholdOptimizer("nonexistent")
        hook = VoltageHook(mode="Max")
        hook(torch.tensor([5.0]))
        with pytest.raises(NotImplementedError):
            opt.compute_threshold(hook)

    def test_default_is_fixed(self):
        opt = ThresholdOptimizer()
        assert opt.strategy == "fixed"

    def test_subclass_can_store_custom_strategy(self):
        class CustomOptimizer(ThresholdOptimizer):
            def compute_threshold(self, hook):
                assert self.strategy == "custom"
                return 3.0

        opt = CustomOptimizer("custom")
        hook = VoltageHook(mode="Max")
        hook(torch.tensor([5.0]))
        assert opt.compute_threshold(hook) == pytest.approx(3.0)


class TestConverterBackwardCompat:
    def test_converter_is_plain_conversion_driver(self):
        converter = _rate_converter(mode="Max", fuse_flag=False)

        assert not isinstance(converter, nn.Module)
        with pytest.raises(TypeError):
            converter(SimpleCNN())

    def test_relu_model_converts(self):
        model = SimpleCNN()
        model.eval()
        converter = _rate_converter(mode="Max", fuse_flag=False)
        snn = converter.convert(model)
        assert snn is not None

    def test_output_shape_preserved(self):
        model = SimpleCNN()
        model.eval()
        dummy = torch.randn(1, 1, 28, 28)
        converter = _rate_converter(mode="Max", fuse_flag=False)
        snn = converter.convert(model)
        out = snn(dummy)
        assert out.shape == (1, 10)

    def test_fuse_conv_bn(self):
        model = SimpleCNN()
        model.eval()
        converter = _rate_converter(mode="Max", fuse_flag=True)
        snn = converter.convert(model)
        assert snn is not None

    def test_mode_max(self):
        model = SimpleCNN()
        model.eval()
        converter = _rate_converter(mode="max")
        snn = converter.convert(model)
        assert snn is not None

    def test_mode_robust(self):
        model = SimpleCNN()
        model.eval()
        converter = _rate_converter(mode="99.9%")
        snn = converter.convert(model)
        assert snn is not None

    def test_mode_scalar(self):
        model = SimpleCNN()
        model.eval()
        converter = _rate_converter(mode=0.5)
        snn = converter.convert(model)
        assert snn is not None

    def test_mode_integer_one(self):
        model = SimpleCNN()
        model.eval()
        converter = _rate_converter(mode=1)
        snn = converter.convert(model)
        assert snn is not None

    def test_invalid_mode_raises(self):
        recipe = RateCodingRecipe(dataloader=[], mode="invalid")
        with pytest.raises(NotImplementedError):
            Converter(recipe=recipe).convert(nn.Identity())

    def test_empty_mode_raises(self):
        recipe = RateCodingRecipe(dataloader=[], mode="")
        with pytest.raises(NotImplementedError):
            Converter(recipe=recipe).convert(nn.Identity())

    def test_invalid_scalar_raises(self):
        recipe = RateCodingRecipe(dataloader=[], mode=1.5)
        with pytest.raises(NotImplementedError):
            Converter(recipe=recipe).convert(nn.Identity())

    def test_invalid_scalar_zero_raises(self):
        recipe = RateCodingRecipe(dataloader=[], mode=0.0)
        with pytest.raises(NotImplementedError):
            Converter(recipe=recipe).convert(nn.Identity())

    def test_invalid_percentile_raises(self):
        recipe = RateCodingRecipe(dataloader=[], mode="101%")
        with pytest.raises(NotImplementedError):
            Converter(recipe=recipe).convert(nn.Identity())

    def test_invalid_scalar_rejected_when_asserts_are_optimized(self):
        code = """
from spikingjelly.activation_based.ann2snn import RateCodingRecipe

recipe = RateCodingRecipe(dataloader=[], mode=1.5)
try:
    recipe.validate(None)
except NotImplementedError:
    raise SystemExit(0)
raise SystemExit(1)
"""
        result = subprocess.run(
            [sys.executable, "-O", "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_tensor_dataloader_converts(self):
        model = SimpleCNNNoBN()
        model.eval()
        imgs = torch.randn(2, 1, 28, 28)
        converter = _rate_converter(dataloader=[imgs], mode="Max", fuse_flag=False)
        snn = converter.convert(model)
        assert snn is not None

    def test_numpy_dataloader_converts_full_batch(self):
        model = SimpleCNNNoBN()
        model.eval()
        imgs = np.random.randn(2, 1, 28, 28).astype(np.float32)
        converter = _rate_converter(dataloader=[imgs], mode="Max", fuse_flag=False)
        snn = converter.convert(model)
        assert snn is not None

    def test_dict_dataloader_converts(self):
        model = SimpleCNNNoBN()
        model.eval()
        imgs = torch.randn(2, 1, 28, 28)
        converter = _rate_converter(
            dataloader=[{"input": imgs}], mode="Max", fuse_flag=False
        )
        snn = converter.convert(model)
        assert snn is not None

    def test_dict_dataloader_prefers_input_key(self):
        model = SimpleCNNNoBN()
        model.eval()
        imgs = torch.randn(2, 1, 28, 28)
        labels = torch.zeros(2, dtype=torch.long)
        converter = _rate_converter(
            dataloader=[{"label": labels, "input": imgs}],
            mode="Max",
            fuse_flag=False,
        )
        snn = converter.convert(model)
        assert snn is not None

    def test_dict_dataloader_prefers_images_key(self):
        imgs = torch.randn(2, 1, 28, 28)
        labels = torch.zeros(2, dtype=torch.long)

        extracted = RateCodingRecipe._extract_batch_input(
            {"labels": labels, "images": imgs}
        )

        assert extracted is imgs

    def test_nested_dataloader_extracts_tensor(self):
        imgs = torch.randn(2, 1, 28, 28)

        extracted = RateCodingRecipe._extract_batch_input(({"input": imgs},))

        assert extracted is imgs

    def test_nested_dict_dataloader_extracts_tensor(self):
        imgs = torch.randn(2, 1, 28, 28)

        extracted = RateCodingRecipe._extract_batch_input(
            {"labels": torch.zeros(2), "images": (imgs,)}
        )

        assert extracted is imgs

    def test_calibration_preserves_tensor_dtype(self):
        model = SimpleCNNNoBN().double()
        model.eval()
        imgs = torch.randn(2, 1, 28, 28, dtype=torch.float64)
        converter = _rate_converter(dataloader=[imgs], mode="Max", fuse_flag=False)

        snn = converter.convert(model)

        assert snn is not None

    def test_extract_batch_input_rejects_empty_sequence(self):
        with pytest.raises(ValueError, match="empty list or tuple"):
            RateCodingRecipe._extract_batch_input([])

    def test_extract_batch_input_rejects_empty_dict(self):
        with pytest.raises(ValueError, match="empty dictionary"):
            RateCodingRecipe._extract_batch_input({})


class TestConverterTDOperatorReplacement:
    def test_replaces_core_modules_with_td_operators(self):
        model = CoreTransformerMLP()
        model.eval()

        converted = _td_converter().convert(model)
        modules = dict(converted.named_modules())

        assert isinstance(modules["norm"], TDLayerNorm)
        assert isinstance(modules["fc0"], TDLinear)
        assert isinstance(modules["act"], TDGELU)
        assert isinstance(modules["fc1"], TDLinear)
        assert isinstance(modules["keep"], nn.Identity)

    def test_replaces_module_subclasses_with_td_operators(self):
        model = SubclassTransformerMLP()
        model.eval()

        converted = _td_converter().convert(model)
        modules = dict(converted.named_modules())

        assert isinstance(modules["norm"], TDLayerNorm)
        assert isinstance(modules["fc0"], TDLinear)
        assert isinstance(modules["act"], TDGELU)
        assert isinstance(modules["fc1"], TDLinear)

    def test_copies_linear_layernorm_and_gelu_configuration(self):
        model = CoreTransformerMLP()
        model.eval()

        converted = _td_converter().convert(model)
        modules = dict(converted.named_modules())

        assert torch.equal(modules["fc0"].weight, model.fc0.weight)
        assert torch.equal(modules["fc0"].bias, model.fc0.bias)
        assert torch.equal(modules["fc1"].weight, model.fc1.weight)
        assert modules["fc1"].bias is None
        assert torch.equal(modules["norm"].weight, model.norm.weight)
        assert torch.equal(modules["norm"].bias, model.norm.bias)
        assert modules["norm"].normalized_shape == model.norm.normalized_shape
        assert modules["norm"].eps == model.norm.eps
        assert modules["act"].approximate == "tanh"

    def test_cumulative_output_matches_ann_reference(self):
        model = CoreTransformerMLP()
        model.eval()
        converted = _td_converter().convert(model)
        x_seq = torch.randn(5, 2, 3, 4)

        y_seq = converted(x_seq)
        expected = model(x_seq.cumsum(dim=0))

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)

    def test_no_affine_layernorm_is_preserved(self):
        model = NoAffineLayerNormMLP()
        model.eval()

        converted = _td_converter().convert(model)
        modules = dict(converted.named_modules())

        assert isinstance(modules["norm"], TDLayerNorm)
        assert modules["norm"].weight is None
        assert modules["norm"].bias is None
        assert isinstance(modules["fc"], TDLinear)

    def test_td_operator_replacement_preserves_training_mode(self):
        model = DropoutCoreMLP()
        model.train()

        converted = _td_converter().convert(model)

        assert converted.training
        assert converted.dropout.training
        assert converted.fc.training

    def test_td_operator_replacement_preserves_eval_mode(self):
        model = DropoutCoreMLP()
        model.eval()

        converted = _td_converter().convert(model)

        assert not converted.training
        assert not converted.dropout.training
        assert not converted.fc.training

    def test_rewrites_sdpa_function_node(self):
        model = SDPABlock()

        converted = _td_converter().convert(model)
        modules = dict(converted.named_modules())

        assert any(
            isinstance(module, TDScaledDotProductAttention)
            for module in modules.values()
        )
        assert not any(
            node.op == "call_function" and node.target is F.scaled_dot_product_attention
            for node in converted.graph.nodes
        )

    def test_sdpa_rewrite_avoids_existing_target_name(self):
        model = SDPAWithExistingTargetBlock()

        converted = _td_converter().convert(model)
        modules = dict(converted.named_modules())

        assert isinstance(modules["td_scaled_dot_product_attention_0"], nn.Identity)
        assert isinstance(
            modules["td_scaled_dot_product_attention_1"],
            TDScaledDotProductAttention,
        )

    def test_sdpa_cumulative_output_matches_ann_reference(self):
        model = SDPABlock()
        converted = _td_converter().convert(model)
        query_seq = torch.randn(4, 2, 3, 5, 8)
        key_seq = torch.randn(4, 2, 3, 6, 8)
        value_seq = torch.randn(4, 2, 3, 6, 7)

        y_seq = converted(query_seq, key_seq, value_seq)
        expected = model(
            query_seq.cumsum(dim=0),
            key_seq.cumsum(dim=0),
            value_seq.cumsum(dim=0),
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_sdpa_rewrite_supports_mask_causal_scale_and_positional_args(self):
        query_seq = torch.randn(4, 2, 3, 5, 8)
        key_seq = torch.randn(4, 2, 3, 5, 8)
        value_seq = torch.randn(4, 2, 3, 5, 7)
        mask = torch.ones(5, 5, dtype=torch.bool).tril()

        masked = _td_converter().convert(SDPABlock())
        y_masked = masked(query_seq, key_seq, value_seq, mask)
        expected_masked = F.scaled_dot_product_attention(
            query_seq.cumsum(dim=0),
            key_seq.cumsum(dim=0),
            value_seq.cumsum(dim=0),
            attn_mask=mask,
            dropout_p=0.0,
        )
        assert torch.allclose(
            y_masked.cumsum(dim=0), expected_masked, atol=1e-6, rtol=1e-6
        )

        causal_model = SDPACausalBlock()
        causal = _td_converter().convert(causal_model)
        y_causal = causal(query_seq, key_seq, value_seq)
        expected_causal = causal_model(
            query_seq.cumsum(dim=0),
            key_seq.cumsum(dim=0),
            value_seq.cumsum(dim=0),
        )
        assert torch.allclose(
            y_causal.cumsum(dim=0), expected_causal, atol=1e-6, rtol=1e-6
        )

        scaled_model = SDPAScaleBlock()
        scaled = _td_converter().convert(scaled_model)
        y_scaled = scaled(query_seq, key_seq, value_seq)
        expected_scaled = scaled_model(
            query_seq.cumsum(dim=0),
            key_seq.cumsum(dim=0),
            value_seq.cumsum(dim=0),
        )
        assert torch.allclose(
            y_scaled.cumsum(dim=0), expected_scaled, atol=1e-6, rtol=1e-6
        )

        positional_model = SDPAPositionalBlock()
        positional = _td_converter().convert(positional_model)
        y_positional = positional(query_seq, key_seq, value_seq)
        expected_positional = positional_model(
            query_seq.cumsum(dim=0),
            key_seq.cumsum(dim=0),
            value_seq.cumsum(dim=0),
        )
        assert torch.allclose(
            y_positional.cumsum(dim=0),
            expected_positional,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_sdpa_rewrite_rejects_unsupported_options(self):
        with pytest.raises(ValueError, match="dropout_p=0.0"):
            _td_converter().convert(SDPANonzeroDropoutBlock())
        with pytest.raises(ValueError, match="enable_gqa"):
            _td_converter().convert(SDPAEnableGQABlock())

    def test_replaces_mha_and_copies_parameters(self):
        model = SelfAttentionBlock()
        model.eval()

        converted = _td_converter().convert(model)
        td_mha = converted.mha
        embed_dim = model.mha.embed_dim

        assert isinstance(td_mha, TDMultiheadAttention)
        assert torch.equal(td_mha.q_proj.weight, model.mha.in_proj_weight[:embed_dim])
        assert torch.equal(
            td_mha.k_proj.weight,
            model.mha.in_proj_weight[embed_dim : 2 * embed_dim],
        )
        assert torch.equal(
            td_mha.v_proj.weight, model.mha.in_proj_weight[2 * embed_dim :]
        )
        assert torch.equal(td_mha.out_proj.weight, model.mha.out_proj.weight)
        assert torch.equal(td_mha.q_proj.bias, model.mha.in_proj_bias[:embed_dim])
        assert torch.equal(
            td_mha.k_proj.bias,
            model.mha.in_proj_bias[embed_dim : 2 * embed_dim],
        )
        assert torch.equal(td_mha.v_proj.bias, model.mha.in_proj_bias[2 * embed_dim :])
        assert torch.equal(td_mha.out_proj.bias, model.mha.out_proj.bias)

    def test_replaces_mha_subclass(self):
        model = SubclassSelfAttentionBlock()
        model.eval()

        converted = _td_converter().convert(model)

        assert isinstance(converted.mha, TDMultiheadAttention)

    def test_mha_replacement_preserves_training_mode(self):
        model = SelfAttentionBlock()
        model.train()

        converted = _td_converter().convert(model)

        assert converted.training
        assert converted.mha.training

    def test_mha_self_attention_cumulative_output_matches_ann_reference(self):
        model = SelfAttentionBlock()
        model.eval()
        converted = _td_converter().convert(model)
        x_seq = torch.randn(4, 2, 5, 8)

        y_seq = converted(x_seq)
        expected = _apply_cumulative(model, x_seq)

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)

    def test_mha_cross_attention_cumulative_output_matches_ann_reference(self):
        model = CrossAttentionBlock()
        model.eval()
        converted = _td_converter().convert(model)
        query_seq = torch.randn(4, 2, 3, 8)
        key_seq = torch.randn(4, 2, 5, 8)
        value_seq = torch.randn(4, 2, 5, 8)
        mask = torch.ones(3, 5, dtype=torch.bool)

        y_seq = converted(query_seq, key_seq, value_seq, mask)
        expected = _apply_cumulative(
            model,
            query_seq,
            key_seq,
            value_seq,
            attn_mask=mask,
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)

    def test_mha_replacement_supports_three_dimensional_attention_mask(self):
        model = SelfAttentionWithMaskBlock()
        model.eval()
        converted = _td_converter().convert(model)
        x_seq = torch.randn(4, 2, 5, 8)
        mask = torch.zeros(4, 5, 5)
        mask[:, :, -1] = float("-inf")

        y_seq = converted(x_seq, mask)
        expected = _apply_cumulative(model, x_seq, attn_mask=mask)

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)

    def test_mha_tuple_return_graph_remains_valid(self):
        model = MHAWithTupleReturn()
        model.eval()
        converted = _td_converter().convert(model)
        x_seq = torch.randn(4, 2, 5, 8)

        y_seq, weights = converted(x_seq)
        expected, expected_weights = _apply_cumulative(model, x_seq)

        assert weights is None
        assert expected_weights is None
        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)

    def test_mha_rewrite_rejects_unsupported_configurations(self):
        with pytest.raises(ValueError, match="dropout=0.0"):
            _td_converter().convert(SelfAttentionBlock(dropout=0.1))
        with pytest.raises(ValueError, match="batch_first=True"):
            _td_converter().convert(SelfAttentionBlock(batch_first=False))
        with pytest.raises(ValueError, match="kdim == vdim == embed_dim"):
            _td_converter().convert(MHAWithSeparateKVDim())
        with pytest.raises(ValueError, match="packed in_proj_weight"):
            _td_converter().convert(MHAWithMissingPackedWeight())
        with pytest.raises(ValueError, match="add_bias_kv"):
            _td_converter().convert(SelfAttentionBlock(add_bias_kv=True))
        with pytest.raises(ValueError, match="add_zero_attn"):
            _td_converter().convert(SelfAttentionBlock(add_zero_attn=True))

    def test_mha_rewrite_rejects_unsupported_calls(self):
        with pytest.raises(ValueError, match="need_weights=False"):
            _td_converter().convert(MHAWithDefaultNeedWeights())
        with pytest.raises(ValueError, match="key_padding_mask"):
            _td_converter().convert(MHAWithKeyPaddingMask())
        with pytest.raises(ValueError, match="average_attn_weights=False"):
            _td_converter().convert(MHAWithAverageWeightsFalse())

    def test_mha_causal_attention_rejects_explicit_mask_at_runtime(self):
        model = MHAWithCausalMask()
        converted = _td_converter().convert(model)
        x_seq = torch.randn(4, 2, 5, 8)
        mask = torch.zeros(5, 5)

        with pytest.raises(ValueError, match="attn_mask"):
            converted(x_seq, mask)


class TestFuse:
    def test_conv_bn_fusion_matches_eval_output(self):
        model = SimpleCNN()
        model.eval()
        fx_model = torch.fx.symbolic_trace(model)
        x = torch.randn(2, 1, 28, 28)
        expected = fx_model(x)

        fused = RateCodingRecipe._fuse(fx_model, fuse_flag=True)
        result = fused(x)

        assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)
        assert not any(isinstance(m, nn.BatchNorm2d) for m in fused.modules())

    def test_conv_bn_fusion_supports_module_subclasses(self):
        model = SubclassCNN()
        model.eval()
        fx_model = torch.fx.symbolic_trace(model)
        x = torch.randn(2, 1, 28, 28)
        expected = fx_model(x)

        fused = RateCodingRecipe._fuse(fx_model, fuse_flag=True)
        result = fused(x)

        assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)
        assert not any(isinstance(m, nn.BatchNorm2d) for m in fused.modules())

    def test_conv_bn_fusion_handles_multiple_pairs(self):
        model = TwoConvBN()
        model.eval()
        fx_model = torch.fx.symbolic_trace(model)
        x = torch.randn(2, 1, 28, 28)
        expected = fx_model(x)

        fused = RateCodingRecipe._fuse(fx_model, fuse_flag=True)
        result = fused(x)

        assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)
        assert not any(isinstance(m, nn.BatchNorm2d) for m in fused.modules())

    def test_no_bn_model_fuse(self):
        model = SimpleCNNNoBN()
        model.eval()
        converter = _rate_converter(mode="Max", fuse_flag=True)
        snn = converter.convert(model)
        assert snn is not None

    def test_fuse_flag_false(self):
        model = SimpleCNN()
        model.eval()
        converter = _rate_converter(mode="Max", fuse_flag=False)
        snn = converter.convert(model)
        assert snn is not None


class TestRuleBasedConversion:
    def test_activation_aware_calibration_channel_last_tensor(self):
        activation = torch.tensor(
            [
                [1.0, 2.0, 4.0],
                [3.0, 2.0, 10.0],
                [5.0, 2.0, 16.0],
            ]
        )

        threshold, offset = _activation_aware_calibration(
            activation, channel_dim=-1, eps=0.5
        )

        expected_offset = torch.tensor([3.0, 2.0, 10.0])
        expected_threshold = torch.clamp(
            activation.std(dim=0, unbiased=False) * 3.0, min=0.5
        )
        assert torch.allclose(offset, expected_offset)
        assert torch.allclose(threshold, expected_threshold)
        assert torch.isfinite(threshold).all()
        assert torch.isfinite(offset).all()
        assert (threshold > 0).all()

    def test_activation_aware_calibration_supports_time_and_nchw_shapes(self):
        time_major = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
        nchw = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).view(2, 3, 2, 2)

        threshold_t, offset_t = _activation_aware_calibration(time_major)
        threshold_nchw, offset_nchw = _activation_aware_calibration(nchw, channel_dim=1)

        assert threshold_t.shape == (3,)
        assert offset_t.shape == (3,)
        assert threshold_nchw.shape == (3,)
        assert offset_nchw.shape == (3,)
        assert torch.allclose(offset_t, time_major.mean(dim=(0, 1)))
        assert torch.allclose(offset_nchw, nchw.mean(dim=(0, 2, 3)))

    def test_activation_aware_calibration_clamps_constant_channels(self):
        activation = torch.ones(4, 3)

        threshold, offset = _activation_aware_calibration(activation, eps=0.125)

        assert torch.allclose(offset, torch.ones(3))
        assert torch.allclose(threshold, torch.full((3,), 0.125))

    def test_activation_aware_calibration_rejects_invalid_channel_dim(self):
        with pytest.raises(ValueError, match="out of range"):
            _activation_aware_calibration(torch.ones(2, 3), channel_dim=2)

    def test_custom_rules_list(self):
        model = SimpleCNN()
        model.eval()
        converter = _rate_converter(
            rules=[ReLURule()],
            fuse_flag=False,
        )
        snn = converter.convert(model)
        assert snn is not None
        assert any(isinstance(m, neuron.IFNode) for m in snn.modules())

    def test_custom_neuron_factory(self):
        model = SimpleCNN()
        model.eval()
        converter = _rate_converter(
            neuron_factory=NeuronFactory(v_threshold=2.0),
            fuse_flag=False,
        )
        snn = converter.convert(model)
        assert snn is not None
        if_nodes = [m for m in snn.modules() if isinstance(m, neuron.IFNode)]
        assert len(if_nodes) == 1
        assert if_nodes[0].v_threshold == 2.0

    def test_custom_neuron_threshold_updates_input_scaler(self):
        model = SimpleCNNNoBN()
        model.eval()
        converter = _rate_converter(
            neuron_factory=NeuronFactory(v_threshold=2.0),
            threshold_optimizer=ThresholdOptimizer("fixed"),
            fuse_flag=False,
        )
        snn = converter.convert(model)

        scalers = [m for m in snn.modules() if isinstance(m, VoltageScaler)]
        if_nodes = [m for m in snn.modules() if isinstance(m, neuron.IFNode)]
        assert len(scalers) == 2
        assert len(if_nodes) == 1
        assert scalers[0].scale.item() == pytest.approx(
            if_nodes[0].v_threshold / scalers[1].scale.item()
        )

    def test_empty_rules_leaves_relu(self):
        model = SimpleCNNNoBN()
        model.eval()
        converter = _rate_converter(
            rules=[],
            fuse_flag=False,
        )
        snn = converter.convert(model)
        assert any(isinstance(m, nn.ReLU) for m in snn.modules())
        assert not any(isinstance(m, neuron.IFNode) for m in snn.modules())

    def test_custom_rule_is_used(self):
        class CountingRule(ReLURule):
            def __init__(self):
                self.match_count = 0
                self.insert_count = 0
                self.replace_count = 0

            def match(self, node, modules):
                matched = super().match(node, modules)
                if matched:
                    self.match_count += 1
                return matched

            def insert_hooks(self, *args, **kwargs):
                self.insert_count += 1
                return super().insert_hooks(*args, **kwargs)

            def find_replacements(self, *args, **kwargs):
                self.find_count = getattr(self, "find_count", 0) + 1
                return super().find_replacements(*args, **kwargs)

            def replace_with_neurons(self, *args, **kwargs):
                self.replace_count += 1
                return super().replace_with_neurons(*args, **kwargs)

        rule = CountingRule()
        model = SimpleCNNNoBN()
        model.eval()
        _rate_converter(
            rules=[rule],
            fuse_flag=False,
        ).convert(model)

        assert rule.match_count >= 2
        assert rule.insert_count == 1
        assert rule.find_count == 1
        assert rule.replace_count == 1

    def test_duck_typed_activation_rule_replaces_identity(self):
        class IdentityRule:
            def __init__(self):
                self.match_count = 0
                self.insert_count = 0
                self.find_count = 0
                self.replace_count = 0
                self.threshold = None

            def match(self, node, modules):
                if (
                    node.op == "call_module"
                    and node.target in modules
                    and type(modules[node.target]) is nn.Identity
                ):
                    self.match_count += 1
                    return True
                return False

            def insert_hooks(
                self, fx_model, node, hook_factory, hook_counts_per_prefix
            ):
                self.insert_count += 1
                target = f"{node.target}_voltage_hook"
                fx_model.add_submodule(target, hook_factory.create())
                with fx_model.graph.inserting_after(node):
                    return fx_model.graph.call_module(target, args=(node,))

            def find_replacements(self, fx_model, modules):
                self.find_count += 1
                for hook_node in fx_model.graph.nodes:
                    if hook_node.op != "call_module":
                        continue
                    if not isinstance(modules.get(hook_node.target), VoltageHook):
                        continue
                    activation_node = hook_node.args[0]
                    if self.match(activation_node, modules):
                        yield activation_node, hook_node

            def replace_with_neurons(
                self,
                fx_model,
                activation_node,
                hook_node,
                neuron_factory,
                threshold_optimizer,
            ):
                self.replace_count += 1
                hook = fx_model.get_submodule(hook_node.target)
                self.threshold = threshold_optimizer.compute_threshold(hook)
                target = f"{activation_node.target}_spiking_identity"
                fx_model.add_submodule(target, nn.Identity())
                with fx_model.graph.inserting_after(hook_node):
                    new_node = fx_model.graph.call_module(
                        target, args=activation_node.args
                    )
                hook_node.replace_all_uses_with(new_node)
                activation_node.replace_all_uses_with(new_node)
                fx_model.graph.erase_node(hook_node)
                fx_model.graph.erase_node(activation_node)

        rule = IdentityRule()
        model = IdentityMLP()
        model.eval()
        with torch.no_grad():
            model.fc0.weight.zero_()
            model.fc0.bias.fill_(1.0)
        snn = _rate_converter(
            dataloader=_make_vector_loader(),
            rules=[rule],
            fuse_flag=False,
        ).convert(model)

        assert rule.match_count >= 2
        assert rule.insert_count == 1
        assert rule.find_count == 1
        assert rule.replace_count == 1
        assert rule.threshold > 0
        assert "identity_spiking_identity" in dict(snn.named_modules())
        assert "identity_voltage_hook" not in dict(snn.named_modules())
        x = torch.randn(2, 4)
        assert torch.allclose(snn(x), model(x), atol=0.0, rtol=0.0)

    def test_set_voltagehook_refreshes_modules_after_rule_insert(self):
        class TwoReLUCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu0 = nn.ReLU()
                self.relu1 = nn.ReLU()

            def forward(self, x):
                return self.relu1(self.relu0(x))

        class MarkerRule:
            def __init__(self):
                self.inserted = False

            def match(self, node, modules):
                return not self.inserted and node.target == "relu0"

            def insert_hooks(self, fx_model, node, *args, **kwargs):
                self.inserted = True
                fx_model.add_submodule("marker", nn.Identity())

        class MarkerAwareRule:
            def __init__(self):
                self.saw_marker = False

            def match(self, node, modules):
                if node.target == "relu1" and "marker" in modules:
                    self.saw_marker = True
                return False

        marker_rule = MarkerRule()
        marker_aware_rule = MarkerAwareRule()
        fx_model = torch.fx.symbolic_trace(TwoReLUCNN())

        RateCodingRecipe(
            dataloader=[],
            rules=[marker_rule, marker_aware_rule],
        )._set_voltagehook(fx_model)

        assert marker_rule.inserted
        assert marker_aware_rule.saw_marker

    def test_duplicate_activation_replacements_are_skipped(self):
        class DuplicateActivationRule(ReLURule):
            def __init__(self):
                self.replace_count = 0

            def find_replacements(self, fx_model, modules):
                replacements = list(super().find_replacements(fx_model, modules))
                if len(replacements) != 1:
                    return iter(replacements)
                activation_node, hook_node = replacements[0]
                return iter(
                    [
                        (activation_node, hook_node),
                        (activation_node, object()),
                    ]
                )

            def replace_with_neurons(self, *args, **kwargs):
                self.replace_count += 1
                return super().replace_with_neurons(*args, **kwargs)

        rule = DuplicateActivationRule()
        model = SimpleCNNNoBN()
        model.eval()
        snn = _rate_converter(
            rules=[rule],
            fuse_flag=False,
        ).convert(model)

        assert rule.replace_count == 1
        assert any(isinstance(m, neuron.IFNode) for m in snn.modules())

    def test_threshold_optimizer_is_used(self):
        class FixedScaleOptimizer:
            def __init__(self):
                self.called = False

            def compute_threshold(self, hook):
                self.called = True
                return 2.0

        optimizer = FixedScaleOptimizer()
        model = SimpleCNNNoBN()
        model.eval()
        snn = _rate_converter(
            threshold_optimizer=optimizer,
            fuse_flag=False,
        ).convert(model)

        assert optimizer.called
        scalers = [m for m in snn.modules() if isinstance(m, VoltageScaler)]
        assert len(scalers) == 2
        assert scalers[0].scale.item() == pytest.approx(0.5)
        assert scalers[1].scale.item() == pytest.approx(2.0)

    def test_tensor_neuron_threshold_updates_input_scaler(self):
        class TensorThresholdFactory(NeuronFactory):
            def create(self, scale):
                n = super().create(scale)
                n.v_threshold = torch.tensor(2.0)
                return n

        model = SimpleCNNNoBN()
        model.eval()
        snn = _rate_converter(
            neuron_factory=TensorThresholdFactory(),
            threshold_optimizer=ThresholdOptimizer("fixed"),
            fuse_flag=False,
        ).convert(model)

        scalers = [m for m in snn.modules() if isinstance(m, VoltageScaler)]
        if_nodes = [m for m in snn.modules() if isinstance(m, neuron.IFNode)]
        assert len(scalers) == 2
        assert len(if_nodes) == 1
        assert scalers[0].scale.item() == pytest.approx(
            if_nodes[0].v_threshold.item() / scalers[1].scale.item()
        )

    def test_custom_rule_can_insert_activation_aware_ifnode(self):
        class ActivationAwareReLURule(ReLURule):
            def replace_with_neurons(
                self,
                fx_model,
                activation_node,
                hook_node,
                neuron_factory,
                threshold_optimizer,
            ):
                hook = fx_model.get_submodule(hook_node.target)
                scale = float(threshold_optimizer.compute_threshold(hook))
                target = f"{activation_node.target}_activation_aware_if"
                fx_model.add_submodule(
                    target,
                    neuron.ActivationAwareIFNode(
                        v_threshold=torch.full((8,), scale),
                        v_offset=torch.zeros(8),
                        channel_dim=1,
                        surrogate_function=surrogate.DeterministicPass(),
                    ),
                )
                with fx_model.graph.inserting_after(hook_node):
                    new_node = fx_model.graph.call_module(
                        target, args=activation_node.args
                    )
                hook_node.replace_all_uses_with(new_node)
                activation_node.replace_all_uses_with(new_node)
                fx_model.graph.erase_node(hook_node)
                fx_model.graph.erase_node(activation_node)

        model = SimpleCNNNoBN()
        model.eval()
        snn = _rate_converter(
            rules=[ActivationAwareReLURule()],
            fuse_flag=False,
        ).convert(model)

        assert any(isinstance(m, neuron.ActivationAwareIFNode) for m in snn.modules())
        assert not any(isinstance(m, neuron.IFNode) for m in snn.modules())
        x = torch.rand(2, 1, 28, 28)
        y = snn(x)
        assert y.shape == (2, 10)

    def test_custom_rule_can_calibrate_activation_aware_ifnode(self):
        class ActivationAwareCalibrationRule(ReLURule):
            def insert_hooks(
                self, fx_model, node, hook_factory, hook_counts_per_prefix
            ):
                if not isinstance(node.target, str):
                    raise TypeError("node.target must be a module path string.")
                parent, _, _ = node.target.rpartition(".")
                target = (
                    f"{parent}.activation_aware_hook"
                    if parent
                    else "activation_aware_hook"
                )
                fx_model.add_submodule(target, _ActivationAwareCalibrationHook(1))
                with fx_model.graph.inserting_after(node):
                    return fx_model.graph.call_module(target, args=(node,))

            def find_replacements(self, fx_model, modules):
                for hook_node in fx_model.graph.nodes:
                    if hook_node.op != "call_module":
                        continue
                    if not isinstance(
                        modules.get(hook_node.target), _ActivationAwareCalibrationHook
                    ):
                        continue
                    activation_node = hook_node.args[0]
                    if isinstance(activation_node, torch.fx.Node) and self.match(
                        activation_node, modules
                    ):
                        yield activation_node, hook_node

            def replace_with_neurons(
                self,
                fx_model,
                activation_node,
                hook_node,
                neuron_factory,
                threshold_optimizer,
            ):
                hook = fx_model.get_submodule(hook_node.target)
                threshold, offset = hook.compute_params()
                target = f"{activation_node.target}_activation_aware_if"
                fx_model.add_submodule(
                    target,
                    neuron.ActivationAwareIFNode(
                        v_threshold=threshold,
                        v_offset=offset,
                        channel_dim=1,
                        surrogate_function=surrogate.DeterministicPass(),
                    ),
                )
                with fx_model.graph.inserting_after(hook_node):
                    new_node = fx_model.graph.call_module(
                        target, args=activation_node.args
                    )
                hook_node.replace_all_uses_with(new_node)
                activation_node.replace_all_uses_with(new_node)
                fx_model.graph.erase_node(hook_node)
                fx_model.graph.erase_node(activation_node)

        loader = _make_loader(batch_size=4)
        model = SimpleCNNNoBN()
        model.eval()
        snn = _rate_converter(
            dataloader=loader,
            rules=[ActivationAwareCalibrationRule()],
            fuse_flag=False,
        ).convert(model)

        aa_nodes = [
            m for m in snn.modules() if isinstance(m, neuron.ActivationAwareIFNode)
        ]
        assert len(aa_nodes) == 1
        assert not any(isinstance(m, neuron.IFNode) for m in snn.modules())
        assert aa_nodes[0].v_threshold.shape == (8,)
        assert aa_nodes[0].v_offset.shape == (8,)
        assert torch.isfinite(aa_nodes[0].v_threshold).all()
        assert torch.isfinite(aa_nodes[0].v_offset).all()
        assert (aa_nodes[0].v_threshold > 0).all()
        y = snn(torch.rand(2, 1, 28, 28))
        assert y.shape == (2, 10)

    def test_default_conversion_still_uses_scalar_ifnode(self):
        model = SimpleCNNNoBN()
        model.eval()
        snn = _rate_converter(
            fuse_flag=False,
        ).convert(model)

        if_nodes = [m for m in snn.modules() if isinstance(m, neuron.IFNode)]
        assert len(if_nodes) == 1
        assert not any(
            isinstance(m, neuron.ActivationAwareIFNode) for m in snn.modules()
        )
        assert isinstance(if_nodes[0].v_threshold, float)

    def test_module_names_with_underscores_convert(self):
        model = UnderscoreModuleCNN()
        model.eval()
        snn = _rate_converter(fuse_flag=False).convert(model)

        assert any(isinstance(m, neuron.IFNode) for m in snn.modules())
        assert not any(isinstance(m, nn.ReLU) for m in snn.modules())
        assert "conv_block.spiking_0.if_node" in dict(snn.named_modules())

    def test_non_positive_threshold_raises(self):
        class ZeroScaleOptimizer:
            def compute_threshold(self, hook):
                return 0.0

        model = SimpleCNNNoBN()
        model.eval()
        converter = _rate_converter(
            threshold_optimizer=ZeroScaleOptimizer(),
            fuse_flag=False,
        )

        with pytest.raises(ValueError, match="finite positive"):
            converter.convert(model)

    def test_nan_threshold_raises(self):
        class NaNScaleOptimizer:
            def compute_threshold(self, hook):
                return float("nan")

        model = SimpleCNNNoBN()
        model.eval()
        converter = _rate_converter(
            threshold_optimizer=NaNScaleOptimizer(),
            fuse_flag=False,
        )

        with pytest.raises(ValueError, match="finite positive"):
            converter.convert(model)

    def test_infinite_threshold_raises(self):
        class InfiniteScaleOptimizer:
            def compute_threshold(self, hook):
                return float("inf")

        model = SimpleCNNNoBN()
        model.eval()
        converter = _rate_converter(
            threshold_optimizer=InfiniteScaleOptimizer(),
            fuse_flag=False,
        )

        with pytest.raises(ValueError, match="finite positive"):
            converter.convert(model)


class TestConverterAlgorithmBoundary:
    @pytest.mark.parametrize(
        "name",
        [
            "convert_to_spiking_neurons",
            "replace_by_td_operators",
            "fuse",
            "set_voltagehook",
            "replace_by_neurons",
            "replace_by_ifnode",
        ],
    )
    def test_converter_exposes_no_algorithm_public_methods(self, name):
        assert not hasattr(Converter, name)

    def test_rate_coding_recipe_sets_voltagehook(self):
        model = SimpleCNNNoBN()
        model.eval()
        ann = torch.fx.symbolic_trace(model)
        ann_with_hook = RateCodingRecipe(
            dataloader=[],
            mode="99.9%",
            momentum=0.2,
        )._set_voltagehook(ann)

        hooks = [m for m in ann_with_hook.modules() if isinstance(m, VoltageHook)]
        assert len(hooks) == 1
        assert hooks[0].mode == "99.9%"
        assert hooks[0].momentum == 0.2


class TestEndToEnd:
    def test_snn_inference(self):
        model = SimpleCNN()
        model.eval()
        converter = _rate_converter(mode="Max", fuse_flag=True)
        snn = converter.convert(model)
        img = torch.randn(1, 1, 28, 28)
        T = 10
        out = 0
        for m in snn.modules():
            if hasattr(m, "reset"):
                m.reset()
        for _ in range(T):
            out = out + snn(img)
        assert out.shape == (1, 10)

    def test_snn_no_bn(self):
        model = SimpleCNNNoBN()
        model.eval()
        converter = _rate_converter(mode="Max", fuse_flag=False)
        snn = converter.convert(model)
        out = snn(torch.randn(1, 1, 28, 28))
        assert out.shape == (1, 10)
