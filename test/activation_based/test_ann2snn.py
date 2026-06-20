from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import ann2snn, neuron
from spikingjelly.activation_based.ann2snn import (
    Converter,
    HookFactory,
    NeuronFactory,
    ReLURule,
    ThresholdOptimizer,
)
from spikingjelly.activation_based.ann2snn.modules import VoltageHook, VoltageScaler


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


def _make_loader(batch_size=2, channels=1, h=28, w=28):
    imgs = torch.randn(batch_size, channels, h, w)
    labels = torch.zeros(batch_size, dtype=torch.long)
    return [(imgs, labels)]


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
            "download_url",
            "ReLURule",
            "NeuronFactory",
            "HookFactory",
            "ThresholdOptimizer",
        }


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
    def test_relu_model_converts(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode="Max", fuse_flag=False)
        snn = converter(model)
        assert snn is not None

    def test_output_shape_preserved(self):
        model = SimpleCNN()
        model.eval()
        dummy = torch.randn(1, 1, 28, 28)
        converter = Converter(dataloader=_make_loader(), mode="Max", fuse_flag=False)
        snn = converter(model)
        out = snn(dummy)
        assert out.shape == (1, 10)

    def test_fuse_conv_bn(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode="Max", fuse_flag=True)
        snn = converter(model)
        assert snn is not None

    def test_mode_max(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode="max")
        snn = converter(model)
        assert snn is not None

    def test_mode_robust(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode="99.9%")
        snn = converter(model)
        assert snn is not None

    def test_mode_scalar(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode=0.5)
        snn = converter(model)
        assert snn is not None

    def test_invalid_mode_raises(self):
        with pytest.raises(NotImplementedError):
            Converter(dataloader=[], mode="invalid")

    def test_invalid_scalar_raises(self):
        with pytest.raises(NotImplementedError):
            Converter(dataloader=[], mode=1.5)

    def test_invalid_scalar_zero_raises(self):
        with pytest.raises(NotImplementedError):
            Converter(dataloader=[], mode=0.0)

    def test_tensor_dataloader_converts(self):
        model = SimpleCNNNoBN()
        model.eval()
        imgs = torch.randn(2, 1, 28, 28)
        converter = Converter(dataloader=[imgs], mode="Max", fuse_flag=False)
        snn = converter(model)
        assert snn is not None

    def test_dict_dataloader_converts(self):
        model = SimpleCNNNoBN()
        model.eval()
        imgs = torch.randn(2, 1, 28, 28)
        converter = Converter(
            dataloader=[{"input": imgs}], mode="Max", fuse_flag=False
        )
        snn = converter(model)
        assert snn is not None

    def test_dict_dataloader_prefers_input_key(self):
        model = SimpleCNNNoBN()
        model.eval()
        imgs = torch.randn(2, 1, 28, 28)
        labels = torch.zeros(2, dtype=torch.long)
        converter = Converter(
            dataloader=[{"label": labels, "input": imgs}],
            mode="Max",
            fuse_flag=False,
        )
        snn = converter(model)
        assert snn is not None


class TestFuse:
    def test_conv_bn_fusion_matches_eval_output(self):
        model = SimpleCNN()
        model.eval()
        fx_model = torch.fx.symbolic_trace(model)
        x = torch.randn(2, 1, 28, 28)
        expected = fx_model(x)

        fused = Converter.fuse(fx_model, fuse_flag=True)
        result = fused(x)

        assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)
        assert not any(isinstance(m, nn.BatchNorm2d) for m in fused.modules())

    def test_no_bn_model_fuse(self):
        model = SimpleCNNNoBN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode="Max", fuse_flag=True)
        snn = converter(model)
        assert snn is not None

    def test_fuse_flag_false(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode="Max", fuse_flag=False)
        snn = converter(model)
        assert snn is not None


class TestRuleBasedConversion:
    def test_custom_rules_list(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(
            dataloader=_make_loader(),
            rules=[ReLURule()],
            fuse_flag=False,
        )
        snn = converter(model)
        assert snn is not None
        assert any(isinstance(m, neuron.IFNode) for m in snn.modules())

    def test_custom_neuron_factory(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(
            dataloader=_make_loader(),
            neuron_factory=NeuronFactory(v_threshold=2.0),
            fuse_flag=False,
        )
        snn = converter(model)
        assert snn is not None
        if_nodes = [m for m in snn.modules() if isinstance(m, neuron.IFNode)]
        assert len(if_nodes) == 1
        assert if_nodes[0].v_threshold == 2.0

    def test_custom_neuron_threshold_updates_input_scaler(self):
        model = SimpleCNNNoBN()
        model.eval()
        converter = Converter(
            dataloader=_make_loader(),
            neuron_factory=NeuronFactory(v_threshold=2.0),
            threshold_optimizer=ThresholdOptimizer("fixed"),
            fuse_flag=False,
        )
        snn = converter(model)

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
        converter = Converter(
            dataloader=_make_loader(),
            rules=[],
            fuse_flag=False,
        )
        snn = converter(model)
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
        Converter(
            dataloader=_make_loader(),
            rules=[rule],
            fuse_flag=False,
        )(model)

        assert rule.match_count >= 2
        assert rule.insert_count == 1
        assert rule.find_count == 1
        assert rule.replace_count == 1

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
        snn = Converter(
            dataloader=_make_loader(),
            rules=[rule],
            fuse_flag=False,
        )(model)

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
        snn = Converter(
            dataloader=_make_loader(),
            threshold_optimizer=optimizer,
            fuse_flag=False,
        )(model)

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
        snn = Converter(
            dataloader=_make_loader(),
            neuron_factory=TensorThresholdFactory(),
            threshold_optimizer=ThresholdOptimizer("fixed"),
            fuse_flag=False,
        )(model)

        scalers = [m for m in snn.modules() if isinstance(m, VoltageScaler)]
        if_nodes = [m for m in snn.modules() if isinstance(m, neuron.IFNode)]
        assert len(scalers) == 2
        assert len(if_nodes) == 1
        assert scalers[0].scale.item() == pytest.approx(
            if_nodes[0].v_threshold.item() / scalers[1].scale.item()
        )

    def test_multi_element_neuron_threshold_updates_input_scaler(self):
        class MultiThresholdFactory(NeuronFactory):
            def create(self, scale):
                n = super().create(scale)
                n.v_threshold = torch.tensor([2.0, 4.0])
                return n

        model = SimpleCNNNoBN()
        model.eval()
        snn = Converter(
            dataloader=_make_loader(),
            neuron_factory=MultiThresholdFactory(),
            threshold_optimizer=ThresholdOptimizer("fixed"),
            fuse_flag=False,
        )(model)

        scalers = [m for m in snn.modules() if isinstance(m, VoltageScaler)]
        assert len(scalers) == 2
        assert scalers[0].scale.shape == torch.Size([2])

    def test_module_names_with_underscores_convert(self):
        model = UnderscoreModuleCNN()
        model.eval()
        snn = Converter(dataloader=_make_loader(), fuse_flag=False)(model)

        assert any(isinstance(m, neuron.IFNode) for m in snn.modules())
        assert not any(isinstance(m, nn.ReLU) for m in snn.modules())
        assert "conv_block.spiking_0.if_node" in dict(snn.named_modules())

    def test_non_positive_threshold_raises(self):
        class ZeroScaleOptimizer:
            def compute_threshold(self, hook):
                return 0.0

        model = SimpleCNNNoBN()
        model.eval()
        converter = Converter(
            dataloader=_make_loader(),
            threshold_optimizer=ZeroScaleOptimizer(),
            fuse_flag=False,
        )

        with pytest.raises(ValueError, match="finite positive"):
            converter(model)

    def test_nan_threshold_raises(self):
        class NaNScaleOptimizer:
            def compute_threshold(self, hook):
                return float("nan")

        model = SimpleCNNNoBN()
        model.eval()
        converter = Converter(
            dataloader=_make_loader(),
            threshold_optimizer=NaNScaleOptimizer(),
            fuse_flag=False,
        )

        with pytest.raises(ValueError, match="finite positive"):
            converter(model)

    def test_infinite_threshold_raises(self):
        class InfiniteScaleOptimizer:
            def compute_threshold(self, hook):
                return float("inf")

        model = SimpleCNNNoBN()
        model.eval()
        converter = Converter(
            dataloader=_make_loader(),
            threshold_optimizer=InfiniteScaleOptimizer(),
            fuse_flag=False,
        )

        with pytest.raises(ValueError, match="finite positive"):
            converter(model)


class TestReplaceByIfnodeDeprecation:
    def test_set_voltagehook_static_legacy_api(self):
        model = SimpleCNNNoBN()
        model.eval()
        ann = torch.fx.symbolic_trace(model)
        ann_with_hook = Converter.set_voltagehook(ann, mode="99.9%", momentum=0.2)

        hooks = [m for m in ann_with_hook.modules() if isinstance(m, VoltageHook)]
        assert len(hooks) == 1
        assert hooks[0].mode == "99.9%"
        assert hooks[0].momentum == 0.2

    def test_deprecation_warning(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), fuse_flag=False)
        ann = torch.fx.symbolic_trace(model)
        ann_fused = converter.fuse(ann, fuse_flag=False)
        ann_with_hook = converter.set_voltagehook(ann_fused)
        for data in _make_loader():
            ann_with_hook(data[0])
        with pytest.warns(DeprecationWarning, match="replace_by_ifnode is deprecated"):
            snn = Converter.replace_by_ifnode(ann_with_hook)
        assert snn is not None


class TestEndToEnd:
    def test_snn_inference(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode="Max", fuse_flag=True)
        snn = converter(model)
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
        converter = Converter(dataloader=_make_loader(), mode="Max", fuse_flag=False)
        snn = converter(model)
        out = snn(torch.randn(1, 1, 28, 28))
        assert out.shape == (1, 10)
