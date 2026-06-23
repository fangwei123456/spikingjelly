from types import SimpleNamespace

import numpy as np
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
from spikingjelly.activation_based.ann2snn.operators import (
    TDGELU,
    TDLayerNorm,
    TDLinear,
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


def _make_loader(batch_size=2, channels=1, h=28, w=28):
    imgs = torch.randn(batch_size, channels, h, w)
    labels = torch.zeros(batch_size, dtype=torch.long)
    return [(imgs, labels)]


def _make_vector_loader(batch_size=2, features=4):
    return [torch.randn(batch_size, features)]


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
    def test_converter_is_plain_conversion_driver(self):
        converter = Converter(dataloader=_make_loader(), mode="Max", fuse_flag=False)

        assert not isinstance(converter, nn.Module)
        with pytest.raises(TypeError):
            converter(SimpleCNN())

    def test_relu_model_converts(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode="Max", fuse_flag=False)
        snn = converter.convert_to_spiking_neurons(model)
        assert snn is not None

    def test_output_shape_preserved(self):
        model = SimpleCNN()
        model.eval()
        dummy = torch.randn(1, 1, 28, 28)
        converter = Converter(dataloader=_make_loader(), mode="Max", fuse_flag=False)
        snn = converter.convert_to_spiking_neurons(model)
        out = snn(dummy)
        assert out.shape == (1, 10)

    def test_fuse_conv_bn(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode="Max", fuse_flag=True)
        snn = converter.convert_to_spiking_neurons(model)
        assert snn is not None

    def test_mode_max(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode="max")
        snn = converter.convert_to_spiking_neurons(model)
        assert snn is not None

    def test_mode_robust(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode="99.9%")
        snn = converter.convert_to_spiking_neurons(model)
        assert snn is not None

    def test_mode_scalar(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode=0.5)
        snn = converter.convert_to_spiking_neurons(model)
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
        snn = converter.convert_to_spiking_neurons(model)
        assert snn is not None

    def test_numpy_dataloader_converts_full_batch(self):
        model = SimpleCNNNoBN()
        model.eval()
        imgs = np.random.randn(2, 1, 28, 28).astype(np.float32)
        converter = Converter(dataloader=[imgs], mode="Max", fuse_flag=False)
        snn = converter.convert_to_spiking_neurons(model)
        assert snn is not None

    def test_dict_dataloader_converts(self):
        model = SimpleCNNNoBN()
        model.eval()
        imgs = torch.randn(2, 1, 28, 28)
        converter = Converter(dataloader=[{"input": imgs}], mode="Max", fuse_flag=False)
        snn = converter.convert_to_spiking_neurons(model)
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
        snn = converter.convert_to_spiking_neurons(model)
        assert snn is not None

    def test_extract_batch_input_rejects_empty_sequence(self):
        with pytest.raises(ValueError, match="empty list or tuple"):
            Converter._extract_batch_input([])

    def test_extract_batch_input_rejects_empty_dict(self):
        with pytest.raises(ValueError, match="empty dictionary"):
            Converter._extract_batch_input({})


class TestConverterTDOperatorReplacement:
    def test_replaces_core_modules_with_td_operators(self):
        model = CoreTransformerMLP()
        model.eval()

        converted = Converter(dataloader=[]).replace_by_td_operators(model)
        modules = dict(converted.named_modules())

        assert isinstance(modules["norm"], TDLayerNorm)
        assert isinstance(modules["fc0"], TDLinear)
        assert isinstance(modules["act"], TDGELU)
        assert isinstance(modules["fc1"], TDLinear)
        assert isinstance(modules["keep"], nn.Identity)

    def test_copies_linear_layernorm_and_gelu_configuration(self):
        model = CoreTransformerMLP()
        model.eval()

        converted = Converter(dataloader=[]).replace_by_td_operators(model)
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
        converted = Converter(dataloader=[]).replace_by_td_operators(model)
        x_seq = torch.randn(5, 2, 3, 4)

        y_seq = converted(x_seq)
        expected = model(x_seq.cumsum(dim=0))

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)

    def test_no_affine_layernorm_is_preserved(self):
        model = NoAffineLayerNormMLP()
        model.eval()

        converted = Converter(dataloader=[]).replace_by_td_operators(model)
        modules = dict(converted.named_modules())

        assert isinstance(modules["norm"], TDLayerNorm)
        assert modules["norm"].weight is None
        assert modules["norm"].bias is None
        assert isinstance(modules["fc"], TDLinear)

    def test_td_operator_replacement_preserves_training_mode(self):
        model = DropoutCoreMLP()
        model.train()

        converted = Converter(dataloader=[]).replace_by_td_operators(model)

        assert converted.training
        assert converted.dropout.training
        assert converted.fc.training

    def test_td_operator_replacement_preserves_eval_mode(self):
        model = DropoutCoreMLP()
        model.eval()

        converted = Converter(dataloader=[]).replace_by_td_operators(model)

        assert not converted.training
        assert not converted.dropout.training
        assert not converted.fc.training


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
        snn = converter.convert_to_spiking_neurons(model)
        assert snn is not None

    def test_fuse_flag_false(self):
        model = SimpleCNN()
        model.eval()
        converter = Converter(dataloader=_make_loader(), mode="Max", fuse_flag=False)
        snn = converter.convert_to_spiking_neurons(model)
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
        snn = converter.convert_to_spiking_neurons(model)
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
        snn = converter.convert_to_spiking_neurons(model)
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
        snn = converter.convert_to_spiking_neurons(model)

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
        snn = converter.convert_to_spiking_neurons(model)
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
        ).convert_to_spiking_neurons(model)

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
        snn = Converter(
            dataloader=_make_vector_loader(),
            rules=[rule],
            fuse_flag=False,
        ).convert_to_spiking_neurons(model)

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

        Converter.set_voltagehook(
            fx_model,
            rules=[marker_rule, marker_aware_rule],
        )

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
        snn = Converter(
            dataloader=_make_loader(),
            rules=[rule],
            fuse_flag=False,
        ).convert_to_spiking_neurons(model)

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
        ).convert_to_spiking_neurons(model)

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
        ).convert_to_spiking_neurons(model)

        scalers = [m for m in snn.modules() if isinstance(m, VoltageScaler)]
        if_nodes = [m for m in snn.modules() if isinstance(m, neuron.IFNode)]
        assert len(scalers) == 2
        assert len(if_nodes) == 1
        assert scalers[0].scale.item() == pytest.approx(
            if_nodes[0].v_threshold.item() / scalers[1].scale.item()
        )

    def test_module_names_with_underscores_convert(self):
        model = UnderscoreModuleCNN()
        model.eval()
        snn = Converter(
            dataloader=_make_loader(), fuse_flag=False
        ).convert_to_spiking_neurons(model)

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
            converter.convert_to_spiking_neurons(model)

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
            converter.convert_to_spiking_neurons(model)

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
            converter.convert_to_spiking_neurons(model)


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
        snn = converter.convert_to_spiking_neurons(model)
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
        snn = converter.convert_to_spiking_neurons(model)
        out = snn(torch.randn(1, 1, 28, 28))
        assert out.shape == (1, 10)
