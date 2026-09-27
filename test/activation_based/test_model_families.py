import copy

import pytest
import torch

from spikingjelly.activation_based import (
    base,
    functional,
    layer,
    model,
    neuron,
    surrogate,
)
from spikingjelly.activation_based.model.masnn import MASNN, AttMSResNet
from spikingjelly.activation_based.model.maxformer import MaxFormer
from spikingjelly.activation_based.model.ms_resnet import MaxResNet, MSResNet
from spikingjelly.activation_based.model.qkformer import QKFormer
from spikingjelly.activation_based.model.spike_driven_transformer import (
    SpikeDrivenTransformer,
)


def _train_step(model):
    model.train()
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 5)
    output.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_model_package_exports_new_models_and_builders():
    expected = {
        "AttMSResNet",
        "MASNN",
        "MSResNet",
        "MaxFormer",
        "MaxResNet",
        "QKFormer",
        "SpikeDrivenTransformer",
        "att_ms_resnet18",
        "masnn_dvs128_gesture",
        "max_resnet18",
        "maxformer_10_384",
        "ms_resnet18",
        "ms_resnet34",
        "qkformer_10_384",
        "sdt_8_384",
    }

    assert expected <= set(model.__all__)


def test_spike_driven_self_attention_preserves_feature_shape():
    attention = layer.SpikeDrivenSelfAttention(dim=8, num_heads=2)
    assert not isinstance(attention, base.StepModule)
    assert isinstance(attention.q_lif, base.StepModule)
    x = torch.randn(2, 2, 8, 4, 4)
    y = attention(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert any(parameter.grad is not None for parameter in attention.parameters())


def test_qkformer_tiny_forward_and_backward():
    model = QKFormer(
        T=2,
        num_classes=5,
        embed_dims=32,
        num_heads=(1, 2, 4),
        depths=(1, 1, 1),
    )
    assert isinstance(model.stage1[0].attn, layer.QKAttention)
    assert not isinstance(model.stage1[0].attn, base.StepModule)
    assert not isinstance(model.stage3[0].attn, base.StepModule)
    _train_step(model)


def test_ms_resnet_and_max_resnet_tiny_forward_and_backward():
    kwargs = dict(
        T=2,
        in_channels=3,
        num_classes=5,
        layers=(1, 1, 1, 1),
        base_channels=8,
        stem_kernel_size=3,
        stem_stride=1,
        stem_pool=False,
    )
    ms_resnet = MSResNet(**kwargs)
    max_resnet = MaxResNet(**kwargs)
    assert not hasattr(ms_resnet.layer2[0], "max_pool")
    assert hasattr(max_resnet.layer2[0], "max_pool")
    assert ms_resnet.layer1[0].conv1.stride == (2, 2)
    assert max_resnet.layer1[0].conv1.stride == (1, 1)
    assert ms_resnet.layer1[0].spike1.v_threshold == 0.5
    assert not ms_resnet.layer1[0].spike1.decay_input
    assert isinstance(ms_resnet.layer1[0].spike1.surrogate_function, surrogate.Rect)
    assert max_resnet.layer1[0].spike1.v_threshold == 1.0
    assert max_resnet.layer1[0].spike1.decay_input
    _train_step(ms_resnet)
    _train_step(max_resnet)


def test_maxformer_tiny_forward_and_backward():
    model = MaxFormer(
        T=2,
        num_classes=5,
        embed_dims=32,
        depths=(1, 1, 1),
    )
    assert isinstance(model.stage3[0].attn, layer.SpikingSelfAttention)
    stage = copy.deepcopy(model.patch_embed2).eval()
    x = torch.randn(2, 2, 8, 8, 8)
    spiked = stage.lif1(x)
    expected = stage.pool(stage.bn1(stage.conv1(spiked)))
    expected = stage.bn2(stage.conv2(stage.lif2(expected))) + stage.shortcut(spiked)

    torch.testing.assert_close(model.patch_embed2.eval()(x), expected)
    functional.reset_net(model)
    _train_step(model)


def test_spike_driven_transformer_tiny_forward_and_backward():
    model = SpikeDrivenTransformer(
        T=2,
        num_classes=5,
        embed_dims=32,
        num_heads=4,
        depths=1,
        pooling_stat="1010",
    )
    reference = copy.deepcopy(model.patch_embed).eval()
    x = torch.randn(2, 2, 3, 32, 32)
    expected = x
    for stage in reference.stages:
        expected = stage(expected)
    expected = reference.final_stage(expected)
    identity = expected
    expected = reference.rpe_bn(reference.rpe_conv(reference.final_lif(expected)))
    expected = expected + identity
    features = model.patch_embed.eval()(x)

    torch.testing.assert_close(features, expected)
    assert features.shape[-2:] == (8, 8)
    functional.reset_net(model)
    _train_step(model)


def test_masnn_tiny_forward_and_backward():
    net = MASNN(
        T=4,
        in_channels=2,
        num_classes=5,
        input_size=(16, 16),
        channels=(8, 16, 16),
        pools=(1, 2, 2),
        fc_hidden=12,
        reduction_t=2,
        reduction_c=4,
    )
    conv_block = net.conv_blocks[0]
    assert isinstance(conv_block.attn, layer.MultiDimensionalAttention)
    assert conv_block.attn.ta is not None
    assert conv_block.attn.ca is not None
    assert conv_block.attn.sa is not None
    assert net.conv_blocks[0].pool is None
    assert isinstance(net.conv_blocks[1].pool, layer.AvgPool2d)
    assert isinstance(net.fc_blocks[0].attn, layer.TemporalWiseAttention)
    cell = conv_block.cell
    assert isinstance(cell, neuron.LIFNode)
    assert cell.tau == 10.0 / 7.0
    assert cell.v_threshold == 0.3
    assert not cell.decay_input
    assert cell.v_reset == 0.0
    assert isinstance(cell.surrogate_function, surrogate.Rect)
    assert cell.surrogate_function.alpha == 2.0

    net.train()
    output = net(torch.randn(2, 2, 16, 16))
    assert output.shape == (2, 5)
    output.mean().backward()
    assert any(parameter.grad is not None for parameter in net.parameters())

    functional.reset_net(net)
    with pytest.raises(ValueError):
        net(torch.randn(3, 2, 2, 16, 16))


def test_att_ms_resnet_tiny_forward_and_backward():
    kwargs = dict(
        T=2,
        in_channels=3,
        num_classes=5,
        layers=(1, 1, 1, 1),
        base_channels=8,
        stem_kernel_size=3,
        stem_stride=1,
        stem_pool=False,
        reduction_c=4,
    )
    net = AttMSResNet(**kwargs)
    block = net.layer2[0]
    assert isinstance(block.attention, layer.MultiDimensionalAttention)
    assert block.attention.ta is None
    assert block.attention.ca is not None
    assert block.attention.sa is not None
    assert isinstance(block.downsample[0], layer.AvgPool2d)
    assert torch.all(block.bn2.weight == 0.1)
    assert block.spike1.v_threshold == 0.5
    assert not block.spike1.decay_input
    assert isinstance(block.spike1.surrogate_function, surrogate.Rect)
    assert block.spike1.surrogate_function.alpha == 1.0

    ms_kwargs = {k: v for k, v in kwargs.items() if k != "reduction_c"}
    assert not hasattr(MSResNet(**ms_kwargs).layer2[0], "attention")
    _train_step(net)


def test_att_ms_resnet_matches_branch_shapes_for_odd_inputs():
    net = AttMSResNet(
        T=1,
        num_classes=5,
        layers=(1, 1, 1, 1),
        base_channels=8,
        stem_kernel_size=3,
        stem_stride=1,
        stem_pool=False,
        reduction_c=4,
    ).eval()

    with torch.no_grad():
        odd = net(torch.randn(2, 3, 33, 33))

    assert odd.shape == (2, 5)


def test_masnn_models_accept_a_custom_spiking_neuron():
    masnn = MASNN(
        T=4,
        in_channels=2,
        num_classes=5,
        input_size=(16, 16),
        channels=(8, 16),
        pools=(1, 2),
        fc_hidden=12,
        reduction_t=2,
        reduction_c=4,
        spiking_neuron=neuron.IFNode,
        v_threshold=0.8,
    )
    cells = [masnn.conv_blocks[0].cell, masnn.fc_blocks[0].cell]
    assert all(isinstance(cell, neuron.IFNode) for cell in cells)
    assert all(cell.v_threshold == 0.8 for cell in cells)
    assert all(cell.step_mode == "m" for cell in cells)
    assert cells[0] is not cells[1]

    resnet = AttMSResNet(
        T=2,
        num_classes=5,
        layers=(1, 1, 1),
        base_channels=8,
        stem_kernel_size=3,
        stem_stride=1,
        reduction_c=4,
        spiking_neuron=neuron.IFNode,
        v_threshold=0.8,
    )
    for module in (resnet.layer1[0].spike1, resnet.layer1[0].spike2, resnet.head_lif):
        assert isinstance(module, neuron.IFNode)
        assert module.v_threshold == 0.8
        assert module.step_mode == "m"

    masnn(torch.randn(2, 2, 16, 16)).mean().backward()
    functional.reset_net(masnn)
    _train_step(resnet)


def test_masnn_custom_spiking_neuron_receives_the_model_backend(monkeypatch):
    monkeypatch.setattr(base, "check_backend_library", lambda _backend: None)

    net = MASNN(
        T=4,
        in_channels=2,
        num_classes=5,
        input_size=(16, 16),
        channels=(8,),
        pools=(1,),
        fc_hidden=12,
        reduction_t=2,
        reduction_c=4,
        backend="cupy",
        spiking_neuron=neuron.IFNode,
    )
    resnet = AttMSResNet(
        T=2,
        num_classes=5,
        layers=(1, 1, 1),
        base_channels=8,
        stem_kernel_size=3,
        stem_stride=1,
        reduction_c=4,
        backend="cupy",
        spiking_neuron=neuron.IFNode,
    )

    for model in (net, resnet):
        cells = [m for m in model.modules() if isinstance(m, neuron.BaseNode)]
        assert cells
        assert all(isinstance(cell, neuron.IFNode) for cell in cells)
        assert {cell.backend for cell in cells} == {"cupy"}


def test_masnn_keyword_arguments_override_the_paper_neuron():
    net = MASNN(
        T=4,
        in_channels=2,
        num_classes=5,
        input_size=(16, 16),
        channels=(8,),
        pools=(1,),
        fc_hidden=12,
        reduction_t=2,
        reduction_c=4,
        v_threshold=0.9,
    )
    cell = net.conv_blocks[0].cell

    assert isinstance(cell, neuron.LIFNode)
    assert cell.v_threshold == 0.9
    assert cell.tau == 10.0 / 7.0
