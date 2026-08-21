import torch

from spikingjelly.activation_based import functional, layer
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


def test_spike_driven_self_attention_preserves_feature_shape():
    attention = layer.SpikeDrivenSelfAttention(dim=8, num_heads=2)
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
    _train_step(ms_resnet)
    _train_step(max_resnet)


def test_maxformer_tiny_forward_and_backward():
    _train_step(
        MaxFormer(
            T=2,
            num_classes=5,
            embed_dims=32,
            depths=(1, 1, 1),
        )
    )


def test_spike_driven_transformer_tiny_forward_and_backward():
    model = SpikeDrivenTransformer(
        T=2,
        num_classes=5,
        embed_dims=32,
        num_heads=4,
        depths=1,
        pooling_stat="1010",
    )
    features = model.patch_embed(torch.randn(2, 2, 3, 32, 32))
    assert features.shape[-2:] == (8, 8)
    functional.reset_net(model)
    _train_step(model)
