import torch

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.model.spikformer import Spikformer, spikformer_ti


def _reset_net(net):
    functional.reset_net(net)


def test_spikformer_forward_accepts_image_and_sequence_inputs():
    model = Spikformer(
        T=2,
        in_channels=3,
        img_size_h=64,
        img_size_w=64,
        num_classes=11,
        embed_dims=64,
        num_heads=4,
        depths=2,
        backend="torch",
    ).eval()

    x_img = torch.randn(3, 3, 64, 64)
    _reset_net(model)
    y_img = model(x_img)
    assert y_img.shape == (2, 3, 11)

    x_seq = torch.randn(2, 3, 3, 64, 64)
    _reset_net(model)
    y_seq = model(x_seq)
    assert y_seq.shape == (2, 3, 11)


def test_spikformer_ti_factory_builds_trainable_model():
    model = spikformer_ti(
        T=2,
        img_size_h=64,
        img_size_w=64,
        num_classes=7,
        backend="torch",
    ).train()
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 7, (2,))

    _reset_net(model)
    y = model(x)
    loss = torch.nn.functional.cross_entropy(y.mean(0), target)
    loss.backward()

    assert y.shape == (2, 2, 7)
    assert any(p.grad is not None for p in model.parameters())
