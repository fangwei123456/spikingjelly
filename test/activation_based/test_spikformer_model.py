import torch

from spikingjelly.activation_based import base, functional, neuron
from spikingjelly.activation_based.model import (
    Spikformer,
    spikformer_cifar10,
    spikformer_ti,
)
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18
from spikingjelly.activation_based.model.spiking_vggws_ottt import ottt_spiking_vggws


def _reset_net(net):
    functional.reset_net(net)


def test_model_package_keeps_training_out_of_model_exports():
    import spikingjelly.activation_based.model as model

    assert not hasattr(model, "Trainer")
    assert not hasattr(model, "SpikformerBlock")


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

    assert not isinstance(model, base.StepModule)
    assert not isinstance(model.patch_embed, base.StepModule)
    assert not isinstance(model.patch_embed.stages[0], base.StepModule)
    assert not isinstance(model.blocks[0], base.StepModule)
    assert not isinstance(model.blocks[0].mlp, base.StepModule)
    assert isinstance(model.patch_embed.stages[0].neuron, base.StepModule)
    functional.set_step_mode(model, "m")
    assert model.patch_embed.stages[0].neuron.step_mode == "m"

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


def test_spikformer_cifar10_factory_runs_official_shape():
    model = spikformer_cifar10(T=1, backend="torch").eval()

    output = model(torch.randn(1, 3, 32, 32))

    assert output.shape == (1, 1, 10)
    assert model.patch_embed.grid_size == (8, 8)
    assert len(model.blocks) == 4


def test_spiking_resnet_family_runs_multistep_forward_and_backward():
    model = spiking_resnet18(
        spiking_neuron=neuron.IFNode,
        num_classes=5,
        step_mode="m",
    ).train()
    functional.set_step_mode(model, "m")
    x = torch.randn(1, 2, 3, 32, 32)

    output = model(x)
    output.sum().backward()

    assert output.shape == (1, 2, 5)
    assert model.conv1.weight.grad is not None


def test_ottt_vgg_family_factory_runs_forward_and_backward():
    model = ottt_spiking_vggws(num_classes=5).train()
    x = torch.randn(1, 3, 32, 32)

    output = model(x)
    output.sum().backward()

    assert output.shape == (1, 5)
    assert model.features[0].weight.grad is not None
