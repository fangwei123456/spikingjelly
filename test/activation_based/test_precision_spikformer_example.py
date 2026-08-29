import torch

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.model import Spikformer
from spikingjelly.activation_based.precision import (
    PrecisionConfig,
    prepare_model_for_precision,
)


def _model():
    return Spikformer(
        T=2,
        in_channels=3,
        img_size_h=64,
        img_size_w=64,
        num_classes=16,
        embed_dims=64,
        num_heads=4,
        depths=2,
        backend="torch",
    ).train()


def test_precision_supports_custom_spikformer_training_loop():
    model = _model()
    precision = prepare_model_for_precision(model, "cpu", PrecisionConfig(mode="fp32"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 16, (2,))

    optimizer.zero_grad(set_to_none=True)
    with precision.autocast_context():
        output = precision.model(x)
        loss = torch.nn.functional.cross_entropy(output.mean(0), target)
    precision.backward(loss, optimizer)
    functional.reset_net(precision.model)

    assert output.shape == (2, 2, 16)
    assert torch.isfinite(loss)
    assert precision.describe()["conversion_report"]["converted_modules"] == []
