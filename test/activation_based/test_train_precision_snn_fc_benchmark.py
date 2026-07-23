from argparse import Namespace
from contextlib import nullcontext

import torch

import benchmark.benchmark_train_precision_snn_fc as benchmark
from spikingjelly.activation_based.precision import PrecisionArtifacts


class _ToyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self.linear(x_seq).mean(0)


class _RecordingScaler:
    def __init__(self, scale: float = 8.0) -> None:
        self.scale_factor = scale
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale_factor

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.step_calls += 1
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale_factor)
        optimizer.step()

    def update(self) -> None:
        self.update_calls += 1


class _ScaledArtifacts:
    backward = PrecisionArtifacts.backward

    def __init__(self, model: torch.nn.Module, scaler: _RecordingScaler) -> None:
        self.model = model
        self.scaler = scaler

    @staticmethod
    def autocast_context():
        return nullcontext()


def test_training_step_routes_scaled_gradients_through_scaler() -> None:
    model = _ToyClassifier()
    scaler = _RecordingScaler()
    artifacts = _ScaledArtifacts(model, scaler)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    x_seq = torch.randn(2, 3, 4)
    target = torch.tensor([0, 1, 0])

    benchmark.run_training_step(
        model,
        artifacts,
        optimizer,
        criterion,
        x_seq,
        target,
        torch.device("cpu"),
    )

    assert scaler.step_calls == 1
    assert scaler.update_calls == 1


def test_benchmark_releases_training_state_before_inference(monkeypatch) -> None:
    created_optimizers: list[torch.optim.SGD] = []
    original_sgd = torch.optim.SGD

    class _RecordingSGD(original_sgd):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            created_optimizers.append(self)

    monkeypatch.setattr(benchmark.torch.optim, "SGD", _RecordingSGD)
    args = Namespace(
        input_dim=4,
        hidden_dim=4,
        num_classes=2,
        tau=2.0,
        backend="torch",
        depth=2,
        attention_every=0,
        num_heads=1,
        lr=0.1,
        momentum=0.9,
        warmup=1,
        steps=1,
        inference_steps=1,
        batch_size=3,
    )
    base_model = benchmark.build_model(args)
    x_seq = torch.randn(2, args.batch_size, args.input_dim)
    target = torch.tensor([0, 1, 0])

    benchmark.benchmark_one_precision(
        args,
        "fp32",
        base_model.state_dict(),
        x_seq,
        target,
        torch.device("cpu"),
    )

    assert len(created_optimizers) == 1
    optimizer = created_optimizers[0]
    assert not optimizer.state
    assert all(
        parameter.grad is None
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
