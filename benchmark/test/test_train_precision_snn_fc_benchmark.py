from argparse import Namespace
from contextlib import nullcontext
import json
import sys

import pytest
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


def test_profile_options_parse_and_validate(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--profile",
            "--no-profile-module-hooks",
            "--profile-steps",
            "3",
            "--precisions",
            "fp8",
            "--fp8-recipe",
            "delayed",
            "--fp8-fallback-dtype",
            "fp16",
            "--triton-storage",
            "float8_e4m3fn",
            "--triton-fwd",
            "fp16",
            "--triton-bwd",
            "bf16",
            "--output",
            str(tmp_path / "profile.json"),
        ],
    )

    args = benchmark.parse_args()
    benchmark.validate_profile_args(args)

    assert args.profile is True
    assert args.profile_module_hooks is False
    assert args.profile_steps == 3
    assert args.precisions == ["fp8"]
    assert args.fp8_recipe == "delayed"
    assert args.fp8_fallback_dtype == "fp16"
    assert args.triton_storage == "float8_e4m3fn"
    assert args.triton_fwd == "fp16"
    assert args.triton_bwd == "bf16"


def test_profile_validation_rejects_multiple_precisions(tmp_path) -> None:
    args = Namespace(
        profile=True,
        profile_steps=3,
        precisions=["fp16", "fp8"],
        output=tmp_path / "profile.json",
    )

    with pytest.raises(ValueError, match="exactly one precision"):
        benchmark.validate_profile_args(args)


def test_nvtx_training_ranges_are_balanced(monkeypatch) -> None:
    ranges: list[tuple[str, str | None]] = []

    def push(name: str) -> None:
        ranges.append(("push", name))

    def pop() -> None:
        ranges.append(("pop", None))

    monkeypatch.setattr(benchmark.torch.cuda.nvtx, "range_push", push)
    monkeypatch.setattr(benchmark.torch.cuda.nvtx, "range_pop", pop)

    model = _ToyClassifier()
    artifacts = _ScaledArtifacts(model, _RecordingScaler())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    benchmark.run_training_step(
        model,
        artifacts,
        optimizer,
        criterion,
        torch.randn(2, 3, 4),
        torch.tensor([0, 1, 0]),
        torch.device("cpu"),
        nvtx_step="benchmark_step:training:0",
    )

    pushed = [name for kind, name in ranges if kind == "push"]
    assert pushed[0] == "benchmark_step:training:0"
    for required in ("reset", "zero_grad", "forward", "loss", "backward", "optimizer"):
        assert required in pushed
    assert len(pushed) == sum(kind == "pop" for kind, _ in ranges)
    assert pushed.index("reset") < pushed.index("forward")
    assert pushed.index("forward") < pushed.index("backward")
    assert pushed.index("backward") < pushed.index("optimizer")


def test_profile_hooks_write_first_tensor_metadata_and_balance_nvtx(
    monkeypatch, tmp_path
) -> None:
    ranges: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        benchmark.torch.cuda.nvtx,
        "range_push",
        lambda name: ranges.append(("push", name)),
    )
    monkeypatch.setattr(
        benchmark.torch.cuda.nvtx,
        "range_pop",
        lambda: ranges.append(("pop", None)),
    )

    model = benchmark.DeepFCSNN(
        input_dim=4,
        hidden_dim=4,
        num_classes=2,
        tau=2.0,
        backend="torch",
        depth=2,
        attention_every=0,
        num_heads=1,
    )
    metadata_path = tmp_path / "tensors.jsonl"
    hooks = benchmark._ProfileHooks(model, metadata_path)
    assert not hooks.records
    hooks.active = True
    try:
        x_seq = torch.randn(2, 3, 4, requires_grad=True)
        benchmark.functional.reset_net(model)
        model(x_seq).sum().backward()
    finally:
        hooks.close()

    assert sum(kind == "push" for kind, _ in ranges) == sum(
        kind == "pop" for kind, _ in ranges
    )
    records = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    assert records
    assert {record["event"] for record in records} >= {
        "forward_input",
        "forward_output",
    }
    assert records[0]["value"]


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
        profile=False,
        profile_steps=10,
        profile_module_hooks=True,
        tensor_metadata_output=None,
        fp8_recipe="delayed",
        fp8_fallback_dtype="bf16",
        triton_storage=None,
        triton_fwd="fp32",
        triton_bwd="fp32",
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
