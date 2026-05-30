import importlib.util

import pytest
import torch

from spikingjelly.activation_based import functional, layer
from spikingjelly.activation_based.model import Spikformer
from spikingjelly.activation_based.precision import (
    Float8LinearStepModule,
    PrecisionConfig,
    prepare_model_for_precision,
)

HAS_TORCHAO = importlib.util.find_spec("torchao") is not None


def _make_tiny_spikformer():
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


def _run_one_training_step(model, device, config: PrecisionConfig, batch_size: int = 2):
    artifacts = prepare_model_for_precision(model, device, config)
    model = artifacts.model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(batch_size, 3, 64, 64, device=device)
    target = torch.randint(0, 16, (batch_size,), device=device)

    functional.reset_net(model)
    optimizer.zero_grad(set_to_none=True)
    with artifacts.autocast_context():
        y = model(x)
        loss = torch.nn.functional.cross_entropy(y.mean(0), target)
    artifacts.backward(loss, optimizer, parameters=model.parameters())
    functional.reset_net(model)
    return artifacts, y, loss


def test_spikformer_precision_tools_support_custom_training_loop_fp32():
    model = _make_tiny_spikformer()
    artifacts, y, loss = _run_one_training_step(
        model,
        torch.device("cpu"),
        PrecisionConfig(mode="fp32"),
    )

    assert y.shape == (2, 2, 16)
    assert torch.isfinite(loss)
    report = artifacts.policy.conversion_report()
    assert report["convertible_linear"] >= 1
    assert report["converted_modules"] == []


def test_precision_artifacts_backward_supports_accumulation_without_scaler():
    model = _make_tiny_spikformer()
    artifacts = prepare_model_for_precision(
        model,
        torch.device("cpu"),
        PrecisionConfig(mode="fp32"),
    )
    model = artifacts.model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 16, (2,))

    functional.reset_net(model)
    optimizer.zero_grad(set_to_none=True)
    with artifacts.autocast_context():
        y = model(x)
        loss = torch.nn.functional.cross_entropy(y.mean(0), target)
    grad_norm = artifacts.backward(
        loss,
        optimizer,
        clip_grad_norm=1.0,
        step_optimizer=False,
    )
    assert grad_norm is not None
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.skipif(
    not HAS_TORCHAO
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability(0) < (8, 9),
    reason="This fp8-torchao smoke test requires CUDA compute capability >= 8.9.",
)
def test_fp8_torchao_aligned_linear_smoke():
    model = torch.nn.Sequential(
        layer.Linear(16, 32),
        torch.nn.ReLU(),
        layer.Linear(32, 16),
    ).train().to("cuda:0")
    artifacts = prepare_model_for_precision(
        model,
        torch.device("cuda:0"),
        PrecisionConfig(mode="fp8-torchao", strictness="strict", device="cuda:0"),
    )
    model = artifacts.model.to("cuda:0")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(16, 16, device="cuda:0")
    optimizer.zero_grad(set_to_none=True)
    with artifacts.autocast_context():
        y = model(x)
        loss = y.sum()
    artifacts.backward(loss, optimizer, parameters=model.parameters())

    assert y.shape == (16, 16)
    assert torch.isfinite(loss)
    assert any(isinstance(m, Float8LinearStepModule) for m in model.modules())


@pytest.mark.skipif(
    not HAS_TORCHAO or not torch.cuda.is_available(),
    reason="CUDA is required to validate fp8-torchao capability handling.",
)
def test_spikformer_precision_tools_fp8_torchao_smoke():
    model = _make_tiny_spikformer()
    config = PrecisionConfig(mode="fp8-torchao", strictness="strict", device="cuda:0")
    if torch.cuda.get_device_capability(0) >= (8, 9):
        artifacts, y, loss = _run_one_training_step(
            model.to("cuda:0"),
            torch.device("cuda:0"),
            config,
            batch_size=16,
        )
        assert y.shape == (2, 16, 16)
        assert torch.isfinite(loss)
        converted = artifacts.policy.conversion_report()["converted_modules"]
        assert converted
        assert any(
            isinstance(m, Float8LinearStepModule) for m in artifacts.model.modules()
        )
    else:
        with pytest.raises(RuntimeError):
            prepare_model_for_precision(model, torch.device("cuda:0"), config)


@pytest.mark.skipif(
    not HAS_TORCHAO or torch.cuda.device_count() < 2,
    reason="This fp8-torchao device mismatch test requires torchao and at least 2 CUDA devices.",
)
def test_fp8_torchao_rejects_model_on_different_cuda_device():
    model = torch.nn.Sequential(
        layer.Linear(16, 32),
        torch.nn.ReLU(),
        layer.Linear(32, 16),
    ).to("cuda:0")
    config = PrecisionConfig(mode="fp8-torchao", strictness="strict", device="cuda:1")

    with pytest.raises(RuntimeError, match="target CUDA device 'cuda:1'"):
        prepare_model_for_precision(model, torch.device("cuda:1"), config)


@pytest.mark.skipif(
    not HAS_TORCHAO
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability(0) < (8, 9),
    reason="This fp8-torchao unindexed CUDA test requires torchao and CUDA compute capability >= 8.9.",
)
def test_fp8_torchao_accepts_unindexed_cuda_device_string():
    model = torch.nn.Sequential(
        layer.Linear(16, 32),
        torch.nn.ReLU(),
        layer.Linear(32, 16),
    ).to("cuda:0")
    artifacts = prepare_model_for_precision(
        model,
        "cuda",
        PrecisionConfig(mode="fp8-torchao", strictness="strict", device="cuda"),
    )

    assert any(isinstance(m, Float8LinearStepModule) for m in artifacts.model.modules())
