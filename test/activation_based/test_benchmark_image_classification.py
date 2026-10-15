from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from benchmark import benchmark_image_classification as benchmark
from benchmark.benchmark_image_classification import _CLIFPoolReference


@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float64),
        pytest.param(
            "cuda",
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            torch.float16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_clif_correctness_pool_retains_spike_rates_and_exact_gradient(device, dtype):
    x = (
        torch.arange(24, device=device, dtype=dtype).reshape(2, 3, 2, 2) % 5 / 4
    ).requires_grad_()
    y = _CLIFPoolReference()(x)
    expected = F.adaptive_avg_pool2d(x.detach().double().cpu(), (7, 7)).to(
        device=device, dtype=dtype
    )
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    g = torch.autograd.grad(y.sum(), x)[0]
    torch.testing.assert_close(g, torch.full_like(x, 12.25), rtol=0, atol=0)
    if device == "cuda":
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            assert torch.isfinite(
                torch.autograd.grad(_CLIFPoolReference()(x).sum(), x)[0]
            ).all()
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_clif_baseline_without_provider_retains_torch_diagnostics(monkeypatch):
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda name: None)
    args = SimpleNamespace(case="clif-tinyimagenet", amp=True)
    result = benchmark._clif_execution(args, torch.device("cuda"), True)
    assert result["selected"] == "torch-reference"
    assert result["state_dtype"] == "torch.float16"


def test_clif_current_provider_diagnostic_errors_are_preserved(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("Triton unavailable")

    monkeypatch.setattr(benchmark.functional, "neuron_implementation", unavailable)
    args = SimpleNamespace(case="clif-tinyimagenet", amp=False)
    with pytest.raises(RuntimeError, match="Triton unavailable"):
        benchmark._clif_execution(args, torch.device("cuda"), False)


def test_clif_correctness_failure_restores_pool_and_training_mode():
    pool = torch.nn.AdaptiveAvgPool2d((7, 7))
    net = torch.nn.Sequential(pool, torch.nn.Flatten(), torch.nn.Linear(147, 200))
    net.eval()
    args = SimpleNamespace(case="clif-tinyimagenet", amp=False, time_steps=1)
    with pytest.raises(ValueError, match="2x2"):
        benchmark._check(
            net,
            torch.zeros(2, 3, 3, 3),
            torch.zeros(2, dtype=torch.long),
            args,
            torch.device("cpu"),
        )
    assert net[0] is pool
    assert not net.training
