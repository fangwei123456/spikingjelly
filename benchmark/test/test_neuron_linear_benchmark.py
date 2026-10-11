import pytest
import torch
from spikingjelly.activation_based import surrogate
from benchmark.binary_kernel.benchmark_neuron_linear import _methods, _paired_rounds


def test_paired_rounds_alternate_and_keep_pairs():
    calls = []

    def measure(name):
        calls.append(name)
        return (2.0 if name == "baseline" else 1.0, len(calls))

    pairs = _paired_rounds(measure, 3)

    assert calls == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    assert [pair["speedup"] for pair in pairs] == [2.0, 2.0, 2.0]
    assert [pair["order"] for pair in pairs] == [
        ["baseline", "candidate"],
        ["candidate", "baseline"],
        ["baseline", "candidate"],
    ]


@pytest.mark.parametrize("kind", ["if", "lif"])
@pytest.mark.parametrize("mode", ["inference", "train"])
def test_automatic_neuron_projection_matches_reference(kind, mode):
    torch.manual_seed(72)
    x = torch.rand(3, 2, 4, requires_grad=mode == "train")
    v = torch.zeros(2, 4, requires_grad=mode == "train")
    weight = torch.randn(5, 4, requires_grad=mode == "train")
    bias = torch.randn(5, requires_grad=mode == "train")
    methods = _methods(x, v, weight, bias, surrogate.Sigmoid(), mode, kind, [128])
    expected, actual = methods["torch_reference"](), methods["automatic_dense"]()
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want)
    if mode == "train":
        inputs = (x, v, weight, bias)
        for got, want in zip(
            torch.autograd.grad(sum(t.sum() for t in actual), inputs),
            torch.autograd.grad(sum(t.sum() for t in expected), inputs),
            strict=True,
        ):
            torch.testing.assert_close(got, want)
