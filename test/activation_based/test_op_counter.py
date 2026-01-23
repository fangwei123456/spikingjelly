import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter, functional


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.lif = neuron.LIFNode(step_mode="m")

    def forward(self, x):
        functional.reset_net(self)
        x = self.fc(x)
        x = self.lif(x)
        return x


class TestCounter(op_counter.BaseCounter):
    def __init__(self):
        super().__init__()
        self.rules = {
            torch.ops.aten.add.Tensor: lambda a, k, o: 1,
            torch.ops.aten.relu.default: lambda a, k, o: 1,
            torch.ops.aten.addmm.default: lambda a, k, o: 1,
            torch.ops.aten.t.default: lambda a, k, o: 1,
            torch.ops.aten.ge.Scalar: lambda a, k, o: 1,
        }
        self.ignore_modules = []


def test_dispatch_counter_basic():
    model = SimpleNet()
    x = torch.randn(2, 3, 4) # [T, N, C]

    counter = TestCounter()

    with op_counter.DispatchCounterMode([counter], verbose=True):
        model(x)

    records = counter.get_counts()
    print(records)

    assert "Global" in records
    assert any("fc" in k for k in records.keys())
    assert any("lif" in k for k in records.keys())

    total = counter.get_total()
    assert total > 0


def test_dispatch_counter_ignore_lif():
    model = SimpleNet()
    x = torch.randn(2, 4)

    counter = TestCounter()
    counter.ignore_modules = [neuron.LIFNode]

    with op_counter.DispatchCounterMode([counter], verbose=True):
        model(x)

    records = counter.get_counts()
    print(records)

    for parent, ops in records.items():
        if "lif" in parent:
            for op in ops:
                assert op != torch.ops.aten.add.Tensor
                assert op != torch.ops.aten.relu.default

    found_linear = any(
        torch.ops.aten.addmm.default in ops.keys()
        for ops in records.values()
    )
    assert found_linear

if __name__ == "__main__":
    test_dispatch_counter_basic()
    test_dispatch_counter_ignore_lif()