import torch
import torch.nn as nn
import pytest

from spikingjelly.activation_based import neuron, op_counter, functional


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, 3, 1, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(5, 4)
        self.ww = nn.Parameter(torch.randn(4, 7))
        self.lif = neuron.LIFNode(step_mode="m")

    def forward(self, x):
        functional.reset_net(self)
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x @ self.ww
        x = torch.stack([x, x], dim=0)
        x = torch.stack([x, x], dim=0) @ x.transpose(-1, -2)
        x = torch.logical_and(x > 1, x > 0).float()
        x = self.lif(x)
        return x


def test_dispatch_counter_basic():
    model = SimpleNet()
    x = torch.randn(2, 3, 5, 5)

    counter = op_counter.FlopCounter()

    with op_counter.DispatchCounterMode([counter], verbose=True):
        model(x)

    records = counter.get_counts()
    print(records)

    assert "Global" in records
    assert any("fc" in k for k in records.keys())
    assert any("lif" in k for k in records.keys())

    total = counter.get_total()
    print("total:", total)
    assert total > 0


def test_dispatch_counter_ignore_lif():
    model = SimpleNet()
    x = torch.randn(2, 3, 5, 5)

    counter = op_counter.FlopCounter()
    counter.ignore_modules = [neuron.LIFNode]

    with op_counter.DispatchCounterMode([counter], verbose=True):
        model(x)

    records = counter.get_counts()
    print(records)

    total = counter.get_total()
    print("total:", total)
    assert total > 0

    for parent, ops in records.items():
        if "lif" in parent:
            raise AssertionError("LIFNode should be ignored")

    found_linear = any(
        torch.ops.aten.addmm.default in ops.keys() for ops in records.values()
    )
    assert found_linear


def test_ignore_modules_list():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 5, 3)
            self.conv2 = nn.Conv2d(5, 8, 3)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    model = Net()
    x = torch.randn(2, 3, 8, 8)

    counter = op_counter.FlopCounter()

    with op_counter.DispatchCounterMode([counter], verbose=True):
        model(x)

    records = counter.get_counts()
    print("records after ignoring conv1 and conv2:", records)

    assert "Net.conv1" in records
    assert "Net.conv2" in records

    total = counter.get_total()
    print("total:", total)
    assert total > 0


def test_ignore_module_type():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 5, 3)
            self.lif = neuron.LIFNode(step_mode="m")

        def forward(self, x):
            x = self.conv(x)
            x = self.lif(x)
            return x

    model = Net()
    x = torch.randn(2, 3, 5, 5)

    counter = op_counter.FlopCounter()
    counter.ignore_modules = [neuron.LIFNode]

    with op_counter.DispatchCounterMode([counter], verbose=True):
        model(x)

    records = counter.get_counts()
    print("records after ignoring LIFNode:", records)

    total = counter.get_total()
    print("total:", total)
    assert total > 0

    for parent, ops in records.items():
        if "lif" in parent:
            raise AssertionError("LIFNode should be ignored")

    found_conv = any(
        torch.ops.aten.convolution.default in ops.keys() for ops in records.values()
    )
    assert found_conv


def test_strict_mode():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 5, 3)

        def forward(self, x):
            x = self.conv(x)
            return x

    model = Net()
    x = torch.randn(2, 3, 5, 5)

    counter = op_counter.FlopCounter()

    with op_counter.DispatchCounterMode([counter], strict=True, verbose=True):
        model(x)

    records = counter.get_counts()
    print("records:", records)

    total = counter.get_total()
    print("total:", total)
    assert total > 0


def test_strict_mode_error():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 5, 3)
            self.gelu = nn.GELU()

        def forward(self, x):
            x = self.conv(x)
            x = self.gelu(x)
            return x

    model = Net()
    x = torch.randn(2, 3, 5, 5)

    counter = op_counter.FlopCounter()

    with pytest.raises(NotImplementedError):
        with op_counter.DispatchCounterMode([counter], strict=True, verbose=True):
            model(x)


def test_multiple_counters():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 5, 3)
            self.fc = nn.Linear(5, 4)

        def forward(self, x):
            x = self.conv(x)
            x = torch.flatten(x.mean(dim=[-2, -1]), start_dim=1)
            x = self.fc(x)
            return x

    model = Net()
    x = torch.randn(2, 3, 5, 5)

    counter1 = op_counter.FlopCounter()
    counter2 = op_counter.FlopCounter()

    with op_counter.DispatchCounterMode([counter1, counter2], verbose=True):
        model(x)

    records1 = counter1.get_counts()
    records2 = counter2.get_counts()

    assert records1 == records2

    total1 = counter1.get_total()
    total2 = counter2.get_total()

    assert total1 == total2
    assert total1 > 0


def test_output_dict_structure():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 5, 3)

        def forward(self, x):
            return self.conv(x)

    model = Net()
    x = torch.randn(2, 3, 5, 5)

    counter = op_counter.FlopCounter()

    with op_counter.DispatchCounterMode([counter], verbose=False):
        model(x)

    records = counter.get_counts()

    assert isinstance(records, dict)
    assert "Global" in records

    for module_name, ops in records.items():
        assert isinstance(ops, dict)
        for op, count in ops.items():
            assert isinstance(count, int)
            assert count >= 0


def test_get_total():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 5, 3)

        def forward(self, x):
            return self.conv(x)

    model = Net()
    x = torch.randn(2, 3, 5, 5)

    counter = op_counter.FlopCounter()

    with op_counter.DispatchCounterMode([counter], verbose=False):
        model(x)

    records = counter.get_counts()
    total = counter.get_total()

    expected_total = sum(records["Global"].values())

    assert total == expected_total


def test_extra_rules():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 20)

        def forward(self, x):
            return self.fc(x)

    model = Net()
    x = torch.randn(3, 10)

    counter = op_counter.FlopCounter()

    def custom_rule(args, kwargs, out):
        return 42

    counter = op_counter.FlopCounter(
        extra_rules={torch.ops.aten.addmm.default: custom_rule}
    )

    with op_counter.DispatchCounterMode([counter], verbose=False):
        y = model(x)

    records = counter.get_counts()
    total = counter.get_total()

    assert records["Net.fc"][torch.ops.aten.addmm.default] == 42
    assert total == 42


def test_extra_ignore_modules():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 5, 3)
            self.conv2 = nn.Conv2d(5, 8, 3)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    model = Net()
    x = torch.randn(2, 3, 8, 8)

    counter = op_counter.FlopCounter()

    class CustomModule(nn.Module):
        def forward(self, x):
            return x * 2

    counter = op_counter.FlopCounter(extra_ignore_modules=[CustomModule])

    with op_counter.DispatchCounterMode([counter], verbose=False):
        model(x)

    records = counter.get_counts()
    print("records with extra ignore modules:", records)

    assert "Net.conv1" in records
    assert "Net.conv2" in records

    total = counter.get_total()
    print("total:", total)
    assert total > 0


if __name__ == "__main__":
    test_dispatch_counter_basic()
    test_dispatch_counter_ignore_lif()
    test_ignore_modules_list()
    test_ignore_module_type()
    test_strict_mode()
    test_strict_mode_error()
    test_multiple_counters()
    test_output_dict_structure()
    test_get_total()
    test_extra_rules()
    test_extra_ignore_modules()
