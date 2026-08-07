import torch

from spikingjelly.activation_based import neuron, op_counter
from test.activation_based._op_counter_test_utils import TinySNN


def test_dispatch_counter_basic():
    net = TinySNN()
    x = torch.randn(2, 2, 3, 8, 8)

    counter1 = op_counter.FlopCounter()
    counter2 = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode([counter1, counter2], strict=True):
        y = net(x)
        l = y.sum()
        l.backward()

    records1 = counter1.get_counts()
    records2 = counter2.get_counts()

    assert "Global" in records1
    assert any("sn" in k for k in records1.keys())
    assert any("fc" in k for k in records1.keys())
    assert "Global" in records2
    assert any("sn" in k for k in records2.keys())
    assert any("fc" in k for k in records2.keys())

    total1 = counter1.get_total()
    assert total1 > 0
    total2 = counter2.get_total()
    assert total2 > 0


def test_dispatch_counter_ignore():
    net = TinySNN()
    x = torch.randn(2, 2, 3, 8, 8)

    counter1 = op_counter.FlopCounter(extra_ignore_modules=(neuron.LIFNode,))
    counter2 = op_counter.MemoryAccessCounter(extra_ignore_modules=(neuron.LIFNode,))

    with op_counter.DispatchCounterMode([counter1, counter2], strict=True):
        y = net(x)
        l = y.sum()
        l.backward()

    records1 = counter1.get_counts()
    records2 = counter2.get_counts()

    assert "Global" in records1
    assert any("conv" in k for k in records1.keys())
    assert any("fc" in k for k in records1.keys())
    assert "Global" in records2
    assert any("conv" in k for k in records2.keys())
    assert any("fc" in k for k in records2.keys())

    total1 = counter1.get_total()
    assert total1 > 0
    total2 = counter2.get_total()
    assert total2 > 0
