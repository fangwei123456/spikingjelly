import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter
from _op_counter_test_utils import TinySNN


def test_synop_basic():
    net = TinySNN()
    x = (torch.rand(2, 2, 3, 8, 8) > 0.5).float()

    counter = op_counter.SynOpCounter()
    with op_counter.DispatchCounterMode([counter]):
        _ = net(x)

    records = counter.get_counts()
    assert "Global" in records
    assert any("conv" in k for k in records.keys())
    assert any("fc" in k for k in records.keys())

    total = counter.get_total()
    assert total > 0


def test_synop_float_vs_spike():
    model = nn.Linear(100, 50, bias=False)
    float_x = torch.randn(32, 100)
    spike_x = (torch.rand(32, 100) > 0.8).float()

    counter_float = op_counter.SynOpCounter()
    with op_counter.DispatchCounterMode([counter_float]):
        model(float_x)
    assert counter_float.get_total() == 0, "float×float should produce 0 SynOps"

    counter_spike = op_counter.SynOpCounter()
    with op_counter.DispatchCounterMode([counter_spike]):
        model(spike_x)
    expected = int(spike_x.count_nonzero().item()) * 50
    assert counter_spike.get_total() == expected


def test_synop_ignore():
    net = TinySNN()
    x = (torch.rand(2, 2, 3, 8, 8) > 0.7).float()

    counter_full = op_counter.SynOpCounter(extra_ignore_modules=[neuron.BaseNode])
    counter_no_conv = op_counter.SynOpCounter(
        extra_ignore_modules=[nn.Conv2d, neuron.BaseNode]
    )

    with op_counter.DispatchCounterMode([counter_full]):
        net(x)

    with op_counter.DispatchCounterMode([counter_no_conv]):
        net(x)

    total_full = counter_full.get_total()
    total_no_conv = counter_no_conv.get_total()
    assert total_no_conv < total_full
