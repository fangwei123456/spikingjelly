import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter, functional
from spikingjelly.activation_based.model.sew_resnet import sew_resnet18


def test_synop_basic():
    net = sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    functional.set_step_mode(net, "m")
    x = (torch.rand(4, 8, 3, 224, 224) > 0.5).float()

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
    print(
        f"nnz={int(spike_x.count_nonzero().item())}, expected SynOps={expected}, got={counter_spike.get_total()}"
    )
    assert counter_spike.get_total() == expected


def test_synop_ignore():
    net = sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    functional.set_step_mode(net, "m")
    x = (torch.rand(4, 8, 3, 224, 224) > 0.7).float()

    counter_full = op_counter.SynOpCounter(extra_ignore_modules=[neuron.BaseNode])
    counter_no_conv = op_counter.SynOpCounter(
        extra_ignore_modules=[nn.Conv2d, neuron.BaseNode]
    )

    with op_counter.DispatchCounterMode([counter_full], verbose=True):
        net(x)

    with op_counter.DispatchCounterMode([counter_no_conv], verbose=True):
        net(x)

    total_full = counter_full.get_total()
    total_no_conv = counter_no_conv.get_total()
    print(f"full SynOps: {total_full}, without Conv2d: {total_no_conv}")
    assert total_no_conv < total_full


if __name__ == "__main__":
    test_synop_basic()
    test_synop_float_vs_spike()
    test_synop_ignore()
