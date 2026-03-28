import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter, functional
from spikingjelly.activation_based.model.sew_resnet import sew_resnet18


def test_ac_basic():
    net = sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    functional.set_step_mode(net, "m")
    x = torch.randn(4, 8, 3, 224, 224)

    counter = op_counter.ACCounter()
    with op_counter.DispatchCounterMode([counter]):
        _ = net(x)

    records = counter.get_counts()
    assert "Global" in records
    assert any("conv" in k for k in records.keys())
    assert any("fc" in k for k in records.keys())

    total = counter.get_total()
    assert total > 0


def test_ac_covers_more_than_synop():
    net = nn.Sequential(
        nn.Linear(100, 50),
        nn.BatchNorm1d(50),
    )
    x = torch.randn(32, 100)

    synop_counter = op_counter.SynOpCounter()
    ac_counter = op_counter.ACCounter()
    with op_counter.DispatchCounterMode([synop_counter, ac_counter], verbose=True):
        _ = net(x)

    total_synop = synop_counter.get_total()
    total_ac = ac_counter.get_total()
    assert total_synop == 0, "float×float matmul should produce 0 SynOps"
    assert total_ac > total_synop


def test_ac_spike_increases_total():
    net = sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    net = nn.Conv2d(3, 3, kernel_size=3, padding=1)
    all_zero = torch.zeros(8, 3, 224, 224)  # all-zero → no spike-driven ACs in conv
    all_one = torch.ones(8, 3, 224, 224)  # all-one → maximum spike-driven ACs

    counter1 = op_counter.ACCounter()
    counter2 = op_counter.ACCounter()
    with op_counter.DispatchCounterMode([counter1]):
        _ = net(all_zero)
    with op_counter.DispatchCounterMode([counter2]):
        _ = net(all_one)

    ac_all_zero = counter1.get_total()
    ac_all_one = counter2.get_total()
    assert ac_all_one > ac_all_zero


def test_ac_element_wise_add():
    x = torch.randn(32, 64)
    y = torch.randn(32, 64)

    counter = op_counter.ACCounter()
    with op_counter.DispatchCounterMode([counter]):
        _ = x + y

    assert counter.get_total() == x.numel()


def test_ac_matmul_float_vs_spike():
    model = nn.Linear(100, 50, bias=False)
    float_x = torch.randn(32, 100)
    spike_x = (torch.rand(32, 100) > 0.8).float()

    counter_float = op_counter.ACCounter()
    with op_counter.DispatchCounterMode([counter_float]):
        model(float_x)
    assert counter_float.get_total() == 0, "float×float should produce 0 AC"

    counter_spike = op_counter.ACCounter()
    with op_counter.DispatchCounterMode([counter_spike]):
        model(spike_x)
    expected = int(spike_x.count_nonzero().item()) * 50
    assert counter_spike.get_total() == expected


def test_ac_ignore():
    net = sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    functional.set_step_mode(net, "m")
    x = torch.randn(4, 8, 3, 224, 224)

    counter_full = op_counter.ACCounter(extra_ignore_modules=[neuron.BaseNode])
    counter_no_bn = op_counter.ACCounter(
        extra_ignore_modules=[nn.BatchNorm2d, neuron.BaseNode]
    )

    with op_counter.DispatchCounterMode([counter_full], verbose=True):
        net(x)
    with op_counter.DispatchCounterMode([counter_no_bn], verbose=True):
        net(x)

    total_full = counter_full.get_total()
    total_no_bn = counter_no_bn.get_total()
    print(
        f"full ACs: {total_full}, without BatchNorm2d: {total_no_bn} ({total_no_bn / total_full * 100:.2f}%)"
    )
    assert total_no_bn < total_full


if __name__ == "__main__":
    test_ac_basic()
    test_ac_covers_more_than_synop()
    test_ac_spike_increases_total()
    test_ac_element_wise_add()
    test_ac_matmul_float_vs_spike()
    test_ac_ignore()
