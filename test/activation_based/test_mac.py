import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter, functional
from spikingjelly.activation_based.model.sew_resnet import sew_resnet18


def test_mac_basic():
    net = sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    functional.set_step_mode(net, "m")
    x = torch.randn(4, 8, 3, 224, 224)

    counter = op_counter.MACCounter()
    with op_counter.DispatchCounterMode([counter]):
        _ = net(x)

    records = counter.get_counts()
    assert "Global" in records
    assert any("conv" in k for k in records.keys())
    assert any("fc" in k for k in records.keys())

    total = counter.get_total()
    assert total > 0


def test_mac_float_vs_spike():
    model = nn.Linear(100, 50, bias=False)
    float_x = torch.randn(32, 100)
    spike_x = (torch.rand(32, 100) > 0.8).float()

    counter_float = op_counter.MACCounter()
    with op_counter.DispatchCounterMode([counter_float]):
        model(float_x)
    # mm: m=32, k=100, n=50 → 32*50*100 MACs
    expected = 32 * 50 * 100
    assert counter_float.get_total() == expected, (
        f"float×float should produce {expected} MACs, got {counter_float.get_total()}"
    )

    counter_spike = op_counter.MACCounter()
    with op_counter.DispatchCounterMode([counter_spike]):
        model(spike_x)
    assert counter_spike.get_total() == 0, "spike×float should produce 0 MACs"


def test_mac_bn_affine():
    # eval mode: no running stats update, only affine FMA
    bn_affine = nn.BatchNorm1d(50)
    bn_no_affine = nn.BatchNorm1d(50, affine=False)
    bn_affine.eval()
    bn_no_affine.eval()
    x = torch.randn(32, 50)

    counter_affine = op_counter.MACCounter()
    with op_counter.DispatchCounterMode([counter_affine]):
        bn_affine(x)
    assert counter_affine.get_total() == x.numel(), (
        f"BN with affine=True should count {x.numel()} MACs, got {counter_affine.get_total()}"
    )

    counter_no_affine = op_counter.MACCounter()
    with op_counter.DispatchCounterMode([counter_no_affine]):
        bn_no_affine(x)
    assert counter_no_affine.get_total() == 0, (
        "BN with affine=False should produce 0 MACs"
    )


def test_mac_bn_running_stats():
    # train mode: affine FMA + running stats FMA (delta form)
    bn = nn.BatchNorm1d(50)  # affine=True, track_running_stats=True
    bn_no_rs = nn.BatchNorm1d(50, track_running_stats=False)  # no running stats update
    x = torch.randn(32, 50)

    counter = op_counter.MACCounter()
    with op_counter.DispatchCounterMode([counter]):
        bn(x)
    # affine: n=1600; running stats: 2*c=100 (mean + var, delta FMA)
    expected = x.numel() + 2 * 50
    assert counter.get_total() == expected, (
        f"BN train+running_stats should count {expected} MACs, got {counter.get_total()}"
    )

    counter_no_rs = op_counter.MACCounter()
    with op_counter.DispatchCounterMode([counter_no_rs]):
        bn_no_rs(x)
    assert counter_no_rs.get_total() == x.numel(), (
        f"BN train, no running_stats should count {x.numel()} MACs (affine only), "
        f"got {counter_no_rs.get_total()}"
    )


def test_mac_ac_mutually_exclusive():
    # float×float matmul: all work is MAC, zero AC
    model = nn.Linear(100, 50, bias=False)
    x = torch.randn(32, 100)

    mac_counter = op_counter.MACCounter()
    ac_counter = op_counter.ACCounter()
    with op_counter.DispatchCounterMode([mac_counter, ac_counter]):
        model(x)

    assert mac_counter.get_total() == 32 * 50 * 100
    assert ac_counter.get_total() == 0, "float×float matmul should produce 0 AC"

    # spike×float matmul: all work is AC, zero MAC
    spike_x = (torch.rand(32, 100) > 0.8).float()
    mac_counter2 = op_counter.MACCounter()
    ac_counter2 = op_counter.ACCounter()
    with op_counter.DispatchCounterMode([mac_counter2, ac_counter2]):
        model(spike_x)

    assert mac_counter2.get_total() == 0
    assert ac_counter2.get_total() == int(spike_x.count_nonzero().item()) * 50


def test_mac_ignore():
    net = sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    functional.set_step_mode(net, "m")
    x = torch.randn(4, 8, 3, 224, 224)

    counter_full = op_counter.MACCounter(extra_ignore_modules=[neuron.BaseNode])
    counter_no_bn = op_counter.MACCounter(
        extra_ignore_modules=[nn.BatchNorm2d, neuron.BaseNode]
    )

    with op_counter.DispatchCounterMode([counter_full], verbose=True):
        net(x)
    with op_counter.DispatchCounterMode([counter_no_bn], verbose=True):
        net(x)

    total_full = counter_full.get_total()
    total_no_bn = counter_no_bn.get_total()
    print(
        f"full MACs: {total_full}, without BatchNorm2d: {total_no_bn} ({total_no_bn / total_full * 100:.2f}%)"
    )
    assert total_no_bn < total_full


if __name__ == "__main__":
    # test_mac_basic()
    # test_mac_float_vs_spike()
    # test_mac_bn_affine()
    # test_mac_bn_running_stats()
    # test_mac_ac_mutually_exclusive()
    test_mac_ignore()
