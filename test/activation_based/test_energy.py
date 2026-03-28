import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter, functional
from spikingjelly.activation_based.model.sew_resnet import sew_resnet18

# 45 nm process, 32-bit floating-point reference costs (pJ)
E_MAC = 4.6
E_AC = 0.9


def test_energy_spike_input():
    net = sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    functional.set_step_mode(net, "m")

    T, B = 8, 4
    x = (torch.rand(T, B, 3, 224, 224) > 0.8).float()

    mac_counter = op_counter.MACCounter(
        extra_ignore_modules=[neuron.BaseNode, nn.modules.batchnorm._BatchNorm]
    )
    ac_counter = op_counter.ACCounter(
        extra_ignore_modules=[neuron.BaseNode, nn.modules.batchnorm._BatchNorm]
    )
    synop_counter = op_counter.SynOpCounter(
        extra_ignore_modules=[neuron.BaseNode, nn.modules.batchnorm._BatchNorm]
    )

    with op_counter.DispatchCounterMode(
        [mac_counter, ac_counter, synop_counter], verbose=True
    ):
        _ = net(x)

    total_mac = mac_counter.get_total()
    total_ac = ac_counter.get_total()
    total_synop = synop_counter.get_total()

    print("\n=== SEW-ResNet-18 forward pass (T=8, B=4, spike input fr≈0.2) ===")
    print(f"  MACs   : {total_mac:>15,}")
    print(f"  ACs    : {total_ac:>15,}")
    print(f"  SynOps : {total_synop:>15,}")
    energy_mac = total_mac * E_MAC
    energy_ac = total_ac * E_AC
    energy_synop = total_synop * E_AC
    print(f"\n  Energy (MAC)   : {energy_mac / 1e9:.4f} nJ  ({E_MAC} pJ/op)")
    print(f"  Energy (AC)    : {energy_ac / 1e9:.4f} nJ  ({E_AC} pJ/op)")
    print(f"  Energy (SynOp) : {energy_synop / 1e9:.4f} nJ  ({E_AC} pJ/op)")
    print(f"  Total          : {(energy_mac + energy_ac) / 1e9:.4f} nJ")

    assert total_mac > 0, "MACs should be positive for float conv/BN layers"
    assert total_ac > 0, "ACs should be positive (BN reductions, residual adds, etc.)"
    assert total_synop > 0, "SynOps should be positive for spike-driven conv layers"


def test_energy_float_input():
    """With dense float input all conv ops become MAC; SynOps should be zero."""
    net = sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    functional.set_step_mode(net, "m")

    T, B = 8, 4
    x = torch.rand(T, B, 3, 224, 224)

    mac_counter = op_counter.MACCounter(
        extra_ignore_modules=[neuron.BaseNode, nn.modules.batchnorm._BatchNorm]
    )
    ac_counter = op_counter.ACCounter(
        extra_ignore_modules=[neuron.BaseNode, nn.modules.batchnorm._BatchNorm]
    )
    synop_counter = op_counter.SynOpCounter(
        extra_ignore_modules=[neuron.BaseNode, nn.modules.batchnorm._BatchNorm]
    )

    with op_counter.DispatchCounterMode(
        [mac_counter, ac_counter, synop_counter], verbose=True
    ):
        _ = net(x)

    total_mac = mac_counter.get_total()
    total_ac = ac_counter.get_total()
    total_synop = synop_counter.get_total()

    print("\n=== SEW-ResNet-18 forward pass (T=4, B=2, float input) ===")
    print(f"  MACs   : {total_mac:>15,}")
    print(f"  ACs    : {total_ac:>15,}")
    print(f"  SynOps : {total_synop:>15,}")
    assert total_mac > 0
    assert total_ac > 0
    assert total_synop > 0
    energy_mac = total_mac * E_MAC
    energy_ac = total_ac * E_AC
    energy_synop = total_synop * E_AC
    print(f"\n  Energy (MAC)   : {energy_mac / 1e9:.4f} nJ  ({E_MAC} pJ/op)")
    print(f"  Energy (AC)    : {energy_ac / 1e9:.4f} nJ  ({E_AC} pJ/op)")
    print(f"  Energy (SynOp) : {energy_synop / 1e9:.4f} nJ  ({E_AC} pJ/op)")
    print(f"  Total          : {(energy_mac + energy_ac) / 1e9:.4f} nJ")

    assert total_mac > 0, "MACs should be positive for float conv/BN layers"
    assert total_ac > 0, "ACs should be positive (BN reductions, residual adds, etc.)"
    assert total_synop > 0, "SynOps should be positive for spike-driven conv layers"


if __name__ == "__main__":
    test_energy_spike_input()
    print("\n")
    for _ in range(10):
        print("=" * 100)
    print("\n")
    test_energy_float_input()
