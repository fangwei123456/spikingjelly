import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter, functional
from spikingjelly.activation_based.model.sew_resnet import sew_resnet18


def test_dispatch_counter_basic():
    net = sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    functional.set_step_mode(net, "m")
    x = torch.randn(4, 8, 3, 224, 224)

    counter1 = op_counter.FlopCounter()
    counter2 = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode([counter1, counter2], verbose=True, strict=True):
       y = net(x)
       l = y.sum()
       l.backward()

    records1 = counter1.get_counts()
    print(records1)
    records2 = counter2.get_counts()
    print(records2)

    assert "Global" in records1
    assert any("sn1" in k for k in records1.keys())
    assert any("fc" in k for k in records1.keys())
    assert "Global" in records2
    assert any("sn2" in k for k in records2.keys())
    assert any("fc" in k for k in records2.keys())

    total1 = counter1.get_total()
    print("total FLOPs:", total1)
    assert total1 > 0
    total2 = counter2.get_total()
    print("total memory access (bytes):", total2)
    print("arithmetic intensity:", total1 / total2, "FLOPs/byte")


def test_dispatch_counter_ignore():
    net = sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    functional.set_step_mode(net, "m")
    x = torch.randn(4, 8, 3, 224, 224)

    counter1 = op_counter.FlopCounter(extra_ignore_modules=(neuron.LIFNode,))
    counter2 = op_counter.MemoryAccessCounter(extra_ignore_modules=(neuron.LIFNode,))

    with op_counter.DispatchCounterMode([counter1, counter2], verbose=True, strict=True):
        y = net(x)
        l = y.sum()
        l.backward()

    records1 = counter1.get_counts()
    print(records1)
    records2 = counter2.get_counts()
    print(records2)

    assert "Global" in records1
    assert any("conv" in k for k in records1.keys())
    assert any("fc" in k for k in records1.keys())
    assert "Global" in records2
    assert any("conv" in k for k in records2.keys())
    assert any("fc" in k for k in records2.keys())

    total1 = counter1.get_total()
    print("total FLOPs:", total1)
    assert total1 > 0
    total2 = counter2.get_total()
    print("total memory access (bytes):", total2)
    print("arithmetic intensity:", total1 / total2, "FLOPs/byte")


if __name__ == "__main__":
    test_dispatch_counter_basic()
    test_dispatch_counter_ignore()
