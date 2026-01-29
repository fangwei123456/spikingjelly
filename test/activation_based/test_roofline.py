import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter, functional


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.pool = nn.Sequential(nn.AvgPool2d(5, 1), nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(32, 10)
        self.ww = nn.Parameter(torch.randn(10, 5))
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
    x = torch.randn(8, 3, 224, 224)

    counter1 = op_counter.FlopCounter()
    counter2 = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode([counter1, counter2], verbose=True):
        model(x)

    records1 = counter1.get_counts()
    print(records1)
    records2 = counter2.get_counts()
    print(records2)

    assert "Global" in records1
    assert any("fc" in k for k in records1.keys())
    assert any("lif" in k for k in records1.keys())
    assert "Global" in records2
    assert any("fc" in k for k in records2.keys())
    assert any("lif" in k for k in records2.keys())

    total1 = counter1.get_total()
    print("total FLOPs:", total1)
    assert total1 > 0
    total2 = counter2.get_total()
    print("total memory access (bytes):", total2)
    print("arithmetic intensity:", total1 / total2, "FLOPs/byte")



if __name__ == "__main__":
    test_dispatch_counter_basic()
