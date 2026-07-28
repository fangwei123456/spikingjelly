import torch
from torch import nn

from spikingjelly.activation_based import monitor


class StatefulModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = 0

    def forward(self, x):
        self.value += 1
        return x


def test_input_and_output_monitor_lifecycle():
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    input_monitor = monitor.InputMonitor(net, nn.Linear)
    output_calls = []

    def record_output_call(output):
        output_calls.append(output)
        return output

    output_monitor = monitor.OutputMonitor(
        net, nn.Linear, function_on_output=record_output_call
    )

    x = torch.ones(1, 2)
    net(x)

    assert input_monitor.monitored_layers == ["0", "2"]
    assert output_monitor.monitored_layers == ["0", "2"]
    assert len(input_monitor.records) == 2
    torch.testing.assert_close(input_monitor["0"][0], input_monitor[0])
    assert len(output_monitor["2"]) == 1
    assert len(output_calls) == 2

    input_monitor.disable()
    output_monitor.disable()
    net(x)
    assert len(input_monitor.records) == 2
    assert len(output_monitor.records) == 2
    assert len(output_calls) == 2

    input_monitor.clear_recorded_data()
    output_monitor.clear_recorded_data()
    assert input_monitor.records == []
    assert input_monitor["0"] == []

    input_monitor.enable()
    output_monitor.enable()
    input_monitor.remove_hooks()
    output_monitor.remove_hooks()
    net(x)
    assert input_monitor.records == []
    assert output_monitor.records == []
    assert input_monitor.hooks == []
    assert output_monitor.hooks == []


def test_attribute_monitor_before_and_after_forward():
    net = StatefulModule()
    before = monitor.AttributeMonitor("value", True, net)
    after = monitor.AttributeMonitor("value", False, net)

    net(torch.ones(1))
    net(torch.ones(1))

    assert before.records == [0, 1]
    assert after.records == [1, 2]


def test_gradient_monitors_record_input_and_output_gradients():
    net = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        net.weight.copy_(2 * torch.eye(2))

    grad_input_monitor = monitor.GradInputMonitor(net)
    grad_output_monitor = monitor.GradOutputMonitor(net)
    x = torch.ones(1, 2, requires_grad=True)
    net(x).sum().backward()

    torch.testing.assert_close(grad_input_monitor.records[0], torch.full_like(x, 2))
    torch.testing.assert_close(grad_output_monitor.records[0], torch.ones_like(x))
    torch.testing.assert_close(grad_input_monitor[""][0], grad_input_monitor[0])
    torch.testing.assert_close(grad_output_monitor[""][0], grad_output_monitor[0])
