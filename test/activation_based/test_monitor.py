import gc
import subprocess
import threading
import weakref
from types import SimpleNamespace

import pytest
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


class InPlaceModule(nn.Module):
    def forward(self, x):
        return x.add_(1)


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


def test_monitor_context_closes_hooks_and_preserves_records():
    net = nn.Linear(2, 1)

    with pytest.raises(RuntimeError):
        with monitor.OutputMonitor(net) as output_monitor:
            net(torch.ones(1, 2))
            raise RuntimeError("stop")

    assert len(output_monitor.records) == 1
    assert output_monitor.hooks == []
    assert net._forward_hooks == {}

    monitor_ref = weakref.ref(output_monitor)
    output_monitor.close()
    del output_monitor
    gc.collect()
    assert monitor_ref() is None


def test_input_monitor_records_before_forward():
    net = InPlaceModule()
    input_monitor = monitor.InputMonitor(net, function_on_input=torch.Tensor.clone)

    x = torch.zeros(1)
    y = net(x)

    torch.testing.assert_close(input_monitor.records[0], torch.zeros(1))
    torch.testing.assert_close(y, torch.ones(1))


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


def test_gpu_monitor_stop_interrupts_interval(monkeypatch):
    sampled = threading.Event()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        sampled.set()
        return SimpleNamespace(returncode=0, stdout="0, 5, 123\n", stderr="")

    monkeypatch.setattr(monitor.subprocess, "run", run)
    gpu_monitor = monitor.GPUMonitor(interval=600)
    assert sampled.wait(1)

    gpu_monitor.stop()
    gpu_monitor.join(timeout=1)

    assert not gpu_monitor.is_alive()
    assert calls[0][0] == [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
    ]
    assert calls[0][1]["timeout"] == 5


def test_gpu_monitor_writes_scalars_and_closes_writer(monkeypatch):
    class Writer:
        def __init__(self, log_dir):
            self.log_dir = log_dir
            self.scalars = []
            self.close_calls = 0

        def add_scalar(self, *args):
            self.scalars.append(args)

        def close(self):
            self.close_calls += 1

    monkeypatch.setattr(monitor, "SummaryWriter", Writer)
    gpu_monitor = monitor.GPUMonitor(
        log_dir="logs", gpu_ids=(2, 3), interval=1, start_now=False
    )

    def run(*_args, **_kwargs):
        gpu_monitor.stop()
        return SimpleNamespace(
            returncode=0,
            stdout="2, 10, 200\n3, 20, 300\n",
            stderr="",
        )

    monkeypatch.setattr(monitor.subprocess, "run", run)
    gpu_monitor.run()

    assert gpu_monitor.writer.log_dir == "logs/gpu_monitor"
    assert gpu_monitor.writer.scalars == [
        ("utilization_2", 10, 0),
        ("memory_used_2", 200, 0),
        ("utilization_3", 20, 0),
        ("memory_used_3", 300, 0),
    ]
    assert gpu_monitor.writer.close_calls == 1


@pytest.mark.parametrize(
    "result",
    (
        SimpleNamespace(returncode=1, stdout="", stderr="failed"),
        SimpleNamespace(returncode=0, stdout="bad output", stderr=""),
    ),
)
def test_gpu_monitor_stops_on_command_or_parse_error(monkeypatch, result):
    errors = []
    close_calls = []
    fake_logger = SimpleNamespace(
        info=lambda *_args: None,
        error=lambda message, *args: errors.append(message.format(*args)),
    )
    monkeypatch.setattr(monitor, "logger", fake_logger)
    monkeypatch.setattr(
        monitor,
        "SummaryWriter",
        lambda _path: SimpleNamespace(close=lambda: close_calls.append(True)),
    )
    monkeypatch.setattr(monitor.subprocess, "run", lambda *_args, **_kwargs: result)

    gpu_monitor = monitor.GPUMonitor(log_dir="logs", start_now=False)
    gpu_monitor.run()

    assert errors
    assert gpu_monitor.step == 0
    assert close_calls == [True]


@pytest.mark.parametrize(
    "command_error",
    (subprocess.TimeoutExpired("nvidia-smi", 5), FileNotFoundError("nvidia-smi")),
)
def test_gpu_monitor_stops_on_command_exception(monkeypatch, command_error):
    errors = []
    close_calls = []
    fake_logger = SimpleNamespace(
        info=lambda *_args: None,
        error=lambda message, *args: errors.append(message.format(*args)),
    )
    monkeypatch.setattr(monitor, "logger", fake_logger)
    monkeypatch.setattr(
        monitor,
        "SummaryWriter",
        lambda _path: SimpleNamespace(close=lambda: close_calls.append(True)),
    )

    def fail(*_args, **_kwargs):
        raise command_error

    monkeypatch.setattr(monitor.subprocess, "run", fail)
    gpu_monitor = monitor.GPUMonitor(log_dir="logs", start_now=False)
    gpu_monitor.run()

    assert errors
    assert gpu_monitor.step == 0
    assert close_calls == [True]


def test_gpu_monitor_validates_configuration_and_stopped_start(monkeypatch):
    with pytest.raises(ValueError, match="gpu_ids"):
        monitor.GPUMonitor(gpu_ids=(), start_now=False)
    with pytest.raises(ValueError, match="interval"):
        monitor.GPUMonitor(interval=0, start_now=False)

    calls = []
    monkeypatch.setattr(
        monitor.subprocess, "run", lambda *_args, **_kwargs: calls.append(True)
    )
    gpu_monitor = monitor.GPUMonitor(start_now=False)
    gpu_monitor.stop()
    gpu_monitor.run()

    assert calls == []
    assert gpu_monitor.writer is None
