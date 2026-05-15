import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import op_counter, neuron


class ToyCLIFNode(neuron.BaseNode):
    def __init__(self):
        super().__init__(v_threshold=1.0, v_reset=None, step_mode="s", backend="torch")
        self.register_memory("m", 0.0)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        if isinstance(self.m, float):
            self.m = torch.full_like(x, self.m, requires_grad=False)
        self.neuronal_charge(x)
        self.m = self.m * torch.sigmoid(self.v)
        spike = self.neuronal_fire()
        self.m = self.m + spike
        self.neuronal_reset(spike)
        self.v = self.v - spike * torch.sigmoid(self.m)
        return spike


def test_neuron_state_counter_ifnode_multi_step_has_state_metrics():
    node = neuron.IFNode(step_mode="m")
    x = torch.rand(4, 2, 3)
    counter = op_counter.NeuronStateCounter()

    with op_counter.DispatchCounterMode([counter]):
        _ = node(x)

    metrics = counter.get_metric_counts()["Global"]
    projection = counter.get_projection_counts()["Global"]

    assert metrics["state_reads"] > 0
    assert metrics["state_writes"] > 0
    assert metrics["state_adds"] > 0
    assert projection["read_potential"] == metrics["state_reads"]
    assert projection["write_potential"] == metrics["state_writes"]


def test_neuron_state_counter_lif_projection_tracks_potential_access():
    node = neuron.LIFNode(step_mode="m", decay_input=False)
    x = torch.rand(3, 2, 4)
    counter = op_counter.NeuronStateCounter()

    with op_counter.DispatchCounterMode([counter]):
        _ = node(x)

    projection = counter.get_projection_counts()["Global"]
    assert projection["read_potential"] > 0
    assert projection["write_potential"] > 0
    assert projection["state_mac_like"] > 0


def test_neuron_state_counter_toy_clif_has_more_state_traffic_than_lif():
    lif = neuron.LIFNode(step_mode="s", decay_input=False)
    clif = ToyCLIFNode()
    x = torch.rand(2, 5)

    lif_counter = op_counter.NeuronStateCounter()
    with op_counter.DispatchCounterMode([lif_counter]):
        _ = lif(x)

    clif_counter = op_counter.NeuronStateCounter()
    with op_counter.DispatchCounterMode([clif_counter]):
        _ = clif(x)

    lif_projection = lif_counter.get_projection_counts()["Global"]
    clif_projection = clif_counter.get_projection_counts()["Global"]
    assert clif_projection["read_potential"] > lif_projection["read_potential"]
    assert clif_projection["write_potential"] > lif_projection["write_potential"]


def test_neuron_state_counter_does_not_change_other_counters_when_neurons_ignored():
    model = nn.Sequential(nn.Linear(6, 6, bias=False), neuron.IFNode())
    x = torch.rand(3, 6)

    def run(include_state: bool):
        mac = op_counter.MACCounter(extra_ignore_modules=[neuron.BaseNode])
        ac = op_counter.ACCounter(extra_ignore_modules=[neuron.BaseNode])
        mem = op_counter.MemoryAccessCounter(extra_ignore_modules=[neuron.BaseNode])
        counters = [mac, ac, mem]
        if include_state:
            counters.append(op_counter.NeuronStateCounter())
        with op_counter.DispatchCounterMode(counters):
            _ = model(x)
        return mac.get_total(), ac.get_total(), mem.get_total()

    assert run(include_state=False) == run(include_state=True)
