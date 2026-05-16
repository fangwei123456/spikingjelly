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
    assert (
        clif_projection["read_potential"] + clif_projection["write_potential"]
        > lif_projection["read_potential"] + lif_projection["write_potential"]
    )


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


def test_neuron_state_counter_inplace_nonlinear_counts_state_write():
    class InplaceNonlinearNode(neuron.BaseNode):
        def __init__(self):
            super().__init__(v_threshold=1.0, v_reset=None, step_mode="s")

        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x

        def single_step_forward(self, x: torch.Tensor):
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            self.v = self.v.sigmoid_()
            return self.v

    node = InplaceNonlinearNode()
    counter = op_counter.NeuronStateCounter()

    with op_counter.DispatchCounterMode([counter]):
        _ = node(torch.rand(2, 4))

    metrics = counter.get_metric_counts()["Global"]
    assert metrics["state_nonlinear_ops"] > 0
    assert metrics["state_writes"] > 0


def test_neuron_state_counter_counts_state_access_in_bytes_for_views():
    class ViewStateNode(neuron.BaseNode):
        def __init__(self):
            super().__init__(v_threshold=1.0, v_reset=None, step_mode="s")

        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x

        def single_step_forward(self, x: torch.Tensor):
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            y = self.v.view_as(x)
            self.v = y + 1.0
            return self.v

    x = torch.rand(2, 4, dtype=torch.float32)
    node = ViewStateNode()
    counter = op_counter.NeuronStateCounter()

    with op_counter.DispatchCounterMode([counter]):
        _ = node(x)

    metrics = counter.get_metric_counts()["Global"]
    expected_bytes = x.numel() * x.element_size()
    assert metrics["state_reads"] >= expected_bytes
    assert metrics["state_writes"] >= expected_bytes


def test_neuron_state_counter_uses_runtime_dtype_bytes():
    class HalfStateNode(neuron.BaseNode):
        def __init__(self):
            super().__init__(v_threshold=1.0, v_reset=None, step_mode="s")

        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x

        def single_step_forward(self, x: torch.Tensor):
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            self.v = self.v + 1.0
            return self.v

    x = torch.rand(2, 4, dtype=torch.float16)
    node = HalfStateNode().half()
    counter = op_counter.NeuronStateCounter()

    with op_counter.DispatchCounterMode([counter]):
        _ = node(x)

    metrics = counter.get_metric_counts()["Global"]
    expected_bytes = x.numel() * x.element_size()
    assert metrics["state_reads"] >= expected_bytes
    assert metrics["state_reads"] % x.element_size() == 0
    assert metrics["state_writes"] >= expected_bytes
    assert metrics["state_writes"] % x.element_size() == 0


def test_neuron_state_counter_supports_meta_tensor_storage_keys():
    class MetaFriendlyNode(neuron.BaseNode):
        def __init__(self):
            super().__init__(v_threshold=1.0, v_reset=None, step_mode="s")

        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x

        def single_step_forward(self, x: torch.Tensor):
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            self.v = self.v + 1.0
            return self.v

    node = MetaFriendlyNode()
    x = torch.empty((2, 4), device="meta")
    counter = op_counter.NeuronStateCounter()

    with op_counter.DispatchCounterMode([counter]):
        _ = node(x)

    assert "Global" in counter.get_metric_counts()
    assert counter.get_metric_counts()["Global"]["state_reads"] > 0


def test_neuron_state_counter_projection_does_not_double_count_reset_tags():
    class SpikeGatedAddNode(neuron.BaseNode):
        def __init__(self):
            super().__init__(v_threshold=1.0, v_reset=None, step_mode="s")

        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x

        def single_step_forward(self, x: torch.Tensor):
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            spike = (self.v > 0.5).to(x)
            self.v = self.v + spike
            return self.v

    node = SpikeGatedAddNode()
    counter = op_counter.NeuronStateCounter()

    with op_counter.DispatchCounterMode([counter]):
        _ = node(torch.rand(2, 4))

    metrics = counter.get_metric_counts()["Global"]
    projection = counter.get_projection_counts()["Global"]
    assert metrics["state_adds"] > 0
    assert metrics["state_reset_ops"] > 0
    expected = (
        metrics["state_adds"]
        + metrics["state_comps"]
        + metrics["state_nonlinear_ops"]
        + metrics["state_select_ops"]
    )
    assert projection["state_acc_like"] == expected


def test_neuron_state_counter_non_binary_zero_sparse_gate_reduces_state_bytes():
    class DenseGateNode(neuron.BaseNode):
        def __init__(self):
            super().__init__(v_threshold=1.0, v_reset=None, step_mode="s")

        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x

        def single_step_forward(self, x: torch.Tensor):
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            gate = torch.full_like(x, 0.25)
            self.v = self.v + gate
            return self.v

    class SparseGateNode(neuron.BaseNode):
        def __init__(self):
            super().__init__(v_threshold=1.0, v_reset=None, step_mode="s")

        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x

        def single_step_forward(self, x: torch.Tensor):
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            gate = torch.zeros_like(x)
            gate[:, :2] = 0.25
            self.v = self.v + gate
            return self.v

    x = torch.rand(2, 4)
    dense_counter = op_counter.NeuronStateCounter()
    sparse_counter = op_counter.NeuronStateCounter()

    with op_counter.DispatchCounterMode([dense_counter]):
        _ = DenseGateNode()(x)
    with op_counter.DispatchCounterMode([sparse_counter]):
        _ = SparseGateNode()(x)

    dense_metrics = dense_counter.get_metric_counts()["Global"]
    sparse_metrics = sparse_counter.get_metric_counts()["Global"]
    dense_projection = dense_counter.get_projection_counts()["Global"]
    sparse_projection = sparse_counter.get_projection_counts()["Global"]
    expected_buffer_bytes = x.numel() * x.element_size()
    assert sparse_metrics["state_reads"] < dense_metrics["state_reads"]
    assert sparse_metrics["state_writes"] < dense_metrics["state_writes"]
    assert dense_projection["potential_buffer_bytes"] == expected_buffer_bytes
    assert sparse_projection["potential_buffer_bytes"] == expected_buffer_bytes


def test_neuron_state_counter_clone_counts_state_write():
    class CloneStateNode(neuron.BaseNode):
        def __init__(self):
            super().__init__(v_threshold=1.0, v_reset=None, step_mode="s")

        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x

        def single_step_forward(self, x: torch.Tensor):
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            self.v = self.v.clone()
            return self.v

    x = torch.rand(2, 4)
    counter = op_counter.NeuronStateCounter()

    with op_counter.DispatchCounterMode([counter]):
        _ = CloneStateNode()(x)

    metrics = counter.get_metric_counts()["Global"]
    expected_bytes = x.numel() * x.element_size()
    assert metrics["state_reads"] >= expected_bytes
    assert metrics["state_writes"] >= expected_bytes


def test_neuron_state_counter_copy_does_not_count_state_target_as_read():
    class CopyStateNode(neuron.BaseNode):
        def __init__(self):
            super().__init__(v_threshold=1.0, v_reset=None, step_mode="s")

        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x

        def single_step_forward(self, x: torch.Tensor):
            self.v_float_to_tensor(x)
            self.v.copy_(torch.zeros_like(x))
            return self.v

    x = torch.rand(2, 4)
    counter = op_counter.NeuronStateCounter()

    with op_counter.DispatchCounterMode([counter]):
        _ = CopyStateNode()(x)

    metrics = counter.get_metric_counts()["Global"]
    expected_bytes = x.numel() * x.element_size()
    assert metrics["state_reads"] == 0
    assert metrics["state_writes"] >= expected_bytes
