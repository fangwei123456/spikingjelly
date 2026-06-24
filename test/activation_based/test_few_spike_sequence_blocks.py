import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.ann2snn.operators import (
    SNNElementWiseProduct,
    TDLinear,
)


def _table(scale=1.0):
    return neuron.FewSpikeTable(
        theta=torch.tensor([0.25, 0.5, 0.75]) * scale,
        h=torch.tensor([0.2, 0.3, 0.4]) * scale,
        d=torch.tensor([1.0, 2.0, 4.0]) * scale,
    )


def _two_step_table(scale=1.0):
    return neuron.FewSpikeTable(
        theta=torch.tensor([0.25, 0.5]) * scale,
        h=torch.tensor([0.1, 0.1]) * scale,
        d=torch.tensor([1.0, 2.0]) * scale,
    )


class _FewSpikeMLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, activation):
        super().__init__()
        self.fc1 = TDLinear(in_features, hidden_features, bias=False)
        self.activation = activation
        self.fc2 = TDLinear(hidden_features, out_features, bias=False)

    def forward(self, x_seq):
        return self.fc2(self.activation(self.fc1(x_seq)))


class _FewSpikeGatedMLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, activation, gate):
        super().__init__()
        self.up_proj = TDLinear(in_features, hidden_features, bias=False)
        self.gate_proj = TDLinear(in_features, hidden_features, bias=False)
        self.activation = activation
        self.gate = gate
        self.product = SNNElementWiseProduct()
        self.down_proj = TDLinear(hidden_features, out_features, bias=False)

    def forward(self, x_seq):
        up_seq = self.activation(self.up_proj(x_seq))
        gate_seq = self.gate(self.gate_proj(x_seq))
        return self.down_proj(self.product(up_seq, gate_seq))


def _single_step_mlp_reference(block, x_seq):
    x = x_seq.sum(dim=0)
    hidden = F.linear(x, block.fc1.weight, None)
    hidden = block.activation.single_step_forward(hidden)
    return F.linear(hidden, block.fc2.weight, None)


def _single_step_gated_reference(block, x_seq):
    x = x_seq.sum(dim=0)
    up = F.linear(x, block.up_proj.weight, None)
    gate = F.linear(x, block.gate_proj.weight, None)
    up = block.activation.single_step_forward(up)
    gate = block.gate.single_step_forward(gate)
    return F.linear(up * gate, block.down_proj.weight, None)


def test_few_spike_mlp_block_final_sum_matches_single_step_reference():
    table = _table()
    activation = neuron.FewSpikeNode(
        table=table,
        surrogate_function=surrogate.DeterministicPass(),
        step_mode="m",
    )
    block = _FewSpikeMLPBlock(4, 6, 3, activation)
    x_seq = torch.randn(table.K, 2, 4)

    y_seq = block(x_seq)
    expected = _single_step_mlp_reference(block, x_seq)

    assert y_seq.shape == (table.K, 2, 3)
    assert torch.allclose(y_seq.sum(dim=0), expected)


def test_few_spike_mlp_block_supports_higher_rank_input():
    table = _table()
    activation = neuron.FewSpikeNode(
        table=table,
        surrogate_function=surrogate.DeterministicPass(),
        step_mode="m",
    )
    block = _FewSpikeMLPBlock(5, 7, 2, activation)
    x_seq = torch.randn(table.K, 2, 3, 5)

    y_seq = block(x_seq)
    expected = _single_step_mlp_reference(block, x_seq)

    assert y_seq.shape == (table.K, 2, 3, 2)
    assert torch.allclose(y_seq.sum(dim=0), expected)


def test_few_spike_mlp_block_rejects_wrong_time_length():
    table = _table()
    activation = neuron.FewSpikeNode(
        table=table,
        surrogate_function=surrogate.DeterministicPass(),
        step_mode="m",
    )
    block = _FewSpikeMLPBlock(4, 6, 3, activation)

    with pytest.raises(ValueError):
        block(torch.randn(table.K - 1, 2, 4))


def test_few_spike_gated_block_final_sum_matches_single_step_reference():
    table = _table()
    block = _FewSpikeGatedMLPBlock(
        in_features=4,
        hidden_features=6,
        out_features=3,
        activation=neuron.FewSpikeNode(
            table=table,
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
        ),
        gate=neuron.FewSpikeNode(
            table=table,
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
        ),
    )
    x_seq = torch.randn(table.K, 2, 4)

    up_seq = block.activation(block.up_proj(x_seq))
    gate_seq = block.gate(block.gate_proj(x_seq))
    product_seq = block.product(up_seq, gate_seq)
    y_seq = block(x_seq)
    expected = _single_step_gated_reference(block, x_seq)

    assert up_seq.shape == (table.K, 2, 6)
    assert gate_seq.shape == (table.K, 2, 6)
    assert product_seq.shape == (table.K, 2, 6)
    assert y_seq.shape == (table.K, 2, 3)
    assert torch.allclose(y_seq.sum(dim=0), expected)


def test_oat_activation_block_matches_single_step_reference():
    normal = _two_step_table()
    outlier = _two_step_table(scale=10.0)
    activation = neuron.OutlierAwareThresholdNode(
        table=normal,
        outlier_table=outlier,
        split_threshold=1.0,
        clamp_value=2.0,
        surrogate_function=surrogate.DeterministicPass(),
        step_mode="m",
    )
    block = _FewSpikeMLPBlock(3, 4, 2, activation)
    with torch.no_grad():
        block.fc1.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
    x_seq = torch.tensor(
        [
            [[1.5, 0.2, -0.1], [-1.2, 0.3, 0.4]],
            [[1.0, 0.5, 0.6], [-0.7, 0.4, 0.5]],
        ]
    )

    y_seq = block(x_seq)
    expected = _single_step_mlp_reference(block, x_seq)

    assert y_seq.shape == (normal.K, 2, 2)
    assert torch.allclose(y_seq.sum(dim=0), expected)


def test_hg_gate_block_matches_single_step_reference():
    table_low = _two_step_table()
    table_mid = _two_step_table(scale=10.0)
    table_high = _two_step_table(scale=100.0)
    table_gate = _two_step_table(scale=2.0)
    block = _FewSpikeGatedMLPBlock(
        in_features=3,
        hidden_features=4,
        out_features=2,
        activation=neuron.FewSpikeNode(
            table=table_gate,
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
        ),
        gate=neuron.HGNode(
            tables=[table_low, table_mid, table_high],
            gate_thresholds=[0.0, 1.0],
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
        ),
    )
    with torch.no_grad():
        block.gate_proj.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
    x_seq = torch.tensor(
        [
            [[-0.4, 0.3, 0.7], [0.2, 0.4, 0.5]],
            [[-0.3, 0.5, 0.6], [0.6, 0.8, 0.7]],
        ]
    )

    y_seq = block(x_seq)
    expected = _single_step_gated_reference(block, x_seq)

    assert y_seq.shape == (table_low.K, 2, 2)
    assert torch.allclose(y_seq.sum(dim=0), expected)


def test_gated_block_autograd_and_stateless_forward():
    table = _table()
    block = _FewSpikeGatedMLPBlock(
        in_features=4,
        hidden_features=5,
        out_features=3,
        activation=neuron.FewSpikeNode(
            table=table,
            surrogate_function=surrogate.Sigmoid(),
            step_mode="m",
        ),
        gate=neuron.FewSpikeNode(
            table=table,
            surrogate_function=surrogate.Sigmoid(),
            step_mode="m",
        ),
    )
    x_seq = torch.randn(table.K, 2, 4, requires_grad=True)

    y0 = block(x_seq)
    y1 = block(x_seq)
    y0.square().sum().backward()

    assert torch.allclose(y0, y1)
    assert x_seq.grad is not None
    assert torch.isfinite(x_seq.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_block_rejects_mixed_table_lengths():
    table = _two_step_table()
    other = _table()

    with pytest.raises(ValueError):
        neuron.OutlierAwareThresholdNode(
            table=table,
            outlier_table=other,
            split_threshold=1.0,
            step_mode="m",
        )
    with pytest.raises(ValueError):
        neuron.HGNode(
            tables=[table, other],
            gate_thresholds=[0.0],
            step_mode="m",
        )
