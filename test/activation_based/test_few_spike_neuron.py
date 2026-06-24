import pytest
import torch

from spikingjelly.activation_based import functional, neuron, surrogate


def _table(scale=1.0):
    return neuron.FewSpikeTable(
        theta=torch.tensor([0.25, 0.5, 0.75]) * scale,
        h=torch.tensor([0.2, 0.3, 0.4]) * scale,
        d=torch.tensor([1.0, 2.0, 4.0]) * scale,
    )


def _manual_fs(gate, table, surrogate_function, return_sequence):
    theta = table.theta.to(device=gate.device, dtype=gate.dtype)
    h = table.h.to(device=gate.device, dtype=gate.dtype)
    d = table.d.to(device=gate.device, dtype=gate.dtype)
    v = gate
    y = torch.zeros_like(gate)
    y_seq = []
    for k in range(table.K):
        z = surrogate_function(v - theta[k])
        weighted_spike = d[k] * z
        if return_sequence:
            y_seq.append(weighted_spike)
        else:
            y = y + weighted_spike
        v = v - h[k] * z
    if return_sequence:
        return torch.stack(y_seq)
    return y


def test_few_spike_table_validation():
    table = _table()

    assert table.K == 3
    assert table.theta.dtype == torch.float32

    with pytest.raises(ValueError):
        neuron.FewSpikeTable(theta=[], h=[], d=[])
    with pytest.raises(ValueError):
        neuron.FewSpikeTable(theta=[1.0, 2.0], h=[1.0], d=[1.0, 2.0])
    with pytest.raises(ValueError):
        neuron.FewSpikeTable(theta=[[1.0]], h=[[1.0]], d=[[1.0]])
    with pytest.raises(ValueError):
        neuron.FewSpikeTable(theta=[float("inf")], h=[1.0], d=[1.0])
    with pytest.raises(TypeError):
        neuron.FewSpikeTable(theta=torch.tensor([1.0 + 1.0j]), h=[1.0], d=[1.0])


def test_few_spike_node_single_step_matches_reference():
    table = _table()
    sg = surrogate.DeterministicPass()
    node = neuron.FewSpikeNode(
        table=table,
        surrogate_function=sg,
        step_mode="s",
    )
    x = torch.tensor([[0.1, 0.6], [1.2, 2.0]])

    y = node(x)
    expected = _manual_fs(x, table, sg, return_sequence=False)

    assert torch.allclose(y, expected)


def test_few_spike_node_multi_step_shape_and_sum_match_single_step():
    table = _table()
    sg = surrogate.DeterministicPass()
    node = neuron.FewSpikeNode(
        table=table,
        surrogate_function=sg,
        step_mode="m",
    )
    x_seq = torch.tensor(
        [
            [[0.1, 0.3], [0.2, 0.5]],
            [[0.2, 0.1], [0.4, 0.3]],
            [[0.3, 0.4], [0.1, 0.2]],
        ]
    )

    y_seq = node(x_seq)
    expected_seq = _manual_fs(x_seq.sum(0), table, sg, return_sequence=True)
    single = node.single_step_forward(x_seq.sum(0))

    assert y_seq.shape == x_seq.shape
    assert torch.allclose(y_seq, expected_seq)
    assert torch.allclose(y_seq.sum(0), single)


def test_few_spike_node_rejects_wrong_multi_step_length_and_non_float_input():
    node = neuron.FewSpikeNode(table=_table(), step_mode="m")

    with pytest.raises(ValueError):
        node(torch.ones(2, 4))
    with pytest.raises(TypeError):
        node(torch.ones(3, 4, dtype=torch.int64))


def test_few_spike_node_is_step_module_and_stateless():
    node = neuron.FewSpikeNode(
        table=_table(),
        surrogate_function=surrogate.DeterministicPass(),
    )
    x = torch.tensor([[0.5, 1.0]])

    functional.set_step_mode(node, "s")
    y0 = node(x)
    y1 = node(x)

    assert torch.allclose(y0, y1)
    assert not hasattr(node, "reset")

    functional.set_step_mode(node, "m")
    x_seq = torch.stack([x, x, x])
    assert node(x_seq).shape == x_seq.shape


def test_few_spike_node_autograd_and_dtype_follow_module_buffers():
    node = neuron.FewSpikeNode(
        table=_table(),
        surrogate_function=surrogate.Sigmoid(),
    ).to(dtype=torch.float64)
    x = torch.tensor([[0.9, 1.8]], dtype=torch.float64, requires_grad=True)

    y = node(x)
    y.sum().backward()

    assert y.dtype == torch.float64
    assert node.theta.dtype == torch.float64
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_few_spike_nodes_do_not_share_default_surrogate_module():
    table = _table()
    node0 = neuron.FewSpikeNode(table=table)
    node1 = neuron.FewSpikeNode(table=table)
    oat = neuron.OutlierAwareThresholdNode(
        table=neuron.FewSpikeTable(theta=[0.25], h=[0.1], d=[1.0]),
        outlier_table=neuron.FewSpikeTable(theta=[0.25], h=[0.1], d=[2.0]),
        split_threshold=1.0,
    )
    hg = neuron.HGNode(
        tables=[neuron.FewSpikeTable(theta=[0.25], h=[0.1], d=[1.0])],
        gate_thresholds=[],
    )

    assert node0.surrogate_function is not node1.surrogate_function
    assert node0.surrogate_function is not oat.surrogate_function
    assert node0.surrogate_function is not hg.surrogate_function


def test_outlier_aware_threshold_node_matches_reference():
    normal = neuron.FewSpikeTable(theta=[0.25, 0.5], h=[0.1, 0.1], d=[1.0, 2.0])
    outlier = neuron.FewSpikeTable(theta=[0.25, 0.5], h=[0.1, 0.1], d=[10.0, 20.0])
    sg = surrogate.DeterministicPass()
    node = neuron.OutlierAwareThresholdNode(
        table=normal,
        outlier_table=outlier,
        split_threshold=1.0,
        clamp_value=2.0,
        surrogate_function=sg,
        step_mode="s",
    )
    x = torch.tensor([[-2.5, -0.8, 0.0, 0.9, 1.5]])

    y = node(x)
    gate = x.clamp(min=-2.0, max=2.0)
    signs = torch.sign(gate).detach()
    magnitude = gate.abs()
    expected = (
        torch.where(
            magnitude <= 1.0,
            _manual_fs(magnitude, normal, sg, return_sequence=False),
            _manual_fs(magnitude, outlier, sg, return_sequence=False),
        )
        * signs
    )

    assert torch.allclose(y, expected)


def test_outlier_aware_threshold_node_multi_step_and_validation():
    normal = neuron.FewSpikeTable(theta=[0.25, 0.5], h=[0.1, 0.1], d=[1.0, 2.0])
    outlier = neuron.FewSpikeTable(theta=[0.25, 0.5], h=[0.1, 0.1], d=[10.0, 20.0])
    node = neuron.OutlierAwareThresholdNode(
        table=normal,
        outlier_table=outlier,
        split_threshold=0.5,
        surrogate_function=surrogate.DeterministicPass(),
        step_mode="m",
    )

    x_seq = torch.tensor([[[-0.2, 0.3]], [[-0.4, 0.5]]])

    assert node(x_seq).shape == x_seq.shape
    with pytest.raises(ValueError):
        neuron.OutlierAwareThresholdNode(
            table=normal,
            outlier_table=_table(),
            split_threshold=0.5,
        )
    with pytest.raises(TypeError, match="table must be FewSpikeTable"):
        neuron.OutlierAwareThresholdNode(
            table=torch.tensor([1.0]),
            outlier_table=outlier,
            split_threshold=0.5,
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        neuron.OutlierAwareThresholdNode(
            table=normal,
            outlier_table=outlier,
            split_threshold=float("nan"),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        neuron.OutlierAwareThresholdNode(
            table=normal,
            outlier_table=outlier,
            split_threshold=0.5,
            clamp_value=float("inf"),
        )
    with pytest.raises(ValueError):
        neuron.OutlierAwareThresholdNode(
            table=normal,
            outlier_table=outlier,
            split_threshold=2.0,
            clamp_value=1.0,
        )


def test_hg_node_matches_region_reference():
    table_low = neuron.FewSpikeTable(theta=[0.25, 0.5], h=[0.1, 0.1], d=[1.0, 2.0])
    table_mid = neuron.FewSpikeTable(theta=[0.25, 0.5], h=[0.1, 0.1], d=[10.0, 20.0])
    table_high = neuron.FewSpikeTable(theta=[0.25, 0.5], h=[0.1, 0.1], d=[100.0, 200.0])
    sg = surrogate.DeterministicPass()
    node = neuron.HGNode(
        tables=[table_low, table_mid, table_high],
        gate_thresholds=[0.0, 1.0],
        surrogate_function=sg,
        step_mode="s",
    )
    x = torch.tensor([[-0.5, 0.5, 1.5]])

    y = node(x)
    expected = torch.where(
        x <= 0.0,
        _manual_fs(x, table_low, sg, return_sequence=False),
        torch.where(
            x <= 1.0,
            _manual_fs(x, table_mid, sg, return_sequence=False),
            _manual_fs(x, table_high, sg, return_sequence=False),
        ),
    )

    assert torch.allclose(y, expected)


def test_hg_node_multi_step_and_validation():
    table0 = neuron.FewSpikeTable(theta=[0.25, 0.5], h=[0.1, 0.1], d=[1.0, 2.0])
    table1 = neuron.FewSpikeTable(theta=[0.25, 0.5], h=[0.1, 0.1], d=[10.0, 20.0])
    node = neuron.HGNode(
        tables=[table0, table1],
        gate_thresholds=[0.0],
        surrogate_function=surrogate.DeterministicPass(),
        step_mode="m",
    )
    x_seq = torch.tensor([[[-0.2, 0.2]], [[-0.1, 0.3]]])

    assert node(x_seq).shape == x_seq.shape
    single_table = neuron.HGNode(
        tables=[table0],
        gate_thresholds=[],
        surrogate_function=surrogate.DeterministicPass(),
    )
    x = torch.tensor([[0.2, 0.8]])
    assert torch.allclose(
        single_table(x),
        _manual_fs(x, table0, surrogate.DeterministicPass(), False),
    )
    with pytest.raises(ValueError):
        neuron.HGNode(tables=[], gate_thresholds=[])
    with pytest.raises(ValueError):
        neuron.HGNode(tables=[table0, _table()], gate_thresholds=[0.0])
    with pytest.raises(ValueError):
        neuron.HGNode(tables=[table0, table1], gate_thresholds=[1.0, 2.0])
    with pytest.raises(ValueError):
        neuron.HGNode(tables=[table0, table1, table0], gate_thresholds=[1.0, 0.0])
