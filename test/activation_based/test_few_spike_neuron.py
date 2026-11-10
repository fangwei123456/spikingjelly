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


def test_outlier_aware_threshold_node_aligns_outlier_buffer_dtype():
    normal = neuron.FewSpikeTable(
        theta=torch.tensor([0.25, 0.5], dtype=torch.float64),
        h=torch.tensor([0.1, 0.1], dtype=torch.float64),
        d=torch.tensor([1.0, 2.0], dtype=torch.float64),
    )
    outlier = neuron.FewSpikeTable(
        theta=torch.tensor([0.25, 0.5], dtype=torch.float32),
        h=torch.tensor([0.1, 0.1], dtype=torch.float32),
        d=torch.tensor([10.0, 20.0], dtype=torch.float32),
    )

    node = neuron.OutlierAwareThresholdNode(
        table=normal,
        outlier_table=outlier,
        split_threshold=0.5,
    )

    assert node.outlier_theta.dtype == node.theta.dtype == torch.float64
    assert node.outlier_h.dtype == node.h.dtype == torch.float64
    assert node.outlier_d.dtype == node.d.dtype == torch.float64


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
        neuron.HGNode(tables=[table0, table1], gate_thresholds=[float("nan")])
    with pytest.raises(ValueError):
        neuron.HGNode(tables=[table0, table1, table0], gate_thresholds=[1.0, 0.0])


def test_hg_node_does_not_register_unused_parent_table_buffers():
    table0 = neuron.FewSpikeTable(theta=[0.25, 0.5], h=[0.1, 0.1], d=[1.0, 2.0])
    table1 = neuron.FewSpikeTable(theta=[0.25, 0.5], h=[0.1, 0.1], d=[10.0, 20.0])

    node = neuron.HGNode(
        tables=[table0, table1],
        gate_thresholds=[0.0],
    )

    assert isinstance(node, neuron.FewSpikeNode)
    assert node.K == table0.K
    assert "theta" not in node.state_dict()
    assert "h" not in node.state_dict()
    assert "d" not in node.state_dict()
    assert set(node.state_dict().keys()) == {
        "region_theta",
        "region_h",
        "region_d",
        "gate_thresholds",
    }


def _reference_decode(node, x):
    # Frozen e18a95aa equations: keep the oracle independent of fused execution.
    sequence = node.step_mode == "m"
    gate = x.sum(0) if sequence else x
    sign = None
    if isinstance(node, neuron.OutlierAwareThresholdNode):
        if node.clamp_value is not None:
            gate = gate.clamp(-node.clamp_value, node.clamp_value)
        sign = gate.sign().detach()
        gate = gate.abs()
        normal = gate <= node.split_threshold
        tables = [
            [torch.where(normal, a, b) for a, b in zip(first, second)]
            for first, second in (
                (node.theta, node.outlier_theta),
                (node.h, node.outlier_h),
                (node.d, node.outlier_d),
            )
        ]
    elif isinstance(node, neuron.HGNode):
        region = torch.bucketize(gate.detach(), node.gate_thresholds)
        tables = [
            [table[:, k][region] for k in range(node.K)]
            for table in (node.region_theta, node.region_h, node.region_d)
        ]
    else:
        tables = node.theta, node.h, node.d
    v = gate
    readout = torch.zeros_like(gate)
    outputs = []
    for theta, h, d in zip(*tables):
        spike = node.surrogate_function(v - theta)
        value = d * spike
        outputs.append(value)
        readout = readout + value
        v = v - h * spike
    result = torch.stack(outputs) if sequence else readout
    if sign is not None:
        result = result * (sign.unsqueeze(0) if sequence else sign)
    return result


def _routing_node(kind, mode, sg=None):
    table = neuron.FewSpikeTable([0.25, 0.5, 0.75], [0.2, 0.3, 0.4], [1, 2, 4])
    other = neuron.FewSpikeTable([0.1, 0.4, 0.8], [0.1, 0.4, 0.2], [2, 4, 8])
    options = dict(step_mode=mode, surrogate_function=sg)
    if kind == "oat":
        return neuron.OutlierAwareThresholdNode(table, other, 0.5, 1.0, **options)
    if kind == "hg":
        low = neuron.FewSpikeTable([-1.0, -0.5, 0.0], [0.25, 0.25, 0.25], [-1, -2, -4])
        return neuron.HGNode([low, table, other], [-0.5, 0.5], **options)
    return neuron.FewSpikeNode(table, **options)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("kind", ["fs", "oat", "hg"])
@pytest.mark.parametrize("mode", ["s", "m"])
@pytest.mark.parametrize(
    "dtype,layout",
    [
        (torch.float32, "contiguous"),
        (torch.float32, "strided"),
        (torch.float16, "contiguous"),
        (torch.bfloat16, "contiguous"),
        (torch.float64, "empty"),
    ],
)
def test_few_spike_decode_preserves_boundaries_dtype_and_layout(
    device, kind, mode, dtype, layout
):
    node = _routing_node(kind, mode).to(device=device, dtype=dtype)
    edge = torch.tensor(
        [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0], device=device, dtype=dtype
    )
    values = torch.cat(
        (
            edge,
            torch.nextafter(edge, torch.full_like(edge, -torch.inf)),
            torch.nextafter(edge, torch.full_like(edge, torch.inf)),
            edge.new_tensor([torch.nan, torch.inf, -torch.inf]),
        )
    )
    x = values.repeat(2, 1)
    if layout == "strided":
        x = x.t()
    elif layout == "empty":
        x = x[:0]
    if mode == "m":
        x = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)))
    with torch.no_grad():
        expected = _reference_decode(node, x)
        actual = node(x)
        repeated = node(x)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("kind", ["fs", "oat", "hg"])
@pytest.mark.parametrize("mode", ["s", "m"])
@pytest.mark.parametrize("spiking", [True, False])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_few_spike_decode_preserves_surrogate_input_gradients(
    kind, mode, spiking, device
):
    node = _routing_node(kind, mode, surrogate.Sigmoid(spiking=spiking)).to(
        device=device, dtype=torch.float64
    )
    x = (
        torch.linspace(-0.9, 1.2, 30, dtype=torch.float64, device=device)
        .reshape(3, 2, 5)
        .requires_grad_()
    )
    expected_x = x.detach().clone().requires_grad_()
    actual = node(x)
    expected = _reference_decode(node, expected_x)
    gradient = torch.linspace(
        -1, 1, actual.numel(), dtype=actual.dtype, device=device
    ).reshape(actual.shape)
    actual_grad = torch.autograd.grad(actual, x, gradient)[0]
    expected_grad = torch.autograd.grad(expected, expected_x, gradient)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("kind", ["fs", "oat", "hg"])
def test_few_spike_surrogate_hooks_remain_observable(device, kind):
    node = _routing_node(kind, "m").to(device)
    handle = node.surrogate_function.register_forward_hook(
        lambda module, inputs, output: torch.zeros_like(output)
    )
    try:
        with torch.no_grad():
            result = node(torch.ones(3, 5, device=device))
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
    finally:
        handle.remove()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_few_spike_subnormal_thresholds(device):
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    table = neuron.FewSpikeTable(torch.stack((tiny, tiny)), [0.0, 0.0], [1.0, 2.0])
    node = neuron.FewSpikeNode(table).to(device)
    x = torch.tensor([0.0, tiny.item(), -tiny.item()], device=device)
    with torch.no_grad():
        torch.testing.assert_close(
            node(x), x.new_tensor([0.0, 3.0, 0.0]), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("kind", ["fs", "oat", "hg"])
@pytest.mark.parametrize(
    "input_dtype,table_dtype",
    [(torch.float16, torch.float32), (torch.float32, torch.float64)],
)
def test_few_spike_mixed_dtype_promotion(device, kind, input_dtype, table_dtype):
    node = _routing_node(kind, "s").to(device=device, dtype=table_dtype)
    for shape in ((), (2, 3)):
        x = torch.full(shape, 0.625, dtype=input_dtype, device=device)
        with torch.no_grad():
            expected = _reference_decode(node, x)
            actual = node(x)
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["fs", "oat", "hg"])
def test_few_spike_fullgraph_keeps_reference_outputs(kind):
    node = _routing_node(kind, "s").cuda()
    x = torch.linspace(-1, 2, 33, device="cuda")
    with torch.no_grad():
        expected = _reference_decode(node, x)
        actual = torch.compile(node, fullgraph=True)(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["fs", "oat", "hg"])
def test_few_spike_fp32_cuda_gradients_are_preserved(kind):
    node = _routing_node(kind, "m").cuda()
    x = torch.linspace(-0.9, 1.2, 30, device="cuda").reshape(3, 2, 5).requires_grad_()
    expected_x = x.detach().clone().requires_grad_()
    actual = node(x)
    expected = _reference_decode(node, expected_x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0],
        torch.autograd.grad(expected.sum(), expected_x)[0],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_few_spike_no_grad_does_not_silently_discard_forward_ad(device):
    node = _routing_node("fs", "s").to(device)
    x = torch.ones(5, device=device)
    with torch.autograd.forward_ad.dual_level(), torch.no_grad():
        dual = torch.autograd.forward_ad.make_dual(x, torch.ones_like(x))
        with pytest.raises(NotImplementedError, match="jvp"):
            node(dual)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("kind", ["fs", "oat", "hg"])
@pytest.mark.parametrize("mode", ["s", "m"])
@pytest.mark.parametrize("negative_view", ["input", "buffers"])
def test_few_spike_lazy_negative_views(device, kind, mode, negative_view):
    node = _routing_node(kind, mode).to(device)
    x = torch.tensor(1 + 1j, device=device).conj().imag
    assert x.is_neg() and x.is_contiguous() and x.item() == -1.0
    if mode == "m":
        x = torch._neg_view(torch.tensor([1.0, 0.0, 0.0], device=device))
    if negative_view == "buffers":
        x = torch.full((3,) if mode == "m" else (), 0.75, device=device)
        for name, value in list(node.named_buffers()):
            setattr(node, name, torch._neg_view(-value))
        assert all(value.is_neg() and value.is_contiguous() for value in node.buffers())
    with torch.no_grad():
        expected = _reference_decode(node, x)
        torch.testing.assert_close(node(x), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_few_spike_fused_offsets_beyond_int32():
    from spikingjelly._ops import few_spike

    if (
        few_spike._selection.diagnostics(torch.device("cuda"))["implementation"]
        == "torch"
    ):
        pytest.skip("CUDA fusion unavailable")
    if torch.cuda.mem_get_info()[0] < 20 * 1024**3:
        pytest.skip("Large-offset validation requires 20 GiB free CUDA memory")
    k = 32
    n = 2**31 // (k - 1) + 17
    node = neuron.FewSpikeNode(
        neuron.FewSpikeTable([0.5] * k, [0.0] * k, list(range(1, k + 1))),
        step_mode="m",
    ).cuda()
    x = torch.zeros((k, n), device="cuda")
    x[0, -1] = 1.0
    with torch.no_grad():
        result = node(x)
        expected = _reference_decode(node, x[:, -64:])
        assert (k - 1) * n > 2**31
        torch.testing.assert_close(result[:, -64:], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            result[:, :64], torch.zeros_like(expected), rtol=0, atol=0
        )
        torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n", [1, 512])
def test_few_spike_public_forward_executes_fused_kernel(tmp_path, n):
    import json
    from spikingjelly._ops import few_spike

    if (
        few_spike._selection.diagnostics(torch.device("cuda"))["implementation"]
        == "torch"
    ):
        pytest.skip("CUDA fusion unavailable")
    for kind in ("fs", "oat", "hg"):
        for mode in ("s", "m"):
            node = _routing_node(kind, mode).cuda()
            x = torch.ones((3, n) if mode == "m" else (n,), device="cuda")
            with torch.no_grad():
                node(x)
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    actual = node(x)
                    torch.cuda.synchronize()
                torch.testing.assert_close(
                    actual, _reference_decode(node, x), rtol=0, atol=0
                )
            trace = tmp_path / f"{kind}-{mode}.json"
            profile.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())["traceEvents"]
            assert any(
                event.get("cat") == "kernel"
                and any(
                    name in event["name"]
                    for name in ("_few_spike_inference", "few_spike_native_kernel")
                )
                for event in events
            )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("thresholds", [[], [0.0]])
def test_hg_replaced_shorter_gate_thresholds_use_reference(device, thresholds):
    node = _routing_node("hg", "s").to(device)
    node.gate_thresholds = torch.tensor(thresholds, device=device)
    x = torch.tensor([-2.0, 0.0, 2.0], device=device)
    with torch.no_grad():
        torch.testing.assert_close(node(x), _reference_decode(node, x), rtol=0, atol=0)


def test_few_spike_cpu_implementation_query():
    from spikingjelly.activation_based import functional

    for execution in ("eager", "compile"):
        assert (
            functional.neuron_implementation(
                "few_spike", torch.device("cpu"), execution=execution
            )["implementation"]
            == "torch-reference"
        )


@pytest.fixture(params=["cuda", "triton"])
def few_spike_provider(request, monkeypatch):
    import importlib
    from spikingjelly._ops import few_spike
    from spikingjelly.activation_based import functional

    try:
        with monkeypatch.context() as context:
            context.setenv("SJ_FEW_SPIKE_CUDA_IMPLEMENTATION", request.param)
            importlib.reload(few_spike)
            try:
                functional.neuron_implementation("few_spike", torch.device("cuda"))
            except RuntimeError as error:
                if not str(error).startswith("No available sj_few_spike"):
                    raise
                pytest.skip(str(error))
            yield request.param
    finally:
        importlib.reload(few_spike)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["fs", "oat", "hg", "oat_unclamped", "hg_single"])
@pytest.mark.parametrize("mode", ["s", "m"])
def test_few_spike_explicit_provider(kind, mode, few_spike_provider, tmp_path):
    import json
    from spikingjelly.activation_based import functional

    node = _routing_node(kind, mode).cuda()
    if kind == "oat_unclamped":
        node = _routing_node("oat", mode).cuda()
        node.clamp_value = None
    elif kind == "hg_single":
        table = neuron.FewSpikeTable([0.25, 0.5, 0.75], [0.2, 0.3, 0.4], [1, 2, 4])
        node = neuron.HGNode([table], [], step_mode=mode).cuda()
    x = torch.tensor(
        [-float("inf"), -1.0, 0.0, 0.25, 0.5, 0.75, 1.0, float("inf"), float("nan")],
        device="cuda",
    )
    if mode == "m":
        x = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)))
    with torch.no_grad():
        expected = _reference_decode(node, x)
        node(x)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            actual = node(x)
            torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert (
        functional.neuron_implementation("few_spike", x.device)["implementation"]
        == few_spike_provider
    )
    trace = tmp_path / "provider.json"
    profile.export_chrome_trace(str(trace))
    kernels = [
        e["name"]
        for e in json.loads(trace.read_text())["traceEvents"]
        if e.get("cat") == "kernel"
    ]
    expected_kernel = (
        "few_spike_native_kernel"
        if few_spike_provider == "cuda"
        else "_few_spike_inference"
    )
    other = (
        "_few_spike_inference"
        if few_spike_provider == "cuda"
        else "few_spike_native_kernel"
    )
    assert any(expected_kernel in name for name in kernels)
    assert not any(other in name for name in kernels)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_few_spike_explicit_provider_rejects_unsupported_profile(few_spike_provider):
    from spikingjelly.activation_based import functional

    node = _routing_node("fs", "s").cuda()
    with (
        torch.no_grad(),
        pytest.raises(RuntimeError, match="cannot run this execution profile"),
    ):
        node(torch.ones(2, device="cuda", dtype=torch.float64))
    with pytest.raises(RuntimeError, match="cannot run this execution profile"):
        node(torch.ones(2, device="cuda", requires_grad=True))
    with pytest.raises(RuntimeError, match="cannot run this execution profile"):
        functional.neuron_implementation(
            "few_spike", torch.device("cuda"), execution="compile"
        )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_few_spike_no_grad_accepts_requires_grad_input(device):
    node = _routing_node("fs", "s").to(device)
    x = torch.ones(3, device=device, requires_grad=True)
    with torch.no_grad():
        actual = node(x)
        torch.testing.assert_close(actual, _reference_decode(node, x), rtol=0, atol=0)
    assert not actual.requires_grad


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_few_spike_oversized_clamp_preserves_reference_error(device):
    node = _routing_node("oat", "s").to(device)
    node.clamp_value = 1e100
    with torch.no_grad(), pytest.raises(RuntimeError, match="overflow"):
        node(torch.ones(3, device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["cuda", "triton"])
def test_few_spike_explicit_unavailable_provider_errors(provider, monkeypatch):
    import importlib
    from spikingjelly._ops import few_spike
    from spikingjelly.activation_based import functional

    try:
        with monkeypatch.context() as context:
            context.setenv("SJ_FEW_SPIKE_CUDA_IMPLEMENTATION", provider)
            importlib.reload(few_spike)
            node = _routing_node("fs", "s").cuda()
            x = torch.ones(2, device="cuda")
            try:
                info = functional.neuron_implementation("few_spike", x.device)
            except RuntimeError as error:
                if not str(error).startswith("No available sj_few_spike"):
                    raise
                with (
                    torch.no_grad(),
                    pytest.raises(RuntimeError, match="No available sj_few_spike"),
                ):
                    node(x)
            else:
                assert info["implementation"] == provider
                with torch.no_grad():
                    torch.testing.assert_close(
                        node(x), _reference_decode(node, x), rtol=0, atol=0
                    )
    finally:
        importlib.reload(few_spike)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_few_spike_provider_subnormal(few_spike_provider):
    tiny = torch.finfo(torch.float32).tiny / 2
    node = neuron.FewSpikeNode(neuron.FewSpikeTable([tiny], [0.0], [1.0])).cuda()
    x = torch.tensor([0.0, tiny, 2 * tiny], device="cuda", dtype=torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(
            node(x), x.new_tensor([0.0, 1.0, 1.0]), rtol=0, atol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_few_spike_provider_current_stream(few_spike_provider):
    node = _routing_node("oat", "m").cuda()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.no_grad():
        x = torch.full((3, 1024), 0.25, device="cuda")
        actual = node(x)
        expected = _reference_decode(node, x)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("failure", [AttributeError, RuntimeError])
def test_few_spike_incompatible_triton_import_is_optional(monkeypatch, failure):
    import builtins
    import runpy
    from pathlib import Path
    from spikingjelly._ops import few_spike

    original_import = builtins.__import__

    def incompatible(name, *args, **kwargs):
        if name == "triton":
            raise failure("incompatible Triton initialization")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", incompatible)
    with pytest.raises(
        ImportError, match="incompatible Triton initialization"
    ) as error:
        runpy.run_path(str(Path(few_spike.__file__).with_name("triton.py")))
    assert isinstance(error.value.__cause__, failure)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["fs", "oat", "hg"])
@pytest.mark.parametrize("mode", ["s", "m"])
def test_few_spike_trace_uses_reference(kind, mode, few_spike_provider, monkeypatch):
    from spikingjelly._ops import few_spike

    node = _routing_node(kind, mode).cuda()
    x = torch.linspace(-1.5, 1.5, 33, device="cuda")
    if mode == "m":
        x = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)))
    with torch.no_grad():
        # The fixture warms a real provider; auto must bypass it while tracing.
        monkeypatch.setattr(few_spike._selection, "_requested", "auto")
        traced = torch.jit.trace(node, x)
        for value in (x, x.flip(-1) * 0.75):
            torch.testing.assert_close(
                traced(value), _reference_decode(node, value), rtol=0, atol=0
            )
        monkeypatch.setattr(few_spike._selection, "_requested", few_spike_provider)
        with pytest.raises(RuntimeError, match="tracing"):
            torch.jit.trace(node, x)
