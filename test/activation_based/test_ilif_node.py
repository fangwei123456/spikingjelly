import pytest
import torch

from spikingjelly.activation_based import neuron, surrogate


def _assert_close(actual: torch.Tensor, expected: torch.Tensor):
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def _safe_ilif_sequence(tau: float, dtype: torch.dtype):
    shape = (12, 4, 16)
    device = torch.device("cuda")
    v_threshold = torch.tensor(0.5, device=device, dtype=dtype)
    decay = torch.tensor(1.0 - 1.0 / tau, device=device, dtype=dtype)
    levels = torch.tensor(
        [-0.25, 0.25, 1.25, 2.25, 3.25, 4.25],
        device=device,
        dtype=dtype,
    )
    level_indices = torch.arange(
        torch.Size(shape).numel(), device=device, dtype=torch.int64
    ).reshape(shape)
    h_seq = levels[level_indices % levels.numel()] * v_threshold
    h_seq[0, 0, 0] = 3.25 * v_threshold

    v_init = torch.linspace(
        -0.2,
        0.2,
        shape[1] * shape[2],
        device=device,
        dtype=dtype,
    ).reshape(shape[1:])
    v = v_init.clone()
    x_seq = torch.empty(shape, device=device, dtype=dtype)
    for t in range(shape[0]):
        x_seq[t] = h_seq[t] - decay * v
        spike = torch.round(torch.clamp(h_seq[t] / v_threshold, 0, 4))
        v = h_seq[t] - spike * v_threshold
    return x_seq, v_init


def test_ilif_rejects_incompatible_surrogate():
    with pytest.raises(ValueError):
        neuron.ILIFNode(surrogate_function=surrogate.Sigmoid())
    with pytest.raises(ValueError):
        neuron.ILIFNode(
            surrogate_function=surrogate.MultiLevelSpikeCount(4, spiking=False)
        )


def test_ilif_reuses_lif_dynamics_and_limits_triton_to_multi_step():
    node = neuron.ILIFNode(step_mode="s")

    assert isinstance(node, neuron.LIFNode)
    assert node.tau == pytest.approx(4.0 / 3.0)
    assert node.decay_input is False
    assert node.v_reset is None
    assert node.supported_backends == ("torch",)
    assert neuron.ILIFNode(step_mode="m").supported_backends == ("torch", "triton")
    with pytest.raises(NotImplementedError):
        neuron.ILIFNode(step_mode="s", backend="triton")


def test_ilif_training_outputs_integer_counts_and_updates_voltage():
    node = neuron.ILIFNode(v_threshold=1.0, tau=4.0 / 3.0)
    x = torch.tensor([[3.2, 0.4, 5.8]])

    y = node(x)

    expected = torch.tensor([[3.0, 0.0, 4.0]])
    assert torch.equal(y, expected)
    torch.testing.assert_close(node.v, x - expected)


def test_ilif_uses_spike_count_surrogate():
    node = neuron.ILIFNode()

    assert isinstance(node.surrogate_function, surrogate.MultiLevelSpikeCount)
    assert node.surrogate_function.max_spike_count == 4


def test_ilif_tau_is_applied_during_charge():
    node = neuron.ILIFNode(v_threshold=1.0, tau=4.0 / 3.0)

    node(torch.tensor([[3.2]]))
    y = node(torch.tensor([[0.0]]))

    assert torch.equal(y, torch.tensor([[0.0]]))
    torch.testing.assert_close(node.v, torch.tensor([[0.05]]))


def test_ilif_eval_returns_integer_counts():
    node = neuron.ILIFNode(v_threshold=1.0, tau=4.0 / 3.0).eval()

    y = node(torch.tensor([[3.2, 5.8]]))

    assert torch.equal(y, torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(node.v, torch.tensor([[0.2, 1.8]]))


def test_ilif_train_and_eval_have_identical_forward_semantics():
    train_node = neuron.ILIFNode(
        v_threshold=0.5,
        tau=4.0 / 3.0,
        step_mode="m",
        store_v_seq=True,
    )
    eval_node = neuron.ILIFNode(
        v_threshold=0.5,
        tau=4.0 / 3.0,
        step_mode="m",
        store_v_seq=True,
    ).eval()
    x_seq = torch.tensor([[[1.6, -0.2]], [[0.0, 2.2]], [[0.4, 0.0]]])
    initial_v = torch.tensor([[0.2, -0.1]])
    train_node.v = initial_v.clone()
    eval_node.v = initial_v.clone()

    train_y = train_node(x_seq)
    eval_y = eval_node(x_seq)

    assert torch.equal(eval_y, train_y)
    torch.testing.assert_close(eval_node.v, train_node.v)
    torch.testing.assert_close(eval_node.v_seq, train_node.v_seq)
    assert eval_y.max().item() > 1.0


def test_ilif_multistep_eval_preserves_sequence_length():
    node = neuron.ILIFNode(
        v_threshold=1.0,
        tau=4.0 / 3.0,
        step_mode="m",
        store_v_seq=True,
    ).eval()
    x_seq = torch.tensor([[[3.2]], [[0.0]], [[4.8]], [[0.0]]])

    y = node(x_seq)

    assert y.shape == x_seq.shape
    assert node.v_seq.shape == x_seq.shape
    assert torch.equal(y.flatten(), torch.tensor([3.0, 0.0, 4.0, 0.0]))


def test_ilif_training_straight_through_gradient_is_windowed():
    node = neuron.ILIFNode(v_threshold=1.0)
    x = torch.tensor([[-0.5, 0.5, 4.5]], requires_grad=True)

    y = node(x)
    y.sum().backward()

    assert torch.equal(y.detach(), torch.tensor([[0.0, 0.0, 4.0]]))
    torch.testing.assert_close(x.grad, torch.tensor([[0.0, 1.0, 0.0]]))


def test_ilif_supports_custom_gradient_window():
    node = neuron.ILIFNode(
        surrogate_function=surrogate.MultiLevelSpikeCount(
            4,
            grad_window=(-0.5, 4.5),
        ),
    )
    x = torch.tensor([[-0.25, 4.25, 4.75]], requires_grad=True)

    y = node(x)
    y.sum().backward()

    assert torch.equal(y.detach(), torch.tensor([[0.0, 4.0, 4.0]]))
    torch.testing.assert_close(x.grad, torch.tensor([[1.0, 1.0, 0.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("store_v_seq", [False, True])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("tau", [4.0 / 3.0, 2.0, 4.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ilif_triton_matches_torch_training(store_v_seq, detach_reset, tau, dtype):
    pytest.importorskip("triton")
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 requires compute capability >= 8.")
    torch_node = neuron.ILIFNode(
        v_threshold=0.5,
        tau=tau,
        detach_reset=detach_reset,
        step_mode="m",
        backend="torch",
        store_v_seq=store_v_seq,
    ).to(device="cuda", dtype=dtype)
    triton_node = neuron.ILIFNode(
        v_threshold=0.5,
        tau=tau,
        detach_reset=detach_reset,
        step_mode="m",
        backend="triton",
        store_v_seq=store_v_seq,
    ).to(device="cuda", dtype=dtype)
    x, v = _safe_ilif_sequence(tau, dtype)
    x_torch = x.clone().requires_grad_()
    x_triton = x.clone().requires_grad_()
    v_torch = v.clone().requires_grad_()
    v_triton = v_torch.detach().clone().requires_grad_()
    torch_node.v = v_torch
    triton_node.v = v_triton
    weight = torch.linspace(
        -0.5,
        0.5,
        x.numel(),
        device=x.device,
        dtype=dtype,
    ).reshape_as(x)

    y_torch = torch_node(x_torch)
    y_triton = triton_node(x_triton)
    assert torch.equal(y_triton, y_torch)
    _assert_close(triton_node.v, torch_node.v)
    loss_torch = (y_torch * weight).sum() + 0.1 * torch_node.v.sum()
    loss_triton = (y_triton * weight).sum() + 0.1 * triton_node.v.sum()
    if store_v_seq:
        _assert_close(triton_node.v_seq, torch_node.v_seq)
        loss_torch = loss_torch + 0.01 * torch_node.v_seq.sum()
        loss_triton = loss_triton + 0.01 * triton_node.v_seq.sum()

    loss_torch.backward()
    loss_triton.backward()

    _assert_close(x_triton.grad, x_torch.grad)
    _assert_close(v_triton.grad, v_torch.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("store_v_seq", [False, True])
@pytest.mark.parametrize("tau", [4.0 / 3.0, 2.0, 4.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ilif_triton_matches_torch_eval(store_v_seq, tau, dtype):
    pytest.importorskip("triton")
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 requires compute capability >= 8.")
    torch_node = (
        neuron.ILIFNode(
            v_threshold=0.5,
            tau=tau,
            step_mode="m",
            backend="torch",
            store_v_seq=store_v_seq,
        )
        .to(device="cuda", dtype=dtype)
        .eval()
    )
    triton_node = (
        neuron.ILIFNode(
            v_threshold=0.5,
            tau=tau,
            step_mode="m",
            backend="triton",
            store_v_seq=store_v_seq,
        )
        .to(device="cuda", dtype=dtype)
        .eval()
    )
    x, v = _safe_ilif_sequence(tau, dtype)
    torch_node.v = v.clone()
    triton_node.v = v.clone()

    with torch.inference_mode():
        y_torch = torch_node(x)
        y_triton = triton_node(x)

    assert torch.equal(y_triton, y_torch)
    assert y_triton[0, 0, 0].item() == 3.0
    _assert_close(triton_node.v, torch_node.v)
    if store_v_seq:
        _assert_close(triton_node.v_seq, torch_node.v_seq)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ilif_triton_rounds_half_to_even():
    pytest.importorskip("triton")
    node = neuron.ILIFNode(
        step_mode="m",
        backend="triton",
    ).cuda()
    x_seq = torch.tensor(
        [[[0.5, 1.5, 2.5, 3.5, 4.5]]],
        device="cuda",
    )

    y = node(x_seq)

    assert torch.equal(
        y,
        torch.tensor([[[0.0, 2.0, 2.0, 4.0, 4.0]]], device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ilif_triton_matches_custom_gradient_window():
    pytest.importorskip("triton")
    torch_node = neuron.ILIFNode(
        tau=4.0 / 3.0,
        surrogate_function=surrogate.MultiLevelSpikeCount(
            4,
            grad_window=(-0.5, 4.5),
        ),
        step_mode="m",
        backend="torch",
    ).cuda()
    triton_node = neuron.ILIFNode(
        tau=4.0 / 3.0,
        surrogate_function=surrogate.MultiLevelSpikeCount(
            4,
            grad_window=(-0.5, 4.5),
        ),
        step_mode="m",
        backend="triton",
    ).cuda()
    x = torch.tensor(
        [[[-0.25, 4.25, 4.75]], [[0.0, 0.0, 0.0]]],
        device="cuda",
    )
    x_torch = x.clone().requires_grad_()
    x_triton = x.clone().requires_grad_()

    torch_node(x_torch).sum().backward()
    triton_node(x_triton).sum().backward()

    torch.testing.assert_close(x_triton.grad, x_torch.grad)


def test_multi_level_spike_count_supports_custom_gradient_window():
    spike_count = surrogate.MultiLevelSpikeCount(
        max_spike_count=4,
        grad_window=(-0.5, 4.5),
    )
    x = torch.tensor([[-0.25, 4.25, 4.75]], requires_grad=True)

    y = spike_count(x)
    y.sum().backward()

    assert torch.equal(y.detach(), torch.tensor([[0.0, 4.0, 4.0]]))
    torch.testing.assert_close(x.grad, torch.tensor([[1.0, 1.0, 0.0]]))


def test_multi_level_spike_count_rejects_invalid_parameters():
    for value in (0, -1):
        with pytest.raises(ValueError):
            surrogate.MultiLevelSpikeCount(value)
    for value in (1.5, False):
        with pytest.raises(TypeError):
            surrogate.MultiLevelSpikeCount(value)
    with pytest.raises(ValueError):
        surrogate.MultiLevelSpikeCount(4, grad_window=(1.0, 0.0))
