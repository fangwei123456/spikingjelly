import pytest
import torch

from spikingjelly.activation_based import functional, neuron, surrogate


def _assert_close(actual: torch.Tensor, expected: torch.Tensor):
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_ilif_rejects_invalid_max_spike_count():
    for value in (0, -1):
        with pytest.raises(ValueError):
            neuron.ILIFNode(max_spike_count=value)
    for value in (1.5, True):
        with pytest.raises(TypeError):
            neuron.ILIFNode(max_spike_count=value)


def test_ilif_exposes_triton_for_both_step_modes():
    assert neuron.ILIFNode(step_mode="s").supported_backends == ("torch", "triton")
    assert neuron.ILIFNode(step_mode="m").supported_backends == ("torch", "triton")


def test_ilif_training_outputs_integer_counts_and_updates_voltage():
    node = neuron.ILIFNode(v_threshold=1.0, max_spike_count=4, decay=0.25)
    x = torch.tensor([[3.2, 0.4, 5.8]])

    y = node(x)

    expected = torch.tensor([[3.0, 0.0, 4.0]])
    assert torch.equal(y, expected)
    torch.testing.assert_close(node.v, x - expected)


def test_ilif_uses_spike_count_surrogate():
    node = neuron.ILIFNode(max_spike_count=4)

    assert isinstance(node.surrogate_function, surrogate.MultiLevelSpikeCount)
    assert node.surrogate_function.max_spike_count == 4


def test_ilif_decay_is_applied_during_charge():
    node = neuron.ILIFNode(v_threshold=1.0, max_spike_count=4, decay=0.25)

    node(torch.tensor([[3.2]]))
    y = node(torch.tensor([[0.0]]))

    assert torch.equal(y, torch.tensor([[0.0]]))
    torch.testing.assert_close(node.v, torch.tensor([[0.05]]))


def test_ilif_eval_returns_integer_counts():
    node = neuron.ILIFNode(v_threshold=1.0, max_spike_count=4, decay=0.25).eval()

    y = node(torch.tensor([[3.2, 5.8]]))

    assert torch.equal(y, torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(node.v, torch.tensor([[0.2, 1.8]]))


def test_ilif_train_and_eval_have_identical_forward_semantics():
    train_node = neuron.ILIFNode(
        v_threshold=0.5,
        max_spike_count=4,
        decay=0.25,
        step_mode="m",
        store_v_seq=True,
    )
    eval_node = neuron.ILIFNode(
        v_threshold=0.5,
        max_spike_count=4,
        decay=0.25,
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
        max_spike_count=4,
        decay=0.25,
        step_mode="m",
        store_v_seq=True,
    ).eval()
    x_seq = torch.tensor([[[3.2]], [[0.0]], [[4.8]], [[0.0]]])

    y = node(x_seq)

    assert y.shape == x_seq.shape
    assert node.v_seq.shape == x_seq.shape
    assert torch.equal(y.flatten(), torch.tensor([3.0, 0.0, 4.0, 0.0]))


def test_ilif_training_straight_through_gradient_is_windowed():
    node = neuron.ILIFNode(v_threshold=1.0, max_spike_count=4, decay=0.0)
    x = torch.tensor([[-0.5, 0.5, 4.5]], requires_grad=True)

    y = node(x)
    y.sum().backward()

    assert torch.equal(y.detach(), torch.tensor([[0.0, 0.0, 4.0]]))
    torch.testing.assert_close(x.grad, torch.tensor([[0.0, 1.0, 0.0]]))


def test_ilif_supports_custom_gradient_window():
    node = neuron.ILIFNode(
        max_spike_count=4,
        decay=0.0,
        grad_window=(-0.5, 4.5),
    )
    x = torch.tensor([[-0.25, 4.25, 4.75]], requires_grad=True)

    y = node(x)
    y.sum().backward()

    assert torch.equal(y.detach(), torch.tensor([[0.0, 4.0, 4.0]]))
    torch.testing.assert_close(x.grad, torch.tensor([[1.0, 1.0, 0.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ilif_single_step_triton_matches_torch_training(detach_reset, dtype):
    pytest.importorskip("triton")
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 requires compute capability >= 8.")
    kwargs = {
        "v_threshold": 0.5,
        "max_spike_count": 4,
        "decay": 0.25,
        "detach_reset": detach_reset,
        "step_mode": "s",
    }
    torch_node = neuron.ILIFNode(backend="torch", **kwargs).to(
        device="cuda", dtype=dtype
    )
    triton_node = neuron.ILIFNode(backend="triton", **kwargs).to(
        device="cuda", dtype=dtype
    )
    x = torch.randn(4, 16, device="cuda", dtype=dtype)
    x_torch = x.clone().requires_grad_()
    x_triton = x.clone().requires_grad_()
    v_torch = torch.randn_like(x).requires_grad_()
    v_triton = v_torch.detach().clone().requires_grad_()
    torch_node.v = v_torch
    triton_node.v = v_triton
    weight = torch.randn_like(x)

    y_torch = torch_node(x_torch)
    y_triton = triton_node(x_triton)
    assert torch.equal(y_triton, y_torch)
    _assert_close(triton_node.v, torch_node.v)
    loss_torch = (y_torch * weight).sum() + 0.1 * torch_node.v.sum()
    loss_triton = (y_triton * weight).sum() + 0.1 * triton_node.v.sum()
    loss_torch.backward()
    loss_triton.backward()

    _assert_close(x_triton.grad, x_torch.grad)
    _assert_close(v_triton.grad, v_torch.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ilif_single_step_triton_rounds_half_to_even():
    pytest.importorskip("triton")
    node = neuron.ILIFNode(
        max_spike_count=4,
        decay=0.0,
        step_mode="s",
        backend="triton",
    ).cuda()
    x = torch.tensor([[0.5, 1.5, 2.5, 3.5, 4.5]], device="cuda")

    y = node(x)

    assert torch.equal(
        y,
        torch.tensor([[0.0, 2.0, 2.0, 4.0, 4.0]], device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ilif_single_step_functional_triton_interface():
    pytest.importorskip("triton")
    x = torch.randn(7, 13, device="cuda")
    v = torch.randn_like(x)
    torch_node = neuron.ILIFNode(
        decay=0.25,
        max_spike_count=4,
        backend="torch",
    ).cuda()
    torch_node.eval()
    torch_node.v = v.clone()

    with torch.inference_mode():
        expected_spike = torch_node(x)
        actual_spike, actual_v = functional.ilif_single_step_triton(
            x,
            v,
            torch_node.decay,
            torch_node.v_threshold,
            torch_node.surrogate_function.max_spike_count,
            torch_node.surrogate_function.grad_min,
            torch_node.surrogate_function.grad_max,
        )
    _assert_close(actual_spike, expected_spike)
    _assert_close(actual_v, torch_node.v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("store_v_seq", [False, True])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("decay", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ilif_triton_matches_torch_training(store_v_seq, detach_reset, decay, dtype):
    pytest.importorskip("triton")
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 requires compute capability >= 8.")
    torch_node = neuron.ILIFNode(
        v_threshold=0.5,
        max_spike_count=4,
        decay=decay,
        detach_reset=detach_reset,
        step_mode="m",
        backend="torch",
        store_v_seq=store_v_seq,
    ).to(device="cuda", dtype=dtype)
    triton_node = neuron.ILIFNode(
        v_threshold=0.5,
        max_spike_count=4,
        decay=decay,
        detach_reset=detach_reset,
        step_mode="m",
        backend="triton",
        store_v_seq=store_v_seq,
    ).to(device="cuda", dtype=dtype)
    x = torch.randn(12, 4, 16, device="cuda", dtype=dtype)
    x_torch = x.clone().requires_grad_()
    x_triton = x.clone().requires_grad_()
    v_torch = torch.randn_like(x[0]).requires_grad_()
    v_triton = v_torch.detach().clone().requires_grad_()
    torch_node.v = v_torch
    triton_node.v = v_triton
    weight = torch.randn_like(x)

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
@pytest.mark.parametrize("decay", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ilif_triton_matches_torch_eval(store_v_seq, decay, dtype):
    pytest.importorskip("triton")
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 requires compute capability >= 8.")
    torch_node = (
        neuron.ILIFNode(
            v_threshold=0.5,
            decay=decay,
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
            decay=decay,
            step_mode="m",
            backend="triton",
            store_v_seq=store_v_seq,
        )
        .to(device="cuda", dtype=dtype)
        .eval()
    )
    x = torch.randn(12, 4, 16, device="cuda", dtype=dtype)
    x[0, 0, 0] = 1.6
    v = torch.zeros_like(x[0])
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
        max_spike_count=4,
        decay=0.0,
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
    kwargs = dict(
        max_spike_count=4,
        decay=0.25,
        grad_window=(-0.5, 4.5),
        step_mode="m",
    )
    torch_node = neuron.ILIFNode(backend="torch", **kwargs).cuda()
    triton_node = neuron.ILIFNode(backend="triton", **kwargs).cuda()
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
