from importlib.util import find_spec

import pytest
import torch

from spikingjelly.activation_based import neuron, surrogate


def _require_cuda_cupy():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CuPy backend tests.")
    if find_spec("cupy") is None:
        pytest.skip("CuPy package is required for CuPy backend tests.")


def _make_node(kind: str, backend: str, dtype: torch.dtype) -> torch.nn.Module:
    common_kwargs = dict(
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
        detach_reset=False,
        step_mode="m",
        backend=backend,
        store_v_seq=True,
    )

    if kind == "if":
        node = neuron.IFNode(**common_kwargs)
    elif kind == "lif":
        node = neuron.LIFNode(tau=2.0, decay_input=True, **common_kwargs)
    elif kind == "plif":
        node = neuron.ParametricLIFNode(
            init_tau=2.0,
            decay_input=True,
            **common_kwargs,
        )
    else:
        raise ValueError(kind)

    return node.to(device="cuda", dtype=dtype).train()


def _assert_close(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype):
    if dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-4, 1e-4
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cupy_vs_torch_multistep_forward_backward(kind, dtype):
    _require_cuda_cupy()

    seed = 20260428
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x = torch.randn(6, 4, 12, device="cuda", dtype=dtype)

    node_torch = _make_node(kind, backend="torch", dtype=dtype)
    node_cupy = _make_node(kind, backend="cupy", dtype=dtype)

    x_torch = x.detach().clone().requires_grad_(True)
    x_cupy = x.detach().clone().requires_grad_(True)

    s_torch = node_torch(x_torch)
    s_cupy = node_cupy(x_cupy)

    v_torch = node_torch.v_seq
    v_cupy = node_cupy.v_seq

    _assert_close(s_cupy, s_torch, dtype)
    _assert_close(v_cupy, v_torch, dtype)

    loss_torch = s_torch.sum() + 0.5 * v_torch.sum()
    loss_cupy = s_cupy.sum() + 0.5 * v_cupy.sum()

    loss_torch.backward()
    loss_cupy.backward()

    _assert_close(x_cupy.grad, x_torch.grad, dtype)

    grads_torch = {
        name: p.grad.detach().clone()
        for name, p in node_torch.named_parameters()
        if p.grad is not None
    }
    grads_cupy = {
        name: p.grad.detach().clone()
        for name, p in node_cupy.named_parameters()
        if p.grad is not None
    }

    assert grads_cupy.keys() == grads_torch.keys()
    for name in grads_torch:
        _assert_close(grads_cupy[name], grads_torch[name], dtype)


@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cupy_batch_size_change_reconciles_v_state(kind, dtype):
    _require_cuda_cupy()

    seed = 20260428
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x_first = torch.randn(5, 3, 10, device="cuda", dtype=dtype)
    x_second = torch.randn(5, 7, 10, device="cuda", dtype=dtype)

    node_torch = _make_node(kind, backend="torch", dtype=dtype)
    node_cupy = _make_node(kind, backend="cupy", dtype=dtype)

    # Pass 1: establish internal state with batch=3.
    node_torch(x_first)
    node_torch.reset()
    node_cupy(x_first)
    node_cupy.reset()

    # Pass 2: change batch size to 7; v_float_to_tensor should reconcile state.
    s_torch_second = node_torch(x_second)
    v_torch_second = node_torch.v_seq

    s_cupy_second = node_cupy(x_second)
    v_cupy_second = node_cupy.v_seq

    assert node_torch.v.shape == x_second[0].shape
    assert node_cupy.v.shape == x_second[0].shape

    # Compare against fresh nodes to ensure shape-mismatch state was reset/reconciled.
    fresh_torch = _make_node(kind, backend="torch", dtype=dtype)
    fresh_cupy = _make_node(kind, backend="cupy", dtype=dtype)

    s_torch_fresh = fresh_torch(x_second)
    v_torch_fresh = fresh_torch.v_seq
    s_cupy_fresh = fresh_cupy(x_second)
    v_cupy_fresh = fresh_cupy.v_seq

    _assert_close(s_torch_second, s_torch_fresh, dtype)
    _assert_close(v_torch_second, v_torch_fresh, dtype)
    _assert_close(s_cupy_second, s_cupy_fresh, dtype)
    _assert_close(v_cupy_second, v_cupy_fresh, dtype)

    # Keep backend parity check under batch-size transition.
    _assert_close(s_cupy_second, s_torch_second, dtype)
    _assert_close(v_cupy_second, v_torch_second, dtype)
