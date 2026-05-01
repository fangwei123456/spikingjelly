from importlib.util import find_spec

import pytest
import torch

from spikingjelly.activation_based import functional, neuron, surrogate


def _require_cuda_cupy():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CuPy backend tests.")
    if find_spec("cupy") is None:
        pytest.skip("CuPy package is required for CuPy backend tests.")


def _require_cuda_cupy_compile():
    _require_cuda_cupy()
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available.")


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


class _CompileProbeModel(torch.nn.Module):
    def __init__(self, node: torch.nn.Module, features: int):
        super().__init__()
        self.proj = torch.nn.Linear(features, features, bias=False)
        self.node = node

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.node(self.proj(x))


def _install_cupy_path_sentinel(monkeypatch, kind: str):
    hits = {"count": 0}

    if kind == "if":
        from spikingjelly.activation_based.cuda_kernel.auto_cuda.neuron_kernel import (
            integrate_and_fire as ac_if,
        )

        original = ac_if.multistep_if

        def _wrapped(*args, **kwargs):
            hits["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ac_if, "multistep_if", _wrapped)
    elif kind == "lif":
        from spikingjelly.activation_based.cuda_kernel.auto_cuda.neuron_kernel import (
            lif as ac_lif,
        )

        original = ac_lif.multistep_lif

        def _wrapped(*args, **kwargs):
            hits["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ac_lif, "multistep_lif", _wrapped)
    elif kind == "plif":
        from spikingjelly.activation_based.cuda_kernel.auto_cuda.neuron_kernel import (
            plif as ac_plif,
        )

        original = ac_plif.multistep_plif

        def _wrapped(*args, **kwargs):
            hits["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ac_plif, "multistep_plif", _wrapped)
    else:
        raise ValueError(kind)

    return hits


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


@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cupy_compile_inductor_runs_forward_backward(kind, dtype, monkeypatch):
    _require_cuda_cupy_compile()

    seed = 20260430
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cupy_hits = _install_cupy_path_sentinel(monkeypatch, kind)

    node_cupy = _make_node(kind, backend="cupy", dtype=dtype)
    model = _CompileProbeModel(node_cupy, features=12).to(device="cuda", dtype=dtype).train()

    compiled_model = torch.compile(
        model,
        backend="inductor",
        options={
            "triton.cudagraphs": False,
            "triton.cudagraph_trees": False,
        },
    )

    for _ in range(2):
        x = torch.randn(6, 4, 12, device="cuda", dtype=dtype, requires_grad=True)
        functional.reset_net(model)
        y = compiled_model(x)
        assert y.shape == x.shape
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
    assert cupy_hits["count"] > 0


@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cupy_compile_inductor_matches_eager(kind, dtype, monkeypatch):
    _require_cuda_cupy_compile()

    seed = 20260430
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cupy_hits = _install_cupy_path_sentinel(monkeypatch, kind)

    node_eager = _make_node(kind, backend="cupy", dtype=dtype)
    node_compiled = _make_node(kind, backend="cupy", dtype=dtype)
    node_compiled.load_state_dict(node_eager.state_dict(), strict=True)

    eager_model = _CompileProbeModel(node_eager, features=10).to(device="cuda", dtype=dtype).train()
    compiled_source_model = _CompileProbeModel(node_compiled, features=10).to(
        device="cuda", dtype=dtype
    ).train()
    compiled_source_model.load_state_dict(eager_model.state_dict(), strict=True)

    compiled_model = torch.compile(
        compiled_source_model,
        backend="inductor",
        options={
            "triton.cudagraphs": False,
            "triton.cudagraph_trees": False,
        },
    )

    x_ref = torch.randn(7, 3, 10, device="cuda", dtype=dtype)
    x_eager = x_ref.clone().detach().requires_grad_(True)
    x_compiled = x_ref.clone().detach().requires_grad_(True)

    functional.reset_net(eager_model)
    functional.reset_net(compiled_source_model)
    y_eager = eager_model(x_eager)
    y_compiled = compiled_model(x_compiled)

    _assert_close(y_compiled, y_eager, dtype)

    y_eager.sum().backward()
    y_compiled.sum().backward()

    _assert_close(x_compiled.grad, x_eager.grad, dtype)
    assert cupy_hits["count"] > 0
