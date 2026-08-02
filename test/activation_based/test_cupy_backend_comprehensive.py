import pytest
import torch

from spikingjelly.activation_based import functional, neuron, surrogate


def _cupy_available() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def _require_cuda_cupy():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CuPy backend tests.")
    if not _cupy_available():
        pytest.skip("CuPy package is required for CuPy backend tests.")


def _require_cuda_cupy_compile():
    _require_cuda_cupy()
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available.")


def _make_node(
    kind: str,
    backend: str,
    dtype: torch.dtype,
    training: bool = True,
    store_v_seq: bool = True,
) -> torch.nn.Module:
    common_kwargs = dict(
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
        detach_reset=False,
        step_mode="m",
        backend=backend,
        store_v_seq=store_v_seq,
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

    return node.to(device="cuda", dtype=dtype).train(training)


def _assert_close(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype):
    if dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-4, 1e-4
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("kind", "v_reset", "detach_reset", "decay_input"),
    [
        ("if", 0.0, False, None),
        ("if", None, True, None),
        ("lif", 0.0, True, True),
        ("lif", None, False, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cupy_single_step_matches_torch(
    kind, v_reset, detach_reset, decay_input, dtype
):
    _require_cuda_cupy()

    common_kwargs = dict(
        v_threshold=0.8,
        v_reset=v_reset,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
        detach_reset=detach_reset,
        step_mode="s",
    )
    if kind == "if":
        node_torch = neuron.IFNode(backend="torch", **common_kwargs)
        node_cupy = neuron.IFNode(backend="cupy", **common_kwargs)
    else:
        node_torch = neuron.LIFNode(
            tau=2.5,
            decay_input=decay_input,
            backend="torch",
            **common_kwargs,
        )
        node_cupy = neuron.LIFNode(
            tau=2.5,
            decay_input=decay_input,
            backend="cupy",
            **common_kwargs,
        )

    x = torch.randn(3, 5, device="cuda", dtype=dtype)
    v = torch.randn_like(x) * 0.2
    x_torch = x.detach().clone().requires_grad_(True)
    x_cupy = x.detach().clone().requires_grad_(True)
    v_torch = v.detach().clone().requires_grad_(True)
    v_cupy = v.detach().clone().requires_grad_(True)
    node_torch.v = v_torch
    node_cupy.v = v_cupy

    spike_torch = node_torch(x_torch)
    spike_cupy = node_cupy(x_cupy)
    _assert_close(spike_cupy, spike_torch, dtype)
    _assert_close(node_cupy.v, node_torch.v, dtype)

    (spike_torch.sum() + node_torch.v.sum()).backward()
    (spike_cupy.sum() + node_cupy.v.sum()).backward()
    _assert_close(x_cupy.grad, x_torch.grad, dtype)
    _assert_close(v_cupy.grad, v_torch.grad, dtype)


@pytest.mark.parametrize(
    ("kind", "dtype", "v_reset", "detach_reset"),
    [
        ("qif", torch.float32, -0.3, True),
        ("qif", torch.float32, None, False),
        ("qif", torch.float16, -0.3, True),
        ("qif", torch.float16, None, False),
        ("eif", torch.float32, -0.3, True),
        ("eif", torch.float32, None, False),
        ("eif", torch.float16, -0.3, True),
        ("eif", torch.float16, None, False),
        ("izhikevich", torch.float32, -0.3, True),
        ("izhikevich", torch.float32, -0.3, False),
    ],
)
@pytest.mark.parametrize(
    ("training", "store_v_seq"),
    [(True, True), (True, False), (False, False)],
)
def test_cupy_nonlinear_multistep_matches_torch(
    kind, dtype, v_reset, detach_reset, training, store_v_seq
):
    _require_cuda_cupy()

    common_kwargs = dict(
        v_threshold=0.9,
        v_reset=v_reset,
        detach_reset=detach_reset,
        step_mode="m",
        store_v_seq=store_v_seq,
    )
    if kind == "qif":
        node_type = neuron.QIFNode
        kind_kwargs = dict(tau=2.5, a0=0.6, v_rest=-0.2, v_c=0.4)
    elif kind == "eif":
        node_type = neuron.EIFNode
        kind_kwargs = dict(tau=2.5, delta_T=0.7, theta_rh=0.4, v_rest=-0.2)
    else:
        node_type = neuron.IzhikevichNode
        kind_kwargs = dict(
            tau=2.5,
            v_c=0.4,
            a0=0.6,
            v_rest=-0.2,
            w_rest=0.0,
            tau_w=3.0,
            a=0.1,
            b=0.2,
        )
    node_torch = node_type(
        backend="torch",
        surrogate_function=surrogate.ATan(alpha=2.0),
        **common_kwargs,
        **kind_kwargs,
    ).to(device="cuda", dtype=dtype)
    node_cupy = node_type(
        backend="cupy",
        surrogate_function=surrogate.ATan(alpha=2.0),
        **common_kwargs,
        **kind_kwargs,
    ).to(device="cuda", dtype=dtype)
    node_torch.train(training)
    node_cupy.train(training)

    x = torch.randn(4, 2, 4, device="cuda", dtype=dtype) * 0.2
    x_torch = x.detach().clone().requires_grad_(training)
    x_cupy = x.detach().clone().requires_grad_(training)
    v = torch.randn_like(x[0]) * 0.1
    v_torch = v.detach().clone().requires_grad_(training)
    v_cupy = v.detach().clone().requires_grad_(training)
    node_torch.v = v_torch
    node_cupy.v = v_cupy
    if kind == "izhikevich":
        w = torch.randn_like(x[0]) * 0.1
        w_torch = w.detach().clone().requires_grad_(training)
        w_cupy = w.detach().clone().requires_grad_(training)
        node_torch.w = w_torch
        node_cupy.w = w_cupy

    spike_torch = node_torch(x_torch)
    spike_cupy = node_cupy(x_cupy)
    _assert_close(spike_cupy, spike_torch, dtype)
    _assert_close(node_cupy.v, node_torch.v, dtype)
    if store_v_seq:
        _assert_close(node_cupy.v_seq, node_torch.v_seq, dtype)
    if kind == "izhikevich":
        _assert_close(node_cupy.w, node_torch.w, dtype)

    if not training:
        return

    loss_torch = spike_torch.sum() + node_torch.v.sum()
    loss_cupy = spike_cupy.sum() + node_cupy.v.sum()
    if kind == "izhikevich":
        loss_torch = loss_torch + node_torch.w.sum()
        loss_cupy = loss_cupy + node_cupy.w.sum()
    loss_torch.backward()
    loss_cupy.backward()
    _assert_close(x_cupy.grad, x_torch.grad, dtype)
    _assert_close(v_cupy.grad, v_torch.grad, dtype)
    if kind == "izhikevich":
        _assert_close(w_cupy.grad, w_torch.grad, dtype)


def test_cupy_izhikevich_backward_with_only_w_init_grad():
    _require_cuda_cupy()

    kwargs = dict(
        tau=2.5,
        v_c=0.4,
        a0=0.6,
        v_rest=-0.2,
        w_rest=0.0,
        tau_w=3.0,
        a=0.1,
        b=0.2,
        v_threshold=0.9,
        v_reset=-0.3,
        detach_reset=False,
        surrogate_function=surrogate.ATan(alpha=2.0),
        step_mode="m",
        store_v_seq=False,
    )
    node_torch = neuron.IzhikevichNode(backend="torch", **kwargs).cuda()
    node_cupy = neuron.IzhikevichNode(backend="cupy", **kwargs).cuda()

    x = torch.randn(4, 2, 4, device="cuda") * 0.2
    v = torch.randn_like(x[0]) * 0.1
    w = torch.randn_like(x[0]) * 0.1
    w_torch = w.detach().clone().requires_grad_(True)
    w_cupy = w.detach().clone().requires_grad_(True)
    node_torch.v = v.detach().clone()
    node_cupy.v = v.detach().clone()
    node_torch.w = w_torch
    node_cupy.w = w_cupy

    spike_torch = node_torch(x)
    spike_cupy = node_cupy(x)
    _assert_close(spike_cupy, spike_torch, torch.float32)
    _assert_close(node_cupy.v, node_torch.v, torch.float32)
    _assert_close(node_cupy.w, node_torch.w, torch.float32)

    (spike_torch.sum() + node_torch.v.sum() + node_torch.w.sum()).backward()
    (spike_cupy.sum() + node_cupy.v.sum() + node_cupy.w.sum()).backward()
    _assert_close(w_cupy.grad, w_torch.grad, torch.float32)


class _CompileProbeModel(torch.nn.Module):
    def __init__(self, node: torch.nn.Module, features: int):
        super().__init__()
        self.proj = torch.nn.Linear(features, features, bias=False)
        self.node = node

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.node(self.proj(x))


@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    ("training", "store_v_seq"),
    [(True, True), (True, False), (False, False)],
)
def test_cupy_vs_torch_multistep_forward_backward(kind, dtype, training, store_v_seq):
    _require_cuda_cupy()

    seed = 20260428
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x = torch.randn(6, 4, 12, device="cuda", dtype=dtype)

    node_torch = _make_node(kind, "torch", dtype, training, store_v_seq)
    node_cupy = _make_node(kind, "cupy", dtype, training, store_v_seq)

    x_torch = x.detach().clone().requires_grad_(training)
    x_cupy = x.detach().clone().requires_grad_(training)

    s_torch = node_torch(x_torch)
    s_cupy = node_cupy(x_cupy)

    _assert_close(s_cupy, s_torch, dtype)
    _assert_close(node_cupy.v, node_torch.v, dtype)
    if store_v_seq:
        _assert_close(node_cupy.v_seq, node_torch.v_seq, dtype)
        v_torch = node_torch.v_seq
        v_cupy = node_cupy.v_seq
    else:
        v_torch = node_torch.v
        v_cupy = node_cupy.v

    if not training:
        return

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
def test_cupy_compile_inductor_runs_forward_backward(kind, dtype):
    _require_cuda_cupy_compile()

    seed = 20260430
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    node_cupy = _make_node(kind, backend="cupy", dtype=dtype)
    model = (
        _CompileProbeModel(node_cupy, features=12)
        .to(device="cuda", dtype=dtype)
        .train()
    )

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


@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cupy_compile_inductor_matches_eager(kind, dtype):
    _require_cuda_cupy_compile()

    seed = 20260430
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    node_eager = _make_node(kind, backend="cupy", dtype=dtype)
    node_compiled = _make_node(kind, backend="cupy", dtype=dtype)
    node_compiled.load_state_dict(node_eager.state_dict(), strict=True)

    eager_model = (
        _CompileProbeModel(node_eager, features=10)
        .to(device="cuda", dtype=dtype)
        .train()
    )
    compiled_source_model = (
        _CompileProbeModel(node_compiled, features=10)
        .to(device="cuda", dtype=dtype)
        .train()
    )
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
