import json
import os
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from importlib.util import find_spec

import pytest
import torch

from spikingjelly.activation_based import functional, neuron, surrogate


def _require_cuda_triton_compile():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton compile compatibility tests.")
    if find_spec("triton") is None:
        pytest.skip("Triton package is required for Triton backend tests.")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available.")
    if not hasattr(torch, "_dynamo") or not hasattr(torch._dynamo, "explain"):
        pytest.skip("torch._dynamo.explain is not available.")


def _make_surrogate(name: str) -> surrogate.SurrogateFunctionBase:
    if name == "Sigmoid":
        return surrogate.Sigmoid(alpha=4.0)
    if name == "ATan":
        return surrogate.ATan(alpha=2.0)
    raise ValueError(name)


@contextmanager
def _inductor_single_process_compile():
    config = getattr(torch, "_inductor", None)
    config = getattr(config, "config", None)
    if config is None or not hasattr(config, "compile_threads"):
        yield
        return

    old_compile_threads = config.compile_threads
    try:
        # Use in-process codegen to avoid multiprocessing pickle issues
        # for Triton JIT surrogate callables.
        config.compile_threads = 1
        yield
    finally:
        config.compile_threads = old_compile_threads


def _build_node(
    kind: str, backend: str, surrogate_fn: surrogate.SurrogateFunctionBase
) -> torch.nn.Module:
    if kind == "lif":
        return neuron.LIFNode(
            tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate_fn,
            detach_reset=False,
            step_mode="m",
            backend=backend,
        )
    if kind == "if":
        return neuron.IFNode(
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate_fn,
            detach_reset=False,
            step_mode="m",
            backend=backend,
        )
    if kind == "plif":
        return neuron.ParametricLIFNode(
            init_tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate_fn,
            detach_reset=False,
            step_mode="m",
            backend=backend,
        )
    raise ValueError(kind)


class _CompileModel(torch.nn.Module):
    def __init__(self, node: torch.nn.Module, features: int):
        super().__init__()
        self.proj = torch.nn.Linear(features, features, bias=False)
        self.node = node

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.node(self.proj(x))


def _graph_break_count(explain_output) -> int:
    if hasattr(explain_output, "graph_break_count"):
        return int(explain_output.graph_break_count)
    if hasattr(explain_output, "break_reasons"):
        return len(explain_output.break_reasons)
    raise TypeError(
        f"Unsupported explain output type: {type(explain_output).__name__}."
    )


@pytest.mark.parametrize("kind", ["lif", "if", "plif"])
def test_dynamo_explain_no_graph_breaks(kind):
    _require_cuda_triton_compile()

    torch.manual_seed(20260419)
    torch.cuda.manual_seed_all(20260419)
    torch._dynamo.reset()

    node = _build_node(kind, "triton", _make_surrogate("Sigmoid")).cuda().train()
    model = _CompileModel(node, features=16).cuda().train()
    x = torch.randn(8, 4, 16, device="cuda", requires_grad=True)

    explain_output = torch._dynamo.explain(model)(x)
    assert _graph_break_count(explain_output) == 0


@pytest.mark.parametrize("kind", ["lif", "if", "plif"])
def test_compile_inductor_runs_forward_backward(kind):
    _require_cuda_triton_compile()

    torch.manual_seed(20260419)
    torch.cuda.manual_seed_all(20260419)

    node = _build_node(kind, "triton", _make_surrogate("Sigmoid")).cuda().train()
    model = _CompileModel(node, features=12).cuda().train()

    with _inductor_single_process_compile():
        compiled_model = torch.compile(
            model,
            backend="inductor",
            options={
                "triton.cudagraphs": False,
                "triton.cudagraph_trees": False,
            },
        )

        for _ in range(2):
            x = torch.randn(6, 3, 12, device="cuda", requires_grad=True)
            functional.reset_net(model)
            y = compiled_model(x)
            assert y.shape == x.shape
            loss = y.sum()
            loss.backward()
            assert x.grad is not None
            del loss
            del y


@pytest.mark.parametrize("kind", ["lif", "if", "plif"])
def test_standard_inductor_backend_matches_torch_backend(kind):
    _require_cuda_triton_compile()

    torch.manual_seed(20260429)
    torch.cuda.manual_seed_all(20260429)

    torch_node = _build_node(kind, "torch", _make_surrogate("Sigmoid")).cuda().train()
    inductor_node = _build_node(kind, "inductor", _make_surrogate("Sigmoid")).cuda().train()
    inductor_node.load_state_dict(torch_node.state_dict(), strict=True)

    x_ref = torch.randn(9, 3, 18, device="cuda", dtype=torch.float32)
    x_torch = x_ref.clone().detach().requires_grad_(True)
    x_inductor = x_ref.clone().detach().requires_grad_(True)

    functional.reset_net(torch_node)
    functional.reset_net(inductor_node)
    y_torch = torch_node(x_torch)
    y_inductor = inductor_node(x_inductor)
    assert torch.allclose(y_torch, y_inductor, atol=1e-5, rtol=1e-4)

    y_torch.sum().backward()
    y_inductor.sum().backward()
    assert torch.allclose(x_torch.grad, x_inductor.grad, atol=1e-5, rtol=1e-4)

    if kind == "plif":
        assert torch.allclose(
            torch_node.w.grad, inductor_node.w.grad, atol=1e-5, rtol=1e-4
        )


@pytest.mark.parametrize("kind", ["lif", "if", "plif"])
def test_standard_inductor_backend_cache_key_tracks_threshold_changes(kind, monkeypatch):
    node = _build_node(kind, "inductor", _make_surrogate("Sigmoid")).train()
    captured_keys = []

    def _fake_compile(cache_key, fn):
        captured_keys.append(cache_key)
        return fn

    monkeypatch.setattr(node, "_compile_inductor_graph", _fake_compile)
    x = torch.randn(4, 2, 8)

    functional.reset_net(node)
    node(x)
    node.v_threshold = 1.5
    functional.reset_net(node)
    node(x)

    assert len(captured_keys) == 2
    assert captured_keys[0] != captured_keys[1]


def test_lif_inductor_backend_cache_key_tracks_tau_changes(monkeypatch):
    node = _build_node("lif", "inductor", _make_surrogate("Sigmoid")).train()
    captured_keys = []

    def _fake_compile(cache_key, fn):
        captured_keys.append(cache_key)
        return fn

    monkeypatch.setattr(node, "_compile_inductor_graph", _fake_compile)
    x = torch.randn(4, 2, 8)

    functional.reset_net(node)
    node(x)
    node.tau = 3.0
    functional.reset_net(node)
    node(x)

    assert len(captured_keys) == 2
    assert captured_keys[0] != captured_keys[1]


@pytest.mark.parametrize("kind", ["lif", "if", "plif"])
def test_standard_inductor_backend_cache_key_tracks_cuda_device_changes(kind, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("At least 2 CUDA devices are required for cache key device-index coverage.")

    node = _build_node(kind, "inductor", _make_surrogate("Sigmoid")).train().to("cuda:0")
    captured_keys = []

    def _fake_compile(cache_key, fn):
        captured_keys.append(cache_key)
        return fn

    monkeypatch.setattr(node, "_compile_inductor_graph", _fake_compile)
    x0 = torch.randn(4, 2, 8, device="cuda:0")
    x1 = torch.randn(4, 2, 8, device="cuda:1")

    functional.reset_net(node)
    node(x0)
    node.to("cuda:1")
    functional.reset_net(node)
    node(x1)

    assert len(captured_keys) == 2
    assert captured_keys[0] != captured_keys[1]


@pytest.mark.parametrize("kind", ["lif", "if", "plif"])
def test_standard_inductor_backend_cache_key_tracks_runtime_shape_changes(kind, monkeypatch):
    node = _build_node(kind, "inductor", _make_surrogate("Sigmoid")).train()
    captured_keys = []

    def _fake_compile(cache_key, fn):
        captured_keys.append(cache_key)
        return fn

    monkeypatch.setattr(node, "_compile_inductor_graph", _fake_compile)
    x3 = torch.randn(4, 2, 8)
    x4 = torch.randn(4, 2, 2, 4)

    functional.reset_net(node)
    node(x3)
    functional.reset_net(node)
    node(x4)

    assert len(captured_keys) == 2
    assert captured_keys[0] != captured_keys[1]


@pytest.mark.parametrize("kind", ["lif", "if", "plif"])
@pytest.mark.parametrize("sg_name", ["Sigmoid", "ATan"])
def test_triton_vs_torch_forward_backward_consistency(kind, sg_name):
    _require_cuda_triton_compile()

    torch.manual_seed(20260419)
    torch.cuda.manual_seed_all(20260419)

    torch_node = _build_node(kind, "torch", _make_surrogate(sg_name)).cuda().train()
    triton_node = _build_node(kind, "triton", _make_surrogate(sg_name)).cuda().train()
    triton_node.load_state_dict(torch_node.state_dict(), strict=True)

    x_ref = torch.randn(10, 2, 20, device="cuda", dtype=torch.float32)
    x_torch = x_ref.clone().detach().requires_grad_(True)
    x_triton = x_ref.clone().detach().requires_grad_(True)

    functional.reset_net(torch_node)
    functional.reset_net(triton_node)
    y_torch = torch_node(x_torch)
    y_triton = triton_node(x_triton)
    assert torch.allclose(y_torch, y_triton, atol=1e-5, rtol=1e-4)

    y_torch.sum().backward()
    y_triton.sum().backward()
    assert torch.allclose(x_torch.grad, x_triton.grad, atol=1e-5, rtol=1e-4)

    if kind == "plif":
        assert torch.allclose(
            torch_node.w.grad, triton_node.w.grad, atol=1e-5, rtol=1e-4
        )


_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import json
    import os

    import torch

    force_custom = os.environ["SJ_FORCE_CUSTOM_OP"] == "1"
    kind = os.environ["SJ_NODE_KIND"]
    if force_custom:
        os.environ["SJ_USE_TRITON_OP"] = "0"
    else:
        os.environ["SJ_USE_TRITON_OP"] = "1"

    from spikingjelly.activation_based import neuron, surrogate

    torch.manual_seed(20260419)
    torch.cuda.manual_seed_all(20260419)

    if kind == "lif":
        node = neuron.LIFNode(
            tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            detach_reset=False,
            step_mode="m",
            backend="triton",
        )
    elif kind == "if":
        node = neuron.IFNode(
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            detach_reset=False,
            step_mode="m",
            backend="triton",
        )
    elif kind == "plif":
        node = neuron.ParametricLIFNode(
            init_tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            detach_reset=False,
            step_mode="m",
            backend="triton",
        )
    else:
        raise ValueError(kind)

    node = node.cuda().train()
    x = torch.randn(4, 2, 7, device="cuda", dtype=torch.float32, requires_grad=True)
    y = node(x)
    y.sum().backward()

    v = os.getenv("SJ_USE_TRITON_OP")
    env_enabled = True if v is None else v.strip().lower() not in ("0", "false")
    use_triton_op = bool(env_enabled and hasattr(torch.library, "triton_op"))

    payload = {
        "use_triton_op": use_triton_op,
        "out": y.detach().cpu().tolist(),
        "x_grad": x.grad.detach().cpu().tolist(),
    }
    if kind == "plif":
        payload["w_grad"] = float(node.w.grad.detach().cpu().item())

    print("JSON_RESULT=" + json.dumps(payload))
    """
)


def _run_subprocess_path(kind: str, force_custom_op: bool) -> dict:
    env = os.environ.copy()
    env["SJ_NODE_KIND"] = kind
    env["SJ_FORCE_CUSTOM_OP"] = "1" if force_custom_op else "0"
    env["SJ_USE_TRITON_OP"] = "0" if force_custom_op else "1"

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SCRIPT],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(
            "Subprocess probe timed out while running _SUBPROCESS_SCRIPT "
            f"for kind={kind}, force_custom_op={force_custom_op}: {e}"
        )

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("JSON_RESULT="):
            return json.loads(line[len("JSON_RESULT=") :])

    raise AssertionError(
        "Missing JSON_RESULT in subprocess stdout. "
        f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
    )


@pytest.mark.parametrize("kind", ["lif", "if", "plif"])
def test_triton_op_and_custom_op_fallback_consistency(kind):
    _require_cuda_triton_compile()

    if not hasattr(torch.library, "triton_op"):
        pytest.skip("torch.library.triton_op is unavailable on this torch build.")

    result_triton_op = _run_subprocess_path(kind, force_custom_op=False)
    result_custom_op = _run_subprocess_path(kind, force_custom_op=True)

    assert result_triton_op["use_triton_op"] is True
    assert result_custom_op["use_triton_op"] is False

    out_triton = torch.as_tensor(result_triton_op["out"])
    out_custom = torch.as_tensor(result_custom_op["out"])
    grad_triton = torch.as_tensor(result_triton_op["x_grad"])
    grad_custom = torch.as_tensor(result_custom_op["x_grad"])

    assert torch.allclose(out_triton, out_custom, atol=1e-5, rtol=1e-4)
    assert torch.allclose(grad_triton, grad_custom, atol=1e-5, rtol=1e-4)

    if kind == "plif":
        assert abs(result_triton_op["w_grad"] - result_custom_op["w_grad"]) <= 1e-5


@pytest.mark.parametrize("kind", ["lif", "if", "plif"])
def test_triton_unsupported_surrogate_raises_not_implemented(kind):
    _require_cuda_triton_compile()

    torch.manual_seed(20260419)
    torch.cuda.manual_seed_all(20260419)

    node = _build_node(
        kind, "triton", surrogate.PiecewiseLeakyReLU(w=1.0, c=0.01)
    ).cuda().train()
    x = torch.randn(6, 2, 9, device="cuda", requires_grad=True)

    with pytest.raises(NotImplementedError, match="PiecewiseLeakyReLU"):
        y = node(x)
        y.sum().backward()
