import json
import os
import subprocess
import sys
import textwrap
from contextlib import contextmanager

import pytest
import torch

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.triton_kernel.neuron_kernel import (
    integrate_and_fire as triton_if_kernel,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel import (
    lif as triton_lif_kernel,
)


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def _require_cuda_triton_compile():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton compile compatibility tests.")
    if not _triton_available():
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


@pytest.mark.parametrize(
    ("surrogate_fn", "detach_reset", "v_reset"),
    [
        (surrogate.Sigmoid(alpha=4.0), False, 0.0),
        (surrogate.ATan(alpha=2.0), True, None),
    ],
)
def test_compiled_triton_lif_backward_matches_eager(
    surrogate_fn, detach_reset, v_reset
):
    _require_cuda_triton_compile()
    torch.manual_seed(20260830)
    kwargs = {
        "tau": 2.0,
        "v_reset": v_reset,
        "surrogate_function": surrogate_fn,
        "detach_reset": detach_reset,
        "step_mode": "m",
        "backend": "triton",
    }
    eager = neuron.LIFNode(**kwargs).cuda().train()
    compiled_node = neuron.LIFNode(**kwargs).cuda().train()
    x = torch.randn(7, 2, 20, device="cuda")
    x_eager = x.clone().requires_grad_()
    x_compiled = x.clone().requires_grad_()

    eager(x_eager).sum().backward()
    with _inductor_single_process_compile():
        compiled = torch.compile(
            compiled_node,
            backend="inductor",
            options={
                "triton.cudagraphs": False,
                "triton.cudagraph_trees": False,
            },
        )
        compiled(x_compiled).sum().backward()

    torch.testing.assert_close(x_compiled.grad, x_eager.grad)


def test_inductor_is_not_a_standard_neuron_backend():
    with pytest.raises(NotImplementedError, match="not a supported backend"):
        _build_node("lif", "inductor", _make_surrogate("Sigmoid"))


@pytest.mark.parametrize(
    ("kind", "sg_name"),
    [("lif", "Sigmoid"), ("if", "ATan"), ("plif", "Sigmoid")],
)
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


@pytest.mark.parametrize(
    ("kind", "T", "dtype", "v_reset", "detach_reset"),
    [
        ("lif", 7, torch.float32, 0.0, False),
        ("lif", 65, torch.float16, None, True),
        ("if", 7, torch.float32, None, True),
        ("if", 65, torch.bfloat16, None, True),
    ],
)
def test_triton_last_state_matches_full_voltage_sequence(
    kind, T, dtype, v_reset, detach_reset
):
    """Match training results and gradients against full voltage storage."""
    _require_cuda_triton_compile()
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")

    def make_node(store_v_seq):
        """Create a Triton node with the requested voltage storage mode."""
        common = {
            "v_threshold": 1.0,
            "v_reset": v_reset,
            "surrogate_function": surrogate.ATan(alpha=2.0),
            "detach_reset": detach_reset,
            "step_mode": "m",
            "backend": "triton",
            "store_v_seq": store_v_seq,
        }
        if kind == "lif":
            return neuron.LIFNode(tau=2.0, decay_input=True, **common)
        return neuron.IFNode(**common)

    torch.manual_seed(20260718)
    torch.cuda.manual_seed_all(20260718)
    full_node = make_node(True).cuda().train()
    last_node = make_node(False).cuda().train()
    x = torch.randn(T, 3, 37, device="cuda", dtype=dtype)
    grad_spike = torch.randn_like(x)
    grad_v = torch.randn_like(x[0])
    x_full = x.detach().clone().requires_grad_(True)
    x_last = x.detach().clone().requires_grad_(True)

    spike_full = full_node(x_full)
    spike_last = last_node(x_last)
    loss_full = (spike_full * grad_spike).sum() + (full_node.v * grad_v).sum()
    loss_last = (spike_last * grad_spike).sum() + (last_node.v * grad_v).sum()
    loss_full.backward()
    loss_last.backward()

    assert full_node.v_seq.shape == x.shape
    assert last_node.v.shape == x.shape[1:]
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-4
    elif dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:
        atol, rtol = 1e-2, 1e-2
    assert torch.allclose(spike_full, spike_last, atol=atol, rtol=rtol)
    assert torch.allclose(full_node.v, last_node.v, atol=atol, rtol=rtol)
    assert torch.allclose(x_full.grad, x_last.grad, atol=atol, rtol=rtol)
    full_node.detach()
    last_node.detach()


@pytest.mark.parametrize(
    ("kind", "T", "dtype", "v_reset"),
    [
        ("lif", 1, torch.float32, 0.0),
        ("lif", 65, torch.float16, None),
        ("if", 1, torch.bfloat16, None),
        ("if", 65, torch.float32, None),
    ],
)
def test_triton_last_state_inference_matches_full_voltage_sequence(
    kind, T, dtype, v_reset
):
    """Match inference results against full voltage storage."""
    _require_cuda_triton_compile()
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")

    common = {
        "v_threshold": 1.0,
        "v_reset": v_reset,
        "step_mode": "m",
        "backend": "triton",
    }
    cls = neuron.LIFNode if kind == "lif" else neuron.IFNode
    full_node = cls(store_v_seq=True, **common).cuda().eval()
    last_node = cls(store_v_seq=False, **common).cuda().eval()
    torch.manual_seed(20260718)
    torch.cuda.manual_seed_all(20260718)
    x = torch.randn(T, 3, 37, device="cuda", dtype=dtype)

    with torch.inference_mode():
        spike_full = full_node(x)
        spike_last = last_node(x)

    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-4
    elif dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:
        atol, rtol = 1e-2, 1e-2
    assert full_node.v_seq.shape == x.shape
    assert last_node.v.shape == x.shape[1:]
    assert torch.allclose(spike_full, spike_last, atol=atol, rtol=rtol)
    assert torch.allclose(full_node.v, last_node.v, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kind", ["lif", "if"])
def test_triton_last_state_fake_output_shapes(kind):
    """Check fake operators return final-state voltage shapes."""
    x_seq = torch.empty(7, 3, 37)
    v_init = torch.empty(3, 37)

    if kind == "lif":
        inference_outputs = triton_lif_kernel._multistep_lif_inference_fake(
            x_seq, v_init, True, 2.0, 1.0, 0.0, False, False
        )
        forward_outputs = triton_lif_kernel._multistep_lif_forward_fake(
            x_seq, v_init, True, 2.0, 1.0, 0.0, False, False, 0, 2.0, False
        )
    else:
        inference_outputs = triton_if_kernel._multistep_if_inference_fake(
            x_seq, v_init, 1.0, 0.0, False, False
        )
        forward_outputs = triton_if_kernel._multistep_if_forward_fake(
            x_seq, v_init, 1.0, 0.0, False, False, 0, 2.0, False
        )

    assert tuple(output.shape for output in inference_outputs) == (
        x_seq.shape,
        v_init.shape,
    )
    assert tuple(output.shape for output in forward_outputs) == (
        x_seq.shape,
        v_init.shape,
        x_seq.shape,
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

    from spikingjelly.activation_based.triton_kernel import triton_utils

    use_triton_op = triton_utils._USE_TRITON_OP

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


@pytest.mark.parametrize("startup_mode", ["0", "1"])
def test_registration_mode_is_fixed_at_import(startup_mode):
    env = os.environ.copy()
    env["SJ_USE_TRITON_OP"] = startup_mode
    env["SJ_USE_WRAP_TRITON"] = "0"
    script = textwrap.dedent(
        """
        import json
        import os
        import torch
        from spikingjelly.activation_based.triton_kernel import triton_utils
        import spikingjelly.activation_based.neuron

        assert hasattr(torch.ops.sj, "multistep_activation_aware_if_inference")

        mode = triton_utils._USE_TRITON_OP
        os.environ["SJ_USE_TRITON_OP"] = "0" if mode else "1"
        torch.library.custom_op = lambda *args, **kwargs: lambda f: "custom"
        triton_utils.triton_op = lambda *args, **kwargs: lambda f: "triton"
        kernel = object()
        if not mode:
            assert triton_utils.wrap_triton(kernel) is kernel
        print(json.dumps({
            "mode": mode,
            "registered": triton_utils.register_op("sj::probe")(lambda: None),
            "wrapped": triton_utils.wrap_triton is getattr(torch.library, "wrap_triton", None),
        }))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["mode"] is (result["registered"] == "triton")
    assert result["wrapped"] is result["mode"]
    if startup_mode == "0":
        assert result["mode"] is False


def test_import_without_triton_has_no_discovery_warnings():
    if _triton_available():
        pytest.skip("This check needs an environment without Triton.")
    script = textwrap.dedent(
        """
        import spikingjelly.activation_based.neuron
        import torch
        try:
            torch.ops.sj.multistep_if_inference(
                torch.zeros(1, 2), torch.zeros(2), 1.0, 0.0, False, False
            )
        except NotImplementedError as error:
            assert "CPU" in str(error)
        else:
            raise AssertionError("CUDA-only fallback accepted a CPU input")
        from spikingjelly.activation_based.triton_kernel import triton_utils
        torch.library.custom_op = lambda *args, **kwargs: lambda f: f
        @triton_utils.register_op("sj::probe")
        def probe(x: torch.Tensor) -> torch.Tensor:
            raise AssertionError("Original implementation reached")
        try:
            probe(torch.zeros(1))
        except ImportError as error:
            assert "Triton is not installed" in str(error)
            assert isinstance(error.__cause__, (ImportError, OSError))
        else:
            raise AssertionError("Missing Triton did not raise")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "find_triton_kernels" not in completed.stderr


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        ("language_import", "ModuleNotFoundError"),
        ("language_symbol", "AttributeError"),
        ("language_runtime", "RuntimeError"),
    ],
)
def test_partial_triton_initialization_stays_optional(failure, error_type):
    env = os.environ.copy()
    env["SJ_BROKEN_TRITON"] = failure
    env["SJ_ERROR_TYPE"] = error_type
    script = textwrap.dedent(
        """
        import os
        import sys
        import types
        import torch

        triton = types.ModuleType("triton")
        triton.__path__ = []
        triton.jit = lambda f: f
        triton.autotune = lambda **kwargs: lambda f: f
        triton.Config = lambda *args, **kwargs: object()
        sys.modules["triton"] = triton
        if os.environ["SJ_BROKEN_TRITON"] != "language_import":
            language = types.ModuleType("triton.language")
            language.constexpr = object()
            if os.environ["SJ_BROKEN_TRITON"] == "language_runtime":
                def missing_symbol(name):
                    if name == "int1":
                        raise RuntimeError("Triton initialization sentinel")
                    raise AttributeError(name)
                language.__getattr__ = missing_symbol
            sys.modules["triton.language"] = language

        import spikingjelly.activation_based.neuron
        from spikingjelly.activation_based.triton_kernel import triton_utils

        torch.library.custom_op = lambda *args, **kwargs: lambda f: f
        @triton_utils.register_op("sj::probe")
        def probe(x: torch.Tensor) -> torch.Tensor:
            raise AssertionError("Dummy implementation reached")
        try:
            probe(torch.zeros(1))
        except ImportError as error:
            assert "failed to initialize" in str(error)
            assert type(error.__cause__).__name__ == os.environ["SJ_ERROR_TYPE"]
        else:
            raise AssertionError("Broken Triton did not fail clearly")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_registered_op_reports_missing_triton():
    if _triton_available():
        pytest.skip("This check needs an environment without Triton.")
    x = torch.zeros(2, 3, device="cuda")
    v = torch.zeros(3, device="cuda")
    with pytest.raises(ImportError, match="Triton is not installed"):
        torch.ops.sj.multistep_if_inference(x, v, 1.0, 0.0, False, False)


def test_registration_failure_is_not_hidden_by_optional_import():
    env = os.environ.copy()
    env["SJ_USE_TRITON_OP"] = "0"
    script = textwrap.dedent(
        """
        import torch
        def fail_registration(*args, **kwargs):
            raise RuntimeError("registration sentinel")
        torch.library.custom_op = fail_registration
        import spikingjelly.activation_based.neuron
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "registration sentinel" in completed.stderr


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

    node = (
        _build_node(kind, "triton", surrogate.PiecewiseLeakyReLU(w=1.0, c=0.01))
        .cuda()
        .train()
    )
    x = torch.randn(6, 2, 9, device="cuda", requires_grad=True)

    with pytest.raises(NotImplementedError, match="PiecewiseLeakyReLU"):
        y = node(x)
        y.sum().backward()
