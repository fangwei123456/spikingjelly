"""The shared device operator is the public execution seam."""

import importlib

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


FAMILIES = (
    "if_",
    "lif",
    "plif",
    "qif",
    "eif",
    "izhikevich",
    "ilif",
    "activation_aware_if",
    "stbif",
)


@pytest.mark.parametrize("family", FAMILIES)
def test_forward_and_backward_have_explicit_device_kernels(family):
    package = importlib.import_module(f"spikingjelly._ops.{family}")
    name = package._forward.name()
    assert name == f"sj_{'if' if family == 'if_' else family}::forward"
    for device in ("CPU", "CUDA"):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(name, device)
        if family not in ("activation_aware_if", "stbif"):
            backward = name.replace("::forward", "::backward")
            assert torch._C._dispatch_has_kernel_for_dispatch_key(backward, device)
    registered = set(torch._C._dispatch_get_all_op_names())
    namespace = name.split("::")[0]
    for provider in ("cpu", "triton"):
        for direction in ("forward", "backward"):
            assert f"{namespace}::{provider}_{direction}" not in registered


def test_import_registers_all_families_without_loading_cuda_providers():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(f"""
        import importlib
        import sys
        import torch

        def forbidden(*args, **kwargs):
            raise AssertionError("operator registration must not initialize CUDA")

        torch.cuda._lazy_init = forbidden
        for family in {FAMILIES!r}:
            package = "spikingjelly._ops." + family
            module = importlib.import_module(package)
            for device in ("CPU", "CUDA"):
                assert torch._C._dispatch_has_kernel_for_dispatch_key(
                    module._forward.name(), device
                )
            assert not any(
                package + "." + provider in sys.modules
                for provider in ("native", "triton")
            )
        assert not torch.cuda.is_initialized()
    """)
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_cpu_and_fake_cuda_do_not_choose_a_cuda_provider(monkeypatch):
    from spikingjelly._ops import lif

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU/Fake execution must not select a CUDA provider")

    monkeypatch.setattr(lif._selection, "_select", forbidden)
    x = torch.full((3, 7), 0.3, requires_grad=True)
    v = torch.zeros(7, requires_grad=True)
    s, voltage, _ = lif._forward(x, v, 2.0, True, 1.0, 0.0, False, 4.0)
    gradients = torch.autograd.grad(s.sum() + voltage.sum(), (x, v))
    assert all(torch.isfinite(g).all() for g in gradients)
    with FakeTensorMode():
        x = torch.empty(3, 7, device="cuda")
        v = torch.empty(7, device="cuda")
        outputs = lif._forward(x, v, 2.0, True, 1.0, 0.0, False, 4.0)
        assert all(t.shape == x.shape and t.device == x.device for t in outputs)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("compile_first", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compilation_keeps_triton_kernels_visible(family, compile_first, monkeypatch):
    # Each case specializes the same OpOverload.__call__ frame to a different op.
    torch.compiler.reset()
    package = importlib.import_module(f"spikingjelly._ops.{family}")
    monkeypatch.setattr(package._selection, "_selections", {})
    monkeypatch.setattr(package._selection, "_compiled_selections", {})
    device = torch.device("cuda", 0)
    trainable = family not in ("activation_aware_if", "stbif")
    x = torch.full((3, 33), 0.3, device=device, requires_grad=trainable)
    v = torch.zeros(33, device=device, requires_grad=trainable)
    w = torch.zeros(
        () if family == "plif" else (33,), device=device, requires_grad=trainable
    )
    th, off, pos, neg = [torch.tensor(t, device=device) for t in (1.0, 0.0, 3.0, -3.0)]
    parameters = {
        "if_": (x, v, 1.0, 0.0, True, 2.0, False, 1),
        "lif": (x, v, 2.3, True, 1.0, 0.0, True, 2.0, False, 1),
        "plif": (x, v, w, True, 1.0, 0.0, True, 2.0, False, 1),
        "qif": (x, v, 2.3, -0.2, 0.8, 0.4, 1.0, 0.0, True, 2.0, False, 1),
        "eif": (x, v, 2.3, -0.2, 0.9, 0.7, 1.0, 0.0, True, 2.0, False, 1),
        "izhikevich": (
            x,
            v,
            w,
            2.3,
            -0.2,
            0.8,
            0.4,
            0.2,
            0.3,
            3.1,
            1.0,
            0.0,
            True,
            2.0,
            False,
            1,
        ),
        "ilif": (x, v, 2.3, 4.0, 0.0, 4.0, 1.0, True, False),
        "activation_aware_if": (x, v, th, off, 1, 1, None, False),
        "stbif": (x, v, w, th, pos, neg),
    }[family]
    inputs = (x, v, w) if family in ("plif", "izhikevich") else (x, v)
    expected = package._selection._cpu_forward(*parameters)
    eager = None if compile_first else package._forward(*parameters)
    selected = package._selection.diagnostics(device, execution="compile")[
        "implementation"
    ]
    if selected != "triton":
        pytest.skip("requires the family's compiled implementation to be triton")
    captured = []
    from torch._dynamo.backends.common import aot_autograd
    from torch._functorch.aot_autograd import make_boxed_func

    def compiler(module, inputs):
        captured.append(module)
        return make_boxed_func(module.forward)

    compiled = torch.compile(
        package._forward,
        backend=aot_autograd(fw_compiler=compiler, bw_compiler=compiler),
        fullgraph=True,
    )
    actual = compiled(*parameters)
    if eager is None:
        eager = package._forward(*parameters)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(eager, expected)
    if trainable:
        visible = 3 if family == "izhikevich" else 2
        torch.testing.assert_close(
            torch.autograd.grad(sum(t.sum() for t in actual[:visible]), inputs),
            torch.autograd.grad(
                sum(t.sum() for t in expected[:visible]), inputs, retain_graph=True
            ),
        )
        torch.testing.assert_close(
            torch.autograd.grad(sum(t.sum() for t in eager[:visible]), inputs),
            torch.autograd.grad(sum(t.sum() for t in expected[:visible]), inputs),
        )
    assert len(captured) == (2 if trainable else 1)
    for module in captured:
        targets = [str(node.target) for node in module.graph.nodes]
        assert any("triton_kernel_wrapper" in target for target in targets), targets
        assert not any("sj_" in target for target in targets), targets
    namespace = package._forward.name().split("::")[0]
    assert not any(
        name.startswith((f"{namespace}::triton_", f"{namespace}::cpu_"))
        for name in torch._C._dispatch_get_all_op_names()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiler_cache_distinguishes_cuda_implementations(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    try:
        from spikingjelly._ops.lif.native import _build_info
        from spikingjelly._ops.native_loader import _check_native_device

        _check_native_device(_build_info, 0)
    except (ImportError, OSError) as error:
        pytest.skip(f"compatible native build required: {error}")

    # These processes deliberately share AOT/Inductor cache files and one graph.
    code = textwrap.dedent("""
        import os
        import torch
        from spikingjelly._ops.lif import _forward as lif
        from spikingjelly.activation_based.functional import neuron_implementation
        from unittest.mock import patch

        torch.compiler.config.cache_key_tag = "caller-tag"
        provider = os.environ["SJ_LIF_CUDA_IMPLEMENTATION"]
        x = torch.full((3, 2, 5), 0.3, device="cuda", requires_grad=True)
        v = torch.zeros(2, 5, device="cuda", requires_grad=True)
        args = (x, v, 2.0, True, 1.0, 0.0, False, 4.0)
        reference = lif(*args)
        assert neuron_implementation("lif", x.device)["implementation"] == provider
        bound_tag = torch.compiler.config.cache_key_tag
        lif(*args)
        assert torch.compiler.config.cache_key_tag == bound_tag
        assert "sj-ops:sj_lif" in bound_tag
        # User-provided compile scopes must keep the provider fingerprint too.
        torch.compiler.config.cache_key_tag = "caller-tag"
        compiled = torch.compile(lif, fullgraph=True)
        compiled(*args)
        calls = []
        original_call = torch._ops.OpOverload.__call__
        def observe(op, *args, **kwargs):
            if op.name() in ("sj_lif::native_forward",):
                calls.append(op.name())
            return original_call(op, *args, **kwargs)
        with patch.object(torch._ops.OpOverload, "__call__", observe):
            output = compiled(*args)
        torch.testing.assert_close(output, reference)
        torch.testing.assert_close(
            torch.autograd.grad(output[0].sum() + output[1].sum(), (x, v)),
            torch.autograd.grad(reference[0].sum() + reference[1].sum(), (x, v)),
        )
        assert torch.compiler.config.cache_key_tag.startswith("caller-tag")
        if provider == "cuda":
            assert calls == ["sj_lif::native_forward"], calls
        else:
            assert not calls, calls
        print(provider, "correct compiled provider", flush=True)
    """)
    for provider in ("cuda", "triton", "cuda"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={
                **os.environ,
                "SJ_LIF_CUDA_IMPLEMENTATION": provider,
                "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
                "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
                "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
            },
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fullgraph_compile_binds_provider_on_first_call():
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import torch
        from spikingjelly._ops.lif import _forward as lif
        from spikingjelly._ops.lif.reference import _forward_impl

        x = torch.full((2, 11), 0.3, device="cuda", requires_grad=True)
        v = torch.zeros(11, device="cuda", requires_grad=True)
        args = (x, v, 2.0, True, 1.0, 0.0, False, 4.0, False)
        expected = _forward_impl(x, v, 2.0, True, 1.0, 0.0, False, 4.0, False, 0)
        compiled = torch.compile(lif, fullgraph=True)
        actual = compiled(*args)
        for got, want in zip(actual, expected, strict=True):
            if got is not None:
                torch.testing.assert_close(got, want)
        got_grads = torch.autograd.grad(
            actual[0].sum() + actual[1].sum(), (x, v)
        )
        want_grads = torch.autograd.grad(
            expected[0].sum() + expected[1].sum(), (x, v)
        )
        torch.testing.assert_close(got_grads, want_grads)
        from spikingjelly.activation_based.functional import neuron_implementation

        assert neuron_implementation("lif", x.device)["implementation"] == "triton"
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "SJ_LIF_CUDA_IMPLEMENTATION": "triton"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("family", ["if", "lif", "plif"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_low_precision_state_reference_fullgraph(family, dtype):
    from spikingjelly.activation_based import functional, surrogate

    torch.compiler.reset()
    torch.manual_seed(11)
    x = (torch.randn(4, 33, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    v = torch.zeros(33, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.zeros((), device="cuda", requires_grad=True)
    sg = surrogate.ATan()
    op = getattr(functional, f"{family}_multi_step")

    def run(x, v, w):
        if family == "if":
            return op(x, v, surrogate_function=sg)[:2]
        if family == "lif":
            return op(x, v, 2.0, surrogate_function=sg)[:2]
        return op(x, v, w, surrogate_function=sg)[:2]

    reference = importlib.import_module(
        f"spikingjelly._ops.{'if_' if family == 'if' else family}.reference"
    ).multi_step

    def run_reference(x, v, w):
        if family == "if":
            return reference(x, v, 1.0, 0.0, sg, False)[:2]
        if family == "lif":
            return reference(x, v, 2.0, True, 1.0, 0.0, sg, False)[:2]
        return reference(x, v, w, True, 1.0, 0.0, sg, False)[:2]

    # Compile the first call too: the fallback has no device-selection work to do.
    actual = torch.compile(run, fullgraph=True)(x, v, w)
    # Inductor fuses low-precision arithmetic; compare the same compiler policy.
    expected = torch.compile(run_reference, fullgraph=True)(x, v, w)
    inputs = (x, v, w) if family == "plif" else (x, v)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(sum(t.sum() for t in actual), inputs),
        torch.autograd.grad(sum(t.sum() for t in expected), inputs),
    )
    assert actual[1].dtype == dtype


@pytest.mark.parametrize("family", ["if", "lif", "plif"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_explicit_precision_node_fullgraph(family):
    from spikingjelly.activation_based import neuron, surrogate
    from spikingjelly.activation_based.precision import (
        PrecisionConfig,
        prepare_model_for_precision,
    )

    torch.compiler.reset()
    node_class = {
        "if": neuron.IFNode,
        "lif": neuron.LIFNode,
        "plif": neuron.ParametricLIFNode,
    }[family]
    model = node_class(step_mode="m", surrogate_function=surrogate.ATan()).cuda()
    prepare_model_for_precision(
        model, "cuda:0", PrecisionConfig(mode="bf16", neuron_storage="fp32")
    )
    x = torch.randn(4, 33, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected = model(x)
    model.reset()
    actual = torch.compile(model, fullgraph=True)(x)
    torch.testing.assert_close(actual, expected)
    inputs = (x, *model.parameters())
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), inputs),
        torch.autograd.grad(expected.sum(), inputs),
    )
    assert model.v.dtype == torch.float32


def test_offline_priorities_bind_per_device_and_strict_override(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from spikingjelly._ops import selection
    from spikingjelly.activation_based import functional

    assert (
        functional.neuron_implementation(
            "lif", torch.device("cpu"), execution="compile"
        )["implementation"]
        == "torch-reference"
    )
    with pytest.raises(ValueError, match="execution"):
        functional.neuron_implementation("lif", torch.device("cpu"), execution="other")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, index))
    monkeypatch.setattr(torch.cuda, "device", lambda index: nullcontext())
    monkeypatch.setattr(
        selection,
        "_CUDA_PRIORITIES",
        {
            (8, 0): {"sj_lif": ("triton", "torch")},
            (8, 1): {"sj_lif": ("torch", "triton")},
        },
    )
    module = SimpleNamespace(
        _forward_impl=lambda *args: None, _backward_impl=lambda *args: None
    )
    monkeypatch.setattr(selection.importlib, "import_module", lambda *args: module)
    monkeypatch.delenv("SJ_LIF_CUDA_IMPLEMENTATION", raising=False)
    selector = selection._CudaSelection(
        "test", "sj_lif", "SJ_LIF_CUDA_IMPLEMENTATION", module._forward_impl
    )
    assert selector.diagnostics(torch.device("cuda", 0))["implementation"] == "triton"
    assert selector.diagnostics(torch.device("cuda", 1))["implementation"] == "torch"
    assert (
        selector.diagnostics(torch.device("cuda", 1), execution="compile")[
            "implementation"
        ]
        == "triton"
    )
    assert (
        selector.get_trace_forward(torch.device("cuda", 1)).func is module._forward_impl
    )
    assert (
        selector.get_trace_backward(torch.device("cuda", 1)).func
        is module._backward_impl
    )
    assert selector.get_cuda_backward(torch.device("cuda", 1)) is module._backward_impl
    # Changing the table cannot silently rebind a device after first use.
    selection._CUDA_PRIORITIES[(8, 0)]["sj_lif"] = ("torch", "triton")
    assert selector.diagnostics(torch.device("cuda", 0))["implementation"] == "triton"
    monkeypatch.setenv("SJ_LIF_CUDA_IMPLEMENTATION", "torch")
    forced = selection._CudaSelection(
        "test", "sj_lif", "SJ_LIF_CUDA_IMPLEMENTATION", module._forward_impl
    )
    assert forced.diagnostics(torch.device("cuda", 0))["implementation"] == "torch"
    assert (
        forced.diagnostics(torch.device("cuda", 0), execution="compile")[
            "implementation"
        ]
        == "torch"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lif_final_state_triton_backward_matches_reference(
    monkeypatch, compiled, dtype
):
    from spikingjelly.activation_based import functional, surrogate
    from spikingjelly._ops import lif

    torch.compiler.reset()
    monkeypatch.setattr(lif._selection, "_requested", "triton")
    monkeypatch.setattr(lif._selection, "_selections", {})
    monkeypatch.setattr(lif._selection, "_compiled_selections", {})
    torch.manual_seed(31)
    x = torch.rand(4, 2, 8, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.zeros(2, 8, device="cuda", requires_grad=True)
    sg = surrogate.ATan()

    def forward(x, v):
        return functional.lif_multi_step(x, v, surrogate_function=sg)

    run = torch.compile(forward, fullgraph=True) if compiled else forward
    actual = run(x, v)
    spike, final, _ = lif.reference.multi_step(
        x.float(), v, 2.0, True, 1.0, 0.0, sg, False
    )
    expected = (spike.to(dtype), final)
    assert actual[2] is None
    for got, want in zip(actual[:2], expected):
        torch.testing.assert_close(got, want)
    weights = tuple(torch.randn_like(tensor) for tensor in actual[:2])
    losses = [
        sum((value * weight).sum() for value, weight in zip(outputs, weights))
        for outputs in (actual[:2], expected)
    ]
    got = torch.autograd.grad(losses[0], (x, v))
    want = torch.autograd.grad(losses[1], (x, v))
    for actual_grad, expected_grad in zip(got, want):
        tolerance = (
            0.015
            if actual_grad.dtype == torch.bfloat16
            else 0.002
            if actual_grad.dtype == torch.float16
            else 2e-5
        )
        torch.testing.assert_close(
            actual_grad, expected_grad, rtol=tolerance, atol=tolerance
        )
