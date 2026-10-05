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
                for provider in ("native", "triton", "cupy")
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
    s, voltage, _ = lif.lif(x, v)
    gradients = torch.autograd.grad(s.sum() + voltage.sum(), (x, v))
    assert all(torch.isfinite(g).all() for g in gradients)
    with FakeTensorMode():
        x = torch.empty(3, 7, device="cuda")
        v = torch.empty(7, device="cuda")
        outputs = lif.lif(x, v)
        assert all(t.shape == x.shape and t.device == x.device for t in outputs)


def test_cpu_compilation_restores_source_identity_after_caller_tag_change(monkeypatch):
    from spikingjelly._ops.lif import lif

    monkeypatch.setattr(torch.compiler.config, "cache_key_tag", "caller-cpu-tag")
    x = torch.full((3, 7), 0.3, requires_grad=True)
    v = torch.zeros(7, requires_grad=True)
    expected = lif(x, v)
    actual = torch.compile(lif, fullgraph=True)(x, v)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual[0].sum() + actual[1].sum(), (x, v)),
        torch.autograd.grad(expected[0].sum() + expected[1].sum(), (x, v)),
    )
    assert torch.compiler.config.cache_key_tag.startswith("caller-cpu-tag")
    assert "|sj-ops:sj_lif:cpu:" in torch.compiler.config.cache_key_tag


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compilation_keeps_triton_kernels_visible(family):
    # Each case specializes the same OpOverload.__call__ frame to a different op.
    torch.compiler.reset()
    package = importlib.import_module(f"spikingjelly._ops.{family}")
    device = torch.device("cuda", 0)
    selected = package._selection.diagnostics(device)["implementation"]
    if selected != "triton":
        pytest.skip("requires the family's CUDA implementation to be triton")
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
    expected = package._forward(*parameters)
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
    torch.testing.assert_close(actual, expected)
    if trainable:
        visible = 3 if family == "izhikevich" else 2
        torch.testing.assert_close(
            torch.autograd.grad(sum(t.sum() for t in actual[:visible]), inputs),
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
        from spikingjelly._ops.lif import lif, get_cuda_implementation
        from unittest.mock import patch

        torch.compiler.config.cache_key_tag = "caller-tag"
        provider = os.environ["SJ_LIF_CUDA_IMPLEMENTATION"]
        x = torch.full((3, 2, 5), 0.3, device="cuda", requires_grad=True)
        v = torch.zeros(2, 5, device="cuda", requires_grad=True)
        args = (x, v, 2.0, True, 1.0, 0.0, False, 4.0)
        reference = lif(*args)
        assert get_cuda_implementation(x.device)["implementation"] == provider
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
            if op.name() in ("sj_lif::native_forward", "sj_lif::cupy_forward"):
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
        if provider == "cupy":
            assert calls == ["sj_lif::cupy_forward"], calls
        elif provider == "cuda":
            assert calls == ["sj_lif::native_forward"], calls
        else:
            assert not calls, calls
        print(provider, "correct compiled provider", flush=True)
    """)
    for provider in ("cuda", "cupy", "triton", "cuda", "cupy"):
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
def test_cuda_compile_requires_warm_binding_in_fresh_process():
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import torch
        from spikingjelly._ops.lif import lif
        x = torch.ones(2, 11, device="cuda")
        v = torch.zeros(11, device="cuda")
        compiled = torch.compile(lif, fullgraph=True)
        try:
            compiled(x, v)
        except Exception as error:
            assert "Warm up the registered neuron" in str(error), str(error)
        else:
            raise AssertionError("cold CUDA capture must not use an unbound cache key")
        expected = lif(x, v)
        torch.testing.assert_close(compiled(x, v), expected)
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "SJ_LIF_CUDA_IMPLEMENTATION": "triton"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
