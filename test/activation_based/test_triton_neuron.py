import pytest
import torch

import spikingjelly.configure as configure
from spikingjelly.activation_based import base as activation_base
from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.triton_kernel.neuron_kernel import (
    integrate_and_fire as if_triton_kernel,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel import (
    lif as lif_triton_kernel,
)
from spikingjelly.activation_based.triton_kernel.neuron_kernel import (
    plif as plif_triton_kernel,
)
from spikingjelly.activation_based.neuron import integrate_and_fire as if_module
from spikingjelly.activation_based.neuron import lif as lif_module
from spikingjelly.activation_based.neuron import plif as plif_module


def _cupy_available() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def _assert_close(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype):
    if dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-6, 1e-6
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("node_factory", "module_obj", "kernel_attr"),
    [
        (
            lambda backend: neuron.IFNode(step_mode="m", backend=backend).eval(),
            if_module,
            "multistep_if",
        ),
        (
            lambda backend: neuron.LIFNode(
                tau=2.0, step_mode="m", backend=backend
            ).eval(),
            lif_module,
            "multistep_lif",
        ),
        (
            lambda backend: neuron.ParametricLIFNode(
                init_tau=2.0, step_mode="m", backend=backend
            ).eval(),
            plif_module,
            "multistep_plif",
        ),
    ],
)
def test_torch_backend_does_not_probe_triton_in_eval(
    node_factory, module_obj, kernel_attr, monkeypatch
):
    if getattr(module_obj, "triton_kernel", None) is None:
        pytest.skip("Triton module import is unavailable in this environment.")

    def _unexpected(*args, **kwargs):
        raise AssertionError("non-triton backend should not call Triton kernel")

    monkeypatch.setattr(module_obj.triton_kernel, kernel_attr, _unexpected)
    x = torch.randn(5, 2, 4)

    node_factory("torch")(x)


def test_lif_torch_backend_does_not_probe_triton_in_training(monkeypatch):
    if getattr(lif_module, "triton_kernel", None) is None:
        pytest.skip("Triton module import is unavailable in this environment.")

    def _unexpected(*args, **kwargs):
        raise AssertionError("torch backend should not call Triton kernel in training")

    monkeypatch.setattr(lif_module.triton_kernel, "multistep_lif", _unexpected)
    node = neuron.LIFNode(tau=2.0, step_mode="m", backend="torch").train()
    x = torch.randn(5, 2, 4)
    node(x)


def test_triton_backend_rejects_non_spiking_surrogate_in_eval():
    if activation_base.triton is None:
        pytest.skip("Triton module import is unavailable in this environment.")

    x = torch.randn(5, 2, 4)

    if_node = neuron.IFNode(
        step_mode="m",
        backend="triton",
        surrogate_function=surrogate.Sigmoid(spiking=False),
    ).eval()
    with pytest.raises(NotImplementedError, match="spiking surrogate functions"):
        if_node(x)

    lif_node = neuron.LIFNode(
        tau=2.0,
        step_mode="m",
        backend="triton",
        surrogate_function=surrogate.Sigmoid(spiking=False),
    ).eval()
    with pytest.raises(NotImplementedError, match="spiking surrogate functions"):
        lif_node(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("kind", "backend", "kernel_module"),
    [
        ("if", "torch", if_triton_kernel),
        ("if", "cupy", if_triton_kernel),
        ("if", "triton", if_triton_kernel),
        ("lif", "torch", lif_triton_kernel),
        ("lif", "cupy", lif_triton_kernel),
        ("lif", "triton", lif_triton_kernel),
        ("plif", "torch", plif_triton_kernel),
        ("plif", "cupy", plif_triton_kernel),
        ("plif", "triton", plif_triton_kernel),
    ],
)
def test_eval_backend_respects_triton_selection(kind, backend, kernel_module):
    if backend == "cupy":
        pytest.importorskip("cupy")

    x = torch.randn(9, 4, 16, device="cuda")

    if kind == "if":
        node = neuron.IFNode(step_mode="m", backend=backend, store_v_seq=True).eval()
    elif kind == "lif":
        node = neuron.LIFNode(
            tau=2.0, step_mode="m", backend=backend, store_v_seq=True
        ).eval()
    elif kind == "plif":
        node = neuron.ParametricLIFNode(
            init_tau=2.0, step_mode="m", backend=backend, store_v_seq=True
        ).eval()
    else:
        raise ValueError(kind)

    kernel_module.LAST_FORWARD_LOOP_MODE = None
    node(x)

    if backend == "triton":
        assert kernel_module.LAST_FORWARD_LOOP_MODE in ("static", "dynamic")
    else:
        assert kernel_module.LAST_FORWARD_LOOP_MODE is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("v_threshold", [1.0, 0.5])
@pytest.mark.parametrize("v_reset", [0.0, -0.2])
def test_if_triton_matches_torch_eval(v_threshold, v_reset):
    if_node = neuron.IFNode(
        v_threshold=v_threshold,
        v_reset=v_reset,
        step_mode="m",
        backend="torch",
        store_v_seq=True,
    ).eval()
    if_triton = neuron.IFNode(
        v_threshold=v_threshold,
        v_reset=v_reset,
        step_mode="m",
        backend="triton",
        store_v_seq=True,
    ).eval()

    x = torch.randn(32, 128, device="cuda", dtype=torch.float32)
    out_torch = if_node(x)
    out_triton = if_triton(x)

    _assert_close(out_torch, out_triton, torch.float32)
    _assert_close(if_node.v_seq, if_triton.v_seq, torch.float32)
    _assert_close(if_node.v, if_triton.v, torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tau", [2.0, 5.0, 10.0])
@pytest.mark.parametrize("detach_reset", [True, False])
@pytest.mark.parametrize("v_threshold", [1.0, 0.5])
@pytest.mark.parametrize("v_reset", [0.0, -0.2])
def test_lif_triton_matches_torch_training(tau, detach_reset, v_threshold, v_reset):
    lif = neuron.LIFNode(
        tau,
        detach_reset=detach_reset,
        v_threshold=v_threshold,
        v_reset=v_reset,
        step_mode="m",
        backend="torch",
    ).to(device="cuda", dtype=torch.float32)
    lif_triton = neuron.LIFNode(
        tau,
        detach_reset=detach_reset,
        v_threshold=v_threshold,
        v_reset=v_reset,
        step_mode="m",
        backend="triton",
    ).to(device="cuda", dtype=torch.float32)

    x = torch.randn(32, 128, device="cuda", dtype=torch.float32)
    x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
    out1 = lif(x1)
    out2 = lif_triton(x2)
    _assert_close(out1, out2, torch.float32)

    out1.sum().backward()
    out2.sum().backward()
    _assert_close(x1.grad, x2.grad, torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("decay_input", [True, False])
@pytest.mark.parametrize("detach_reset", [True, False])
@pytest.mark.parametrize("v_threshold", [1.0, 0.5])
@pytest.mark.parametrize("v_reset", [0.0, -0.2])
def test_plif_triton_matches_torch_training(
    decay_input, detach_reset, v_threshold, v_reset
):
    lif = neuron.ParametricLIFNode(
        decay_input=decay_input,
        detach_reset=detach_reset,
        v_threshold=v_threshold,
        v_reset=v_reset,
        step_mode="m",
        backend="torch",
    ).to(device="cuda", dtype=torch.float32)
    lif_triton = neuron.ParametricLIFNode(
        decay_input=decay_input,
        detach_reset=detach_reset,
        v_threshold=v_threshold,
        v_reset=v_reset,
        step_mode="m",
        backend="triton",
    ).to(device="cuda", dtype=torch.float32)

    x = torch.randn(32, 128, device="cuda", dtype=torch.float32)
    x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
    out1 = lif(x1)
    out2 = lif_triton(x2)
    _assert_close(out1, out2, torch.float32)

    out1.sum().backward()
    out2.sum().backward()
    _assert_close(x1.grad, x2.grad, torch.float32)
    _assert_close(lif.w.grad, lif_triton.w.grad, torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("kernel_module", "runner"),
    [
        (
            if_triton_kernel,
            lambda T: neuron.IFNode(step_mode="m", backend="triton")
            .to("cuda")(torch.randn(T, 2, 8, device="cuda")),
        ),
        (
            lif_triton_kernel,
            lambda T: neuron.LIFNode(tau=2.0, step_mode="m", backend="triton")
            .to("cuda")(torch.randn(T, 2, 8, device="cuda")),
        ),
        (
            plif_triton_kernel,
            lambda T: neuron.ParametricLIFNode(
                init_tau=2.0, step_mode="m", backend="triton"
            )
            .to("cuda")(torch.randn(T, 2, 8, device="cuda")),
        ),
    ],
)
def test_triton_loop_mode_switches_for_large_T(kernel_module, runner):
    original_threshold = configure.triton_neuron_kernel_static_range_max_T
    try:
        configure.triton_neuron_kernel_static_range_max_T = 16
        kernel_module.LAST_FORWARD_LOOP_MODE = None
        runner(8)
        assert kernel_module.LAST_FORWARD_LOOP_MODE == "static"

        kernel_module.LAST_FORWARD_LOOP_MODE = None
        runner(32)
        assert kernel_module.LAST_FORWARD_LOOP_MODE == "dynamic"
    finally:
        configure.triton_neuron_kernel_static_range_max_T = original_threshold


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("kernel_module", "node_factory"),
    [
        (
            if_triton_kernel,
            lambda: neuron.IFNode(step_mode="m", backend="triton").to("cuda"),
        ),
        (
            lif_triton_kernel,
            lambda: neuron.LIFNode(tau=2.0, step_mode="m", backend="triton").to(
                "cuda"
            ),
        ),
        (
            plif_triton_kernel,
            lambda: neuron.ParametricLIFNode(
                init_tau=2.0, step_mode="m", backend="triton"
            ).to("cuda"),
        ),
    ],
)
def test_triton_backward_loop_mode_switches_for_large_T(kernel_module, node_factory):
    original_threshold = configure.triton_neuron_kernel_static_range_max_T
    try:
        configure.triton_neuron_kernel_static_range_max_T = 16
        node = node_factory()
        x = torch.randn(32, 2, 8, device="cuda", requires_grad=True)
        kernel_module.LAST_BACKWARD_LOOP_MODE = None
        node(x).sum().backward()
        assert kernel_module.LAST_BACKWARD_LOOP_MODE == "dynamic"
    finally:
        configure.triton_neuron_kernel_static_range_max_T = original_threshold


if __name__ == "__main__":
    pytest.main([__file__])
