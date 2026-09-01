import gc
import os
from types import SimpleNamespace

import pytest
import torch

from spikingjelly import configure
from spikingjelly.activation_based import functional, surrogate
from spikingjelly.activation_based.cuda_kernel.cuda_utils import (
    register_python_object,
    resolve_python_object,
)
from spikingjelly.activation_based.cuda_kernel import spike_op
from spikingjelly.activation_based.cuda_kernel.spike_op import spike_linear
from spikingjelly.activation_based.cuda_kernel.tensor_cache import BoolTensorCache


@pytest.mark.parametrize(
    "surrogate_factory",
    [
        surrogate.Sigmoid,
        surrogate.ATan,
        surrogate.PiecewiseLeakyReLU,
        surrogate.LogTailedReLU,
    ],
)
@pytest.mark.parametrize(
    ("dtype", "input_size"),
    [("float", 1), ("half2", 2)],
    ids=["float", "half2"],
)
def test_surrogate_cuda_codes_compile(surrogate_factory, dtype, input_size):
    _require_cuda()
    cupy = pytest.importorskip("cupy")

    sg = surrogate_factory()
    codes = sg.cuda_codes(y=f"{dtype} result", x="over_th", dtype=dtype)
    source = f"""
    #include <cuda_fp16.h>
    extern "C" __global__ void generated(
        const {dtype}* x, {dtype}* y
    ) {{
        const int index = threadIdx.x;
        if (index == 0) {{
            {dtype} over_th = x[index];
            {codes}
            y[index] = result;
        }}
    }}
    """

    kernel = cupy.RawKernel(source, "generated")
    cupy_dtype = cupy.float32 if dtype == "float" else cupy.float16
    x = cupy.asarray([0.25] * input_size, dtype=cupy_dtype)
    y = cupy.empty_like(x)
    kernel((1,), (1,), (x, y))
    cupy.cuda.Stream.null.synchronize()


@pytest.mark.parametrize("dtype", ["float", "half2"])
def test_log_tailed_relu_cuda_codes_match_definition(dtype):
    _require_cuda()
    cupy = pytest.importorskip("cupy")

    alpha = 0.25
    values = [-0.5, 0.5, 1.0, 2.0, 4.0, -1.0]
    expected = [alpha, 1.0, 1.0, 0.5, 0.25, alpha]
    sg = surrogate.LogTailedReLU(alpha=alpha)
    codes = sg.cuda_codes(y=f"{dtype} result", x="over_th", dtype=dtype)
    source = f"""
    #include <cuda_fp16.h>
    extern "C" __global__ void generated(
        const {dtype}* x, {dtype}* y
    ) {{
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        {dtype} over_th = x[index];
        {codes}
        y[index] = result;
    }}
    """

    kernel = cupy.RawKernel(source, "generated")
    cupy_dtype = cupy.float32 if dtype == "float" else cupy.float16
    x = cupy.asarray(values, dtype=cupy_dtype)
    y = cupy.empty_like(x)
    element_count = len(values) // (2 if dtype == "half2" else 1)
    kernel((element_count,), (1,), (x, y))
    cupy.cuda.Stream.null.synchronize()
    cupy.testing.assert_allclose(
        y,
        cupy.asarray(expected, dtype=cupy_dtype),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("context_factory", [torch.no_grad, torch.inference_mode])
def test_capture_token_skips_context_stash_without_grad(context_factory):
    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.multi_step import (
        runtime,
    )

    input_tensor = torch.ones(1, requires_grad=True)
    captured_ctx = runtime._CapturedAutogradCtx()
    with runtime._CAPTURE_CTX_LOCK:
        before_ids = set(runtime._CAPTURE_CTX_BY_ID)

    with context_factory():
        token = runtime._capture_token(captured_ctx, (input_tensor,))

    assert token.device.type == "cpu"
    assert token.item() == -1
    with runtime._CAPTURE_CTX_LOCK:
        assert set(runtime._CAPTURE_CTX_BY_ID) == before_ids


def test_capture_context_setup_skips_fake_token():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from spikingjelly.activation_based.cuda_kernel.neuron_kernel.multi_step import (
        runtime,
    )

    with FakeTensorMode():
        token = torch.empty((), dtype=torch.int64)
    ctx = SimpleNamespace()

    runtime._setup_capture_ctx(ctx, (), (token,))

    assert ctx.captured is None


def test_raw_kernel_cache_reuses_compile_identity(monkeypatch):
    from spikingjelly.activation_based.cuda_kernel.auto_cuda import base

    calls = []

    def fake_raw_kernel(code, name, *, options, backend):
        kernel = object()
        calls.append((code, name, options, backend, kernel))
        return kernel

    class FakeCupy:
        RawKernel = staticmethod(fake_raw_kernel)

    monkeypatch.setattr(base, "cupy", FakeCupy)
    base._get_raw_kernel.cache_clear()
    try:
        first = base._get_raw_kernel("code", "kernel", ("-O3",), "nvrtc")
        second = base._get_raw_kernel("code", "kernel", ("-O3",), "nvrtc")
        different = base._get_raw_kernel("other", "kernel", ("-O3",), "nvrtc")
        different_options = base._get_raw_kernel(
            "code", "kernel", ("--use_fast_math",), "nvrtc"
        )
    finally:
        base._get_raw_kernel.cache_clear()

    assert first is second
    assert different is not first
    assert different_options is not first
    assert len(calls) == 3


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required.")


def _require_cupy():
    pytest.importorskip("cupy")


def _maybe_skip_custom_op_unavailable():
    if not all(
        hasattr(torch.library, name)
        for name in ("custom_op", "register_fake", "register_autograd")
    ):
        pytest.skip("torch.library custom_op/register_autograd are unavailable.")


def test_spike_convolution_reports_extension_build_error(monkeypatch):
    build_error = OSError("build failed")
    monkeypatch.setattr(spike_op, "cpp_wrapper", None)
    monkeypatch.setattr(spike_op, "cpp_wrapper_error", build_error)

    with pytest.raises(RuntimeError) as exc_info:
        spike_op._spike_conv_backward_common(None, None, None, None, None, None, 1, ())

    assert exc_info.value.__cause__ is build_error


def test_python_object_registry_uses_identity_and_releases_objects():
    class Kernel:
        pass

    first = Kernel()
    second = Kernel()
    first_id = register_python_object(first)

    assert register_python_object(first) == first_id
    assert register_python_object(second) != first_id
    assert resolve_python_object(first_id) is first

    del first
    gc.collect()
    with pytest.raises(RuntimeError, match="Unknown python object"):
        resolve_python_object(first_id)


@pytest.mark.parametrize("level", [0, 1])
def test_bool_tensor_cache_balances_repeated_stores(level, monkeypatch):
    monkeypatch.setattr(configure, "save_bool_spike_level", level)
    spike = (torch.arange(17) % 3 == 0).float()
    cache = BoolTensorCache()

    first_key = cache.store_bool(spike)
    second_key = cache.store_bool(spike)

    assert first_key == second_key
    assert torch.equal(cache.get_float(first_key, spike.shape), spike)
    assert torch.equal(cache.get_float(second_key, spike.shape), spike)


def test_spike_linear_backward_no_bias_cuda():
    _require_cuda()
    _require_cupy()
    _maybe_skip_custom_op_unavailable()

    x = (torch.randn(8, 32, device="cuda") > 0).float().requires_grad_(True)
    weight = torch.randn(16, 32, device="cuda", requires_grad=True)

    y = spike_linear(x, weight, None)
    loss = y.square().mean()
    loss.backward()

    assert y.shape == (8, 16)
    assert x.grad is not None
    assert weight.grad is not None


@pytest.mark.parametrize("kind", ["qif", "eif", "izhikevich"])
def test_nonlinear_functional_cupy_forward_backward(kind):
    _require_cuda()
    _require_cupy()
    _maybe_skip_custom_op_unavailable()

    sg = surrogate.ATan()
    x_seq = (torch.randn(4, 64, device="cuda") * 0.2).requires_grad_(True)
    v_init = torch.zeros(64, device="cuda", requires_grad=True)

    if kind == "qif":
        spike_seq, v_next, v_seq = functional.qif_multi_step_cupy(
            x_seq=x_seq,
            v=v_init,
            tau=2.5,
            v_threshold=0.9,
            v_reset=-0.3,
            v_rest=-0.2,
            v_c=0.4,
            a0=0.6,
            detach_reset=True,
            surrogate_function=sg,
            store_v_seq=True,
        )
    elif kind == "eif":
        spike_seq, v_next, v_seq = functional.eif_multi_step_cupy(
            x_seq=x_seq,
            v=v_init,
            tau=2.5,
            v_threshold=0.9,
            v_reset=-0.3,
            v_rest=-0.2,
            theta_rh=0.4,
            delta_t=0.7,
            detach_reset=True,
            surrogate_function=sg,
            store_v_seq=True,
        )
    else:
        w_init = torch.zeros(64, device="cuda", requires_grad=True)
        spike_seq, v_next, w_next, v_seq, w_seq = functional.izhikevich_multi_step_cupy(
            x_seq=x_seq,
            v=v_init,
            w=w_init,
            tau=2.5,
            v_threshold=0.9,
            v_reset=-0.3,
            v_rest=-0.2,
            a=0.1,
            b=0.2,
            tau_w=3.0,
            v_c=0.4,
            a0=0.6,
            detach_reset=True,
            surrogate_function=sg,
            store_state_seq=True,
        )

    if kind == "izhikevich":
        assert w_seq is not None
        torch.testing.assert_close(w_next, w_seq[-1])

    assert spike_seq.shape == x_seq.shape
    assert v_seq is not None
    torch.testing.assert_close(v_next, v_seq[-1])

    loss = spike_seq.mean() + v_seq.mean()
    if kind == "izhikevich":
        loss = loss + w_seq.mean()

    loss.backward()

    assert x_seq.grad is not None
    assert v_init.grad is not None
    if kind == "izhikevich":
        assert w_init.grad is not None


def test_disable_cupy_custom_op_env_fallback():
    _require_cuda()
    _require_cupy()

    old = os.environ.get("SJ_USE_CUPY_OP")
    os.environ["SJ_USE_CUPY_OP"] = "0"
    try:
        x = (torch.randn(2, 8, device="cuda") > 0).float().requires_grad_(True)
        w = torch.randn(4, 8, device="cuda", requires_grad=True)
        y = spike_linear(x, w, None)
        y.sum().backward()
        assert y.shape == (2, 4)
        assert x.grad is not None
        assert w.grad is not None
    finally:
        if old is None:
            del os.environ["SJ_USE_CUPY_OP"]
        else:
            os.environ["SJ_USE_CUPY_OP"] = old
