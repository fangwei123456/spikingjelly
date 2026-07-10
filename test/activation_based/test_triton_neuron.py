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
from spikingjelly.activation_based.triton_kernel.neuron_kernel import (
    utils as neuron_triton_utils,
)
from spikingjelly.activation_based.triton_kernel.fp8_capability import (
    supports_triton_fp8_neuron_backward,
    supports_triton_fp8_neuron_forward,
    triton_fp8_neuron_capability_report,
)
from spikingjelly.activation_based.triton_kernel import triton_utils
from spikingjelly.activation_based.neuron import integrate_and_fire as if_module
from spikingjelly.activation_based.neuron import lif as lif_module
from spikingjelly.activation_based.neuron import plif as plif_module
from spikingjelly.activation_based.cuda_kernel.auto_cuda import (
    neuron_kernel as ac_neuron_kernel,
)


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


def _call_mixed_precision_forward(
    kind,
    x,
    v,
    *,
    decay_input=True,
    v_reset=0.0,
    storage_dtype=torch.float32,
    compute_dtype="fp32",
    backward_compute_dtype="fp32",
    spike_dtype=torch.float32,
    save_intermediates=True,
    detach_reset=False,
    r_tau=None,
):
    if kind == "if":
        return if_triton_kernel.multistep_if_mp(
            x,
            v,
            v_threshold=1.0,
            v_reset=v_reset,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            backward_compute_dtype=backward_compute_dtype,
            spike_dtype=spike_dtype,
            save_intermediates=save_intermediates,
            detach_reset=detach_reset,
        )
    if kind == "lif":
        return lif_triton_kernel.multistep_lif_mp(
            x,
            v,
            decay_input=decay_input,
            tau=2.0,
            v_threshold=1.0,
            v_reset=v_reset,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            backward_compute_dtype=backward_compute_dtype,
            spike_dtype=spike_dtype,
            save_intermediates=save_intermediates,
            detach_reset=detach_reset,
        )
    if kind == "plif":
        if r_tau is None:
            r_tau = torch.tensor(0.5, device=x.device, dtype=torch.float32)
        return plif_triton_kernel.multistep_plif_mp(
            x,
            v,
            r_tau,
            decay_input=decay_input,
            v_threshold=1.0,
            v_reset=v_reset,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            backward_compute_dtype=backward_compute_dtype,
            spike_dtype=spike_dtype,
            save_intermediates=save_intermediates,
            detach_reset=detach_reset,
        )
    raise ValueError(f"Unsupported mixed-precision neuron kind: {kind}.")


def _call_mixed_precision_forward_with_plan(
    kind,
    x,
    v,
    plan,
    *,
    decay_input=True,
    v_reset=0.0,
    detach_reset=False,
    r_tau=None,
):
    if kind == "if":
        return if_triton_kernel.multistep_if_mp_with_plan(
            x,
            v,
            plan,
            v_threshold=1.0,
            v_reset=v_reset,
            detach_reset=detach_reset,
        )
    if kind == "lif":
        return lif_triton_kernel.multistep_lif_mp_with_plan(
            x,
            v,
            plan,
            decay_input=decay_input,
            tau=2.0,
            v_threshold=1.0,
            v_reset=v_reset,
            detach_reset=detach_reset,
        )
    if kind == "plif":
        if r_tau is None:
            r_tau = torch.tensor(0.5, device=x.device, dtype=torch.float32)
        return plif_triton_kernel.multistep_plif_mp_with_plan(
            x,
            v,
            r_tau,
            plan,
            decay_input=decay_input,
            v_threshold=1.0,
            v_reset=v_reset,
            detach_reset=detach_reset,
        )
    raise ValueError(f"Unsupported mixed-precision neuron kind: {kind}.")


def _fp8_storage_dtype_or_skip() -> torch.dtype:
    dtype_candidates = []
    if hasattr(torch, "float8_e5m2"):
        dtype_candidates.append(torch.float8_e5m2)
    if hasattr(torch, "float8_e4m3fn"):
        dtype_candidates.append(torch.float8_e4m3fn)
    for storage_dtype in dtype_candidates:
        if supports_triton_fp8_neuron_forward(
            storage_dtype, torch.device("cuda"), compute_dtype="fp32"
        ):
            return storage_dtype
    pytest.skip("No supported Triton FP8 storage dtype is available on this CUDA device.")


def _quantize_for_storage(x: torch.Tensor, storage_dtype: torch.dtype) -> torch.Tensor:
    return x.to(storage_dtype).to(torch.float32)


def _mixed_precision_reference(
    kind: str,
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    *,
    storage_dtype: torch.dtype,
    decay_input: bool = True,
    v_reset: float | None = 0.0,
    r_tau: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_seq = _quantize_for_storage(x_seq, storage_dtype)
    v = _quantize_for_storage(v_init, storage_dtype)
    s_seq = torch.empty_like(x_seq, dtype=torch.float32)
    v_seq = torch.empty_like(x_seq, dtype=torch.float32)
    h_seq = torch.empty_like(x_seq, dtype=torch.float32)
    reset_value = 0.0 if v_reset is None else v_reset
    for t in range(x_seq.shape[0]):
        x = x_seq[t]
        if kind == "if":
            h = v + x
        elif kind in ("lif", "plif"):
            if decay_input:
                h = v + r_tau * (reset_value - v + x)
            else:
                h = v + r_tau * (reset_value - v) + x
        else:
            raise ValueError(f"Unsupported mixed-precision neuron kind: {kind}.")
        s = (h >= 1.0).to(torch.float32)
        if v_reset is None:
            v = h - s
        else:
            v = s * reset_value + (1.0 - s) * h
        s_seq[t] = s
        v_seq[t] = _quantize_for_storage(v, storage_dtype)
        h_seq[t] = _quantize_for_storage(h, storage_dtype)
    return s_seq, v_seq, h_seq


def test_triton_fp8_capability_report_cpu_is_unavailable():
    report = triton_fp8_neuron_capability_report(torch.device("cpu"))
    assert report["device_type"] == "cpu"
    for dtype_report in report["dtypes"].values():
        assert dtype_report["available"] is False
        assert dtype_report["reason"]
        assert dtype_report["forward"]["available"] is False
        assert dtype_report["backward"]["available"] is False


def test_triton_fp8_capability_rejects_invalid_dtype():
    with pytest.raises(ValueError, match="Unsupported Triton FP8 dtype"):
        supports_triton_fp8_neuron_forward(torch.float32, torch.device("cpu"))
    with pytest.raises(ValueError, match="Unsupported Triton FP8 dtype"):
        supports_triton_fp8_neuron_backward(torch.float32, torch.device("cpu"))


def test_normalize_triton_compute_dtype_accepts_native_fp8_dtype():
    if hasattr(torch, "float8_e4m3fn"):
        assert (
            triton_utils.normalize_triton_compute_dtype_name(torch.float8_e4m3fn)
            == "fp8"
        )
    if hasattr(torch, "float8_e5m2"):
        assert (
            triton_utils.normalize_triton_compute_dtype_name(torch.float8_e5m2)
            == "fp8"
        )


def test_resolve_triton_compute_dtype_reports_unavailable_type_dict(monkeypatch):
    monkeypatch.setattr(triton_utils, "type_dict", {})
    with pytest.raises(ValueError, match="fp32 compute dtype is unavailable"):
        triton_utils.resolve_triton_compute_dtype("fp32")
    with pytest.raises(ValueError, match="fp16 compute dtype is unavailable"):
        triton_utils.resolve_triton_compute_dtype("fp16")


def test_triton_neuron_dtype_id_roundtrip():
    dtypes = [torch.float32, torch.float16]
    if hasattr(torch, "bfloat16"):
        dtypes.append(torch.bfloat16)
    if hasattr(torch, "float8_e4m3fn"):
        dtypes.append(torch.float8_e4m3fn)
    if hasattr(torch, "float8_e5m2"):
        dtypes.append(torch.float8_e5m2)
    for dtype in dtypes:
        dtype_id = triton_utils.torch_dtype_to_triton_neuron_dtype_id(dtype)
        assert triton_utils.triton_neuron_dtype_id_to_torch_dtype(dtype_id) == dtype

    with pytest.raises(ValueError, match="dtype id"):
        triton_utils.triton_neuron_dtype_id_to_torch_dtype(999)


def test_mixed_precision_uses_custom_op_not_autograd_function():
    assert not hasattr(if_triton_kernel, "_MixedPrecisionIF")
    assert not hasattr(lif_triton_kernel, "_MixedPrecisionLIF")
    assert not hasattr(plif_triton_kernel, "_MixedPrecisionPLIF")
    assert hasattr(if_triton_kernel, "multistep_if_mp_forward")
    assert hasattr(lif_triton_kernel, "multistep_lif_mp_forward")
    assert hasattr(plif_triton_kernel, "multistep_plif_mp_forward")
    assert if_triton_kernel.multistep_if_mixed_precision_forward is (
        if_triton_kernel.multistep_if_mp
    )
    assert lif_triton_kernel.multistep_lif_mixed_precision_forward is (
        lif_triton_kernel.multistep_lif_mp
    )
    assert plif_triton_kernel.multistep_plif_mixed_precision_forward is (
        plif_triton_kernel.multistep_plif_mp
    )


def test_fp8_backward_capability_uses_backward_probe(monkeypatch):
    storage_dtype = getattr(torch, "float8_e4m3fn", None)
    if storage_dtype is None:
        pytest.skip("This PyTorch build does not expose torch.float8_e4m3fn.")

    from spikingjelly.activation_based.triton_kernel import fp8_capability

    calls = {"forward": 0, "backward": 0}

    def _unexpected_forward(*args, **kwargs):
        del args, kwargs
        calls["forward"] += 1
        return {"available": True, "reason": None}

    def _backward_probe(*args, **kwargs):
        del args, kwargs
        calls["backward"] += 1
        return {"available": False, "reason": "backward probe failed"}

    monkeypatch.setattr(
        fp8_capability, "triton_fp8_neuron_capability", _unexpected_forward
    )
    monkeypatch.setattr(
        fp8_capability, "triton_fp8_neuron_backward_capability", _backward_probe
    )

    with pytest.raises(RuntimeError, match="backward probe failed"):
        neuron_triton_utils._check_fp8_backward_capability(
            storage_dtype, torch.device("cuda", 0), "fp32", "LIF"
        )
    assert calls == {"forward": 0, "backward": 1}


def test_prepare_triton_neuron_forward_plan_rejects_cpu_device():
    with pytest.raises(RuntimeError, match="CUDA device"):
        neuron_triton_utils.prepare_triton_neuron_forward_plan(
            neuron_type="lif",
            device="cpu",
            storage_dtype=torch.float32,
            compute_dtype="fp32",
        )


def test_prepare_triton_neuron_forward_plan_rejects_invalid_options():
    with pytest.raises(ValueError, match="neuron_type"):
        neuron_triton_utils.prepare_triton_neuron_forward_plan(
            neuron_type="bad",
            device="cpu",
            storage_dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="storage dtype"):
        neuron_triton_utils.prepare_triton_neuron_forward_plan(
            neuron_type="lif",
            device="cpu",
            storage_dtype=torch.int32,
        )
    with pytest.raises(ValueError, match="compute_dtype"):
        neuron_triton_utils.prepare_triton_neuron_forward_plan(
            neuron_type="lif",
            device="cpu",
            storage_dtype=torch.float32,
            compute_dtype="bad",
        )
    with pytest.raises(ValueError, match="requires an FP8 storage_dtype"):
        neuron_triton_utils.prepare_triton_neuron_forward_plan(
            neuron_type="lif",
            device="cpu",
            storage_dtype=torch.float32,
            compute_dtype="fp8",
        )
    with pytest.raises(ValueError, match="backward_compute_dtype"):
        neuron_triton_utils.prepare_triton_neuron_forward_plan(
            neuron_type="lif",
            device="cpu",
            storage_dtype=torch.float32,
            backward_compute_dtype="bad",
        )
    with pytest.raises(ValueError, match="backward_compute_dtype='fp8'"):
        neuron_triton_utils.prepare_triton_neuron_forward_plan(
            neuron_type="lif",
            device="cpu",
            storage_dtype=torch.float32,
            backward_compute_dtype="fp8",
        )


def test_triton_neuron_forward_plan_matches_config():
    plan = neuron_triton_utils.TritonNeuronExecutionPlan(
        neuron_type="lif",
        device=torch.device("cuda", 0),
        storage_dtype=torch.float32,
        forward_compute_dtype_name="fp32",
        forward_compute_tl_dtype=object(),
        backward_compute_dtype_name="fp32",
        backward_compute_tl_dtype=object(),
        spike_dtype=torch.float32,
        storage_dtype_id=0,
        forward_compute_dtype_id=0,
        backward_compute_dtype_id=0,
        spike_dtype_id=0,
        save_intermediates=True,
    )

    assert plan.matches(
        neuron_type="lif",
        device=torch.device("cuda", 0),
        storage_dtype="fp32",
        compute_dtype=torch.float32,
        backward_compute_dtype="fp32",
        spike_dtype=torch.float32,
        save_intermediates=True,
    )
    assert not plan.matches(
        neuron_type="if",
        device=torch.device("cuda", 0),
        storage_dtype="fp32",
        compute_dtype=torch.float32,
        backward_compute_dtype="fp32",
        spike_dtype=torch.float32,
        save_intermediates=True,
    )
    assert not plan.matches(
        neuron_type="lif",
        device=torch.device("cuda", 1),
        storage_dtype="fp32",
        compute_dtype=torch.float32,
        backward_compute_dtype="fp32",
        spike_dtype=torch.float32,
        save_intermediates=True,
    )
    assert not plan.matches(
        neuron_type="lif",
        device=torch.device("cuda", 0),
        storage_dtype=torch.float16,
        compute_dtype=torch.float32,
        backward_compute_dtype="fp32",
        spike_dtype=torch.float32,
        save_intermediates=True,
    )
    assert not plan.matches(
        neuron_type="lif",
        device=torch.device("cuda", 0),
        storage_dtype="fp32",
        compute_dtype="fp16",
        backward_compute_dtype="fp32",
        spike_dtype=torch.float32,
        save_intermediates=True,
    )
    assert not plan.matches(
        neuron_type="lif",
        device=torch.device("cuda", 0),
        storage_dtype="fp32",
        compute_dtype=torch.float32,
        backward_compute_dtype="fp16",
        spike_dtype=torch.float32,
        save_intermediates=True,
    )
    assert not plan.matches(
        neuron_type="lif",
        device=torch.device("cuda", 0),
        storage_dtype="fp32",
        compute_dtype=torch.float32,
        backward_compute_dtype="fp32",
        spike_dtype=torch.float16,
        save_intermediates=True,
    )
    assert not plan.matches(
        neuron_type="lif",
        device=torch.device("cuda", 0),
        storage_dtype="fp32",
        compute_dtype=torch.float32,
        backward_compute_dtype="fp32",
        spike_dtype=torch.float32,
        save_intermediates=False,
    )


@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
def test_mixed_precision_forward_rejects_invalid_options(kind):
    x = torch.randn(4, 2)
    v = torch.zeros(2)
    with pytest.raises(ValueError, match="storage dtype"):
        _call_mixed_precision_forward(
            kind,
            x,
            v,
            storage_dtype=torch.int32,
        )
    with pytest.raises(ValueError, match="compute_dtype"):
        _call_mixed_precision_forward(
            kind,
            x,
            v,
            storage_dtype=torch.float32,
            compute_dtype="bad",
        )


@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
def test_mixed_precision_forward_fp8_cpu_fails_with_capability_reason(kind):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("This PyTorch build does not expose torch.float8_e4m3fn.")
    x = torch.randn(4, 2)
    v = torch.zeros(2)
    with pytest.raises(RuntimeError, match="unavailable"):
        _call_mixed_precision_forward(
            kind,
            x,
            v,
            storage_dtype=torch.float8_e4m3fn,
        )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("node_factory", "kernel_attr"),
    [
        (
            lambda: neuron.IFNode(step_mode="m", backend="cupy", store_v_seq=True).to(
                "cuda"
            ),
            "multistep_if",
        ),
        (
            lambda: neuron.LIFNode(
                tau=2.0, step_mode="m", backend="cupy", store_v_seq=True
            ).to("cuda"),
            "multistep_lif",
        ),
    ],
)
def test_cupy_backend_uses_cupy_path_in_eval(node_factory, kernel_attr, monkeypatch):
    pytest.importorskip("cupy")
    hits = {"count": 0}
    original = getattr(ac_neuron_kernel, kernel_attr)

    def _wrapped(*args, **kwargs):
        hits["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ac_neuron_kernel, kernel_attr, _wrapped)
    node = node_factory().eval()
    x = torch.randn(6, 3, 10, device="cuda")
    node(x)
    assert hits["count"] > 0


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

    original_loop_mode = kernel_module.LAST_FORWARD_LOOP_MODE
    try:
        kernel_module.LAST_FORWARD_LOOP_MODE = None
        node(x)

        if backend == "triton":
            assert kernel_module.LAST_FORWARD_LOOP_MODE in ("static", "dynamic")
        else:
            assert kernel_module.LAST_FORWARD_LOOP_MODE is None
    finally:
        kernel_module.LAST_FORWARD_LOOP_MODE = original_loop_mode


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
@pytest.mark.parametrize(
    ("kind", "kernel_module"),
    [
        ("if", if_triton_kernel),
        ("lif", lif_triton_kernel),
        ("plif", plif_triton_kernel),
    ],
)
@pytest.mark.parametrize("T", [8, 16, 32])
@pytest.mark.parametrize("decay_input", [True, False])
@pytest.mark.parametrize("v_reset", [0.0, None])
def test_mixed_precision_float32_matches_torch_eval(
    kind, kernel_module, T, decay_input, v_reset
):
    pytest.importorskip("triton")
    original_threshold = configure.triton_neuron_kernel_static_range_max_T
    try:
        configure.triton_neuron_kernel_static_range_max_T = 16
        x = torch.randn(T, 4, 8, device="cuda", dtype=torch.float32)
        v_init = torch.zeros_like(x[0])
        if kind == "if":
            node = neuron.IFNode(
                v_threshold=1.0,
                v_reset=v_reset,
                step_mode="m",
                backend="torch",
                store_v_seq=True,
            ).to("cuda").eval()
            r_tau = None
        elif kind == "lif":
            node = neuron.LIFNode(
                tau=2.0,
                decay_input=decay_input,
                v_threshold=1.0,
                v_reset=v_reset,
                step_mode="m",
                backend="torch",
                store_v_seq=True,
            ).to("cuda").eval()
            r_tau = None
        elif kind == "plif":
            node = neuron.ParametricLIFNode(
                init_tau=2.0,
                decay_input=decay_input,
                v_threshold=1.0,
                v_reset=v_reset,
                step_mode="m",
                backend="torch",
                store_v_seq=True,
            ).to("cuda").eval()
            r_tau = node.w.sigmoid().to(x)
        else:
            raise ValueError(kind)

        original_loop_mode = kernel_module.LAST_FORWARD_LOOP_MODE
        try:
            kernel_module.LAST_FORWARD_LOOP_MODE = None
            plan = neuron_triton_utils.prepare_triton_neuron_forward_plan(
                neuron_type=kind,
                device=x.device,
                storage_dtype=torch.float32,
                compute_dtype="fp32",
                spike_dtype=torch.float32,
                save_intermediates=True,
            )
            with torch.no_grad():
                expected = node(x)
                spike, v_seq, h_seq = _call_mixed_precision_forward(
                    kind,
                    x,
                    v_init,
                    decay_input=decay_input,
                    v_reset=v_reset,
                    storage_dtype=torch.float32,
                    compute_dtype="fp32",
                    spike_dtype=torch.float32,
                    save_intermediates=True,
                    r_tau=r_tau,
                )
                plan_spike, plan_v_seq, plan_h_seq = (
                    _call_mixed_precision_forward_with_plan(
                        kind,
                        x,
                        v_init,
                        plan,
                        decay_input=decay_input,
                        v_reset=v_reset,
                        r_tau=r_tau,
                    )
                )
            _assert_close(expected, spike, torch.float32)
            _assert_close(node.v_seq, v_seq, torch.float32)
            assert h_seq is not None and h_seq.dtype == torch.float32
            _assert_close(spike, plan_spike, torch.float32)
            _assert_close(v_seq, plan_v_seq, torch.float32)
            assert plan_h_seq is not None and plan_h_seq.dtype == torch.float32
            _assert_close(h_seq, plan_h_seq, torch.float32)
            assert kernel_module.LAST_FORWARD_LOOP_MODE == (
                "static" if T <= 16 else "dynamic"
            )
        finally:
            kernel_module.LAST_FORWARD_LOOP_MODE = original_loop_mode
    finally:
        configure.triton_neuron_kernel_static_range_max_T = original_threshold


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
def test_mixed_precision_forward_with_plan_skips_repeated_preflight(kind, monkeypatch):
    x = torch.randn(8, 4, device="cuda", dtype=torch.float32)
    v_init = torch.zeros_like(x[0])
    r_tau = torch.tensor(0.5, device=x.device, dtype=torch.float32)
    plan = neuron_triton_utils.prepare_triton_neuron_forward_plan(
        neuron_type=kind,
        device=x.device,
        storage_dtype=torch.float32,
        compute_dtype="fp32",
        spike_dtype=torch.float32,
        save_intermediates=True,
    )

    def _unexpected_preflight(*args, **kwargs):
        del args, kwargs
        raise AssertionError("with-plan forward must not repeat preflight")

    monkeypatch.setattr(
        neuron_triton_utils,
        "normalize_triton_storage_dtype",
        _unexpected_preflight,
    )
    monkeypatch.setattr(
        neuron_triton_utils,
        "normalize_triton_compute_dtype_name",
        _unexpected_preflight,
    )
    monkeypatch.setattr(
        neuron_triton_utils,
        "resolve_triton_compute_dtype",
        _unexpected_preflight,
    )
    monkeypatch.setattr(
        neuron_triton_utils,
        "_check_fp8_forward_capability",
        _unexpected_preflight,
    )
    monkeypatch.setattr(
        neuron_triton_utils,
        "_check_fp8_backward_capability",
        _unexpected_preflight,
    )

    with torch.no_grad():
        for _ in range(3):
            spike, v_seq, h_seq = _call_mixed_precision_forward_with_plan(
                kind,
                x,
                v_init,
                plan,
                decay_input=True,
                v_reset=0.0,
                r_tau=r_tau,
            )
            assert spike.shape == x.shape
            assert v_seq.shape == x.shape
            assert h_seq is not None and h_seq.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
@pytest.mark.parametrize("use_plan", [False, True])
def test_mixed_precision_rejects_v_init_shape_mismatch(kind, use_plan):
    x = torch.randn(8, 2, 4, device="cuda", dtype=torch.float32)
    v_init = torch.zeros(4, device="cuda", dtype=torch.float32)
    r_tau = torch.tensor(0.5, device=x.device, dtype=torch.float32)
    plan = None
    if use_plan:
        plan = neuron_triton_utils.prepare_triton_neuron_forward_plan(
            neuron_type=kind,
            device=x.device,
            storage_dtype=torch.float32,
            compute_dtype="fp32",
            spike_dtype=torch.float32,
            save_intermediates=True,
        )

    with torch.no_grad(), pytest.raises(RuntimeError, match="v_init shape"):
        if use_plan:
            _call_mixed_precision_forward_with_plan(
                kind,
                x,
                v_init,
                plan,
                decay_input=True,
                v_reset=0.0,
                r_tau=r_tau,
            )
        else:
            _call_mixed_precision_forward(
                kind,
                x,
                v_init,
                decay_input=True,
                v_reset=0.0,
                storage_dtype=torch.float32,
                compute_dtype="fp32",
                spike_dtype=torch.float32,
                save_intermediates=True,
                r_tau=r_tau,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
def test_mixed_precision_backward_with_plan_skips_repeated_preflight(
    kind, monkeypatch
):
    x = torch.randn(8, 4, device="cuda", dtype=torch.float32)
    v_init = torch.zeros_like(x[0])
    r_tau = torch.tensor(0.5, device=x.device, dtype=torch.float32)
    plan = neuron_triton_utils.prepare_triton_neuron_forward_plan(
        neuron_type=kind,
        device=x.device,
        storage_dtype=torch.float32,
        compute_dtype="fp32",
        backward_compute_dtype="fp32",
        spike_dtype=torch.float32,
        save_intermediates=True,
    )

    def _unexpected_preflight(*args, **kwargs):
        del args, kwargs
        raise AssertionError("with-plan backward must not repeat preflight")

    monkeypatch.setattr(
        neuron_triton_utils,
        "normalize_triton_storage_dtype",
        _unexpected_preflight,
    )
    monkeypatch.setattr(
        neuron_triton_utils,
        "normalize_triton_compute_dtype_name",
        _unexpected_preflight,
    )
    monkeypatch.setattr(
        neuron_triton_utils,
        "resolve_triton_compute_dtype",
        _unexpected_preflight,
    )
    monkeypatch.setattr(
        neuron_triton_utils,
        "_check_fp8_forward_capability",
        _unexpected_preflight,
    )
    monkeypatch.setattr(
        neuron_triton_utils,
        "_check_fp8_backward_capability",
        _unexpected_preflight,
    )

    for _ in range(3):
        x_req = x.clone().requires_grad_()
        v_req = v_init.clone().requires_grad_()
        if kind == "plif":
            r_tau_req = r_tau.clone().requires_grad_()
        else:
            r_tau_req = r_tau
        spike, v_seq, h_seq = _call_mixed_precision_forward_with_plan(
            kind,
            x_req,
            v_req,
            plan,
            decay_input=True,
            v_reset=0.0,
            r_tau=r_tau_req,
        )
        assert h_seq is not None
        (spike.sum() + v_seq.float().sum() * 0.125).backward()
        assert torch.isfinite(x_req.grad).all()
        assert torch.isfinite(v_req.grad).all()
        if kind == "plif":
            assert torch.isfinite(r_tau_req.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
@pytest.mark.parametrize("v_reset", [0.0, None])
def test_mixed_precision_autograd_with_plan_matches_safe_wrapper(kind, v_reset):
    x = torch.randn(8, 4, device="cuda", dtype=torch.float32)
    v_init = torch.randn(4, device="cuda", dtype=torch.float32) * 0.1
    r_tau_safe = torch.tensor(0.5, device=x.device, dtype=torch.float32)
    r_tau_plan = torch.tensor(0.5, device=x.device, dtype=torch.float32)
    if kind == "plif":
        r_tau_safe.requires_grad_()
        r_tau_plan.requires_grad_()
    x_safe = x.clone().requires_grad_()
    x_plan = x.clone().requires_grad_()
    v_safe = v_init.clone().requires_grad_()
    v_plan = v_init.clone().requires_grad_()
    plan = neuron_triton_utils.prepare_triton_neuron_forward_plan(
        neuron_type=kind,
        device=x.device,
        storage_dtype=torch.float32,
        compute_dtype="fp32",
        backward_compute_dtype="fp32",
        spike_dtype=torch.float32,
        save_intermediates=True,
    )

    safe_out = _call_mixed_precision_forward(
        kind,
        x_safe,
        v_safe,
        decay_input=True,
        v_reset=v_reset,
        storage_dtype=torch.float32,
        compute_dtype="fp32",
        backward_compute_dtype="fp32",
        r_tau=r_tau_safe,
    )
    plan_out = _call_mixed_precision_forward_with_plan(
        kind,
        x_plan,
        v_plan,
        plan,
        decay_input=True,
        v_reset=v_reset,
        r_tau=r_tau_plan,
    )
    for safe_tensor, plan_tensor in zip(safe_out, plan_out):
        if safe_tensor is not None:
            torch.testing.assert_close(safe_tensor, plan_tensor, atol=0.0, rtol=0.0)

    safe_loss = safe_out[0].sum() + safe_out[1].float().sum() * 0.125
    plan_loss = plan_out[0].sum() + plan_out[1].float().sum() * 0.125
    safe_loss.backward()
    plan_loss.backward()
    torch.testing.assert_close(x_safe.grad, x_plan.grad, atol=0.0, rtol=0.0)
    torch.testing.assert_close(v_safe.grad, v_plan.grad, atol=0.0, rtol=0.0)
    if kind == "plif":
        torch.testing.assert_close(
            r_tau_safe.grad, r_tau_plan.grad, atol=0.0, rtol=0.0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
def test_mixed_precision_fp8_storage_backward_fp32_compute_produces_finite_grad(kind):
    storage_dtype = _fp8_storage_dtype_or_skip()
    x = torch.randn(8, 4, device="cuda", dtype=torch.float32).requires_grad_()
    v_init = torch.zeros(4, device="cuda", dtype=torch.float32).requires_grad_()
    r_tau = torch.tensor(0.5, device=x.device, dtype=torch.float32)
    if kind == "plif":
        r_tau.requires_grad_()
    spike, v_seq, h_seq = _call_mixed_precision_forward(
        kind,
        x,
        v_init,
        decay_input=True,
        v_reset=0.0,
        storage_dtype=storage_dtype,
        compute_dtype="fp32",
        backward_compute_dtype="fp32",
        spike_dtype=torch.float32,
        save_intermediates=False,
        r_tau=r_tau,
    )

    assert h_seq is None
    spike.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(v_init.grad).all()
    if kind == "plif":
        assert torch.isfinite(r_tau.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["if", "lif", "plif"])
@pytest.mark.parametrize("v_reset", [0.0, None])
def test_mixed_precision_fp8_storage_matches_quantized_reference(kind, v_reset):
    storage_dtype = _fp8_storage_dtype_or_skip()
    x = torch.tensor(
        [
            [0.25, 0.50, 0.75, 1.00],
            [0.50, -0.25, 0.25, -0.50],
            [0.25, 0.75, -0.50, 0.50],
            [-0.25, 0.50, 0.25, 0.75],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    v_init = torch.zeros_like(x[0])
    r_tau = torch.tensor(0.5, device=x.device, dtype=torch.float32)
    plan = neuron_triton_utils.prepare_triton_neuron_forward_plan(
        neuron_type=kind,
        device=x.device,
        storage_dtype=storage_dtype,
        compute_dtype="fp32",
        spike_dtype=torch.float32,
        save_intermediates=True,
    )

    with torch.no_grad():
        spike, v_seq, h_seq = _call_mixed_precision_forward(
            kind,
            x,
            v_init,
            decay_input=True,
            v_reset=v_reset,
            storage_dtype=storage_dtype,
            compute_dtype="fp32",
            spike_dtype=torch.float32,
            save_intermediates=True,
            r_tau=r_tau,
        )
        plan_spike, plan_v_seq, plan_h_seq = _call_mixed_precision_forward_with_plan(
            kind,
            x,
            v_init,
            plan,
            decay_input=True,
            v_reset=v_reset,
            r_tau=r_tau,
        )
    expected_s, expected_v, expected_h = _mixed_precision_reference(
        kind,
        x,
        v_init,
        storage_dtype=storage_dtype,
        decay_input=True,
        v_reset=v_reset,
        r_tau=float(r_tau.item()),
    )

    torch.testing.assert_close(spike, expected_s, atol=0.0, rtol=0.0)
    torch.testing.assert_close(plan_spike, expected_s, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        v_seq.to(torch.float32), expected_v, atol=0.0625, rtol=0.0
    )
    torch.testing.assert_close(
        plan_v_seq.to(torch.float32), expected_v, atol=0.0625, rtol=0.0
    )
    assert h_seq is not None
    assert plan_h_seq is not None
    torch.testing.assert_close(
        h_seq.to(torch.float32), expected_h, atol=0.0625, rtol=0.0
    )
    torch.testing.assert_close(
        plan_h_seq.to(torch.float32), expected_h, atol=0.0625, rtol=0.0
    )


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
            lambda T: neuron.IFNode(step_mode="m", backend="triton").to("cuda")(
                torch.randn(T, 2, 8, device="cuda")
            ),
        ),
        (
            lif_triton_kernel,
            lambda T: neuron.LIFNode(tau=2.0, step_mode="m", backend="triton").to(
                "cuda"
            )(torch.randn(T, 2, 8, device="cuda")),
        ),
        (
            plif_triton_kernel,
            lambda T: neuron.ParametricLIFNode(
                init_tau=2.0, step_mode="m", backend="triton"
            ).to("cuda")(torch.randn(T, 2, 8, device="cuda")),
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
            lambda: neuron.LIFNode(tau=2.0, step_mode="m", backend="triton").to("cuda"),
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
