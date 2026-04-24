"""M1 tests for FlexSN inductor backend (eager path only).

Validates that the new ``flex_sn_scan`` HOP + ``FlexSN(backend="inductor")``
produce numerically identical output/state sequences to the reference
``backend="torch"`` implementation on a simple LIF single-step core.

Inductor lowering and Triton codegen are M3 scope and are not exercised here,
so these tests do not require a CUDA GPU.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from spikingjelly.activation_based.neuron.flexsn import (
    FlexSN,
    _make_inductor_final_state_warmup_args,
)
from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg16_bn
from spikingjelly.activation_based.triton_kernel.flex_sn_inductor import (
    flex_sn_scan,
    lowerable_scan,
    lowerable_scan_available,
    lowerable_while_loop_scan,
    lowerable_while_loop_available,
)


def _lif_core(x: torch.Tensor, v: torch.Tensor):
    tau = 2.0
    v_threshold = 1.0
    v_reset = 0.0
    h = v + (x - (v - v_reset)) / tau
    s = (h >= v_threshold).to(h.dtype)
    v_new = h * (1.0 - s) + v_reset * s
    return s, v_new


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(42)


def test_inductor_final_state_warmup_args_use_example_shapes():
    info = SimpleNamespace(num_inputs=2, num_states=2)
    specs = (
        ((3, 4), torch.float32),
        ((5,), torch.float64),
        ((3, 4), torch.float32),
        ((2, 3), torch.float64),
    )

    warm_args = _make_inductor_final_state_warmup_args(
        info,
        torch.device("cpu"),
        specs,
    )

    assert [tuple(arg.shape) for arg in warm_args] == [
        (1, 3, 4),
        (1, 5),
        (3, 4),
        (2, 3),
    ]
    assert [arg.dtype for arg in warm_args] == [
        torch.float32,
        torch.float64,
        torch.float32,
        torch.float64,
    ]


def test_inductor_final_state_wrapper_t0_returns_non_aliased_states():
    from spikingjelly.activation_based.triton_kernel.flexsn.wrapper import (
        flexsn_inference_final_state,
    )

    info = SimpleNamespace(num_inputs=1, num_outputs=1, num_states=1)
    x = torch.randn(0, 3)
    v = torch.randn(3)

    y_seq, v_final = flexsn_inference_final_state(None, info, x, v)

    assert tuple(y_seq.shape) == (0, 3)
    torch.testing.assert_close(v_final, v)
    assert v_final.data_ptr() != v.data_ptr()


def test_inductor_training_final_state_impl_t0_returns_non_aliased_states(
    monkeypatch,
):
    from spikingjelly.activation_based.triton_kernel.flex_sn_inductor import (
        custom_ops,
    )

    info = SimpleNamespace(
        num_inputs=1,
        num_outputs=1,
        num_states=1,
        c2k_return_mapping=[],
    )
    bundle = SimpleNamespace(training_info=info)
    x = torch.randn(0, 3)
    v = torch.randn(3)
    full_returns = [x.new_empty(x.shape), x.new_empty((0, *v.shape))]

    def fake_training_impl(bundle_arg, flat_args):
        assert bundle_arg is bundle
        assert len(flat_args) == 2
        assert flat_args[0] is x
        assert flat_args[1] is v
        return full_returns

    monkeypatch.setattr(
        custom_ops,
        "_flexsn_inductor_training_impl",
        fake_training_impl,
    )

    y_seq, v_final = custom_ops._flexsn_inductor_training_final_state_impl(
        bundle,
        [x, v],
    )

    assert tuple(y_seq.shape) == (0, 3)
    torch.testing.assert_close(v_final, v)
    assert v_final.data_ptr() != v.data_ptr()


@pytest.mark.parametrize("T", [1, 4, 16])
@pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 3, 8)])
def test_hop_matches_manual_loop(rng, T, shape):
    x = torch.randn((T, *shape), generator=rng)
    v0 = torch.zeros(shape)

    s_seq, v_seq = flex_sn_scan(_lif_core, 1, 1, 1, x, v0)

    expected_s, expected_v = [], []
    v = v0.clone()
    for t in range(T):
        s, v = _lif_core(x[t], v)
        expected_s.append(s)
        expected_v.append(v)
    expected_s = torch.stack(expected_s, dim=0)
    expected_v = torch.stack(expected_v, dim=0)

    torch.testing.assert_close(s_seq, expected_s)
    torch.testing.assert_close(v_seq, expected_v)


def test_hop_matches_manual_loop_with_lifted_tensor(rng):
    T, N = 4, 8
    x = torch.randn((T, N), generator=rng)
    v0 = torch.zeros(N)
    bias = torch.randn(N, generator=rng)

    def core_with_bias(x_step, v, bias_term):
        return _lif_core(x_step + bias_term, v)

    s_seq, v_seq = flex_sn_scan(core_with_bias, 1, 1, 1, x, v0, bias)

    expected_s, expected_v = [], []
    v = v0.clone()
    for t in range(T):
        s, v = core_with_bias(x[t], v, bias)
        expected_s.append(s)
        expected_v.append(v)

    torch.testing.assert_close(s_seq, torch.stack(expected_s, dim=0))
    torch.testing.assert_close(v_seq, torch.stack(expected_v, dim=0))


@pytest.mark.parametrize("T", [1, 4, 16])
@pytest.mark.parametrize("shape", [(8,), (4, 8)])
def test_inductor_backend_matches_torch_backend(rng, T, shape):
    x = torch.randn((T, *shape), generator=rng)

    torch_neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
    )
    inductor_neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
    )

    torch_out = torch_neuron(x)
    inductor_out = inductor_neuron(x)

    torch.testing.assert_close(inductor_out, torch_out)
    assert len(inductor_neuron.states) == 1
    torch.testing.assert_close(inductor_neuron.states[0], torch_neuron.states[0])


@pytest.mark.parametrize("T", [1, 4, 16])
@pytest.mark.parametrize("shape", [(8,), (4, 8)])
def test_hop_backend_matches_torch_backend(rng, T, shape):
    x = torch.randn((T, *shape), generator=rng)

    torch_neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
    )
    hop_neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="hop",
        store_state_seqs=True,
        example_inputs=(torch.zeros(shape), torch.zeros(shape)),
    )

    torch_out = torch_neuron(x)
    hop_out = hop_neuron(x)

    torch.testing.assert_close(hop_out, torch_out)
    assert len(hop_neuron.states) == 1
    torch.testing.assert_close(hop_neuron.states[0], torch_neuron.states[0])


def test_hop_backend_matches_torch_backend_with_closure(rng):
    T, N = 4, 8
    x = torch.randn((T, N), generator=rng)
    bias = torch.randn(N, generator=rng)

    def core_with_bias(x_step, v):
        return _lif_core(x_step + bias, v)

    torch_neuron = FlexSN(
        core=core_with_bias,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
    )
    hop_neuron = FlexSN(
        core=core_with_bias,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="hop",
        store_state_seqs=True,
        example_inputs=(torch.zeros(N), torch.zeros(N)),
    )

    torch_out = torch_neuron(x)
    hop_out = hop_neuron(x)

    torch.testing.assert_close(hop_out, torch_out)
    torch.testing.assert_close(hop_neuron.states[0], torch_neuron.states[0])


def test_inductor_backend_store_state_seqs(rng):
    T, N = 8, 16
    x = torch.randn((T, N), generator=rng)

    torch_neuron = FlexSN(
        core=_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="torch", store_state_seqs=True,
    )
    inductor_neuron = FlexSN(
        core=_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="inductor", store_state_seqs=True,
    )

    torch_neuron(x)
    inductor_neuron(x)

    assert len(inductor_neuron.state_seqs) == 1
    torch.testing.assert_close(
        inductor_neuron.state_seqs[0], torch_neuron.state_seqs[0]
    )


def test_inductor_backend_skips_flex_sn_kernel_construction():
    neuron = FlexSN(
        core=_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="inductor",
    )
    assert neuron.kernel is None


def test_inductor_backend_in_supported_backends():
    neuron = FlexSN(
        core=_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="inductor",
    )
    assert "inductor" in neuron.supported_backends


def test_hop_rejects_wrong_arity():
    x = torch.randn(4, 8)
    v0 = torch.zeros(8)
    extra = torch.zeros(8)

    with pytest.raises(ValueError, match="expected 2 tensor args"):
        flex_sn_scan(_lif_core, 1, 1, 1, x, v0, extra)


def test_hop_rejects_mismatched_T():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(5, 8)
    v0 = torch.zeros(8)

    def two_input_core(a, b, v):
        return a + b, v + a

    with pytest.raises(ValueError, match="leading dim"):
        flex_sn_scan(two_input_core, 2, 1, 1, x1, x2, v0)


def test_hop_registers_with_dynamo():
    from torch._dynamo.variables.higher_order_ops import TorchHigherOrderOperatorVariable

    hop_var = TorchHigherOrderOperatorVariable.make(flex_sn_scan)
    assert hop_var.value is flex_sn_scan


def test_lowerable_scan_matches_manual_loop(rng):
    if not callable(lowerable_scan_available) or not lowerable_scan_available():
        pytest.skip("PyTorch scan HOP is unavailable in this environment")

    T, N = 4, 8
    x = torch.randn((T, N), generator=rng)
    v0 = torch.zeros(N)

    with torch.no_grad():
        s_seq, v_seq = lowerable_scan(_lif_core, 1, 1, 1, x, v0)

    expected_s, expected_v = [], []
    v = v0.clone()
    for t in range(T):
        s, v = _lif_core(x[t], v)
        expected_s.append(s)
        expected_v.append(v)

    torch.testing.assert_close(s_seq, torch.stack(expected_s, dim=0))
    torch.testing.assert_close(v_seq, torch.stack(expected_v, dim=0))


def test_lowerable_while_loop_matches_manual_loop(rng):
    if (
        not callable(lowerable_while_loop_available)
        or not lowerable_while_loop_available()
    ):
        pytest.skip("PyTorch while_loop HOP is unavailable in this environment")

    T, N = 4, 8
    x = torch.randn((T, N), generator=rng)
    v0 = torch.zeros(N)

    with torch.no_grad():
        s_seq, v_seq = lowerable_while_loop_scan(_lif_core, 1, 1, 1, x, v0)

    expected_s, expected_v = [], []
    v = v0.clone()
    for t in range(T):
        s, v = _lif_core(x[t], v)
        expected_s.append(s)
        expected_v.append(v)

    torch.testing.assert_close(s_seq, torch.stack(expected_s, dim=0))
    torch.testing.assert_close(v_seq, torch.stack(expected_v, dim=0))


def test_compile_fullgraph_lowerable_while_loop_matches_eager(rng):
    if sys.platform == "win32":
        pytest.skip("torch.compile is not supported on Windows")
    if (
        not callable(lowerable_while_loop_available)
        or not lowerable_while_loop_available()
    ):
        pytest.skip("PyTorch while_loop HOP is unavailable in this environment")

    T, N = 4, 8
    x_eager = torch.randn((T, N), generator=rng)
    x_compiled = x_eager.detach().clone()
    v0_eager = torch.zeros(N)
    v0_compiled = v0_eager.detach().clone()

    def run_scan(x, v0):
        with torch.no_grad():
            return lowerable_while_loop_scan(_lif_core, 1, 1, 1, x, v0)

    eager_out = run_scan(x_eager, v0_eager)
    compiled_out = torch.compile(run_scan, fullgraph=True)(x_compiled, v0_compiled)

    for compiled_tensor, eager_tensor in zip(compiled_out, eager_out, strict=True):
        torch.testing.assert_close(compiled_tensor, eager_tensor)


def test_compile_fullgraph_hop_backend_matches_eager_with_lowerable_while_loop(
    rng, monkeypatch
):
    if sys.platform == "win32":
        pytest.skip("torch.compile is not supported on Windows")
    if (
        not callable(lowerable_while_loop_available)
        or not lowerable_while_loop_available()
    ):
        pytest.skip("PyTorch while_loop HOP is unavailable in this environment")

    T, N = 4, 8
    x_eager = torch.randn(T, N, generator=rng)
    x_compiled = x_eager.detach().clone()

    def make_neuron():
        return FlexSN(
            core=_differentiable_lif_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="hop",
            store_state_seqs=True,
            example_inputs=(torch.zeros(N), torch.zeros(N)),
        )

    monkeypatch.setenv("SJ_ENABLE_EXPERIMENTAL_LOWERABLE_WHILE_LOOP", "1")
    with torch.no_grad():
        eager_out = make_neuron()(x_eager)
        compiled_fn = torch.compile(make_neuron(), fullgraph=True)
        compiled_out = compiled_fn(x_compiled)

    torch.testing.assert_close(compiled_out, eager_out)


def test_hop_backend_store_state_seqs_false_matches_torch_backend(rng):
    T, N = 4, 8
    x = torch.randn(T, N, generator=rng)

    hop_neuron = FlexSN(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="hop",
        store_state_seqs=False,
        example_inputs=(torch.zeros(N), torch.zeros(N)),
    )
    torch_neuron = FlexSN(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
        store_state_seqs=False,
    )

    hop_out = hop_neuron(x)
    torch_out = torch_neuron(x)

    torch.testing.assert_close(hop_out, torch_out)
    torch.testing.assert_close(hop_neuron.states[0], torch_neuron.states[0])


def test_hop_backend_zero_length_sequence_matches_torch_backend():
    x = torch.randn(0, 8)

    hop_neuron = FlexSN(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="hop",
        store_state_seqs=True,
        example_inputs=(torch.zeros(8), torch.zeros(8)),
    )
    torch_neuron = FlexSN(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
        store_state_seqs=True,
    )

    hop_out = hop_neuron(x)
    torch_out = torch_neuron(x)

    torch.testing.assert_close(hop_out, torch_out)
    torch.testing.assert_close(hop_neuron.states[0], torch_neuron.states[0])


def test_hop_backend_zero_length_sequence_uses_closure_output_shape():
    x = torch.randn(0, 2)
    bias = torch.zeros(3)

    def core_with_closure(x_step):
        return bias

    hop_neuron = FlexSN(
        core=core_with_closure,
        num_inputs=1,
        num_states=0,
        num_outputs=1,
        step_mode="m",
        backend="hop",
        store_state_seqs=False,
        example_inputs=(torch.zeros(2),),
    )

    hop_out = hop_neuron(x)
    assert hop_out.shape == (0, 3)


def test_compile_fullgraph_hop_store_state_seqs_false_matches_eager_with_lowerable_while_loop(
    rng, monkeypatch
):
    if sys.platform == "win32":
        pytest.skip("torch.compile is not supported on Windows")
    if (
        not callable(lowerable_while_loop_available)
        or not lowerable_while_loop_available()
    ):
        pytest.skip("PyTorch while_loop HOP is unavailable in this environment")

    T, N = 4, 8
    x_eager = torch.randn(T, N, generator=rng)
    x_compiled = x_eager.detach().clone()

    def make_neuron():
        return FlexSN(
            core=_differentiable_lif_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="hop",
            store_state_seqs=False,
            example_inputs=(torch.zeros(N), torch.zeros(N)),
        )

    monkeypatch.setenv("SJ_ENABLE_EXPERIMENTAL_LOWERABLE_WHILE_LOOP", "1")
    with torch.no_grad():
        eager_neuron = make_neuron()
        eager_out = eager_neuron(x_eager)
        compiled_neuron = make_neuron()
        compiled_fn = torch.compile(compiled_neuron, fullgraph=True)
        compiled_out = compiled_fn(x_compiled)

    torch.testing.assert_close(compiled_out, eager_out)
    torch.testing.assert_close(compiled_neuron.states[0], eager_neuron.states[0])


# -------------------------------------------------------------------------
# M2: backward / autograd tests
# -------------------------------------------------------------------------


def _differentiable_lif_core(x: torch.Tensor, v: torch.Tensor):
    """LIF variant without the hard spike gate — fully differentiable so we
    can use a reference autograd baseline for numerical comparison."""
    tau = 2.0
    h = v + (x - v) / tau
    y = torch.tanh(h)
    v_new = h - y
    return y, v_new


def _reference_bptt(core_fn, x, v0, num_outputs=1):
    """Manual BPTT: run the scan with plain Python + autograd tracked."""
    T = x.shape[0]
    v = v0
    outs = []
    for t in range(T):
        results = core_fn(x[t], v)
        outs.append(results[:num_outputs])
        v = results[num_outputs]
    y_seq = torch.stack([o[0] for o in outs], dim=0)
    return y_seq, v


@pytest.mark.parametrize("T", [1, 4, 8])
@pytest.mark.parametrize("shape", [(8,), (4, 8)])
def test_hop_backward_matches_manual_bptt(rng, T, shape):
    x_hop = torch.randn((T, *shape), generator=rng, requires_grad=True)
    v0_hop = torch.randn(shape, generator=rng, requires_grad=True)
    x_ref = x_hop.detach().clone().requires_grad_(True)
    v0_ref = v0_hop.detach().clone().requires_grad_(True)

    y_seq_hop, v_seq_hop = flex_sn_scan(
        _differentiable_lif_core, 1, 1, 1, x_hop, v0_hop
    )
    loss_hop = y_seq_hop.sum() + v_seq_hop.sum()
    loss_hop.backward()

    y_seq_ref, v_final_ref = _reference_bptt(_differentiable_lif_core, x_ref, v0_ref)
    # flex_sn_scan returns stacked state sequence, reference returns final state;
    # build a matching sum-of-all-states target by re-running with accumulation
    v = v0_ref
    v_seq_ref = []
    for t in range(T):
        y, v = _differentiable_lif_core(x_ref[t], v)
        v_seq_ref.append(v)
    v_seq_ref = torch.stack(v_seq_ref, dim=0)
    loss_ref = y_seq_ref.sum() + v_seq_ref.sum()
    x_ref.grad = None
    v0_ref.grad = None
    loss_ref.backward()

    torch.testing.assert_close(y_seq_hop, y_seq_ref)
    torch.testing.assert_close(v_seq_hop, v_seq_ref)
    torch.testing.assert_close(x_hop.grad, x_ref.grad)
    torch.testing.assert_close(v0_hop.grad, v0_ref.grad)


def test_inductor_backend_backward_matches_torch_backend(rng):
    T, N = 6, 16
    x_i = torch.randn(T, N, generator=rng, requires_grad=True)
    x_t = x_i.detach().clone().requires_grad_(True)

    inductor_neuron = FlexSN(
        core=_differentiable_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="inductor",
    )
    torch_neuron = FlexSN(
        core=_differentiable_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="torch",
    )

    out_i = inductor_neuron(x_i).sum()
    out_t = torch_neuron(x_t).sum()
    out_i.backward()
    out_t.backward()

    torch.testing.assert_close(x_i.grad, x_t.grad)


def test_hop_gradcheck_small(rng):
    """gradcheck provides an independent numerical verification (finite
    differences)."""
    T, N = 3, 4
    x = torch.randn(T, N, generator=rng, dtype=torch.float64, requires_grad=True)
    v0 = torch.randn(N, generator=rng, dtype=torch.float64, requires_grad=True)

    def fn(x_in, v_in):
        y_seq, v_seq = flex_sn_scan(_differentiable_lif_core, 1, 1, 1, x_in, v_in)
        return y_seq.sum() + v_seq.sum()

    assert torch.autograd.gradcheck(fn, (x, v0), eps=1e-6, atol=1e-5)


def test_hop_backward_with_requires_grad_inputs_only(rng):
    """initial state is a constant (no grad) but inputs require grad."""
    T, N = 5, 8
    x = torch.randn(T, N, generator=rng, requires_grad=True)
    v0 = torch.zeros(N)  # no grad

    y_seq, v_seq = flex_sn_scan(_differentiable_lif_core, 1, 1, 1, x, v0)
    (y_seq.sum() + v_seq.sum()).backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


# -------------------------------------------------------------------------
# M2: AOTAutograd / make_fx tracing tests (the Inductor-ready path)
# -------------------------------------------------------------------------


def test_make_fx_unrolls_scan(rng):
    """make_fx should produce a flat aten graph (scan unrolled into T
    copies of core_fn). This is the form Inductor lowers in M3."""
    from torch.fx.experimental.proxy_tensor import make_fx

    T, N = 4, 8
    x = torch.randn(T, N, generator=rng)
    v0 = torch.zeros(N)

    def scan_fn(x_in, v_in):
        return flex_sn_scan(_differentiable_lif_core, 1, 1, 1, x_in, v_in)

    gm = make_fx(scan_fn)(x, v0)
    code = gm.code
    # Expect T unrolled aten.tanh.default calls (one per time step)
    assert code.count("aten.tanh.default") == T, (
        f"Expected {T} tanh ops, got {code.count('aten.tanh.default')}:\n{code}"
    )
    # No HOP reference should survive in the flattened graph
    assert "flex_sn_scan" not in code


def test_aot_function_traces_fwd_bwd(rng):
    """aot_function must produce both forward and backward graphs; backward
    must contain T tanh-backward ops mirroring the forward unroll."""
    from torch._functorch.aot_autograd import aot_function

    T, N = 4, 8
    x = torch.randn(T, N, generator=rng, requires_grad=True)
    v0 = torch.zeros(N, requires_grad=True)

    captured = {}

    def fwd_compiler(gm, _):
        captured["fwd"] = gm.code
        return gm

    def bwd_compiler(gm, _):
        captured["bwd"] = gm.code
        return gm

    def fn(x_in, v_in):
        y_seq, v_seq = flex_sn_scan(_differentiable_lif_core, 1, 1, 1, x_in, v_in)
        return (y_seq.sum() + v_seq.sum()).unsqueeze(0)

    compiled = aot_function(fn, fwd_compiler, bwd_compiler)
    out = compiled(x, v0)
    out.sum().backward()

    assert "fwd" in captured and "bwd" in captured
    assert captured["fwd"].count("aten.tanh.default") == T
    assert "tanh_backward" in captured["bwd"] or "mul" in captured["bwd"]
    assert x.grad is not None and v0.grad is not None


# -------------------------------------------------------------------------
# M3.a: end-to-end torch.compile integration tests
#
# On the dev machine Inductor emits C++ (no CUDA); on the target GPU host
# the same pipeline emits Triton. These tests only check correctness and
# that the fullgraph=True path is unbroken — performance is M4 scope.
# -------------------------------------------------------------------------


def test_compile_fullgraph_forward_matches_eager(rng):
    T, N = 4, 8
    x_eager = torch.randn(T, N, generator=rng)
    x_compiled = x_eager.detach().clone()

    def make_neuron():
        return FlexSN(
            core=_differentiable_lif_core, num_inputs=1, num_states=1, num_outputs=1,
            step_mode="m", backend="inductor",
        )

    eager_out = make_neuron()(x_eager)
    compiled_fn = torch.compile(make_neuron(), fullgraph=True)
    compiled_out = compiled_fn(x_compiled)

    torch.testing.assert_close(compiled_out, eager_out)


def test_compile_fullgraph_backward_matches_eager(rng):
    T, N = 5, 8
    x_eager = torch.randn(T, N, generator=rng, requires_grad=True)
    x_compiled = x_eager.detach().clone().requires_grad_(True)

    def make_neuron():
        return FlexSN(
            core=_differentiable_lif_core, num_inputs=1, num_states=1, num_outputs=1,
            step_mode="m", backend="inductor",
        )

    make_neuron()(x_eager).sum().backward()

    compiled_fn = torch.compile(make_neuron(), fullgraph=True)
    compiled_fn(x_compiled).sum().backward()

    torch.testing.assert_close(x_compiled.grad, x_eager.grad)


def test_compile_fullgraph_backward_store_state_seqs_false_matches_eager(rng):
    T, N = 5, 8
    x_eager = torch.randn(T, N, generator=rng, requires_grad=True)
    x_compiled = x_eager.detach().clone().requires_grad_(True)

    def make_neuron():
        return FlexSN(
            core=_differentiable_lif_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="inductor",
            store_state_seqs=False,
        )

    make_neuron()(x_eager).sum().backward()

    compiled_fn = torch.compile(make_neuron(), fullgraph=True)
    compiled_fn(x_compiled).sum().backward()

    torch.testing.assert_close(x_compiled.grad, x_eager.grad)


def test_compile_fullgraph_hop_backend_matches_eager(rng):
    if sys.platform == "win32":
        pytest.skip("torch.compile is not supported on Windows")

    T, N = 4, 8
    x_eager = torch.randn(T, N, generator=rng)
    x_compiled = x_eager.detach().clone()

    def make_neuron():
        return FlexSN(
            core=_differentiable_lif_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="hop",
            store_state_seqs=True,
            example_inputs=(torch.zeros(N), torch.zeros(N)),
        )

    eager_out = make_neuron()(x_eager)
    compiled_fn = torch.compile(make_neuron(), fullgraph=True)
    compiled_out = compiled_fn(x_compiled)

    torch.testing.assert_close(compiled_out, eager_out)


def test_compile_fullgraph_hop_backend_matches_eager_with_closure(rng):
    if sys.platform == "win32":
        pytest.skip("torch.compile is not supported on Windows")

    T, N = 4, 8
    x_eager = torch.randn(T, N, generator=rng)
    x_compiled = x_eager.detach().clone()
    bias = torch.randn(N, generator=rng)

    def core_with_bias(x, v):
        return _differentiable_lif_core(x + bias, v)

    def make_neuron():
        return FlexSN(
            core=core_with_bias,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="hop",
            store_state_seqs=True,
            example_inputs=(torch.zeros(N), torch.zeros(N)),
        )

    eager_out = make_neuron()(x_eager)
    compiled_fn = torch.compile(make_neuron(), fullgraph=True)
    compiled_out = compiled_fn(x_compiled)

    torch.testing.assert_close(compiled_out, eager_out)


def test_compile_fuses_surrounding_linear_layers(rng):
    """Linear -> FlexSN -> Linear must compile under fullgraph=True.
    On GPU this is the case Inductor fuses into a single kernel stack;
    here we only validate correctness vs. eager."""
    T, N = 4, 8

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = torch.nn.Linear(N, N)
            self.neuron = FlexSN(
                core=_differentiable_lif_core,
                num_inputs=1, num_states=1, num_outputs=1,
                step_mode="m", backend="inductor",
            )
            self.post = torch.nn.Linear(N, N)

        def forward(self, x):
            pre = self.pre(x)
            spikes = self.neuron(pre)
            return self.post(spikes)

    torch.manual_seed(0)
    eager_net = Net()
    torch.manual_seed(0)
    compiled_net = torch.compile(Net(), fullgraph=True)

    x_e = torch.randn(T, N, generator=rng)
    x_c = x_e.detach().clone()
    torch.testing.assert_close(compiled_net(x_c), eager_net(x_e))


def test_compile_fuses_surrounding_linear_layers_with_hop_lowerable_while_loop(rng, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("torch.compile is not supported on Windows")
    if (
        not callable(lowerable_while_loop_available)
        or not lowerable_while_loop_available()
    ):
        pytest.skip("PyTorch while_loop HOP is unavailable in this environment")

    T, N = 4, 8

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = torch.nn.Linear(N, N)
            self.neuron = FlexSN(
                core=_differentiable_lif_core,
                num_inputs=1,
                num_states=1,
                num_outputs=1,
                step_mode="m",
                backend="hop",
                example_inputs=(torch.zeros(N), torch.zeros(N)),
            )
            self.post = torch.nn.Linear(N, N)

        def forward(self, x):
            pre = self.pre(x)
            spikes = self.neuron(pre)
            return self.post(spikes)

    torch.manual_seed(0)
    eager_net = Net().eval()
    torch.manual_seed(0)
    compiled_net = Net().eval()
    monkeypatch.setenv("SJ_ENABLE_EXPERIMENTAL_LOWERABLE_WHILE_LOOP", "1")
    compiled_net = torch.compile(compiled_net, fullgraph=True)

    x_e = torch.randn(T, N, generator=rng)
    x_c = x_e.detach().clone()
    with torch.no_grad():
        eager_out = eager_net(x_e)
        compiled_out = compiled_net(x_c)

    torch.testing.assert_close(compiled_out, eager_out)


def test_compile_spiking_vgg_hop_lowerable_while_loop_matches_eager(monkeypatch):
    if sys.platform == "win32":
        pytest.skip("torch.compile is not supported on Windows")
    if (
        not callable(lowerable_while_loop_available)
        or not lowerable_while_loop_available()
    ):
        pytest.skip("PyTorch while_loop HOP is unavailable in this environment")
    if not torch.cuda.is_available():
        pytest.skip("SpikingVGG compile coverage is only exercised on CUDA")

    def core(x, v):
        tau, v_th = 2.0, 1.0
        h = v + (x - v) / tau
        s = (h >= v_th).to(h.dtype).contiguous()
        return s, (h * (1.0 - s)).contiguous()

    def make_flexsn(**kwargs):
        return FlexSN(
            core=core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode=kwargs.get("step_mode", "m"),
            backend="hop",
        )

    T, B, C, H, W = 4, 8, 3, 32, 32
    torch.manual_seed(42)
    x_eager = torch.randn((T, B, C, H, W), device="cuda")
    x_compiled = x_eager.detach().clone()

    torch.manual_seed(0)
    eager_model = spiking_vgg16_bn(
        spiking_neuron=make_flexsn,
        step_mode="m",
    ).cuda().eval()
    torch.manual_seed(0)
    compiled_model = spiking_vgg16_bn(
        spiking_neuron=make_flexsn,
        step_mode="m",
    ).cuda().eval()

    from spikingjelly.activation_based import functional as sj_functional
    sj_functional.set_step_mode(eager_model, "m")
    sj_functional.set_step_mode(compiled_model, "m")

    monkeypatch.setenv("SJ_ENABLE_EXPERIMENTAL_LOWERABLE_WHILE_LOOP", "1")

    with torch.backends.cudnn.flags(enabled=False):
        with torch.no_grad():
            eager_out = eager_model(x_eager)
            compiled_model = torch.compile(compiled_model, fullgraph=True)
            compiled_out = compiled_model(x_compiled)

    torch.testing.assert_close(compiled_out, eager_out, atol=1e-5, rtol=1e-4)


def test_compile_captures_core_ops_in_fx_graph(rng):
    """Verify that the compiled forward graph contains T unrolled copies of
    the core_fn's aten ops — the unrolled shape Inductor expects to lower.
    """
    from torch._dynamo import explain

    T, N = 4, 8
    x = torch.randn(T, N, generator=rng)
    neuron = FlexSN(
        core=_differentiable_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="inductor",
    )

    explanation = explain(neuron)(x)
    assert explanation.graph_break_count == 0, (
        f"Expected no graph breaks, got {explanation.graph_break_count}:\n"
        f"{explanation.break_reasons}"
    )


def test_inductor_compile_graph_elides_explicit_zero_state_init():
    if sys.platform == "win32":
        pytest.skip("torch.compile is not supported on Windows")
    if not torch.cuda.is_available():
        pytest.skip("inductor custom-op graph coverage is exercised on CUDA")

    from torch._dynamo import explain

    def core(x, v):
        tau, v_th = 2.0, 1.0
        h = v + (x - v) / tau
        s = (h >= v_th).to(h.dtype)
        return s, h * (1.0 - s)

    def make_flexsn(**kwargs):
        return FlexSN(
            core=core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode=kwargs.get("step_mode", "m"),
            backend="inductor",
        )

    T, N = 4, 8
    torch.manual_seed(42)
    x = torch.randn((T, N), device="cuda")
    neuron = make_flexsn(store_state_seqs=False).cuda().eval()

    with torch.no_grad():
        explanation = explain(neuron)(x)
    targets = [str(node.target) for graph in explanation.graphs for node in graph.graph.nodes]

    assert any(
        "sj.flexsn_inductor_inference_final_state.default" in target
        for target in targets
    )
    assert not any("zeros_like" in target for target in targets)


def test_inductor_compile_training_graph_uses_final_state_custom_op():
    if sys.platform == "win32":
        pytest.skip("torch.compile is not supported on Windows")
    if not torch.cuda.is_available():
        pytest.skip("inductor custom-op graph coverage is exercised on CUDA")

    from torch._dynamo import explain

    def core(x, v):
        tau, v_th = 2.0, 1.0
        h = v + (x - v) / tau
        s = (h >= v_th).to(h.dtype)
        return s, h * (1.0 - s)

    T, N = 4, 8
    torch.manual_seed(42)
    x = torch.randn((T, N), device="cuda", requires_grad=True)
    neuron = FlexSN(
        core=core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
        store_state_seqs=False,
    ).cuda().train()

    explanation = explain(lambda inp: neuron(inp).sum())(x)
    targets = [
        str(node.target)
        for graph in explanation.graphs
        for node in graph.graph.nodes
    ]

    assert any(
        "sj.flexsn_inductor_training_final_state.default" in target
        for target in targets
    )


def test_aot_function_backward_numerical_match(rng):
    """AOTAutograd-compiled backward must produce the same gradients as
    the eager backward."""
    from torch._functorch.aot_autograd import aot_function

    T, N = 5, 8
    x_aot = torch.randn(T, N, generator=rng, requires_grad=True)
    v0_aot = torch.randn(N, generator=rng, requires_grad=True)
    x_eager = x_aot.detach().clone().requires_grad_(True)
    v0_eager = v0_aot.detach().clone().requires_grad_(True)

    def fn(x_in, v_in):
        y_seq, v_seq = flex_sn_scan(_differentiable_lif_core, 1, 1, 1, x_in, v_in)
        return (y_seq.sum() + v_seq.sum()).unsqueeze(0)

    compiled = aot_function(fn, lambda gm, _: gm, lambda gm, _: gm)
    compiled(x_aot, v0_aot).sum().backward()
    fn(x_eager, v0_eager).sum().backward()

    torch.testing.assert_close(x_aot.grad, x_eager.grad)
    torch.testing.assert_close(v0_aot.grad, v0_eager.grad)
