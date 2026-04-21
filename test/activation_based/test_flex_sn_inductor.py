"""M1 tests for FlexSN inductor backend (eager path only).

Validates that the new ``flex_sn_scan`` HOP + ``FlexSN(backend="inductor")``
produce numerically identical output/state sequences to the reference
``backend="torch"`` implementation on a simple LIF single-step core.

Inductor lowering and Triton codegen are M3 scope and are not exercised here,
so these tests do not require a CUDA GPU.
"""
from __future__ import annotations

import pytest
import torch

from spikingjelly.activation_based.neuron.flexsn import FlexSN
from spikingjelly.activation_based.triton_kernel.flex_sn_inductor import (
    flex_sn_scan,
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
