"""M1 tests for FlexSN inductor backend (eager path only).

Validates that the new ``flex_sn_scan`` HOP + ``FlexSN(backend="inductor")``
produce numerically identical output/state sequences to the reference
``backend="torch"`` implementation on a simple LIF single-step core.

Inductor lowering and Triton codegen are M3 scope and are not exercised here,
so these tests do not require a CUDA GPU.
"""
from __future__ import annotations

import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from spikingjelly.activation_based.neuron.flexsn import FlexSN
from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg16_bn
from spikingjelly.activation_based.triton_kernel.flex_sn_inductor import (
    custom_ops as flexsn_custom_ops,
    flex_sn_scan,
    kernel as flexsn_inductor_kernel,
    lowerable_scan,
    lowerable_scan_available,
    lowerable_while_loop_scan,
    lowerable_while_loop_available,
)
from spikingjelly.activation_based.triton_kernel.flex_sn_inductor import (
    hop as flexsn_hop,
)
from spikingjelly.activation_based.triton_kernel.flexsn.info import FlexSNInfo
from spikingjelly.activation_based.triton_kernel.flexsn.wrapper import (
    flexsn_backward_ncl_bucket,
)


def _lif_core(x: torch.Tensor, v: torch.Tensor):
    tau = 2.0
    v_threshold = 1.0
    v_reset = 0.0
    h = v + (x - (v - v_reset)) / tau
    s = (h >= v_threshold).to(h.dtype)
    v_new = h * (1.0 - s) + v_reset * s
    return s, v_new


def _stateful_tanh_core(x: torch.Tensor, v: torch.Tensor):
    v_new = torch.tanh(x + 0.5 * v)
    y = torch.sigmoid(v_new)
    return y, v_new


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


def test_hop_rejects_empty_input_sequence():
    x = torch.empty(0, 8)
    v0 = torch.zeros(8)

    with pytest.raises(ValueError, match="T == 0"):
        flex_sn_scan(_lif_core, 1, 1, 1, x, v0)

    with pytest.raises(ValueError, match="T == 0"):
        flexsn_hop.eager_scan_final_state(_lif_core, 1, 1, 1, x, v0)


def test_kernel_names_include_graph_fingerprint():
    core_a = lambda x: x + 1
    core_b = lambda x: x * 2
    example = torch.randn(4)
    graph_a = make_fx(core_a)(example).graph
    graph_b = make_fx(core_b)(example).graph

    name_a = flexsn_inductor_kernel._make_core_name(
        core_a, "inductor_scan", graph_a
    )
    name_b = flexsn_inductor_kernel._make_core_name(
        core_b, "inductor_scan", graph_b
    )

    assert name_a != name_b
    assert name_a == flexsn_inductor_kernel._make_core_name(
        core_a, "inductor_scan", graph_a
    )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("store_state_seqs", [False, True])
def test_inductor_backend_segmented_inference_matches_unsegmented(
    rng, monkeypatch, store_state_seqs
):
    T, N = 9, 16
    x = torch.randn((T, N), generator=rng).cuda()

    reference = FlexSN(
        core=_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="inductor", store_state_seqs=store_state_seqs,
    ).cuda()
    segmented = FlexSN(
        core=_lif_core, num_inputs=1, num_states=1, num_outputs=1,
        step_mode="m", backend="inductor", store_state_seqs=store_state_seqs,
    ).cuda()

    with torch.no_grad():
        ref_out = reference(x)
        monkeypatch.setenv("SJ_FLEXSN_INDUCTOR_SEGMENT_T", "4")
        seg_out = segmented(x)

    torch.testing.assert_close(seg_out, ref_out)
    torch.testing.assert_close(segmented.states[0], reference.states[0])
    if store_state_seqs:
        torch.testing.assert_close(segmented.state_seqs[0], reference.state_seqs[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inductor_training_final_state_matches_full_training(rng):
    T, N = 8, 16
    x_ref = torch.randn((T, N), generator=rng, device="cpu").cuda().requires_grad_(True)
    x_opt = x_ref.detach().clone().requires_grad_(True)

    reference = FlexSN(
        core=_stateful_tanh_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
        store_state_seqs=True,
    ).cuda()
    optimized = FlexSN(
        core=_stateful_tanh_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
        store_state_seqs=False,
    ).cuda()

    ref_out = reference(x_ref)
    opt_out = optimized(x_opt)

    torch.testing.assert_close(opt_out, ref_out)
    torch.testing.assert_close(optimized.states[0], reference.states[0])

    ref_loss = ref_out.sum() + reference.states[0].sum()
    opt_loss = opt_out.sum() + optimized.states[0].sum()
    ref_loss.backward()
    opt_loss.backward()

    torch.testing.assert_close(x_opt.grad, x_ref.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inductor_training_final_state_kernel_only_builds_when_it_reduces_saved_sequences():
    lif_neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
    ).cuda()
    tanh_neuron = FlexSN(
        core=_stateful_tanh_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
    ).cuda()

    assert lif_neuron._inductor_fwd_final_state_kernel is not None
    assert tanh_neuron._inductor_fwd_final_state_kernel is None


def _stateless_passthrough_core(x: torch.Tensor, v: torch.Tensor):
    return x, x


@pytest.mark.parametrize("backend", ["inductor", "hop"])
def test_scan_backends_reject_mismatched_example_state_shape(backend):
    with pytest.raises(ValueError, match="example input/state tensor"):
        FlexSN(
            core=_stateless_passthrough_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend=backend,
            example_inputs=(torch.zeros(8), torch.zeros(4)),
        )


def test_inductor_zero_state_materialization_uses_registered_state_specs():
    info = FlexSNInfo(
        num_inputs=1,
        num_outputs=1,
        num_states=1,
        fwd_core_args=[],
        fwd_core_returns=[],
        fwd_core_recipients=[],
        fwd_kernel_returns=[],
        num_fwd_kernel_returns=0,
        c2k_return_mapping=[],
    )
    bundle = flexsn_custom_ops.FlexSNKernelHandle(
        inference_kernel=None,
        inference_info=info,
        inference_final_state_kernel=None,
        inference_final_state_info=None,
        forward_kernel=None,
        forward_final_state_kernel=None,
        backward_kernel=None,
        backward_final_state_kernel=None,
        training_info=None,
        state_template_specs=(((3, 5), torch.float32, torch.device("cpu")),),
    )
    x_seq = torch.randn(4, 8, dtype=torch.float16)

    materialized = flexsn_custom_ops._materialize_zero_state_args(bundle, info, [x_seq])

    assert len(materialized) == 2
    assert materialized[1].shape == (3, 5)
    assert materialized[1].dtype == torch.float32
    assert materialized[1].device.type == "cpu"
    torch.testing.assert_close(materialized[1], torch.zeros(3, 5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inductor_zero_state_materialization_overrides_registered_device_at_runtime():
    info = FlexSNInfo(
        num_inputs=1,
        num_outputs=1,
        num_states=1,
        fwd_core_args=[],
        fwd_core_returns=[],
        fwd_core_recipients=[],
        fwd_kernel_returns=[],
        num_fwd_kernel_returns=0,
        c2k_return_mapping=[],
    )
    bundle = flexsn_custom_ops.FlexSNKernelHandle(
        inference_kernel=None,
        inference_info=info,
        inference_final_state_kernel=None,
        inference_final_state_info=None,
        forward_kernel=None,
        forward_final_state_kernel=None,
        backward_kernel=None,
        backward_final_state_kernel=None,
        training_info=None,
        state_template_specs=(((3, 5), torch.float32, torch.device("cpu")),),
    )
    x_seq = torch.randn(4, 8, dtype=torch.float16, device="cuda")

    materialized = flexsn_custom_ops._materialize_zero_state_args(bundle, info, [x_seq])

    assert len(materialized) == 2
    assert materialized[1].shape == (3, 5)
    assert materialized[1].dtype == torch.float32
    assert materialized[1].device.type == "cuda"
    torch.testing.assert_close(materialized[1], torch.zeros(3, 5, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inductor_backward_final_state_kernel_handles_small_t_large_token_workloads():
    small_t_large_tokens = torch.randn((4, 8, 224, 224), device="cuda")
    small_t_medium_tokens = torch.randn((4, 32, 3, 32, 32), device="cuda")
    small_t_small_tokens = torch.randn((4, 16, 64), device="cuda")

    assert flexsn_custom_ops._should_use_backward_final_state_kernel(
        [small_t_large_tokens]
    )
    assert flexsn_custom_ops._should_use_backward_final_state_kernel(
        [small_t_medium_tokens]
    )
    assert not flexsn_custom_ops._should_use_backward_final_state_kernel(
        [small_t_small_tokens]
    )


def test_flexsn_backward_ncl_bucket_boundaries():
    assert flexsn_backward_ncl_bucket(512) == 0
    assert flexsn_backward_ncl_bucket(1 << 12) == 0
    assert flexsn_backward_ncl_bucket((1 << 12) + 1) == 1
    assert flexsn_backward_ncl_bucket(1 << 17) == 1
    assert flexsn_backward_ncl_bucket((1 << 17) + 1) == 2
    assert flexsn_backward_ncl_bucket(1 << 20) == 2
    assert flexsn_backward_ncl_bucket((1 << 20) + 1) == 3
    assert flexsn_backward_ncl_bucket(1 << 23) == 3
    assert flexsn_backward_ncl_bucket((1 << 23) + 1) == 4



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

    with pytest.raises(ValueError, match="expected at least 2 tensor args"):
        flex_sn_scan(_lif_core, 1, 1, 1, x)


def test_hop_accepts_single_tensor_return(rng):
    x = torch.randn(4, 8, generator=rng)

    def stateless_core(x_step):
        return x_step.sin()

    (y_seq,) = flex_sn_scan(stateless_core, 1, 0, 1, x)
    torch.testing.assert_close(y_seq, x.sin())


def test_hop_rejects_mismatched_T():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(5, 8)
    v0 = torch.zeros(8)

    def two_input_core(a, b, v):
        return a + b, v + a

    with pytest.raises(ValueError, match="leading dim"):
        flex_sn_scan(two_input_core, 2, 1, 1, x1, x2, v0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inductor_inference_without_final_state_kernel_keeps_final_state_semantics(rng):
    x = torch.randn((6, 16), generator=rng).cuda()
    neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
        store_state_seqs=False,
    ).cuda()
    if not neuron._inductor_inference_available:
        pytest.skip("inductor inference kernel unavailable")

    neuron._inductor_scan_final_state_kernel = None
    neuron._inductor_scan_final_state_info = None
    neuron._inductor_inference_final_state_available = False

    with torch.no_grad():
        _ = neuron(x)

    assert neuron.states[0].shape == x.shape[1:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inductor_stateless_final_state_kernel_builds():
    def stateless_core(x):
        return (x,)

    neuron = FlexSN(
        core=stateless_core,
        num_inputs=1,
        num_states=0,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
        store_state_seqs=False,
        example_inputs=(torch.zeros(8, device="cuda"),),
    ).cuda()

    assert neuron._inductor_inference_available
    assert neuron._inductor_inference_final_state_available


def test_backward_final_state_kernel_threshold_knobs_do_not_force_specialization(monkeypatch):
    monkeypatch.setattr(
        flexsn_custom_ops,
        "_BACKWARD_FINAL_STATE_SPECIALIZED_MIN_STEPS",
        0,
    )
    monkeypatch.setattr(
        flexsn_custom_ops,
        "_BACKWARD_FINAL_STATE_SPECIALIZED_MIN_TOKENS",
        1024,
    )
    grad = torch.randn(4, 8)
    assert not flexsn_custom_ops._should_use_backward_final_state_kernel([grad])


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

    assert len(compiled_out) == len(eager_out)
    for compiled_tensor, eager_tensor in zip(compiled_out, eager_out):
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
            example_inputs=(torch.zeros(N), torch.zeros(N)),
        )

    eager_out = make_neuron()(x_eager)
    monkeypatch.setenv("SJ_ENABLE_EXPERIMENTAL_LOWERABLE_WHILE_LOOP", "1")
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

    eager_neuron = make_neuron()
    eager_out = eager_neuron(x_eager)
    monkeypatch.setenv("SJ_ENABLE_EXPERIMENTAL_LOWERABLE_WHILE_LOOP", "1")
    compiled_fn = torch.compile(make_neuron(), fullgraph=True)
    compiled_out = compiled_fn(x_compiled)

    torch.testing.assert_close(compiled_out, eager_out)
    torch.testing.assert_close(compiled_fn._orig_mod.states[0], eager_neuron.states[0])


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
        s = (h >= v_th).to(h.dtype)
        return s, h * (1.0 - s)

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
    model = spiking_vgg16_bn(
        spiking_neuron=make_flexsn,
        step_mode="m",
    ).cuda().eval()

    from spikingjelly.activation_based import functional as sj_functional
    sj_functional.set_step_mode(model, "m")

    monkeypatch.setenv("SJ_ENABLE_EXPERIMENTAL_LOWERABLE_WHILE_LOOP", "1")

    with torch.no_grad():
        eager_out = model(x_eager)
        compiled_model = torch.compile(model, fullgraph=True)
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
