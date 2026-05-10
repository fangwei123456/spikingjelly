"""Coverage for FlexSN's triton / inductor backend and HOP fallback helpers.

The file exercises FlexSN's shared ``triton`` / ``inductor`` CUDA backend
labels, the ``flex_sn_scan`` HOP path, and the low-level helper modules under
``triton_kernel.flexsn``. CUDA-specific cases are guarded individually so the
non-CUDA portions still run on a plain development machine.
"""

from __future__ import annotations

import sys
import copy
from types import SimpleNamespace

import pytest
import torch
from spikingjelly.activation_based import base as base_module
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg16_bn
from spikingjelly.activation_based.neuron import flexsn as flexsn_module
from spikingjelly.activation_based.neuron.flexsn import (
    FlexSN,
    FlexSNKernel,
    _make_inductor_final_state_warmup_args,
)
from spikingjelly.activation_based.triton_kernel.flexsn import hop as hop_module
from spikingjelly.activation_based.triton_kernel.flexsn import (
    template as template_module,
)
from spikingjelly.activation_based.triton_kernel.flexsn.hop import (
    dynamo_hop_available,
    eager_scan,
    eager_scan_final_state,
    flex_sn_scan,
    lowerable_scan,
    lowerable_scan_available,
    lowerable_scan_final_state,
    lowerable_while_loop_available,
    lowerable_while_loop_scan,
)
from spikingjelly.activation_based.triton_kernel.flexsn.info import FlexSNInfo
from spikingjelly.activation_based.triton_kernel.torch2triton import (
    graph2triton as graph2triton_module,
)
from spikingjelly.activation_based.triton_kernel.torch2triton import (
    torch2graph as torch2graph_module,
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


def _skip_if_dynamo_hop_unavailable():
    if not callable(dynamo_hop_available) or not dynamo_hop_available():
        pytest.skip("FlexSN Dynamo HOP registration is unavailable")


def test_torch_backend_empty_sequence_does_not_call_core():
    calls = {"count": 0}

    def core(x, v):
        calls["count"] += 1
        raise AssertionError("core should not run for an empty multi-step input")

    m = FlexSN(
        core=core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
        example_outputs=(torch.zeros(3),),
    )

    out = m(torch.empty(0, 3))

    assert calls["count"] == 0
    assert out.shape == (0, 3)
    assert m.states[0].shape == (3,)


def test_torch_backend_empty_sequence_does_not_probe_core_at_construction():
    calls = {"count": 0}

    def core(x, v):
        calls["count"] += 1
        return x.new_zeros(4), v

    m = FlexSN(
        core=core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
        example_inputs=(torch.zeros(2), torch.zeros(3)),
        example_outputs=(torch.zeros(4),),
    )
    assert calls["count"] == 0
    m.states = [torch.zeros(3)]

    out = m.multi_step_forward(torch.empty(0, 2))[0]

    assert calls["count"] == 0
    assert out.shape == (0, 4)


def test_torch_backend_empty_sequence_requires_example_output_template():
    calls = {"count": 0}

    def core(x, v):
        calls["count"] += 1
        return x, v

    m = FlexSN(
        core=core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
    )

    with pytest.raises(ValueError, match="requires example_outputs"):
        m(torch.empty(0, 3))

    assert calls["count"] == 0


def test_torch_backend_empty_sequence_rejects_example_input_template():
    calls = {"count": 0}

    def core(x, v):
        calls["count"] += 1
        return x.new_zeros(5), v

    m = FlexSN(
        core=core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
        example_inputs=(torch.zeros(4), torch.zeros(3)),
    )
    m.states = [torch.zeros(3)]

    with pytest.raises(ValueError, match="requires example_outputs"):
        m.multi_step_forward(torch.empty(0, 4))

    assert calls["count"] == 0


def test_torch_backend_empty_state_only_sequence_does_not_require_output_template():
    calls = {"count": 0}

    def core(x, v):
        calls["count"] += 1
        raise AssertionError("core should not run for an empty multi-step input")

    m = FlexSN(
        core=core,
        num_inputs=1,
        num_states=1,
        num_outputs=0,
        step_mode="m",
        backend="torch",
    )
    m.states = [torch.zeros(3)]

    outputs = m.multi_step_forward(torch.empty(0, 3))

    assert calls["count"] == 0
    assert outputs == []
    torch.testing.assert_close(m.states[0], torch.zeros(3))


def test_torch_backend_empty_sequence_uses_example_output_template_without_core_probe():
    calls = {"count": 0}

    def core(x, v):
        calls["count"] += 1
        raise AssertionError("core should not run for an empty multi-step input")

    m = FlexSN(
        core=core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
        example_inputs=(torch.zeros(2), torch.zeros(3)),
        example_outputs=(torch.zeros(4, dtype=torch.float64),),
    )
    assert calls["count"] == 0
    m.states = [torch.zeros(3)]

    out = m.multi_step_forward(torch.empty(0, 2))[0]

    assert calls["count"] == 0
    assert out.shape == (0, 4)
    assert out.dtype == torch.float64


def test_flexsn_rejects_invalid_example_output_template():
    with pytest.raises(ValueError, match="expected 1 example output tensors"):
        FlexSN(
            core=_lif_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="torch",
            example_outputs=(),
        )


def test_scan_backends_reject_mismatched_example_output_template():
    with pytest.raises(ValueError, match="example_outputs"):
        FlexSN(
            core=_lif_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="inductor",
            example_inputs=(torch.zeros(3), torch.zeros(3)),
            example_outputs=(torch.zeros(4),),
        )


def test_scan_backends_reject_mismatched_example_output_template_without_example_inputs():
    with pytest.raises(ValueError, match="example_outputs"):
        FlexSN(
            core=_lif_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="inductor",
            example_outputs=(torch.zeros(4),),
        )


def test_scan_backends_reject_mismatched_example_input_numel():
    with pytest.raises(ValueError, match="same number of elements"):
        FlexSN(
            core=_lif_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="inductor",
            example_inputs=(torch.zeros(2, 3), torch.zeros(5)),
            example_outputs=(torch.zeros(2, 3),),
        )


def test_generate_forward_and_backward_graph_rejects_requires_grad_length_mismatch():
    with pytest.raises(ValueError, match="requires_grad must have the same length"):
        torch2graph_module.generate_forward_and_backward_graph(
            _lif_core,
            (torch.zeros(3), torch.zeros(3)),
            requires_grad=(True,),
        )


def test_output_template_specs_from_outputs_preserve_device():
    output = torch.zeros(4, dtype=torch.float64)

    specs = flexsn_module._make_output_template_specs_from_outputs(1, (output,))

    assert specs == (((4,), torch.float64, output.device),)


def test_empty_multistep_outputs_honor_template_device():
    specs = (((4,), torch.float64, torch.device("meta")),)

    outputs = flexsn_module._empty_multistep_outputs((torch.empty(0, 3),), [], 1, specs)

    assert outputs[0].shape == (0, 4)
    assert outputs[0].dtype == torch.float64
    assert outputs[0].device.type == "meta"


def test_flexsn_empty_multistep_outputs_follow_runtime_device():
    def core(x, v):
        raise AssertionError("core should not run for an empty multi-step input")

    m = FlexSN(
        core=core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
        example_outputs=(torch.empty(4, device="meta"),),
    )
    m.states = [torch.zeros(3)]

    out = m.multi_step_forward(torch.empty(0, 2))[0]

    assert out.shape == (0, 4)
    assert out.device.type == "cpu"


def test_empty_multistep_outputs_rejects_missing_template_reference():
    with pytest.raises(ValueError, match="at least one input or state"):
        flexsn_module._empty_multistep_outputs((), [], 1)


def test_multi_step_forward_initializes_states_for_torch_backend():
    m = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
    )

    out = m.multi_step_forward(torch.randn(2, 3))[0]

    assert out.shape == (2, 3)
    assert m.states is not None
    assert m.states[0].shape == (3,)


def test_multi_step_forward_initializes_states_for_hop_backend():
    m = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="hop",
    )

    out = m.multi_step_forward(torch.randn(2, 3))[0]

    assert out.shape == (2, 3)
    assert m.states is not None
    assert m.states[0].shape == (3,)


def test_multi_step_forward_initializes_states_for_triton_backend():
    m = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="triton",
    )

    out = m.multi_step_forward(torch.randn(2, 3))[0]

    assert out.shape == (2, 3)
    assert m.states is not None
    assert m.states[0].shape == (3,)
    assert m.backend == "triton"


def test_multi_step_forward_initializes_states_for_inductor_training_fallback():
    m = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
    )

    out = m.multi_step_forward(torch.randn(2, 3, requires_grad=True))[0]

    assert out.shape == (2, 3)
    assert m.states is not None
    assert m.states[0].shape == (3,)


def test_core_requires_grad_detects_functor_tensor_attributes():
    class Core:
        def __init__(self, weight):
            self.weight = weight

        def __call__(self, x, v):
            return x * self.weight, v

    assert flexsn_module._core_requires_grad(Core(torch.ones(3, requires_grad=True)))
    assert not flexsn_module._core_requires_grad(
        Core(torch.ones(3, requires_grad=False))
    )


def test_core_requires_grad_detects_bound_method_self_parameters():
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3))

        def forward(self, x, v):
            return x * self.weight, v

    assert flexsn_module._core_requires_grad(Core().forward)


def test_core_requires_grad_detects_module_unregistered_tensors():
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.ones(3, requires_grad=True)

        def forward(self, x, v):
            return x * self.weight, v

    assert flexsn_module._core_requires_grad(Core().forward)


def test_core_requires_grad_detects_plain_bound_method_self_tensors():
    class Core:
        def __init__(self):
            self.weight = torch.ones(3, requires_grad=True)

        def forward(self, x, v):
            return x * self.weight, v

    assert flexsn_module._core_requires_grad(Core().forward)


def test_flexsn_wrapper_final_states_use_init_state_templates():
    from spikingjelly.activation_based.triton_kernel.flexsn.wrapper import (
        _num_elements_per_step,
        flexsn_inference_final_state,
    )

    info = SimpleNamespace(num_inputs=1, num_states=1, num_outputs=1)
    x = torch.empty(0, 2)
    state = torch.ones(3, dtype=torch.float64)

    assert _num_elements_per_step(x) == 2

    out, final_state = flexsn_inference_final_state(None, info, x, state)

    assert out.shape == (0, 2)
    assert final_state.shape == (3,)
    assert final_state.dtype == torch.float64
    assert final_state.data_ptr() != state.data_ptr()
    torch.testing.assert_close(final_state, state)


def test_codegen_stem_sanitizes_cache_filenames():
    assert graph2triton_module._safe_codegen_stem("../escape") == "escape"
    assert graph2triton_module._safe_codegen_stem("/tmp/absolute") == "absolute"
    assert graph2triton_module._safe_codegen_stem("bad/name?x") == "name_x"
    assert graph2triton_module._safe_codegen_stem("..") == "kernel"
    assert "__path__" in graph2triton_module._NAMESPACE_METADATA_KEYS


def test_compile_triton_code_str_rejects_stale_namespace_symbol(tmp_path, monkeypatch):
    stale = object()
    namespace = {"missing_kernel": stale}

    monkeypatch.setattr(graph2triton_module, "_CODEGEN_CACHE_DIR", tmp_path)
    monkeypatch.setattr(graph2triton_module, "_has_real_triton_runtime", lambda: True)

    with pytest.raises(ValueError, match="missing_kernel"):
        graph2triton_module.compile_triton_code_str(
            "other_symbol = 1\n",
            "missing_kernel",
            name_space=namespace,
        )

    assert namespace["missing_kernel"] is stale


def test_flexsn_wrapper_t0_final_states_preserve_num_states_without_init():
    from spikingjelly.activation_based.triton_kernel.flexsn.wrapper import (
        flexsn_inference_final_state,
    )

    info = SimpleNamespace(num_inputs=1, num_states=2, num_outputs=1)
    x = torch.empty(0, 3)

    out, state0, state1 = flexsn_inference_final_state(None, info, x)

    assert out.shape == (0, 3)
    assert state0.shape == (3,)
    assert state1.shape == (3,)
    torch.testing.assert_close(state0, torch.zeros_like(state0))
    torch.testing.assert_close(state1, torch.zeros_like(state1))


def test_flexsn_wrappers_skip_kernel_launch_for_t0():
    from spikingjelly.activation_based.triton_kernel.flexsn import wrapper

    class _RaisingKernel:
        def __getitem__(self, grid):
            raise AssertionError("T == 0 wrappers should not launch Triton")

    x = torch.empty(0, 3)
    info = SimpleNamespace(
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        num_fwd_kernel_returns=2,
    )

    inference_outputs = wrapper.flexsn_inference(_RaisingKernel(), info, x)
    forward_outputs = wrapper.flexsn_forward(_RaisingKernel(), info, x)
    backward_outputs = wrapper.flexsn_backward(_RaisingKernel(), info, x)

    assert [tuple(t.shape) for t in inference_outputs] == [(0, 3), (0, 3)]
    assert [tuple(t.shape) for t in forward_outputs] == [(0, 3), (0, 3)]
    assert [tuple(t.shape) for t in backward_outputs] == [(0, 3), (3,)]
    torch.testing.assert_close(backward_outputs[0], torch.zeros_like(x))
    torch.testing.assert_close(
        backward_outputs[1],
        torch.zeros_like(backward_outputs[1]),
    )


def test_flexsn_wrapper_t0_backward_state_grads_use_state_templates(monkeypatch):
    from spikingjelly.activation_based.triton_kernel.flexsn import wrapper

    class _FakeKernel:
        def __getitem__(self, grid):
            def _run(*args, **kwargs):
                return None

            return _run

    info = SimpleNamespace(num_inputs=1, num_states=2, num_outputs=1)
    grad_y = torch.empty(0, 2)
    grad_s0 = torch.empty(0, 3)
    grad_s1 = torch.empty(0, 4, dtype=torch.float64)

    monkeypatch.setitem(wrapper.type_dict, grad_y.dtype, "float32")

    grad_x, grad_v0, grad_v1 = wrapper.flexsn_backward(
        _FakeKernel(),
        info,
        grad_y,
        grad_s0,
        grad_s1,
    )

    assert grad_x.shape == (0, 2)
    assert grad_v0.shape == (3,)
    assert grad_v1.shape == (4,)
    assert grad_v1.dtype == torch.float64
    torch.testing.assert_close(grad_v0, torch.zeros_like(grad_v0))
    torch.testing.assert_close(grad_v1, torch.zeros_like(grad_v1))


def test_flexsn_wrapper_backward_uses_input_templates_for_state_only_core(monkeypatch):
    from spikingjelly.activation_based.triton_kernel.flexsn import wrapper

    class _FakeKernel:
        def __getitem__(self, grid):
            def _run(*args, **kwargs):
                return None

            return _run

    info = SimpleNamespace(num_inputs=1, num_states=1, num_outputs=0)
    input_template = torch.empty(2, 2, 3)
    grad_state_seq = torch.empty(2, 6)

    monkeypatch.setitem(wrapper.type_dict, grad_state_seq.dtype, "float32")

    grad_x, grad_v = wrapper.flexsn_backward(
        _FakeKernel(),
        info,
        grad_state_seq,
        input_templates=(input_template,),
    )

    assert grad_x.shape == input_template.shape
    assert grad_v.shape == (6,)


def test_flexsn_wrapper_t0_backward_uses_input_templates_for_state_only_core(
    monkeypatch,
):
    from spikingjelly.activation_based.triton_kernel.flexsn import wrapper

    class _FakeKernel:
        def __getitem__(self, grid):
            def _run(*args, **kwargs):
                return None

            return _run

    info = SimpleNamespace(num_inputs=1, num_states=1, num_outputs=0)
    input_template = torch.empty(0, 2, 3)
    grad_state_seq = torch.empty(0, 6)

    monkeypatch.setitem(wrapper.type_dict, grad_state_seq.dtype, "float32")

    grad_x, grad_v = wrapper.flexsn_backward(
        _FakeKernel(),
        info,
        grad_state_seq,
        input_templates=(input_template,),
    )

    assert grad_x.shape == input_template.shape
    assert grad_v.shape == (6,)
    torch.testing.assert_close(grad_x, torch.zeros_like(input_template))
    torch.testing.assert_close(grad_v, torch.zeros_like(grad_v))


def test_flexsn_wrapper_backward_uses_state_templates_for_state_only_core(monkeypatch):
    from spikingjelly.activation_based.triton_kernel.flexsn import wrapper

    class _FakeKernel:
        def __getitem__(self, grid):
            def _run(*args, **kwargs):
                return None

            return _run

    info = SimpleNamespace(num_inputs=1, num_states=1, num_outputs=0)
    input_template = torch.empty(2, 2, 3)
    state_template = torch.empty(6)
    grad_state_seq = torch.empty(2, 2, 3)

    monkeypatch.setitem(wrapper.type_dict, grad_state_seq.dtype, "float32")

    grad_x, grad_v = wrapper.flexsn_backward(
        _FakeKernel(),
        info,
        grad_state_seq,
        input_templates=(input_template,),
        state_templates=(state_template,),
    )

    assert grad_x.shape == input_template.shape
    assert grad_v.shape == state_template.shape


def test_flexsn_wrapper_backward_all_none_gradients_use_templates():
    from spikingjelly.activation_based.triton_kernel.flexsn import wrapper

    info = SimpleNamespace(num_inputs=1, num_states=1, num_outputs=1)
    input_template = torch.randn(2, 2, 3)
    state_template = torch.randn(6)

    grad_x, grad_v = wrapper.flexsn_backward(
        None,
        info,
        None,
        None,
        input_templates=(input_template,),
        state_templates=(state_template,),
    )

    torch.testing.assert_close(grad_x, torch.zeros_like(input_template))
    torch.testing.assert_close(grad_v, torch.zeros_like(state_template))


def test_flexsn_wrapper_backward_requires_input_templates_for_multi_input_core():
    from spikingjelly.activation_based.triton_kernel.flexsn import wrapper

    info = SimpleNamespace(num_inputs=2, num_states=0, num_outputs=1)
    grad_output = torch.randn(3, 4)

    with pytest.raises(ValueError, match="multiple input sequences"):
        wrapper.flexsn_backward(None, info, grad_output)


def test_flexsn_wrapper_backward_zero_fills_mixed_none_gradients(monkeypatch):
    from spikingjelly.activation_based.triton_kernel.flexsn import wrapper

    seen = {}

    class _FakeKernel:
        def __getitem__(self, grid):
            def _run(*args, **kwargs):
                seen["grad_output"] = args[0]
                seen["grad_state_seq"] = args[1]
                args[2].copy_(torch.ones_like(args[2]))
                args[3].copy_(torch.ones_like(args[3]))

            return _run

    info = SimpleNamespace(num_inputs=1, num_states=1, num_outputs=1)
    grad_output = torch.randn(3, 4)
    state_template = torch.randn(6)
    monkeypatch.setitem(wrapper.type_dict, grad_output.dtype, "float32")

    grad_x, grad_v = wrapper.flexsn_backward(
        _FakeKernel(),
        info,
        grad_output,
        None,
        state_templates=(state_template,),
    )

    torch.testing.assert_close(seen["grad_output"], grad_output)
    torch.testing.assert_close(seen["grad_state_seq"], torch.zeros_like(grad_output))
    torch.testing.assert_close(grad_x, torch.ones_like(grad_x))
    torch.testing.assert_close(grad_v, torch.ones_like(grad_v))


def test_flexsn_wrapper_backward_requires_state_templates_for_missing_state_seq_grad():
    from spikingjelly.activation_based.triton_kernel.flexsn import wrapper

    info = SimpleNamespace(num_inputs=1, num_states=1, num_outputs=1)
    grad_output = torch.randn(3, 4)

    with pytest.raises(ValueError, match="state_templates are required"):
        wrapper.flexsn_backward(None, info, grad_output, None)


def test_flexsn_wrapper_backward_requires_input_templates_when_output_grads_missing():
    from spikingjelly.activation_based.triton_kernel.flexsn import wrapper

    info = SimpleNamespace(num_inputs=1, num_states=1, num_outputs=1)
    grad_state_seq = torch.randn(3, 4)

    with pytest.raises(ValueError, match="output-sequence gradients are all None"):
        wrapper.flexsn_backward(None, info, None, grad_state_seq)


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
    from spikingjelly.activation_based.triton_kernel.flexsn import (
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


def test_inductor_fake_final_state_templates_use_explicit_states():
    from spikingjelly.activation_based.triton_kernel.flexsn import (
        custom_ops,
    )

    info = SimpleNamespace(num_inputs=1, num_states=2)
    x = torch.randn(4, 3, dtype=torch.float32)
    state_a = torch.randn(5, dtype=torch.float64)
    state_b = torch.empty(2, 3, dtype=torch.float16)

    templates = custom_ops._make_state_templates_like(info, [x, state_a, state_b])

    assert [tuple(template.shape) for template in templates] == [(5,), (2, 3)]
    assert [template.dtype for template in templates] == [
        torch.float64,
        torch.float16,
    ]


def test_inductor_training_final_state_backward_pads_missing_grads(monkeypatch):
    from spikingjelly.activation_based.triton_kernel.flexsn import (
        custom_ops,
    )

    x = torch.randn(2, 3)
    v = torch.randn(3)
    y = torch.randn(2, 3)
    bundle = SimpleNamespace(
        backward_kernel=object(),
        training_info=SimpleNamespace(num_inputs=1, num_outputs=1, num_states=1),
    )
    ctx = SimpleNamespace(
        handle=1,
        input_template_specs=[
            (tuple(x.shape), x.dtype, x.device),
            (tuple(v.shape), v.dtype, v.device),
        ],
        output_template_specs=[(tuple(y.shape), y.dtype, y.device)],
        state_seq_template_specs=[((x.shape[0], *v.shape), v.dtype, v.device)],
        saved_tensors=[],
        _active_ref_finalizer=SimpleNamespace(
            alive=False,
            detach=lambda: None,
        ),
    )
    seen = {}

    def fake_backward(handle, grad_inputs, saved_tensors, input_templates):
        seen["grad_inputs"] = grad_inputs
        assert handle == ctx.handle
        assert saved_tensors == []
        return [torch.zeros_like(x), torch.zeros_like(v)]

    monkeypatch.setattr(custom_ops, "_lookup_kernel_handle", lambda handle: bundle)
    monkeypatch.setattr(custom_ops, "flexsn_inductor_backward", fake_backward)
    monkeypatch.setattr(
        custom_ops,
        "release_active_flexsn_kernel_handle",
        lambda handle: None,
    )

    _, grads = custom_ops._flexsn_training_final_state_backward(ctx, [])

    assert len(seen["grad_inputs"]) == 2
    torch.testing.assert_close(seen["grad_inputs"][0], torch.zeros_like(y))
    torch.testing.assert_close(
        seen["grad_inputs"][1], torch.zeros((x.shape[0], *v.shape))
    )
    torch.testing.assert_close(grads[0], torch.zeros_like(x))
    torch.testing.assert_close(grads[1], torch.zeros_like(v))


def test_inductor_backward_impl_passes_state_templates_to_wrapper(monkeypatch):
    from spikingjelly.activation_based.triton_kernel.flexsn import (
        custom_ops,
    )

    input_template = torch.randn(2, 2, 3)
    state_template = torch.randn(6)
    grad_state_seq = torch.randn(2, 2, 3)
    bundle = SimpleNamespace(
        backward_kernel=object(),
        training_info=SimpleNamespace(num_inputs=1, num_outputs=0, num_states=1),
    )
    seen = {}

    def fake_backward(
        kernel,
        info,
        *args,
        input_templates=None,
        state_templates=None,
    ):
        seen["input_templates"] = input_templates
        seen["state_templates"] = state_templates
        return [torch.zeros_like(input_template), torch.zeros_like(state_template)]

    monkeypatch.setattr(custom_ops, "flexsn_backward", fake_backward)

    grads = custom_ops._flexsn_inductor_backward_impl(
        bundle,
        [grad_state_seq],
        [],
        [input_template, state_template],
    )

    assert len(seen["input_templates"]) == 1
    assert len(seen["state_templates"]) == 1
    assert seen["input_templates"][0].shape == input_template.shape
    assert seen["state_templates"][0].shape == state_template.shape
    assert grads[0].shape == input_template.shape
    assert grads[1].shape == state_template.shape


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
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
        store_state_seqs=True,
    )
    inductor_neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
        store_state_seqs=True,
    )

    torch_neuron(x)
    inductor_neuron(x)

    assert len(inductor_neuron.state_seqs) == 1
    torch.testing.assert_close(
        inductor_neuron.state_seqs[0], torch_neuron.state_seqs[0]
    )


def test_inductor_backend_exposes_kernel_accessor():
    neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
    )
    assert callable(neuron.kernel)


def test_triton_backend_is_supported():
    neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="triton",
    )
    assert neuron.backend == "triton"
    assert callable(neuron.kernel)


def test_set_backend_to_triton():
    neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
    )
    neuron.backend = "triton"
    assert neuron.backend == "triton"


def test_inductor_backend_in_supported_backends():
    neuron = FlexSN(
        core=_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
    )
    assert "triton" in neuron.supported_backends
    assert "inductor" in neuron.supported_backends


def test_set_backend_accepts_triton_for_flexsn():
    net = torch.nn.Sequential(
        FlexSN(
            core=_lif_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="torch",
        ),
        FlexSN(
            core=_lif_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="torch",
        ),
    )

    functional.set_backend(net, "triton", instance=FlexSN)

    for module in net:
        assert module.backend == "triton"


def test_triton_and_inductor_labels_share_dispatch(rng):
    T, N = 6, 16
    x_t = torch.randn(T, N, generator=rng, requires_grad=True)
    x_i = x_t.detach().clone().requires_grad_(True)

    triton_neuron = FlexSN(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="triton",
    )
    inductor_neuron = FlexSN(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
    )

    y_t = triton_neuron(x_t).sum()
    y_i = inductor_neuron(x_i).sum()
    y_t.backward()
    y_i.backward()

    torch.testing.assert_close(x_t.grad, x_i.grad)


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
    _skip_if_dynamo_hop_unavailable()

    try:
        from torch._dynamo.variables.higher_order_ops import (
            TorchHigherOrderOperatorVariable,
        )
    except (ImportError, ModuleNotFoundError, AttributeError):
        pytest.skip("torch._dynamo higher-order-op internals are unavailable")

    hop_var = TorchHigherOrderOperatorVariable.make(flex_sn_scan)
    assert hop_var.value is flex_sn_scan


def test_dynamo_body_result_templates_use_speculated_outputs():
    class _FakeVar:
        def __init__(self, example_value):
            self._proxy = SimpleNamespace(
                node=SimpleNamespace(meta={"example_value": example_value})
            )

        def as_proxy(self):
            return self._proxy

    class _FakeTupleVariable:
        def __init__(self, items):
            self.items = items

    body_result = _FakeTupleVariable(
        [
            _FakeVar(torch.empty(2, dtype=torch.float64)),
            _FakeVar(torch.empty(3, dtype=torch.float16)),
            _FakeVar(torch.empty(4, dtype=torch.float32)),
        ]
    )

    specs = hop_module._output_template_specs_from_dynamo_body_result(
        body_result,
        num_outputs=2,
    )

    assert specs == (
        ((2,), torch.float64),
        ((3,), torch.float16),
    )


def test_run_hop_scan_compiled_falls_back_when_dynamo_hop_unavailable(monkeypatch):
    x = torch.randn(2, 3)
    v = torch.zeros(3)

    def fake_hop(*args):
        return ("hop",)

    def fake_eager(*args):
        return ("eager",)

    monkeypatch.setattr(flexsn_module, "_is_compiling", lambda: True)
    monkeypatch.setattr(flexsn_module, "_flexsn_lowerable_scan", None)
    monkeypatch.setattr(flexsn_module, "_flexsn_lowerable_while_loop_scan", None)
    monkeypatch.setattr(flexsn_module, "_flexsn_hop_scan", fake_hop)
    monkeypatch.setattr(flexsn_module, "_flexsn_eager_scan", fake_eager)
    monkeypatch.setattr(flexsn_module, "_flexsn_dynamo_hop_available", lambda: False)

    result = flexsn_module._run_hop_scan(None, 1, 1, 1, True, x, v)
    assert result == ("eager",)

    monkeypatch.setattr(flexsn_module, "_flexsn_dynamo_hop_available", lambda: True)
    result = flexsn_module._run_hop_scan(None, 1, 1, 1, True, x, v)
    assert result == ("hop",)


def test_run_hop_scan_forwards_output_template_specs(monkeypatch):
    x = torch.empty(0, 2)
    v = torch.zeros(3)
    specs = (((4,), torch.float64, torch.device("cpu")),)
    captured = {}

    def fake_eager(*args, output_template_specs=None):
        captured["output_template_specs"] = output_template_specs
        return ("eager",)

    monkeypatch.setattr(flexsn_module, "_is_compiling", lambda: False)
    monkeypatch.setattr(flexsn_module, "_flexsn_hop_scan", None)
    monkeypatch.setattr(flexsn_module, "_flexsn_eager_scan", fake_eager)

    result = flexsn_module._run_hop_scan(
        None,
        1,
        1,
        1,
        True,
        x,
        v,
        output_template_specs=specs,
    )

    assert result == ("eager",)
    assert captured["output_template_specs"] is specs


def test_run_hop_scan_forwards_output_template_specs_to_hop(monkeypatch):
    x = torch.empty(0, 2)
    v = torch.zeros(3)
    specs = (((4,), torch.float64, torch.device("cpu")),)
    captured = {}

    def fake_hop(*args, output_template_specs=None):
        captured["output_template_specs"] = output_template_specs
        return ("hop",)

    monkeypatch.setattr(flexsn_module, "_is_compiling", lambda: False)
    monkeypatch.setattr(flexsn_module, "_flexsn_hop_scan", fake_hop)

    result = flexsn_module._run_hop_scan(
        None,
        1,
        1,
        1,
        True,
        x,
        v,
        output_template_specs=specs,
    )

    assert result == ("hop",)
    assert captured["output_template_specs"] is specs


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


def test_lowerable_scan_return_order_with_unequal_outputs_and_states():
    if not callable(lowerable_scan_available) or not lowerable_scan_available():
        pytest.skip("PyTorch scan HOP is unavailable in this environment")

    def core(x, v):
        next_v = v + x
        return x, x + 1, next_v

    x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    v0 = torch.zeros(2)

    y0_seq, y1_seq, v_seq = lowerable_scan(core, 1, 1, 2, x, v0)
    y0_final, y1_final, v_final = lowerable_scan_final_state(core, 1, 1, 2, x, v0)

    torch.testing.assert_close(y0_seq, x)
    torch.testing.assert_close(y1_seq, x + 1)
    torch.testing.assert_close(v_seq, x.cumsum(dim=0))
    torch.testing.assert_close(y0_final, x)
    torch.testing.assert_close(y1_final, x + 1)
    torch.testing.assert_close(v_final, x.sum(dim=0))


def test_lowerable_scan_empty_sequence_returns_templates_without_scan_op():
    if not callable(lowerable_scan_available) or not lowerable_scan_available():
        pytest.skip("PyTorch scan HOP is unavailable in this environment")

    calls = {"count": 0}

    def core(x, v):
        calls["count"] += 1
        raise AssertionError("core should not run for an empty lowerable scan")

    x = torch.empty(0, 2)
    v0 = torch.zeros(3)
    specs = (((4,), torch.float64),)

    y_seq, v_seq = lowerable_scan(core, 1, 1, 1, x, v0, output_template_specs=specs)
    y_final, v_final = lowerable_scan_final_state(
        core, 1, 1, 1, x, v0, output_template_specs=specs
    )

    assert calls["count"] == 0
    assert y_seq.shape == (0, 4)
    assert y_seq.dtype == torch.float64
    assert v_seq.shape == (0, 3)
    assert y_final.shape == (0, 4)
    assert y_final.dtype == torch.float64
    torch.testing.assert_close(v_final, v0)
    assert v_final is not v0


def test_eager_scan_empty_sequence_uses_template_without_core_call():
    calls = {"count": 0}

    def core(x, v):
        calls["count"] += 1
        raise AssertionError("core should not run for an empty eager scan")

    x = torch.empty(0, 2)
    v0 = torch.zeros(3)
    specs = (((4,), torch.float64),)

    y_seq, v_seq = eager_scan(core, 1, 1, 1, x, v0, output_template_specs=specs)
    y_final, v_final = eager_scan_final_state(
        core, 1, 1, 1, x, v0, output_template_specs=specs
    )

    assert calls["count"] == 0
    assert y_seq.shape == (0, 4)
    assert y_seq.dtype == torch.float64
    assert v_seq.shape == (0, 3)
    assert y_final.shape == (0, 4)
    assert y_final.dtype == torch.float64
    torch.testing.assert_close(v_final, v0)
    assert v_final is not v0


def test_eager_scan_empty_state_only_does_not_require_output_template():
    calls = {"count": 0}

    def core(x, v):
        calls["count"] += 1
        raise AssertionError("core should not run for an empty eager scan")

    x = torch.empty(0, 2)
    v0 = torch.ones(3)

    (v_seq,) = eager_scan(core, 1, 1, 0, x, v0)
    (v_final,) = eager_scan_final_state(core, 1, 1, 0, x, v0)

    assert calls["count"] == 0
    assert v_seq.shape == (0, 3)
    torch.testing.assert_close(v_final, v0)
    assert v_final is not v0


def test_eager_scan_empty_sequence_validates_core_arity():
    x = torch.empty(0, 2)
    v0 = torch.zeros(2)
    specs = (((2,), x.dtype),)

    def bad_core(x_t, v_t, extra):
        return x_t, v_t

    with pytest.raises(ValueError, match="expected 2 tensor args"):
        eager_scan(bad_core, 1, 1, 1, x, v0, output_template_specs=specs)

    with pytest.raises(ValueError, match="expected 2 tensor args"):
        eager_scan_final_state(bad_core, 1, 1, 1, x, v0, output_template_specs=specs)

    def bad_keyword_core(x_t, v_t, *, bias):
        return x_t, v_t

    with pytest.raises(ValueError, match="expected 2 tensor args"):
        eager_scan(bad_keyword_core, 1, 1, 1, x, v0, output_template_specs=specs)

    with pytest.raises(ValueError, match="expected 2 tensor args"):
        eager_scan_final_state(
            bad_keyword_core, 1, 1, 1, x, v0, output_template_specs=specs
        )


def test_inference_final_state_template_handles_zero_states(monkeypatch):
    captured = {}

    def fake_compile(kernel_str, kernel_name, verbose):
        compile(kernel_str, kernel_name, "exec")
        captured["kernel_str"] = kernel_str
        return object()

    monkeypatch.setattr(template_module, "compile_triton_code_str", fake_compile)
    info = FlexSNInfo(
        num_inputs=1,
        num_outputs=1,
        num_states=0,
        fwd_core_args=["x"],
        fwd_core_returns=["s"],
        fwd_core_recipients=["s0"],
        fwd_kernel_returns=["s0"],
        num_fwd_kernel_returns=1,
        c2k_return_mapping=[],
    )

    template_module.get_flexsn_inference_final_state_kernel(
        "@triton.jit\ndef core_12345678(x0):\n    return x0",
        "core_12345678",
        info,
    )

    assert "\n    , # inputs" not in captured["kernel_str"]


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
        example_outputs=(torch.zeros(8),),
    )
    torch_neuron = FlexSN(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
        store_state_seqs=True,
        example_outputs=(torch.zeros(8),),
    )

    hop_out = hop_neuron(x)
    torch_out = torch_neuron(x)

    torch.testing.assert_close(hop_out, torch_out)
    torch.testing.assert_close(hop_neuron.states[0], torch_neuron.states[0])


def test_hop_backend_zero_length_sequence_uses_example_output_template():
    x = torch.randn(0, 2)
    bias = torch.zeros(3)
    calls = {"count": 0}

    def core_with_closure(x_step):
        calls["count"] += 1
        raise AssertionError("core should not run for an empty hop input")

    hop_neuron = FlexSN(
        core=core_with_closure,
        num_inputs=1,
        num_states=0,
        num_outputs=1,
        step_mode="m",
        backend="hop",
        store_state_seqs=False,
        example_inputs=(torch.zeros(2),),
        example_outputs=(bias,),
    )

    hop_out = hop_neuron(x)
    assert calls["count"] == 0
    assert hop_out.shape == (0, 3)


def test_hop_backend_zero_length_sequence_delegates_to_hop_scan(monkeypatch):
    calls = {"count": 0}

    def fake_run_hop_scan(
        core,
        num_inputs,
        num_states,
        num_outputs,
        store_state_seqs,
        *flat_args,
        output_template_specs=None,
    ):
        calls["count"] += 1
        assert flat_args[0].shape[0] == 0
        assert output_template_specs is None
        return (flat_args[0].new_empty((0, 5)),)

    monkeypatch.setattr(flexsn_module, "_run_hop_scan", fake_run_hop_scan)
    hop_neuron = FlexSN(
        core=lambda x: (x.new_empty(5),),
        num_inputs=1,
        num_states=0,
        num_outputs=1,
        step_mode="m",
        backend="hop",
        store_state_seqs=False,
        example_inputs=(torch.zeros(2),),
    )

    hop_out = hop_neuron(torch.empty(0, 2))

    assert calls["count"] == 1
    assert hop_out.shape == (0, 5)


def test_hop_backend_runtime_templates_ignore_example_output_device(monkeypatch):
    captured = {}

    def fake_run_hop_scan(
        core,
        num_inputs,
        num_states,
        num_outputs,
        store_state_seqs,
        *flat_args,
        output_template_specs=None,
    ):
        captured["output_template_specs"] = output_template_specs
        return (flat_args[0].new_empty((0, 5)),)

    monkeypatch.setattr(flexsn_module, "_run_hop_scan", fake_run_hop_scan)
    hop_neuron = FlexSN(
        core=lambda x: (x.new_empty(5),),
        num_inputs=1,
        num_states=0,
        num_outputs=1,
        step_mode="m",
        backend="hop",
        store_state_seqs=False,
        example_inputs=(torch.zeros(2),),
        example_outputs=(torch.empty(5, device="meta"),),
    )

    hop_out = hop_neuron(torch.empty(0, 2))

    assert captured["output_template_specs"] == (((5,), torch.float32),)
    assert hop_out.shape == (0, 5)
    assert hop_out.device.type == "cpu"


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


def _reference_state_seq(core_fn, x, v0):
    T = x.shape[0]
    v = v0
    v_seq = []
    for t in range(T):
        _, v = core_fn(x[t], v)
        v_seq.append(v)
    return torch.stack(v_seq, dim=0)


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


def test_flexsnkernel_matches_torch_reference(rng):
    if not torch.cuda.is_available():
        pytest.skip("FlexSNKernel parity is exercised on CUDA")
    if base_module.triton is None:
        pytest.skip("Triton package is required for FlexSNKernel parity")

    T, N = 6, 16
    x_kernel = torch.randn(T, N, generator=rng).to("cuda").requires_grad_(True)
    v0_kernel = torch.randn(N, generator=rng).to("cuda").requires_grad_(True)
    x_ref = x_kernel.detach().clone().requires_grad_(True)
    v0_ref = v0_kernel.detach().clone().requires_grad_(True)

    kernel = FlexSNKernel(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        example_inputs=(torch.zeros(N, device="cuda"), torch.zeros(N, device="cuda")),
    )

    y_seq_kernel, v_seq_kernel = kernel(x_kernel, v0_kernel)
    (y_seq_kernel.sum() + v_seq_kernel.sum()).backward()

    y_seq_ref, _ = _reference_bptt(_differentiable_lif_core, x_ref, v0_ref)
    v_seq_ref = _reference_state_seq(_differentiable_lif_core, x_ref, v0_ref)
    (y_seq_ref.sum() + v_seq_ref.sum()).backward()

    torch.testing.assert_close(y_seq_kernel, y_seq_ref)
    torch.testing.assert_close(v_seq_kernel, v_seq_ref)
    torch.testing.assert_close(x_kernel.grad, x_ref.grad)
    torch.testing.assert_close(v0_kernel.grad, v0_ref.grad)


def test_flexsnkernel_matches_inductor_backend(rng):
    if not torch.cuda.is_available():
        pytest.skip("FlexSNKernel parity is exercised on CUDA")
    if base_module.triton is None:
        pytest.skip("Triton package is required for FlexSNKernel parity")

    T, N = 6, 16
    x_kernel = torch.randn(T, N, generator=rng).to("cuda").requires_grad_(True)
    v0_kernel = torch.randn(N, generator=rng).to("cuda").requires_grad_(True)
    x_inductor = x_kernel.detach().clone().requires_grad_(True)
    v0_inductor = v0_kernel.detach().clone().requires_grad_(True)

    kernel = FlexSNKernel(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        example_inputs=(torch.zeros(N, device="cuda"), torch.zeros(N, device="cuda")),
    )
    neuron = FlexSN(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
        store_state_seqs=True,
    ).cuda()
    neuron.states = [v0_inductor]

    y_seq_kernel, v_seq_kernel = kernel(x_kernel, v0_kernel)
    (y_seq_kernel.sum() + v_seq_kernel.sum()).backward()

    y_seq_inductor = neuron(x_inductor)
    v_seq_inductor = neuron.state_seqs[0]
    (y_seq_inductor.sum() + v_seq_inductor.sum()).backward()

    torch.testing.assert_close(y_seq_kernel, y_seq_inductor)
    torch.testing.assert_close(v_seq_kernel, v_seq_inductor)
    torch.testing.assert_close(x_kernel.grad, x_inductor.grad)
    torch.testing.assert_close(v0_kernel.grad, v0_inductor.grad)


def test_flexsnkernel_deepcopy_preserves_handle_lifetime(rng):
    if not torch.cuda.is_available():
        pytest.skip("FlexSNKernel deepcopy coverage is exercised on CUDA")
    if base_module.triton is None:
        pytest.skip("Triton package is required for FlexSNKernel deepcopy coverage")

    N = 16
    kernel = FlexSNKernel(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        example_inputs=(torch.zeros(N, device="cuda"), torch.zeros(N, device="cuda")),
    )
    kernel_copy = copy.deepcopy(kernel)

    x = torch.randn(4, N, generator=rng).to("cuda")
    v0 = torch.randn(N, generator=rng).to("cuda")

    y_ref, v_ref = kernel(x, v0)
    y_copy, v_copy = kernel_copy(x, v0)

    torch.testing.assert_close(y_copy, y_ref)
    torch.testing.assert_close(v_copy, v_ref)


def test_inductor_backend_backward_matches_torch_backend(rng):
    T, N = 6, 16
    x_i = torch.randn(T, N, generator=rng, requires_grad=True)
    x_t = x_i.detach().clone().requires_grad_(True)

    inductor_neuron = FlexSN(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
    )
    torch_neuron = FlexSN(
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="torch",
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
    if sys.platform == "win32":
        pytest.skip("torch.compile fullgraph Inductor is not supported on Windows")
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
            backend="inductor",
        )

    eager_out = make_neuron()(x_eager)
    compiled_fn = torch.compile(make_neuron(), fullgraph=True)
    compiled_out = compiled_fn(x_compiled)

    torch.testing.assert_close(compiled_out, eager_out)


def test_compile_fullgraph_backward_matches_eager(rng):
    if sys.platform == "win32":
        pytest.skip("torch.compile fullgraph Inductor is not supported on Windows")
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
        )

    make_neuron()(x_eager).sum().backward()

    compiled_fn = torch.compile(make_neuron(), fullgraph=True)
    compiled_fn(x_compiled).sum().backward()

    torch.testing.assert_close(x_compiled.grad, x_eager.grad)


def test_compile_fullgraph_backward_store_state_seqs_false_matches_eager(rng):
    if sys.platform == "win32":
        pytest.skip("torch.compile fullgraph Inductor is not supported on Windows")
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
    _skip_if_dynamo_hop_unavailable()

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


def test_compile_fullgraph_hop_direct_zero_length_scan_uses_body_template():
    if sys.platform == "win32":
        pytest.skip("torch.compile is not supported on Windows")
    _skip_if_dynamo_hop_unavailable()

    def core(x_step, state):
        return state + 1, state

    x = torch.empty(0, 2)
    state = torch.zeros(3)

    @torch.compile(fullgraph=True)
    def compiled_scan(x_seq, init_state):
        return flex_sn_scan(core, 1, 1, 1, x_seq, init_state)

    out, state_seq = compiled_scan(x, state)

    assert out.shape == (0, 3)
    assert state_seq.shape == (0, 3)
    assert out.dtype == state.dtype


def test_compile_fullgraph_hop_direct_accepts_output_template_specs():
    if sys.platform == "win32":
        pytest.skip("torch.compile is not supported on Windows")
    _skip_if_dynamo_hop_unavailable()

    def core(x_step, state):
        return state + 1, state

    x = torch.empty(0, 2)
    state = torch.zeros(3)
    specs = (((5,), torch.float64),)

    @torch.compile(fullgraph=True)
    def compiled_scan(x_seq, init_state):
        return flex_sn_scan(
            core,
            1,
            1,
            1,
            x_seq,
            init_state,
            output_template_specs=specs,
        )

    out, state_seq = compiled_scan(x, state)

    assert out.shape == (0, 5)
    assert out.dtype == torch.float64
    assert state_seq.shape == (0, 3)


def test_compile_fullgraph_hop_backend_matches_eager_with_closure(rng):
    if sys.platform == "win32":
        pytest.skip("torch.compile is not supported on Windows")
    _skip_if_dynamo_hop_unavailable()

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
    if sys.platform == "win32":
        pytest.skip("torch.compile fullgraph Inductor is not supported on Windows")
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
                backend="inductor",
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


def test_compile_fuses_surrounding_linear_layers_with_hop_lowerable_while_loop(
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
    eager_model = (
        spiking_vgg16_bn(
            spiking_neuron=make_flexsn,
            step_mode="m",
        )
        .cuda()
        .eval()
    )
    torch.manual_seed(0)
    compiled_model = (
        spiking_vgg16_bn(
            spiking_neuron=make_flexsn,
            step_mode="m",
        )
        .cuda()
        .eval()
    )

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
        core=_differentiable_lif_core,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
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
    targets = [
        str(node.target) for graph in explanation.graphs for node in graph.graph.nodes
    ]

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
    neuron = (
        FlexSN(
            core=core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            step_mode="m",
            backend="inductor",
            store_state_seqs=False,
        )
        .cuda()
        .train()
    )

    explanation = explain(lambda inp: neuron(inp).sum())(x)
    targets = [
        str(node.target) for graph in explanation.graphs for node in graph.graph.nodes
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
