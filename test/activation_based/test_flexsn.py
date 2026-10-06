import copy

import pytest
import torch

from spikingjelly.activation_based.neuron.flexsn import FlexSN


def _lif_core(x, v, threshold):
    h = v + (x - v) * 0.5
    spike = torch.sigmoid(h - threshold)
    return spike, h * (1.0 - spike)


def _reference(x_seq, v, threshold):
    outputs, states = [], []
    for x in x_seq:
        output, v = _lif_core(x, v, threshold)
        outputs.append(output)
        states.append(v)
    return torch.stack(outputs), v, torch.stack(states)


def test_cpu_sequence_matches_reference_and_tracks_state_sequences():
    threshold = torch.nn.Parameter(torch.tensor(0.8))
    module = FlexSN(_lif_core, 1, (threshold,), store_state_seqs=True)
    x = torch.randn(4, 2, 3, requires_grad=True)
    initial = torch.randn(2, 3, requires_grad=True)
    module.states = (initial,)

    actual = module(x)
    expected, final, state_seq = _reference(x, initial, threshold)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(module.states[0], final)
    torch.testing.assert_close(module.state_seqs[0], state_seq)
    loss = actual.sum() + module.states[0].sum()
    reference_loss = expected.sum() + final.sum()
    got = torch.autograd.grad(loss, (x, initial, threshold))
    want = torch.autograd.grad(reference_loss, (x, initial, threshold))
    for actual_grad, expected_grad in zip(got, want, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_functional_forward_does_not_mutate_managed_state():
    module = FlexSN(_lif_core, 1, (torch.tensor(0.8),), store_state_seqs=True)
    x = torch.randn(3, 2)
    state = torch.zeros_like(x[0])

    outputs, updated = module.multi_step_functional_forward(
        (x,), (state,), static_inputs=module.static_inputs
    )

    assert module.states == (None,)
    assert module.state_seqs is None
    torch.testing.assert_close(outputs[0], module(x))
    torch.testing.assert_close(updated[0], module.states[0])
    assert module.state_seqs is not None


def test_cpu_scan_is_visible_to_fullgraph_compile():
    x = torch.randn(3, 2, 4)
    eager = FlexSN(_lif_core, 1, (torch.tensor(0.8),))(x)
    compiled = torch.compile(
        FlexSN(_lif_core, 1, (torch.tensor(0.8),)), fullgraph=True
    )(x)
    torch.testing.assert_close(compiled, eager)


def test_multi_input_output_and_state_counts():
    def core(x, y, v, w, scale):
        v_next = v + x
        w_next = w + y * scale
        return x + y, x - y, v_next, w_next

    module = FlexSN(core, 2, (torch.tensor(2.0),), store_state_seqs=True)
    x = torch.randn(3, 2)
    y = torch.randn(3, 2)
    outputs = module(x, y)
    assert len(outputs) == 2
    assert all(output.shape == x.shape for output in outputs)
    assert len(module.states) == 2
    assert len(module.state_seqs) == 2


def test_constructor_rejects_invalid_or_captured_core_values():
    with pytest.raises(ValueError, match="num_states"):
        FlexSN(_lif_core, -1)
    with pytest.raises(TypeError, match="static_inputs"):
        FlexSN(_lif_core, 1, (1.0,))

    captured = torch.tensor(1.0)

    def invalid_core(x, v):
        return x + captured, v

    with pytest.raises(TypeError, match="capture tensors"):
        FlexSN(invalid_core, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_automatically_builds_triton_for_supported_core():
    pytest.importorskip("triton")
    threshold = torch.nn.Parameter(torch.tensor(0.8, device="cuda"))
    module = FlexSN(_lif_core, 1, (threshold,), store_state_seqs=True).cuda()
    x = torch.randn(4, 2, 3, device="cuda", requires_grad=True)
    initial = torch.randn(2, 3, device="cuda", requires_grad=True)
    module.states = (initial,)

    actual = module(x)
    expected, final, state_seq = _reference(x, initial, threshold)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(module.states[0], final)
    torch.testing.assert_close(module.state_seqs[0], state_seq)
    torch.autograd.grad(actual.sum() + module.states[0].sum(), (x, initial, threshold))
    assert module._triton_handle is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_known_unsupported_lowering_falls_back_to_hop(monkeypatch):
    def unsupported(*args, **kwargs):
        raise NotImplementedError("core contains an unsupported Triton operator")

    module = FlexSN(_lif_core, 1, (torch.tensor(0.8, device="cuda"),)).cuda()
    monkeypatch.setattr(module, "_ensure_triton_runtime", unsupported)
    x = torch.randn(3, 2, device="cuda")

    actual = module(x)
    expected, _, _ = _reference(x, torch.zeros_like(x[0]), module.static_inputs[0])

    torch.testing.assert_close(actual, expected)
    assert module._triton_capability[x.device, x.dtype] is False
    assert module._triton_handle is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_runtime_lifetime_survives_copy_and_rebuilds():
    pytest.importorskip("triton")
    module = FlexSN(_lif_core, 1, (torch.tensor(0.8, device="cuda"),)).cuda()
    x = torch.randn(3, 2, device="cuda")
    module(x)
    assert module._triton_handle is not None

    copied = copy.deepcopy(module)
    assert copied._triton_handle is None
    torch.testing.assert_close(copied(x), module(x))
    assert copied._triton_handle is not None
