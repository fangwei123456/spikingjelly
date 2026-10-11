import pytest
import torch

from spikingjelly.activation_based import neuron, surrogate


def _reference(x_seq, threshold, offset, reset, surrogate_function, detach_reset=False):
    threshold = torch.as_tensor(threshold, device=x_seq.device, dtype=x_seq.dtype)
    offset = torch.as_tensor(offset, device=x_seq.device, dtype=x_seq.dtype)
    v = (
        torch.zeros_like(x_seq[0])
        if reset is None
        else torch.full_like(x_seq[0], reset)
    )
    spikes, voltages = [], []
    for x in x_seq:
        h = v + x
        spike = surrogate_function(h + offset - threshold)
        spike_reset = spike.detach() if detach_reset else spike
        v = (
            h - spike_reset * threshold
            if reset is None
            else spike_reset * reset + (1 - spike_reset) * h
        )
        spikes.append(spike)
        voltages.append(v)
    return torch.stack(spikes), v, torch.stack(voltages)


def test_channelwise_sequence_matches_torch_reference():
    torch.manual_seed(17)
    x = torch.randn(4, 2, 5, 3).transpose(-1, -2)
    threshold = torch.tensor([0.6, 0.8, 1.0])
    offset = torch.tensor([0.0, 0.1, -0.2])
    node = neuron.ActivationAwareIFNode(
        v_threshold=threshold,
        v_offset=offset,
        channel_dim=1,
        step_mode="m",
        store_v_seq=True,
    ).eval()

    actual = node(x)
    expected, v_expected, v_seq_expected = _reference(
        x, threshold.view(1, 3, 1), offset.view(1, 3, 1), None, surrogate.Sigmoid()
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(node.v, v_expected)
    torch.testing.assert_close(node.v_seq, v_seq_expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_inference_automatically_uses_registered_operator():
    torch.manual_seed(18)
    x = torch.randn(4, 2, 5, 3, device="cuda").transpose(-1, -2)
    threshold = torch.tensor([0.6, 0.8, 1.0])
    offset = torch.tensor([0.0, 0.1, -0.2])
    node = (
        neuron.ActivationAwareIFNode(
            v_threshold=threshold,
            v_offset=offset,
            channel_dim=1,
            step_mode="m",
            store_v_seq=True,
        )
        .cuda()
        .eval()
    )

    with torch.inference_mode():
        actual = node(x)
        expected, v_expected, v_seq_expected = _reference(
            x,
            threshold.to(x.device).view(1, 3, 1),
            offset.to(x.device).view(1, 3, 1),
            None,
            surrogate.Sigmoid(),
        )

    assert torch.equal(actual, expected)
    torch.testing.assert_close(node.v, v_expected)
    torch.testing.assert_close(node.v_seq, v_seq_expected)
    assert not hasattr(node, "backend")


def test_training_preserves_surrogate_gradient_and_detach_reset():
    x = torch.tensor([[0.2, 0.4], [0.5, 0.7]], requires_grad=True)
    node = neuron.ActivationAwareIFNode(
        v_threshold=torch.tensor([0.7, 0.9]),
        v_offset=0.1,
        channel_dim=-1,
        detach_reset=False,
        step_mode="m",
    )
    output = node(x)
    grad = torch.autograd.grad(output.sum() + node.v.sum(), x)[0]
    expected, expected_v, _ = _reference(
        x,
        torch.tensor([0.7, 0.9]),
        0.1,
        None,
        node.surrogate_function,
    )
    expected_grad = torch.autograd.grad(expected.sum() + expected_v.sum(), x)[0]
    torch.testing.assert_close(grad, expected_grad)
    assert torch.isfinite(grad).all()


def test_store_v_seq_reset_and_constructor_contract():
    with pytest.raises(TypeError, match="backend"):
        neuron.ActivationAwareIFNode(backend="triton")

    node = neuron.ActivationAwareIFNode(step_mode="m", store_v_seq=True)
    x = torch.ones(2, 1, 3) * 0.2
    node(x)
    assert node.v_seq is not None
    node.store_v_seq = False
    assert node.v_seq is None
    node(x)
    assert node.v_seq is None
    node.reset()
    assert node.v == 0.0


def test_channel_validation_and_invalid_modes():
    with pytest.raises(ValueError, match="step_mode"):
        neuron.ActivationAwareIFNode(step_mode="invalid")

    node = neuron.ActivationAwareIFNode(
        v_threshold=torch.ones(4), channel_dim=1, step_mode="m"
    )
    with pytest.raises(ValueError, match="v_threshold has length"):
        node(torch.ones(2, 1, 3))
