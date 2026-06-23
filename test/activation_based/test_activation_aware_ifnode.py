import pytest
import torch

from spikingjelly.activation_based import neuron, surrogate


def _manual_activation_aware_if(
    x_seq,
    threshold,
    offset=0.0,
    channel_dim=-1,
    v_reset=None,
):
    threshold = torch.as_tensor(threshold, device=x_seq.device, dtype=x_seq.dtype)
    offset = torch.as_tensor(offset, device=x_seq.device, dtype=x_seq.dtype)

    def broadcast(param, x):
        if param.dim() == 0:
            return param
        dim = channel_dim if channel_dim >= 0 else channel_dim + x.dim()
        shape = [1] * x.dim()
        shape[dim] = param.numel()
        return param.view(shape)

    if v_reset is None:
        v = torch.zeros_like(x_seq[0])
    else:
        v = torch.full_like(x_seq[0], v_reset)
    y_seq = []
    v_seq = []
    for t in range(x_seq.shape[0]):
        x = x_seq[t]
        th = broadcast(threshold, x)
        off = broadcast(offset, x)
        h = v + x
        spike = (h + off >= th).to(x)
        if v_reset is None:
            v = h - spike * th
        else:
            v = spike * v_reset + (1.0 - spike) * h
        y_seq.append(spike)
        v_seq.append(v)
    return torch.stack(y_seq), torch.stack(v_seq)


class TestActivationAwareIFNode:
    def test_scalar_threshold_and_offset_match_manual_reference(self):
        node = neuron.ActivationAwareIFNode(
            v_threshold=1.5,
            v_offset=0.25,
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
        )
        x_seq = torch.tensor(
            [
                [[0.8, 1.0], [1.2, 0.2]],
                [[0.6, 0.2], [0.4, 1.4]],
                [[0.3, 0.9], [0.1, 0.2]],
            ]
        )

        y = node(x_seq)
        expected, expected_v = _manual_activation_aware_if(
            x_seq, threshold=1.5, offset=0.25
        )

        assert torch.allclose(y, expected)
        assert torch.allclose(node.v, expected_v[-1])

    def test_channel_last_broadcast_for_transformer_shapes(self):
        node = neuron.ActivationAwareIFNode(
            v_threshold=torch.tensor([1.0, 1.5, 2.0]),
            v_offset=torch.tensor([0.0, 0.25, -0.25]),
            channel_dim=-1,
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
        )
        x_seq = torch.tensor(
            [
                [[0.6, 1.0, 2.4], [1.1, 1.3, 1.0]],
                [[0.5, 0.4, 0.7], [0.2, 0.4, 1.3]],
            ]
        )

        y = node(x_seq)
        expected, _ = _manual_activation_aware_if(
            x_seq,
            threshold=torch.tensor([1.0, 1.5, 2.0]),
            offset=torch.tensor([0.0, 0.25, -0.25]),
            channel_dim=-1,
        )

        assert torch.allclose(y, expected)

    def test_channel_dim_one_broadcast_for_nchw_shapes(self):
        node = neuron.ActivationAwareIFNode(
            v_threshold=torch.tensor([1.0, 2.0]),
            v_offset=torch.tensor([0.0, 0.5]),
            channel_dim=1,
            surrogate_function=surrogate.DeterministicPass(),
        )
        x = torch.tensor(
            [
                [
                    [[0.8, 1.2], [0.1, 0.9]],
                    [[1.0, 1.6], [2.0, 0.4]],
                ]
            ]
        )

        y = node(x)
        expected, _ = _manual_activation_aware_if(
            x.unsqueeze(0),
            threshold=torch.tensor([1.0, 2.0]),
            offset=torch.tensor([0.0, 0.5]),
            channel_dim=1,
        )

        assert torch.allclose(y, expected[0])

    def test_hard_reset_and_store_v_seq(self):
        node = neuron.ActivationAwareIFNode(
            v_threshold=torch.tensor([1.0, 2.0]),
            channel_dim=-1,
            v_reset=0.25,
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
            store_v_seq=True,
        )
        x_seq = torch.tensor(
            [
                [[1.1, 1.5]],
                [[0.2, 0.7]],
                [[0.8, 0.1]],
            ]
        )

        y = node(x_seq)
        expected, expected_v = _manual_activation_aware_if(
            x_seq, threshold=torch.tensor([1.0, 2.0]), v_reset=0.25
        )

        assert torch.allclose(y, expected)
        assert torch.allclose(node.v_seq, expected_v)
        assert torch.allclose(node.v, expected_v[-1])

    def test_shape_change_reinitializes_v_to_v_reset_with_hard_reset(self):
        node = neuron.ActivationAwareIFNode(
            v_threshold=1.0,
            v_reset=0.25,
            surrogate_function=surrogate.DeterministicPass(),
        )

        node(torch.tensor([[1.1]]))
        y = node(torch.tensor([[0.3, 0.3]]))

        assert torch.allclose(y, torch.zeros_like(y))
        assert torch.allclose(node.v, torch.full((1, 2), 0.55))

    def test_reset_restores_v_to_v_reset_with_hard_reset(self):
        node = neuron.ActivationAwareIFNode(
            v_threshold=1.0,
            v_reset=0.25,
            surrogate_function=surrogate.DeterministicPass(),
        )

        node(torch.tensor([[1.1]]))
        node.reset()

        assert node.v == 0.25

    def test_dtype_and_device_follow_input(self):
        node = neuron.ActivationAwareIFNode(
            v_threshold=torch.tensor([1.0, 2.0], dtype=torch.float32),
            v_offset=torch.tensor([0.0, 0.5], dtype=torch.float32),
            surrogate_function=surrogate.DeterministicPass(),
        )
        x = torch.tensor([[1.0, 1.0]], dtype=torch.float64)

        y = node(x)

        assert y.dtype == torch.float64
        assert node.v.dtype == torch.float64
        assert node.v.device == x.device

    def test_autograd_smoke(self):
        node = neuron.ActivationAwareIFNode(
            v_threshold=torch.tensor([1.0, 2.0]),
            v_offset=torch.tensor([0.0, 0.25]),
            surrogate_function=surrogate.Sigmoid(),
        )
        x = torch.tensor([[0.9, 1.8]], requires_grad=True)

        y = node(x)
        loss = y.sum() + node.v.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_detach_reset_changes_reset_gradient_path(self):
        x0 = torch.tensor([[1.2]], requires_grad=True)
        x1 = x0.detach().clone().requires_grad_(True)
        attached = neuron.ActivationAwareIFNode(
            v_threshold=1.0,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=False,
        )
        detached = neuron.ActivationAwareIFNode(
            v_threshold=1.0,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True,
        )

        attached(x0)
        detached(x1)
        attached.v.sum().backward()
        detached.v.sum().backward()

        assert not torch.allclose(x0.grad, x1.grad)

    @pytest.mark.parametrize("bad_backend", ["cupy", "triton", "inductor"])
    def test_rejects_non_torch_backend(self, bad_backend):
        with pytest.raises(ValueError, match="backend='torch'"):
            neuron.ActivationAwareIFNode(backend=bad_backend)

    @pytest.mark.parametrize(
        "threshold",
        [0.0, -1.0, float("nan"), float("inf"), torch.tensor([1.0, 0.0])],
    )
    def test_rejects_invalid_threshold(self, threshold):
        with pytest.raises(ValueError, match="finite positive"):
            neuron.ActivationAwareIFNode(v_threshold=threshold)

    @pytest.mark.parametrize(
        "offset",
        [float("nan"), float("inf"), torch.tensor([0.0, float("nan")])],
    )
    def test_rejects_invalid_offset(self, offset):
        with pytest.raises(ValueError, match="finite"):
            neuron.ActivationAwareIFNode(v_offset=offset)

    def test_rejects_non_scalar_non_1d_parameters(self):
        with pytest.raises(ValueError, match="scalar or 1D"):
            neuron.ActivationAwareIFNode(v_threshold=torch.ones(2, 2))
        with pytest.raises(ValueError, match="scalar or 1D"):
            neuron.ActivationAwareIFNode(v_offset=torch.zeros(2, 2))

    def test_rejects_invalid_channel_dim_and_length(self):
        node = neuron.ActivationAwareIFNode(
            v_threshold=torch.ones(3),
            channel_dim=-1,
        )
        with pytest.raises(ValueError, match="has length 3"):
            node(torch.ones(2, 4))

        node = neuron.ActivationAwareIFNode(
            v_threshold=torch.ones(3),
            channel_dim=3,
        )
        with pytest.raises(ValueError, match="out of range"):
            node(torch.ones(2, 3))
