import pytest
import torch

import spikingjelly.configure as configure
from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.neuron import integrate_and_fire


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
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_triton_multistep_matches_torch_channelwise_inference(self, dtype):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("CUDA device does not support bfloat16")
        torch.manual_seed(20260718)
        x_seq = torch.rand(16, 2, 3, 5, device="cuda", dtype=dtype) * 0.4
        kwargs = {
            "v_threshold": torch.linspace(0.7, 1.3, 5),
            "v_offset": torch.linspace(-0.1, 0.1, 5),
            "channel_dim": -1,
            "surrogate_function": surrogate.DeterministicPass(),
            "step_mode": "m",
            "store_v_seq": True,
        }
        reference = (
            neuron.ActivationAwareIFNode(**kwargs, backend="torch").cuda().eval()
        )
        candidate = (
            neuron.ActivationAwareIFNode(**kwargs, backend="triton").cuda().eval()
        )

        with torch.inference_mode():
            expected = reference(x_seq)
            actual = candidate(x_seq)

        assert torch.equal(actual, expected)
        atol = rtol = 1e-2 if dtype == torch.bfloat16 else 1e-6
        torch.testing.assert_close(candidate.v, reference.v, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            candidate.v_seq, reference.v_seq, atol=atol, rtol=rtol
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    @pytest.mark.parametrize("threshold_is_scalar", [False, True])
    def test_triton_multistep_supports_mixed_scalar_channelwise_parameters(
        self, threshold_is_scalar
    ):
        x_seq = torch.rand(8, 2, 3, 5, device="cuda") * 0.25
        threshold = 1.0 if threshold_is_scalar else torch.linspace(0.8, 1.2, 5)
        offset = torch.linspace(-0.1, 0.1, 5) if threshold_is_scalar else 0.05
        kwargs = {
            "v_threshold": threshold,
            "v_offset": offset,
            "channel_dim": -1,
            "surrogate_function": surrogate.DeterministicPass(),
            "step_mode": "m",
        }
        reference = (
            neuron.ActivationAwareIFNode(**kwargs, backend="torch").cuda().eval()
        )
        candidate = (
            neuron.ActivationAwareIFNode(**kwargs, backend="triton").cuda().eval()
        )

        with torch.inference_mode():
            expected = reference(x_seq)
            actual = candidate(x_seq)

        assert torch.equal(actual, expected)
        torch.testing.assert_close(candidate.v, reference.v, atol=1e-6, rtol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    def test_triton_multistep_matches_torch_scalar_parameters(self):
        torch.manual_seed(20260718)
        x_seq = torch.rand(32, 3, 7, device="cuda") * 0.3
        kwargs = {
            "v_threshold": 0.9,
            "v_offset": 0.1,
            "surrogate_function": surrogate.DeterministicPass(),
            "step_mode": "m",
        }
        reference = (
            neuron.ActivationAwareIFNode(**kwargs, backend="torch").cuda().eval()
        )
        candidate = (
            neuron.ActivationAwareIFNode(**kwargs, backend="triton").cuda().eval()
        )

        with torch.inference_mode():
            expected = reference(x_seq)
            actual = candidate(x_seq)

        assert torch.equal(actual, expected)
        torch.testing.assert_close(candidate.v, reference.v, atol=1e-6, rtol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    @pytest.mark.parametrize("v_reset", [None, 0.25])
    @pytest.mark.parametrize("store_v_seq", [False, True])
    def test_triton_preserves_multistep_state_and_reset(self, v_reset, store_v_seq):
        torch.manual_seed(20260718)
        first = (torch.rand(16, 2, 3, 10, device="cuda") * 0.4)[..., ::2]
        second = (torch.rand(17, 2, 3, 10, device="cuda") * 0.4)[..., ::2]
        assert not first.is_contiguous()
        kwargs = {
            "v_threshold": torch.tensor([0.8, 1.0, 1.2]),
            "v_offset": torch.tensor([-0.1, 0.0, 0.1]),
            "channel_dim": 1,
            "v_reset": v_reset,
            "surrogate_function": surrogate.DeterministicPass(),
            "step_mode": "m",
            "store_v_seq": store_v_seq,
        }
        reference = (
            neuron.ActivationAwareIFNode(**kwargs, backend="torch").cuda().eval()
        )
        candidate = (
            neuron.ActivationAwareIFNode(**kwargs, backend="triton").cuda().eval()
        )
        original_static_limit = configure.triton_neuron_kernel_static_range_max_T
        configure.triton_neuron_kernel_static_range_max_T = 16
        try:
            with torch.inference_mode():
                expected_first = reference(first)
                actual_first = candidate(first)
                expected_second = reference(second)
                actual_second = candidate(second)
                reference.reset()
                candidate.reset()
                expected_replay = reference(first)
                actual_replay = candidate(first)
        finally:
            configure.triton_neuron_kernel_static_range_max_T = original_static_limit

        assert torch.equal(actual_first, expected_first)
        assert torch.equal(actual_second, expected_second)
        assert torch.equal(actual_replay, expected_replay)
        assert torch.equal(actual_replay, actual_first)
        torch.testing.assert_close(candidate.v, reference.v, atol=1e-6, rtol=1e-6)
        if store_v_seq:
            torch.testing.assert_close(
                candidate.v_seq, reference.v_seq, atol=1e-6, rtol=1e-6
            )
        else:
            assert not hasattr(candidate, "v_seq")
            assert candidate.v.untyped_storage().nbytes() == (
                candidate.v.numel() * candidate.v.element_size()
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    def test_triton_inference_rejects_training_autograd_and_nonspiking(self):
        x_seq = torch.rand(4, 2, 3, device="cuda")
        training_node = neuron.ActivationAwareIFNode(
            step_mode="m", backend="triton"
        ).cuda()
        with pytest.raises(RuntimeError, match="inference only"):
            training_node(x_seq)

        autograd_node = (
            neuron.ActivationAwareIFNode(step_mode="m", backend="triton").cuda().eval()
        )
        with pytest.raises(RuntimeError, match="autograd"):
            autograd_node(x_seq.requires_grad_())

        nonspiking_node = (
            neuron.ActivationAwareIFNode(
                surrogate_function=surrogate.DeterministicPass(spiking=False),
                step_mode="m",
                backend="triton",
            )
            .cuda()
            .eval()
        )
        with pytest.raises(RuntimeError, match="spiking surrogate"):
            nonspiking_node(x_seq.detach())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    def test_triton_inference_rejects_unsupported_dtype(self):
        node = (
            neuron.ActivationAwareIFNode(step_mode="m", backend="triton").cuda().eval()
        )

        with pytest.raises(RuntimeError, match="float32 and bfloat16"):
            node(torch.rand(4, 2, 3, device="cuda", dtype=torch.float16))

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

    def test_disabling_store_v_seq_releases_previous_sequence_storage(self):
        node = neuron.ActivationAwareIFNode(
            v_threshold=1.0,
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
            store_v_seq=True,
        )
        x_seq = torch.full((8, 2, 3), 0.2)

        node(x_seq)
        assert isinstance(node.v_seq, torch.Tensor)
        node.store_v_seq = False
        node(x_seq)

        assert node.v_seq is None
        assert node.v.untyped_storage().nbytes() == (
            node.v.numel() * node.v.element_size()
        )

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

    def test_multistep_advertises_triton_backend(self):
        node = neuron.ActivationAwareIFNode(step_mode="m")

        assert node.supported_backends == ("torch", "triton")

    def test_multistep_triton_backend_is_cuda_only(self):
        pytest.importorskip("triton")
        node = neuron.ActivationAwareIFNode(
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
            backend="triton",
        )

        with pytest.raises(RuntimeError, match="CUDA"):
            node(torch.ones(2, 3))

    def test_single_step_rejects_triton_backend(self):
        with pytest.raises(ValueError, match="step_mode='m'"):
            neuron.ActivationAwareIFNode(step_mode="s", backend="triton")

    def test_step_mode_mutation_does_not_silently_fallback_from_triton(self):
        pytest.importorskip("triton")
        node = neuron.ActivationAwareIFNode(step_mode="m", backend="triton")
        node.step_mode = "s"

        with pytest.raises(RuntimeError, match="single-step.*torch"):
            node(torch.ones(2, 3))

    def test_triton_backend_rejects_an_unavailable_kernel(self, monkeypatch):
        monkeypatch.setattr(
            integrate_and_fire, "activation_aware_if_triton_kernel", None
        )

        with pytest.raises(RuntimeError, match="kernel is unavailable"):
            neuron.ActivationAwareIFNode(step_mode="m", backend="triton")

    @pytest.mark.parametrize(
        "x_seq",
        (
            torch.tensor(1.0),
            torch.empty(0, 2, 3),
            torch.empty(2, 0, 3),
        ),
    )
    def test_multistep_rejects_invalid_sequence_shapes(self, x_seq):
        node = neuron.ActivationAwareIFNode(step_mode="m")

        with pytest.raises(ValueError, match="non-empty shape.*T"):
            node(x_seq)

    @pytest.mark.parametrize("bad_backend", ["cupy", "inductor"])
    def test_rejects_non_torch_backend(self, bad_backend):
        with pytest.raises(ValueError, match="backend"):
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
