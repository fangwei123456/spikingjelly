import pytest
import torch
import torch.nn.functional as F

from spikingjelly.activation_based.ann2snn.operators import TDLayerNorm, TDSoftmax


class TestTDSoftmax:
    def test_shape_is_preserved(self):
        x_seq = torch.randn(4, 2, 3)
        op = TDSoftmax(dim=-1)

        y_seq = op(x_seq)

        assert y_seq.shape == x_seq.shape

    def test_cumulative_output_matches_softmax_on_cumulative_input(self):
        x_seq = torch.randn(5, 2, 4)
        op = TDSoftmax(dim=-1)

        y_seq = op(x_seq)
        expected = torch.softmax(x_seq.cumsum(dim=0), dim=-1)

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_final_cumulative_output_matches_softmax_on_total_input(self):
        x_seq = torch.randn(6, 3, 5)
        op = TDSoftmax(dim=-1)

        y_seq = op(x_seq)
        expected = torch.softmax(x_seq.sum(dim=0), dim=-1)

        assert torch.allclose(
            y_seq.cumsum(dim=0)[-1], expected, atol=1e-6, rtol=1e-6
        )

    def test_positive_non_last_softmax_dim(self):
        x_seq = torch.randn(5, 2, 3, 4)
        op = TDSoftmax(dim=2)

        y_seq = op(x_seq)
        expected = torch.softmax(x_seq.cumsum(dim=0), dim=2)

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_single_timestep_returns_softmax_of_input(self):
        x_seq = torch.randn(1, 2, 3)
        op = TDSoftmax(dim=-1)

        y_seq = op(x_seq)

        assert y_seq.shape == x_seq.shape
        assert torch.allclose(y_seq[0], torch.softmax(x_seq[0], dim=-1))

    def test_gradients_flow(self):
        x_seq = torch.randn(3, 4, requires_grad=True)
        op = TDSoftmax(dim=-1)

        y_seq = op(x_seq)
        y_seq.sum().backward()

        assert x_seq.grad is not None
        assert torch.isfinite(x_seq.grad).all()

    def test_negative_values_are_allowed(self):
        x_seq = torch.tensor(
            [
                [[5.0, 0.0]],
                [[-5.0, 5.0]],
            ]
        )
        op = TDSoftmax(dim=-1)

        y_seq = op(x_seq)

        assert (y_seq < 0).any()
        assert torch.allclose(
            y_seq.cumsum(dim=0),
            torch.softmax(x_seq.cumsum(dim=0), dim=-1),
        )

    def test_rejects_dim_zero(self):
        op = TDSoftmax(dim=0)

        with pytest.raises(ValueError, match="time dimension"):
            op(torch.randn(4, 2, 3))

    def test_rejects_negative_dim_resolving_to_zero(self):
        op = TDSoftmax(dim=-3)

        with pytest.raises(ValueError, match="time dimension"):
            op(torch.randn(4, 2, 3))

    def test_rejects_out_of_range_dim(self):
        op = TDSoftmax(dim=3)

        with pytest.raises(ValueError, match="dim must be in the range"):
            op(torch.randn(4, 2, 3))

    def test_rejects_one_dimensional_input(self):
        op = TDSoftmax(dim=-1)

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            op(torch.randn(4))

    def test_extra_repr(self):
        op = TDSoftmax(dim=-1)

        assert op.extra_repr() == "dim=-1"


class TestTDLayerNorm:
    def test_shape_is_preserved(self):
        x_seq = torch.randn(4, 2, 3)
        op = TDLayerNorm(normalized_shape=3)

        y_seq = op(x_seq)

        assert y_seq.shape == x_seq.shape

    def test_cumulative_output_matches_layer_norm_on_cumulative_input(self):
        x_seq = torch.randn(5, 2, 4)
        op = TDLayerNorm(normalized_shape=4)

        y_seq = op(x_seq)
        expected = F.layer_norm(
            x_seq.cumsum(dim=0),
            op.normalized_shape,
            op.weight,
            op.bias,
            op.eps,
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_final_cumulative_output_matches_layer_norm_on_total_input(self):
        x_seq = torch.randn(6, 3, 5)
        op = TDLayerNorm(normalized_shape=5)

        y_seq = op(x_seq)
        expected = F.layer_norm(
            x_seq.sum(dim=0),
            op.normalized_shape,
            op.weight,
            op.bias,
            op.eps,
        )

        assert torch.allclose(
            y_seq.cumsum(dim=0)[-1], expected, atol=1e-6, rtol=1e-6
        )

    def test_multi_dimensional_normalized_shape(self):
        x_seq = torch.randn(5, 2, 3, 4)
        op = TDLayerNorm(normalized_shape=(3, 4))

        y_seq = op(x_seq)
        expected = F.layer_norm(
            x_seq.cumsum(dim=0),
            op.normalized_shape,
            op.weight,
            op.bias,
            op.eps,
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_single_timestep_returns_layer_norm_of_input(self):
        x_seq = torch.randn(1, 2, 3)
        op = TDLayerNorm(normalized_shape=3)

        y_seq = op(x_seq)
        expected = F.layer_norm(
            x_seq[0],
            op.normalized_shape,
            op.weight,
            op.bias,
            op.eps,
        )

        assert y_seq.shape == x_seq.shape
        assert torch.allclose(y_seq[0], expected)

    def test_gradients_flow_to_input_and_affine_parameters(self):
        x_seq = torch.randn(3, 4, requires_grad=True)
        op = TDLayerNorm(normalized_shape=4)

        y_seq = op(x_seq)
        y_seq.square().sum().backward()

        assert x_seq.grad is not None
        assert torch.isfinite(x_seq.grad).all()
        assert op.weight.grad is not None
        assert torch.isfinite(op.weight.grad).all()
        assert op.bias.grad is not None
        assert torch.isfinite(op.bias.grad).all()

    def test_elementwise_affine_false_has_no_parameters(self):
        op = TDLayerNorm(normalized_shape=4, elementwise_affine=False)

        assert op.weight is None
        assert op.bias is None
        assert op.state_dict() == {}

    def test_elementwise_affine_false_ignores_bias_flag(self):
        op_true = TDLayerNorm(
            normalized_shape=4, elementwise_affine=False, bias=True
        )
        op_false = TDLayerNorm(
            normalized_shape=4, elementwise_affine=False, bias=False
        )

        assert op_true.bias is None
        assert op_false.bias is None
        assert op_true.state_dict() == op_false.state_dict() == {}

    def test_forward_with_elementwise_affine_false(self):
        x_seq = torch.randn(5, 2, 4)
        op = TDLayerNorm(normalized_shape=4, elementwise_affine=False)

        y_seq = op(x_seq)
        expected = F.layer_norm(
            x_seq.cumsum(dim=0),
            op.normalized_shape,
            None,
            None,
            op.eps,
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_affine_without_bias_has_weight_only(self):
        op = TDLayerNorm(normalized_shape=4, elementwise_affine=True, bias=False)

        assert op.weight is not None
        assert op.bias is None
        assert set(op.state_dict().keys()) == {"weight"}

    def test_reset_parameters_initializes_affine(self):
        op = TDLayerNorm(normalized_shape=4)
        with torch.no_grad():
            op.weight.fill_(2.0)
            op.bias.fill_(3.0)

        op.reset_parameters()

        assert torch.equal(op.weight, torch.ones(4))
        assert torch.equal(op.bias, torch.zeros(4))

    def test_device_and_dtype_initialize_parameters(self):
        op = TDLayerNorm(normalized_shape=4, device="cpu", dtype=torch.float64)

        assert op.weight.device.type == "cpu"
        assert op.bias.device.type == "cpu"
        assert op.weight.dtype == torch.float64
        assert op.bias.dtype == torch.float64

    def test_rejects_one_dimensional_input(self):
        op = TDLayerNorm(normalized_shape=4)

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            op(torch.randn(4))

    def test_invalid_trailing_shape_raises_clear_value_error(self):
        op = TDLayerNorm(normalized_shape=4)

        with pytest.raises(ValueError, match="trailing shape"):
            op(torch.randn(3, 2, 5))

    def test_extra_repr(self):
        op = TDLayerNorm(normalized_shape=(2, 3), eps=1e-4, bias=False)

        assert op.extra_repr() == (
            "(2, 3), eps=0.0001, elementwise_affine=True, bias=False"
        )
