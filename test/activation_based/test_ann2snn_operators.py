import pytest
import torch
import torch.nn.functional as F

from spikingjelly.activation_based.ann2snn.operators import (
    TDGELU,
    TDLayerNorm,
    TDLinear,
    TDScaledDotProductAttention,
    TDSoftmax,
)


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

        assert torch.allclose(y_seq.cumsum(dim=0)[-1], expected, atol=1e-6, rtol=1e-6)

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

    def test_rejects_empty_time_dimension(self):
        op = TDSoftmax(dim=-1)

        with pytest.raises(ValueError, match="non-empty time dimension"):
            op(torch.empty(0, 2, 3))

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

        assert torch.allclose(y_seq.cumsum(dim=0)[-1], expected, atol=1e-6, rtol=1e-6)

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
        op_true = TDLayerNorm(normalized_shape=4, elementwise_affine=False, bias=True)
        op_false = TDLayerNorm(normalized_shape=4, elementwise_affine=False, bias=False)

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

    def test_rejects_empty_time_dimension(self):
        op = TDLayerNorm(normalized_shape=4)

        with pytest.raises(ValueError, match="non-empty time dimension"):
            op(torch.empty(0, 2, 4))

    def test_extra_repr(self):
        op = TDLayerNorm(normalized_shape=(2, 3), eps=1e-4, bias=False)

        assert op.extra_repr() == (
            "(2, 3), eps=0.0001, elementwise_affine=True, bias=False"
        )


class TestTDGELU:
    def test_shape_is_preserved(self):
        x_seq = torch.randn(4, 2, 3)
        op = TDGELU()

        y_seq = op(x_seq)

        assert y_seq.shape == x_seq.shape

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_cumulative_output_matches_gelu_on_cumulative_input(self, approximate):
        x_seq = torch.randn(5, 2, 4)
        op = TDGELU(approximate=approximate)

        y_seq = op(x_seq)
        expected = F.gelu(x_seq.cumsum(dim=0), approximate=approximate)

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_final_cumulative_output_matches_gelu_on_total_input(self):
        x_seq = torch.randn(6, 3, 5)
        op = TDGELU()

        y_seq = op(x_seq)
        expected = F.gelu(x_seq.sum(dim=0), approximate=op.approximate)

        assert torch.allclose(y_seq.cumsum(dim=0)[-1], expected, atol=1e-6, rtol=1e-6)

    def test_single_timestep_returns_gelu_of_input(self):
        x_seq = torch.randn(1, 2, 3)
        op = TDGELU()

        y_seq = op(x_seq)
        expected = F.gelu(x_seq[0], approximate=op.approximate)

        assert y_seq.shape == x_seq.shape
        assert torch.allclose(y_seq[0], expected)

    def test_gradients_flow(self):
        x_seq = torch.randn(3, 4)
        op = TDGELU()

        x_seq_ref = x_seq.clone().detach().requires_grad_()
        y_cum_ref = F.gelu(x_seq_ref.cumsum(dim=0), approximate=op.approximate)
        y_seq_ref = torch.empty_like(y_cum_ref)
        y_seq_ref[0] = y_cum_ref[0]
        y_seq_ref[1:] = y_cum_ref[1:] - y_cum_ref[:-1]
        y_seq_ref.sum().backward()

        x_seq = x_seq.clone().detach().requires_grad_()
        y_seq = op(x_seq)
        y_seq.sum().backward()

        assert x_seq.grad is not None
        assert torch.isfinite(x_seq.grad).all()
        assert torch.allclose(x_seq.grad, x_seq_ref.grad, atol=1e-6, rtol=1e-6)

    def test_negative_values_are_allowed(self):
        x_seq = torch.tensor(
            [
                [[-2.0, 1.0]],
                [[3.0, -1.0]],
            ]
        )
        op = TDGELU()

        y_seq = op(x_seq)

        assert (y_seq < 0).any()
        assert torch.allclose(
            y_seq.cumsum(dim=0),
            F.gelu(x_seq.cumsum(dim=0), approximate=op.approximate),
        )

    def test_rejects_one_dimensional_input(self):
        op = TDGELU()

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            op(torch.randn(4))

    def test_rejects_empty_time_dimension(self):
        op = TDGELU()

        with pytest.raises(ValueError, match="non-empty time dimension"):
            op(torch.empty(0, 2, 3))

    def test_rejects_invalid_approximate(self):
        with pytest.raises(ValueError, match="approximate must be 'none' or 'tanh'"):
            TDGELU(approximate="foo")

    def test_extra_repr(self):
        op = TDGELU(approximate="tanh")

        assert op.extra_repr() == "approximate='tanh'"


class TestTDLinear:
    def test_shape_is_preserved_for_batched_input(self):
        x_seq = torch.randn(4, 2, 3)
        op = TDLinear(3, 5)

        y_seq = op(x_seq)

        assert y_seq.shape == (4, 2, 5)

    def test_higher_rank_input_shape(self):
        x_seq = torch.randn(4, 2, 6, 3)
        op = TDLinear(3, 5)

        y_seq = op(x_seq)

        assert y_seq.shape == (4, 2, 6, 5)

    def test_cumulative_output_matches_linear_on_cumulative_input(self):
        x_seq = torch.randn(5, 2, 4)
        op = TDLinear(4, 3)

        y_seq = op(x_seq)
        expected = F.linear(x_seq.cumsum(dim=0), op.weight, op.bias)

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_final_cumulative_output_matches_linear_on_total_input(self):
        x_seq = torch.randn(6, 3, 5)
        op = TDLinear(5, 4)

        y_seq = op(x_seq)
        expected = F.linear(x_seq.sum(dim=0), op.weight, op.bias)

        assert torch.allclose(
            y_seq.cumsum(dim=0)[-1], expected, atol=1e-6, rtol=1e-6
        )

    def test_single_timestep_returns_linear_of_input(self):
        x_seq = torch.randn(1, 2, 3)
        op = TDLinear(3, 5)

        y_seq = op(x_seq)
        expected = F.linear(x_seq[0], op.weight, op.bias)

        assert y_seq.shape == (1, 2, 5)
        assert torch.allclose(y_seq[0], expected, atol=1e-6, rtol=1e-6)

    def test_bias_is_not_repeatedly_accumulated(self):
        x_seq = torch.zeros(4, 2, 3)
        op = TDLinear(3, 5)
        with torch.no_grad():
            op.weight.zero_()
            op.bias.copy_(torch.arange(5, dtype=x_seq.dtype))

        y_seq = op(x_seq)

        assert torch.allclose(y_seq.cumsum(dim=0)[-1], op.bias.expand(2, 5))
        assert torch.allclose(y_seq[0], op.bias.expand(2, 5))
        assert torch.count_nonzero(y_seq[1:]) == 0

    def test_bias_false_has_weight_only(self):
        op = TDLinear(3, 5, bias=False)

        assert op.weight is not None
        assert op.bias is None
        assert set(op.state_dict().keys()) == {"weight"}

    def test_reset_parameters_preserves_shapes_and_reinitializes_values(self):
        op = TDLinear(3, 5)
        with torch.no_grad():
            op.weight.fill_(1.0)
            op.bias.fill_(1.0)

        op.reset_parameters()

        assert op.weight.shape == (5, 3)
        assert op.bias.shape == (5,)
        assert not torch.equal(op.weight, torch.ones_like(op.weight))
        assert not torch.equal(op.bias, torch.ones_like(op.bias))

    def test_device_and_dtype_initialize_parameters(self):
        op = TDLinear(3, 5, device="cpu", dtype=torch.float64)

        assert op.weight.device.type == "cpu"
        assert op.bias.device.type == "cpu"
        assert op.weight.dtype == torch.float64
        assert op.bias.dtype == torch.float64

    def test_gradients_match_reference(self):
        x_seq = torch.randn(3, 2, 4)
        op = TDLinear(4, 5)

        x_ref = x_seq.clone().detach().requires_grad_()
        weight_ref = op.weight.clone().detach().requires_grad_()
        bias_ref = op.bias.clone().detach().requires_grad_()
        y_cum_ref = F.linear(x_ref.cumsum(dim=0), weight_ref, bias_ref)
        y_seq_ref = torch.empty_like(y_cum_ref)
        y_seq_ref[0] = y_cum_ref[0]
        y_seq_ref[1:] = y_cum_ref[1:] - y_cum_ref[:-1]
        y_seq_ref.square().sum().backward()

        x_seq = x_seq.clone().detach().requires_grad_()
        y_seq = op(x_seq)
        y_seq.square().sum().backward()

        assert torch.allclose(x_seq.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
        assert torch.allclose(op.weight.grad, weight_ref.grad, atol=1e-6, rtol=1e-6)
        assert torch.allclose(op.bias.grad, bias_ref.grad, atol=1e-6, rtol=1e-6)

    def test_rejects_one_dimensional_input(self):
        op = TDLinear(3, 5)

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            op(torch.randn(3))

    def test_rejects_empty_time_dimension(self):
        op = TDLinear(3, 5)

        with pytest.raises(ValueError, match="non-empty time dimension"):
            op(torch.empty(0, 2, 3))

    def test_invalid_trailing_feature_size_raises_from_pytorch(self):
        op = TDLinear(3, 5)

        with pytest.raises(RuntimeError):
            op(torch.randn(4, 2, 4))

    def test_extra_repr(self):
        op = TDLinear(3, 5, bias=False)

        assert op.extra_repr() == "in_features=3, out_features=5, bias=False"


class TestTDScaledDotProductAttention:
    def test_shape_is_preserved(self):
        q_seq = torch.randn(4, 2, 3, 5)
        k_seq = torch.randn(4, 2, 6, 5)
        v_seq = torch.randn(4, 2, 6, 7)
        op = TDScaledDotProductAttention()

        y_seq = op(q_seq, k_seq, v_seq)

        assert y_seq.shape == (4, 2, 3, 7)

    def test_cumulative_output_matches_sdpa_on_cumulative_input(self):
        q_seq = torch.randn(5, 2, 3, 4)
        k_seq = torch.randn(5, 2, 6, 4)
        v_seq = torch.randn(5, 2, 6, 7)
        op = TDScaledDotProductAttention()

        y_seq = op(q_seq, k_seq, v_seq)
        expected = F.scaled_dot_product_attention(
            q_seq.cumsum(dim=0),
            k_seq.cumsum(dim=0),
            v_seq.cumsum(dim=0),
            dropout_p=0.0,
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_final_cumulative_output_matches_sdpa_on_total_input(self):
        q_seq = torch.randn(6, 2, 3, 4)
        k_seq = torch.randn(6, 2, 5, 4)
        v_seq = torch.randn(6, 2, 5, 7)
        op = TDScaledDotProductAttention()

        y_seq = op(q_seq, k_seq, v_seq)
        expected = F.scaled_dot_product_attention(
            q_seq.sum(dim=0),
            k_seq.sum(dim=0),
            v_seq.sum(dim=0),
            dropout_p=0.0,
        )

        assert torch.allclose(y_seq.cumsum(dim=0)[-1], expected, atol=1e-6, rtol=1e-6)

    def test_single_timestep_returns_sdpa_of_input(self):
        q_seq = torch.randn(1, 2, 3, 4)
        k_seq = torch.randn(1, 2, 5, 4)
        v_seq = torch.randn(1, 2, 5, 7)
        op = TDScaledDotProductAttention()

        y_seq = op(q_seq, k_seq, v_seq)
        expected = F.scaled_dot_product_attention(
            q_seq[0],
            k_seq[0],
            v_seq[0],
            dropout_p=0.0,
        )

        assert y_seq.shape == (1, 2, 3, 7)
        assert torch.allclose(y_seq[0], expected, atol=1e-6, rtol=1e-6)

    def test_multi_head_style_shape(self):
        q_seq = torch.randn(4, 2, 3, 5, 4)
        k_seq = torch.randn(4, 2, 3, 6, 4)
        v_seq = torch.randn(4, 2, 3, 6, 7)
        op = TDScaledDotProductAttention()

        y_seq = op(q_seq, k_seq, v_seq)
        expected = F.scaled_dot_product_attention(
            q_seq.cumsum(dim=0),
            k_seq.cumsum(dim=0),
            v_seq.cumsum(dim=0),
            dropout_p=0.0,
        )

        assert y_seq.shape == (4, 2, 3, 5, 7)
        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize(
        "attn_mask",
        [
            torch.tensor(
                [
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, True, True, True],
                ]
            ),
            torch.tensor(
                [
                    [0.0, 0.0, float("-inf"), float("-inf")],
                    [0.0, 0.0, 0.0, float("-inf")],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ],
    )
    def test_supports_attention_mask(self, attn_mask):
        q_seq = torch.randn(5, 2, 3, 4)
        k_seq = torch.randn(5, 2, 4, 4)
        v_seq = torch.randn(5, 2, 4, 6)
        op = TDScaledDotProductAttention()

        y_seq = op(q_seq, k_seq, v_seq, attn_mask=attn_mask)
        expected = F.scaled_dot_product_attention(
            q_seq.cumsum(dim=0),
            k_seq.cumsum(dim=0),
            v_seq.cumsum(dim=0),
            attn_mask=attn_mask,
            dropout_p=0.0,
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_supports_causal_attention(self):
        q_seq = torch.randn(4, 2, 5, 4)
        k_seq = torch.randn(4, 2, 5, 4)
        v_seq = torch.randn(4, 2, 5, 6)
        op = TDScaledDotProductAttention(is_causal=True)

        y_seq = op(q_seq, k_seq, v_seq)
        expected = F.scaled_dot_product_attention(
            q_seq.cumsum(dim=0),
            k_seq.cumsum(dim=0),
            v_seq.cumsum(dim=0),
            dropout_p=0.0,
            is_causal=True,
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_supports_custom_scale(self):
        q_seq = torch.randn(4, 2, 3, 4)
        k_seq = torch.randn(4, 2, 5, 4)
        v_seq = torch.randn(4, 2, 5, 6)
        op = TDScaledDotProductAttention(scale=0.25)

        y_seq = op(q_seq, k_seq, v_seq)
        expected = F.scaled_dot_product_attention(
            q_seq.cumsum(dim=0),
            k_seq.cumsum(dim=0),
            v_seq.cumsum(dim=0),
            dropout_p=0.0,
            scale=0.25,
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_gradients_match_reference(self):
        q_seq = torch.randn(3, 2, 4, 5)
        k_seq = torch.randn(3, 2, 6, 5)
        v_seq = torch.randn(3, 2, 6, 7)
        op = TDScaledDotProductAttention()

        q_ref = q_seq.clone().detach().requires_grad_()
        k_ref = k_seq.clone().detach().requires_grad_()
        v_ref = v_seq.clone().detach().requires_grad_()
        y_cum_ref = F.scaled_dot_product_attention(
            q_ref.cumsum(dim=0),
            k_ref.cumsum(dim=0),
            v_ref.cumsum(dim=0),
            dropout_p=0.0,
        )
        y_seq_ref = torch.empty_like(y_cum_ref)
        y_seq_ref[0] = y_cum_ref[0]
        y_seq_ref[1:] = y_cum_ref[1:] - y_cum_ref[:-1]
        y_seq_ref.square().sum().backward()

        q_seq = q_seq.clone().detach().requires_grad_()
        k_seq = k_seq.clone().detach().requires_grad_()
        v_seq = v_seq.clone().detach().requires_grad_()
        y_seq = op(q_seq, k_seq, v_seq)
        y_seq.square().sum().backward()

        assert torch.allclose(q_seq.grad, q_ref.grad, atol=1e-6, rtol=1e-6)
        assert torch.allclose(k_seq.grad, k_ref.grad, atol=1e-6, rtol=1e-6)
        assert torch.allclose(v_seq.grad, v_ref.grad, atol=1e-6, rtol=1e-6)

    def test_negative_values_are_allowed(self):
        q_seq = torch.zeros(2, 1, 1, 1)
        k_seq = torch.zeros(2, 1, 2, 1)
        v_seq = torch.tensor([[[[2.0], [2.0]]], [[[-3.0], [-3.0]]]])
        op = TDScaledDotProductAttention()

        y_seq = op(q_seq, k_seq, v_seq)

        assert (y_seq < 0).any()

    def test_rejects_input_with_too_few_dimensions(self):
        op = TDScaledDotProductAttention()

        with pytest.raises(ValueError, match="at least 3 dimensions"):
            op(torch.randn(4, 3), torch.randn(4, 3, 5), torch.randn(4, 3, 6))

    def test_rejects_empty_time_dimension(self):
        op = TDScaledDotProductAttention()

        with pytest.raises(ValueError, match="non-empty time dimension"):
            op(
                torch.empty(0, 2, 3, 4),
                torch.empty(0, 2, 5, 4),
                torch.empty(0, 2, 5, 6),
            )

    def test_rejects_mismatched_time_lengths(self):
        op = TDScaledDotProductAttention()

        with pytest.raises(ValueError, match="same time length"):
            op(
                torch.randn(4, 2, 3, 4),
                torch.randn(5, 2, 6, 4),
                torch.randn(4, 2, 6, 7),
            )

    def test_rejects_mask_with_causal_attention(self):
        op = TDScaledDotProductAttention(is_causal=True)

        with pytest.raises(ValueError, match="attn_mask"):
            op(
                torch.randn(4, 2, 3, 4),
                torch.randn(4, 2, 3, 4),
                torch.randn(4, 2, 3, 6),
                attn_mask=torch.ones(3, 3, dtype=torch.bool),
            )

    def test_invalid_attention_shape_raises_from_pytorch(self):
        op = TDScaledDotProductAttention()

        with pytest.raises(RuntimeError):
            op(
                torch.randn(4, 2, 3, 4),
                torch.randn(4, 2, 5, 6),
                torch.randn(4, 2, 5, 7),
            )

    def test_extra_repr(self):
        op = TDScaledDotProductAttention(is_causal=True, scale=0.25)

        assert op.extra_repr() == "is_causal=True, scale=0.25"
