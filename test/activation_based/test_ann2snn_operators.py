import pytest
import torch
import torch.nn.functional as F

from spikingjelly.activation_based import base, functional
from spikingjelly.activation_based.ann2snn.operators import (
    SNNElementWiseProduct,
    SNNMatrixOperator,
    TDModule,
    TDGELU,
    TDLayerNorm,
    TDLinear,
    TDMultiheadAttention,
    TDScaledDotProductAttention,
    TDSoftmax,
)
from spikingjelly.activation_based import neuron, surrogate


def _copy_td_mha_to_ann(td_op, ann_op):
    with torch.no_grad():
        ann_op.in_proj_weight.copy_(
            torch.cat(
                [
                    td_op.q_proj.weight,
                    td_op.k_proj.weight,
                    td_op.v_proj.weight,
                ],
                dim=0,
            )
        )
        if ann_op.in_proj_bias is not None:
            ann_op.in_proj_bias.copy_(
                torch.cat(
                    [
                        td_op.q_proj.bias,
                        td_op.k_proj.bias,
                        td_op.v_proj.bias,
                    ],
                    dim=0,
                )
            )
        ann_op.out_proj.weight.copy_(td_op.out_proj.weight)
        if ann_op.out_proj.bias is not None:
            ann_op.out_proj.bias.copy_(td_op.out_proj.bias)


def _mha_cumulative_reference(td_op, query_seq, key_seq, value_seq, **kwargs):
    ann_op = torch.nn.MultiheadAttention(
        td_op.embed_dim,
        td_op.num_heads,
        dropout=0.0,
        bias=td_op.q_proj.bias is not None,
        batch_first=True,
    )
    _copy_td_mha_to_ann(td_op, ann_op)
    q_cum = query_seq.cumsum(dim=0)
    k_cum = key_seq.cumsum(dim=0)
    v_cum = value_seq.cumsum(dim=0)
    return torch.stack(
        [
            ann_op(q_cum[t], k_cum[t], v_cum[t], need_weights=False, **kwargs)[0]
            for t in range(query_seq.shape[0])
        ]
    )


def _temporal_difference(y_cum):
    y_seq = torch.empty_like(y_cum)
    y_seq[0] = y_cum[0]
    y_seq[1:] = y_cum[1:] - y_cum[:-1]
    return y_seq


def _snn_matrix_loop_reference(a_seq, b_seq):
    a_sum = None
    b_sum = None
    y_seq = []
    for t in range(a_seq.shape[0]):
        if t == 0:
            a_sum = a_seq[t]
            b_sum = b_seq[t]
            y_t = torch.matmul(a_sum, b_sum)
        else:
            prev = torch.matmul(a_sum, b_sum)
            a_sum = a_sum + a_seq[t]
            b_sum = b_sum + b_seq[t]
            y_t = torch.matmul(a_sum, b_sum) - prev
        y_seq.append(y_t)
    return torch.stack(y_seq)


def _ann_mha_reference(td_op, query, key, value, **kwargs):
    ann_op = torch.nn.MultiheadAttention(
        td_op.embed_dim,
        td_op.num_heads,
        dropout=0.0,
        bias=td_op.q_proj.bias is not None,
        batch_first=True,
    )
    _copy_td_mha_to_ann(td_op, ann_op)
    return ann_op(query, key, value, need_weights=False, **kwargs)[0]


def test_td_modules_support_step_mode_switching():
    modules = [
        TDSoftmax(),
        TDLayerNorm(3),
        TDGELU(),
        TDLinear(3, 4),
        SNNMatrixOperator(),
        SNNElementWiseProduct(),
        TDScaledDotProductAttention(),
        TDMultiheadAttention(embed_dim=4, num_heads=2),
    ]

    for module in modules:
        assert isinstance(module, TDModule)
        assert isinstance(module, base.StepModule)
        assert module.step_mode == "m"

        functional.set_step_mode(module, "s")
        assert module.step_mode == "s"
        functional.set_step_mode(module, "m")
        assert module.step_mode == "m"

        with pytest.raises(ValueError, match="step_mode can only be"):
            module.step_mode = "bogus"


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

    def test_single_step_mode_matches_softmax(self):
        x = torch.randn(2, 3)
        op = TDSoftmax(dim=0, step_mode="s")

        y = op(x)

        assert torch.allclose(y, torch.softmax(x, dim=0))

    def test_single_step_softmax_scalar_matches_torch(self):
        x = torch.tensor(1.0)
        op = TDSoftmax(dim=0, step_mode="s")

        y = op(x)

        assert torch.allclose(y, torch.softmax(x, dim=0))

    def test_one_step_multi_step_matches_single_step(self):
        x = torch.randn(2, 3)
        op = TDSoftmax(dim=-1)

        y_seq = op(x.unsqueeze(0))
        functional.set_step_mode(op, "s")

        assert torch.allclose(y_seq[0], op(x))

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

    def test_single_step_mode_matches_layer_norm(self):
        x = torch.randn(2, 3)
        op = TDLayerNorm(normalized_shape=3, step_mode="s")

        y = op(x)
        expected = F.layer_norm(x, op.normalized_shape, op.weight, op.bias, op.eps)

        assert torch.allclose(y, expected)

    def test_one_step_multi_step_matches_single_step(self):
        x = torch.randn(2, 3)
        op = TDLayerNorm(normalized_shape=3)

        y_seq = op(x.unsqueeze(0))
        functional.set_step_mode(op, "s")

        assert torch.allclose(y_seq[0], op(x))

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

    def test_single_step_mode_matches_gelu(self):
        x = torch.randn(2, 3)
        op = TDGELU(approximate="tanh", step_mode="s")

        y = op(x)

        assert torch.allclose(y, F.gelu(x, approximate="tanh"))

    def test_one_step_multi_step_matches_single_step(self):
        x = torch.randn(2, 3)
        op = TDGELU(approximate="tanh")

        y_seq = op(x.unsqueeze(0))
        functional.set_step_mode(op, "s")

        assert torch.allclose(y_seq[0], op(x))

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

    def test_bias_false_matches_per_timestep_linear(self):
        x_seq = torch.randn(5, 2, 4)
        op = TDLinear(4, 3, bias=False)

        y_seq = op(x_seq)
        expected = F.linear(x_seq, op.weight, None)

        assert torch.allclose(y_seq, expected, atol=1e-6, rtol=1e-6)

    def test_final_cumulative_output_matches_linear_on_total_input(self):
        x_seq = torch.randn(6, 3, 5)
        op = TDLinear(5, 4)

        y_seq = op(x_seq)
        expected = F.linear(x_seq.sum(dim=0), op.weight, op.bias)

        assert torch.allclose(y_seq.cumsum(dim=0)[-1], expected, atol=1e-6, rtol=1e-6)

    def test_single_timestep_returns_linear_of_input(self):
        x_seq = torch.randn(1, 2, 3)
        op = TDLinear(3, 5)

        y_seq = op(x_seq)
        expected = F.linear(x_seq[0], op.weight, op.bias)

        assert y_seq.shape == (1, 2, 5)
        assert torch.allclose(y_seq[0], expected, atol=1e-6, rtol=1e-6)

    def test_single_step_mode_matches_linear(self):
        x = torch.randn(2, 3)
        op = TDLinear(3, 5, step_mode="s")

        y = op(x)
        expected = F.linear(x, op.weight, op.bias)

        assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)

    def test_one_step_multi_step_matches_single_step(self):
        x = torch.randn(2, 3)
        op = TDLinear(3, 5)

        y_seq = op(x.unsqueeze(0))
        functional.set_step_mode(op, "s")

        assert torch.allclose(y_seq[0], op(x), atol=1e-6, rtol=1e-6)

    def test_bias_is_not_repeatedly_accumulated(self):
        x_seq = torch.zeros(4, 2, 3)
        op = TDLinear(3, 5)
        with torch.no_grad():
            op.weight.zero_()
            op.bias.copy_(torch.arange(5, dtype=x_seq.dtype))

        y_seq = op(x_seq)

        assert torch.allclose(y_seq.cumsum(dim=0)[-1], op.bias.expand(2, 5))
        assert not torch.allclose(
            y_seq.cumsum(dim=0)[-1], op.bias.mul(x_seq.shape[0]).expand(2, 5)
        )
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


class TestSNNMatrixOperator:
    def test_matches_las_style_loop_reference(self):
        a_seq = torch.randn(4, 2, 3, 5, dtype=torch.float64)
        b_seq = torch.randn(4, 2, 5, 7, dtype=torch.float64)
        op = SNNMatrixOperator()

        y_seq = op(a_seq, b_seq)
        expected = _snn_matrix_loop_reference(a_seq, b_seq)

        assert y_seq.shape == (4, 2, 3, 7)
        assert torch.allclose(y_seq, expected, atol=1e-6, rtol=1e-6)

    def test_cumulative_output_matches_matmul_on_cumulative_inputs(self):
        a_seq = torch.randn(5, 2, 3, 4)
        b_seq = torch.randn(5, 2, 4, 6)
        op = SNNMatrixOperator()

        y_seq = op(a_seq, b_seq)
        expected = torch.matmul(a_seq.cumsum(dim=0), b_seq.cumsum(dim=0))

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_single_step_mode_matches_matmul(self):
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 6)
        op = SNNMatrixOperator(step_mode="s")

        y = op(a, b)

        assert torch.allclose(y, torch.matmul(a, b))

    def test_one_step_multi_step_matches_single_step(self):
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 6)
        op = SNNMatrixOperator()

        y_seq = op(a.unsqueeze(0), b.unsqueeze(0))
        functional.set_step_mode(op, "s")

        assert torch.allclose(y_seq[0], op(a, b))

    def test_preserves_cross_time_terms(self):
        a_seq = torch.tensor([[[[1.0]]], [[[2.0]]]])
        b_seq = torch.tensor([[[[3.0]]], [[[5.0]]]])
        op = SNNMatrixOperator()

        y_seq = op(a_seq, b_seq)
        naive = torch.matmul(a_seq, b_seq)

        assert torch.allclose(y_seq.flatten(), torch.tensor([3.0, 21.0]))
        assert not torch.allclose(y_seq, naive)

    def test_supports_broadcast_batch_dimensions(self):
        a_seq = torch.randn(3, 2, 4, 5)
        b_seq = torch.randn(3, 1, 5, 6)
        op = SNNMatrixOperator()

        y_seq = op(a_seq, b_seq)
        expected = torch.matmul(a_seq.cumsum(dim=0), b_seq.cumsum(dim=0))

        assert y_seq.shape == (3, 2, 4, 6)
        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_broadcast_ignores_time_dimension_when_ranks_differ(self):
        a_seq = torch.randn(3, 2, 4, 5)
        b_seq = torch.randn(3, 5, 6)
        op = SNNMatrixOperator()

        y_seq = op(a_seq, b_seq)
        expected = torch.matmul(a_seq.cumsum(dim=0), b_seq.unsqueeze(1).cumsum(dim=0))

        assert y_seq.shape == (3, 2, 4, 6)
        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_gradients_match_reference(self):
        a_seq = torch.randn(3, 2, 4, 5)
        b_seq = torch.randn(3, 2, 5, 6)
        op = SNNMatrixOperator()

        a_ref = a_seq.clone().detach().requires_grad_()
        b_ref = b_seq.clone().detach().requires_grad_()
        y_ref = _temporal_difference(
            torch.matmul(a_ref.cumsum(dim=0), b_ref.cumsum(dim=0))
        )
        y_ref.square().sum().backward()

        a_seq = a_seq.clone().detach().requires_grad_()
        b_seq = b_seq.clone().detach().requires_grad_()
        y_seq = op(a_seq, b_seq)
        y_seq.square().sum().backward()

        assert torch.allclose(a_seq.grad, a_ref.grad, atol=1e-6, rtol=1e-6)
        assert torch.allclose(b_seq.grad, b_ref.grad, atol=1e-6, rtol=1e-6)

    def test_preserves_dtype_and_device(self):
        a_seq = torch.randn(3, 2, 4, 5, dtype=torch.float64)
        b_seq = torch.randn(3, 2, 5, 6, dtype=torch.float64)
        op = SNNMatrixOperator()

        y_seq = op(a_seq, b_seq)

        assert y_seq.dtype == torch.float64
        assert y_seq.device == a_seq.device

    def test_rejects_mismatched_time_lengths(self):
        op = SNNMatrixOperator()

        with pytest.raises(ValueError, match="same time length"):
            op(torch.randn(3, 2, 4, 5), torch.randn(4, 2, 5, 6))

    def test_rejects_empty_time_dimension(self):
        op = SNNMatrixOperator()

        with pytest.raises(ValueError, match="non-empty time dimension"):
            op(torch.empty(0, 2, 4, 5), torch.empty(0, 2, 5, 6))

    def test_rejects_inputs_with_too_few_dimensions(self):
        op = SNNMatrixOperator()

        with pytest.raises(ValueError, match="at least 3 dimensions"):
            op(torch.randn(3, 5), torch.randn(3, 5, 6))

    def test_invalid_matmul_shape_raises_from_pytorch(self):
        op = SNNMatrixOperator()

        with pytest.raises(RuntimeError):
            op(torch.randn(3, 2, 4, 5), torch.randn(3, 2, 4, 6))


class TestSNNElementWiseProduct:
    def test_cumulative_output_matches_product_on_cumulative_inputs(self):
        a_seq = torch.randn(5, 2, 3)
        b_seq = torch.randn(5, 2, 3)
        op = SNNElementWiseProduct()

        y_seq = op(a_seq, b_seq)
        expected = a_seq.cumsum(dim=0) * b_seq.cumsum(dim=0)

        assert y_seq.shape == (5, 2, 3)
        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_final_sum_matches_product_of_final_sums(self):
        a_seq = torch.randn(5, 2, 3)
        b_seq = torch.randn(5, 2, 3)
        op = SNNElementWiseProduct()

        y_seq = op(a_seq, b_seq)
        expected = a_seq.sum(dim=0) * b_seq.sum(dim=0)

        assert torch.allclose(y_seq.sum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_single_step_mode_matches_product(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        op = SNNElementWiseProduct(step_mode="s")

        y = op(a, b)

        assert torch.allclose(y, a * b)

    def test_one_step_multi_step_matches_single_step(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        op = SNNElementWiseProduct()

        y_seq = op(a.unsqueeze(0), b.unsqueeze(0))
        functional.set_step_mode(op, "s")

        assert torch.allclose(y_seq[0], op(a, b))

    def test_supports_broadcast(self):
        a_seq = torch.randn(4, 2, 3)
        b_seq = torch.randn(4, 1, 3)
        op = SNNElementWiseProduct()

        y_seq = op(a_seq, b_seq)
        expected = a_seq.cumsum(dim=0) * b_seq.cumsum(dim=0)

        assert y_seq.shape == (4, 2, 3)
        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_broadcast_ignores_time_dimension_when_ranks_differ(self):
        a_seq = torch.randn(4, 2, 3)
        b_seq = torch.randn(4, 3)
        op = SNNElementWiseProduct()

        y_seq = op(a_seq, b_seq)
        expected = a_seq.cumsum(dim=0) * b_seq.unsqueeze(1).cumsum(dim=0)

        assert y_seq.shape == (4, 2, 3)
        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-6, rtol=1e-6)

    def test_gradients_match_reference(self):
        a_seq = torch.randn(3, 2, 4)
        b_seq = torch.randn(3, 2, 4)
        op = SNNElementWiseProduct()

        a_ref = a_seq.clone().detach().requires_grad_()
        b_ref = b_seq.clone().detach().requires_grad_()
        y_ref = _temporal_difference(a_ref.cumsum(dim=0) * b_ref.cumsum(dim=0))
        y_ref.square().sum().backward()

        a_seq = a_seq.clone().detach().requires_grad_()
        b_seq = b_seq.clone().detach().requires_grad_()
        y_seq = op(a_seq, b_seq)
        y_seq.square().sum().backward()

        assert torch.allclose(a_seq.grad, a_ref.grad, atol=1e-6, rtol=1e-6)
        assert torch.allclose(b_seq.grad, b_ref.grad, atol=1e-6, rtol=1e-6)

    def test_rejects_mismatched_time_lengths(self):
        op = SNNElementWiseProduct()

        with pytest.raises(ValueError, match="same time length"):
            op(torch.randn(3, 2, 4), torch.randn(4, 2, 4))

    def test_rejects_empty_time_dimension(self):
        op = SNNElementWiseProduct()

        with pytest.raises(ValueError, match="non-empty time dimension"):
            op(torch.empty(0, 2, 4), torch.empty(0, 2, 4))

    def test_rejects_one_dimensional_input(self):
        op = SNNElementWiseProduct()

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            op(torch.randn(3), torch.randn(3))

    def test_invalid_broadcast_shape_raises_from_pytorch(self):
        op = SNNElementWiseProduct()

        with pytest.raises(RuntimeError):
            op(torch.randn(3, 2, 4), torch.randn(3, 2, 5))


def test_few_spike_tdlinear_few_spike_sequence_smoke():
    table = neuron.FewSpikeTable(
        theta=torch.tensor([0.25, 0.5, 0.75]),
        h=torch.tensor([0.1, 0.1, 0.1]),
        d=torch.tensor([1.0, 2.0, 4.0]),
    )
    first = neuron.FewSpikeNode(
        table, surrogate_function=surrogate.Sigmoid(), step_mode="m"
    )
    linear = TDLinear(4, 5)
    second = neuron.FewSpikeNode(
        table, surrogate_function=surrogate.Sigmoid(), step_mode="m"
    )
    x_seq = torch.randn(table.K, 2, 4, requires_grad=True)

    y_seq = second(linear(first(x_seq)))
    y_seq.sum().backward()

    assert y_seq.shape == (table.K, 2, 5)
    assert x_seq.grad is not None
    assert torch.isfinite(x_seq.grad).all()


def test_td_mlp_switches_between_single_and_multi_step_modes():
    model = torch.nn.Sequential(
        TDLinear(4, 6),
        TDGELU(),
        TDLinear(6, 3),
    )

    functional.set_step_mode(model, "s")
    x = torch.randn(2, 4)
    y = model(x)
    expected = model[2].single_step_forward(
        model[1].single_step_forward(model[0].single_step_forward(x))
    )

    assert y.shape == (2, 3)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)

    functional.set_step_mode(model, "m")
    x_seq = torch.randn(5, 2, 4)
    y_seq = model(x_seq)
    functional.set_step_mode(model, "s")
    expected_from_total = model(x_seq.sum(dim=0))

    assert y_seq.shape == (5, 2, 3)
    assert torch.allclose(y_seq.sum(dim=0), expected_from_total, atol=1e-6, rtol=1e-6)


def test_few_spike_tdlinear_few_spike_switches_step_modes():
    table = neuron.FewSpikeTable(
        theta=torch.tensor([0.25, 0.5, 0.75]),
        h=torch.tensor([0.1, 0.1, 0.1]),
        d=torch.tensor([1.0, 2.0, 4.0]),
    )
    model = torch.nn.Sequential(
        neuron.FewSpikeNode(
            table, surrogate_function=surrogate.Sigmoid(), step_mode="m"
        ),
        TDLinear(4, 5),
        neuron.FewSpikeNode(
            table, surrogate_function=surrogate.Sigmoid(), step_mode="m"
        ),
    )

    functional.set_step_mode(model, "s")
    x = torch.randn(2, 4)
    y = model(x)

    assert y.shape == (2, 5)

    functional.set_step_mode(model, "m")
    x_seq = torch.randn(table.K, 2, 4)
    y_seq = model(x_seq)
    functional.set_step_mode(model, "s")
    expected_from_total = model(x_seq.sum(dim=0))

    assert y_seq.shape == (table.K, 2, 5)
    assert torch.allclose(y_seq.sum(dim=0), expected_from_total, atol=1e-5, rtol=1e-5)


class TestTDScaledDotProductAttention:
    # CUDA SDPA kernels can differ from the reference cumulative comparison by
    # a few ulps, especially after cumsum and temporal differencing. Keep this
    # tolerance scoped to cumulative SDPA/reference-gradient checks.
    sdpa_cumulative_atol = 1e-5
    sdpa_cumulative_rtol = 1e-5

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

        assert torch.allclose(
            y_seq.cumsum(dim=0),
            expected,
            atol=self.sdpa_cumulative_atol,
            rtol=self.sdpa_cumulative_rtol,
        )

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

        assert torch.allclose(
            y_seq.cumsum(dim=0)[-1],
            expected,
            atol=self.sdpa_cumulative_atol,
            rtol=self.sdpa_cumulative_rtol,
        )

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

    def test_single_step_mode_matches_sdpa(self):
        q = torch.randn(2, 3, 4)
        k = torch.randn(2, 5, 4)
        v = torch.randn(2, 5, 7)
        op = TDScaledDotProductAttention(step_mode="s")

        y = op(q, k, v)
        expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)

    def test_one_step_multi_step_matches_single_step(self):
        q = torch.randn(2, 3, 4)
        k = torch.randn(2, 5, 4)
        v = torch.randn(2, 5, 7)
        op = TDScaledDotProductAttention()

        y_seq = op(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        functional.set_step_mode(op, "s")

        assert torch.allclose(y_seq[0], op(q, k, v), atol=1e-6, rtol=1e-6)

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
        assert torch.allclose(
            y_seq.cumsum(dim=0),
            expected,
            atol=self.sdpa_cumulative_atol,
            rtol=self.sdpa_cumulative_rtol,
        )

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

        assert torch.allclose(
            y_seq.cumsum(dim=0),
            expected,
            atol=self.sdpa_cumulative_atol,
            rtol=self.sdpa_cumulative_rtol,
        )

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

        assert torch.allclose(
            y_seq.cumsum(dim=0),
            expected,
            atol=self.sdpa_cumulative_atol,
            rtol=self.sdpa_cumulative_rtol,
        )

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

        assert torch.allclose(
            y_seq.cumsum(dim=0),
            expected,
            atol=self.sdpa_cumulative_atol,
            rtol=self.sdpa_cumulative_rtol,
        )

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

        assert torch.allclose(
            q_seq.grad,
            q_ref.grad,
            atol=self.sdpa_cumulative_atol,
            rtol=self.sdpa_cumulative_rtol,
        )
        assert torch.allclose(
            k_seq.grad,
            k_ref.grad,
            atol=self.sdpa_cumulative_atol,
            rtol=self.sdpa_cumulative_rtol,
        )
        assert torch.allclose(
            v_seq.grad,
            v_ref.grad,
            atol=self.sdpa_cumulative_atol,
            rtol=self.sdpa_cumulative_rtol,
        )

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


class TestTDMultiheadAttention:
    def test_shape_is_preserved_and_weights_are_none(self):
        x_seq = torch.randn(4, 2, 5, 8)
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        y_seq, weights = op(x_seq, x_seq, x_seq, need_weights=False)

        assert y_seq.shape == x_seq.shape
        assert weights is None

    def test_cumulative_output_matches_ann_mha_on_cumulative_input(self):
        x_seq = torch.randn(5, 2, 4, 8)
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        y_seq, _ = op(x_seq, x_seq, x_seq, need_weights=False)
        expected = _mha_cumulative_reference(op, x_seq, x_seq, x_seq)

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)

    def test_final_cumulative_output_matches_ann_mha_on_total_input(self):
        x_seq = torch.randn(6, 2, 4, 8)
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        y_seq, _ = op(x_seq, x_seq, x_seq, need_weights=False)
        expected = _mha_cumulative_reference(op, x_seq, x_seq, x_seq)[-1]

        assert torch.allclose(y_seq.cumsum(dim=0)[-1], expected, atol=1e-5, rtol=1e-5)

    def test_single_timestep_matches_ann_mha(self):
        x_seq = torch.randn(1, 2, 4, 8)
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        y_seq, _ = op(x_seq, x_seq, x_seq, need_weights=False)
        expected = _mha_cumulative_reference(op, x_seq, x_seq, x_seq)

        assert torch.allclose(y_seq, expected, atol=1e-5, rtol=1e-5)

    def test_single_step_mode_matches_ann_mha(self):
        x = torch.randn(2, 4, 8)
        op = TDMultiheadAttention(embed_dim=8, num_heads=2, step_mode="s")

        y, weights = op(x, x, x, need_weights=False)
        expected = _ann_mha_reference(op, x, x, x)

        assert weights is None
        assert torch.allclose(y, expected, atol=1e-5, rtol=1e-5)

    def test_one_step_multi_step_matches_single_step(self):
        x = torch.randn(2, 4, 8)
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        y_seq, _ = op(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        functional.set_step_mode(op, "s")
        y, _ = op(x, x, x)

        assert torch.allclose(y_seq[0], y, atol=1e-5, rtol=1e-5)

    def test_direct_step_entrypoints_force_projection_step_modes(self):
        x = torch.randn(2, 4, 8)
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        y, _ = op.single_step_forward(x, x, x)
        expected = _ann_mha_reference(op, x, x, x)

        assert torch.allclose(y, expected, atol=1e-5, rtol=1e-5)
        assert op.step_mode == "m"
        assert op.q_proj.step_mode == "m"
        assert op.k_proj.step_mode == "m"
        assert op.v_proj.step_mode == "m"
        assert op.out_proj.step_mode == "m"

        functional.set_step_mode(op, "s")
        x_seq = x.unsqueeze(0)
        y_seq, _ = op.multi_step_forward(x_seq, x_seq, x_seq)
        expected_seq = _mha_cumulative_reference(op, x_seq, x_seq, x_seq)

        assert torch.allclose(y_seq, expected_seq, atol=1e-5, rtol=1e-5)
        assert op.step_mode == "s"
        assert op.q_proj.step_mode == "s"
        assert op.k_proj.step_mode == "s"
        assert op.v_proj.step_mode == "s"
        assert op.out_proj.step_mode == "s"

    def test_parent_step_mode_propagates_to_projection_modules_and_uses_hooks(self):
        x = torch.randn(2, 4, 8)
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)
        hook_calls = []

        op.q_proj.register_forward_hook(
            lambda module, inputs, output: hook_calls.append(module.step_mode)
        )
        op.step_mode = "s"
        assert op.q_proj.step_mode == "s"
        assert op.k_proj.step_mode == "s"
        assert op.v_proj.step_mode == "s"
        assert op.out_proj.step_mode == "s"
        y, _ = op(x, x, x)
        expected = _ann_mha_reference(op, x, x, x)

        assert torch.allclose(y, expected, atol=1e-5, rtol=1e-5)
        assert hook_calls == ["s"]

        functional.set_step_mode(op, "m")
        assert op.q_proj.step_mode == "m"
        assert op.k_proj.step_mode == "m"
        assert op.v_proj.step_mode == "m"
        assert op.out_proj.step_mode == "m"

    def test_cross_attention_matches_ann_mha(self):
        query_seq = torch.randn(4, 2, 3, 8)
        key_seq = torch.randn(4, 2, 5, 8)
        value_seq = torch.randn(4, 2, 5, 8)
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        y_seq, _ = op(query_seq, key_seq, value_seq, need_weights=False)
        expected = _mha_cumulative_reference(op, query_seq, key_seq, value_seq)

        assert y_seq.shape == (4, 2, 3, 8)
        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)

    def test_supports_attention_mask(self):
        x_seq = torch.randn(4, 2, 4, 8)
        attn_mask = torch.tensor(
            [
                [0.0, 0.0, float("-inf"), float("-inf")],
                [0.0, 0.0, 0.0, float("-inf")],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        y_seq, _ = op(x_seq, x_seq, x_seq, attn_mask=attn_mask, need_weights=False)
        expected = _mha_cumulative_reference(
            op, x_seq, x_seq, x_seq, attn_mask=attn_mask
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)

    def test_bool_attention_mask_uses_mha_semantics(self):
        x_seq = torch.randn(4, 2, 4, 8)
        attn_mask = torch.tensor(
            [
                [False, False, True, True],
                [False, False, False, True],
                [False, False, False, False],
                [False, False, False, False],
            ]
        )
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        y_seq, _ = op(x_seq, x_seq, x_seq, attn_mask=attn_mask, need_weights=False)
        expected = _mha_cumulative_reference(
            op, x_seq, x_seq, x_seq, attn_mask=attn_mask
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)

    def test_three_dimensional_attention_mask_matches_mha(self):
        x_seq = torch.randn(4, 2, 4, 8)
        attn_mask = torch.zeros(4, 4, 4)
        attn_mask[:, :, -1] = float("-inf")
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        y_seq, _ = op(x_seq, x_seq, x_seq, attn_mask=attn_mask, need_weights=False)
        expected = _mha_cumulative_reference(
            op,
            x_seq,
            x_seq,
            x_seq,
            attn_mask=attn_mask,
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)

    def test_supports_causal_attention(self):
        x_seq = torch.randn(4, 2, 5, 8)
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        y_seq, _ = op(x_seq, x_seq, x_seq, need_weights=False, is_causal=True)
        q_seq = op._split_heads(op.q_proj(x_seq))
        k_seq = op._split_heads(op.k_proj(x_seq))
        v_seq = op._split_heads(op.v_proj(x_seq))
        expected_attn = F.scaled_dot_product_attention(
            q_seq.cumsum(dim=0),
            k_seq.cumsum(dim=0),
            v_seq.cumsum(dim=0),
            dropout_p=0.0,
            is_causal=True,
        )
        expected = F.linear(
            op._merge_heads(expected_attn),
            op.out_proj.weight,
            op.out_proj.bias,
        )

        assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)

    def test_bias_false_has_no_bias_parameters(self):
        op = TDMultiheadAttention(embed_dim=8, num_heads=2, bias=False)

        assert op.q_proj.bias is None
        assert op.k_proj.bias is None
        assert op.v_proj.bias is None
        assert op.out_proj.bias is None
        assert all("bias" not in key for key in op.state_dict())

    def test_gradients_flow_to_input_and_parameters(self):
        x_seq = torch.randn(3, 2, 4, 8, requires_grad=True)
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        y_seq, _ = op(x_seq, x_seq, x_seq, need_weights=False)
        y_seq.square().sum().backward()

        assert x_seq.grad is not None
        assert torch.isfinite(x_seq.grad).all()
        for parameter in op.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

    def test_rejects_invalid_constructor_arguments(self):
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            TDMultiheadAttention(embed_dim=0, num_heads=2)
        with pytest.raises(ValueError, match="num_heads must be positive"):
            TDMultiheadAttention(embed_dim=8, num_heads=0)
        with pytest.raises(ValueError, match="num_heads must be positive"):
            TDMultiheadAttention(embed_dim=8, num_heads=-2)
        with pytest.raises(ValueError, match="divisible"):
            TDMultiheadAttention(embed_dim=7, num_heads=2)
        with pytest.raises(ValueError, match="dropout=0.0"):
            TDMultiheadAttention(embed_dim=8, num_heads=2, dropout=0.1)
        with pytest.raises(ValueError, match="batch_first=True"):
            TDMultiheadAttention(embed_dim=8, num_heads=2, batch_first=False)

    def test_rejects_invalid_input_shape(self):
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        with pytest.raises(ValueError, match="\\[T, batch, seq, embed_dim\\]"):
            op(torch.randn(4, 5, 8), torch.randn(4, 5, 8), torch.randn(4, 5, 8))

    def test_rejects_mismatched_attention_batch_dimensions(self):
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        with pytest.raises(ValueError, match="leading dimensions"):
            op(
                torch.randn(4, 2, 3, 8),
                torch.randn(4, 1, 5, 8),
                torch.randn(4, 1, 5, 8),
            )

        functional.set_step_mode(op, "s")
        with pytest.raises(ValueError, match="leading dimensions"):
            op(
                torch.randn(2, 3, 8),
                torch.randn(1, 5, 8),
                torch.randn(1, 5, 8),
            )

    def test_rejects_unsupported_forward_options(self):
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)
        x_seq = torch.randn(4, 2, 5, 8)

        with pytest.raises(ValueError, match="need_weights=False"):
            op(x_seq, x_seq, x_seq, need_weights=True)
        with pytest.raises(ValueError, match="key_padding_mask"):
            op(
                x_seq,
                x_seq,
                x_seq,
                need_weights=False,
                key_padding_mask=torch.zeros(2, 5, dtype=torch.bool),
            )
        with pytest.raises(ValueError, match="average_attn_weights=False"):
            op(x_seq, x_seq, x_seq, need_weights=False, average_attn_weights=False)

    def test_extra_repr(self):
        op = TDMultiheadAttention(embed_dim=8, num_heads=2)

        assert op.extra_repr() == (
            "embed_dim=8, num_heads=2, dropout=0.0, batch_first=True"
        )
