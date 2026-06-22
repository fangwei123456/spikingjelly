import pytest
import torch

from spikingjelly.activation_based.ann2snn.operators import SpikeSoftmax


class TestSpikeSoftmax:
    def test_shape_is_preserved(self):
        x_seq = torch.randn(4, 2, 3)
        op = SpikeSoftmax(dim=-1)

        y_seq = op(x_seq)

        assert y_seq.shape == x_seq.shape

    def test_cumulative_output_matches_softmax_on_cumulative_input(self):
        x_seq = torch.randn(5, 2, 4)
        op = SpikeSoftmax(dim=-1)

        y_seq = op(x_seq)
        expected = torch.softmax(x_seq.cumsum(dim=0), dim=-1)

        assert torch.allclose(y_seq.cumsum(dim=0), expected)

    def test_final_cumulative_output_matches_softmax_on_total_input(self):
        x_seq = torch.randn(6, 3, 5)
        op = SpikeSoftmax(dim=-1)

        y_seq = op(x_seq)
        expected = torch.softmax(x_seq.sum(dim=0), dim=-1)

        assert torch.allclose(y_seq.cumsum(dim=0)[-1], expected)

    def test_positive_non_last_softmax_dim(self):
        x_seq = torch.randn(5, 2, 3, 4)
        op = SpikeSoftmax(dim=2)

        y_seq = op(x_seq)
        expected = torch.softmax(x_seq.cumsum(dim=0), dim=2)

        assert torch.allclose(y_seq.cumsum(dim=0), expected)

    def test_single_timestep_returns_softmax_of_input(self):
        x_seq = torch.randn(1, 2, 3)
        op = SpikeSoftmax(dim=-1)

        y_seq = op(x_seq)

        assert y_seq.shape == x_seq.shape
        assert torch.allclose(y_seq[0], torch.softmax(x_seq[0], dim=-1))

    def test_gradients_flow(self):
        x_seq = torch.randn(3, 4, requires_grad=True)
        op = SpikeSoftmax(dim=-1)

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
        op = SpikeSoftmax(dim=-1)

        y_seq = op(x_seq)

        assert (y_seq < 0).any()
        assert torch.allclose(
            y_seq.cumsum(dim=0),
            torch.softmax(x_seq.cumsum(dim=0), dim=-1),
        )

    def test_rejects_dim_zero(self):
        op = SpikeSoftmax(dim=0)

        with pytest.raises(ValueError, match="time dimension"):
            op(torch.randn(4, 2, 3))

    def test_rejects_negative_dim_resolving_to_zero(self):
        op = SpikeSoftmax(dim=-3)

        with pytest.raises(ValueError, match="time dimension"):
            op(torch.randn(4, 2, 3))

    def test_rejects_out_of_range_dim(self):
        op = SpikeSoftmax(dim=3)

        with pytest.raises(ValueError, match="dim must be in the range"):
            op(torch.randn(4, 2, 3))

    def test_rejects_one_dimensional_input(self):
        op = SpikeSoftmax(dim=-1)

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            op(torch.randn(4))

    def test_extra_repr(self):
        op = SpikeSoftmax(dim=-1)

        assert op.extra_repr() == "dim=-1"
