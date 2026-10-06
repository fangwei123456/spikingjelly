import itertools
import math

import pytest
import torch

from spikingjelly.activation_based import neuron
from spikingjelly._ops.layout import _empty_like, _layout_args


def _layout(x, kind):
    if kind == "contiguous":
        return x.contiguous()
    if kind == "time_inner":
        order = (*range(1, x.ndim), 0)
        return x.permute(order).contiguous().permute(x.ndim - 1, *range(x.ndim - 1))
    if kind == "sliced":
        base = x.new_empty((*x.shape[:-1], x.shape[-1] * 2 + 1))
        result = base[..., 1::2]
        result.copy_(x)
        return result
    if kind == "offset":
        base = x.new_empty(x.numel() + 1)
        result = base[1:].view(x.shape)
        result.copy_(x)
        return result
    if kind == "broadcast":
        return x[:1].expand_as(x)
    raise ValueError(kind)


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_layout_offsets_and_dense_allocation(order):
    x = torch.empty(2, 3, 5).permute(order)
    output = _empty_like(x)
    assert output.stride() == x.stride()
    state = torch.empty_like(x[0]).transpose(0, 1).contiguous().transpose(0, 1)
    sizes, layouts = _layout_args(x, x, state)
    for index in range(math.prod(sizes)):
        coordinates = []
        n = index
        for size in sizes:
            coordinates.append(n % size)
            n //= size
        for time in range(x.shape[0]):
            offset = time * layouts[0][0] + sum(
                c * s for c, s in zip(coordinates, layouts[0][1:])
            )
            assert 0 <= offset < x.numel()
    assert layouts[1][0] == 0


@pytest.mark.parametrize("kind", ["sliced", "offset", "broadcast"])
def test_nondense_output_does_not_alias(kind):
    x = _layout(torch.arange(30.0).view(2, 3, 5), kind)
    result = _empty_like(x)
    result.copy_(torch.arange(30.0).view_as(x))
    assert result.data_ptr() != x.data_ptr()
    assert torch.equal(result.flatten(), torch.arange(30.0))


@pytest.mark.parametrize(
    "input_strides,output_strides",
    [
        ((1, 10, 2), (1, 15, 3)),
        ((1, 20, 4), (15, 5, 1)),
        ((0, 0, 1), (15, 1, 3)),
        ((1, 1, 1), (15, 1, 3)),
    ],
)
def test_resized_neuron_buffer_layout(input_strides, output_strides):
    x = torch.empty_strided((2, 3, 5), input_strides)
    output = _empty_like(x, shape=(3, 3, 5), dtype=torch.float16)
    assert output.shape == (3, 3, 5)
    assert output.stride() == output_strides
    assert output.dtype == torch.float16
    values = torch.arange(output.numel(), dtype=output.dtype).view_as(output)
    output.copy_(values)
    assert torch.equal(output, values)


@pytest.mark.parametrize("sequence", [False, True])
def test_layout_helpers_symbolic_trace(sequence):
    from torch.fx.experimental.proxy_tensor import make_fx

    def run(x, state):
        shape = (x.shape[0] + 1, *x.shape[1:]) if sequence else None
        return (
            _empty_like(x, shape=shape, sequence=sequence),
            _layout_args(x, x, state, sequence=sequence),
        )

    for batch in (2, 4):
        shape = (batch, 3, 5) if sequence else (3, 5)
        strides = (1, 20, 4) if sequence else (20, 4)
        x = torch.empty_strided(shape, strides)
        state = torch.empty_strided((3, 5), (1, 6))
        expected, metadata = run(x, state)
        traced = make_fx(run, tracing_mode="symbolic")(x, state)
        actual, actual_metadata = traced(x, state)
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert actual_metadata == metadata
        assert metadata == ((5, 3), ((1 if sequence else 0, 4, 20), (0, 6, 1)))


@pytest.mark.parametrize("kind", ["IFNode", "LIFNode", "ParametricLIFNode"])
@pytest.mark.parametrize("layout", ["time_inner", "sliced", "broadcast"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_registered_neuron_preserves_noncontiguous_values_and_gradients(
    kind, layout, device
):
    torch.manual_seed(42)
    node_type = getattr(neuron, kind)
    node = node_type(step_mode="m", store_v_seq=False).to(device)
    reference = node_type(step_mode="m", store_v_seq=False).to(device)
    reference.load_state_dict(node.state_dict())
    x = _layout(torch.rand(4, 2, 3, 5, device=device) * 0.8, layout).requires_grad_()
    state = _layout(torch.rand(2, 3, 5, device=device) * 0.1, layout).requires_grad_()
    reference_state = state.detach().contiguous().requires_grad_()
    node.v = state
    reference.v = reference_state

    actual = node(x)
    expected = reference(x.contiguous())
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(node.v, reference.v, rtol=2e-5, atol=2e-6)
    actual_grads = torch.autograd.grad(actual.sum() + node.v.sum(), (x, state))
    expected_grads = torch.autograd.grad(
        expected.sum() + reference.v.sum(), (x, reference_state)
    )
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("kind", ["IFNode", "LIFNode", "ParametricLIFNode"])
def test_single_step_noncontiguous_input(kind):
    node = getattr(neuron, kind)(step_mode="s")
    base = torch.rand(2, 10)
    x = base[:, ::2].requires_grad_()
    y = node(x)
    assert y.shape == x.shape
    assert torch.isfinite(torch.autograd.grad(y.sum(), x)[0]).all()
