import copy

import torch
import torch.nn as nn

from spikingjelly.activation_based.memopt.checkpointing import (
    in_gc_1st_forward,
    query_autocast,
    input_compressed_gc,
    to_gc_function,
    GCContainer,
    TCGCContainer,
    _gc_1st_forward,
    _separate_args,
    _combine_args,
)
from spikingjelly.activation_based.memopt.compress import (
    BaseSpikeCompressor,
    NullSpikeCompressor,
)
from spikingjelly.activation_based import neuron


def simple_forward_fn(x, weight, bias=None):
    y = torch.matmul(x, weight.t())
    if bias is not None:
        y = y + bias
    return y


class MockCompressor(BaseSpikeCompressor):
    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq * 2

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        return s_seq / 2


def test_thread_local_functions():
    """Test thread-local functions and context managers."""
    # Test initial state
    assert not in_gc_1st_forward()

    # Test context manager
    with _gc_1st_forward():
        assert in_gc_1st_forward()
    assert not in_gc_1st_forward()

    # Test nested context managers
    with _gc_1st_forward():
        assert in_gc_1st_forward()
        with _gc_1st_forward():
            assert in_gc_1st_forward()
        assert in_gc_1st_forward()
    assert not in_gc_1st_forward()


def test_autocast_query():
    """Test autocast querying functionality."""
    device_type, dtype, enabled = query_autocast()
    assert device_type is not None
    assert dtype is not None
    assert not enabled

    with torch.amp.autocast("cpu", dtype=torch.float16):
        device_type, dtype, enabled = query_autocast()
        assert device_type == "cpu"
        assert dtype == torch.float16
        assert enabled


def test_argument_separate_combine():
    """Test argument separation and combination functions."""
    # Test with only tensors
    tensor1 = torch.randn(3, 4)
    tensor2 = torch.randn(2, 5)

    input_args, tensor_args, tensor_indices = _separate_args(tensor1, tensor2)

    assert len(input_args) == 2
    assert all(arg is None for arg in input_args)
    assert tensor_args == [tensor1, tensor2]
    assert tensor_indices == [0, 1]

    # Test with mixed types
    non_tensor = "string"
    input_args, tensor_args, tensor_indices = _separate_args(
        tensor1, non_tensor, tensor2
    )

    assert input_args == [None, "string", None]
    assert tensor_args == [tensor1, tensor2]
    assert tensor_indices == [0, 2]

    combined = _combine_args([None, "string", None], [tensor1, tensor2], [0, 2])
    assert len(combined) == 3
    assert torch.equal(combined[0], tensor1)
    assert combined[1] == "string"
    assert torch.equal(combined[2], tensor2)


def test_input_compressed_gc():
    """Test InputCompressedGC functionality."""
    # Test without gradients
    with torch.no_grad():
        x = torch.randn(5, 3)
        weight = torch.randn(4, 3)
        result = input_compressed_gc(
            simple_forward_fn, NullSpikeCompressor(), x, weight
        )
        expected = torch.matmul(x, weight.t())
        assert torch.allclose(result, expected)
        assert not result.requires_grad

    # Test with gradients
    x = torch.randn(5, 3, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)

    result = input_compressed_gc(simple_forward_fn, NullSpikeCompressor(), x, weight)
    expected = torch.matmul(x, weight.t())

    assert torch.allclose(result, expected)
    assert result.requires_grad

    # Test backward pass
    loss = result.sum()
    loss.backward()

    assert x.grad is not None
    assert weight.grad is not None

    # Test with bias
    bias = torch.randn(4, requires_grad=True)
    result = input_compressed_gc(
        simple_forward_fn, NullSpikeCompressor(), x, weight, bias
    )
    expected = torch.matmul(x, weight.t()) + bias
    assert torch.allclose(result, expected)


def test_to_gc_function():
    """Test to_gc_function decorator and converter."""
    compressor = MockCompressor()

    # Test decorator mode
    @to_gc_function(compressor)
    def decorated_forward(x, weight):
        return torch.matmul(x, weight.t())

    x = torch.randn(5, 3, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)
    result = decorated_forward(x, weight)
    expected = torch.matmul(x, weight.t())
    assert torch.allclose(result, expected)
    assert result.requires_grad

    # Test conversion mode
    def simple_forward(x, weight):
        return torch.matmul(x, weight.t())

    converted_forward = to_gc_function(compressor, simple_forward)
    result = converted_forward(x, weight)
    assert torch.allclose(result, expected)
    assert result.requires_grad


def test_gc_container():
    """Test GCContainer module."""
    compressor = NullSpikeCompressor()
    layer1 = nn.Linear(10, 20)
    layer2 = nn.ReLU()
    container = GCContainer(compressor, layer1, layer2)
    x = torch.randn(3, 10, requires_grad=True)
    result = container(x)
    expected = layer2(layer1(x))
    repr_str = container.extra_repr()
    assert len(container) == 2
    assert container[0] == layer1
    assert container[1] == layer2
    assert isinstance(container.x_compressor, NullSpikeCompressor)
    assert torch.allclose(result, expected)
    assert result.requires_grad
    assert "x_compressor=NullSpikeCompressor" in repr_str

    container_null = GCContainer(None, layer1)
    assert isinstance(container_null.x_compressor, NullSpikeCompressor)

    container_stateful = GCContainer(
        compressor,
        neuron.IFNode(step_mode="m"),
    )
    result = container_stateful(x)
    assert len(container_stateful) == 1
    assert isinstance(container_stateful[0], neuron.IFNode)


def test_tcgc_container():
    """Test TCGCContainer module."""
    compressor = NullSpikeCompressor()
    layer = [nn.Linear(10, 5), nn.ReLU()]
    net = nn.Sequential(*layer)
    container = TCGCContainer(compressor, *layer, n_chunk=4)
    assert len(container) == 2
    assert container.n_chunk == 4

    x_seq = torch.randn(8, 3, 10, requires_grad=True)  # 8 time steps
    result = container(x_seq)
    expected = net(x_seq)
    assert torch.allclose(result, expected)
    assert result.requires_grad

    container_single = TCGCContainer(compressor, *layer, n_chunk=1)
    result_single = container_single(x_seq)
    assert torch.allclose(result_single, expected)

    repr_str = container.extra_repr()
    assert "x_compressor=NullSpikeCompressor" in repr_str
    assert "n_chunk=4" in repr_str


def test_integration():
    """Integration tests for checkpointing functionality."""
    compressor = MockCompressor()

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(50, 100),
                neuron.LIFNode(),
                nn.Linear(100, 50),
                neuron.IFNode(),
                nn.Linear(50, 10),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.layers(x)

    net = SimpleNet()
    gc_net = GCContainer(compressor, *net.layers)
    net = copy.deepcopy(net)

    x = torch.randn(16, 50, requires_grad=True)
    regular_result = net(x)
    gc_result = gc_net(x)
    assert torch.allclose(regular_result, gc_result, atol=1e-6)

    regular_loss = regular_result.sum()
    regular_loss.backward()
    gc_loss = gc_result.sum()
    gc_loss.backward()
    for param, gc_param in zip(net.parameters(), gc_net.parameters()):
        if param.grad is not None and gc_param.grad is not None:
            assert torch.allclose(param.grad, gc_param.grad, atol=1e-5)


if __name__ == "__main__":
    test_thread_local_functions()
    test_autocast_query()
    test_argument_separate_combine()
    test_input_compressed_gc()
    test_to_gc_function()
    test_gc_container()
    test_tcgc_container()
    test_integration()
