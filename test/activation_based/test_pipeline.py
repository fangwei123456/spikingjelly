import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, layer, neuron
from spikingjelly.activation_based.memopt.checkpointing import (
    GCContainer,
    TCGCContainer,
)
from spikingjelly.activation_based.memopt.compress import (
    BitSpikeCompressor,
    NullSpikeCompressor,
)
from spikingjelly.activation_based.memopt.pipeline import (
    _dummy_input_to_device,
    _dummy_train_step,
    _probe_binary_inputs,
    _spatially_split_gc_container,
    _temporally_split_gc_container,
    _unwrap_gc_container,
    apply_gc,
    get_module_and_parent,
    memory_optimization,
)


def _create_simple_snn_model():
    """Create a simple SNN model for testing."""
    net = nn.Sequential(
        layer.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        neuron.LIFNode(tau=2.0),
        layer.AdaptiveAvgPool2d((1, 1)),
        layer.Flatten(),
        layer.Linear(64, 10),
        neuron.IFNode(),
    )
    functional.set_step_mode(net, "m")
    return net


def test_dummy_input_to_device():
    dummy_input = {
        "float": [
            torch.randn(2, 3, dtype=torch.float64),
            ("label", torch.ones(2, dtype=torch.int64)),
        ],
        "none": None,
    }

    result = _dummy_input_to_device(dummy_input, "meta")

    assert list(result) == ["float", "none"]
    assert isinstance(result["float"], list)
    assert isinstance(result["float"][1], tuple)
    assert result["float"][1][0] == "label" and result["none"] is None
    assert result["float"][0].device.type == "meta"
    assert result["float"][0].dtype == torch.float64
    assert result["float"][1][1].device.type == "meta"
    assert result["float"][1][1].dtype == torch.int64


def test_probe_binary_inputs():
    """Test _probe_binary_inputs function."""
    net = _create_simple_snn_model()
    dummy_input = (torch.randn(4, 8, 1, 28, 28),)

    result = _probe_binary_inputs(net, (neuron.IFNode, neuron.LIFNode), dummy_input)
    assert isinstance(result, dict)
    # All modules should be marked as non-binary
    for value in result.values():
        assert value is False

    # Test with binary inputs
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("torch.all", lambda x, dim=None: torch.tensor(True))
        result = _probe_binary_inputs(net, (neuron.IFNode, neuron.LIFNode), dummy_input)
        assert isinstance(result, dict)
        # All modules should be marked as binary
        for value in result.values():
            assert value is True

    # Test with non-binary inputs
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("torch.all", lambda x, dim=None: torch.tensor(False))
        result = _probe_binary_inputs(net, (neuron.IFNode, neuron.LIFNode), dummy_input)
        assert isinstance(result, dict)
        # All modules should be marked as non-binary
        for value in result.values():
            assert value is False


def test_apply_gc():
    """Test _apply_gc function."""
    net = _create_simple_snn_model()
    dummy_input = (torch.randn(4, 8, 1, 28, 28),)

    # Test basic GC application
    optimized_net = apply_gc(
        net, (neuron.IFNode, neuron.LIFNode), dummy_input, device="cpu"
    )

    # Check that neuron modules are wrapped in GCContainer
    neuron_modules_found = 0
    for module in optimized_net.modules():
        if isinstance(module, GCContainer):
            neuron_modules_found += 1

    assert neuron_modules_found == 3


def test_apply_gc_with_compression():
    """Test _apply_gc function with compression."""
    net = _create_simple_snn_model()
    dummy_input = (torch.randn(4, 8, 1, 28, 28),)

    # Test with compression enabled
    optimized_net = apply_gc(
        net, (neuron.IFNode, neuron.LIFNode), dummy_input, compress_x=True, device="cpu"
    )

    # Check that containers have appropriate compressors
    for module in optimized_net.modules():
        if isinstance(module, GCContainer):
            assert isinstance(
                module.x_compressor, (BitSpikeCompressor, NullSpikeCompressor)
            )


def test_apply_gc_without_compression():
    """Test _apply_gc function without compression."""
    net = _create_simple_snn_model()
    dummy_input = (torch.randn(4, 8, 1, 28, 28),)

    # Test with compression disabled
    optimized_net = apply_gc(
        net,
        (neuron.IFNode, neuron.LIFNode),
        dummy_input,
        compress_x=False,
        device="cpu",
    )

    # Check that containers use NullSpikeCompressor
    for module in optimized_net.modules():
        if isinstance(module, GCContainer):
            assert isinstance(module.x_compressor, NullSpikeCompressor)


def test_dummy_train_step():
    """Test _dummy_train_step function."""
    net = _create_simple_snn_model()
    source = torch.randn(4, 8, 1, 28, 28, requires_grad=True)
    dummy_input = (source * 2,)

    # Test that the function runs without error
    _dummy_train_step(net, dummy_input)

    # Check that network was reset
    for module in net.modules():
        if hasattr(module, "v"):
            assert module.v == 0.0
    assert source.grad is None


def test_get_module_and_parent():
    """Test _get_module_and_parent function."""
    net = _create_simple_snn_model()

    # Test getting a specific module
    module_name = "1"  # IFNode
    module, parent, child_name = get_module_and_parent(net, module_name)

    assert isinstance(module, neuron.IFNode)
    assert parent is net
    assert child_name == "1"


def test_spatially_split_gc_container():
    """Test _spatially_split_gc_container function."""
    container = GCContainer(NullSpikeCompressor(), nn.Linear(10, 5))
    result = _spatially_split_gc_container(container)
    assert result is None

    container = GCContainer(NullSpikeCompressor(), nn.Linear(10, 5), nn.ReLU())
    result = _spatially_split_gc_container(container)
    assert isinstance(result, nn.Sequential)


def test_temporally_split_gc_container():
    """Test _temporally_split_gc_container function."""
    container = GCContainer(NullSpikeCompressor(), nn.Linear(10, 5))
    result = _temporally_split_gc_container(container, factor=2)
    assert isinstance(result, TCGCContainer)
    assert result.n_chunk == 2

    result = _temporally_split_gc_container(result, factor=3)
    assert isinstance(result, TCGCContainer)
    assert result.n_chunk == 6


def test_unwrap_gc_container():
    """Test _unwrap_gc_container function."""
    linear = nn.Linear(10, 5)
    container = GCContainer(NullSpikeCompressor(), linear)
    result = _unwrap_gc_container(container)
    assert result is linear

    seq = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
    container = GCContainer(NullSpikeCompressor(), seq)
    result = _unwrap_gc_container(container)
    assert result[0] is seq[0]
    assert result[1] is seq[1]
    assert isinstance(seq, nn.Sequential)


def test_memory_optimization_level_0():
    """Test memory_optimization with level 0 (no optimization)."""
    net = _create_simple_snn_model()
    dummy_input = (torch.randn(4, 8, 1, 28, 28),)

    # Test level 0 - should return original network
    optimized_net = memory_optimization(
        net, (neuron.IFNode, neuron.LIFNode), dummy_input, level=0
    )
    assert optimized_net is net


def test_memory_optimization_level_1():
    """Test memory_optimization with level 1 (basic GC)."""
    net = _create_simple_snn_model()
    dummy_input = (torch.randn(4, 8, 1, 28, 28),)

    # Test level 1 - basic gradient checkpointing
    optimized_net = memory_optimization(
        net,
        (neuron.IFNode, neuron.LIFNode),
        dummy_input,
        level=1,
    )
    gc_containers = [m for m in optimized_net.modules() if isinstance(m, GCContainer)]
    assert len(gc_containers) > 0


def test_memory_optimization_missing_dummy_input():
    """Test memory_optimization with missing dummy_input for higher levels."""
    net = _create_simple_snn_model()

    with pytest.raises(ValueError, match="dummy_input must be provided"):
        memory_optimization(
            net,
            (neuron.IFNode, neuron.LIFNode),
            None,  # No dummy input
            level=2,
        )


def test_memory_optimization_integration():
    """Integration test for memory_optimization function."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    net = _create_simple_snn_model()
    dummy_input = (torch.randn(2, 4, 1, 28, 28),)
    # Test full optimization pipeline
    optimized_net = memory_optimization(
        net,
        (neuron.IFNode, neuron.LIFNode),
        dummy_input,
        level=4,
        compress_x=True,
    )
    optimized_net = optimized_net.to("cuda")
    dummy_input = tuple(x.to("cuda") for x in dummy_input)
    optimized_net.eval()
    with torch.no_grad():
        output = optimized_net(*dummy_input)
        assert output.shape == (2, 4, 10)
