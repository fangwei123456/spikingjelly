import torch
import torch.nn as nn
import pytest

from spikingjelly.activation_based import layer, functional, neuron, base


def _create_test_model():
    net = nn.Sequential(
        layer.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        neuron.IFNode(),
        layer.AdaptiveAvgPool2d((1, 1)),
        layer.Flatten(),
        layer.Linear(64, 10),
        neuron.IFNode(),
    )
    functional.set_step_mode(net, "m")
    return net


def test_named_memories():
    """Test named_memories function."""
    # Test with simple model
    net = _create_test_model()

    # Get named memories
    named_memory_list = list(base.named_memories(net))
    assert len(named_memory_list) == 3
    for name, value in named_memory_list:
        assert isinstance(name, str)
        assert "v" in name
        print(name, value)


def test_named_memories_with_prefix():
    """Test named_memories function with prefix."""
    net = _create_test_model()

    # Test with prefix
    named_memory_list = list(base.named_memories(net, prefix="test_model"))

    # Check that all names start with the prefix
    for name, value in named_memory_list:
        assert name.startswith("test_model.")
        print(name, value)


def test_named_memories_nested():
    """Test named_memories with nested modules."""

    # Create a nested model
    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = neuron.IFNode()
            self.submodule = nn.Sequential(neuron.IFNode(), neuron.IFNode())
            self.layer2 = neuron.IFNode()

        def forward(self, x):
            x = self.layer1(x)
            x = self.submodule(x)
            x = self.layer2(x)
            return x

    model = NestedModel()
    functional.set_step_mode(model, "m")

    named_memory_list = list(base.named_memories(model))
    assert len(named_memory_list) == 4

    # Check nested naming
    names = [name for name, _ in named_memory_list]
    assert any("layer1" in name for name in names)
    assert any("submodule.0" in name for name in names)
    assert any("submodule.1" in name for name in names)
    assert any("layer2" in name for name in names)


def test_memories():
    """Test memories function."""
    net = _create_test_model()
    memory_gen = base.memories(net)
    memory_list = list(memory_gen)

    assert len(memory_list) == 3
    for memory in memory_list:
        assert memory == 0.0


def test_extract_memories():
    """Test extract_memories function."""
    net = _create_test_model()
    memory_list = base.extract_memories(net)

    assert isinstance(memory_list, list)
    assert len(memory_list) == 3
    for memory in memory_list:
        memory == 0.0

    memory_gen_list = list(base.memories(net))
    assert len(memory_list) == len(memory_gen_list)
    for m1, m2 in zip(memory_list, memory_gen_list):
        assert m1 == m2


def test_load_memories():
    """Test load_memories function."""
    net = _create_test_model()
    original_memories = base.extract_memories(net)
    modified_memories = []
    for _ in original_memories:
        modified_memories.append(torch.tensor([42.0], dtype=torch.float32))
    base.load_memories(net, modified_memories)

    current_memories = base.extract_memories(net)
    for current, modified in zip(current_memories, modified_memories):
        assert torch.equal(current, modified)
        assert torch.all(current == 42.0)


def test_load_memories_length_mismatch():
    net = _create_test_model()
    original_memories = base.extract_memories(net)

    # Try to load with wrong number of memories
    with pytest.raises(ValueError, match="Memory length mismatch"):
        base.load_memories(net, original_memories[:-1])  # One less memory

    with pytest.raises(ValueError, match="Memory length mismatch"):
        base.load_memories(
            net, original_memories + [torch.randn(10)]
        )  # One extra memory


def test_to_functional_forward_stateless():
    """Test to_functional_forward with stateless module."""

    class StatelessModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.linear(x))

    module = StatelessModule()
    func_forward = base.to_functional_forward(module)
    x = torch.randn(3, 10)
    result1 = func_forward(x)
    result2 = module(x)
    assert torch.equal(result1, result2)


def test_to_functional_forward_stateful():
    """Test to_functional_forward with stateful module."""

    class StatefulModule(base.MemoryModule):
        def __init__(self):
            super().__init__()
            self.register_memory("counter", torch.tensor(0.0))
            self.linear = nn.Linear(10, 5)

        def single_step_forward(self, x):
            self.counter = self.counter + 1.0
            return self.linear(x)

    module = StatefulModule()
    func_forward = base.to_functional_forward(module)

    # Test functional interface: (input, state) -> (output, new_state)
    x = torch.randn(3, 10)
    initial_state = torch.tensor(0.0)
    output, new_state = func_forward(x, initial_state)
    expected_output = module.linear(x)
    expected_state = initial_state + 1.0

    assert torch.equal(output, expected_output)
    assert torch.equal(new_state, expected_state)


def test_to_functional_forward_nested_stateful():
    """Test to_functional_forward with nested stateful modules."""

    # Create nested stateful model
    class InnerModule(base.MemoryModule):
        def __init__(self):
            super().__init__()
            self.register_memory("inner_counter", 0)

        def single_step_forward(self, x):
            self.inner_counter += 1
            return x

    class OuterModule(base.MemoryModule):
        def __init__(self):
            super().__init__()
            self.register_memory("outer_counter", torch.tensor(0.0))
            self.inner = InnerModule()

        def single_step_forward(self, x):
            self.outer_counter = self.outer_counter + 2.0
            return self.inner(x)

    module = OuterModule()
    func_forward = base.to_functional_forward(module)
    x = torch.randn(3, 10)
    outer_state = torch.tensor(5.0)
    inner_state = 3
    output, new_outer_state, new_inner_state = func_forward(x, outer_state, inner_state)

    assert torch.equal(output, x)
    assert torch.equal(new_outer_state, outer_state + 2.0)
    assert new_inner_state == inner_state + 1


def test_to_functional_forward_state_restoration():
    """Test that to_functional_forward restores original state."""

    class StatefulModule(base.MemoryModule):
        def __init__(self):
            super().__init__()
            self.register_memory("state", torch.tensor(10.0))

        def single_step_forward(self, x):
            self.state = self.state + x.sum()
            return x

    module = StatefulModule()
    original_state = module.state.clone()
    func_forward = base.to_functional_forward(module)
    x = torch.randn(3, 10)
    func_forward(x, torch.tensor(5.0))

    assert torch.equal(module.state, original_state)


def test_to_functional_forward_custom_fn():
    """Test to_functional_forward with custom function."""

    class TestModule(base.MemoryModule):
        def __init__(self):
            super().__init__()
            self.register_memory("state", torch.tensor(0.0))
            self.linear = nn.Linear(10, 5)

        def single_step_forward(self, x):
            self.state = self.state + 1.0
            return self.linear(x)

        def custom_forward(self, x):
            self.state = self.state + 2.0
            return self.linear(x) * 2

    module = TestModule()
    func_forward = base.to_functional_forward(module, module.custom_forward)
    x = torch.randn(3, 10)
    initial_state = torch.tensor(0.0)
    output, new_state = func_forward(x, initial_state)

    assert torch.equal(output, module.linear(x) * 2)
    assert torch.equal(new_state, initial_state + 2.0)


if __name__ == "__main__":
    # Run individual tests
    test_named_memories()
    test_named_memories_with_prefix()
    test_named_memories_nested()
    test_memories()
    test_extract_memories()
    test_load_memories()
    test_load_memories_length_mismatch()
    test_to_functional_forward_stateless()
    test_to_functional_forward_stateful()
    test_to_functional_forward_nested_stateful()
    test_to_functional_forward_state_restoration()
    test_to_functional_forward_custom_fn()
    print("All tests passed!")
