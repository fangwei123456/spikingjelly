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


def test_get_reset_value_reports_missing_reset_value():
    class StatefulModule(base.MemoryModule):
        def __init__(self):
            super().__init__()
            self.register_memory("state", 0)

        def single_step_forward(self, x):
            return x

    module = StatefulModule()
    del module._memories_rv["state"]

    with pytest.raises(KeyError, match="state has no reset value"):
        module.get_reset_value("state")


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
    outputs, states = func_forward((x,), ())
    result2 = module(x)
    assert states == ()
    assert torch.equal(outputs[0], result2)


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

    x = torch.randn(3, 10)
    initial_state = torch.tensor(0.0)
    outputs, new_states = func_forward((x,), (initial_state,))
    expected_output = module.linear(x)
    expected_state = initial_state + 1.0

    assert torch.equal(outputs[0], expected_output)
    assert torch.equal(new_states[0], expected_state)


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
    outputs, new_states = func_forward((x,), (outer_state, inner_state))

    assert torch.equal(outputs[0], x)
    assert torch.equal(new_states[0], outer_state + 2.0)
    assert new_states[1] == inner_state + 1


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
    func_forward((x,), (torch.tensor(5.0),))

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
    outputs, new_states = func_forward((x,), (initial_state,))

    assert torch.equal(outputs[0], module.linear(x) * 2)
    assert torch.equal(new_states[0], initial_state + 2.0)


def test_memory_module_functional_forward_is_pure_and_backs_regular_forward():
    x = torch.tensor(1.0)
    node = neuron.IFNode()
    initial_v = torch.zeros_like(x)
    outputs, updated_states = node.functional_forward((x,), (initial_v,))
    assert node.v == 0.0
    torch.testing.assert_close(node(x), outputs[0])
    torch.testing.assert_close(node.v, updated_states[0])


def test_memory_module_step_forwards_are_functional_backed():
    class FunctionalOnlyModule(base.MemoryModule):
        def __init__(self):
            super().__init__()
            self.register_memory("left", torch.tensor(2.0))
            self.register_memory("right", torch.tensor(5.0))

        def single_step_functional_forward(self, inputs, states, **kwargs):
            x, y = inputs
            increment = kwargs.get("increment", 1.0)
            left = states[0] + increment
            right = states[1] + increment
            return (x + y + left, x - y + right), (left, right)

    module = FunctionalOnlyModule()
    outputs = module.single_step_forward(
        torch.tensor(1.0), torch.tensor(3.0), increment=2.0
    )
    torch.testing.assert_close(outputs[0], torch.tensor(8.0))
    torch.testing.assert_close(outputs[1], torch.tensor(5.0))
    torch.testing.assert_close(module.left, torch.tensor(4.0))
    torch.testing.assert_close(module.right, torch.tensor(7.0))

    outputs = module.multi_step_forward(
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
        increment=3.0,
    )
    torch.testing.assert_close(outputs[0], torch.tensor([11.0, 16.0]))
    torch.testing.assert_close(outputs[1], torch.tensor([8.0, 11.0]))
    torch.testing.assert_close(module.left, torch.tensor(10.0))
    torch.testing.assert_close(module.right, torch.tensor(13.0))


def test_materialize_states_uses_single_step_shape_for_multistep_forward():
    class MaterializedStateModule(base.MemoryModule):
        def __init__(self):
            super().__init__()
            self.register_memory("state", None)
            self.step_mode = "m"

        def materialize_states(self, inputs, states, step_mode):
            x = inputs[0][0] if step_mode == "m" else inputs[0]
            return (torch.zeros_like(x),)

        def single_step_functional_forward(self, inputs, states, **kwargs):
            state = states[0] + inputs[0]
            return (state,), (state,)

    module = MaterializedStateModule()
    x_seq = torch.randn(4, 2, 3)

    module(x_seq)

    assert module.state.shape == x_seq.shape[1:]


def test_simple_base_node_forward_uses_charge_fire_reset_hooks():
    class SquareIFNode(neuron.SimpleBaseNode):
        def neuronal_charge(self, x):
            self.v = self.v + x.square()

    node = SquareIFNode(v_threshold=10.0)
    x_seq = torch.tensor([[1.0], [2.0], [1.0]])

    node.step_mode = "s"
    spike_steps = torch.stack([node(x) for x in x_seq])
    torch.testing.assert_close(spike_steps, torch.zeros_like(x_seq))
    torch.testing.assert_close(node.v, torch.tensor([6.0]))

    node.reset()
    node.step_mode = "m"
    spike_seq = node(x_seq)
    torch.testing.assert_close(spike_seq, torch.zeros_like(x_seq))
    torch.testing.assert_close(node.v, torch.tensor([6.0]))


@pytest.mark.parametrize(
    ("simple_factory", "production_factory"),
    [
        (neuron.SimpleIFNode, neuron.IFNode),
        (
            lambda **kwargs: neuron.SimpleLIFNode(tau=2.5, decay_input=False, **kwargs),
            lambda **kwargs: neuron.LIFNode(tau=2.5, decay_input=False, **kwargs),
        ),
    ],
)
@pytest.mark.parametrize("step_mode", ["s", "m"])
def test_simple_nodes_match_production_nodes(
    simple_factory, production_factory, step_mode
):
    simple = simple_factory(v_threshold=0.8, v_reset=0.0, step_mode=step_mode)
    production = production_factory(v_threshold=0.8, v_reset=0.0, step_mode=step_mode)
    x = torch.randn(4, 2, 3) if step_mode == "m" else torch.randn(2, 3)
    v = torch.randn_like(x[0] if step_mode == "m" else x)
    simple.v = v.clone()
    production.v = v.clone()

    torch.testing.assert_close(simple(x), production(x))
    torch.testing.assert_close(simple.v, production.v)


def test_to_functional_forward_converts_simple_node_without_mutating_it():
    class SquareIFNode(neuron.SimpleBaseNode):
        def neuronal_charge(self, x):
            self.v = self.v + x.square()

    node = SquareIFNode(v_threshold=2.0, v_reset=None, step_mode="m")
    node.v = torch.tensor([7.0])
    x_seq = torch.tensor([[0.5], [1.0], [0.5]])
    initial_v = torch.tensor([0.25])

    with pytest.raises(NotImplementedError):
        node.functional_forward((x_seq,), (initial_v,))

    functional_forward = base.to_functional_forward(node)
    outputs, updated_states = functional_forward((x_seq,), (initial_v,))

    torch.testing.assert_close(node.v, torch.tensor([7.0]))
    reference = SquareIFNode(v_threshold=2.0, v_reset=None, step_mode="m")
    reference.v = initial_v
    torch.testing.assert_close(outputs[0], reference(x_seq))
    torch.testing.assert_close(updated_states[0], reference.v)


def test_to_functional_forward_preserves_overridden_simple_forward():
    class NonSpikingLIFNode(neuron.SimpleLIFNode):
        def single_step_forward(self, x):
            self.neuronal_charge(x)
            return self.v

    node = NonSpikingLIFNode(tau=2.0, decay_input=True, step_mode="m")
    node.v = torch.tensor([4.0])
    x_seq = torch.tensor([[1.0], [2.0], [3.0]])
    initial_v = torch.tensor([0.0])

    outputs, states = base.to_functional_forward(node)((x_seq,), (initial_v,))

    reference = NonSpikingLIFNode(tau=2.0, decay_input=True, step_mode="m")
    reference.v = initial_v
    torch.testing.assert_close(outputs[0], reference(x_seq))
    torch.testing.assert_close(states[0], reference.v)
    torch.testing.assert_close(node.v, torch.tensor([4.0]))


def test_to_functional_forward_uses_default_multi_step_functional_forward():
    class FunctionalOnlyModule(base.MemoryModule):
        def __init__(self):
            super().__init__()
            self.register_memory("state", torch.tensor(0.0))
            self.step_mode = "m"

        def single_step_functional_forward(self, inputs, states, **kwargs):
            state = states[0] + inputs[0]
            return (state,), (state,)

        def forward(self, *args, **kwargs):
            raise AssertionError("to_functional_forward used the stateful path")

    module = FunctionalOnlyModule()
    functional_forward = base.to_functional_forward(module)
    outputs, states = functional_forward(
        (torch.tensor([1.0, 2.0, 3.0]),), (torch.tensor(4.0),)
    )

    torch.testing.assert_close(outputs[0], torch.tensor([5.0, 7.0, 10.0]))
    torch.testing.assert_close(states[0], torch.tensor(10.0))


def test_to_functional_forward_sequential_multiple_inputs_outputs_and_kwargs():
    class Add(nn.Module):
        def forward(self, x, y, *, scale=1.0):
            return (x + y) * scale, x - y

    class Combine(nn.Module):
        def forward(self, total, difference):
            return total + difference

    module = nn.Sequential(Add(), Combine())
    functional_forward = base.to_functional_forward(module)
    x = torch.tensor(3.0)
    y = torch.tensor(2.0)
    outputs, states = functional_forward((x, y), (), scale=2.0)

    assert states == ()
    torch.testing.assert_close(outputs[0], torch.tensor(11.0))


def test_to_functional_forward_fallback_multiple_inputs_states_outputs_and_kwargs():
    class Accumulator(base.MemoryModule):
        def __init__(self):
            super().__init__()
            self.register_memory("left", torch.tensor(10.0))
            self.register_memory("right", torch.tensor(20.0))

        def single_step_forward(self, x, y, *, scale=1.0):
            self.left = self.left + x * scale
            self.right = self.right + y * scale
            return self.left, self.right

    module = Accumulator()
    functional_forward = base.to_functional_forward(module)
    outputs, states = functional_forward(
        (torch.tensor(2.0), torch.tensor(3.0)),
        (torch.tensor(1.0), torch.tensor(4.0)),
        scale=2.0,
    )

    torch.testing.assert_close(outputs[0], torch.tensor(5.0))
    torch.testing.assert_close(outputs[1], torch.tensor(10.0))
    torch.testing.assert_close(states[0], torch.tensor(5.0))
    torch.testing.assert_close(states[1], torch.tensor(10.0))
    torch.testing.assert_close(module.left, torch.tensor(10.0))
    torch.testing.assert_close(module.right, torch.tensor(20.0))


def test_to_functional_forward_fallback_restores_state_after_error():
    class FailingModule(base.MemoryModule):
        def __init__(self):
            super().__init__()
            self.register_memory("state", torch.tensor(7.0))

        def single_step_forward(self, x):
            self.state = self.state + x
            raise RuntimeError("expected failure")

    module = FailingModule()
    functional_forward = base.to_functional_forward(module)

    with pytest.raises(RuntimeError, match="expected failure"):
        functional_forward((torch.tensor(1.0),), (torch.tensor(3.0),))

    torch.testing.assert_close(module.state, torch.tensor(7.0))


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
