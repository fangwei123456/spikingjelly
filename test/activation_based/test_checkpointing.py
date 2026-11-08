import copy

import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import memopt, neuron


@pytest.mark.parametrize(
    "compressor",
    [
        memopt.BooleanSpikeCompressor(),
        memopt.Uint8SpikeCompressor(),
        memopt.BitSpikeCompressor(),
        memopt.SparseSpikeCompressor(),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_compressors_are_stateless_and_preserve_dtype(compressor, dtype):
    x = torch.randint(0, 2, (3, 5), dtype=torch.int64).to(dtype)
    first = compressor.compress(x)
    second = compressor.compress(x.reshape(5, 3))

    assert torch.equal(compressor.decompress(first), x)
    assert torch.equal(compressor.decompress(second), x.reshape(5, 3))
    assert not hasattr(compressor, "s_seq_dtype")


def test_custom_compressor_inherits_base_class():
    class NegatingCompressor(memopt.SpikeCompressor):
        def compress(self, x):
            return -x

        def decompress(self, packed):
            return -packed

    x = torch.randn(4, requires_grad=True)
    y = memopt.checkpoint(
        lambda value: value.square(), x, compressor=NegatingCompressor()
    )
    y.sum().backward()

    assert torch.allclose(x.grad, 2 * x)

    class IncompleteCompressor(memopt.SpikeCompressor):
        def compress(self, x):
            return x

    with pytest.raises(TypeError):
        IncompleteCompressor()


def test_checkpoint_supports_kwargs_pytree_and_gradients():
    x = torch.randint(0, 2, (4, 6), dtype=torch.float32, requires_grad=True)
    weight = torch.randn(6, 3, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()

    def function(value, *, matrix):
        output = (value @ matrix).relu()
        return {"output": output, "metadata": "fixed"}

    expected = function(x_ref, matrix=weight_ref)
    actual = memopt.checkpoint(
        function,
        x,
        matrix=weight,
        compressor=memopt.BitSpikeCompressor(),
    )
    expected["output"].sum().backward()
    actual["output"].sum().backward()

    assert actual["metadata"] == "fixed"
    assert torch.equal(actual["output"], expected["output"])
    assert torch.allclose(x.grad, x_ref.grad)
    assert torch.allclose(weight.grad, weight_ref.grad)


def _compare_module(reference: nn.Module, wrapped: nn.Module, x: torch.Tensor):
    x_ref = x.detach().clone().requires_grad_()
    x_actual = x.detach().clone().requires_grad_()
    expected = reference(x_ref)
    actual = wrapped(x_actual)
    expected.sum().backward()
    actual.sum().backward()
    assert torch.allclose(actual, expected)
    assert torch.allclose(x_actual.grad, x_ref.grad)


def test_checkpoint_module_preserves_names_state_dict_and_parameter_identity():
    module = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2))
    state_keys = list(module.state_dict())
    parameter_names = [name for name, _ in module.named_parameters()]
    parameter_ids = [id(parameter) for parameter in module.parameters()]

    wrapped = memopt.checkpoint_module(module)

    assert list(wrapped.state_dict()) == state_keys
    assert [name for name, _ in wrapped.named_parameters()] == parameter_names
    assert [id(parameter) for parameter in wrapped.parameters()] == parameter_ids
    _compare_module(copy.deepcopy(module), wrapped, torch.randn(3, 5))


def test_checkpoint_module_commits_neuron_state_once():
    module = neuron.LIFNode(step_mode="m")
    reference = copy.deepcopy(module)
    wrapped = memopt.checkpoint_module(module)

    _compare_module(reference, wrapped, torch.randn(6, 2, 3))

    assert torch.equal(module.v, reference.v)


def _flexsn_core(x, v):
    h = 0.5 * v + x
    spike = torch.sigmoid(h)
    return spike, h * (1.0 - spike)


def _parameterized_flexsn_core(x, v, threshold):
    h = 0.5 * v + x
    spike = torch.sigmoid(h - threshold)
    return spike, h * (1.0 - spike)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "step_mode,depth,static_kind,shared,chunks",
    [
        ("s", 1, "none", False, 1),
        ("s", 1, "parameter", False, 1),
        ("m", 1, "none", False, 3),
        ("m", 2, "parameter", False, 1),
        ("m", 2, "parameter", True, 3),
        ("m", 2, "buffer", False, 3),
        ("m", 2, "differentiable_buffer", False, 3),
    ],
)
def test_checkpoint_module_flexsn_preserves_states_and_gradients(
    device, step_mode, depth, static_kind, shared, chunks
):
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("triton")
    torch.manual_seed(813)
    nodes = []
    for index in range(depth):
        if static_kind == "none":
            static = ()
            core = _flexsn_core
        else:
            tensor = torch.tensor(0.7 + 0.2 * index, device=device)
            if static_kind == "parameter":
                tensor = nn.Parameter(tensor)
            elif static_kind == "differentiable_buffer":
                tensor.requires_grad_()
            static = nodes[0].static_inputs if shared and nodes else (tensor,)
            core = _parameterized_flexsn_core
        nodes.append(neuron.FlexSN(core, 1, static, step_mode=step_mode))
    module = nodes[0] if depth == 1 else nn.Sequential(*nodes)
    reference = copy.deepcopy(module)
    reference_nodes = [reference] if depth == 1 else list(reference)
    initial_states = []
    for node, ref in zip(nodes, reference_nodes):
        initial = torch.randn(2, 3, device=device, requires_grad=True)
        ref_initial = initial.detach().clone().requires_grad_()
        node.states = (initial,)
        ref.states = (ref_initial,)
        initial_states.append((initial, ref_initial))
    x = torch.randn((2, 3) if step_mode == "s" else (5, 2, 3), device=device)
    _compare_module(reference, memopt.checkpoint_module(module, chunks=chunks), x)
    if device == "cuda" and step_mode == "m":
        assert all(node._triton_handle is not None for node in nodes)
    for node, ref in zip(nodes, reference_nodes):
        torch.testing.assert_close(node.states[0], ref.states[0])
    for initial, ref_initial in initial_states:
        torch.testing.assert_close(initial.grad, ref_initial.grad)
    for parameter, ref_parameter in zip(module.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad)
    for buffer, ref_buffer in zip(module.buffers(), reference.buffers()):
        if buffer.requires_grad:
            torch.testing.assert_close(buffer.grad, ref_buffer.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_checkpoint_module_flexsn_cold_tracing_preserves_dropout_rng():
    pytest.importorskip("triton")
    module = nn.Sequential(neuron.FlexSN(_flexsn_core, 1), nn.Dropout(0.5))
    reference = copy.deepcopy(module)
    x = torch.randn(5, 2, 3, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    torch.manual_seed(813)
    expected = reference(x_ref)
    torch.manual_seed(813)
    actual = memopt.checkpoint_module(module)(x)
    torch.testing.assert_close(actual, expected)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    assert module[0]._triton_handle is not None


def test_checkpoint_module_commits_batch_norm_buffers_once():
    module = nn.BatchNorm1d(3)
    reference = copy.deepcopy(module)
    wrapped = memopt.checkpoint_module(module)

    _compare_module(reference, wrapped, torch.randn(5, 3))

    assert module.num_batches_tracked.item() == 1
    assert torch.equal(module.running_mean, reference.running_mean)
    assert torch.equal(module.running_var, reference.running_var)


def test_checkpoint_module_uses_buffers_registered_after_wrapping():
    class LateBuffer(nn.Module):
        def forward(self, x):
            self.total.add_(x.sum())
            return x * self.total

    module = LateBuffer()
    wrapped = memopt.checkpoint_module(module)
    module.register_buffer("total", torch.zeros(()))
    reference = copy.deepcopy(module)

    _compare_module(reference, wrapped, torch.randn(3, 2))

    assert torch.equal(module.total, reference.total)


def test_temporal_chunks_support_uneven_inputs_and_pytree_outputs():
    class PytreeModule(nn.Module):
        def forward(self, x, scale):
            return {"x": x * scale, "tag": "constant"}

    wrapped = memopt.checkpoint_module(
        PytreeModule(), chunks=3, chunked_args=(0,), time_dim=0
    )
    x = torch.randn(7, 2, 3, requires_grad=True)
    output = wrapped(x, torch.tensor(2.0))
    output["x"].sum().backward()

    assert output["x"].shape == x.shape
    assert output["tag"] == "constant"
    assert torch.allclose(output["x"], x * 2)
    assert torch.allclose(x.grad, torch.full_like(x, 2))


@pytest.mark.parametrize("shape,chunks", [((0, 2), 2), ((2, 2), 3)])
def test_temporal_chunks_reject_empty_or_excessive_chunks(shape, chunks):
    wrapped = memopt.checkpoint_module(nn.Identity(), chunks=chunks)
    with pytest.raises(ValueError, match="temporal length"):
        wrapped(torch.randn(shape, requires_grad=True))


def test_checkpoint_without_temporal_chunks_accepts_empty_input():
    x = torch.empty(0, 2, requires_grad=True)

    assert memopt.checkpoint_module(nn.Identity())(x).shape == x.shape


def test_temporal_chunks_require_matching_input_lengths():
    class Add(nn.Module):
        def forward(self, x, y):
            return x + y

    wrapped = memopt.checkpoint_module(Add(), chunks=2, chunked_args=(0, 1))
    with pytest.raises(ValueError, match="same temporal length"):
        wrapped(torch.randn(4, 2), torch.randn(3, 2))


def test_checkpoint_module_compiles_with_compression():
    module = memopt.checkpoint_module(
        nn.Linear(3, 2), compressor=memopt.BitSpikeCompressor()
    )
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    x = torch.randint(0, 2, (4, 3), dtype=torch.float32, requires_grad=True)

    compiled(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
