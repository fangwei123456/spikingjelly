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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_compressors_are_stateless_and_preserve_dtype(compressor, dtype):
    x = torch.randint(0, 2, (3, 5), dtype=torch.int64).to(dtype)
    first = compressor.compress(x)
    second = compressor.compress(x.reshape(5, 3))

    assert torch.equal(compressor.decompress(first), x)
    assert torch.equal(compressor.decompress(second), x.reshape(5, 3))
    assert not hasattr(compressor, "s_seq_dtype")


def test_structural_compressor_requires_no_inheritance():
    class NegatingCompressor:
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
    module = neuron.LIFNode(step_mode="m", backend="torch")
    reference = copy.deepcopy(module)
    wrapped = memopt.checkpoint_module(module)

    _compare_module(reference, wrapped, torch.randn(6, 2, 3))

    assert torch.equal(module.v, reference.v)


def test_checkpoint_module_commits_batch_norm_buffers_once():
    module = nn.BatchNorm1d(3)
    reference = copy.deepcopy(module)
    wrapped = memopt.checkpoint_module(module)

    _compare_module(reference, wrapped, torch.randn(5, 3))

    assert module.num_batches_tracked.item() == 1
    assert torch.equal(module.running_mean, reference.running_mean)
    assert torch.equal(module.running_var, reference.running_var)


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
