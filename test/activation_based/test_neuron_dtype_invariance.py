import pytest
import torch

from spikingjelly.activation_based import neuron

# Concrete BaseNode subclasses that construct with no required arguments.
# Single-step neurons that take a plain ``(batch, features)`` input.
# (DSR* are multi-step only; CUBALIFNode has a distinct state contract --
# both are exercised in test_functional_neuron.py / test_neuron_variants.py.)
# The intent is coverage: a neuron's recurrent state and its
# (floating) forward output must follow the module's dtype after ``.to(...)`` --
# the same invariant class as #743 (MemoryModule._apply) and #744 (MSTDPLearner).
NO_ARG_NODES = [
    neuron.IFNode,
    neuron.LIFNode,
    neuron.ParametricLIFNode,
    neuron.QIFNode,
    neuron.EIFNode,
    neuron.IzhikevichNode,
    neuron.KLIFNode,
    neuron.ComplementaryLIFNode,
    neuron.ILIFNode,
    neuron.OTTTLIFNode,
    neuron.SLTTLIFNode,
    neuron.HalfThresholdIFNode,
    neuron.ActivationAwareIFNode,
    neuron.SimpleIFNode,
]

_IDS = [cls.__name__ for cls in NO_ARG_NODES]


@pytest.mark.parametrize("cls", NO_ARG_NODES, ids=_IDS)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float16], ids=["f64", "f16"])
def test_neuron_state_and_output_follow_module_dtype(cls, dtype):
    node = cls(step_mode="s").to(dtype)
    x = torch.rand(4, 3, dtype=dtype)

    out = node(x)
    if isinstance(out, torch.Tensor) and out.is_floating_point():
        assert out.dtype == dtype, f"{cls.__name__}: {dtype} in -> {out.dtype} out"

    node.reset()
    for name, mem in node.named_memories():
        if isinstance(mem, torch.Tensor) and mem.is_floating_point():
            assert mem.dtype == dtype, (
                f"{cls.__name__}: memory {name!r} is {mem.dtype} after .to({dtype})+reset()"
            )


@pytest.mark.parametrize("cls", NO_ARG_NODES, ids=_IDS)
def test_neuron_reset_forgets_the_batch_size(cls):
    node = cls(step_mode="s")
    for _ in range(4):
        node(torch.rand(3, 5))
    node.reset()
    out = node(torch.rand(2, 5))  # a different batch size after reset
    if isinstance(out, torch.Tensor) and out.dim() >= 1:
        assert out.shape[0] == 2, f"{cls.__name__}: stale batch dim after reset()"
