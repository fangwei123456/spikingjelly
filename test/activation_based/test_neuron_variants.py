import copy

import pytest
import torch

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.neuron import DSRIFNode, DSRLIFNode, MPBNLIFNode


@pytest.mark.parametrize(
    ("node_type", "kwargs"),
    [
        (DSRIFNode, {}),
        (DSRLIFNode, {"tau": 2.0, "delta_t": 0.05}),
    ],
)
def test_dsr_neuron_backward_does_not_require_distributed(node_type, kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    node = node_type(T=4, **kwargs).to(device)
    x = torch.randn(4, 2, 3, device=device, requires_grad=True)

    node(x).sum().backward()

    assert x.grad is not None
    assert node.v_threshold.grad is not None
    assert torch.isfinite(node.v_threshold.grad)


@pytest.mark.parametrize("shape", [(8, 4), (8, 4, 3, 3)])
@pytest.mark.parametrize("learnable_vth", [False, True])
def test_mpbn_threshold_reparameterization_preserves_eval_result(shape, learnable_vth):
    width = shape[1]
    kwargs = {"out_features": width} if len(shape) == 2 else {"out_channels": width}
    node = MPBNLIFNode(
        tau=2.0,
        decay_input=False,
        learnable_vth=learnable_vth,
        **kwargs,
    ).train()
    for _ in range(3):
        functional.reset_net(node)
        node(torch.randn(*shape))

    functional.reset_net(node)
    node.eval()
    folded = copy.deepcopy(node)
    folded.re_parameterize_v_threshold(normalize_residual=True)
    x = torch.randn(*shape)
    functional.reset_net(node)
    functional.reset_net(folded)

    expected = node(x)
    actual = folded(x)

    assert torch.equal(actual, expected)
    assert torch.allclose(folded.v, node.v, atol=1e-5, rtol=1e-5)
