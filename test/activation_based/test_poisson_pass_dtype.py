import pytest
import torch

from spikingjelly.activation_based.surrogate import PoissonPass


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("spiking", [False, True])
def test_poisson_pass_composes_with_matching_linear_dtype(dtype, spiking):
    """Sampling must not prevent a same-dtype network from forwarding and training."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        x = torch.full((2, 3), 0.5, dtype=dtype, requires_grad=True)
        original = x.detach().clone()
        linear = torch.nn.Linear(3, 2, bias=False).to(dtype)
        with torch.no_grad():
            linear.weight.copy_(x.new_tensor([[1, 2, 3], [-1, 0.5, 0.25]]))
        model = torch.nn.Sequential(PoissonPass(spiking=spiking), linear)

        output = model(x)
        output.sum().backward()

    assert output.shape == (2, 2)
    assert output.dtype == dtype
    assert output.device == x.device
    assert torch.equal(x.detach(), original)
    assert torch.equal(x.grad, x.new_tensor([[0, 2.5, 3.25], [0, 2.5, 3.25]]))
