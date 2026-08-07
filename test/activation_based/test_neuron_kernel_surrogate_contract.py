import pytest
import torch

from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.cuda_kernel.neuron_kernel import multi_step

_PRIMITIVE_SURROGATES = [
    surrogate.PiecewiseQuadratic,
    surrogate.PiecewiseExp,
    surrogate.Sigmoid,
    surrogate.SoftSign,
    surrogate.ATan,
    surrogate.NonzeroSignLogAbs,
    surrogate.Erf,
    surrogate.PiecewiseLeakyReLU,
    surrogate.SquarewaveFourierSeries,
    surrogate.S2NN,
    surrogate.QPseudoSpike,
    surrogate.LeakyKReLU,
    surrogate.LogTailedReLU,
    surrogate.Rect,
]


@pytest.mark.parametrize(
    "surrogate_type",
    _PRIMITIVE_SURROGATES,
    ids=lambda surrogate_type: surrogate_type.__name__,
)
def test_spiking_surrogate_gradient_matches_primitive_finite_difference(
    surrogate_type,
):
    x = torch.tensor([-0.75, -0.25, 0.25, 0.75], dtype=torch.float64)
    spiking_x = x.clone().requires_grad_()
    function = surrogate_type()

    spiking_output = function(spiking_x)
    spiking_output.sum().backward()
    function.set_spiking_mode(False)
    epsilon = 1e-5
    numerical_gradient = (function(x + epsilon) - function(x - epsilon)) / (2 * epsilon)

    assert torch.equal(spiking_output, surrogate.heaviside(x))
    torch.testing.assert_close(spiking_x.grad, numerical_gradient, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (surrogate.DeterministicPass(), torch.tensor([1.0, 1.0])),
        (surrogate.PoissonPass(), torch.tensor([0.0, 1.0])),
    ],
    ids=["deterministic", "poisson"],
)
def test_straight_through_surrogates_preserve_boundary_values_and_gradients(
    function, expected
):
    x = torch.tensor([0.0, 1.0], requires_grad=True)

    output = function(x)
    output.sum().backward()

    assert torch.equal(output, expected)
    assert torch.equal(x.grad, torch.ones_like(x))


def test_super_spike_has_step_forward_and_finite_surrogate_gradient():
    x = torch.tensor([-0.75, -0.25, 0.25, 0.75], requires_grad=True)

    output = surrogate.SuperSpike()(x)
    output.sum().backward()

    assert torch.equal(output, surrogate.heaviside(x))
    assert torch.isfinite(x.grad).all()


def test_fake_numerical_gradient_is_finite_away_from_zero():
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)

    surrogate.FakeNumericalGradient(alpha=0.3)(x).sum().backward()

    assert torch.isfinite(x.grad).all()
    assert (x.grad >= 0.3).all()


@pytest.mark.parametrize(
    "kernel_type, extra_kwargs",
    [
        (multi_step.NeuronBPTTKernel, {}),
        (multi_step.LIFNodeBPTTKernel, {"decay_input": True}),
        (multi_step.ParametricLIFNodeBPTTKernel, {"decay_input": True}),
    ],
)
@pytest.mark.parametrize("surrogate_type", [surrogate.Sigmoid, surrogate.ATan])
@pytest.mark.parametrize("dtype", ["float", "half2"])
def test_bptt_kernel_accepts_surrogate_cuda_codes(
    kernel_type, extra_kwargs, surrogate_type, dtype
):
    kernel = kernel_type(
        surrogate_function=surrogate_type().cuda_codes,
        hard_reset=True,
        detach_reset=False,
        dtype=dtype,
        **extra_kwargs,
    )

    surrogate_code = surrogate_type().cuda_codes(
        y=f"const {dtype} grad_s_to_h", x="over_th", dtype=dtype
    )
    assert "".join(surrogate_code.split()) in "".join(kernel.core.split())
