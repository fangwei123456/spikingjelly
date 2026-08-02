import pytest

from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.cuda_kernel.neuron_kernel import multi_step


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

    assert "grad_s_to_h" in kernel.core
