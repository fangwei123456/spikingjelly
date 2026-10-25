import pytest
import torch

from spikingjelly.activation_based.encoding import PoissonEncoder


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("step_mode", ["s", "m"])
def test_poisson_encoder_keeps_binary_image_probabilities_exact(dtype, step_mode):
    """Black pixels never spike and white pixels always spike, also in sequences."""
    shape = (32, 1, 28, 28) if step_mode == "s" else (4, 8, 1, 28, 28)
    image = torch.zeros(shape, dtype=dtype)
    image.flatten()[1::2] = 1
    original = image.clone()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        encoded = PoissonEncoder(step_mode=step_mode)(image)

    assert torch.equal(encoded, original)
    assert torch.equal(image, original)
    assert encoded.dtype == image.dtype
    assert encoded.device == image.device
    assert encoded.shape == image.shape
