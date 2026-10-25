import pytest
import torch
import torch.nn.functional as F

from spikingjelly.activation_based.examples.common.tv_ref_classify.transforms import (
    RandomMixup,
)
from spikingjelly.activation_based.functional import (
    temporal_efficient_training_cross_entropy,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("spatial", [(), (3,), (2, 3)])
@pytest.mark.parametrize("soft", [False, True])
def test_temporal_cross_entropy_matches_per_step_loss_and_gradients(
    dtype, spatial, soft
):
    """Time-invariant targets must match the documented per-step CE definition."""
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(
            torch.Generator(device="cpu").manual_seed(41).get_state()
        )
        logits = torch.randn(3, 2, 4, *spatial, dtype=dtype, requires_grad=True)
        if soft:
            target = torch.randn(4, 2, *spatial, dtype=dtype).softmax(0).movedim(0, 1)
            target = target.detach().requires_grad_(True)
        else:
            target = torch.randint(4, (2, *spatial))
    original_logits = logits.detach().clone()
    original_target = target.detach().clone()
    reference_logits = logits.detach().clone().requires_grad_(True)
    reference_target = target.detach().clone().requires_grad_(soft)

    loss = temporal_efficient_training_cross_entropy(logits, target)
    reference = torch.stack(
        [F.cross_entropy(step, reference_target) for step in reference_logits]
    ).mean()
    torch.testing.assert_close(loss, reference)
    variables = (logits, target) if soft else (logits,)
    references = (reference_logits, reference_target) if soft else (reference_logits,)
    gradients = torch.autograd.grad(loss, variables)
    expected_gradients = torch.autograd.grad(reference, references)
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected)
    assert torch.equal(logits.detach(), original_logits)
    assert torch.equal(target.detach(), original_target)


def test_temporal_cross_entropy_accepts_project_mixup_targets():
    """The project's image MixUp distribution can supervise temporal logits."""
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(
            torch.Generator(device="cpu").manual_seed(17).get_state()
        )
        _, target = RandomMixup(num_classes=3, p=1.0)(
            torch.rand(2, 3, 8, 8), torch.tensor([0, 1])
        )
        logits = torch.randn(3, 2, 3, requires_grad=True)
    expected = torch.stack([F.cross_entropy(step, target) for step in logits]).mean()
    actual = temporal_efficient_training_cross_entropy(logits, target)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad.isfinite().all()


def test_temporal_cross_entropy_soft_targets_under_cpu_autocast():
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(
            torch.Generator(device="cpu").manual_seed(43).get_state()
        )
        logits = torch.randn(3, 2, 4, dtype=torch.bfloat16, requires_grad=True)
        target = torch.rand(2, 4).softmax(1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = temporal_efficient_training_cross_entropy(logits, target)
        expected = torch.stack(
            [F.cross_entropy(step, target) for step in logits]
        ).mean()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad.isfinite().all()


def test_temporal_cross_entropy_rejects_integer_probability_targets():
    with pytest.raises(RuntimeError):
        temporal_efficient_training_cross_entropy(
            torch.zeros(3, 2, 4), torch.zeros(2, 4, dtype=torch.long)
        )
