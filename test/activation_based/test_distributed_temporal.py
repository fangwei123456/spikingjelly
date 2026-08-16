import pytest
import torch
from torch.utils.checkpoint import checkpoint

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.distributed.llm.temporal import (
    pack_time_batch,
    run_functional_sequence,
    unpack_time_batch,
)


def test_time_batch_round_trip_keeps_time_slices_independent():
    hidden = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)

    packed = pack_time_batch(hidden)

    assert packed.shape == (5, 6, 7)
    assert packed.is_contiguous()
    torch.testing.assert_close(unpack_time_batch(packed, 2), hidden)
    torch.testing.assert_close(packed[:, :3], hidden[0].permute(1, 0, 2))
    torch.testing.assert_close(packed[:, 3:], hidden[1].permute(1, 0, 2))


def test_unpack_time_batch_rejects_incompatible_time_steps():
    with pytest.raises(ValueError, match="must be positive and divide"):
        unpack_time_batch(torch.empty(5, 6, 7), 4)


def test_run_functional_sequence_restarts_without_mutating_module_state():
    module = neuron.IFNode(step_mode="m")
    module.v = torch.tensor(9.0)
    eager_input = torch.full((3, 2), 0.6, requires_grad=True)
    checkpoint_input = eager_input.detach().clone().requires_grad_()

    eager = run_functional_sequence(module, (eager_input,))[0]
    recomputed = checkpoint(
        lambda hidden: run_functional_sequence(module, (hidden,))[0],
        checkpoint_input,
        use_reentrant=False,
    )

    torch.testing.assert_close(eager, recomputed)
    torch.testing.assert_close(module.v, torch.tensor(9.0))
    eager.sum().backward()
    recomputed.sum().backward()
    torch.testing.assert_close(eager_input.grad, checkpoint_input.grad)
    torch.testing.assert_close(module.v, torch.tensor(9.0))
