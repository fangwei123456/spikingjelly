from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from spikingjelly.activation_based.distributed.llm.metrics import (
    _loss_totals,
    _reduce_data_parallel_metrics,
)
from spikingjelly.activation_based.distributed.llm.training import (
    _iterator,
)


def test_training_helpers_resume_data_and_reduce_weighted_losses():
    dataset = TensorDataset(torch.arange(8))
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
    batches = _iterator(DataLoader(dataset, batch_size=2, sampler=sampler), 3)

    assert torch.equal(next(batches)[0], torch.tensor([6, 7]))
    assert torch.equal(next(batches)[0], torch.tensor([0, 1]))
    assert _loss_totals(
        [{"loss": torch.tensor([6.0, 2.0])}, {"loss": torch.tensor([3.0, 1.0])}]
    ) == {"loss": (9.0, 3.0)}


def test_metric_reduction_preserves_large_token_counts(monkeypatch):
    count = 2**24 + 1
    totals = _loss_totals([{"loss": torch.tensor([1.0, count], dtype=torch.float64)}])
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)
    parallel_state = SimpleNamespace(get_data_parallel_group=lambda **_kwargs: None)

    metrics = _reduce_data_parallel_metrics(totals, parallel_state, torch.device("cpu"))

    assert totals["loss"][1] == count
    assert metrics["loss"] == 1.0 / count


def test_loss_totals_rejects_ambiguous_metric_tensors():
    with pytest.raises(ValueError, match="one or two elements"):
        _loss_totals([{"loss": torch.ones(3)}])
