from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from benchmark.snn_llm.spikelm import SpikeLMConfig
from spikingjelly.activation_based.distributed.llm.metrics import (
    _loss_totals,
    _reduce_data_parallel_metrics,
)
from spikingjelly.activation_based.distributed.llm.training import (
    _build_training_inputs,
    _iterator,
)


def test_training_config_resolves_model_and_dataset_builders():
    config = SimpleNamespace(
        model=SpikeLMConfig(
            transformer=object(),
            vocab_size=128,
            max_sequence_length=8,
            time_steps=2,
        ),
        use_snn_memopt=False,
        resume=None,
        dataset_builder="benchmark.snn_llm.cli._dataset_provider",
        dataset_kwargs={"data_dir": "tokens", "sequence_length": 8},
    )

    model_provider, dataset_provider, forward_step = _build_training_inputs(config)

    assert callable(model_provider)
    assert dataset_provider.keywords == {
        "data_dir": "tokens",
        "sequence_length": 8,
    }
    assert callable(forward_step)


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
