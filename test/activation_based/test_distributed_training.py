from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from benchmark.snn_llm.spikelm import SpikeLMConfig
from spikingjelly.activation_based.distributed.llm.training import (
    _build_training_inputs,
    _iterator,
    _loss_totals,
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
