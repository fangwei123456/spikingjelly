import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmark.snn_llm.cli import _dataset_provider
from benchmark.snn_llm.spikelm import (
    SpikeLMConfig,
    _ElasticBiSpike,
    _SpikingLayerNorm,
)
from spikingjelly.activation_based.distributed.llm.temporal import (
    run_functional_sequence,
)


def test_elastic_bi_spike_is_functional_and_bidirectional():
    module = _ElasticBiSpike(time_steps=3, decay=0.25, amplitude=1.0)
    sequence = torch.tensor([[[[0.6]]], [[[-0.8]]], [[[1.4]]]], requires_grad=True)

    output = run_functional_sequence(module, (sequence,))[0]
    output.sum().backward()

    assert torch.equal(output.detach().flatten(), torch.tensor([1.0, -1.0, 1.0]))
    assert module.v == 0.0
    assert sequence.grad is not None
    assert torch.isfinite(sequence.grad).all()


def test_spikelm_config_rejects_invalid_temporal_values():
    values = dict(
        transformer=object(),
        vocab_size=128,
        max_sequence_length=64,
        time_steps=4,
    )
    config = SpikeLMConfig(**values)

    assert config.spike_decay == 0.25
    with pytest.raises(ValueError, match="spike_amplitude"):
        SpikeLMConfig(**values, spike_amplitude=0.0)


def test_spikelm_npy_dataset_repeats_to_requested_sample_count():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        np.save(path / "train.npy", np.arange(10).reshape(2, 5))
        train, valid, test = _dataset_provider(
            (3, 0, 0), data_dir=path, sequence_length=4
        )

        assert len(train) == 3
        assert torch.equal(train[2]["input_ids"], torch.tensor([0, 1, 2, 3]))
        assert len(valid) == len(test) == 0


def test_spikelm_npy_dataset_rejects_non_integer_tokens():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        np.save(path / "train.npy", np.zeros((1, 5), dtype=np.float32))
        with pytest.raises(ValueError, match="integer array"):
            _dataset_provider((1, 0, 0), data_dir=path, sequence_length=4)


def test_spikelm_memopt_preserves_output_and_gradient():
    plain = _SpikingLayerNorm(
        config=object(),
        hidden_size=3,
        eps=1e-5,
        time_steps=2,
        decay=0.25,
        amplitude=1.0,
        use_snn_memopt=False,
    )
    optimized = _SpikingLayerNorm(
        config=object(),
        hidden_size=3,
        eps=1e-5,
        time_steps=2,
        decay=0.25,
        amplitude=1.0,
        use_snn_memopt=True,
    )
    optimized.load_state_dict(plain.state_dict())
    plain_input = torch.randn(4, 4, 3, requires_grad=True)
    optimized_input = plain_input.detach().clone().requires_grad_(True)

    plain(plain_input).sum().backward()
    optimized(optimized_input).sum().backward()

    assert torch.equal(plain(plain_input.detach()), optimized(optimized_input.detach()))
    assert torch.equal(plain_input.grad, optimized_input.grad)


def test_spiking_layer_norm_keeps_deep_residual_gradients_bounded():
    torch.manual_seed(0)
    hidden = torch.randn(8, 4, 16, requires_grad=True)
    output = hidden
    for _ in range(12):
        layer = _SpikingLayerNorm(
            config=object(),
            hidden_size=16,
            eps=1e-5,
            time_steps=4,
            decay=0.25,
            amplitude=1.0,
            use_snn_memopt=False,
        )
        output = output + layer(output)

    output.square().mean().backward()

    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad.abs().max() < 10.0
