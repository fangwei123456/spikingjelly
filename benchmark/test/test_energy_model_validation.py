import pytest
import torch
import torch.nn as nn

from benchmark.energy_model_validation import (
    CROSS_ESTIMATORS,
    LEMAIRE_CASES,
    MIN_COMPARABLE_CASES,
    SPIKESIM_CASES,
    Score,
    _cross_validation_metrics,
    _metrics,
    _neuromc_runtime_score,
)
from spikingjelly.activation_based import neuron, op_counter


def test_energy_validation_grid_and_raw_scale_metric():
    assert len(SPIKESIM_CASES) == 216
    assert {case[-1] for case in SPIKESIM_CASES} == {1, 3, 5}
    assert len(LEMAIRE_CASES) >= MIN_COMPARABLE_CASES["Lemaire"]
    assert len({case[0] for case in (*SPIKESIM_CASES, *LEMAIRE_CASES)}) == len(
        SPIKESIM_CASES
    ) + len(LEMAIRE_CASES)

    scores = [
        Score("test", str(value), "", value, value / 4, "", "", {})
        for value in (1.0, 2.0, 4.0, 8.0)
    ]
    metrics = _metrics(scores)
    assert metrics["scale_ratio"] == pytest.approx(0.25)
    assert metrics["raw_p90_factor"] == pytest.approx(4.0)
    assert metrics["scale_adjusted_p90_factor"] == pytest.approx(1.0)


def test_cross_validation_metrics_are_pairwise_and_symmetric():
    rows = [
        {
            name: float(index + 1) * (factor + 1)
            for factor, name in enumerate(CROSS_ESTIMATORS)
        }
        for index in range(4)
    ]

    for matrix in _cross_validation_metrics(rows).values():
        assert matrix.shape == (len(CROSS_ESTIMATORS), len(CROSS_ESTIMATORS))
        assert matrix == pytest.approx(matrix.T)
        assert matrix.diagonal() == pytest.approx(1.0)
        assert matrix[0, 1] == pytest.approx(1.0)


def test_neuromc_we_score_handles_even_spatial_kernel():
    item = {
        "phase": "we",
        "dims": {"C": 2, "K": 3, "OY": 2, "OX": 4, "FY": 5, "FX": 7},
    }
    model = nn.Sequential(neuron.IFNode(), nn.Conv2d(2, 3, (2, 4), bias=False)).train()
    x = torch.zeros(1, 2, 6, 10, requires_grad=True)
    expected = op_counter.estimate_neuromc_runtime_energy(
        model,
        x,
        target=torch.empty(0),
        loss_fn=lambda output, target: output.sum(),
    ).energy_by_core_type["wg"]

    assert _neuromc_runtime_score(item) == pytest.approx(expected)
