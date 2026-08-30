import pytest

from benchmark.energy_model_validation import (
    CROSS_ESTIMATORS,
    LEMAIRE_CASES,
    MIN_COMPARABLE_CASES,
    SPIKESIM_CASES,
    Score,
    _cross_validation_metrics,
    _metrics,
)


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
