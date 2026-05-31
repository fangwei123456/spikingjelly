import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import op_counter


def test_compute_energy_uses_mac_and_ac_as_authoritative_total():
    class AddModel(nn.Module):
        def forward(self, x, y):
            return x + y

    model = AddModel()
    x = torch.ones(2, 3)
    y = torch.ones(2, 3)

    report = op_counter.estimate_compute_energy(model, (x, y))

    assert report.counts["mac"] == 0
    assert report.counts["ac"] == 12
    assert report.energy_mac_pj == pytest.approx(0.0)
    assert report.energy_ac_pj == pytest.approx(12 * 0.9)
    assert report.energy_total_pj == pytest.approx(
        report.energy_mac_pj + report.energy_ac_pj
    )


def test_compute_energy_spike_linear_counts_synop_as_auxiliary_only():
    model = nn.Linear(4, 3, bias=False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    report = op_counter.estimate_compute_energy(model, x)

    assert report.counts["synop"] == 6
    assert report.counts["ac"] == 6
    assert report.counts["mac"] == 0
    assert report.energy_total_pj == pytest.approx(6 * 0.9)


def test_compute_energy_dense_linear_counts_mac():
    model = nn.Linear(4, 3, bias=False)
    x = torch.full((2, 4), 0.5)

    report = op_counter.estimate_compute_energy(model, x)

    assert report.counts["mac"] == 24
    assert report.counts["ac"] == 0
    assert report.energy_total_pj == pytest.approx(24 * 4.6)


def test_compute_energy_profiler_matches_convenience_function():
    model = nn.Linear(4, 2, bias=False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    profiler = op_counter.ComputeEnergyProfiler()
    with profiler:
        _ = model(x)

    report_ctx = profiler.get_report()
    report_fn = op_counter.estimate_compute_energy(model, x)

    assert report_ctx.energy_total_pj == pytest.approx(report_fn.energy_total_pj)
    assert report_ctx.counts == report_fn.counts


def test_compute_energy_custom_cost_config_is_applied():
    model = nn.Linear(4, 2, bias=False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    cfg = op_counter.ComputeEnergyConfig(
        cost_config=op_counter.ComputeEnergyCostConfig(e_mac_pj=10.0, e_ac_pj=2.0)
    )

    report = op_counter.estimate_compute_energy(model, x, config=cfg)

    assert report.counts["ac"] == 4
    assert report.energy_total_pj == pytest.approx(8.0)


def test_compute_energy_cost_config_presets_match_horowitz_reference_table():
    fp32 = op_counter.ComputeEnergyCostConfig.fp32()
    fp16 = op_counter.ComputeEnergyCostConfig.fp16()
    int8 = op_counter.ComputeEnergyCostConfig.int8()

    assert fp32.e_mac_pj == pytest.approx(4.6)
    assert fp32.e_ac_pj == pytest.approx(0.9)
    assert fp16.e_mac_pj == pytest.approx(1.5)
    assert fp16.e_ac_pj == pytest.approx(0.4)
    assert int8.e_mac_pj == pytest.approx(0.23)
    assert int8.e_ac_pj == pytest.approx(0.03)


def test_compute_energy_default_cost_config_matches_fp32_preset():
    cfg = op_counter.ComputeEnergyConfig()
    fp32 = op_counter.ComputeEnergyCostConfig.fp32()

    assert cfg.cost_config.e_mac_pj == pytest.approx(fp32.e_mac_pj)
    assert cfg.cost_config.e_ac_pj == pytest.approx(fp32.e_ac_pj)


def test_compute_energy_fp16_preset_changes_only_comparison_regime():
    model = nn.Linear(4, 2, bias=False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    cfg = op_counter.ComputeEnergyConfig(
        cost_config=op_counter.ComputeEnergyCostConfig.fp16()
    )

    report = op_counter.estimate_compute_energy(model, x, config=cfg)

    assert report.counts["ac"] == 4
    assert report.counts["mac"] == 0
    assert report.energy_total_pj == pytest.approx(4 * 0.4)


def test_compute_energy_warns_when_no_supported_ops_are_profiled():
    model = nn.ReLU()
    x = torch.ones(2, 3)

    report = op_counter.estimate_compute_energy(model, x)

    assert report.energy_total_pj == pytest.approx(0.0)
    assert any(
        "did not match any supported operators" in msg for msg in report.warnings
    )


def test_compute_energy_strict_raises_when_no_supported_ops_are_profiled():
    model = nn.ReLU()
    x = torch.ones(2, 3)
    cfg = op_counter.ComputeEnergyConfig(strict=True)

    with pytest.raises(RuntimeError, match="did not match any supported operators"):
        op_counter.estimate_compute_energy(model, x, config=cfg)


def test_compute_energy_strict_allows_zero_work_when_supported_op_matches():
    model = nn.Linear(4, 3, bias=False)
    x = torch.empty(0, 4)
    cfg = op_counter.ComputeEnergyConfig(strict=True)

    report = op_counter.estimate_compute_energy(model, x, config=cfg)

    assert report.counts["mac"] == 0
    assert report.counts["ac"] == 0
    assert report.counts["synop"] == 0
    assert report.counts["flop"] == 0
    assert report.warnings == []


def test_compute_energy_supports_dict_inputs_for_keyword_only_models():
    class KeywordOnlyAdd(nn.Module):
        def forward(self, *, x, y):
            return x + y

    model = KeywordOnlyAdd()
    inputs = {
        "x": torch.ones(2, 3),
        "y": torch.ones(2, 3),
    }

    report = op_counter.estimate_compute_energy(model, inputs)

    assert report.counts["mac"] == 0
    assert report.counts["ac"] == 12
