import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter


def test_snn_energy_inference_report_has_lemaire_compatible_fields():
    model = nn.Sequential(nn.Linear(8, 8, bias=False), neuron.IFNode())
    x = torch.rand(4, 8)

    report = op_counter.estimate_analytical_energy(model, x)

    assert report.energy_total_pj == pytest.approx(
        report.energy_by_component["totals"]["total_pj"]
    )
    assert report.inference_only_lemaire_compatible.available is True
    assert report.inference_only_lemaire_compatible.inference_only_E_total_pj >= 0.0
    assert report.inference_only_lemaire_compatible.inference_only_E_op_pj >= 0.0
    assert report.inference_only_lemaire_compatible.inference_only_E_addr_pj >= 0.0
    assert report.inference_only_lemaire_compatible.inference_only_E_inout_pj >= 0.0
    assert report.inference_only_lemaire_compatible.inference_only_E_params_pj >= 0.0
    assert (
        report.inference_only_lemaire_compatible.inference_only_E_potential_pj >= 0.0
    )
    assert report.inference_only_lemaire_compatible.inference_only_E_inout_pj > 0.0


def test_snn_energy_training_report_marks_inference_only_projection_unavailable():
    model = nn.Sequential(
        nn.Linear(8, 16),
        neuron.IFNode(),
        nn.Linear(16, 4),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    x = torch.rand(3, 8)
    target = torch.randn(3, 4)

    report = op_counter.estimate_analytical_energy(
        model,
        x,
        target=target,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )

    assert "forward" in report.energy_by_stage
    assert "backward" in report.energy_by_stage
    assert "optimizer" in report.energy_by_stage
    assert report.inference_only_lemaire_compatible.available is False
    assert (
        report.inference_only_lemaire_compatible.unavailable_reason
        == "Lemaire-compatible projection is inference-only."
    )


def test_snn_energy_profiler_bind_model_rejects_non_torch_backend_when_strict():
    model = neuron.IFNode()
    model._backend = "triton"
    profiler = op_counter.AnalyticalEnergyProfiler(
        config=op_counter.AnalyticalEnergyConfig(strict=True)
    )

    with pytest.raises(ValueError, match="only supports torch backend"):
        profiler.bind_model(model)


def test_snn_energy_profiler_bind_model_warns_non_torch_backend_when_not_strict():
    model = neuron.IFNode()
    model._backend = "triton"
    profiler = op_counter.AnalyticalEnergyProfiler(
        config=op_counter.AnalyticalEnergyConfig(strict=False)
    )

    with pytest.warns(RuntimeWarning, match="only supports torch backend"):
        profiler.bind_model(model)


def test_snn_energy_conv_inference_report_has_lemaire_compatible_fields():
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(),
    )
    x = torch.rand(2, 3, 8, 8)

    report = op_counter.estimate_analytical_energy(model, x)

    assert report.inference_only_lemaire_compatible.available is True
    assert report.inference_only_lemaire_compatible.inference_only_E_params_pj > 0.0
    assert report.inference_only_lemaire_compatible.inference_only_E_inout_pj > 0.0


def test_snn_energy_manual_profiler_usage_defaults_to_forward_stage():
    model = nn.Sequential(nn.Linear(8, 8, bias=False), neuron.IFNode())
    x = torch.rand(4, 8)
    profiler = op_counter.AnalyticalEnergyProfiler()
    profiler.bind_model(model)

    with profiler:
        _ = model(x)

    report = profiler.get_report()
    assert report.inference_only_lemaire_compatible.available is True
    assert report.inference_only_lemaire_compatible.inference_only_E_inout_pj > 0.0


def test_analytical_energy_cost_config_validates_memory_breakpoints():
    with pytest.raises(ValueError, match="exactly 4"):
        op_counter.AnalyticalEnergyCostConfig(memory_breakpoints=((0.0, 0.0),))

    with pytest.raises(ValueError, match="strictly increasing"):
        op_counter.AnalyticalEnergyCostConfig(
            memory_breakpoints=((0.0, 0.0), (1.0, 1.0), (1.0, 2.0), (2.0, 3.0))
        )


def test_snn_energy_suspend_does_not_contribute_to_lemaire_projection():
    model = nn.Linear(8, 8, bias=False)
    profiler = op_counter.AnalyticalEnergyProfiler()
    profiler.bind_model(model)
    x = torch.rand(4, 8)

    with profiler:
        with profiler.stage("forward"):
            _ = model(x)
        baseline = profiler.get_report().inference_only_lemaire_compatible
        with profiler.suspend():
            _ = model(x)

    report = profiler.get_report().inference_only_lemaire_compatible
    assert report.available is True
    assert report.inference_only_E_inout_pj == pytest.approx(
        baseline.inference_only_E_inout_pj
    )
    assert report.inference_only_E_params_pj == pytest.approx(
        baseline.inference_only_E_params_pj
    )


def test_snn_energy_binary_linear_does_not_produce_lemaire_mac_addr():
    model = nn.Linear(8, 4, bias=False)
    x = (torch.rand(3, 8) > 0.5).float()

    report = op_counter.estimate_analytical_energy(model, x)

    assert report.inference_only_lemaire_compatible.available is True
    assert report.inference_only_lemaire_compatible.inference_only_E_addr_pj >= 0.0

    profiler = op_counter.AnalyticalEnergyProfiler()
    profiler.bind_model(model)
    with profiler:
        with profiler.stage("forward"):
            _ = model(x)
    assert profiler._addr_estimator.mac_addr == 0


def test_snn_energy_binary_conv_produces_lemaire_mac_addr():
    model = nn.Conv2d(2, 4, kernel_size=3, bias=False)
    x = (torch.rand(1, 2, 5, 5) > 0.5).float()

    profiler = op_counter.AnalyticalEnergyProfiler()
    profiler.bind_model(model)
    with profiler:
        with profiler.stage("forward"):
            _ = model(x)

    assert profiler._addr_estimator.mac_addr > 0


def test_snn_energy_linear_inout_uses_byte_sized_accesses():
    model = nn.Linear(8, 4, bias=False)
    x = torch.rand(3, 8, dtype=torch.float32)
    profiler = op_counter.AnalyticalEnergyProfiler()
    profiler.bind_model(model)

    with profiler:
        with profiler.stage("forward"):
            _ = model(x)

    report = profiler.get_report()
    read_in_bytes = x.numel() * x.element_size()
    write_out_bytes = (3 * 4) * x.element_size()
    assert profiler._lemaire_tracker.read_in == read_in_bytes
    assert profiler._lemaire_tracker.write_out == write_out_bytes
    assert (
        report.inference_only_lemaire_compatible.inference_only_E_inout_pj
        == pytest.approx(
            read_in_bytes
            * op_counter.AnalyticalEnergyCostConfig().memory_cost_pj(read_in_bytes)
            + write_out_bytes
            * op_counter.AnalyticalEnergyCostConfig().memory_cost_pj(
                write_out_bytes
            )
        )
    )
