import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter


def test_snn_energy_inference_report_has_lemaire_compatible_fields():
    model = nn.Sequential(nn.Linear(8, 8, bias=False), neuron.IFNode())
    x = torch.rand(4, 8)

    report = op_counter.estimate_snn_energy(model, x)

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

    report = op_counter.estimate_snn_energy(
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
    profiler = op_counter.SNNEnergyProfiler(
        config=op_counter.SNNEnergyConfig(strict=True)
    )

    with pytest.raises(ValueError, match="only supports torch backend"):
        profiler.bind_model(model)


def test_snn_energy_profiler_bind_model_warns_non_torch_backend_when_not_strict():
    model = neuron.IFNode()
    model._backend = "triton"
    profiler = op_counter.SNNEnergyProfiler(
        config=op_counter.SNNEnergyConfig(strict=False)
    )

    with pytest.warns(RuntimeWarning, match="only supports torch backend"):
        profiler.bind_model(model)
