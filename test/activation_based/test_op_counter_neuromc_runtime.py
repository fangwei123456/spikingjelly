import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter


def test_neuromc_exact_linear_report_fields():
    model = nn.Linear(8, 4, bias=False)
    x = torch.randn(3, 8)

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    assert report.energy_total_pj > 0.0
    assert report.energy_compute_pj == report.energy_mac_pj + report.energy_extra_compute_pj
    assert report.energy_memory_pj == report.energy_base_memory_pj + report.energy_extra_memory_pj
    assert report.energy_total_pj == pytest.approx(
        report.energy_compute_pj + report.energy_memory_pj
    )
    assert report.energy_by_core_type["fp_soma"] > 0.0
    assert report.counts_by_core_type["fp_soma"]["mac"] == 3 * 8 * 4
    assert report.primitive_counts["totals"]["mac"] == 3 * 8 * 4


def test_neuromc_exact_ifnode_supports_sg_breakdown():
    model = nn.Sequential(nn.Linear(6, 6, bias=False), neuron.IFNode())
    x = (torch.rand(2, 6) > 0.6).float()

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    assert report.energy_by_process_key["with_sg"] > 0.0
    assert report.counts_by_process_key["with_sg"]["mux"] > 0
    assert report.counts_by_process_key["with_sg"]["comp"] > 0


def test_neuromc_exact_batchnorm_supports_bn_breakdown():
    model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8))
    x = torch.randn(4, 8)

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    assert report.energy_by_core_type["fp_bn"] > 0.0
    assert report.energy_by_process_key["with_bn"] > 0.0
    assert report.counts_by_process_key["with_bn"]["sqrt"] > 0


def test_neuromc_exact_training_uses_bp_wg_and_optimizer():
    model = nn.Sequential(nn.Linear(8, 16), neuron.IFNode(), nn.Linear(16, 4))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x = (torch.rand(3, 8) > 0.7).float()
    target = torch.randn(3, 4)
    report = op_counter.estimate_neuromc_runtime_energy(
        model, x, target=target, loss_fn=loss_fn, optimizer=optimizer
    )

    assert report.energy_by_stage["forward"] > 0.0
    assert report.energy_by_stage["backward"] > 0.0
    assert report.energy_by_stage["optimizer"] > 0.0
    assert report.energy_by_core_type["bp_grad"] > 0.0
    assert report.energy_by_core_type["wg"] > 0.0
    assert report.energy_by_core_type["bp_grad_opt"] > 0.0
    assert report.energy_by_process_key["with_opt"] > 0.0
    assert sum(report.energy_by_stage.values()) == pytest.approx(report.energy_total_pj)


def test_neuromc_exact_repeated_forward_reuses_weights():
    model = nn.Linear(16, 8, bias=False)
    x = (torch.rand(4, 16) > 0.75).float()

    with op_counter.NeuroMCEnergyProfiler() as profiler:
        profiler.bind_model(model)
        with profiler.stage("t0_forward"):
            _ = model(x)
        with profiler.stage("t1_forward"):
            _ = model(x)
    report = profiler.get_report()

    assert report.energy_by_stage["t0_forward"] > report.energy_by_stage["t1_forward"]
    assert any(item["t_type"] == 1 for item in report.mapping_summary)


def test_neuromc_exact_sgd_optimizer_is_rejected():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    x = torch.randn(2, 4)
    target = torch.randn(2, 2)

    with pytest.raises(ValueError, match="Adam/AdamW"):
        op_counter.estimate_neuromc_runtime_energy(
            model, x, target=target, loss_fn=loss_fn, optimizer=optimizer
        )


def test_neuromc_exact_unsupported_op_raises():
    class UnsupportedModel(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    model = UnsupportedModel()
    x = torch.randn(8, 8)

    with pytest.raises(ValueError, match="does not support"):
        op_counter.estimate_neuromc_runtime_energy(model, x)


def test_neuromc_exact_memory_config_api():
    cfg = op_counter.MemoryHierarchyConfig.neuromc_like_v1()
    cfg.validate()
    assert cfg.technology_nm == 32
    assert "dram" in cfg.memory_instances
