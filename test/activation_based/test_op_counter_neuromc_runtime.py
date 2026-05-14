import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter
from spikingjelly.activation_based.op_counter.memory_residency import (
    MemoryResidencyCounter,
    MemoryResidencySimulator,
)
from spikingjelly.activation_based.op_counter.neuromc.add_counter import NeuroMCAddCounter


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


def test_neuromc_exact_mixed_supported_and_unsupported_ops_still_raise():
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def forward(self, x):
            return torch.tanh(self.linear(x))

    model = MixedModel()
    x = torch.randn(4, 8)

    with pytest.raises(ValueError, match="does not support"):
        op_counter.estimate_neuromc_runtime_energy(model, x)


def test_neuromc_exact_frozen_linear_does_not_count_wg():
    model = nn.Linear(4, 3, bias=False)
    model.weight.requires_grad_(False)
    x = torch.randn(2, 4, requires_grad=True)
    target = torch.randn(2, 3)
    loss_fn = nn.MSELoss()

    report = op_counter.estimate_neuromc_runtime_energy(
        model, x, target=target, loss_fn=loss_fn
    )

    assert report.energy_by_core_type.get("bp_grad", 0.0) > 0.0
    assert report.energy_by_core_type.get("wg", 0.0) == 0.0


def test_neuromc_exact_functional_mm_backward_has_wg_fragment():
    class FunctionalMM(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 5))

        def forward(self, x):
            return torch.mm(x, self.weight)

    model = FunctionalMM()
    x = (torch.rand(4, 3) > 0.7).float().requires_grad_()
    target = torch.randn(4, 5)
    loss_fn = nn.MSELoss()

    report = op_counter.estimate_neuromc_runtime_energy(
        model, x, target=target, loss_fn=loss_fn
    )

    assert report.energy_by_core_type.get("bp_grad", 0.0) > 0.0
    assert report.energy_by_core_type.get("wg", 0.0) > 0.0


def test_neuromc_exact_b_type_reuse_hint_reduces_later_forward_energy():
    model = nn.Linear(16, 8, bias=False)
    x = (torch.rand(4, 16) > 0.75).float()

    with op_counter.NeuroMCEnergyProfiler() as profiler:
        profiler.bind_model(model)
        with profiler.stage("b0_t0_forward"):
            _ = model(x)
        with profiler.stage("b1_t0_forward"):
            _ = model(x)
    report = profiler.get_report()

    assert report.energy_by_stage["b0_t0_forward"] > report.energy_by_stage["b1_t0_forward"]
    assert any(
        item["b_type"] == 1 and item["t_type"] == 0 for item in report.mapping_summary
    )


def test_neuromc_exact_stage_conv_type_controls_bn_backward_counts():
    model = nn.Sequential(nn.BatchNorm1d(8), nn.Linear(8, 4))
    loss_fn = nn.MSELoss()
    x = torch.randn(3, 8, requires_grad=True)
    target = torch.randn(3, 4)

    def run(stage_name: str):
        model.zero_grad(set_to_none=True)
        with op_counter.NeuroMCEnergyProfiler() as profiler:
            profiler.bind_model(model)
            with profiler.stage("forward"):
                out = model(x)
            with profiler.suspend():
                loss = loss_fn(out, target)
            with profiler.stage(stage_name):
                loss.backward()
        return profiler.get_report()

    report_default = run("backward")
    report_without = run("without_bp_bn_backward")

    assert report_default.counts_by_process_key["with_bn"]["add"] > 0
    assert (
        report_default.counts_by_process_key["with_bn"]["add"]
        > report_without.counts_by_process_key["with_bn"]["add"]
    )


def test_neuromc_exact_bn_backward_counts_scale_with_batch():
    model = nn.Sequential(nn.BatchNorm2d(8), nn.Flatten(), nn.Linear(32, 4))
    loss_fn = nn.MSELoss()

    def run(batch_size: int):
        x = torch.randn(batch_size, 8, 2, 2, requires_grad=True)
        target = torch.randn(batch_size, 4)
        model.zero_grad(set_to_none=True)
        with op_counter.NeuroMCEnergyProfiler() as profiler:
            profiler.bind_model(model)
            with profiler.stage("forward"):
                out = model(x)
            with profiler.suspend():
                loss = loss_fn(out, target)
            with profiler.stage("backward"):
                loss.backward()
        return profiler.get_report()

    report_b1 = run(1)
    report_b4 = run(4)
    assert report_b4.counts_by_process_key["with_bn"]["add"] > report_b1.counts_by_process_key["with_bn"]["add"]


def test_memory_residency_reset_clears_state():
    counter = MemoryResidencyCounter()
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    func = torch.ops.aten.mm.default
    out = torch.mm(x, y)
    _ = counter.count(func, (x, y), {}, out)
    assert counter.get_level_bits()
    counter.reset()
    assert counter.get_level_bits() == {}
    assert counter.get_level_rw_bits() == {}
    assert counter.get_move_bits_by_edge() == {}


def test_memory_residency_views_share_storage_identity():
    sim = MemoryResidencySimulator()
    base = torch.randn(16)
    view1 = base[:8]
    view2 = base[4:12]
    sim.on_tensor_read(view1, "op")
    before = dict(sim.sram_cache)
    sim.on_tensor_read(view2, "op")
    after = dict(sim.sram_cache)
    assert len(after) == len(before)


def test_neuromc_base_counter_unknown_op_returns_zero():
    counter = NeuroMCAddCounter()
    x = torch.randn(2, 2)
    out = torch.sin(x)
    value = counter.count(torch.ops.aten.sin.default, (x,), {}, out)
    assert value == 0


def test_neuromc_exact_memory_config_api():
    cfg = op_counter.MemoryHierarchyConfig.neuromc_like_v1()
    cfg.validate()
    assert cfg.technology_nm == 32
    assert "dram" in cfg.memory_instances
