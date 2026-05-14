import pytest
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, op_counter
from spikingjelly.activation_based.op_counter.memory_residency import (
    MemoryResidencyCounter,
    MemoryResidencySimulator,
)
from spikingjelly.activation_based.op_counter.neuromc.base_counter import (
    NeuroMCBaseCounter,
)
from spikingjelly.activation_based.op_counter.neuromc.memory_residency_counter import (
    NeuroMCMemoryResidencyCounter,
)
from spikingjelly.activation_based.op_counter.neuromc.utils import _is_spike, _spike_nnz


def test_neuromc_exact_linear_report_fields():
    model = nn.Linear(8, 4, bias=False)
    x = (torch.rand(3, 8) > 0.6).float()

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    assert report.energy_total_pj > 0.0
    assert (
        report.energy_compute_pj
        == report.energy_mac_pj + report.energy_extra_compute_pj
    )
    assert (
        report.energy_memory_pj
        == report.energy_base_memory_pj + report.energy_extra_memory_pj
    )
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
    x = (torch.rand(4, 8) > 0.5).float()

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
    x = (torch.rand(2, 4) > 0.5).float()
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
    x = (torch.rand(4, 8) > 0.5).float()

    with pytest.raises(ValueError, match="does not support"):
        op_counter.estimate_neuromc_runtime_energy(model, x)


def test_neuromc_exact_frozen_linear_does_not_count_wg():
    model = nn.Linear(4, 3, bias=False)
    model.weight.requires_grad_(False)
    x = (torch.rand(2, 4) > 0.5).float().requires_grad_()
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


def test_neuromc_exact_mixed_hook_and_functional_mm_counts_both_paths():
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=False)
            self.weight = nn.Parameter(torch.randn(4, 3))

        def forward(self, x):
            return self.linear(x)[:, :3] + torch.mm(x, self.weight)

    model = MixedModel()
    x = (torch.rand(5, 4) > 0.5).float()

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    expected_mac = 5 * 4 * 4 + 5 * 4 * 3
    assert report.primitive_counts["totals"]["mac"] == expected_mac
    assert any(item["op_name"] == "linear.forward" for item in report.mapping_summary)
    assert any(item["op_name"] == "aten.mm.default" for item in report.mapping_summary)


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

    assert (
        report.energy_by_stage["b0_t0_forward"]
        > report.energy_by_stage["b1_t0_forward"]
    )
    assert any(
        item["b_type"] == 1 and item["t_type"] == 0 for item in report.mapping_summary
    )


def test_neuromc_exact_stage_conv_type_controls_bn_backward_counts():
    model = nn.BatchNorm1d(8)
    loss_fn = nn.MSELoss()
    x = torch.randn(3, 8, requires_grad=True)
    target = torch.randn(3, 8)

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
    model = nn.BatchNorm2d(8)
    loss_fn = nn.MSELoss()

    def run(batch_size: int):
        x = torch.randn(batch_size, 8, 2, 2, requires_grad=True)
        target = torch.randn(batch_size, 8, 2, 2)
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
    assert (
        report_b4.counts_by_process_key["with_bn"]["add"]
        > report_b1.counts_by_process_key["with_bn"]["add"]
    )


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


def test_memory_residency_unknown_op_returns_zero():
    counter = MemoryResidencyCounter()
    x = torch.randn(2, 2)
    out = torch.sin(x)
    value = counter.count(torch.ops.aten.sin.default, (x,), {}, out)
    assert value == 0


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
    counter = NeuroMCBaseCounter()
    x = torch.randn(2, 2)
    out = torch.sin(x)
    value = counter.count(torch.ops.aten.sin.default, (x,), {}, out)
    assert value == 0


def test_neuromc_spike_nnz_empty_tensor_returns_none():
    x = torch.empty(0)
    assert _is_spike(x) is False
    assert _spike_nnz(x) is None


def test_neuromc_profiler_cleans_last_input_attribute():
    model = nn.Linear(4, 3)
    x = (torch.rand(2, 4) > 0.5).float()
    with op_counter.NeuroMCEnergyProfiler() as profiler:
        profiler.bind_model(model)
        with profiler.stage("forward"):
            _ = model(x)
        assert hasattr(model, "_neuromc_last_input")
    assert not hasattr(model, "_neuromc_last_input")


def test_neuromc_exact_conv3d_is_rejected():
    model = nn.Conv3d(2, 4, kernel_size=3, padding=1)
    x = torch.randn(1, 2, 4, 4, 4)
    with pytest.raises(ValueError, match="Conv3d"):
        op_counter.estimate_neuromc_runtime_energy(model, x)


def test_neuromc_bind_model_rejects_conv3d_before_registering_hooks():
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.conv3d = nn.Conv3d(1, 1, kernel_size=3, padding=1)

    model = MixedModel()
    profiler = op_counter.NeuroMCEnergyProfiler()
    with pytest.raises(ValueError, match="Conv3d"):
        profiler.bind_model(model)
    assert len(model.linear._forward_hooks) == 0
    assert len(model.linear._backward_hooks) == 0


def test_neuromc_bind_model_rejects_rebinding_different_model():
    profiler = op_counter.NeuroMCEnergyProfiler()
    model1 = nn.Linear(4, 3)
    model2 = nn.Linear(4, 3)
    profiler.bind_model(model1)
    with pytest.raises(RuntimeError, match="already bound"):
        profiler.bind_model(model2)


def test_neuromc_exact_ann_forward_input_is_rejected():
    model = nn.Linear(8, 4, bias=False)
    x = torch.randn(3, 8)
    with pytest.raises(ValueError, match="ANN-like dense activations"):
        op_counter.estimate_neuromc_runtime_energy(model, x)


def test_neuromc_exact_memory_config_api():
    cfg = op_counter.MemoryHierarchyConfig.neuromc_like_v1()
    cfg.validate()
    assert cfg.technology_nm == 32
    assert "dram" in cfg.memory_instances
    assert hasattr(op_counter.neuromc, "MemoryResidencySimulator")


def test_neuromc_exact_trace_events_do_not_retain_live_tensors():
    model = nn.Linear(4, 3, bias=False)
    x = (torch.rand(2, 4) > 0.5).float()

    with op_counter.NeuroMCEnergyProfiler() as profiler:
        profiler.bind_model(model)
        with profiler.stage("forward"):
            _ = model(x)

    assert profiler._trace_events
    first_event = profiler._trace_events[0]
    assert not any(torch.is_tensor(v) for v in first_event.args)


def test_neuromc_exact_optimizer_counts_mixed_parameter_shapes():
    model = nn.Linear(2, 3, bias=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    profiler = op_counter.NeuroMCEnergyProfiler()
    profiler.bind_optimizer(optimizer)

    fragment = profiler._optimizer_fragment("optimizer")
    counts = profiler._extra_counts(fragment)
    bits, _ = profiler._extra_memory_for_fragment(fragment)

    assert counts["add"] == 48
    assert counts["mul"] == 132
    assert counts["sqrt"] == 12
    assert bits["sram"]["rh2l"] == (14 * 3 + 7 * 6) * 32
    assert bits["sram"]["wl2h"] == (14 * 3 + 7 * 6) * 32


def test_neuromc_memory_residency_legacy_memory_config_alias():
    cfg = op_counter.MemoryHierarchyConfig.neuromc_like_v1()
    with pytest.deprecated_call():
        counter = NeuroMCMemoryResidencyCounter(memory_config=cfg)
    assert isinstance(counter, MemoryResidencyCounter)


def test_neuromc_exact_multi_step_conv2d_uses_true_channel_and_time_dims():
    model = layer.Conv2d(2, 3, kernel_size=3, padding=1, bias=False, step_mode="m")
    x = (torch.rand(4, 2, 2, 5, 5) > 0.5).float()

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    expected_mac = 4 * 2 * 2 * 3 * 5 * 5 * 3 * 3
    assert report.primitive_counts["totals"]["mac"] == expected_mac
    assert any(
        item["loop_dims"]["T"] == 4
        and item["loop_dims"]["B"] == 2
        and item["loop_dims"]["C"] == 2
        and item["loop_dims"]["K"] == 3
        for item in report.mapping_summary
        if item["op_name"] == "conv.forward"
    )


def test_neuromc_exact_multi_step_batchnorm2d_uses_true_channel_dim():
    model = layer.BatchNorm2d(4, step_mode="m")
    x = torch.randn(3, 2, 4, 5, 5)

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    assert report.counts_by_process_key["with_bn"]["sqrt"] == 4 * 3
    assert any(
        item["loop_dims"]["T"] == 3
        and item["loop_dims"]["B"] == 2
        and item["loop_dims"]["K"] == 4
        for item in report.mapping_summary
        if item["op_name"] == "bn.forward"
    )
