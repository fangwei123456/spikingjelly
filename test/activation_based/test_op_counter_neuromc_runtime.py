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

    assert report.energy_by_core_type["ann_bn"] > 0.0
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


def _optimizer_fragment_totals(profiler, fragments):
    total_add = 0
    total_mul = 0
    total_sqrt = 0
    total_rh2l = 0
    total_wl2h = 0
    for fragment in fragments:
        counts = profiler._extra_counts(fragment)
        bits, _ = profiler._extra_memory_for_fragment(fragment)
        total_add += counts["add"]
        total_mul += counts["mul"]
        total_sqrt += counts["sqrt"]
        total_rh2l += bits["sram"].get("rh2l", 0)
        total_wl2h += bits["sram"].get("wl2h", 0)
    return total_add, total_mul, total_sqrt, total_rh2l, total_wl2h


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


def test_neuromc_exact_plain_sgd_optimizer_is_supported():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    x = (torch.rand(2, 4) > 0.5).float()
    target = torch.randn(2, 2)

    report = op_counter.estimate_neuromc_runtime_energy(
        model, x, target=target, loss_fn=loss_fn, optimizer=optimizer
    )

    assert report.energy_by_stage["optimizer"] > 0.0
    assert report.energy_by_core_type["bp_grad_opt"] > 0.0
    assert report.energy_by_process_key["with_opt"] > 0.0
    assert any(
        item["source"] == "optimizer"
        and item["op_name"] == "sgd"
        and item["optimizer_has_momentum"] is False
        and item["optimizer_has_weight_decay"] is False
        for item in report.mapping_summary
    )


def test_neuromc_exact_sgd_with_momentum_and_weight_decay_is_supported():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01
    )
    loss_fn = nn.MSELoss()
    x = (torch.rand(2, 4) > 0.5).float()
    target = torch.randn(2, 2)

    report = op_counter.estimate_neuromc_runtime_energy(
        model, x, target=target, loss_fn=loss_fn, optimizer=optimizer
    )

    assert report.energy_by_stage["optimizer"] > 0.0
    assert any(
        item["source"] == "optimizer"
        and item["op_name"] == "sgd"
        and item["optimizer_has_momentum"] is True
        and item["optimizer_has_weight_decay"] is True
        for item in report.mapping_summary
    )


def test_neuromc_exact_online_learning_optimizer_step_each_time_step():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.MSELoss()
    xs = [torch.randn(3, 4), torch.randn(3, 4)]
    targets = [torch.randn(3, 2), torch.randn(3, 2)]

    with op_counter.NeuroMCEnergyProfiler() as profiler:
        profiler.bind_model(model)
        profiler.bind_optimizer(optimizer)
        for t, (x, target) in enumerate(zip(xs, targets, strict=True)):
            with profiler.stage(f"t{t}_forward"):
                out = model(x)
            with profiler.suspend():
                loss = loss_fn(out, target)
            with profiler.stage(f"t{t}_backward"):
                loss.backward()
            with profiler.stage(f"t{t}_optimizer"):
                profiler.record_optimizer_step(f"t{t}_optimizer")
                with profiler.suspend():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        report = profiler.get_report()

    optimizer_items = [
        item for item in report.mapping_summary if item["source"] == "optimizer"
    ]
    assert report.energy_by_stage["t0_optimizer"] > 0.0
    assert report.energy_by_stage["t1_optimizer"] > 0.0
    assert {item["stage"] for item in optimizer_items} == {
        "t0_optimizer",
        "t1_optimizer",
    }
    assert any(
        item["stage"] == "t0_optimizer"
        and item["op_name"] == "sgd"
        and item["optimizer_has_momentum"] is True
        and item["optimizer_has_momentum_buffer"] is False
        for item in optimizer_items
    )
    assert any(
        item["stage"] == "t1_optimizer"
        and item["op_name"] == "sgd"
        and item["optimizer_has_momentum"] is True
        and item["optimizer_has_momentum_buffer"] is True
        for item in optimizer_items
    )


def test_neuromc_exact_sgd_optimizer_rejects_nesterov():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    profiler = op_counter.NeuroMCEnergyProfiler()
    profiler.bind_optimizer(optimizer)

    with pytest.raises(ValueError, match="nesterov=False"):
        profiler._optimizer_fragments("optimizer")


def test_neuromc_exact_sgd_optimizer_rejects_dampening():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, dampening=0.1)
    profiler = op_counter.NeuroMCEnergyProfiler()
    profiler.bind_optimizer(optimizer)

    with pytest.raises(ValueError, match="dampening=0"):
        profiler._optimizer_fragments("optimizer")


def test_neuromc_exact_sgd_optimizer_rejects_maximize():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, maximize=True)
    profiler = op_counter.NeuroMCEnergyProfiler()
    profiler.bind_optimizer(optimizer)

    with pytest.raises(ValueError, match="maximize=False"):
        profiler._optimizer_fragments("optimizer")


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


def test_neuromc_exact_dense_functional_mm_backward_uses_ann_paths():
    class FunctionalMM(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 5))

        def forward(self, x):
            return torch.mm(x, self.weight)

    model = FunctionalMM()
    x = torch.randn(4, 3, requires_grad=True)
    target = torch.randn(4, 5)
    loss_fn = nn.MSELoss()

    report = op_counter.estimate_neuromc_runtime_energy(
        model, x, target=target, loss_fn=loss_fn
    )

    assert report.energy_by_core_type.get("ann_be", 0.0) > 0.0
    assert report.energy_by_core_type.get("ann_we", 0.0) > 0.0


def test_neuromc_exact_same_shape_functional_mm_backward_keeps_spike_and_ann_labels():
    class DualFunctionalMM(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_dense = nn.Parameter(torch.randn(3, 5))
            self.weight_spike = nn.Parameter(torch.randn(3, 5))

        def forward(self, x_dense, x_spike):
            return torch.mm(x_dense, self.weight_dense) + torch.mm(
                x_spike, self.weight_spike
            )

    model = DualFunctionalMM()
    x_dense = torch.randn(4, 3, requires_grad=True)
    x_spike = (torch.rand(4, 3) > 0.7).float().requires_grad_()
    target = torch.randn(4, 5)
    loss_fn = nn.MSELoss()

    report = op_counter.estimate_neuromc_runtime_energy(
        model, (x_dense, x_spike), target=target, loss_fn=loss_fn
    )

    assert report.energy_by_core_type.get("ann_be", 0.0) > 0.0
    assert report.energy_by_core_type.get("bp_grad", 0.0) > 0.0
    assert report.energy_by_core_type.get("ann_we", 0.0) > 0.0
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


def test_neuromc_exact_mixed_hook_and_functional_addmm_counts_both_paths():
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=False)
            self.bias = nn.Parameter(torch.randn(3))
            self.weight = nn.Parameter(torch.randn(4, 3))

        def forward(self, x):
            return self.linear(x)[:, :3] + torch.addmm(self.bias, x, self.weight)

    model = MixedModel()
    x = (torch.rand(5, 4) > 0.5).float()

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    expected_mac = 5 * 4 * 4 + 5 * 4 * 3
    assert report.primitive_counts["totals"]["mac"] == expected_mac
    assert any(item["op_name"] == "linear.forward" for item in report.mapping_summary)
    assert any(
        item["op_name"] == "aten.addmm.default" for item in report.mapping_summary
    )


def test_neuromc_exact_dense_linear_uses_ann_forward_path():
    model = nn.Linear(8, 4, bias=False)
    x = torch.randn(3, 8)

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    assert report.energy_by_core_type["ann_fe"] > 0.0
    assert report.energy_base_memory_pj > 0.0
    assert any(
        item["op_name"] == "linear.forward"
        and item["core_type"] == "ann_fe"
        and item["input_precision_bits"] == 16
        for item in report.mapping_summary
    )


def test_neuromc_exact_dense_conv2d_uses_ann_forward_path():
    model = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
    x = torch.randn(4, 2, 5, 5)

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    assert report.energy_by_core_type["ann_fe"] > 0.0
    assert report.energy_base_memory_pj > 0.0
    assert any(
        item["op_name"] == "conv.forward"
        and item["core_type"] == "ann_fe"
        and item["input_precision_bits"] == 16
        for item in report.mapping_summary
    )


def test_neuromc_exact_dense_training_uses_ann_backward_paths():
    model = nn.Linear(8, 4, bias=False)
    x = torch.randn(3, 8, requires_grad=True)
    target = torch.randn(3, 4)
    loss_fn = nn.MSELoss()

    report = op_counter.estimate_neuromc_runtime_energy(
        model, x, target=target, loss_fn=loss_fn
    )

    assert report.energy_by_core_type["ann_fe"] > 0.0
    assert report.energy_by_core_type["ann_be"] > 0.0
    assert report.energy_by_core_type["ann_we"] > 0.0


def test_neuromc_exact_dense_conv2d_backward_uses_ann_paths():
    model = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
    x = torch.randn(3, 2, 5, 5)
    grad_in = torch.randn_like(x)
    grad_out = torch.randn(3, 3, 5, 5)
    profiler = op_counter.NeuroMCEnergyProfiler()

    model._neuromc_last_input = x
    fragments = profiler._make_conv_backward_fragments(
        "backward", model, (grad_in,), (grad_out,)
    )

    assert {fragment.core_type for fragment in fragments} == {"ann_be", "ann_we"}
    assert all(fragment.input_precision_bits == 16 for fragment in fragments)


def test_neuromc_exact_dense_weight_grad_costs_more_than_spike_weight_grad():
    torch.manual_seed(0)
    model_spike = nn.Linear(8, 4, bias=False)
    x_spike = (torch.rand(3, 8) > 0.6).float().requires_grad_()
    target_spike = torch.randn(3, 4)
    loss_fn = nn.MSELoss()
    spike_report = op_counter.estimate_neuromc_runtime_energy(
        model_spike, x_spike, target=target_spike, loss_fn=loss_fn
    )

    model_dense = nn.Linear(8, 4, bias=False)
    x_dense = torch.randn(3, 8, requires_grad=True)
    target_dense = torch.randn(3, 4)
    dense_report = op_counter.estimate_neuromc_runtime_energy(
        model_dense, x_dense, target=target_dense, loss_fn=loss_fn
    )

    assert spike_report.energy_by_core_type["wg"] > 0.0
    assert dense_report.energy_by_core_type["ann_we"] > 0.0
    assert (
        dense_report.energy_by_core_type["ann_we"]
        > spike_report.energy_by_core_type["wg"]
    )


def test_neuromc_exact_dense_batchnorm_training_uses_ann_bn_labels():
    model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8))
    x = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8)
    loss_fn = nn.MSELoss()

    report = op_counter.estimate_neuromc_runtime_energy(
        model, x, target=target, loss_fn=loss_fn
    )

    assert report.energy_by_core_type["ann_bn"] > 0.0
    assert report.energy_by_core_type["ann_bn_bp"] > 0.0
    assert report.energy_by_process_key["with_bn"] > 0.0


def test_neuromc_exact_dense_batchnorm2d_forward_uses_ann_bn_label():
    model = nn.Sequential(nn.Conv2d(2, 4, kernel_size=3, padding=1), nn.BatchNorm2d(4))
    x = torch.randn(2, 2, 5, 5)

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    assert report.energy_by_core_type["ann_bn"] > 0.0
    assert any(
        item["op_name"] == "bn.forward" and item["core_type"] == "ann_bn"
        for item in report.mapping_summary
    )


def test_neuromc_exact_mixed_model_dense_bn_keeps_ann_label_before_spike_node():
    model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), neuron.IFNode())
    x = torch.randn(3, 8)

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    assert report.energy_by_core_type["ann_bn"] > 0.0
    assert not any(
        item["op_name"] == "bn.forward" and item["core_type"] == "fp_bn"
        for item in report.mapping_summary
    )


def test_neuromc_exact_mixed_model_reports_ann_and_snn_paths():
    model = nn.Sequential(nn.Linear(8, 8), neuron.IFNode(), nn.Linear(8, 4))
    x = torch.randn(3, 8)

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    assert report.energy_by_core_type["ann_fe"] > 0.0
    assert report.energy_by_core_type["fp_soma"] > 0.0
    assert any(
        item["op_name"] == "linear.forward" and item["core_type"] == "ann_fe"
        for item in report.mapping_summary
    )
    assert any(
        item["op_name"] == "linear.forward" and item["core_type"] == "fp_soma"
        for item in report.mapping_summary
    )


def test_neuromc_exact_dense_mixed_hook_and_functional_mm_use_ann_path():
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=False)
            self.weight = nn.Parameter(torch.randn(4, 3))

        def forward(self, x):
            return self.linear(x)[:, :3] + torch.mm(x, self.weight)

    model = MixedModel()
    x = torch.randn(5, 4)

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    expected_mac = 5 * 4 * 4 + 5 * 4 * 3
    assert report.primitive_counts["totals"]["mac"] == expected_mac
    assert report.counts_by_core_type["ann_fe"]["mac"] == expected_mac
    assert any(
        item["op_name"] == "aten.mm.default" and item["core_type"] == "ann_fe"
        for item in report.mapping_summary
    )


def test_neuromc_exact_dense_mixed_hook_and_functional_addmm_use_ann_path():
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=False)
            self.bias = nn.Parameter(torch.randn(3))
            self.weight = nn.Parameter(torch.randn(4, 3))

        def forward(self, x):
            return self.linear(x)[:, :3] + torch.addmm(self.bias, x, self.weight)

    model = MixedModel()
    x = torch.randn(5, 4)

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    expected_mac = 5 * 4 * 4 + 5 * 4 * 3
    assert report.primitive_counts["totals"]["mac"] == expected_mac
    assert report.counts_by_core_type["ann_fe"]["mac"] == expected_mac
    assert any(
        item["op_name"] == "aten.addmm.default" and item["core_type"] == "ann_fe"
        for item in report.mapping_summary
    )


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


def test_neuromc_exact_ann_forward_input_is_supported():
    model = nn.Linear(8, 4, bias=False)
    x = torch.randn(3, 8)
    report = op_counter.estimate_neuromc_runtime_energy(model, x)
    assert report.energy_total_pj > 0.0
    assert report.energy_by_core_type["ann_fe"] > 0.0


def test_neuromc_exact_memory_config_api():
    cfg = op_counter.MemoryHierarchyConfig.neuromc_like_v1()
    cfg.validate()
    assert cfg.technology_nm == 32
    assert "dram" in cfg.memory_instances
    assert hasattr(op_counter.neuromc, "MemoryResidencySimulator")
    assert hasattr(op_counter.neuromc, "NeuroMCMemoryResidencyCounter")


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

    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)

    fragment = profiler._optimizer_fragment("optimizer")
    counts = profiler._extra_counts(fragment)
    bits, _ = profiler._extra_memory_for_fragment(fragment)

    assert counts["add"] == 48
    assert counts["mul"] == 132
    assert counts["sqrt"] == 12
    assert bits["sram"]["rh2l"] == (14 * 3 + 7 * 6) * 32
    assert bits["sram"]["wl2h"] == (14 * 3 + 7 * 6) * 32


def test_neuromc_exact_sgd_optimizer_counts_plain_step():
    model = nn.Linear(2, 3, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    profiler = op_counter.NeuroMCEnergyProfiler()
    profiler.bind_optimizer(optimizer)

    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)

    fragments = profiler._optimizer_fragments("optimizer")
    add, mul, sqrt, rh2l, wl2h = _optimizer_fragment_totals(profiler, fragments)

    total_params = model.weight.numel() + model.bias.numel()
    assert add == total_params
    assert mul == total_params
    assert sqrt == 0
    assert rh2l == total_params * 2 * 32
    assert wl2h == total_params * 32


def test_neuromc_exact_sgd_optimizer_counts_weight_decay_only_step():
    model = nn.Linear(2, 3, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)
    profiler = op_counter.NeuroMCEnergyProfiler()
    profiler.bind_optimizer(optimizer)

    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)

    fragments = profiler._optimizer_fragments("optimizer")
    add, mul, sqrt, rh2l, wl2h = _optimizer_fragment_totals(profiler, fragments)

    total_params = model.weight.numel() + model.bias.numel()
    assert add == total_params * 2
    assert mul == total_params * 2
    assert sqrt == 0
    assert rh2l == total_params * 2 * 32
    assert wl2h == total_params * 32


def test_neuromc_exact_sgd_optimizer_counts_momentum_first_step():
    model = nn.Linear(2, 3, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    profiler = op_counter.NeuroMCEnergyProfiler()
    profiler.bind_optimizer(optimizer)

    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)

    fragments = profiler._optimizer_fragments("optimizer")
    add, mul, sqrt, rh2l, wl2h = _optimizer_fragment_totals(profiler, fragments)

    total_params = model.weight.numel() + model.bias.numel()
    assert add == total_params
    assert mul == total_params
    assert sqrt == 0
    assert rh2l == total_params * 2 * 32
    assert wl2h == total_params * 2 * 32
    assert any(
        fragment.op_name == "sgd"
        and fragment.optimizer_has_momentum
        and not fragment.optimizer_has_momentum_buffer
        for fragment in fragments
    )


def test_neuromc_exact_sgd_optimizer_counts_momentum_with_existing_buffer():
    model = nn.Linear(2, 3, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    profiler = op_counter.NeuroMCEnergyProfiler()
    profiler.bind_optimizer(optimizer)

    for param in model.parameters():
        optimizer.state[param]["momentum_buffer"] = torch.ones_like(param)
        param.grad = torch.ones_like(param)

    fragments = profiler._optimizer_fragments("optimizer")
    add, mul, sqrt, rh2l, wl2h = _optimizer_fragment_totals(profiler, fragments)

    total_params = model.weight.numel() + model.bias.numel()
    assert add == total_params * 2
    assert mul == total_params * 2
    assert sqrt == 0
    assert rh2l == total_params * 3 * 32
    assert wl2h == total_params * 2 * 32
    assert all(
        fragment.optimizer_has_momentum_buffer
        for fragment in fragments
        if fragment.op_name == "sgd"
    )


def test_neuromc_exact_sgd_optimizer_counts_momentum_and_weight_decay():
    model = nn.Linear(2, 3, bias=True)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01
    )
    profiler = op_counter.NeuroMCEnergyProfiler()
    profiler.bind_optimizer(optimizer)

    for param in model.parameters():
        optimizer.state[param]["momentum_buffer"] = torch.ones_like(param)
        param.grad = torch.ones_like(param)

    fragments = profiler._optimizer_fragments("optimizer")
    add, mul, sqrt, rh2l, wl2h = _optimizer_fragment_totals(profiler, fragments)

    total_params = model.weight.numel() + model.bias.numel()
    assert add == total_params * 3
    assert mul == total_params * 3
    assert sqrt == 0
    assert rh2l == total_params * 3 * 32
    assert wl2h == total_params * 2 * 32


def test_neuromc_exact_sgd_optimizer_counts_mixed_param_groups():
    model = nn.Linear(2, 3, bias=True)
    optimizer = torch.optim.SGD(
        [
            {"params": [model.bias], "lr": 0.1},
            {
                "params": [model.weight],
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 0.01,
            },
        ]
    )
    profiler = op_counter.NeuroMCEnergyProfiler()
    profiler.bind_optimizer(optimizer)

    model.bias.grad = torch.ones_like(model.bias)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.state[model.weight]["momentum_buffer"] = torch.ones_like(model.weight)

    fragments = profiler._optimizer_fragments("optimizer")
    add, mul, sqrt, rh2l, wl2h = _optimizer_fragment_totals(profiler, fragments)

    bias_numel = model.bias.numel()
    weight_numel = model.weight.numel()
    assert add == bias_numel + weight_numel * 3
    assert mul == bias_numel + weight_numel * 3
    assert sqrt == 0
    assert rh2l == (bias_numel * 2 + weight_numel * 3) * 32
    assert wl2h == (bias_numel + weight_numel * 2) * 32
    assert len(fragments) == 2


def test_neuromc_memory_residency_legacy_memory_config_alias():
    cfg = op_counter.MemoryHierarchyConfig.neuromc_like_v1()
    with pytest.deprecated_call():
        counter = NeuroMCMemoryResidencyCounter(memory_config=cfg)
    assert isinstance(counter, MemoryResidencyCounter)


def test_neuromc_memory_residency_legacy_memory_config_positional_alias():
    cfg = op_counter.MemoryHierarchyConfig.neuromc_like_v1()
    with pytest.deprecated_call():
        counter = NeuroMCMemoryResidencyCounter(cfg)
    assert isinstance(counter, MemoryResidencyCounter)


def test_neuromc_like_v1_legacy_memory_model_is_ignored():
    with pytest.deprecated_call():
        cfg = op_counter.MemoryHierarchyConfig.neuromc_like_v1(memory_model="weighted")
    cfg.validate()


def test_neuromc_exact_optimizer_skips_params_without_grad():
    model = nn.Linear(2, 3, bias=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    profiler = op_counter.NeuroMCEnergyProfiler()
    profiler.bind_optimizer(optimizer)

    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = None

    fragment = profiler._optimizer_fragment("optimizer")
    counts = profiler._extra_counts(fragment)

    assert fragment.input_numel == 0
    assert fragment.weight_numel == model.weight.numel()
    assert counts["add"] == 4 * model.weight.numel()
    assert counts["mul"] == 11 * model.weight.numel()
    assert counts["sqrt"] == model.weight.numel()


def test_neuromc_exact_multi_step_trace_batchnorm_is_rejected():
    class FunctionalBN(nn.Module):
        def forward(self, x):
            return torch.nn.functional.batch_norm(
                x,
                running_mean=None,
                running_var=None,
                weight=None,
                bias=None,
                training=True,
            )

    model = FunctionalBN()
    x = torch.randn(3, 2, 4, 5, 5)

    with pytest.raises(ValueError, match="multi-step or 3D BatchNorm trace fallback"):
        op_counter.estimate_neuromc_runtime_energy(model, x)


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
