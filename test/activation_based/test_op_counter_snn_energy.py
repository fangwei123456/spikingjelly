import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, op_counter


def test_lemaire_energy_report_uses_single_authoritative_total():
    model = nn.Sequential(nn.Linear(8, 8, bias=False), neuron.IFNode())
    x = torch.rand(4, 8)

    report = op_counter.estimate_lemaire_energy(model, x)

    assert report.total_pj == pytest.approx(
        report.breakdown_pj["ops_pj"]
        + report.breakdown_pj["addressing_pj"]
        + report.breakdown_pj["memory_pj"]
    )
    assert report.breakdown_pj["memory_pj"] == pytest.approx(
        report.breakdown_pj["inout_pj"]
        + report.breakdown_pj["params_pj"]
        + report.breakdown_pj["potential_pj"]
    )
    assert report.breakdown_pj["addressing_pj"] >= 0.0
    assert report.breakdown_pj["inout_pj"] > 0.0


def test_lemaire_energy_profiler_bind_model_rejects_non_torch_backend_when_strict():
    model = neuron.IFNode()
    model._backend = "triton"
    profiler = op_counter.LemaireEnergyProfiler(
        config=op_counter.LemaireEnergyConfig(strict=True)
    )

    with pytest.raises(ValueError, match="only supports torch backend"):
        profiler.bind_model(model)


def test_lemaire_energy_profiler_bind_model_warns_non_torch_backend_when_not_strict():
    model = neuron.IFNode()
    model._backend = "triton"
    profiler = op_counter.LemaireEnergyProfiler(
        config=op_counter.LemaireEnergyConfig(strict=False)
    )

    with pytest.warns(RuntimeWarning, match="only supports torch backend"):
        profiler.bind_model(model)


def test_lemaire_energy_conv_inference_report_has_memory_and_addressing():
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(),
    )
    x = torch.rand(2, 3, 8, 8)

    report = op_counter.estimate_lemaire_energy(model, x)

    assert report.breakdown_pj["params_pj"] > 0.0
    assert report.breakdown_pj["inout_pj"] > 0.0
    assert report.counts["acc_addr"] > 0


def test_lemaire_energy_sparse_linear_memory_is_lower_than_dense():
    model = nn.Sequential(nn.Linear(8, 8, bias=False), neuron.IFNode())
    dense_x = torch.rand(4, 8)
    sparse_x = torch.zeros(4, 8)
    sparse_x[0, 0] = 0.25
    sparse_x[1, 3] = 0.5

    dense_report = op_counter.estimate_lemaire_energy(model, dense_x)
    sparse_report = op_counter.estimate_lemaire_energy(model, sparse_x)

    assert sparse_report.breakdown_pj["inout_pj"] < dense_report.breakdown_pj["inout_pj"]
    assert sparse_report.breakdown_pj["params_pj"] < dense_report.breakdown_pj["params_pj"]
    assert sparse_report.total_pj < dense_report.total_pj


def test_lemaire_energy_non_binary_sparse_linear_memory_is_lower_than_dense():
    model = nn.Sequential(nn.Linear(8, 8, bias=False), neuron.IFNode())
    dense_x = torch.full((4, 8), 0.25)
    sparse_x = torch.zeros(4, 8)
    sparse_x[:, :2] = 0.25

    dense_report = op_counter.estimate_lemaire_energy(model, dense_x)
    sparse_report = op_counter.estimate_lemaire_energy(model, sparse_x)

    assert sparse_report.breakdown_pj["inout_pj"] < dense_report.breakdown_pj["inout_pj"]


def test_lemaire_energy_sparse_zero_ratio_below_threshold_stays_dense():
    model = nn.Sequential(nn.Linear(8, 8, bias=False), neuron.IFNode())
    dense_x = torch.full((4, 8), 0.25)
    near_dense_x = dense_x.clone()
    near_dense_x[:, 0] = 0.0

    dense_report = op_counter.estimate_lemaire_energy(model, dense_x)
    near_dense_report = op_counter.estimate_lemaire_energy(model, near_dense_x)

    assert near_dense_report.breakdown_pj["inout_pj"] == pytest.approx(
        dense_report.breakdown_pj["inout_pj"]
    )


def test_lemaire_energy_sparse_conv_memory_is_lower_than_dense():
    model = nn.Sequential(
        nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(),
    )
    dense_x = torch.rand(1, 2, 6, 6)
    sparse_x = torch.zeros(1, 2, 6, 6)
    sparse_x[:, 0, 1, 1] = 0.25

    dense_report = op_counter.estimate_lemaire_energy(model, dense_x)
    sparse_report = op_counter.estimate_lemaire_energy(model, sparse_x)

    assert sparse_report.breakdown_pj["inout_pj"] < dense_report.breakdown_pj["inout_pj"]
    assert sparse_report.breakdown_pj["params_pj"] < dense_report.breakdown_pj["params_pj"]


def test_lemaire_energy_conv_transpose_sparse_input_falls_back_with_warning():
    model = nn.Sequential(nn.ConvTranspose2d(2, 4, kernel_size=3, bias=False))
    x = torch.zeros(1, 2, 5, 5)
    x[:, 0, 1, 1] = 0.25

    report = op_counter.estimate_lemaire_energy(model, x)

    assert any("ConvTranspose2d" in message for message in report.warnings)


def test_lemaire_energy_manual_profiler_usage_defaults_to_forward_only():
    model = nn.Sequential(nn.Linear(8, 8, bias=False), neuron.IFNode())
    x = torch.rand(4, 8)
    profiler = op_counter.LemaireEnergyProfiler()
    profiler.bind_model(model)

    with profiler:
        _ = model(x)

    report = profiler.get_report()
    assert report.breakdown_pj["inout_pj"] > 0.0
    assert not hasattr(profiler, "stage")
    assert not hasattr(profiler, "suspend")


def test_lemaire_energy_cost_config_validates_memory_breakpoints():
    with pytest.raises(ValueError, match="exactly 4"):
        op_counter.LemaireEnergyCostConfig(memory_breakpoints=((0.0, 0.0),))

    with pytest.raises(ValueError, match="strictly increasing"):
        op_counter.LemaireEnergyCostConfig(
            memory_breakpoints=((0.0, 0.0), (1.0, 1.0), (1.0, 2.0), (2.0, 3.0))
        )


def test_lemaire_energy_cost_config_converts_access_costs_to_pj_per_byte():
    cost = op_counter.LemaireEnergyCostConfig()
    assert cost.memory_cost_pj(8.0 * 1024.0) == pytest.approx(2.5)
    assert cost.memory_cost_pj(32.0 * 1024.0) == pytest.approx(5.0)
    assert cost.memory_cost_pj(1024.0 * 1024.0) == pytest.approx(25.0)


def test_lemaire_addressing_counter_linear_counts_dense_and_binary():
    model = nn.Linear(8, 4, bias=False)
    counter = op_counter.LemaireAddressingCounter()
    dense_x = torch.rand(3, 8)
    spike_x = (torch.rand(3, 8) > 0.5).float()

    with op_counter.DispatchCounterMode([counter]):
        _ = model(dense_x)
    dense_counts = counter.get_metric_counts()["Global"]
    assert dense_counts["mac_addr"] == 0
    assert dense_counts["acc_addr"] == dense_x.numel() + 3 * 4

    counter = op_counter.LemaireAddressingCounter()
    with op_counter.DispatchCounterMode([counter]):
        _ = model(spike_x)
    spike_counts = counter.get_metric_counts()["Global"]
    assert spike_counts["mac_addr"] == 0
    assert spike_counts["acc_addr"] == int(spike_x.count_nonzero().item()) * model.out_features


def test_lemaire_addressing_counter_conv_counts_dense_binary_and_grouped():
    dense_counter = op_counter.LemaireAddressingCounter()
    dense_model = nn.Conv2d(2, 4, kernel_size=3, bias=False)
    dense_x = torch.rand(1, 2, 5, 5)
    with op_counter.DispatchCounterMode([dense_counter]):
        dense_out = dense_model(dense_x)
    dense_counts = dense_counter.get_metric_counts()["Global"]
    assert dense_counts["mac_addr"] == 0
    assert dense_counts["acc_addr"] == (
        dense_x.numel() + dense_out.numel() + dense_model.out_channels * 9
    )

    grouped_counter = op_counter.LemaireAddressingCounter()
    grouped_model = nn.Conv2d(4, 8, kernel_size=3, bias=False, groups=2)
    spike_x = torch.zeros(1, 4, 5, 5)
    spike_x[:, 0, 1, 1] = 1.0
    spike_x[:, 3, 2, 2] = 1.0
    with op_counter.DispatchCounterMode([grouped_counter]):
        _ = grouped_model(spike_x)
    grouped_counts = grouped_counter.get_metric_counts()["Global"]
    spike_num_in = int(spike_x.count_nonzero().item())
    assert grouped_counts["mac_addr"] == spike_num_in * 2
    assert grouped_counts["acc_addr"] == (
        spike_num_in * (grouped_model.out_channels // grouped_model.groups) * 9
    )


def test_lemaire_addressing_counter_only_counts_supported_modules():
    class MatmulWrapper(nn.Module):
        def forward(self, x, y):
            return torch.mm(x, y)

    model = MatmulWrapper()
    counter = op_counter.LemaireAddressingCounter()

    with op_counter.DispatchCounterMode([counter]):
        _ = model(torch.rand(3, 8), torch.rand(8, 4))

    assert counter.get_total() == 0


def test_lemaire_energy_linear_inout_uses_runtime_dtype_bytes():
    model = nn.Linear(8, 4, bias=False).half()
    x = torch.rand(3, 8, dtype=torch.float16)
    report = op_counter.estimate_lemaire_energy(model, x)

    assert report.counts["read_in_bytes"] == x.numel() * x.element_size()
    assert report.counts["write_out_bytes"] == (3 * 4) * x.element_size()
    assert report.counts["read_params_bytes"] == (
        model.weight.numel() * model.weight.element_size()
    )


def test_lemaire_energy_config_passes_extra_state_rules_to_counter():
    calls = {"count": 0}

    class RuleNode(neuron.BaseNode):
        def __init__(self):
            super().__init__(v_threshold=1.0, v_reset=None, step_mode="s")

        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + x

    def rule(module, func, args, kwargs, out, state_tensor_keys):
        del module, func, args, kwargs, out, state_tensor_keys
        calls["count"] += 1
        return {"state_reads": 1, "state_writes": 1}

    node = RuleNode()
    profiler = op_counter.LemaireEnergyProfiler(
        config=op_counter.LemaireEnergyConfig(extra_state_rules={RuleNode: rule})
    )
    profiler.bind_model(node)
    with profiler:
        _ = node(torch.rand(2, 4))

    assert calls["count"] > 0


def test_richer_counters_still_expose_scalar_and_structured_views():
    model = nn.Sequential(nn.Linear(8, 8, bias=False), neuron.IFNode())
    x = torch.rand(4, 8)
    memory = op_counter.AnalyticalMemoryCounter(extra_ignore_modules=[neuron.BaseNode])
    state = op_counter.NeuronStateCounter()

    with op_counter.DispatchCounterMode([memory, state]):
        _ = model(x)

    assert memory.get_total() > 0
    assert memory.get_metric_counts()["Global"]["memory_access_bytes"] > 0
    assert memory.get_extra_counts()["Global"]["memory_access_bytes"] > 0
    assert state.get_total() >= 0
    assert "Global" in state.get_metric_counts()
    assert "Global" in state.get_projection_counts()
    assert "Global" in state.get_extra_counts()


def test_old_analytical_energy_names_are_not_exported():
    assert not hasattr(op_counter, "estimate_analytical_energy")
    assert not hasattr(op_counter, "AnalyticalEnergyProfiler")
    assert not hasattr(op_counter, "AnalyticalEnergyConfig")
    assert not hasattr(op_counter, "AnalyticalEnergyCostConfig")
    assert not hasattr(op_counter, "AnalyticalEnergyReport")


def test_training_related_legacy_arguments_are_rejected():
    model = nn.Linear(8, 4)
    x = torch.rand(3, 8)
    target = torch.rand(3, 4)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(TypeError):
        op_counter.estimate_lemaire_energy(
            model,
            x,
            target=target,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
