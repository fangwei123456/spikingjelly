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
    assert report.model_info.model_id == "lemaire_2022_runtime_v1"
    assert report.model_info.fidelity == "paper"
    assert report.config.snn_fifo_capacity_elements == 1000


def test_lemaire_energy_charges_spike_synops_once_through_ac():
    model = nn.Linear(4, 3, bias=False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    cost = op_counter.LemaireEnergyCostConfig(
        e_add_pj=1.0,
        e_mul_pj=0.0,
        memory_breakpoints=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
    )
    config = op_counter.LemaireEnergyConfig(cost_config=cost)

    report = op_counter.estimate_lemaire_energy(model, x, config=config)

    assert report.counts["synop"] == 6
    assert report.counts["ac"] == 6
    assert report.breakdown_pj["ops_pj"] == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("node", "expected_mac"), ((neuron.IFNode(), 0), (neuron.LIFNode(), 3))
)
def test_lemaire_runtime_counts_follow_paper_if_lif_buckets(node, expected_mac):
    linear = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        linear.weight.fill_(0.1)
    model = nn.Sequential(linear, node).eval()
    cost = op_counter.LemaireEnergyCostConfig(
        e_add_pj=1.0,
        e_mul_pj=1.0,
        memory_breakpoints=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
    )

    report = op_counter.estimate_lemaire_energy(
        model,
        torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        config=op_counter.LemaireEnergyConfig(cost_config=cost),
    )

    assert report.counts["ac"] == 6
    assert report.counts["mac"] == expected_mac
    assert report.breakdown_pj["ops_pj"] == pytest.approx(6 + expected_mac * 2)


def test_lemaire_energy_profiler_bind_model_rejects_non_torch_backend_when_strict():
    model = neuron.IFNode()
    model._backend = "triton"
    profiler = op_counter.LemaireEnergyProfiler(
        config=op_counter.LemaireEnergyConfig(strict=True)
    )

    with pytest.raises(ValueError, match="only supports torch backend"):
        profiler.bind_model(model)


def test_lemaire_energy_strict_profiles_supported_linear():
    model = nn.Linear(4, 3, bias=False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    report = op_counter.estimate_lemaire_energy(
        model, x, config=op_counter.LemaireEnergyConfig(strict=True)
    )

    assert report.counts["synop"] == 6
    assert report.counts["read_params_bytes"] == 6 * 4


def test_lemaire_energy_supports_keyword_module_inputs():
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    linear_report = op_counter.estimate_lemaire_energy(
        nn.Linear(4, 3, bias=False), {"input": x}
    )
    neuron_report = op_counter.estimate_lemaire_energy(neuron.IFNode(), {"x": x})

    assert linear_report.counts["synop"] == 6
    assert neuron_report.counts["write_potential_bytes"] == x.numel() * 4


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


def test_lemaire_energy_binary_linear_uses_event_driven_access_formulas():
    model = nn.Sequential(nn.Linear(8, 8, bias=False), neuron.IFNode())
    dense_x = torch.full((4, 8), 0.25)
    spike_x = torch.zeros(4, 8)
    spike_x[0, 0] = 1.0
    spike_x[1, 3] = 1.0

    dense_report = op_counter.estimate_lemaire_energy(model, dense_x)
    spike_report = op_counter.estimate_lemaire_energy(model, spike_x)

    assert spike_report.counts["read_in_bytes"] == 2 * 4
    assert spike_report.counts["read_params_bytes"] == 2 * 8 * 4
    assert (
        spike_report.counts["read_params_bytes"]
        < dense_report.counts["read_params_bytes"]
    )


def test_lemaire_energy_non_binary_sparse_linear_stays_dense():
    model = nn.Linear(8, 8, bias=False)
    dense_x = torch.full((4, 8), 0.25)
    sparse_x = torch.zeros(4, 8)
    sparse_x[:, :2] = 0.25

    dense_report = op_counter.estimate_lemaire_energy(model, dense_x)
    sparse_report = op_counter.estimate_lemaire_energy(model, sparse_x)

    assert sparse_report.counts["read_in_bytes"] == dense_report.counts["read_in_bytes"]
    assert (
        sparse_report.counts["read_params_bytes"]
        == dense_report.counts["read_params_bytes"]
    )


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


def test_lemaire_energy_binary_conv_uses_paper_event_fanout():
    model = nn.Sequential(
        nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False),
        neuron.IFNode(),
    )
    dense_x = torch.rand(1, 2, 6, 6)
    sparse_x = torch.zeros(1, 2, 6, 6)
    sparse_x[:, 0, 1, 1] = 1.0

    dense_report = op_counter.estimate_lemaire_energy(model, dense_x)
    sparse_report = op_counter.estimate_lemaire_energy(model, sparse_x)

    expected_fanout = 4 * 3 * 3
    assert sparse_report.counts["read_in_bytes"] == 4
    assert sparse_report.counts["read_params_bytes"] == expected_fanout * 4
    assert (
        sparse_report.counts["read_params_bytes"]
        < dense_report.counts["read_params_bytes"]
    )


def test_lemaire_energy_binary_conv_uses_paper_fanout_at_borders():
    model = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    x = torch.zeros(1, 1, 3, 3)
    x[0, 0, 0, 0] = 1.0

    report = op_counter.estimate_lemaire_energy(model, x)

    assert report.counts["synop"] == 9
    assert report.counts["read_params_bytes"] == 9 * 4
    assert report.counts["read_potential_bytes"] == 9 * 4
    assert report.counts["write_potential_bytes"] == 9 * 4


def test_lemaire_energy_conv_transpose_sparse_input_falls_back_with_warning():
    model = nn.Sequential(nn.ConvTranspose2d(2, 4, kernel_size=3, bias=False))
    x = torch.zeros(1, 2, 5, 5)
    x[:, 0, 1, 1] = 0.25

    report = op_counter.estimate_lemaire_energy(
        model, x, config=op_counter.LemaireEnergyConfig(strict=False)
    )

    assert any("ConvTranspose2d" in message for message in report.warnings)


def test_lemaire_energy_conv_transpose_raises_when_strict():
    model = nn.ConvTranspose2d(2, 4, kernel_size=3, bias=False)
    config = op_counter.LemaireEnergyConfig(strict=True)

    with pytest.raises(ValueError, match="do not support ConvTranspose2d"):
        op_counter.estimate_lemaire_energy(
            model, torch.zeros(1, 2, 5, 5), config=config
        )


def test_lemaire_energy_manual_profiler_usage_defaults_to_forward_only():
    model = nn.Sequential(nn.Linear(8, 8, bias=False), neuron.IFNode())
    x = torch.rand(4, 8)
    profiler = op_counter.LemaireEnergyProfiler()
    profiler.bind_model(model)

    with profiler:
        _ = model(x)

    report = profiler.get_report()
    assert report.breakdown_pj["inout_pj"] > 0.0


def test_lemaire_energy_profiler_reuse_does_not_accumulate_counters():
    model = nn.Linear(8, 8, bias=False)
    x = torch.rand(4, 8)
    profiler = op_counter.LemaireEnergyProfiler()
    profiler.bind_model(model)

    with profiler:
        _ = model(x)
    first_report = profiler.get_report()

    with profiler:
        _ = model(x)
    second_report = profiler.get_report()

    assert second_report.total_pj == pytest.approx(first_report.total_pj)
    assert second_report.counts == first_report.counts


def test_lemaire_energy_profiler_rejects_rebinding_while_active():
    profiler = op_counter.LemaireEnergyProfiler()
    profiler.bind_model(nn.Linear(4, 3))

    with profiler, pytest.raises(RuntimeError, match="while profiling"):
        profiler.bind_model(nn.Linear(4, 3))


def test_lemaire_energy_cost_config_validates_memory_breakpoints():
    with pytest.raises(ValueError, match="exactly 4"):
        op_counter.LemaireEnergyCostConfig(memory_breakpoints=((0.0, 0.0),))

    with pytest.raises(ValueError, match="strictly increasing"):
        op_counter.LemaireEnergyCostConfig(
            memory_breakpoints=((0.0, 0.0), (1.0, 1.0), (1.0, 2.0), (2.0, 3.0))
        )

    with pytest.raises(ValueError, match="finite and nonnegative"):
        op_counter.LemaireEnergyCostConfig(e_add_pj=-1.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        op_counter.LemaireEnergyCostConfig(
            memory_breakpoints=(
                (0.0, 0.0),
                (1.0, 1.0),
                (2.0, float("nan")),
                (3.0, 3.0),
            )
        )


def test_lemaire_energy_cost_config_converts_access_costs_to_pj_per_byte():
    cost = op_counter.LemaireEnergyCostConfig()
    assert cost.memory_cost_pj(8.0 * 1024.0) == pytest.approx(2.5)
    assert cost.memory_cost_pj(32.0 * 1024.0) == pytest.approx(5.0)
    assert cost.memory_cost_pj(1024.0 * 1024.0) == pytest.approx(25.0)


def test_lemaire_energy_config_validates_fifo_capacity():
    with pytest.raises(ValueError, match="snn_fifo_capacity_elements"):
        op_counter.LemaireEnergyConfig(snn_fifo_capacity_elements=0)


def test_lemaire_energy_prices_parameter_sram_per_layer():
    model = nn.Sequential(
        nn.Linear(2, 2, bias=False),
        nn.Linear(2, 8, bias=False),
    )
    with torch.no_grad():
        for module in model:
            module.weight.fill_(0.5)
    cost = op_counter.LemaireEnergyCostConfig(
        memory_breakpoints=((0.0, 0.0), (16.0, 1.0), (32.0, 2.0), (64.0, 4.0))
    )
    report = op_counter.estimate_lemaire_energy(
        model,
        torch.full((1, 2), 0.5),
        config=op_counter.LemaireEnergyConfig(cost_config=cost),
    )

    # Layer capacities are 16 B and 64 B; their read traffic is also 16 B and 64 B.
    assert report.counts["read_params_bytes"] == 80
    assert report.breakdown_pj["params_pj"] == pytest.approx(16 * 1 + 64 * 4)
    assert report.breakdown_pj["params_pj"] != pytest.approx(80 * 4)


def test_lemaire_energy_linear_addressing_counts_dense_and_binary():
    model = nn.Linear(8, 4, bias=False)
    dense_x = torch.rand(3, 8)
    spike_x = (torch.rand(3, 8) > 0.5).float()

    dense_counts = op_counter.estimate_lemaire_energy(model, dense_x).counts
    assert dense_counts["mac_addr"] == 0
    assert dense_counts["acc_addr"] == dense_x.numel() + 3 * 4

    spike_counts = op_counter.estimate_lemaire_energy(model, spike_x).counts
    assert spike_counts["mac_addr"] == 0
    assert (
        spike_counts["acc_addr"]
        == int(spike_x.count_nonzero().item()) * model.out_features
    )


def test_lemaire_energy_conv_addressing_counts_dense_binary_and_grouped():
    dense_model = nn.Conv2d(2, 4, kernel_size=3, bias=False)
    dense_x = torch.rand(1, 2, 5, 5)
    dense_out = dense_model(dense_x)
    dense_counts = op_counter.estimate_lemaire_energy(dense_model, dense_x).counts
    assert dense_counts["mac_addr"] == 0
    assert dense_counts["acc_addr"] == (
        dense_x.numel() + dense_out.numel() + dense_model.out_channels * 9
    )

    grouped_model = nn.Conv2d(4, 8, kernel_size=3, bias=False, groups=2)
    spike_x = torch.zeros(1, 4, 5, 5)
    spike_x[:, 0, 1, 1] = 1.0
    spike_x[:, 3, 2, 2] = 1.0
    grouped_counts = op_counter.estimate_lemaire_energy(grouped_model, spike_x).counts
    spike_num_in = int(spike_x.count_nonzero().item())
    assert grouped_counts["synop"] == spike_num_in * 4 * 9
    assert grouped_counts["read_params_bytes"] == spike_num_in * 4 * 9 * 4
    assert grouped_counts["mac_addr"] == spike_num_in * 2
    assert grouped_counts["acc_addr"] == (
        spike_num_in * (grouped_model.out_channels // grouped_model.groups) * 9
    )


def test_lemaire_energy_ignores_unsupported_functional_matmul():
    class MatmulWrapper(nn.Module):
        def forward(self, x, y):
            return torch.mm(x, y)

    model = MatmulWrapper()
    report = op_counter.estimate_lemaire_energy(
        model,
        (torch.rand(3, 8), torch.rand(8, 4)),
        config=op_counter.LemaireEnergyConfig(strict=False),
    )

    assert report.counts["acc_addr"] == 0
    assert report.counts["mac_addr"] == 0
    assert any("MatmulWrapper" in message for message in report.warnings)


def test_lemaire_energy_strict_rejects_unknown_leaf_module():
    with pytest.raises(ValueError, match="ReLU"):
        op_counter.estimate_lemaire_energy(nn.ReLU(), torch.rand(2, 4))


def test_lemaire_energy_rejects_neurons_outside_paper_scope():
    model = nn.Sequential(
        nn.Linear(4, 3, bias=False),
        neuron.ParametricLIFNode(),
    ).eval()

    with pytest.raises(ValueError, match="ParametricLIFNode"):
        op_counter.estimate_lemaire_energy(model, torch.rand(2, 4))


def test_lemaire_energy_uses_paper_32bit_access_width():
    model = nn.Linear(8, 4, bias=False).half()
    x = torch.rand(3, 8, dtype=torch.float16)
    report = op_counter.estimate_lemaire_energy(model, x)

    assert report.counts["read_in_bytes"] == x.numel() * 4
    assert report.counts["write_out_bytes"] == (3 * 4) * 4
    assert report.counts["read_params_bytes"] == 3 * model.weight.numel() * 4


def test_lemaire_energy_counts_output_spikes_and_potential_accesses():
    linear = nn.Linear(4, 3, bias=True)
    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.zero_()
    model = nn.Sequential(linear, neuron.IFNode(v_threshold=0.5))
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    report = op_counter.estimate_lemaire_energy(model, x)

    event_fanout = 2 * 3
    base_potential_updates = 3
    assert report.counts["write_out_bytes"] == 3 * 4
    assert (
        report.counts["read_potential_bytes"]
        == (event_fanout + base_potential_updates) * 4
    )
    assert (
        report.counts["write_potential_bytes"]
        == (event_fanout + base_potential_updates) * 4
    )
    assert report.buffer_sizes_bytes["inout_buffer_bytes"] == 1000 * 4


def test_neuron_state_counter_still_exposes_scalar_and_structured_views():
    model = nn.Sequential(nn.Linear(8, 8, bias=False), neuron.IFNode())
    x = torch.rand(4, 8)
    state = op_counter.NeuronStateCounter()

    with op_counter.DispatchCounterMode([state]):
        _ = model(x)

    assert state.get_total() >= 0
    assert "Global" in state.get_metric_counts()
    assert "Global" in state.get_projection_counts()
    assert "Global" in state.get_extra_counts()


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
