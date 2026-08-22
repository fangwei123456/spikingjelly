import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import layer, neuron, op_counter


def test_function_counter_mode_accepts_function_rules():
    class AddCounter(op_counter.BaseCounter):
        def __init__(self):
            super().__init__()
            self.rules = {torch.add: lambda args, kwargs, out: 1}

    counter = AddCounter()
    with op_counter.FunctionCounterMode([counter]):
        torch.add(torch.ones(2), torch.ones(2))

    assert counter.get_total() == 1


def test_simple_energy_uses_mac_and_ac_as_authoritative_total():
    class AddModel(nn.Module):
        def forward(self, x, y):
            return x + y

    model = AddModel()
    x = torch.ones(2, 3)
    y = torch.ones(2, 3)

    report = op_counter.estimate_simple_energy(model, (x, y))

    assert report.counts["mac"] == 0
    assert report.counts["ac"] == 12
    assert report.counts["memory_access_bytes"] == 0
    assert report.energy_mac_pj == pytest.approx(0.0)
    assert report.energy_ac_pj == pytest.approx(12 * 0.9)
    assert report.energy_compute_pj == pytest.approx(12 * 0.9)
    assert report.energy_memory_pj == pytest.approx(0.0)
    assert report.energy_total_pj == pytest.approx(
        report.energy_mac_pj + report.energy_ac_pj + report.energy_memory_pj
    )


def test_simple_energy_spike_linear_counts_synop_as_auxiliary_only():
    model = nn.Linear(4, 3, bias=False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    report = op_counter.estimate_simple_energy(model, x)

    assert report.counts["synop"] == 6
    assert report.counts["ac"] == 6
    assert report.counts["mac"] == 0
    assert report.counts["weight_read_bytes"] == 24
    assert report.counts["memory_access_bytes"] == 24
    assert report.energy_total_pj == pytest.approx(6 * 0.9 + 24 * 24.96)


def test_simple_energy_dense_linear_counts_mac():
    model = nn.Linear(4, 3, bias=False)
    x = torch.full((2, 4), 0.5)

    report = op_counter.estimate_simple_energy(model, x)

    assert report.counts["mac"] == 24
    assert report.counts["ac"] == 0
    assert report.counts["weight_read_bytes"] == 96
    assert report.counts["memory_access_bytes"] == 96
    assert report.energy_total_pj == pytest.approx(24 * 4.6 + 96 * 24.96)


def test_simple_energy_profiler_matches_convenience_function():
    model = nn.Linear(4, 2, bias=False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    profiler = op_counter.SimpleEnergyProfiler()
    profiler.bind_model(model)
    with profiler:
        _ = model(x)

    report_ctx = profiler.get_report()
    report_fn = op_counter.estimate_simple_energy(model, x)

    assert report_ctx.energy_total_pj == pytest.approx(report_fn.energy_total_pj)
    assert report_ctx.counts == report_fn.counts


def test_simple_energy_custom_cost_config_is_applied():
    model = nn.Linear(4, 2, bias=False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    cfg = op_counter.SimpleEnergyConfig(
        cost_config=op_counter.SimpleEnergyCostConfig(
            e_mac_pj=10.0,
            e_ac_pj=2.0,
            e_memory_pj_per_byte=0.0,
        )
    )

    report = op_counter.estimate_simple_energy(model, x, config=cfg)

    assert report.counts["ac"] == 4
    assert report.energy_total_pj == pytest.approx(8.0)


def test_simple_energy_custom_memory_cost_is_applied_to_runtime_bytes():
    model = nn.Linear(4, 2, bias=False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    cfg = op_counter.SimpleEnergyConfig(
        cost_config=op_counter.SimpleEnergyCostConfig(
            e_mac_pj=0.0,
            e_ac_pj=0.0,
            e_memory_pj_per_byte=2.0,
        )
    )

    report = op_counter.estimate_simple_energy(model, x, config=cfg)

    assert report.counts["memory_access_bytes"] == 16
    assert report.energy_memory_pj == pytest.approx(32.0)
    assert report.energy_total_pj == pytest.approx(32.0)


def test_simple_energy_does_not_treat_bmm_operands_as_neuromorphic_memory():
    class BMMModel(nn.Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    x = torch.full((2, 3, 4), 0.5)
    y = torch.full((2, 4, 5), 0.5)
    cfg = op_counter.SimpleEnergyConfig(
        cost_config=op_counter.SimpleEnergyCostConfig(
            e_mac_pj=1.0,
            e_ac_pj=0.0,
            e_memory_pj_per_byte=1.0,
        )
    )

    report = op_counter.estimate_simple_energy(BMMModel(), (x, y), config=cfg)

    assert report.counts["mac"] == 120
    assert report.counts["ac"] == 0
    assert report.counts["memory_access_bytes"] == 0
    assert report.energy_total_pj == pytest.approx(120.0)


def test_simple_energy_cost_config_presets_match_horowitz_reference_table():
    fp32 = op_counter.SimpleEnergyCostConfig.fp32()
    fp16 = op_counter.SimpleEnergyCostConfig.fp16()
    int8 = op_counter.SimpleEnergyCostConfig.int8()

    assert fp32.e_mac_pj == pytest.approx(4.6)
    assert fp32.e_ac_pj == pytest.approx(0.9)
    assert fp32.e_memory_pj_per_byte == pytest.approx(24.96)
    assert fp16.e_mac_pj == pytest.approx(1.5)
    assert fp16.e_ac_pj == pytest.approx(0.4)
    assert int8.e_mac_pj == pytest.approx(0.23)
    assert int8.e_ac_pj == pytest.approx(0.03)
    assert int8.e_memory_pj_per_byte == pytest.approx(24.96)


def test_simple_energy_default_cost_config_matches_fp32_preset():
    cfg = op_counter.SimpleEnergyConfig()
    fp32 = op_counter.SimpleEnergyCostConfig.fp32()

    assert cfg.cost_config.e_mac_pj == pytest.approx(fp32.e_mac_pj)
    assert cfg.cost_config.e_ac_pj == pytest.approx(fp32.e_ac_pj)
    assert cfg.cost_config.e_memory_pj_per_byte == pytest.approx(
        fp32.e_memory_pj_per_byte
    )


def test_simple_energy_fp16_preset_changes_only_comparison_regime():
    model = nn.Linear(4, 2, bias=False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    cfg = op_counter.SimpleEnergyConfig(
        cost_config=op_counter.SimpleEnergyCostConfig.fp16()
    )

    report = op_counter.estimate_simple_energy(model, x, config=cfg)

    assert report.counts["ac"] == 4
    assert report.counts["mac"] == 0
    assert report.counts["memory_access_bytes"] == 16
    assert report.energy_total_pj == pytest.approx(4 * 0.4 + 16 * 24.96)


def test_simple_energy_warns_when_no_supported_ops_are_profiled():
    model = nn.ReLU()
    x = torch.ones(2, 3)

    report = op_counter.estimate_simple_energy(model, x)

    assert report.energy_total_pj == pytest.approx(0.0)
    assert any(
        "did not match any supported operators" in msg for msg in report.warnings
    )


def test_simple_energy_strict_raises_when_no_supported_ops_are_profiled():
    model = nn.ReLU()
    x = torch.ones(2, 3)
    cfg = op_counter.SimpleEnergyConfig(strict=True)

    with pytest.raises(RuntimeError, match="did not match any supported operators"):
        op_counter.estimate_simple_energy(model, x, config=cfg)


def test_simple_energy_strict_allows_zero_work_when_supported_op_matches():
    model = nn.Linear(4, 3, bias=False)
    x = torch.empty(0, 4)
    cfg = op_counter.SimpleEnergyConfig(strict=True)

    report = op_counter.estimate_simple_energy(model, x, config=cfg)

    assert report.counts["mac"] == 0
    assert report.counts["ac"] == 0
    assert report.counts["synop"] == 0
    assert report.counts["memory_access_bytes"] == 0
    assert report.energy_memory_pj == pytest.approx(0.0)
    assert report.warnings == []


def test_simple_energy_supports_dict_inputs_for_keyword_only_models():
    class KeywordOnlyAdd(nn.Module):
        def forward(self, *, x, y):
            return x + y

    model = KeywordOnlyAdd()
    inputs = {
        "x": torch.ones(2, 3),
        "y": torch.ones(2, 3),
    }

    report = op_counter.estimate_simple_energy(model, inputs)

    assert report.counts["mac"] == 0
    assert report.counts["ac"] == 12


def test_neuromorphic_memory_counter_requires_bound_model():
    counter = op_counter.NeuromorphicMemoryAccessCounter()

    with pytest.raises(RuntimeError, match="bind_model"):
        with counter:
            pass


def test_neuromorphic_memory_counter_counts_dense_and_spike_weight_reads():
    model = nn.Linear(4, 3, bias=True)
    counter = op_counter.NeuromorphicMemoryAccessCounter()
    counter.bind_model(model)

    with counter:
        _ = model(torch.full((2, 4), 0.5))
    dense = counter.get_counts()["Global"]
    assert dense["weight_read_bytes"] == 24 * 4
    assert dense["bias_read_bytes"] == 6 * 4

    with counter:
        _ = model(torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
    spike = counter.get_counts()["Global"]
    assert spike["weight_read_bytes"] == 6 * 4
    assert spike["bias_read_bytes"] == 3 * 4


def test_neuromorphic_memory_counter_uses_actual_conv_spike_fanout():
    model = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
    x = torch.zeros(1, 1, 3, 3)
    x[0, 0, 0, 0] = 1.0
    counter = op_counter.NeuromorphicMemoryAccessCounter()
    counter.bind_model(model)

    with counter:
        _ = model(x)

    # A corner spike reaches 2 x 2 positions in each of two output channels.
    assert counter.get_counts()["Global"]["weight_read_bytes"] == 8 * 4


def test_simple_energy_ignores_module_subtrees():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3, bias=False)

        def forward(self, x):
            return self.linear(x)

    report = op_counter.estimate_simple_energy(
        Block(),
        torch.ones(1, 4),
        config=op_counter.SimpleEnergyConfig(extra_ignore_modules=[Block]),
    )

    assert report.counts["mac"] == 0
    assert report.counts["memory_access_bytes"] == 0


def test_neuromorphic_memory_counter_counts_state_once_per_timestep():
    single = neuron.IFNode(step_mode="s")
    single_counter = op_counter.NeuromorphicMemoryAccessCounter()
    single_counter.bind_model(single)
    with single_counter:
        _ = single(torch.rand(2, 3))

    multi = neuron.IFNode(step_mode="m")
    multi_counter = op_counter.NeuromorphicMemoryAccessCounter()
    multi_counter.bind_model(multi)
    with multi_counter:
        _ = multi(torch.rand(4, 2, 3))

    single_counts = single_counter.get_counts()["Global"]
    multi_counts = multi_counter.get_counts()["Global"]
    assert single_counts["neuron_state_read_bytes"] == 2 * 3 * 4
    assert single_counts["neuron_state_write_bytes"] == 2 * 3 * 4
    assert multi_counts["neuron_state_read_bytes"] == 4 * 2 * 3 * 4
    assert multi_counts["neuron_state_write_bytes"] == 4 * 2 * 3 * 4


def test_neuromorphic_memory_counter_supports_multistep_conv_and_runtime_dtype():
    model = layer.Conv2d(1, 2, kernel_size=1, bias=False, step_mode="m").half()
    x = torch.tensor([[[[[1.0]]]], [[[[0.0]]]], [[[[1.0]]]]], dtype=torch.float16)
    counter = op_counter.NeuromorphicMemoryAccessCounter()
    counter.bind_model(model)

    with counter:
        _ = model(x)

    assert counter.get_counts()["Global"]["weight_read_bytes"] == 4 * 2


def test_simple_energy_breaks_memory_energy_into_parameter_and_state_parts():
    model = nn.Sequential(nn.Linear(3, 3, bias=False), neuron.IFNode())
    x = torch.tensor([[1.0, 0.0, 1.0]])

    report = op_counter.estimate_simple_energy(model, x)

    assert report.counts["weight_read_bytes"] == 6 * 4
    assert report.counts["neuron_state_read_bytes"] == 3 * 4
    assert report.counts["neuron_state_write_bytes"] == 3 * 4
    assert report.breakdown_pj["parameter_memory_pj"] == pytest.approx(24 * 24.96)
    assert report.breakdown_pj["neuron_state_memory_pj"] == pytest.approx(24 * 24.96)
