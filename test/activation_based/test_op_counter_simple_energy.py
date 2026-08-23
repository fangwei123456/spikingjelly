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
    assert report.model_info.model_id == "simple_horowitz_step_composite_v1"
    assert report.model_info.fidelity == "spikingjelly-defined"
    assert report.config.cost_config.e_mac_pj == 4.6


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


def test_simple_energy_costs_must_be_finite_and_nonnegative():
    with pytest.raises(ValueError, match="finite and nonnegative"):
        op_counter.SimpleEnergyCostConfig(e_mac_pj=-1.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        op_counter.SimpleEnergyCostConfig(e_memory_pj_per_byte=float("nan"))


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


def test_simple_energy_supports_keyword_module_inputs():
    model = nn.Linear(4, 3, bias=False)
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    report = op_counter.estimate_simple_energy(model, {"input": x})

    assert report.counts["weight_read_bytes"] == 6 * 4


def test_simple_energy_profiler_rejects_rebinding_while_active():
    profiler = op_counter.SimpleEnergyProfiler()
    profiler.bind_model(nn.Linear(4, 3))

    with profiler, pytest.raises(RuntimeError, match="while profiling"):
        profiler.bind_model(nn.Linear(4, 3))


def test_neuromorphic_memory_counter_uses_module_counter_mode():
    counter = op_counter.NeuromorphicMemoryAccessCounter()
    assert isinstance(counter, op_counter.ModuleCounter)
    assert not hasattr(counter, "bind_model")


def test_neuromorphic_memory_counter_counts_dense_and_spike_weight_reads():
    model = nn.Linear(4, 3, bias=True)
    counter = op_counter.NeuromorphicMemoryAccessCounter()

    with op_counter.ModuleCounterMode([counter], model=model):
        _ = model(torch.full((2, 4), 0.5))
    dense = counter.get_counts()["Global"]
    assert dense["weight_read_bytes"] == 24 * 4
    assert dense["bias_read_bytes"] == 6 * 4

    counter.reset()
    with op_counter.ModuleCounterMode([counter], model=model):
        _ = model(torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
    spike = counter.get_counts()["Global"]
    assert spike["weight_read_bytes"] == 6 * 4
    assert spike["bias_read_bytes"] == 3 * 4


def test_neuromorphic_memory_counter_uses_actual_conv_spike_fanout():
    model = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
    x = torch.zeros(1, 1, 3, 3)
    x[0, 0, 0, 0] = 1.0
    counter = op_counter.NeuromorphicMemoryAccessCounter()

    with op_counter.ModuleCounterMode([counter], model=model):
        _ = model(x)

    # A corner spike reaches 2 x 2 positions in each of two output channels.
    assert counter.get_counts()["Global"]["weight_read_bytes"] == 8 * 4


def test_neuromorphic_memory_counter_preserves_large_integer_fanout():
    size = 192
    model = nn.Conv2d(64, 16, kernel_size=3, padding=1, bias=False)
    x = torch.ones(1, 64, size, size)
    x[0, 0, size // 2, size // 2] = 0
    counter = op_counter.NeuromorphicMemoryAccessCounter()

    with op_counter.ModuleCounterMode([counter], model=model):
        model(x)

    expected_weight_uses = (64 * (3 * size - 2) ** 2 - 9) * 16
    assert counter.get_counts()["Global"]["weight_read_bytes"] == (
        expected_weight_uses * model.weight.element_size()
    )


@pytest.mark.parametrize(("padding", "weight_uses"), (("same", 4), ("valid", 1)))
def test_neuromorphic_memory_counter_supports_string_conv_padding(padding, weight_uses):
    model = nn.Conv2d(1, 1, kernel_size=3, padding=padding, bias=False)
    x = torch.zeros(1, 1, 3, 3)
    x[0, 0, 0, 0] = 1.0
    counter = op_counter.NeuromorphicMemoryAccessCounter()

    with op_counter.ModuleCounterMode([counter], model=model):
        _ = model(x)

    assert counter.get_counts()["Global"]["weight_read_bytes"] == weight_uses * 4


def test_neuromorphic_memory_probe_does_not_dispatch_extra_padding():
    model = nn.Conv2d(
        1, 1, kernel_size=3, padding=1, padding_mode="reflect", bias=False
    )
    memory_counter = op_counter.NeuromorphicMemoryAccessCounter()
    pad_counter = op_counter.BaseCounter()
    pad_counter.rules = {
        torch.ops.aten.reflection_pad2d.default: lambda args, kwargs, out: 1
    }

    with (
        op_counter.DispatchCounterMode([pad_counter]),
        op_counter.ModuleCounterMode([memory_counter], model=model),
    ):
        model(torch.ones(1, 1, 3, 3))

    assert pad_counter.get_total() == 1


def test_neuromorphic_memory_counter_empty_reads_do_not_create_scopes():
    counter = op_counter.NeuromorphicMemoryAccessCounter()

    assert counter.get_total() == 0
    assert counter.get_counts() == {}


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
    with op_counter.ModuleCounterMode([single_counter], model=single):
        _ = single(torch.rand(2, 3))

    multi = neuron.IFNode(step_mode="m")
    multi_counter = op_counter.NeuromorphicMemoryAccessCounter()
    with op_counter.ModuleCounterMode([multi_counter], model=multi):
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

    with op_counter.ModuleCounterMode([counter], model=model):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_simple_and_lemaire_match_cpu_counts():
    cpu_model = nn.Linear(8, 4, bias=True).eval()
    cuda_model = nn.Linear(8, 4, bias=True).cuda().eval()
    cuda_model.load_state_dict(cpu_model.state_dict())
    x = (torch.rand(3, 8) > 0.6).float()

    simple_cpu = op_counter.estimate_simple_energy(cpu_model, x)
    simple_cuda = op_counter.estimate_simple_energy(cuda_model, x.cuda())
    lemaire_cpu = op_counter.estimate_lemaire_energy(cpu_model, x)
    lemaire_cuda = op_counter.estimate_lemaire_energy(cuda_model, x.cuda())

    assert simple_cuda.counts == simple_cpu.counts
    assert simple_cuda.energy_total_pj == pytest.approx(simple_cpu.energy_total_pj)
    assert lemaire_cuda.counts == lemaire_cpu.counts
    assert lemaire_cuda.total_pj == pytest.approx(lemaire_cpu.total_pj)
