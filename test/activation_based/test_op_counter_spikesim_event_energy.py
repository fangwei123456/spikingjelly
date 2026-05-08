import math

import pytest
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, op_counter
from spikingjelly.activation_based.op_counter.spikesim.formulas import (
    compute_spikesim_event_energy_breakdown,
)


def _dense_stage_energy(
    *,
    in_channels: int,
    out_channels: int,
    out_shape: tuple[int, int, int, int],
    kernel_size: tuple[int, int],
    config: op_counter.SpikeSimEnergyConfig,
) -> float:
    num_sites = out_shape[0] * out_shape[2] * out_shape[3]
    p_i = math.ceil(in_channels / config.xbar_size)
    q_i = math.ceil(out_channels / config.xbar_size)
    k_h, k_w = kernel_size
    pe_cycle_energy = (
        config.patch_control_energy_pj
        + config.neuron_pj
        + (config.xbar_size / 8.0) * k_h * k_w * config.xbar_array_energy_pj
    )
    return num_sites * p_i * q_i * pe_cycle_energy


def test_spikesim_event_energy_dense_equivalence():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=4)
    model = nn.Conv2d(3, 5, kernel_size=3, padding=0, bias=False).eval()
    x = torch.ones(2, 3, 8, 8)

    report = op_counter.estimate_spikesim_event_energy(model, x, config=config)

    out = model(x)
    expected = _dense_stage_energy(
        in_channels=3,
        out_channels=5,
        out_shape=tuple(out.shape),
        kernel_size=(3, 3),
        config=config,
    )
    assert len(report.energy_by_stage) == 1
    assert report.energy_total_pj == pytest.approx(expected)


def test_spikesim_event_energy_sparse_counts_hand_checked():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=2)
    model = nn.Conv2d(2, 2, kernel_size=2, bias=False).eval()
    x = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ]
    )

    report = op_counter.estimate_spikesim_event_energy(model, x, config=config)
    stage = next(iter(report.event_stats_by_stage))
    stats = report.event_stats_by_stage[stage]

    assert stats["active_patch_tile_count"] == 2
    assert stats["active_row_count"] == 2
    assert stats["active_output_tile_site_count"] == 2
    assert stats["dense_patch_tile_count"] == 4
    assert stats["dense_row_count"] == 32
    assert stats["dense_output_tile_site_count"] == 4


def test_spikesim_event_energy_multi_step_matches_repeated_single_step():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=4)
    model_m = layer.Conv2d(
        3, 4, kernel_size=3, padding=1, bias=False, step_mode="m"
    ).eval()
    model_s = layer.Conv2d(
        3, 4, kernel_size=3, padding=1, bias=False, step_mode="s"
    ).eval()
    with torch.no_grad():
        model_s.weight.copy_(model_m.weight)

    x_seq = (torch.rand(3, 2, 3, 8, 8) > 0.7).float()
    report_m = op_counter.estimate_spikesim_event_energy(model_m, x_seq, config=config)

    total_s = 0.0
    for t in range(x_seq.shape[0]):
        total_s += op_counter.estimate_spikesim_event_energy(
            model_s, x_seq[t], config=config
        ).energy_total_pj

    assert report_m.energy_total_pj == pytest.approx(total_s)


def test_spikesim_event_energy_mixed_dense_and_event_driven_stages():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=4)

    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)

        def forward(self, x):
            x = self.conv1(x)
            x = (x > 0).float()
            x = self.conv2(x)
            return x

    model = MixedModel().eval()
    x = torch.rand(2, 3, 8, 8)
    report = op_counter.estimate_spikesim_event_energy(model, x, config=config)

    conv1_stats = report.stage_metadata["MixedModel.conv1"]
    conv2_stats = report.stage_metadata["MixedModel.conv2"]
    assert conv1_stats["dense_fallback_calls"] == 1
    assert conv1_stats["event_driven_calls"] == 0
    assert conv2_stats["dense_fallback_calls"] == 0
    assert conv2_stats["event_driven_calls"] == 1


def test_spikesim_event_energy_repeated_same_scope_accumulates():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=4)

    class ReuseConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)

        def forward(self, x):
            y1 = self.conv(x)
            y2 = self.conv(x)
            return y1 + y2

    model = ReuseConvModel().eval()
    x = (torch.rand(2, 3, 8, 8) > 0.5).float()
    report = op_counter.estimate_spikesim_event_energy(model, x, config=config)

    assert list(report.stage_metadata.keys()) == ["ReuseConvModel.conv"]
    assert report.stage_metadata["ReuseConvModel.conv"]["total_calls"] == 2
    assert any("merge-like op aten.add.Tensor" in msg for msg in report.warnings)


def test_spikesim_event_energy_counter_base_contract_nonempty():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=4)
    model = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False).eval()
    x = (torch.rand(1, 3, 8, 8) > 0.5).float()

    with op_counter.SpikeSimEventEnergyProfiler(config=config) as profiler:
        _ = model(x)

    records = profiler._counter.get_counts()
    assert "Global" in records
    assert records["Global"][torch.ops.aten.convolution.default] == 1
    assert profiler._counter.get_total() == 1


def test_spikesim_event_energy_config_validation():
    with pytest.raises(ValueError):
        op_counter.SpikeSimEnergyConfig(xbar_size=0).validate()
    with pytest.raises(ValueError):
        op_counter.SpikeSimEnergyConfig(device="foo").validate()


def test_spikesim_event_energy_no_supported_stage_warning():
    model = nn.ReLU().eval()
    x = torch.randn(2, 3)
    report = op_counter.estimate_spikesim_event_energy(model, x)
    assert any("No supported Conv2d stages" in msg for msg in report.warnings)


def test_spikesim_event_energy_shape_mismatch_warns_without_crash():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=4)

    class ShapeMismatchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)

        def forward(self, x):
            y1 = self.conv(x)
            y2 = self.conv(x[:, :, :-1, :-1])
            return y1, y2

    model = ShapeMismatchModel().eval()
    x = (torch.rand(1, 3, 8, 8) > 0.5).float()
    report = op_counter.estimate_spikesim_event_energy(model, x, config=config)

    assert any("inconsistent shapes" in msg for msg in report.warnings)


def test_spikesim_event_energy_formula_requires_both_tile_inputs():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=4)
    stats = {
        "active_patch_tile_count": 1,
        "active_row_count": 1,
        "active_output_tile_site_count": 1,
        "active_row_count_by_tile": [1],
    }
    metadata = {
        "in_channels": 4,
        "out_channel_tiles": 1,
    }

    with pytest.raises(ValueError, match="both be present or both be absent"):
        compute_spikesim_event_energy_breakdown(stats, metadata, config)


def test_spikesim_event_energy_profiler_context_cleanup_on_enter_failure():
    profiler = op_counter.SpikeSimEventEnergyProfiler()
    baseline_parents = set(profiler._tracker.parents)
    baseline_active_modules = set(profiler._tracker.active_modules)

    original_enter = profiler._trace_mode.__enter__

    def raising_enter():
        raise RuntimeError("boom")

    profiler._trace_mode.__enter__ = raising_enter
    try:
        with pytest.raises(RuntimeError):
            profiler.__enter__()
        assert profiler._tracker.parents == baseline_parents
        assert profiler._tracker.active_modules == baseline_active_modules
    finally:
        profiler._trace_mode.__enter__ = original_enter


def test_spikesim_event_energy_grouped_conv_warns_or_raises():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=4)
    grouped = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=2, bias=False).eval()
    x = (torch.rand(1, 4, 8, 8) > 0.5).float()

    report = op_counter.estimate_spikesim_event_energy(grouped, x, config=config)
    assert any("grouped/depthwise" in msg for msg in report.warnings)

    with pytest.raises(NotImplementedError):
        op_counter.estimate_spikesim_event_energy(
            grouped, x, config=config, strict=True
        )


def test_spikesim_event_energy_transposed_conv_and_linear_warn():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=4)
    conv_t = nn.ConvTranspose2d(4, 4, kernel_size=3, padding=1, bias=False).eval()
    x = (torch.rand(1, 4, 8, 8) > 0.5).float()
    report_t = op_counter.estimate_spikesim_event_energy(conv_t, x, config=config)
    assert any("transposed convolutions" in msg for msg in report_t.warnings)

    linear = nn.Linear(8, 4, bias=False).eval()
    y = torch.randn(2, 8)
    report_linear = op_counter.estimate_spikesim_event_energy(linear, y, config=config)
    assert any("linear-like op" in msg for msg in report_linear.warnings)


def test_spikesim_event_energy_scalar_add_does_not_warn():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=4)

    class ScalarAddModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)

        def forward(self, x):
            return self.conv(x) + 1.5

    model = ScalarAddModel().eval()
    x = (torch.rand(1, 3, 8, 8) > 0.5).float()
    report = op_counter.estimate_spikesim_event_energy(model, x, config=config)

    assert not any("merge-like op" in msg for msg in report.warnings)


def test_spikesim_event_energy_tail_tile_rows_cost_more():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=4)
    model = nn.Conv2d(5, 4, kernel_size=1, bias=False).eval()
    x_full_tile = torch.zeros(1, 5, 1, 1)
    x_full_tile[:, 0, 0, 0] = 1.0
    x_tail_tile = torch.zeros(1, 5, 1, 1)
    x_tail_tile[:, 4, 0, 0] = 1.0

    report_full = op_counter.estimate_spikesim_event_energy(
        model, x_full_tile, config=config
    )
    report_tail = op_counter.estimate_spikesim_event_energy(
        model, x_tail_tile, config=config
    )

    stage = next(iter(report_full.energy_by_stage))
    full_xbar = report_full.energy_by_component["by_stage"][stage]["xbar_pj"]
    tail_xbar = report_tail.energy_by_component["by_stage"][stage]["xbar_pj"]
    assert tail_xbar > full_xbar


def test_spikesim_event_energy_monotonic_with_input_sparsity():
    config = op_counter.SpikeSimEnergyConfig(xbar_size=4)
    model = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False).eval()
    dense_x = (torch.rand(2, 3, 8, 8) > 0.2).float()
    sparse_x = (torch.rand(2, 3, 8, 8) > 0.8).float()

    dense_report = op_counter.estimate_spikesim_event_energy(
        model, dense_x, config=config
    )
    sparse_report = op_counter.estimate_spikesim_event_energy(
        model, sparse_x, config=config
    )

    assert dense_report.energy_total_pj >= sparse_report.energy_total_pj
