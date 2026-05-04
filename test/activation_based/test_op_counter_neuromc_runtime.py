import torch
import torch.nn as nn
from spikingjelly.activation_based import op_counter
from spikingjelly.activation_based.op_counter.neuromc.memory_residency_counter import (
    _access_convolution_backward,
)
from spikingjelly.activation_based.op_counter.neuromc.residency import (
    MemoryResidencySimulator,
)


def test_neuromc_runtime_sparse_changes_counts():
    model = nn.Linear(16, 8, bias=False)
    dense_x = torch.randn(4, 16)
    spike_x = (torch.rand(4, 16) > 0.8).float()

    dense_report = op_counter.estimate_neuromc_runtime_energy(model, dense_x)
    spike_report = op_counter.estimate_neuromc_runtime_energy(model, spike_x)

    dense_mul = dense_report.primitive_counts["totals"]["mul"]
    spike_mul = spike_report.primitive_counts["totals"]["mul"]
    assert dense_mul > spike_mul
    assert dense_report.energy_total_pj > spike_report.energy_total_pj


def test_neuromc_runtime_backward_optimizer_stage():
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    x = torch.randn(6, 8)
    target = torch.randn(6, 4)
    report = op_counter.estimate_neuromc_runtime_energy(
        model, x, target=target, loss_fn=loss_fn, optimizer=optimizer
    )

    assert report.energy_by_stage.get("forward", 0.0) > 0.0
    assert report.energy_by_stage.get("backward", 0.0) > 0.0
    assert report.energy_by_stage.get("optimizer", 0.0) >= 0.0
    assert abs(sum(report.energy_by_stage.values()) - report.energy_total_pj) < 1e-3


def test_neuromc_runtime_mm_hand_calculated():
    class MMModel(nn.Module):
        def forward(self, x, y):
            return torch.mm(x, y)

    model = MMModel()
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    report = op_counter.estimate_neuromc_runtime_energy(model, (x, y))

    assert report.primitive_counts["totals"]["mul"] == 2 * 3 * 4
    assert report.primitive_counts["totals"]["add"] == 2 * 4 * (3 - 1)


def test_neuromc_runtime_bn_conv_where_sqrt_nonzero():
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
            self.bn = nn.BatchNorm2d(4)
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = x.mean(dim=(2, 3))
            x = self.fc(x)
            x = torch.where(x > 0, x, -x)
            x = torch.sqrt(x + 1e-6)
            return x

    model = MixedModel().eval()
    x = torch.rand(5, 3, 8, 8)
    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    totals = report.primitive_counts["totals"]
    assert totals["mul"] > 0
    assert totals["add"] > 0
    assert totals["cmp"] > 0
    assert totals["sqrt"] > 0
    assert totals["mux"] > 0


def test_neuromc_runtime_sparse_conv_add_counts_actual_additions():
    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=2, bias=False)
            with torch.no_grad():
                self.conv.weight.fill_(1.0)

        def forward(self, x):
            return self.conv(x)

    model = ConvModel().eval()
    x = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

    report = op_counter.estimate_neuromc_runtime_energy(model, x)

    assert report.primitive_counts["totals"]["add"] == 0


def test_neuromc_runtime_unsupported_warning():
    class UnsupportedModel(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    model = UnsupportedModel()
    x = torch.randn(8, 8)
    report = op_counter.estimate_neuromc_runtime_energy(model, x)
    assert any("Unsupported aten ops" in msg for msg in report.warnings)


def test_neuromc_runtime_bn_affine_false_backward_no_crash():
    model = nn.Sequential(
        nn.BatchNorm1d(8, affine=False),
        nn.Linear(8, 4),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    x = torch.randn(6, 8)
    target = torch.randn(6, 4)
    report = op_counter.estimate_neuromc_runtime_energy(
        model, x, target=target, loss_fn=loss_fn, optimizer=optimizer
    )
    assert report.energy_by_stage.get("backward", 0.0) > 0.0


def test_neuromc_runtime_add_alpha_counts_mul():
    class AddAlphaModel(nn.Module):
        def forward(self, x, y):
            return torch.add(x, y, alpha=2.0)

    model = AddAlphaModel()
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    report = op_counter.estimate_neuromc_runtime_energy(model, (x, y))
    assert report.primitive_counts["totals"]["mul"] == x.numel()


def test_neuromc_runtime_add_scalar_alpha_counts_mul():
    class AddScalarAlphaModel(nn.Module):
        def forward(self, x):
            return torch.add(x, 1.5, alpha=2.0)

    model = AddScalarAlphaModel()
    x = torch.randn(3, 4)
    report = op_counter.estimate_neuromc_runtime_energy(model, x)
    assert report.primitive_counts["totals"]["mul"] == x.numel()


def test_neuromc_runtime_addmm_beta_scaled_bias_no_extra_add():
    class AddMMModel(nn.Module):
        def forward(self, bias, x, y):
            return torch.addmm(bias, x, y, beta=2.0, alpha=1.0)

    model = AddMMModel()
    m, k, n = 3, 4, 5
    bias = torch.randn(n)
    x = torch.randn(m, k)
    y = torch.randn(k, n)
    report = op_counter.estimate_neuromc_runtime_energy(model, (bias, x, y))

    expected_add = m * n * (k - 1) + m * n
    assert report.primitive_counts["totals"]["add"] == expected_add


def test_neuromc_runtime_profiler_arbitrary_inference_flow():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    x = torch.randn(5, 8)

    with op_counter.NeuroMCEnergyProfiler() as profiler:
        with profiler.stage("custom_inference"):
            _ = model(x)
    report = profiler.get_report()

    assert report.energy_by_stage.get("custom_inference", 0.0) > 0.0
    assert "backward" not in report.energy_by_stage


def test_neuromc_runtime_profiler_arbitrary_online_like_training_flow():
    model = nn.Linear(6, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    xs = torch.randn(3, 4, 6)
    targets = torch.randn(3, 4, 3)

    with op_counter.NeuroMCEnergyProfiler() as profiler:
        for t in range(xs.shape[0]):
            with profiler.stage(f"t{t}_forward"):
                out = model(xs[t])
                loss = loss_fn(out, targets[t])
            with profiler.stage(f"t{t}_backward"):
                loss.backward()
            with profiler.stage(f"t{t}_update"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    report = profiler.get_report()

    assert report.energy_by_stage.get("t0_forward", 0.0) > 0.0
    assert report.energy_by_stage.get("t1_backward", 0.0) > 0.0
    assert report.energy_by_stage.get("t2_update", 0.0) >= 0.0
    assert abs(sum(report.energy_by_stage.values()) - report.energy_total_pj) < 1e-3


def test_neuromc_runtime_residency_reuse_lowers_dram_traffic():
    x = torch.randn(1024)
    xs = [torch.randn(1024) for _ in range(13)]

    with op_counter.NeuroMCEnergyProfiler(
        memory_config=op_counter.MemoryHierarchyConfig.neuromc_like_v1(
            memory_model="residency"
        )
    ) as reuse_profiler:
        with reuse_profiler.stage("reuse"):
            y = x
            for _ in range(12):
                y = torch.add(y, x)
    reuse_report = reuse_profiler.get_report()

    with op_counter.NeuroMCEnergyProfiler(
        memory_config=op_counter.MemoryHierarchyConfig.neuromc_like_v1(
            memory_model="residency"
        )
    ) as non_reuse_profiler:
        with non_reuse_profiler.stage("non_reuse"):
            y = xs[0]
            for i in range(1, len(xs)):
                y = torch.add(y, xs[i])
    non_reuse_report = non_reuse_profiler.get_report()

    reuse_dram_bits = reuse_report.memory_bits_by_level["totals"].get("dram", 0)
    non_reuse_dram_bits = non_reuse_report.memory_bits_by_level["totals"].get("dram", 0)
    assert non_reuse_dram_bits > reuse_dram_bits


def test_neuromc_runtime_residency_small_capacity_triggers_eviction():
    tiny_cfg = op_counter.MemoryHierarchyConfig.neuromc_like_v1(
        memory_model="residency"
    )
    tiny_cfg.capacity_bits["reg"] = float(1024)
    tiny_cfg.capacity_bits["sram"] = float(2048)

    xs = [torch.randn(16) for _ in range(12)]
    with op_counter.NeuroMCEnergyProfiler(memory_config=tiny_cfg) as profiler:
        with profiler.stage("eviction"):
            y = xs[0]
            for i in range(1, len(xs)):
                y = torch.add(y, xs[i])
    report = profiler.get_report()
    move_bits = report.memory_bits_by_level["move_bits_by_edge"]
    assert move_bits.get("sram->dram", 0) > 0


def test_neuromc_runtime_weighted_model_compatibility():
    model = nn.Linear(16, 8, bias=False)
    x = torch.randn(4, 16)
    report = op_counter.estimate_neuromc_runtime_energy(
        model, x, memory_model="weighted"
    )

    assert report.memory_bits_by_level["memory_model"] == "weighted"
    assert report.energy_memory_pj > 0


def test_neuromc_runtime_memory_config_model_preserved():
    model = nn.Linear(16, 8, bias=False)
    x = torch.randn(4, 16)
    cfg = op_counter.MemoryHierarchyConfig.neuromc_like_v1(memory_model="weighted")

    report = op_counter.estimate_neuromc_runtime_energy(model, x, memory_config=cfg)

    assert report.memory_bits_by_level["memory_model"] == "weighted"


def test_neuromc_access_convolution_backward_fixed_output_slots():
    grad_out = torch.randn(2, 4, 8, 8)
    x = torch.randn(2, 3, 8, 8)
    w = torch.randn(4, 3, 3, 3)
    output_mask = (False, True, True)

    grad_input = None
    grad_weight = torch.randn_like(w)
    grad_bias = torch.randn(w.shape[0])
    out = (grad_input, grad_weight, grad_bias)
    args = (
        grad_out,
        x,
        w,
        None,
        [1, 1],
        [1, 1],
        [1, 1],
        False,
        [0, 0],
        1,
        output_mask,
    )

    reads, writes = _access_convolution_backward(args, {}, out)

    assert any(t is grad_weight for t in writes)
    assert any(t is grad_bias for t in writes)


def test_memory_residency_reg_hit_updates_sram_lru():
    cfg = op_counter.MemoryHierarchyConfig.neuromc_like_v1(memory_model="residency")
    cfg.capacity_bits.update({"reg": 1024.0, "sram": 1024.0, "dram": float("inf")})
    sim = MemoryResidencySimulator(cfg)

    a = torch.randn(16)
    b = torch.randn(16)
    c = torch.randn(16)

    sim.on_tensor_read(a, "op")
    sim.on_tensor_read(b, "op")
    for _ in range(3):
        sim.on_tensor_read(a, "op")
    sim.on_tensor_read(c, "op")

    key_a = sim._tensor_key(a)
    key_b = sim._tensor_key(b)
    assert key_a in sim.sram_cache
    assert key_b not in sim.sram_cache
