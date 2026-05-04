import torch
import torch.nn as nn

from spikingjelly.activation_based import op_counter


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
