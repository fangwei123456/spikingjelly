import torch
import torch.nn as nn

from spikingjelly.activation_based import op_counter


def test_analytical_memory_counter_matches_dense_lower_bound_for_dense_linear():
    model = nn.Linear(8, 4, bias=False)
    x = torch.rand(3, 8)
    analytical = op_counter.AnalyticalMemoryCounter()
    baseline = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode([analytical, baseline]):
        _ = model(x)

    assert analytical.get_total() == baseline.get_total()
    metrics = analytical.get_metric_counts()["Global"]
    assert metrics["dense_memory_bytes"] == analytical.get_total()
    assert metrics["sparse_memory_bytes"] == 0


def test_analytical_memory_counter_reduces_sparse_linear_memory():
    model = nn.Linear(8, 4, bias=False)
    x = torch.zeros(3, 8)
    x[0, 0] = 0.2
    x[1, 3] = 0.5
    analytical = op_counter.AnalyticalMemoryCounter()
    baseline = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode([analytical, baseline]):
        _ = model(x)

    assert analytical.get_total() < baseline.get_total()
    metrics = analytical.get_metric_counts()["Global"]
    assert metrics["sparse_memory_bytes"] == analytical.get_total()
    assert metrics["read_in_bytes"] == int(x.count_nonzero().item()) * x.element_size()
    assert metrics["read_params_buffer_bytes"] == (
        model.weight.numel() * model.weight.element_size()
    )
    assert metrics["memory_buffer_bytes"] == max(
        metrics["read_in_buffer_bytes"],
        metrics["read_params_buffer_bytes"],
        metrics["write_out_buffer_bytes"],
    )


def test_analytical_memory_counter_reduces_sparse_conv_memory():
    model = nn.Conv2d(2, 4, kernel_size=3, bias=False)
    x = torch.zeros(1, 2, 5, 5)
    x[:, 0, 1, 1] = 0.5
    x[:, 1, 3, 3] = 0.25
    analytical = op_counter.AnalyticalMemoryCounter()
    baseline = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode([analytical, baseline]):
        _ = model(x)

    assert analytical.get_total() < baseline.get_total()
    metrics = analytical.get_metric_counts()["Global"]
    assert metrics["sparse_memory_bytes"] == analytical.get_total()


def test_analytical_memory_counter_warns_dense_fallback_for_bn():
    model = nn.BatchNorm1d(8)
    x = torch.zeros(2, 8)
    x[:, :2] = 0.25
    analytical = op_counter.AnalyticalMemoryCounter()

    with op_counter.DispatchCounterMode([analytical]):
        _ = model(x)

    assert analytical.get_metric_counts()["Global"]["fallback_dense_ops"] > 0
    assert analytical.warnings
