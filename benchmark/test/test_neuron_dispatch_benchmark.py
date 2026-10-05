import pytest
import torch

from benchmark.check_neuron_dispatch import _arguments, _check_gpu_processes, _summary


def test_dispatch_summary_uses_paired_ratios():
    rows = [
        {"direct": 8, "cached": 10, "unified": 9},
        {"direct": 80, "cached": 100, "unified": 90},
    ]
    summary = _summary(rows)
    assert summary["unified_vs_cached_percent"] == pytest.approx(-10)
    assert summary["paired_bootstrap_95_percent"] == pytest.approx([-10, -10])
    assert summary["median_us"]["unified"] == 49.5


def test_izhikevich_benchmark_includes_both_state_gradients():
    from spikingjelly._ops import izhikevich

    torch.manual_seed(42)
    args, inputs = _arguments("izhikevich", 4, 7, "cpu")
    spikes, voltage, recovery, *_ = izhikevich._forward(*args)
    assert spikes.shape == (4, 7)
    assert voltage.shape == recovery.shape == (7,)
    gradients = torch.autograd.grad(
        (spikes, voltage, recovery),
        inputs,
        tuple(torch.ones_like(t) for t in (spikes, voltage, recovery)),
    )
    assert len(gradients) == 3
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


def test_gpu_process_check_rejects_only_other_processes_on_target(monkeypatch):
    import os
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="test")
    )
    snapshot = f"GPU-test, {os.getpid()}\nGPU-other, 999999\n"
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: snapshot)
    _check_gpu_processes()
    snapshot += "GPU-test, 999999\n"
    with pytest.raises(RuntimeError, match="discard this round"):
        _check_gpu_processes()
