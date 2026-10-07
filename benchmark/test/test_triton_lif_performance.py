import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import benchmark.check_triton_lif_performance as check


def _round(ratios):
    return {
        "status": "complete",
        "samples": [
            {
                "baseline_ms": 1.0,
                "candidate_ms": r,
                "baseline_samples_ms": [1.0, 1.0],
                "candidate_samples_ms": [r, r],
            }
            for r in ratios
        ],
    }


@pytest.mark.parametrize(
    "rounds,status",
    [
        ([[1.01, 1.01], [1.0, 1.0]], "pass"),
        ([[0.99, 1.1, 1.1], [0.98, 1.08, 1.08]], "fail"),
        ([[1.01, 1.01], [1.1, 1.1]], "inconclusive"),
    ],
)
def test_regression_verdict_uses_round_medians(rounds, status):
    result = check._summarize([_round(r) for r in rounds], 2.0)
    assert result["status"] == status
    assert result["pair_count"] == sum(map(len, rounds))


def test_summary_uses_paired_ratios_not_ratio_of_timing_medians():
    rows = _round([1.1, 2.0, 0.5])
    for row, scale in zip(rows["samples"], [1, 2, 100]):
        for key in ("baseline_ms", "candidate_ms"):
            row[key] *= scale
        for key in ("baseline_samples_ms", "candidate_samples_ms"):
            row[key] = [value * scale for value in row[key]]
    result = check._summarize([rows], 2.0)
    assert result["median_percent"] == pytest.approx(10.0)
    assert result["worst_pair_percent"] == 100.0


def test_unstable_repeats_cannot_certify_a_small_median_difference():
    result = _round([1.0, 1.0])
    result["samples"][0]["baseline_samples_ms"] = [0.9, 1.1]
    summary = check._summarize([result], 2.0)
    assert summary["status"] == "inconclusive"
    assert summary["reason"] == "repeat_spread_exceeds_limit"
    assert summary["unstable_pair_count"] == 1


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_timings_cannot_pass(bad):
    with pytest.raises(ValueError, match="finite, positive"):
        check._summarize([_round([bad])], 2.0)


def test_pairing_alternates_orders_and_preserves_samples(monkeypatch):
    clocks = iter([1, 2, 4, 3, 8, 5, 7, 10])
    observed = []

    def measure(graph, replays):
        observed.append(graph)
        return next(clocks) / replays

    monkeypatch.setattr(check, "_measure", measure)
    rows = check._paired_samples({"baseline": "A", "candidate": "B"}, 2, 2, 1)
    assert observed == list("ABBABAAB")
    assert rows[0]["baseline_ms"] == 1.0
    assert rows[0]["candidate_ms"] == 1.5
    assert rows[1]["baseline_ms"] == 3.0
    assert rows[1]["candidate_ms"] == 4.5
    assert rows[0]["baseline_samples_ms"] == [0.5, 1.5]


def test_tiny_gradients_are_not_hidden_by_absolute_tolerance():
    expected = torch.tensor([1e-12, -2e-12])
    check._check_gradient(expected * (1 + 1e-5), expected)
    with pytest.raises(AssertionError):
        check._check_gradient(torch.zeros_like(expected), expected)
    with pytest.raises(AssertionError):
        check._check_gradient(torch.full_like(expected, float("nan")), expected)
    check._check_gradient(torch.zeros(2), torch.zeros(2))


@pytest.mark.parametrize(
    "status,exit_code", [("pass", 0), ("fail", 1), ("inconclusive", 2)]
)
def test_cli_writes_all_rounds_and_returns_verdict(
    tmp_path, monkeypatch, status, exit_code
):
    ratios = {"pass": [1.01, 1.0], "fail": [1.1, 1.1], "inconclusive": [1.0, 1.1]}[
        status
    ]

    def worker(command, stdout, stderr):
        index = int(command[command.index("--round-index") + 1])
        output = Path(command[command.index("--output") + 1])
        check._write(output, _round([ratios[index - 1]] * 2))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(check.subprocess, "run", worker)
    output = tmp_path / "check.json"
    assert (
        check._main(["--rounds", "2", "--pairs", "2", "--output", str(output)])
        == exit_code
    )
    report = json.loads(output.read_text())
    assert report["status"] == status
    assert report["arguments"]["surrogate"] == "ATan"
    assert report["arguments"]["rounds"] == 2
    assert len(report["rounds"]) == 2
    assert report["summary"]["pair_count"] == 4


def test_worker_failure_does_not_reuse_old_pass_report(tmp_path, monkeypatch):
    output = tmp_path / "check.json"
    check._write(tmp_path / "check.round-1.json", _round([1.0]))

    def worker(command, stdout, stderr):
        stdout.write("CUDA unavailable\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(check.subprocess, "run", worker)
    assert check._main(["--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "error"
    assert report["rounds"] == []
    assert Path(report["error_log"]).read_text() == "CUDA unavailable\n"


@pytest.mark.parametrize(
    "option,value",
    [("--pairs", "1"), ("--rounds", "0"), ("--max-regression-percent", "nan")],
)
def test_invalid_cli_is_rejected(option, value, tmp_path):
    with pytest.raises(SystemExit) as error:
        check._main(["--output", str(tmp_path / "result.json"), option, value])
    assert error.value.code == 2


@pytest.mark.skipif(
    not torch.cuda.is_available() or bool(torch.version.hip),
    reason="NVIDIA CUDA required",
)
def test_captured_graphs_write_same_buffers_and_read_live_inputs():
    pytest.importorskip("triton")
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("shared graph output buffers require the native CUDA allocator")
    from spikingjelly._ops.lif.triton import _backward_impl
    from spikingjelly._ops.lif import triton_precision as lif

    torch.manual_seed(13)
    inputs = (
        torch.randn(4, 257, device="cuda"),
        torch.randn(257, device="cuda"),
        torch.randn(4, 257, device="cuda"),
    )
    record = {
        "inputs": inputs,
        "parameters": dict(
            tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            soft_reset=False,
            detach_reset=True,
            sg_alpha=2.0,
            sg_triton_id=1,
            store_v_seq=False,
            compute_dtype_id=0,
            storage_dtype_id=0,
        ),
    }
    with torch.no_grad():
        graphs, outputs, checks = check._make_graphs(
            [record], lif._launch_lif_backward_kernel, _backward_impl
        )
        assert set(checks) == {"baseline", "candidate"}
        inputs[0].zero_()
        inputs[1].zero_()
        for graph in graphs.values():
            for tensor in outputs[0]:
                tensor.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for tensor in outputs[0]:
                torch.testing.assert_close(tensor, torch.zeros_like(tensor))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_round_captures_the_automatically_dispatched_triton_path(tmp_path, monkeypatch):
    pytest.importorskip("triton")
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("shared graph buffers require the native CUDA allocator")
    from spikingjelly.activation_based import neuron, surrogate
    from spikingjelly._ops.lif import _selection

    monkeypatch.setattr(_selection, "_requested", "triton")
    monkeypatch.setattr(_selection, "_selections", {})
    monkeypatch.setattr(_selection, "_compiled_selections", {})

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 1000)
            self.lif = neuron.LIFNode(
                step_mode="m", surrogate_function=surrogate.ATan()
            )

        def forward(self, x):
            return self.lif(self.linear(x.mean((2, 3))).unsqueeze(0).repeat(2, 1, 1))

    monkeypatch.setattr(check, "_build_model", lambda *a, **kw: Network())
    args = check._parser().parse_args(
        [
            "--output",
            str(tmp_path / "result.json"),
            "--batch-size",
            "2",
            "--T",
            "2",
            "--image-size",
            "4",
            "--warmup-steps",
            "1",
            "--rounds",
            "1",
            "--pairs",
            "2",
            "--replays",
            "1",
            "--round-index",
            "1",
        ]
    )
    result = check._run_round(args)
    assert result["status"] == "complete"
    assert result["workload"]["backward_calls_per_replay"] == 1
    assert set(result["correctness"]) == {"baseline", "candidate"}
