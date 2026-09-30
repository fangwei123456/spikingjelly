import json
import os
import shlex
import sqlite3
import subprocess
from pathlib import Path

import pytest
import torch

from benchmark import nsys_lif_example
from benchmark.analyze_nsys_snn import _category, _write_report, analyze, compare
from spikingjelly import nsys
from spikingjelly.activation_based import neuron


@pytest.mark.parametrize(
    ("mode", "graph_trace", "options", "trace", "command"),
    [
        ("capture", "node", [], "cuda,nvtx,osrt,python-gil", ["python", "-c", "pass"]),
        (
            "capture-graph",
            "node:nvtx-precapture",
            [],
            "cuda,nvtx,osrt,python-gil",
            ["python"],
        ),
        (
            "capture",
            "node",
            ["--control=api", "--trace=cuda,nvtx,cublas,cudnn"],
            "cuda,nvtx,cublas,cudnn",
            ["python"],
        ),
        (
            "capture",
            "node",
            ["--control=manual", "--session=sj test"],
            "cuda,nvtx,osrt,python-gil",
            ["python", "service.py", "a b"],
        ),
        (
            "capture-graph",
            "node:nvtx-precapture",
            ["--control=manual", "--session=sj test"],
            "cuda,nvtx,osrt,python-gil",
            ["python"],
        ),
    ],
)
def test_shell_capture_trace_mode(tmp_path, mode, graph_trace, options, trace, command):
    nsys_command = tmp_path / "nsys"
    nsys_command.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = --version ]; then echo 'NVIDIA Nsight Systems test'; exit 0; fi\n"
        'printf \'%s\\n\' "$@" > "$NSYS_ARGS"\n',
        encoding="utf-8",
    )
    nsys_command.chmod(0o755)
    args_file = tmp_path / "nsys-args.txt"
    output = tmp_path / "capture output"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "NSYS_ARGS": str(args_file),
        "SJ_USE_TRITON_OP": "0",
    }
    result = subprocess.run(
        [
            "bash",
            str(Path(__file__).resolve().parents[1] / "nsys_snn.sh"),
            mode,
            *options,
            str(output),
            "--",
            *command,
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = args_file.read_text().splitlines()
    assert args[-len(command) :] == command
    assert f"--trace={trace}" in args
    assert "--wait=all" in args
    assert f"--cuda-graph-trace={graph_trace}" in args
    assert "--pytorch=none" in args
    assert "--python-sampling=false" in args
    assert not any(arg.startswith("--python-functions-trace") for arg in args)
    manifest = json.loads(Path(f"{output}.manifest.json").read_text())
    assert manifest["trace"] == trace
    assert manifest["cuda_graph_trace"] == graph_trace
    assert manifest["sj_use_triton_op"] == "0"
    assert manifest["pytorch_trace"] == "none"
    assert manifest["python_sampling"] is False
    assert manifest["command"] == command
    if "--control=manual" in options:
        assert args[0] == "launch"
        assert "--session-new=sj test" in args
        assert "--sample=none" not in args
        assert "--cpuctxsw=none" not in args
        assert not any(arg.startswith("--capture-range") for arg in args)
        assert manifest["control"] == "manual"
        assert manifest["session"] == "sj test"
        assert manifest["capture_range"] == "none"
        start = next(
            line for line in result.stdout.splitlines() if line.startswith("nsys start")
        )
        assert shlex.split(start) == [
            "nsys",
            "start",
            "--session=sj test",
            "--capture-range=none",
            "--sample=none",
            "--cpuctxsw=none",
            "--stop-on-exit=false",
            f"--output={output}",
        ]
        assert "nsys stop" in result.stdout
        assert "nsys shutdown" in result.stdout
    else:
        assert args[0] == "profile"
        assert "--kill=none" in args
        assert "--capture-range=cudaProfilerApi" in args
        assert manifest["control"] == "api"
        assert manifest["session"] is None
        assert manifest["capture_range"] == "cudaProfilerApi"


@pytest.mark.parametrize(
    "options",
    [
        ["--control=invalid"],
        ["--control=manual"],
        ["--session=sj"],
        ["--trace="],
        ["--unknown=1"],
    ],
)
def test_shell_capture_rejects_invalid_control_options(tmp_path, options):
    result = subprocess.run(
        [
            "bash",
            str(Path(__file__).resolve().parents[1] / "nsys_snn.sh"),
            "capture",
            *options,
            str(tmp_path / "capture"),
            "--",
            "python",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert not (tmp_path / "capture.manifest.json").exists()


def test_service_profile_marks_capture_from_step_zero(monkeypatch):
    ranges = []
    profiler = []
    original_randn = torch.randn
    original_randint = torch.randint
    monkeypatch.setattr(nsys_lif_example.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nsys_lif_example.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        nsys_lif_example.torch.cuda.profiler,
        "start",
        lambda: profiler.append("start"),
    )
    monkeypatch.setattr(
        nsys_lif_example.torch.cuda.profiler,
        "stop",
        lambda: profiler.append("stop"),
    )
    monkeypatch.setattr(nsys_lif_example.torch.cuda.nvtx, "range_push", ranges.append)
    monkeypatch.setattr(nsys_lif_example.torch.cuda.nvtx, "range_pop", lambda: None)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(
        nsys_lif_example.torch,
        "randn",
        lambda *args, **kwargs: original_randn(*args),
    )
    monkeypatch.setattr(
        nsys_lif_example.torch,
        "randint",
        lambda *args, **kwargs: original_randint(*args),
    )
    monkeypatch.setattr(
        nsys_lif_example.sys,
        "argv",
        ["nsys_lif_example.py", "--mode", "serve", "--warmup", "0"],
    )
    monkeypatch.setattr(
        nsys_lif_example.sys,
        "stdin",
        iter(["run\n", "ignored\n", "profile\n", "quit\n"]),
    )

    nsys_lif_example.main()

    assert profiler == ["start", "stop"]
    assert ranges.count("sj.step:inference:0") == 1


def test_capture_and_ranges_balance_on_error(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nsys.torch.cuda.profiler, "start", lambda: calls.append("start")
    )
    monkeypatch.setattr(nsys.torch.cuda.profiler, "stop", lambda: calls.append("stop"))
    monkeypatch.setattr(nsys.torch.cuda.nvtx, "range_push", calls.append)
    monkeypatch.setattr(nsys.torch.cuda.nvtx, "range_pop", lambda: calls.append("pop"))

    with pytest.raises(ZeroDivisionError):
        with nsys.capture(True), nsys.step(0, "training", True):
            with nsys.region("forward", True):
                1 / 0
    assert calls == ["start", "sj.step:training:0", "forward", "pop", "pop", "stop"]

    calls.clear()
    with nsys.capture(), nsys.step(0, "inference"):
        pass
    assert calls == []


@pytest.mark.parametrize(("index", "phase"), [(-1, "training"), (0, "other")])
def test_step_rejects_invalid_labels(index, phase):
    with pytest.raises(ValueError, match="step requires"):
        nsys.step(index, phase)


def test_sqlite_attribution_uses_launch_correlation_and_gpu_union(tmp_path):
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT,
                textId INTEGER, globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER,
                correlationId INTEGER, globalTid INTEGER, nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER,
                correlationId INTEGER, demangledName INTEGER, globalPid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER,
                correlationId INTEGER, globalPid INTEGER, copyKind INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER,
                correlationId INTEGER, globalPid INTEGER, memKind INTEGER);
            INSERT INTO StringIds VALUES
                (1, 'lif_forward'), (2, 'conv_kernel'),
                (3, 'cudaStreamSynchronize_v3020'),
                (4, 'cudaLaunchKernel_v7000');
            INSERT INTO NVTX_EVENTS VALUES
                (0, 50000000, 'sj.step:training:0', NULL, 4294967303),
                (1000000, 25000000, 'forward', NULL, 4294967303),
                (1000000, 4000000, 'module:lif.0', NULL, 4294967303),
                (26000000, 49000000, 'reset', NULL, 4294967303),
                (50000000, 100000000, 'sj.step:training:1', NULL, 4294967303),
                (51000000, 75000000, 'forward', NULL, 4294967303),
                (99000000, 99900000, 'reset', NULL, 4294967303),
                (100000000, 150000000, 'sj.step:training:2', NULL, 4294967303),
                (101000000, 125000000, 'forward', NULL, 4294967303),
                (149000000, 149900000, 'reset', NULL, 4294967303),
                (150000000, 200000000, 'sj.step:training:3', NULL, 4294967303),
                (151000000, 175000000, 'forward', NULL, 4294967303),
                (199000000, 199900000, 'reset', NULL, 4294967303);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                (2000000, 3000000, 1, 4294967303, NULL),
                (20000000, 21000000, 2, 4294967303, NULL),
                (22000000, 25000000, 5, 4294967303, 3),
                (24000000, 28000000, NULL, 4294967303, 3),
                (60000000, 60010000, 4, 4294967304, 4),
                (110000000, 110010000, 6, 4294967303, NULL),
                (112000000, 112010000, 7, 4294967303, NULL),
                (199050000, 199060000, 3, 4294967303, NULL);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (10000000, 30000000, 1, 1, 4294967296),
                (20000000, 40000000, 2, 2, 4294967296),
                (10000000, 40000000, 1, 2, 4311744512),
                (61000000, 62000000, 4, 2, 4294967296),
                (199100000, 199200000, 3, 2, 4294967296);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
                (111000000, 111100000, 6, 4294967296, 2);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES
                (113000000, 113100000, 7, 4294967296, 2);
            """
        )
    report = analyze(path)
    step = report["steps"][0]
    assert step["gpu_event_count"] == 2
    assert step["gpu_busy_union_ms"] == 30
    assert step["cuda_api_union_ms"] == 8
    assert step["categories_ms"] == {"conv_gemm": 20, "neuron": 20}
    assert step["phases_ms"] == {"forward": 40}
    assert report["steps"][1]["gpu_event_count"] == 1
    assert report["steps"][1]["phases_ms"] == {"forward": 1}
    assert any(
        event["name"] == "cudaStreamSynchronize_v3020"
        and event["end_ns"] - event["start_ns"] == 3_000_000
        for event in report["cuda_api_events"]
    )
    assert any(
        event["correlation_id"] is None
        and event["name"] == "cudaStreamSynchronize_v3020"
        for event in report["cuda_api_events"]
    )
    assert sum(e["step_id"] == 0 for e in report["gpu_events"]) == 2
    assert len(report["steps"]) == 4
    assert report["gpu_events"][4]["phase"] == "reset"
    assert any(r["name"] == "module:lif.0" for r in report["ranges"])
    output = tmp_path / "analysis"
    _write_report(report, output)
    assert (output / "event_timeline.png").is_file()
    assert (output / "four_step_timeline.png").is_file()
    assert (output / "kernel_cost.png").is_file()
    assert (output / "module_timeline.png").is_file()
    with pytest.raises(FileExistsError, match="report already exists"):
        _write_report(report, output)
    assert not (output / "step_timeline.png").exists()
    assert report["gil_collected"] is False
    assert not (output / "gil_timeline.png").exists()

    later = analyze(path, step_index=1)
    assert later["selection"]["step_ids"] == [1]
    assert sum(e["step_id"] == 1 for e in later["gpu_events"]) == 1
    memory_step = analyze(path, step_index=2)
    assert [
        event["name"] for event in memory_step["gpu_events"] if event["step_id"] == 2
    ] == ["memcpy", "memset"]
    with pytest.raises(ValueError, match="index 9"):
        analyze(path, step_index=9)

    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text('{"memory": {"other": 1}}')
    assert analyze(path, benchmark_path)["peak_allocated_mib"] is None
    benchmark_path.write_text('{"memory": {"peak_allocated_bytes": 1048576}}')
    assert analyze(path, benchmark_path)["peak_allocated_mib"] == 1


def test_shell_analyze_rejects_nonempty_output_directory(tmp_path):
    output = tmp_path / "analysis"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve")
    result = subprocess.run(
        [
            "bash",
            str(Path(__file__).resolve().parents[1] / "nsys_snn.sh"),
            "analyze",
            str(tmp_path / "capture.nsys-rep"),
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "not empty" in result.stderr
    assert marker.read_text() == "preserve"


def test_graph_precapture_ranges_label_replayed_gpu_nodes(tmp_path):
    path = tmp_path / "graph.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT,
                textId INTEGER, globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER,
                correlationId INTEGER, globalTid INTEGER, nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER,
                correlationId INTEGER, demangledName INTEGER, globalPid INTEGER,
                streamId INTEGER, graphNodeId INTEGER);
            CREATE TABLE CUDA_GRAPH_NODE_EVENTS (start INTEGER, end INTEGER,
                graphNodeId INTEGER, originalGraphNodeId INTEGER);
            INSERT INTO StringIds VALUES
                (1, 'cudaGraphLaunch_v10000'), (2, 'lif_forward');
            INSERT INTO NVTX_EVENTS VALUES
                (-30000000, -20000000, 'forward', NULL, 4294967303),
                (-20000000, -10000000, 'backward', NULL, 4294967303),
                (0, 50000000, 'sj.step:training:0', NULL, 4294967303),
                (1000000, 2000000, 'graph_runner', NULL, 4294967303),
                (2000000, 40000000, 'optimizer', NULL, 4294967303);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                (1100000, 1200000, 1, 4294967303, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (2000000, 3000000, 1, 2, 4294967296, 7, 20),
                (3000000, 4000000, 1, 2, 4294967296, 7, 21);
            INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES
                (-25000000, -25000000, 10, NULL),
                (-15000000, -15000000, 11, NULL),
                (-5000000, -5000000, 20, 10),
                (-4000000, -4000000, 21, 11);
            """
        )
    report = analyze(path)
    assert [event["graph_stage"] for event in report["gpu_events"]] == [
        "forward",
        "backward",
    ]
    assert [
        phase["name"]
        for phase in report["ranges"]
        if phase["kind"] == "region" and phase["start_ns"] >= 0
    ] == [
        "graph_runner",
        "optimizer",
    ]
    output = tmp_path / "analysis"
    _write_report(report, output)
    assert (output / "graph_stage_timeline.png").is_file()
    assert (output / "graph_stage_timeline.png").read_bytes() == (
        output / "event_timeline.png"
    ).read_bytes()
    empty_output = tmp_path / "empty-window"
    _write_report(analyze(path, time_range_ms=(1000, 1001)), empty_output)
    assert not (empty_output / "event_timeline.png").exists()
    assert not (empty_output / "graph_stage_timeline.png").exists()


def test_gil_intervals_are_clipped_to_steps_and_kept_per_thread(tmp_path):
    path = tmp_path / "gil.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT,
                textId INTEGER, globalTid INTEGER, eventType INTEGER,
                domainId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER,
                correlationId INTEGER, globalTid INTEGER, nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER,
                correlationId INTEGER, demangledName INTEGER, globalPid INTEGER);
            INSERT INTO StringIds VALUES
                (1, 'Holding GIL'), (2, 'Waiting for GIL');
            INSERT INTO NVTX_EVENTS VALUES
                (0, NULL, 'GIL Trace', NULL, 4294967303, 75, 1),
                (0, 10000000, 'sj.step:training:0', NULL, 4294967303, 59, 0),
                (10000000, 20000000, 'sj.step:training:1', NULL, 4294967303, 59, 0),
                (0, 2000000, NULL, 1, 4294967303, 59, 1),
                (8000000, 12000000, NULL, 1, 4294967303, 59, 1),
                (-1000000, 4000000, NULL, 2, 4294967304, 59, 1),
                (0, 5000000, NULL, 2, 4311744519, 59, 1);
            """
        )
    report = analyze(path)
    assert report["gil_collected"] is True
    assert [step["main_thread_gil_holding_ms"] for step in report["steps"]] == [4, 2]
    assert [step["main_thread_gil_waiting_ms"] for step in report["steps"]] == [0, 0]
    assert [
        {key: t[key] for key in ("global_tid", "holding_ms", "waiting_ms")}
        for t in report["threads"]
        if t["process_id"] == 4294967296
    ] == [
        {"global_tid": 4294967303, "holding_ms": 6, "waiting_ms": 0},
        {"global_tid": 4294967304, "holding_ms": 0, "waiting_ms": 4},
    ]
    assert len([r for r in report["ranges"] if r["kind"] == "gil"]) == 3
    output = tmp_path / "analysis"
    _write_report(report, output)
    assert (output / "gil_timeline.png").is_file()
    assert (output / "threads.csv").is_file()


def test_compare_rejects_different_workloads():
    baseline = {
        "schema_version": 2,
        "benchmark": {"case": {"model": "a"}},
        "steps": [{}],
    }
    candidate = {
        "schema_version": 2,
        "benchmark": {"case": {"model": "b"}},
        "steps": [{}],
    }
    with pytest.raises(ValueError, match="workload metadata differs"):
        compare(baseline, candidate)


def test_compare_keeps_profiled_costs_separate_from_speed_claims():
    def report(neuron_ms):
        return {
            "schema_version": 2,
            "global_steps": [],
            "device_summary": [
                {
                    "device_key": "gpu:0",
                    "gpu_busy_union_ms": neuron_ms,
                    "gpu_idle_within_span_ms": 2,
                    "communication_union_ms": 0,
                    "compute_communication_overlap_ms": 0,
                }
            ],
            "steps": [
                {
                    "cpu_range_ms": 10,
                    "gpu_span_ms": 8,
                    "gpu_busy_union_ms": 6,
                    "gpu_idle_within_span_ms": 2,
                    "cuda_api_union_ms": 1,
                    "categories_ms": {"neuron": neuron_ms},
                }
            ],
        }

    result = compare(report(3), report(2))
    assert result["devices"][0]["metrics"]["gpu_busy_union_ms"]["delta"] == -1
    assert "unprofiled" in result["note"]
    assert result["workload_verified"] is False


def test_compare_allows_execution_and_precision_controls():
    workload = {
        "model": "snn",
        "phase": "inference",
        "T": 4,
        "batch_size": 1,
        "image_size": 32,
        "num_classes": 10,
        "seed": 42,
        "world_size": 1,
        "parallelism": "single",
        "microbatches": 1,
    }
    step = {
        "cpu_range_ms": 10,
        "gpu_span_ms": 8,
        "gpu_busy_union_ms": 6,
        "gpu_idle_within_span_ms": 2,
        "cuda_api_union_ms": 1,
        "categories_ms": {"neuron": 3},
    }
    baseline = {
        "schema_version": 2,
        "device_summary": [],
        "global_steps": [],
        "benchmark": {"case": {**workload, "precision": "fp32", "execution": "eager"}},
        "steps": [step],
    }
    candidate = {
        "schema_version": 2,
        "device_summary": [],
        "global_steps": [],
        "benchmark": {
            "case": {**workload, "precision": "fp16", "execution": "compile"}
        },
        "steps": [step],
    }
    result = compare(baseline, candidate)
    assert result["workload_verified"] is True
    assert result["candidate_case"]["precision"] == "fp16"


def test_empty_service_trace_explains_missing_nvtx(tmp_path):
    path = tmp_path / "service.sqlite"
    with sqlite3.connect(path):
        pass
    with pytest.raises(ValueError, match="no CUDA"):
        analyze(path)


@pytest.mark.parametrize(
    ("kernel", "category"),
    [
        ("cudnn::bn_fw_tr_1C11_kernel_NCHW", "normalization"),
        ("cudnn::winograd_nonfused::winogradForwardOutput4x4", "conv_gemm"),
        ("fft2d_r2c_32x32(cudnn::reduced_divisor)", "conv_gemm"),
        ("cutlass::Kernel2", "conv_gemm"),
        ("cutlass__5x_cudnn::Kernel<cutlass_tensorop_f16_fprop>", "conv_gemm"),
        ("cudnn::engines_precompiled::nhwcToNchwKernel", "layout_copy"),
        ("triton_poi_fused_add_view_16", "elementwise"),
        ("user_fft2d", "unknown"),
    ],
)
def test_native_kernel_categories_keep_unknowns(kernel, category):
    assert _category(kernel) == category


def test_module_ranges_include_stateful_snn_leaf(monkeypatch, tmp_path):
    ranges = []
    monkeypatch.setattr(nsys.torch.cuda.nvtx, "range_push", ranges.append)
    monkeypatch.setattr(nsys.torch.cuda.nvtx, "range_pop", lambda: None)
    model = torch.nn.Sequential(neuron.LIFNode())
    with nsys.module_ranges(model, tmp_path / "modules.jsonl"):
        model(torch.rand(2, 3))
    assert "module:0" in ranges
    record = json.loads((tmp_path / "modules.jsonl").read_text().splitlines()[0])
    assert (record["step_mode"], record["backend"]) == ("s", "torch")
    assert record["value"][0]["shape"] == [2, 3]


def test_module_ranges_remove_hooks_after_registration_failure(monkeypatch, tmp_path):
    layer = torch.nn.Linear(2, 2)

    def fail_registration(*args, **kwargs):
        raise RuntimeError("registration")

    monkeypatch.setattr(layer, "register_forward_hook", fail_registration)
    with pytest.raises(RuntimeError, match="registration"):
        with nsys.module_ranges(torch.nn.Sequential(layer), tmp_path / "modules.jsonl"):
            pass
    assert not layer._forward_pre_hooks
