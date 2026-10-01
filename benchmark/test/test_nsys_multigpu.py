from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from benchmark.analyze_nsys_snn import (
    analyze,
    compare,
    _decode_label,
    _gpu_metrics,
    _write_report,
)
from spikingjelly import nsys


def gid(pid, tid=0):
    return (1 << 48) | (pid << 24) | tid


@pytest.fixture
def trace(tmp_path):
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT,
                textId INTEGER, globalTid INTEGER, eventType INTEGER, domainId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER,
                correlationId INTEGER, globalTid INTEGER, nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER,
                correlationId INTEGER, demangledName INTEGER, globalPid INTEGER,
                deviceId INTEGER, contextId INTEGER, streamId INTEGER, graphNodeId INTEGER);
            CREATE TABLE TARGET_INFO_CUDA_DEVICE (gpuId INTEGER, cudaId INTEGER,
                pid INTEGER, uuid TEXT);
            CREATE TABLE CUDA_GRAPH_NODE_EVENTS (start INTEGER, globalTid INTEGER,
                graphNodeId INTEGER, originalGraphNodeId INTEGER);
            INSERT INTO StringIds VALUES (1, 'lif_forward'), (2, 'ncclDevKernel_AllReduce'),
                (3, 'cudaLaunchKernel'), (4, 'cudaStreamSynchronize');
            INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES
                (0, 0, 101, 'GPU-A'), (1, 0, 102, 'GPU-B');
        """)
        for rank, pid in enumerate((101, 102)):
            label = f"training step 0 | rank {rank} of 2"
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES (0,10000000,?,NULL,?,59,0)",
                (label, gid(pid, 1)),
            )
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES (1000000,9000000,?,NULL,?,59,0)",
                (
                    f"forward | stage {rank} | microbatch 0",
                    gid(pid, 1),
                ),
            )
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2000000,2100000,7,?,3)",
                (gid(pid, 2),),
            )
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (3000000,6000000,7,1,?,?,1,7,NULL)",
                (gid(pid), rank),
            )
    return path


def test_rank_steps_and_device_streams_are_independent(trace, tmp_path):
    report = analyze(trace)
    assert report["schema_version"] == 2
    assert len(report["selection"]["step_ids"]) == 2
    assert report["global_steps"][0]["complete"]
    assert report["global_steps"][0]["cpu_envelope_ms"] == 10
    assert [s["gpu_busy_union_ms"] for s in report["steps"]] == [3, 3]
    assert {e["device_key"] for e in report["gpu_events"]} == {"GPU-A", "GPU-B"}
    assert [e["stage"] for e in report["gpu_events"]] == [0, 1]
    assert report["coverage"]["assigned_gpu_events"] == 2
    assert analyze(trace, rank=1)["selection"]["process_ids"] == [gid(102)]
    _write_report(report, tmp_path / "out")
    assert (tmp_path / "out/pipeline_timeline.png").is_file()


def test_overlap_is_per_device_and_not_summed_latency(trace):
    with sqlite3.connect(trace) as db:
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (3000000,3100000,8,?,3)",
            (gid(101, 2),),
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (4000000,8000000,8,2,?,0,1,8,NULL)",
            (gid(101),),
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (4000000,4100000,9,?,3)",
            (gid(101, 2),),
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (11000000,12000000,9,1,?,0,1,7,NULL)",
            (gid(101),),
        )
    report = analyze(trace)
    metrics = report["steps"][0]["devices"][0]
    assert metrics["gpu_busy_union_ms"] == 6
    assert metrics["compute_communication_overlap_ms"] == 2
    assert report["global_steps"][0]["gpu_end_ns"] == 12_000_000
    assert report["selection"]["end_ns"] == 12_000_000
    assert _gpu_metrics([])["gpu_busy_union_ms"] == 0


def test_same_process_multigpu_has_no_combined_busy_scalar(trace):
    with sqlite3.connect(trace) as db:
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (3000000,6000000,7,1,?,1,2,7,NULL)",
            (gid(101),),
        )
    step = analyze(trace)["steps"][0]
    assert len(step["devices"]) == 2
    assert step["gpu_busy_union_ms"] is None


def test_orphans_and_ambiguous_threads_are_retained(trace):
    with sqlite3.connect(trace) as db:
        db.execute(
            "INSERT INTO NVTX_EVENTS VALUES (0,10000000,'training step 1',NULL,?,59,0)",
            (gid(101, 3),),
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (4000000,5000000,NULL,1,?,0,1,7,NULL)",
            (gid(101),),
        )
    report = analyze(trace)
    assert report["coverage"]["raw_gpu_events"] == 3
    assert report["coverage"]["assigned_gpu_events"] == 1
    assert report["coverage"]["unassigned_reasons"] == {
        "no_unique_step": 1,
        "missing_correlation": 1,
    }
    assert len(report["gpu_events"]) == 3


def test_graph_node_ids_are_process_scoped(trace):
    with sqlite3.connect(trace) as db:
        for pid, stage, phase in ((101, 0, "forward"), (102, 1, "backward")):
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES (-3000000,-1000000,?,NULL,?,59,0)",
                (phase, gid(pid, 1)),
            )
            db.execute(
                "INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (-2000000,?,1,NULL)",
                (gid(pid, 1),),
            )
            db.execute(
                "INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (-500000,?,2,1)",
                (gid(pid, 1),),
            )
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=2")
    report = analyze(trace)
    assert [e["graph_stage"] for e in report["gpu_events"]] == ["forward", "backward"]


def test_uninstrumented_trace_and_time_window(trace):
    with sqlite3.connect(trace) as db:
        db.execute("DROP TABLE NVTX_EVENTS")
    report = analyze(trace, time_range_ms=(0, 2))
    assert not report["steps"]
    assert not report["global_steps"]
    assert (
        report["coverage"]["raw_gpu_events"]
        == report["coverage"]["unassigned_gpu_events"]
        == 2
    )
    assert report["selection"]["end_ns"] - report["selection"]["start_ns"] == 2_000_000


def test_missing_rank_and_open_step_are_incomplete(trace):
    with sqlite3.connect(trace) as db:
        db.execute("DELETE FROM NVTX_EVENTS WHERE globalTid=?", (gid(102, 1),))
        db.execute("UPDATE NVTX_EVENTS SET end=NULL WHERE text LIKE 'training step %'")
    report = analyze(trace)
    assert not report["global_steps"][0]["complete"]
    assert report["global_steps"][0]["missing_ranks"] == [1]
    assert report["coverage"]["incomplete_steps"] == 1
    assert report["steps"][0]["cpu_range_ms"] is None
    assert report["global_steps"][0]["cpu_envelope_ms"] is None
    assert report["global_steps"][0]["end_skew_ms"] is None
    with sqlite3.connect(trace) as db:
        db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
    nvtx_only = analyze(trace)
    assert nvtx_only["steps"][0]["end_ns"] == nvtx_only["capture_end_ns"] == 9_000_000


@pytest.mark.parametrize("start_ns", [0, 20_000_000])
@pytest.mark.parametrize("cuda_events", [False, True])
def test_open_step_bounds_without_a_later_cuda_event(trace, start_ns, cuda_events):
    with sqlite3.connect(trace) as db:
        db.execute("DELETE FROM NVTX_EVENTS")
        db.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?,NULL,?,NULL,?,59,0)",
            (start_ns, "training step 0 | rank 0 of 1", gid(101, 1)),
        )
        if not cuda_events:
            db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME")
            db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
    report = analyze(trace)
    step = report["steps"][0]
    assert not step["complete"]
    assert step["cpu_range_ms"] is None
    assert report["capture_start_ns"] <= start_ns <= report["capture_end_ns"]
    assert step["start_ns"] <= step["end_ns"]
    assert report["selection"]["start_ns"] <= report["selection"]["end_ns"]
    assert report["global_steps"][0]["cpu_envelope_ms"] is None


def test_graph_example_rejects_unsupported_validation(monkeypatch, tmp_path, capsys):
    from benchmark import nsys_multigpu_example

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "example",
            "--parallel",
            "graph",
            "--validate",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        nsys_multigpu_example.main()
    assert error.value.code == 2
    assert "--validate requires" in capsys.readouterr().err


def test_compare_requires_schema_v2(trace):
    with pytest.raises(ValueError, match="capture with current markers"):
        compare({"schema_version": 1}, analyze(trace))


def test_shell_analyzes_unmarked_export_when_optional_stats_are_missing(
    trace, tmp_path
):
    with sqlite3.connect(trace) as db:
        db.execute("DROP TABLE NVTX_EVENTS")
    fake = tmp_path / "nsys"
    fake.write_text(
        "#!/usr/bin/env python\n"
        "import os, shutil, sys\n"
        "if sys.argv[1] != 'export': sys.exit(1)\n"
        "dest = next(x.split('=',1)[1] for x in sys.argv if x.startswith('--output='))\n"
        "shutil.copyfile(os.environ['TRACE_DB'], dest)\n"
    )
    fake.chmod(0o755)
    output = tmp_path / "unmarked"
    result = subprocess.run(
        [
            "bash",
            str(Path(__file__).resolve().parents[1] / "nsys_snn.sh"),
            "analyze",
            str(tmp_path / "input.nsys-rep"),
            str(output),
            "--pid",
            "101",
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "TRACE_DB": str(trace),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "summary.json").read_text())
    assert not report["steps"]
    assert report["selection"]["process_ids"] == [gid(101)]
    assert report["coverage"]["raw_gpu_events"] == 2


def test_metadata_markers_roundtrip(monkeypatch):
    labels = []
    monkeypatch.setattr(nsys.torch.cuda.nvtx, "range_push", labels.append)
    monkeypatch.setattr(nsys.torch.cuda.nvtx, "range_pop", lambda: None)
    with nsys.step(3, "training", True, rank=1, world_size=2):
        with nsys.region("backward", True, stage=1, microbatch=3):
            pass
    assert labels == [
        "training step 3 | rank 1 of 2",
        "backward | stage 1 | microbatch 3",
    ]
    assert _decode_label(labels[0]) == ("training step 3", {"rank": 1, "world_size": 2})
    assert _decode_label(labels[1]) == ("backward", {"stage": 1, "microbatch": 3})


@pytest.mark.parametrize(
    "name,metadata,expected",
    [
        ("forward", {}, "forward"),
        ("forward", {"stage": 0}, "forward | stage 0"),
        ("forward", {"microbatch": 12}, "forward | microbatch 12"),
        ("custom | layout", {"stage": 2}, "custom | layout | stage 2"),
    ],
)
def test_region_labels_preserve_names_and_optional_metadata(
    monkeypatch, name, metadata, expected
):
    labels = []
    monkeypatch.setattr(nsys.torch.cuda.nvtx, "range_push", labels.append)
    monkeypatch.setattr(nsys.torch.cuda.nvtx, "range_pop", lambda: None)
    with nsys.region(name, True, **metadata):
        pass
    assert labels == [expected]
    assert _decode_label(labels[0]) == (name, metadata)


@pytest.mark.parametrize("rank,world_size", [(None, None), (0, None), (0, 2), (11, 12)])
def test_step_labels_preserve_zero_based_rank(monkeypatch, rank, world_size):
    labels = []
    monkeypatch.setattr(nsys.torch.cuda.nvtx, "range_push", labels.append)
    monkeypatch.setattr(nsys.torch.cuda.nvtx, "range_pop", lambda: None)
    with nsys.step(1, "inference", True, rank=rank, world_size=world_size):
        pass
    metadata = {
        key: value
        for key, value in {"rank": rank, "world_size": world_size}.items()
        if value is not None
    }
    assert _decode_label(labels[0]) == ("inference step 1", metadata)


def test_device_capture_cleans_up_partial_start(monkeypatch):
    current = [9]
    stopped = []

    @contextmanager
    def device(index):
        old = current[0]
        current[0] = index
        try:
            yield
        finally:
            current[0] = old

    def start():
        if current[0] == 1:
            raise RuntimeError("start failed")

    monkeypatch.setattr(nsys.torch.cuda, "device", device)
    monkeypatch.setattr(nsys.torch.cuda.profiler, "start", start)
    monkeypatch.setattr(
        nsys.torch.cuda.profiler, "stop", lambda: stopped.append(current[0])
    )
    with pytest.raises(RuntimeError, match="start failed"):
        with nsys.capture(True, devices=[0, 1]):
            pass
    assert stopped == [0]
    assert current == [9]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("devices", [[], [0, 0], [-1]])
def test_capture_rejects_invalid_devices_in_both_modes(enabled, devices):
    with pytest.raises(ValueError, match="unique nonnegative"):
        with nsys.capture(enabled, devices=devices):
            pytest.fail("invalid devices entered the capture scope")
