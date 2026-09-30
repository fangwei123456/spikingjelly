from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest

from benchmark.analyze_nsys_snn import analyze, compare, _gpu_metrics, _write_report
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
            label = "sj.step:training:0|sj:" + json.dumps(
                {"rank": rank, "world_size": 2}
            )
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES (0,10000000,?,NULL,?,59,0)",
                (label, gid(pid, 1)),
            )
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES (1000000,9000000,?,NULL,?,59,0)",
                (
                    "forward|sj:" + json.dumps({"stage": rank, "microbatch": 0}),
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
            "INSERT INTO NVTX_EVENTS VALUES (0,10000000,'sj.step:training:1',NULL,?,59,0)",
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
        db.execute("UPDATE NVTX_EVENTS SET end=NULL WHERE text LIKE 'sj.step%'")
    report = analyze(trace)
    assert not report["global_steps"][0]["complete"]
    assert report["global_steps"][0]["missing_ranks"] == [1]
    assert report["coverage"]["incomplete_steps"] == 1


def test_compare_requires_schema_v2(trace):
    with pytest.raises(ValueError, match="re-run analyze"):
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
    assert json.loads(labels[0].split("|sj:")[1]) == {"rank": 1, "world_size": 2}
    assert json.loads(labels[1].split("|sj:")[1]) == {"stage": 1, "microbatch": 3}


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
