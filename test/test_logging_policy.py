from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger as loguru_logger

from spikingjelly.activation_based.functional import net_config
from spikingjelly.activation_based.op_counter.compute_energy import (
    ComputeEnergyProfiler,
)
from spikingjelly.logger import logger


def _load_checker():
    checker_path = (
        Path(__file__).resolve().parent.parent / "tools/check_logging_policy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "logging_policy_checker", checker_path
    )
    checker = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(checker)
    return checker


def test_package_exports_loguru_logger():
    assert logger is loguru_logger


def test_package_does_not_reexport_logger_object():
    import importlib
    import types

    import spikingjelly

    logger_module = importlib.import_module("spikingjelly.logger")
    assert isinstance(spikingjelly.logger, types.ModuleType)
    assert spikingjelly.logger is logger_module
    assert logger_module.logger is logger


def test_import_is_silent_and_does_not_change_stdlib_root():
    script = """
import contextlib
import io
import logging
root = logging.getLogger()
before = (root.level, tuple(root.handlers))
stdout = io.StringIO()
stderr = io.StringIO()
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    import spikingjelly
assert stdout.getvalue() == ""
assert stderr.getvalue() == ""
assert before == (root.level, tuple(root.handlers))
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_import_preserves_user_loguru_sink():
    script = """
import io
from loguru import logger
output = io.StringIO()
sink_id = logger.add(output, format="{message}")
import spikingjelly
logger.info("user-sink-still-active")
logger.remove(sink_id)
assert output.getvalue().strip() == "user-sink-still-active"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_package_is_disabled_until_enabled():
    script = """
import io
import torch.nn as nn
from loguru import logger
from spikingjelly.activation_based.functional import net_config
output = io.StringIO()
sink_id = logger.add(
    output,
    format="{message}",
    filter=lambda record: record["name"].startswith("spikingjelly"),
)
module = nn.Module()
module.backend = "torch"
module.supported_backends = ("torch",)
net_config.set_backend(module, "unsupported")
assert output.getvalue() == ""
logger.enable("spikingjelly")
net_config.set_backend(module, "unsupported")
assert "unsupported" in output.getvalue()
logger.disable("spikingjelly")
before = output.getvalue()
net_config.set_backend(module, "unsupported")
assert output.getvalue() == before
logger.remove(sink_id)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_net_config_warning_uses_package_logger(loguru_records):
    import torch.nn as nn

    module = nn.Module()
    module.backend = "torch"
    module.supported_backends = ("torch",)
    net_config.set_backend(module, "unsupported")
    assert loguru_records
    assert all(record["name"].startswith("spikingjelly") for record in loguru_records)


def test_compute_energy_summary_is_emitted_once(loguru_records):
    import torch

    model = torch.nn.Linear(2, 2, bias=False)
    profiler = ComputeEnergyProfiler()
    with profiler:
        model(torch.ones(1, 2))
    profiler.get_report()
    profiler.get_report()
    summaries = [
        record
        for record in loguru_records
        if record["extra"].get("event") == "op_counter_summary"
    ]
    assert len(summaries) == 1
    assert summaries[0]["extra"]["total_operations"] >= 0


def test_data_preprocess_summary_reports_completed_tasks(loguru_records):
    from spikingjelly.datasets.base import NeuromorphicDatasetBuilder

    processed = []
    NeuromorphicDatasetBuilder._run_preprocess_tasks(
        object(), processed.append, ((1,), (2,), (3,))
    )
    summaries = [
        record
        for record in loguru_records
        if record["extra"].get("event") == "data_preprocess_summary"
    ]
    assert sorted(processed) == [1, 2, 3]
    assert len(summaries) == 1
    assert summaries[0]["extra"]["task_count"] == 3
    assert summaries[0]["extra"]["duration_seconds"] >= 0


def test_precision_prepare_emits_one_lifecycle_summary(loguru_records):
    import torch

    from spikingjelly.activation_based.precision import prepare_model_for_precision

    prepare_model_for_precision(torch.nn.Linear(2, 2), "cpu", "fp32")
    summaries = [
        record
        for record in loguru_records
        if record["extra"].get("event") == "precision_prepare_summary"
    ]
    assert len(summaries) == 1


def test_distributed_configuration_emits_one_lifecycle_summary(loguru_records):
    import torch

    from spikingjelly.activation_based.distributed.config import SNNDistributedConfig
    from spikingjelly.activation_based.distributed.execution import (
        configure_snn_distributed,
    )

    configure_snn_distributed(torch.nn.Linear(2, 2), SNNDistributedConfig())
    summaries = [
        record
        for record in loguru_records
        if record["extra"].get("event") == "distributed_configuration_summary"
    ]
    assert len(summaries) == 1
    assert summaries[0]["extra"]["rank"] == 0
    assert summaries[0]["extra"]["world_size"] == 1
    assert summaries[0]["extra"]["distributed_backend"] is None
    assert summaries[0]["extra"]["mesh_shape"] is None


def test_metric_logger_formats_progress_with_loguru_arguments(loguru_records):
    from spikingjelly.activation_based.model.tv_ref_classify.utils import MetricLogger

    list(MetricLogger().log_every(["sample"], 1, "Train"))
    messages = [record["message"] for record in loguru_records]
    assert any("Train" in message and "[0/1]" in message for message in messages)


def test_serialized_sink_contains_standard_and_event_fields(tmp_path):
    import torch

    from spikingjelly.activation_based.precision import prepare_model_for_precision

    path = tmp_path / "spikingjelly.jsonl"
    sink_id = logger.add(
        path,
        serialize=True,
        diagnose=False,
        filter=lambda record: record["name"].startswith("spikingjelly"),
    )
    logger.enable("spikingjelly")
    try:
        prepare_model_for_precision(torch.nn.Linear(2, 2), "cpu", "fp32")
    finally:
        logger.disable("spikingjelly")
        logger.remove(sink_id)

    payload = json.loads(path.read_text().splitlines()[0])
    record = payload["record"]
    assert record["extra"]["event"] == "precision_prepare_summary"
    for field in ("time", "level", "name", "function", "line", "process", "thread"):
        assert field in record


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import logging\n", "stdlib logging import"),
        ("from loguru import logger\nlogger.info('x')\n", "must be imported"),
        (
            "from spikingjelly.logger import logger\nlogger.info('value=%s', value)\n",
            "percent placeholder",
        ),
        (
            "from spikingjelly.logger import logger\nlogger.info('value=%08.3f', value)\n",
            "percent placeholder",
        ),
        ("logging.warning('not allowed')\n", "stdlib logging reference"),
        (
            "from spikingjelly.logger import logger\nlogger.info('value={}')\n",
            "placeholder count",
        ),
        (
            "from spikingjelly.logger import logger\nlogger.info(f'value={value}')\n",
            "eager formatting",
        ),
        (
            "from spikingjelly.logger import logger\nlogger.info('work_summary result={}', value)\n",
            "lifecycle summary requires event",
        ),
        (
            "from spikingjelly.logger import logger\nlogger.bind(event='Bad Event').info('done')\n",
            "literal snake_case",
        ),
        ("import builtins\nbuiltins.print('not allowed')\n", "builtins.print"),
    ],
)
def test_policy_checker_rejects_invalid_logging(tmp_path, source, expected):
    path = tmp_path / "invalid_logging.py"
    path.write_text(source, encoding="utf-8")
    violations = _load_checker().check(tmp_path)
    assert any(expected in violation for violation in violations)


def test_policy_checker_accepts_bound_loguru_summary(tmp_path):
    path = tmp_path / "valid_logging.py"
    path.write_text(
        "from spikingjelly.logger import logger\n"
        "logger.bind(event='work_summary', result=value).info(\n"
        "    'work_summary result={}', value\n"
        ")\n",
        encoding="utf-8",
    )
    assert _load_checker().check(tmp_path) == []


def test_policy_checker_exists():
    assert Path("tools/check_logging_policy.py").is_file()
