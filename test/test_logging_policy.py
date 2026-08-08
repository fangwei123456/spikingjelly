from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from spikingjelly.activation_based.functional import net_config
from spikingjelly.activation_based.op_counter.compute_energy import (
    ComputeEnergyProfiler,
)
from spikingjelly.logger import logger


def test_package_logger_is_named_and_does_not_configure_root():
    assert logger.name == "spikingjelly"
    assert any(isinstance(handler, logging.NullHandler) for handler in logger.handlers)


def test_package_does_not_reexport_logger_object():
    import importlib
    import types

    import spikingjelly

    logger_module = importlib.import_module("spikingjelly.logger")
    assert isinstance(spikingjelly.logger, types.ModuleType)
    assert spikingjelly.logger is logger_module
    assert logger_module.logger is logger


def test_import_does_not_change_root_configuration():
    script = """
import logging
root = logging.getLogger()
def snapshot():
    return (
        root.level,
        tuple(
            (
                type(h).__name__,
                h.level,
                h.formatter._fmt if h.formatter is not None else None,
            )
            for h in root.handlers
        ),
    )
before = snapshot()
import spikingjelly
after = snapshot()
if before != after:
    raise SystemExit(f"root logging configuration changed: {before!r} != {after!r}")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_net_config_warning_uses_package_logger(caplog):
    import torch.nn as nn

    module = nn.Module()
    module.backend = "torch"
    module.supported_backends = ("torch",)
    with caplog.at_level(logging.WARNING, logger="spikingjelly"):
        net_config.set_backend(module, "unsupported")
    assert caplog.records
    assert all(record.name == "spikingjelly" for record in caplog.records)


def test_compute_energy_summary_is_emitted_once(caplog):
    import torch

    model = torch.nn.Linear(2, 2, bias=False)
    profiler = ComputeEnergyProfiler()
    with caplog.at_level(logging.INFO, logger="spikingjelly"):
        with profiler:
            model(torch.ones(1, 2))
        profiler.get_report()
        profiler.get_report()
    summaries = [
        record
        for record in caplog.records
        if record.__dict__.get("event") == "op_counter_summary"
    ]
    assert len(summaries) == 1


def test_precision_prepare_emits_one_lifecycle_summary(caplog):
    import torch

    from spikingjelly.activation_based.precision import prepare_model_for_precision

    with caplog.at_level(logging.INFO, logger="spikingjelly"):
        prepare_model_for_precision(torch.nn.Linear(2, 2), "cpu", "fp32")
    summaries = [
        record
        for record in caplog.records
        if record.getMessage().startswith("precision_prepare_summary")
    ]
    assert len(summaries) == 1
    assert all(record.name == "spikingjelly" for record in summaries)


def test_distributed_configuration_emits_one_lifecycle_summary(caplog):
    import torch

    from spikingjelly.activation_based.distributed.config import (
        SNNDistributedConfig,
    )
    from spikingjelly.activation_based.distributed.execution import (
        configure_snn_distributed,
    )

    with caplog.at_level(logging.INFO, logger="spikingjelly"):
        configure_snn_distributed(torch.nn.Linear(2, 2), SNNDistributedConfig())
    summaries = [
        record
        for record in caplog.records
        if record.getMessage().startswith("distributed_configuration_summary")
    ]
    assert len(summaries) == 1


def test_metric_logger_formats_progress_with_logging_arguments(caplog):
    from spikingjelly.activation_based.model.tv_ref_classify.utils import (
        MetricLogger,
    )

    with caplog.at_level(logging.INFO, logger="spikingjelly"):
        list(MetricLogger().log_every(["sample"], 1, "Train"))

    messages = [record.getMessage() for record in caplog.records]
    assert any("Train" in message and "[0/1]" in message for message in messages)


def test_policy_checker_rejects_nested_get_logger(tmp_path):
    import importlib.util

    checker_path = (
        Path(__file__).resolve().parent.parent / "tools/check_logging_policy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "logging_policy_checker", checker_path
    )
    checker = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(checker)

    source = tmp_path / "nested_logger.py"
    source.write_text(
        'import logging\nlogging.getLogger(__name__).info("not allowed")\n',
        encoding="utf-8",
    )
    violations = checker.check(tmp_path)
    assert any("logger must be imported" in violation for violation in violations)


def test_policy_checker_rejects_standalone_get_logger(tmp_path):
    import importlib.util

    checker_path = (
        Path(__file__).resolve().parent.parent / "tools/check_logging_policy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "logging_policy_checker", checker_path
    )
    checker = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(checker)

    source = tmp_path / "standalone_logger.py"
    source.write_text(
        "import logging\n"
        'logging.getLogger("spikingjelly")\n'
        'logger = logging.getLogger("wrong_name")\n',
        encoding="utf-8",
    )
    violations = checker.check(tmp_path)
    assert any("logger must be imported" in violation for violation in violations)
    assert any("non-package logger name" in violation for violation in violations)


def test_policy_checker_rejects_nested_null_handler(tmp_path):
    import importlib.util

    checker_path = (
        Path(__file__).resolve().parent.parent / "tools/check_logging_policy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "logging_policy_checker", checker_path
    )
    checker = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(checker)

    package = tmp_path / "nested" / "__init__.py"
    package.parent.mkdir()
    package.write_text("import logging\nlogging.NullHandler()\n", encoding="utf-8")
    violations = checker.check(tmp_path)
    assert any("logging configuration call" in violation for violation in violations)


def test_policy_checker_rejects_builtins_print(tmp_path):
    import importlib.util

    checker_path = (
        Path(__file__).resolve().parent.parent / "tools/check_logging_policy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "logging_policy_checker", checker_path
    )
    checker = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(checker)

    source = tmp_path / "builtins_print.py"
    source.write_text(
        "import builtins\nbuiltins.print('not allowed')\n", encoding="utf-8"
    )
    violations = checker.check(tmp_path)
    assert any("builtins.print" in violation for violation in violations)


def test_policy_checker_rejects_eager_logging_formatting(tmp_path):
    import importlib.util

    checker_path = (
        Path(__file__).resolve().parent.parent / "tools/check_logging_policy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "logging_policy_checker", checker_path
    )
    checker = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(checker)

    source = tmp_path / "eager_logging.py"
    source.write_text(
        "from spikingjelly.logger import logger\n"
        'logger.info("value=%s" % value)\n'
        'logger.warning("prefix " + value)\n',
        encoding="utf-8",
    )
    violations = checker.check(tmp_path)
    assert sum("manual string formatting" in violation for violation in violations) == 2


@pytest.mark.parametrize("path", [Path("tools/check_logging_policy.py")])
def test_policy_checker_exists(path):
    assert path.is_file()
