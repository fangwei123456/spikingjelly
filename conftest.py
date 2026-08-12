"""Pytest session hooks for SpikingJelly.

Prepends the repository root to PYTHONPATH during pytest startup so that
subprocess invocations of CLI scripts can ``import benchmark.*`` and
similar repo-local packages without needing each test to set its own
``env=`` argument.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent


def pytest_configure(config):
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in (str(_REPO_ROOT), existing) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


@pytest.fixture
def loguru_records():
    from spikingjelly.logger import logger

    records = []
    sink_id = logger.add(
        lambda message: records.append(message.record),
        level="DEBUG",
        filter=lambda record: record["name"].startswith("spikingjelly"),
    )
    logger.enable("spikingjelly")
    try:
        yield records
    finally:
        logger.disable("spikingjelly")
        logger.remove(sink_id)
