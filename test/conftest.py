"""Pytest session hooks for SpikingJelly.

Adds the repository root to PYTHONPATH during pytest startup so that
subprocess invocations of CLI scripts can ``import benchmark.*`` and
similar repo-local packages without needing each test to set its own
``env=`` argument. The repo root is appended to any existing
PYTHONPATH rather than replacing it.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in (existing, str(_REPO_ROOT)) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
