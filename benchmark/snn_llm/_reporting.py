"""Shared report writing for private SNN-LLM benchmark runners."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional


def write_report(output_dir: Path, report: Mapping[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "report.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}.")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir, prefix=".report.json.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite {target}.") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_rank0_report(
    output_dir: Path, report: Mapping[str, object], *, rank: int
) -> Optional[Path]:
    if rank != 0:
        return None
    return write_report(output_dir, report)
