from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MCore SNN run metrics.")
    parser.add_argument("runs", nargs="+", type=Path)
    return parser.parse_args()


def _main() -> None:
    rows = []
    for run in _parse_args().runs:
        path = run / "metrics.json" if run.is_dir() else run
        rows.append((run.name, json.loads(path.read_text(encoding="utf-8"))))
    names = sorted({name for _, metrics in rows for name in metrics})
    print("run\t" + "\t".join(names))
    for run, metrics in rows:
        print(run + "\t" + "\t".join(str(metrics.get(name, "-")) for name in names))


if __name__ == "__main__":
    _main()
