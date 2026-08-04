"""Microbenchmark the cost of filtered and active logging calls."""

from __future__ import annotations

import logging
import time

from spikingjelly.logger import logger


class _FormattingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.format(record)


def _measure(logger: logging.Logger, calls: int) -> float:
    started = time.perf_counter()
    for index in range(calls):
        logger.info("benchmark index=%s", index)
    return time.perf_counter() - started


def _measure_baseline(calls: int) -> float:
    started = time.perf_counter()
    for _ in range(calls):
        pass
    return time.perf_counter() - started


def main() -> None:
    calls = 1_000_000
    logger.propagate = False

    elapsed = _measure_baseline(calls)
    print(
        f"baseline: total_seconds={elapsed:.6f} ns_per_call={elapsed / calls * 1e9:.1f}"
    )

    scenarios = {
        "logger_filtered": (logging.WARNING, []),
        "null_handler": (logging.INFO, [logging.NullHandler()]),
        "formatted_handler": (logging.INFO, [_FormattingHandler()]),
        "handler_filtered": (logging.INFO, [logging.NullHandler()]),
    }
    for name, (level, handlers) in scenarios.items():
        logger.handlers.clear()
        logger.setLevel(level)
        for handler in handlers:
            handler.setLevel(logging.WARNING if name == "handler_filtered" else level)
            logger.addHandler(handler)
        elapsed = _measure(logger, calls)
        print(
            f"{name}: total_seconds={elapsed:.6f} ns_per_call={elapsed / calls * 1e9:.1f}"
        )


if __name__ == "__main__":
    main()
