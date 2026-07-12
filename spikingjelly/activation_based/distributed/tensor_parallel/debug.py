from __future__ import annotations

import threading

import torch


_TP_COMMUNICATION_DEBUG_ENABLED = False
_TP_COMMUNICATION_DEBUG_STATS = {
    "all_reduce_calls": 0,
    "all_reduce_bytes": 0,
}
_TP_COMMUNICATION_DEBUG_LOCK = threading.Lock()


def enable_tp_communication_debug(enabled: bool = True) -> None:
    """Enable or disable tensor-parallel communication counters.

    .. admonition:: Chinese

        启用或关闭张量并行通信计数器。

    :param enabled: Whether debug counting is enabled.
    :type enabled: bool
    """
    global _TP_COMMUNICATION_DEBUG_ENABLED
    _TP_COMMUNICATION_DEBUG_ENABLED = bool(enabled)


def reset_tp_communication_debug_stats() -> None:
    """Reset tensor-parallel communication counters to zero.

    .. admonition:: Chinese

        将张量并行通信调试计数器清零。
    """
    with _TP_COMMUNICATION_DEBUG_LOCK:
        for key in _TP_COMMUNICATION_DEBUG_STATS:
            _TP_COMMUNICATION_DEBUG_STATS[key] = 0


def get_tp_communication_debug_stats() -> dict[str, int]:
    """Return a snapshot of tensor-parallel communication counters.

    .. admonition:: Chinese

        返回张量并行通信调试计数器的快照。

    :return: Counter names mapped to integer values.
    :rtype: dict[str, int]
    """
    with _TP_COMMUNICATION_DEBUG_LOCK:
        return {key: int(value) for key, value in _TP_COMMUNICATION_DEBUG_STATS.items()}


def _record_tp_all_reduce(tensor: torch.Tensor) -> None:
    if not _TP_COMMUNICATION_DEBUG_ENABLED:
        return
    with _TP_COMMUNICATION_DEBUG_LOCK:
        _TP_COMMUNICATION_DEBUG_STATS["all_reduce_calls"] += 1
        _TP_COMMUNICATION_DEBUG_STATS["all_reduce_bytes"] += int(
            tensor.numel() * tensor.element_size()
        )
