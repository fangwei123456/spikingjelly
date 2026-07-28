from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import torch


_MAX_ENTRIES = 64
_CACHE_PID = os.getpid()
_COMPILED_GRAPHS: OrderedDict[tuple[Any, ...], Callable] = OrderedDict()
_CACHE_LOCK = threading.RLock()


def _reset_after_fork() -> None:
    global _CACHE_LOCK, _CACHE_PID
    _CACHE_LOCK = threading.RLock()
    _COMPILED_GRAPHS.clear()
    _CACHE_PID = os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _reset_if_forked() -> None:
    if os.getpid() != _CACHE_PID:
        _reset_after_fork()


def clear() -> None:
    _reset_if_forked()
    with _CACHE_LOCK:
        _COMPILED_GRAPHS.clear()


def info() -> dict[str, int]:
    _reset_if_forked()
    with _CACHE_LOCK:
        return {
            "entries": len(_COMPILED_GRAPHS),
            "max_entries": _MAX_ENTRIES,
            "pid": _CACHE_PID,
        }


def surrogate_key(surrogate_function: Callable) -> tuple[Any, ...] | None:
    if any(
        getattr(surrogate_function, name, None)
        for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")
    ):
        return None
    surrogate_type = type(surrogate_function)
    if surrogate_type.__module__ != "spikingjelly.activation_based.surrogate":
        return None
    params = tuple(sorted(getattr(surrogate_function, "_sg_params", {}).items()))
    return (
        surrogate_type.__module__,
        surrogate_type.__qualname__,
        getattr(surrogate_function, "spiking", None),
        params,
    )


def runtime_key(*tensors: torch.Tensor) -> tuple[Any, ...]:
    return tuple(
        (tuple(tensor.shape), tensor.dtype, tensor.device, tensor.requires_grad)
        for tensor in tensors
    )


def surrogate_callable(surrogate_function: Callable) -> Callable:
    if surrogate_key(surrogate_function) is None:
        return surrogate_function
    surrogate_type = type(surrogate_function)
    fn = (
        surrogate_type.spiking_function
        if surrogate_function.spiking
        else surrogate_type.primitive_function
    )
    params = dict(surrogate_function._sg_params)

    def call(x: torch.Tensor) -> torch.Tensor:
        return fn(x, **params)

    return call


def _build_if_scan(
    v_threshold: float,
    v_reset: float | None,
    surrogate_function: Callable,
    detach_reset: bool,
    store_v_seq: bool,
) -> Callable:
    soft_reset = v_reset is None
    v_reset_value = 0.0 if soft_reset else v_reset
    surrogate_function = surrogate_callable(surrogate_function)

    def _graph(x_seq: torch.Tensor, v_init: torch.Tensor):
        v = v_init
        spike_seq = torch.empty_like(x_seq)
        if store_v_seq:
            v_seq = torch.empty_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = surrogate_function(v - v_threshold)
            spike_d = spike.detach() if detach_reset else spike
            if soft_reset:
                v = v - spike_d * v_threshold
            else:
                v = spike_d * v_reset_value + (1.0 - spike_d) * v
            spike_seq[t] = spike
            if store_v_seq:
                v_seq[t] = v
        if store_v_seq:
            return spike_seq, v, v_seq
        return spike_seq, v

    return _graph


def _build_lif_scan(
    tau: float,
    decay_input: bool,
    v_threshold: float,
    v_reset: float | None,
    surrogate_function: Callable,
    detach_reset: bool,
    store_v_seq: bool,
) -> Callable:
    soft_reset = v_reset is None
    v_reset_value = 0.0 if soft_reset else v_reset
    surrogate_function = surrogate_callable(surrogate_function)

    def _graph(x_seq: torch.Tensor, v_init: torch.Tensor):
        v = v_init
        spike_seq = torch.empty_like(x_seq)
        if store_v_seq:
            v_seq = torch.empty_like(x_seq)
        for t in range(x_seq.shape[0]):
            if decay_input:
                v = v + (x_seq[t] - (v - v_reset_value)) / tau
            else:
                v = v - (v - v_reset_value) / tau + x_seq[t]
            spike = surrogate_function(v - v_threshold)
            spike_d = spike.detach() if detach_reset else spike
            if soft_reset:
                v = v - spike_d * v_threshold
            else:
                v = v_reset_value * spike_d + (1.0 - spike_d) * v
            spike_seq[t] = spike
            if store_v_seq:
                v_seq[t] = v
        if store_v_seq:
            return spike_seq, v, v_seq
        return spike_seq, v

    return _graph


def _build_plif_scan(
    decay_input: bool,
    v_threshold: float,
    v_reset: float | None,
    surrogate_function: Callable,
    detach_reset: bool,
    store_v_seq: bool,
) -> Callable:
    soft_reset = v_reset is None
    v_reset_value = 0.0 if soft_reset else v_reset
    surrogate_function = surrogate_callable(surrogate_function)

    def _graph(x_seq: torch.Tensor, v_init: torch.Tensor, reciprocal_tau: torch.Tensor):
        v = v_init
        spike_seq = torch.empty_like(x_seq)
        if store_v_seq:
            v_seq = torch.empty_like(x_seq)
        for t in range(x_seq.shape[0]):
            if decay_input:
                v = v + (x_seq[t] - (v - v_reset_value)) * reciprocal_tau
            else:
                v = v - (v - v_reset_value) * reciprocal_tau + x_seq[t]
            spike = surrogate_function(v - v_threshold)
            spike_d = spike.detach() if detach_reset else spike
            if soft_reset:
                v = v - spike_d * v_threshold
            else:
                v = v_reset_value * spike_d + (1.0 - spike_d) * v
            spike_seq[t] = spike
            if store_v_seq:
                v_seq[t] = v
        if store_v_seq:
            return spike_seq, v, v_seq
        return spike_seq, v

    return _graph


def compile_graph(cache_key: tuple[Any, ...] | None, fn: Callable) -> Callable:
    _reset_if_forked()
    if not hasattr(torch, "compile"):
        raise RuntimeError("backend='inductor' requires torch.compile.")

    with _CACHE_LOCK:
        if cache_key is not None:
            compiled = _COMPILED_GRAPHS.get(cache_key)
            if compiled is not None:
                _COMPILED_GRAPHS.move_to_end(cache_key)
                return compiled

        compile_kwargs = {"backend": "inductor"}
        try:
            compiled = torch.compile(
                fn,
                **compile_kwargs,
                options={
                    "triton.cudagraphs": False,
                    "triton.cudagraph_trees": False,
                },
            )
        except TypeError:
            compiled = torch.compile(fn, **compile_kwargs)

        if cache_key is not None:
            _COMPILED_GRAPHS[cache_key] = compiled
            _COMPILED_GRAPHS.move_to_end(cache_key)
            while len(_COMPILED_GRAPHS) > _MAX_ENTRIES:
                _COMPILED_GRAPHS.popitem(last=False)
        return compiled
