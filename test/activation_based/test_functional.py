"""Tests for functional.net_config reset cache semantics."""

import gc
import logging
import weakref

import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import base, neuron
from spikingjelly.activation_based.functional import (
    collect_reset_modules,
    invalidate_reset_cache,
    reset_collected_modules,
    reset_net,
)
from spikingjelly.activation_based.functional.net_config import _RESET_MODULE_CACHE


class _ResetCounter(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _NonCallableReset(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset = 1


class _StatefulCounter(base.MemoryModule):
    def __init__(self):
        super().__init__()
        self.register_memory("state", 0.0)
        self.reset_calls = 0

    def single_step_forward(self, x: torch.Tensor):
        self.state = x.detach().clone()
        return self.state

    def reset(self):
        self.reset_calls += 1
        super().reset()


class _EqualHashableResetCounter(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_calls = 0

    def __eq__(self, other):
        return isinstance(other, _EqualHashableResetCounter)

    def __hash__(self):
        return 1

    def reset(self):
        self.reset_calls += 1


class _EqualUnhashableResetCounter(nn.Module):
    __hash__ = None

    def __init__(self):
        super().__init__()
        self.reset_calls = 0

    def __eq__(self, other):
        return self is other

    def reset(self):
        self.reset_calls += 1


def test_collect_reset_modules_ignores_non_callable_reset_attributes():
    net = nn.Sequential(_NonCallableReset(), _ResetCounter())
    modules = collect_reset_modules(net)
    assert modules == (net[1],)


def test_reset_net_caches_modules():
    net = nn.Sequential(_ResetCounter(), nn.ReLU(), _ResetCounter())
    reset_net(net)
    try:
        assert net in _RESET_MODULE_CACHE
        assert tuple(module() for module in _RESET_MODULE_CACHE[net]) == (net[0], net[2])
    finally:
        invalidate_reset_cache(net)


def test_reset_net_cache_hit_reuses_same_module_tuple():
    net = nn.Sequential(_ResetCounter(), nn.ReLU(), _ResetCounter())
    reset_net(net)
    try:
        cached = _RESET_MODULE_CACHE[net]
        reset_net(net)
        assert _RESET_MODULE_CACHE[net] is cached
        assert net[0].reset_calls == 2
        assert net[2].reset_calls == 2
    finally:
        invalidate_reset_cache(net)


def test_invalidate_reset_cache_clears_entry():
    net = nn.Sequential(_ResetCounter(), nn.ReLU(), _ResetCounter())
    reset_net(net)
    assert net in _RESET_MODULE_CACHE
    invalidate_reset_cache(net)
    assert net not in _RESET_MODULE_CACHE


def test_invalidate_then_reset_recollects():
    net = nn.Sequential(_ResetCounter(), nn.ReLU(), _ResetCounter())
    reset_net(net)
    old_cached = _RESET_MODULE_CACHE[net]
    invalidate_reset_cache(net)
    reset_net(net)
    try:
        new_cached = _RESET_MODULE_CACHE[net]
        assert tuple(module() for module in new_cached) == tuple(
            module() for module in old_cached
        )
        assert new_cached is not old_cached
    finally:
        invalidate_reset_cache(net)


def test_invalidate_reset_cache_ignores_unknown_net():
    net = nn.Linear(4, 2)
    invalidate_reset_cache(net)


def test_reset_net_with_ifnode_actually_resets():
    net = nn.Sequential(nn.Linear(4, 8), neuron.IFNode())
    x = torch.randn(2, 4)
    net(x)
    assert torch.is_tensor(net[1].v)
    reset_net(net)
    try:
        assert net[1].v == 0.0
    finally:
        invalidate_reset_cache(net)


def test_reset_net_cached_produces_same_result_as_fresh():
    net = nn.Sequential(nn.Linear(4, 8), neuron.IFNode())
    x = torch.randn(2, 4)
    net(x)
    reset_net(net)
    assert net[1].v == 0.0
    net(x)
    reset_net(net)
    try:
        assert net[1].v == 0.0
    finally:
        invalidate_reset_cache(net)


def test_reset_net_with_lifnode():
    net = nn.Sequential(nn.Linear(4, 8), neuron.LIFNode())
    x = torch.randn(2, 4)
    net(x)
    assert net[1].v is not None
    reset_net(net)
    try:
        assert net[1].v == 0.0
    finally:
        invalidate_reset_cache(net)


def test_reset_net_with_deeply_nested_modules():
    net = nn.Sequential(
        nn.Sequential(nn.Sequential(nn.Linear(4, 8), neuron.IFNode())),
        nn.Sequential(nn.Linear(8, 4), neuron.IFNode()),
    )
    x = torch.randn(2, 4)
    net(x)
    reset_net(net)
    try:
        assert net[0][0][1].v == 0.0
        assert net[1][1].v == 0.0
    finally:
        invalidate_reset_cache(net)


def test_reset_net_with_module_list():
    net = nn.ModuleList(
        [nn.Sequential(nn.Linear(4, 8), neuron.IFNode()) for _ in range(3)]
    )
    x = torch.randn(2, 4)
    for block in net:
        block(x)
    reset_net(net)
    try:
        for block in net:
            assert block[1].v == 0.0
    finally:
        invalidate_reset_cache(net)


def test_independent_models_have_independent_caches():
    net1 = nn.Sequential(_ResetCounter(), _ResetCounter())
    net2 = nn.Sequential(_ResetCounter())
    reset_net(net1)
    reset_net(net2)
    try:
        assert net1 in _RESET_MODULE_CACHE
        assert net2 in _RESET_MODULE_CACHE
        assert len(_RESET_MODULE_CACHE[net1]) == 2
        assert len(_RESET_MODULE_CACHE[net2]) == 1
    finally:
        invalidate_reset_cache(net1)
        invalidate_reset_cache(net2)


def test_collect_reset_modules_remains_stateless():
    net = nn.Sequential(_ResetCounter(), nn.ReLU(), _ResetCounter())
    m1 = collect_reset_modules(net)
    m2 = collect_reset_modules(net)
    assert m1 == m2
    assert m1 is not m2


def test_reset_collected_modules_works_after_invalidate():
    net = nn.Sequential(_ResetCounter(), nn.ReLU(), _ResetCounter())
    reset_net(net)
    invalidate_reset_cache(net)
    modules = collect_reset_modules(net)
    reset_collected_modules(modules)
    assert net[0].reset_calls == 2
    assert net[2].reset_calls == 2


def test_reset_net_with_no_resettable_modules():
    net = nn.Sequential(nn.ReLU(), nn.Linear(4, 2))
    reset_net(net)
    try:
        assert net in _RESET_MODULE_CACHE
        assert _RESET_MODULE_CACHE[net] == ()
    finally:
        invalidate_reset_cache(net)


def test_reset_net_idempotent():
    net = nn.Sequential(_ResetCounter(), _ResetCounter())
    reset_net(net)
    reset_net(net)
    reset_net(net)
    try:
        assert net[0].reset_calls == 3
        assert net[1].reset_calls == 3
    finally:
        invalidate_reset_cache(net)


def test_reset_net_in_training_loop():
    net = nn.Sequential(nn.Linear(4, 8), neuron.IFNode(), nn.Linear(8, 2))
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    for _ in range(5):
        x = torch.randn(2, 4)
        y = torch.randn(2, 2)
        out = net(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        reset_net(net)
        assert net[1].v == 0.0
    invalidate_reset_cache(net)


def test_reset_net_preserves_parameters():
    net = nn.Sequential(nn.Linear(4, 8), neuron.IFNode())
    params_before = [p.clone() for p in net.parameters()]
    reset_net(net)
    try:
        params_after = list(net.parameters())
        for pb, pa in zip(params_before, params_after, strict=True):
            assert torch.equal(pb, pa)
    finally:
        invalidate_reset_cache(net)


def test_reset_net_warns_for_non_memorymodule(caplog: pytest.LogCaptureFixture):
    net = nn.Sequential(_ResetCounter())
    with caplog.at_level(logging.WARNING):
        reset_net(net)
    try:
        assert net[0].reset_calls == 1
        assert any("not spikingjelly.activation_based.base.MemoryModule" in r.message for r in caplog.records)
    finally:
        invalidate_reset_cache(net)


def test_reset_net_propagates_submodule_reset_exception():
    class _BrokenReset(nn.Module):
        def reset(self):
            raise RuntimeError("boom")

    net = nn.Sequential(_BrokenReset())
    with pytest.raises(RuntimeError, match="boom"):
        reset_net(net)
    invalidate_reset_cache(net)


def test_reset_net_cache_must_be_invalidated_after_module_swap():
    net = nn.Sequential(_ResetCounter())
    reset_net(net)
    old_module = net[0]
    replacement = _ResetCounter()
    net[0] = replacement

    reset_net(net)
    assert old_module.reset_calls == 2
    assert replacement.reset_calls == 0

    invalidate_reset_cache(net)
    reset_net(net)
    try:
        assert old_module.reset_calls == 2
        assert replacement.reset_calls == 1
    finally:
        invalidate_reset_cache(net)


def test_reset_net_cache_entry_is_released_with_network():
    net = nn.Sequential(_ResetCounter())
    reset_net(net)
    net_ref = weakref.ref(net)
    cache_size = len(_RESET_MODULE_CACHE)
    assert net in _RESET_MODULE_CACHE
    del net
    gc.collect()

    assert net_ref() is None
    assert len(_RESET_MODULE_CACHE) == cache_size - 1


def test_reset_net_cache_entry_is_released_for_top_level_memorymodule():
    net = _StatefulCounter()
    reset_net(net)
    net_ref = weakref.ref(net)
    cache_size = len(_RESET_MODULE_CACHE)
    assert net in _RESET_MODULE_CACHE
    del net
    gc.collect()

    assert net_ref() is None
    assert len(_RESET_MODULE_CACHE) == cache_size - 1


def test_reset_net_bypasses_cache_for_equal_hashable_modules():
    net1 = _EqualHashableResetCounter()
    net2 = _EqualHashableResetCounter()

    reset_net(net1)
    reset_net(net2)

    assert net1.reset_calls == 1
    assert net2.reset_calls == 1
    assert net1 not in _RESET_MODULE_CACHE
    assert net2 not in _RESET_MODULE_CACHE


def test_invalidate_reset_cache_ignores_unhashable_modules():
    net = _EqualUnhashableResetCounter()

    invalidate_reset_cache(net)


def test_reset_net_cached_modules_follow_memorymodule_reset_semantics():
    net = nn.Sequential(_StatefulCounter())
    x = torch.randn(2, 3)
    net(x)
    reset_net(net)
    reset_net(net)
    try:
        assert net[0].reset_calls == 2
        assert net[0].state == 0.0
    finally:
        invalidate_reset_cache(net)
