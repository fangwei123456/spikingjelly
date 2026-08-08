import copy
import gc
import pickle
import threading
import time
import weakref

import torch

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.neuron import inductor_cache


def _make_node() -> neuron.IFNode:
    return neuron.IFNode(
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
        detach_reset=False,
        step_mode="m",
        backend="inductor",
    ).train()


def test_inductor_cache_reuses_same_config_across_modules_and_function(monkeypatch):
    inductor_cache.clear()
    compile_calls = []

    def fake_compile(fn, **_kwargs):
        compile_calls.append(fn)
        return fn

    monkeypatch.setattr(inductor_cache.torch, "compile", fake_compile)
    x = torch.randn(4, 2, 3)

    node0 = _make_node()
    node1 = _make_node()
    functional.reset_net(node0)
    functional.reset_net(node1)
    y0 = node0(x)
    y_function, _, _ = functional.if_multi_step_inductor(
        x,
        torch.zeros_like(x[0]),
        node0.v_threshold,
        node0.v_reset,
        surrogate.Sigmoid(alpha=4.0),
        node0.detach_reset,
        node0.store_v_seq,
    )
    y1 = node1(x)

    assert len(compile_calls) == 1
    assert inductor_cache.info()["entries"] == 1
    assert torch.equal(y0, y_function)
    assert torch.equal(y0, y1)

    inductor_cache.clear()


def test_inductor_cache_is_bounded(monkeypatch):
    inductor_cache.clear()

    def fake_compile(fn, **_kwargs):
        return fn

    monkeypatch.setattr(inductor_cache.torch, "compile", fake_compile)
    max_entries = inductor_cache.info()["max_entries"]
    for i in range(max_entries + 3):
        inductor_cache.compile_graph(("op", i), lambda x, i=i: x + i)

    assert inductor_cache.info()["entries"] == max_entries
    inductor_cache.clear()


def test_inductor_cache_clears_after_pid_change(monkeypatch):
    inductor_cache.clear()
    original_pid = inductor_cache.info()["pid"]

    def fake_compile(fn, **_kwargs):
        return fn

    monkeypatch.setattr(inductor_cache.torch, "compile", fake_compile)
    inductor_cache.compile_graph(("op", 0), lambda x: x)
    assert inductor_cache.info()["entries"] == 1

    monkeypatch.setattr(inductor_cache.os, "getpid", lambda: original_pid + 1)
    assert inductor_cache.info()["entries"] == 0
    assert inductor_cache.info()["pid"] == original_pid + 1

    inductor_cache.clear()


def test_inductor_cache_reset_after_fork_replaces_lock(monkeypatch):
    inductor_cache.clear()
    monkeypatch.setattr(inductor_cache.torch, "compile", lambda fn, **_kwargs: fn)
    inductor_cache.compile_graph(("op",), lambda x: x)
    old_lock = inductor_cache._CACHE_LOCK

    inductor_cache._reset_after_fork()

    assert inductor_cache._CACHE_LOCK is not old_lock
    assert inductor_cache.info()["entries"] == 0


def test_inductor_cache_is_not_serialized_with_node(monkeypatch):
    inductor_cache.clear()

    def fake_compile(fn, **_kwargs):
        return fn

    monkeypatch.setattr(inductor_cache.torch, "compile", fake_compile)
    node = _make_node()
    functional.reset_net(node)
    node(torch.randn(4, 2, 3))
    assert inductor_cache.info()["entries"] == 1

    copy.deepcopy(node)
    pickle.loads(pickle.dumps(node))

    assert inductor_cache.info()["entries"] == 1

    inductor_cache.clear()


def test_inductor_cache_compiles_same_key_once_across_threads(monkeypatch):
    inductor_cache.clear()
    compile_calls = []

    def fake_compile(fn, **_kwargs):
        compile_calls.append(fn)
        time.sleep(0.02)
        return fn

    monkeypatch.setattr(inductor_cache.torch, "compile", fake_compile)
    barrier = threading.Barrier(2)

    def compile_once():
        barrier.wait()
        inductor_cache.compile_graph(("shared",), lambda x: x)

    threads = [threading.Thread(target=compile_once) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(compile_calls) == 1
    assert inductor_cache.info()["entries"] == 1
    inductor_cache.clear()


def test_inductor_cache_uses_forward_only_surrogate_modules(monkeypatch):
    inductor_cache.clear()
    monkeypatch.setattr(inductor_cache.torch, "compile", lambda fn, **_kwargs: fn)
    x_seq = torch.zeros(2, 1)
    v = torch.zeros(1)

    for surrogate_function in (
        surrogate.DeterministicPass(spiking=False),
        surrogate.PoissonPass(spiking=False),
    ):
        spike, _, _ = functional.if_multi_step_inductor(
            x_seq, v, 1.0, 0.0, surrogate_function
        )
        assert spike.shape == x_seq.shape

    assert inductor_cache.info()["entries"] == 0


def test_inductor_cache_does_not_share_custom_surrogate_closures(monkeypatch):
    class ScaledSurrogate(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return torch.full_like(x, self.scale)

    inductor_cache.clear()
    compile_calls = []

    def fake_compile(fn, **_kwargs):
        compile_calls.append(fn)
        return fn

    monkeypatch.setattr(inductor_cache.torch, "compile", fake_compile)
    x_seq = torch.zeros(2, 1)
    v = torch.zeros(1)
    surrogate0 = ScaledSurrogate(1.0)
    surrogate1 = ScaledSurrogate(2.0)

    spike0, _, _ = functional.if_multi_step_inductor(x_seq, v, 1.0, 0.0, surrogate0)
    spike1, _, _ = functional.if_multi_step_inductor(x_seq, v, 1.0, 0.0, surrogate1)
    spike0_again, _, _ = functional.if_multi_step_inductor(
        x_seq, v, 1.0, 0.0, surrogate0
    )

    assert torch.equal(spike0, torch.ones_like(spike0))
    assert torch.equal(spike1, torch.full_like(spike1, 2.0))
    assert torch.equal(spike0_again, spike0)
    assert len(compile_calls) == 3
    assert inductor_cache.info()["entries"] == 0
    inductor_cache.clear()


def test_inductor_cache_does_not_retain_builtin_surrogate_module(monkeypatch):
    inductor_cache.clear()

    def fake_compile(fn, **_kwargs):
        return fn

    monkeypatch.setattr(inductor_cache.torch, "compile", fake_compile)
    node = _make_node()
    surrogate_ref = weakref.ref(node.surrogate_function)
    node(torch.randn(4, 2, 3))

    del node
    gc.collect()

    assert surrogate_ref() is None
    assert inductor_cache.info()["entries"] == 1
    inductor_cache.clear()


def test_inductor_cache_does_not_retain_custom_surrogate_instances(monkeypatch):
    class ScaledSurrogate(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return torch.full_like(x, self.scale)

    inductor_cache.clear()
    monkeypatch.setattr(inductor_cache.torch, "compile", lambda fn, **_kwargs: fn)
    references = []
    x_seq = torch.zeros(1, 1)
    v = torch.zeros(1)

    for i in range(3):
        surrogate = ScaledSurrogate(float(i))
        references.append(weakref.ref(surrogate))
        functional.if_multi_step_inductor(x_seq, v, 1.0, 0.0, surrogate)

    del surrogate
    gc.collect()
    assert inductor_cache.info()["entries"] == 0
    assert all(reference() is None for reference in references)
