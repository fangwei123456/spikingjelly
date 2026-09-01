"""Tests for activation-based functional helpers."""

import gc
import weakref

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import base, layer, neuron
from spikingjelly.activation_based.functional import (
    chunk_multi_step_forward,
    collect_reset_modules,
    invalidate_reset_cache,
    kernel_dot_product,
    multi_step_forward,
    reset_collected_modules,
    reset_net,
    seq_to_ann_forward,
    spike_similar_loss,
    t_last_multi_step_forward,
    t_last_seq_to_ann_forward,
    temporal_efficient_training_cross_entropy,
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


def test_collect_reset_modules_ignores_non_callable_reset_attributes():
    net = nn.Sequential(_NonCallableReset(), _ResetCounter())
    modules = collect_reset_modules(net)
    assert modules == (net[1],)


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


def test_reset_collected_modules_works_after_invalidate():
    net = nn.Sequential(_ResetCounter(), nn.ReLU(), _ResetCounter())
    reset_net(net)
    invalidate_reset_cache(net)
    modules = collect_reset_modules(net)
    reset_collected_modules(modules)
    assert net[0].reset_calls == 2
    assert net[2].reset_calls == 2


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


def test_reset_net_warns_for_non_memorymodule(loguru_records):
    net = nn.Sequential(_ResetCounter())
    reset_net(net)
    try:
        assert net[0].reset_calls == 1
        assert any(
            "not a MemoryModule" in record["message"] for record in loguru_records
        )
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
    assert net in _RESET_MODULE_CACHE
    del net
    gc.collect()

    assert net_ref() is None


def test_reset_net_bypasses_cache_for_equal_hashable_modules():
    net1 = _EqualHashableResetCounter()
    net2 = _EqualHashableResetCounter()

    reset_net(net1)
    reset_net(net2)

    assert net1.reset_calls == 1
    assert net2.reset_calls == 1
    assert net1 not in _RESET_MODULE_CACHE
    assert net2 not in _RESET_MODULE_CACHE


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


def test_multi_step_forward_supports_both_time_axes_and_module_lists():
    modules = [nn.Linear(4, 3), nn.ReLU()]
    x_seq = torch.randn(5, 2, 4)
    expected = torch.stack([modules[1](modules[0](x)) for x in x_seq])

    assert torch.equal(multi_step_forward(x_seq, modules), expected)
    assert torch.equal(
        t_last_multi_step_forward(x_seq.movedim(0, -1), modules),
        expected.movedim(0, -1),
    )


def test_chunk_multi_step_forward_matches_unsplit_forward():
    module = nn.Linear(4, 3)
    x_seq = torch.randn(7, 2, 4)
    expected = module(x_seq)
    chunk_sizes = []
    module.register_forward_pre_hook(
        lambda _module, inputs: chunk_sizes.append(inputs[0].shape[0])
    )

    actual = chunk_multi_step_forward(3, x_seq, module)

    torch.testing.assert_close(actual, expected)
    assert chunk_sizes == [3, 3, 1]


@pytest.mark.parametrize(
    ("kernel", "args"),
    [("linear", ()), ("polynomial", (2,)), ("sigmoid", (0.5,)), ("gaussian", (1.5,))],
)
def test_kernel_dot_product_matches_direct_reference(kernel, args):
    x = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    y = torch.tensor([[0.5, -2.0], [3.0, 1.0]])
    dot = x @ y.T
    if kernel == "linear":
        expected = dot
    elif kernel == "polynomial":
        expected = dot.square()
    elif kernel == "sigmoid":
        expected = torch.sigmoid(0.5 * dot)
    else:
        expected = torch.exp(-torch.cdist(x, y).square() / (2 * 1.5**2))

    torch.testing.assert_close(kernel_dot_product(x, y, kernel, *args), expected)


def test_spike_similar_loss_matches_cosine_similarity_reference_and_backpropagates():
    spikes = torch.tensor(
        [[[1.0, 0.0]], [[0.25, 1.0]], [[1.0, 1.0]]], requires_grad=True
    )
    labels = torch.tensor([[1, 0], [0, 1], [1, 0]])
    flat_spikes = spikes.flatten(start_dim=1)
    similarity = flat_spikes @ flat_spikes.T
    lengths = flat_spikes.norm(dim=1, keepdim=True)
    similarity = similarity / (lengths @ lengths.T + 1e-8)
    expected = F.mse_loss(similarity, (labels.float() @ labels.float().T).clamp_max(1))

    actual = spike_similar_loss(spikes, labels)
    actual.backward()

    torch.testing.assert_close(actual, expected)
    assert spikes.grad is not None


def test_temporal_efficient_cross_entropy_matches_per_step_mean_and_gradient():
    logits = torch.randn(4, 3, 5, requires_grad=True)
    target = torch.tensor([0, 3, 1])
    expected = torch.stack([F.cross_entropy(step, target) for step in logits]).mean()

    actual = temporal_efficient_training_cross_entropy(logits, target)
    actual.backward()

    torch.testing.assert_close(actual, expected)
    assert logits.grad is not None


def test_t_last_multi_step_forward_preserves_contiguous_output_layout():
    x_seq = torch.randn(2, 4, 5)
    expected = torch.stack([x_seq[..., t] for t in range(x_seq.shape[-1])], dim=-1)
    actual = t_last_multi_step_forward(x_seq, nn.Identity())

    assert torch.equal(actual, expected)
    assert actual.stride() == expected.stride()


def test_t_last_multi_step_container_preserves_stateful_execution():
    class RunningSum(nn.Module):
        def __init__(self):
            super().__init__()
            self.total = 0

        def forward(self, x):
            self.total = self.total + x
            return self.total

    x_seq = torch.tensor([[1.0, 2.0, 3.0]])
    container = layer.TLastMultiStepContainer(RunningSum())

    assert torch.equal(container(x_seq), torch.tensor([[1.0, 3.0, 6.0]]))


def test_seq_to_ann_forward_supports_both_time_axes_and_module_lists():
    modules = [nn.Linear(4, 3), nn.ReLU()]
    x_seq = torch.randn(5, 2, 4)
    expected = torch.stack([modules[1](modules[0](x)) for x in x_seq])

    torch.testing.assert_close(seq_to_ann_forward(x_seq, modules), expected)
    torch.testing.assert_close(
        t_last_seq_to_ann_forward(x_seq.movedim(0, -1), modules),
        expected.movedim(0, -1),
    )


def test_seq_to_ann_forward_restores_tuple_outputs():
    x_seq = torch.randn(3, 2, 1, 4, 4)
    pool = nn.MaxPool2d(2, return_indices=True)
    expected = pool(x_seq.flatten(0, 1))
    values, indices = seq_to_ann_forward(x_seq, pool)

    assert torch.equal(values.flatten(0, 1), expected[0])
    assert torch.equal(indices.flatten(0, 1), expected[1])


def test_seq_to_ann_forward_tuple_inputs_match_per_timestep_loop():
    torch.manual_seed(0)
    a = torch.randn(4, 3, 2)
    b = torch.randn(4, 3, 2)
    expected = torch.stack([a[t] * 2.0 + b[t] for t in range(a.shape[0])])

    assert torch.equal(seq_to_ann_forward((a, b), lambda u, v: u * 2.0 + v), expected)


def test_seq_to_ann_forward_tuple_inputs_feed_two_argument_modules():
    torch.manual_seed(0)
    x_seq = torch.randn(3, 2, 1, 4, 4)
    pooled, indices = seq_to_ann_forward(x_seq, nn.MaxPool2d(2, return_indices=True))
    unpool = nn.MaxUnpool2d(2)
    expected = torch.stack(
        [unpool(pooled[t], indices[t]) for t in range(pooled.shape[0])]
    )

    assert torch.equal(seq_to_ann_forward((pooled, indices), unpool), expected)


def test_seq_to_ann_forward_rejects_mismatched_tuple_leading_dimensions():
    with pytest.raises(ValueError, match=r"\[T, batch_size\] leading dimensions"):
        seq_to_ann_forward((torch.randn(3, 2, 4), torch.randn(2, 3, 4)), nn.Identity())


def test_t_last_seq_to_ann_forward_preserves_vmap_output_layout():
    def stateless_module(x):
        return x.square(), x + 1

    x_seq = torch.randn(2, 3, 5)
    expected = torch.vmap(stateless_module, in_dims=-1, out_dims=-1)(x_seq)
    actual = t_last_seq_to_ann_forward(x_seq, stateless_module)

    for actual_item, expected_item in zip(actual, expected):
        assert torch.equal(actual_item, expected_item)
        assert actual_item.stride() == expected_item.stride()


def test_t_last_seq_to_ann_forward_calls_sequential_container():
    calls = []
    modules = nn.Sequential(nn.Identity())
    modules.register_forward_hook(lambda *args: calls.append(None))

    x_seq = torch.randn(2, 3, 5)
    actual = t_last_seq_to_ann_forward(x_seq, modules)

    assert torch.equal(actual, x_seq)
    assert calls == [None]


def test_multistep_max_pool_wrapper_restores_tuple_outputs():
    x_seq = torch.randn(3, 2, 1, 4, 4)
    wrapper = layer.MaxPool2d(2, return_indices=True, step_mode="m")
    expected = nn.MaxPool2d(2, return_indices=True)(x_seq.flatten(0, 1))
    values, indices = wrapper(x_seq)

    assert torch.equal(values.flatten(0, 1), expected[0])
    assert torch.equal(indices.flatten(0, 1), expected[1])
