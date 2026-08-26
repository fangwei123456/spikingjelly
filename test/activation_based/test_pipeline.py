import copy

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from spikingjelly.activation_based import memopt
from spikingjelly.activation_based.memopt import pipeline


class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


class BudgetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.small = Target()
        self.large = Target()
        self.medium = Target()

    def forward(self, x):
        return (
            self.small(x[:, :1]).sum()
            + self.large(x).sum()
            + self.medium(x[:, :2]).sum()
        )


def test_level_zero_is_strict_noop():
    model = BudgetModel()
    state = copy.deepcopy(model.state_dict())

    result = memopt.optimize_memory(model, Target, level=0)

    assert result is model
    assert all(
        torch.equal(model.state_dict()[key], value) for key, value in state.items()
    )


@pytest.mark.parametrize("level", [-1, 5])
def test_optimize_memory_rejects_invalid_level(level):
    with pytest.raises(ValueError, match="level"):
        memopt.optimize_memory(BudgetModel(), Target, level=level)


def test_positive_level_requires_example_forward():
    with pytest.raises(ValueError, match="example_forward"):
        memopt.optimize_memory(BudgetModel(), Target, level=1)


@pytest.mark.parametrize(
    "budget,wrapped",
    [
        ("speed", {"large", "medium"}),
        ("balanced", {"small", "large", "medium"}),
        ("memory", {"small", "large", "medium"}),
    ],
)
def test_level_one_selects_largest_observed_inputs(budget, wrapped):
    model = BudgetModel()
    calls = 0

    def example(current):
        nonlocal calls
        calls += 1
        return current(torch.ones(2, 4))

    result = memopt.optimize_memory(
        model,
        Target,
        example,
        level=1,
        checkpoint_budget=budget,
    )

    assert result is model
    assert calls == 1
    assert {
        name
        for name in ("small", "large", "medium")
        if isinstance(getattr(model, name), CheckpointWrapper)
    } == wrapped


def test_probe_preserves_mode_rng_buffers_memories_and_grads():
    model = nn.Sequential(nn.BatchNorm1d(4), Target())
    model.eval()
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    rng = torch.get_rng_state().clone()
    state = copy.deepcopy(model.state_dict())

    memopt.optimize_memory(
        model,
        Target,
        lambda current: current(torch.randint(0, 2, (3, 4)).float()).sum(),
        level=1,
    )

    assert not model.training
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(parameter.grad, torch.ones_like(parameter))
    assert all(
        torch.equal(model.state_dict()[key], value) for key, value in state.items()
    )


class Splittable(Target):
    def __init__(self):
        super().__init__()
        self.first = nn.ReLU()
        self.second = nn.Identity()

    def forward(self, x):
        return self.second(self.first(x)) * self.scale


class SplitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Splittable()

    def forward(self, x):
        return self.block(x)


def _split(module):
    return (module.first, module.second) if isinstance(module, Splittable) else ()


def _wrapper_count(model):
    return sum(isinstance(module, CheckpointWrapper) for module in model.modules())


def test_level_two_replaces_outer_checkpoint_with_descendant_checkpoints(monkeypatch):
    model = SplitModel()
    monkeypatch.setattr(
        pipeline,
        "_measure_peak",
        lambda current, example: 100 / _wrapper_count(current),
    )

    memopt.optimize_memory(
        model,
        Splittable,
        lambda current: current(torch.ones(4, 3)).sum(),
        level=2,
        split_fn=_split,
    )

    assert isinstance(model.block, Splittable)
    assert isinstance(model.block.first, CheckpointWrapper)
    assert isinstance(model.block.second, CheckpointWrapper)


def test_level_three_checks_final_leaves_once_and_keeps_best_chunks(monkeypatch):
    model = SplitModel()

    def peak(current, example):
        options = [
            pipeline._checkpoint_options(module)
            for module in current.modules()
            if isinstance(module, CheckpointWrapper)
        ]
        chunks = max(int(option["chunks"]) for option in options)
        if len(options) == 1:
            return 100
        return {1: 80, 2: 70, 4: 75}[chunks]

    monkeypatch.setattr(pipeline, "_measure_peak", peak)
    checked = []

    def can_chunk(module):
        checked.append(module)
        return isinstance(module, nn.ReLU)

    memopt.optimize_memory(
        model,
        Splittable,
        lambda current: current(torch.ones(4, 3)).sum(),
        level=3,
        split_fn=_split,
        can_chunk=can_chunk,
    )

    assert len(checked) == 2
    assert pipeline._checkpoint_options(model.block.first)["chunks"] == 2
    assert pipeline._checkpoint_options(model.block.second)["chunks"] == 1


def test_spatial_oom_reverts_candidate(monkeypatch):
    model = SplitModel()

    def peak(current, example):
        if _wrapper_count(current) > 1:
            raise torch.cuda.OutOfMemoryError("candidate")
        return 100

    monkeypatch.setattr(pipeline, "_measure_peak", peak)
    memopt.optimize_memory(
        model,
        Splittable,
        lambda current: current(torch.ones(4, 3)).sum(),
        level=2,
        split_fn=_split,
    )

    assert isinstance(model.block, CheckpointWrapper)


def test_non_oom_candidate_error_reverts_and_propagates(monkeypatch):
    model = SplitModel()

    def peak(current, example):
        if _wrapper_count(current) > 1:
            raise LookupError("original failure")
        return 100

    monkeypatch.setattr(pipeline, "_measure_peak", peak)
    with pytest.raises(LookupError, match="original failure"):
        memopt.optimize_memory(
            model,
            Splittable,
            lambda current: current(torch.ones(4, 3)).sum(),
            level=2,
            split_fn=_split,
        )

    assert isinstance(model.block, CheckpointWrapper)


def test_level_four_unwraps_only_memory_neutral_candidates(monkeypatch):
    model = BudgetModel()
    monkeypatch.setattr(pipeline, "_measure_peak", lambda current, example: 100)
    monkeypatch.setattr(
        pipeline,
        "_forward_costs",
        lambda current, example, group: {"large": 2.0, "medium": 1.0, "small": 0.5},
    )

    memopt.optimize_memory(
        model,
        Target,
        lambda current: current(torch.ones(2, 4)),
        level=4,
        split_fn=None,
        can_chunk=None,
    )

    assert not any(isinstance(module, CheckpointWrapper) for module in model.children())
