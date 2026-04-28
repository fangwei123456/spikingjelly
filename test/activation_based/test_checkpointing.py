import copy

import pytest
import torch
import torch.nn as nn

import spikingjelly.activation_based.memopt as memopt
import spikingjelly.activation_based.memopt.compress as memopt_compress
import spikingjelly.activation_based.memopt.pipeline as memopt_pipeline
import spikingjelly.activation_based.triton_kernel.compress as triton_compress
from spikingjelly.activation_based.memopt.checkpointing import (
    in_gc_1st_forward,
    query_autocast,
    input_compressed_gc,
    to_gc_function,
    GCContainer,
    TCGCContainer,
    _gc_1st_forward,
    _separate_args,
    _combine_args,
)
from spikingjelly.activation_based.memopt.compress import (
    BaseSpikeCompressor,
    BooleanSpikeCompressor,
    BitSpikeCompressor,
    NullSpikeCompressor,
    SparseSpikeCompressor,
)
from spikingjelly.activation_based import neuron


def simple_forward_fn(x, weight, bias=None):
    y = torch.matmul(x, weight.t())
    if bias is not None:
        y = y + bias
    return y


class MockCompressor(BaseSpikeCompressor):
    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq * 2

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        return s_seq / 2


class BinaryProject(nn.Module):
    def forward(self, x):
        return (x > 0).to(x.dtype)


class TargetBlock(nn.Module):
    def forward(self, x):
        return x + 1.0


class ParameterizedTargetBlock(nn.Module):
    def __init__(self, features=4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))

    def forward(self, x):
        return x + self.weight


class SpatialSplitBlock(nn.Module):
    def __init__(self, features=4):
        super().__init__()
        self.features = features

    def forward(self, x):
        return torch.relu(x)

    def __spatial_split__(self):
        return [nn.Identity(), nn.ReLU()]


class TemporalSplitBlock(nn.Sequential):
    def __init__(self, features=4):
        super().__init__(nn.Linear(features, features), nn.ReLU())
        self.n_seq_inputs = 1
        self.n_outputs = 1


class TemporalOnlyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_seq_inputs = 1
        self.n_outputs = 1

    def forward(self, x):
        return x + 1.0


class UnsplittableBlock(nn.Module):
    def forward(self, x):
        return x


class CountedNonBinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.block = TargetBlock()
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return self.block(self.proj(x))


class DualTargetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.large = TargetBlock()
        self.small = TargetBlock()

    def forward(self, x):
        y1 = self.large(x)
        y2 = self.small(x[..., :2])
        return y1[..., :2] + y2


def _result_row(module_name: str):
    return [(module_name, 0.0)]


def _profile_result_with_optional_peak(result, return_peak):
    if return_peak:
        return result, 100, 100
    return result


def test_thread_local_functions():
    """Test thread-local functions and context managers."""
    # Test initial state
    assert not in_gc_1st_forward()

    # Test context manager
    with _gc_1st_forward():
        assert in_gc_1st_forward()
    assert not in_gc_1st_forward()

    # Test nested context managers
    with _gc_1st_forward():
        assert in_gc_1st_forward()
        with _gc_1st_forward():
            assert in_gc_1st_forward()
        assert in_gc_1st_forward()
    assert not in_gc_1st_forward()


def test_memopt_package_imports_pipeline_api():
    assert hasattr(memopt, "memory_optimization")
    assert hasattr(memopt, "MemOptSummary")
    assert hasattr(memopt, "MEMOPT_PROFILES")
    assert hasattr(memopt, "MEMOPT_CHECKPOINT_BUDGETS")
    assert hasattr(memopt, "MEMOPT_PREFERENCES")


def test_probe_binary_inputs_stops_early_when_all_targets_are_non_binary():
    net = CountedNonBinaryNet()
    result = memopt_pipeline._probe_binary_inputs(
        net,
        TargetBlock,
        dummy_input=(torch.randn(2, 4),),
        n_trials=5,
    )

    assert result[net.block] is False
    assert net.forward_calls == 1


def test_probe_binary_inputs_uses_dummy_input_before_random_trials(monkeypatch):
    net = CountedNonBinaryNet()

    def fail_randomize(dummy_input):
        raise AssertionError("_randomize_input_like should not be called")

    monkeypatch.setattr(memopt_pipeline, "_randomize_input_like", fail_randomize)
    result = memopt_pipeline._probe_binary_inputs(
        net,
        TargetBlock,
        dummy_input=(torch.randn(2, 4),),
        n_trials=5,
    )

    assert result[net.block] is False
    assert net.forward_calls == 1


def test_randomize_input_like_supports_bool_and_integer_tensors():
    bool_input = torch.tensor([[True, False], [False, True]])
    int_input = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

    randomized_bool = memopt_pipeline._randomize_input_like(bool_input)
    randomized_int = memopt_pipeline._randomize_input_like(int_input)

    assert randomized_bool.dtype == torch.bool
    assert randomized_bool.shape == bool_input.shape
    assert randomized_int.dtype == torch.int64
    assert randomized_int.shape == int_input.shape


def test_autocast_query():
    """Test autocast querying functionality."""
    device_type, dtype, enabled = query_autocast()
    assert device_type is not None
    assert dtype is not None
    assert not enabled

    with torch.amp.autocast("cpu", dtype=torch.float16):
        device_type, dtype, enabled = query_autocast()
        assert device_type == "cpu"
        assert dtype == torch.float16
        assert enabled


def test_argument_separate_combine():
    """Test argument separation and combination functions."""
    # Test with only tensors
    tensor1 = torch.randn(3, 4)
    tensor2 = torch.randn(2, 5)

    input_args, tensor_args, tensor_indices = _separate_args(tensor1, tensor2)

    assert len(input_args) == 2
    assert all(arg is None for arg in input_args)
    assert tensor_args == [tensor1, tensor2]
    assert tensor_indices == [0, 1]

    # Test with mixed types
    non_tensor = "string"
    input_args, tensor_args, tensor_indices = _separate_args(
        tensor1, non_tensor, tensor2
    )

    assert input_args == [None, "string", None]
    assert tensor_args == [tensor1, tensor2]
    assert tensor_indices == [0, 2]

    combined = _combine_args([None, "string", None], [tensor1, tensor2], [0, 2])
    assert len(combined) == 3
    assert torch.equal(combined[0], tensor1)
    assert combined[1] == "string"
    assert torch.equal(combined[2], tensor2)


def test_compressors_accept_tuple_shapes_on_decompress():
    bit_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bit_spikes = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=bit_device
    )
    shape = tuple(bit_spikes.shape)

    bit = BitSpikeCompressor()
    bit_decompressed = bit.decompress(bit.compress(bit_spikes), shape)
    torch.testing.assert_close(bit_decompressed, bit_spikes)

    spikes = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    sparse = SparseSpikeCompressor()
    sparse_decompressed = sparse.decompress(sparse.compress(spikes), shape)
    torch.testing.assert_close(sparse_decompressed, spikes)


def test_memory_optimization_requires_dummy_input_for_higher_levels():
    net = nn.Sequential(TargetBlock())
    with pytest.raises(ValueError, match="dummy_input must be provided"):
        memopt.memory_optimization(net, TargetBlock, level=2)


def test_memory_optimization_profile_balanced_falls_back_without_dummy_input(
    monkeypatch,
):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    net, summary = memopt.memory_optimization(
        nn.Sequential(TargetBlock()),
        TargetBlock,
        dummy_input=None,
        profile="balanced",
        return_summary=True,
    )

    assert isinstance(net[0], GCContainer)
    assert isinstance(summary, memopt.MemOptSummary)
    assert summary.profile == "balanced"
    assert summary.requested_level == 2
    assert summary.applied_level == 1
    assert "fallback:level>1_requires_dummy_input" in summary.notes


def test_memory_optimization_rejects_unknown_profile():
    with pytest.raises(ValueError, match="Unsupported memopt profile"):
        memopt.memory_optimization(
            nn.Sequential(TargetBlock()),
            TargetBlock,
            profile="mystery",
        )


def test_memory_optimization_rejects_unknown_prefer():
    with pytest.raises(ValueError, match="Unsupported prefer"):
        memopt.memory_optimization(
            nn.Sequential(TargetBlock()),
            TargetBlock,
            prefer="mystery",
        )


def test_memory_optimization_prefer_speed_sets_defaults(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    net, summary = memopt.memory_optimization(
        nn.Sequential(TargetBlock(), TargetBlock(), TargetBlock(), TargetBlock()),
        TargetBlock,
        dummy_input=None,
        prefer="speed",
        return_summary=True,
    )

    assert isinstance(net[0], GCContainer)
    assert isinstance(net[1], GCContainer)
    assert isinstance(net[2], TargetBlock)
    assert isinstance(net[3], TargetBlock)
    assert summary.prefer == "speed"
    assert summary.profile == "safe"
    assert summary.checkpoint_budget == "speed"
    assert "prefer:profile=safe" in summary.notes
    assert "prefer:checkpoint_budget=speed" in summary.notes
    assert summary.gc_selected_count == 2
    assert len(summary.gc_selected_modules) == 2
    assert "no dummy_input was available" in summary.gc_selection_explanation
    assert "favor training speed" in summary.recommendation


def test_memory_optimization_explicit_profile_overrides_prefer_profile(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    _, summary = memopt.memory_optimization(
        nn.Sequential(TargetBlock(), TargetBlock(), TargetBlock(), TargetBlock()),
        TargetBlock,
        dummy_input=None,
        prefer="speed",
        profile="memory",
        return_summary=True,
    )

    assert summary.prefer == "speed"
    assert summary.profile == "memory"
    assert summary.checkpoint_budget == "speed"
    assert "prefer:profile=safe" not in summary.notes
    assert "prefer:checkpoint_budget=speed" in summary.notes


def test_apply_gc_selects_largest_input_targets_when_budgeted(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    net, summary = memopt_pipeline.apply_gc(
        DualTargetNet(),
        TargetBlock,
        dummy_input=(torch.randn(2, 4),),
        compress_x=False,
        device="cpu",
        max_gc_wrapped_modules=1,
        return_summary=True,
    )

    assert isinstance(net.large, GCContainer)
    assert isinstance(net.small, TargetBlock)
    assert summary["gc_candidate_count"] == 2
    assert summary["gc_selected_count"] == 1
    assert summary["gc_selection_policy"] == "largest_input_activations"
    assert summary["gc_selected_modules"] == ["large"]
    assert "largest observed input activations" in summary["gc_selection_explanation"]


def test_apply_gc_budget_without_dummy_input_falls_back_to_module_order():
    net, summary = memopt_pipeline.apply_gc(
        nn.Sequential(TargetBlock(), TargetBlock()),
        TargetBlock,
        dummy_input=None,
        compress_x=False,
        device="cpu",
        max_gc_wrapped_modules=1,
        return_summary=True,
    )

    assert isinstance(net[0], GCContainer)
    assert isinstance(net[1], TargetBlock)
    assert summary["gc_candidate_count"] == 2
    assert summary["gc_selected_count"] == 1
    assert summary["gc_selection_policy"] == "fallback_module_order"


def test_apply_gc_checkpoint_budget_speed_selects_half_of_targets():
    net, summary = memopt_pipeline.apply_gc(
        nn.Sequential(TargetBlock(), TargetBlock(), TargetBlock(), TargetBlock()),
        TargetBlock,
        dummy_input=None,
        compress_x=False,
        device="cpu",
        checkpoint_budget="speed",
        return_summary=True,
    )

    assert isinstance(net[0], GCContainer)
    assert isinstance(net[1], GCContainer)
    assert isinstance(net[2], TargetBlock)
    assert isinstance(net[3], TargetBlock)
    assert summary["checkpoint_budget"] == "speed"
    assert summary["gc_candidate_count"] == 4
    assert summary["gc_selected_count"] == 2
    assert "checkpoint_budget:speed" in summary["budget_notes"]


def test_apply_gc_checkpoint_budget_memory_covers_all_targets():
    net, summary = memopt_pipeline.apply_gc(
        nn.Sequential(TargetBlock(), TargetBlock()),
        TargetBlock,
        dummy_input=None,
        compress_x=False,
        device="cpu",
        checkpoint_budget="memory",
        return_summary=True,
    )

    assert isinstance(net[0], GCContainer)
    assert isinstance(net[1], GCContainer)
    assert summary["gc_selected_count"] == 2
    assert summary["gc_selection_policy"] == "budget_covers_all"


def test_apply_gc_manual_budget_takes_priority_over_checkpoint_budget():
    net, summary = memopt_pipeline.apply_gc(
        nn.Sequential(TargetBlock(), TargetBlock(), TargetBlock(), TargetBlock()),
        TargetBlock,
        dummy_input=None,
        compress_x=False,
        device="cpu",
        checkpoint_budget="memory",
        max_gc_wrapped_modules=1,
        return_summary=True,
    )

    assert isinstance(net[0], GCContainer)
    assert isinstance(net[1], TargetBlock)
    assert isinstance(net[2], TargetBlock)
    assert isinstance(net[3], TargetBlock)
    assert summary["gc_selected_count"] == 1


def test_memory_optimization_level1_without_dummy_input_skips_main_process_warmup(
    monkeypatch,
):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    warmup_calls = {"count": 0}

    def fake_dummy_train_step(*args, **kwargs):
        warmup_calls["count"] += 1

    monkeypatch.setattr(memopt_pipeline, "_dummy_train_step", fake_dummy_train_step)

    net = nn.Sequential(TargetBlock())
    optimized = memopt.memory_optimization(
        net,
        TargetBlock,
        dummy_input=None,
        compress_x=False,
        level=1,
    )

    assert isinstance(optimized[0], GCContainer)
    assert warmup_calls["count"] == 0


def test_memory_optimization_level1_wraps_target_modules_and_uses_bit_compressor(
    monkeypatch,
):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(memopt_compress, "triton", object())
    monkeypatch.setattr(
        memopt_compress, "bit_spike_compress", triton_compress.bit_spike_compress
    )
    monkeypatch.setattr(
        memopt_compress, "bit_spike_decompress", triton_compress.bit_spike_decompress
    )

    net = nn.Sequential(BinaryProject(), ParameterizedTargetBlock())
    optimized = memopt.memory_optimization(
        net,
        ParameterizedTargetBlock,
        dummy_input=(torch.randn(2, 4),),
        compress_x=True,
        level=1,
    )

    assert isinstance(optimized[1], GCContainer)
    assert isinstance(optimized[1].x_compressor, BitSpikeCompressor)
    assert optimized[1][0].weight.device.type == "cpu"


def test_memory_optimization_can_disable_main_process_warmup(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    warmup_calls = {"count": 0}

    def fake_dummy_train_step(*args, **kwargs):
        warmup_calls["count"] += 1

    monkeypatch.setattr(memopt_pipeline, "_dummy_train_step", fake_dummy_train_step)

    net = nn.Sequential(TargetBlock())
    optimized = memopt.memory_optimization(
        net,
        TargetBlock,
        dummy_input=(torch.randn(2, 4),),
        compress_x=False,
        level=1,
        warmup_in_main_process=False,
    )

    assert isinstance(optimized[0], GCContainer)
    assert warmup_calls["count"] == 0


def test_memory_optimization_profile_respects_explicit_warmup_flags(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    _, summary = memopt.memory_optimization(
        nn.Sequential(TargetBlock()),
        TargetBlock,
        dummy_input=(torch.randn(2, 4),),
        profile="safe",
        warmup_in_main_process=True,
        warmup_in_profile_workers=True,
        return_summary=True,
    )

    assert summary.options["warmup_in_main_process"] is True
    assert summary.options["warmup_in_profile_workers"] is True


def test_memory_optimization_forwards_profile_worker_warmup_flag(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    peak_flags = []
    profile_flags = []

    def fake_train_peak_memory(*args, **kwargs):
        peak_flags.append(kwargs.get("worker_warmup"))
        return 100, 100

    def fake_train_memory_profile(*args, **kwargs):
        profile_flags.append(
            (kwargs.get("worker_warmup"), kwargs.get("return_peak", False))
        )
        return _profile_result_with_optional_peak(
            [], kwargs.get("return_peak", False)
        )

    monkeypatch.setattr(memopt_pipeline, "_train_peak_memory", fake_train_peak_memory)
    monkeypatch.setattr(
        memopt_pipeline, "_train_memory_profile", fake_train_memory_profile
    )

    net = nn.Sequential(SpatialSplitBlock())
    optimized = memopt.memory_optimization(
        net,
        SpatialSplitBlock,
        dummy_input=(torch.randn(2, 4),),
        compress_x=False,
        level=2,
        warmup_in_profile_workers=False,
        warmup_in_main_process=False,
    )

    assert isinstance(optimized[0], GCContainer)
    assert peak_flags == [False]
    assert profile_flags == [(False, False)]


def test_memory_optimization_profile_memory_honors_allow_expensive_profiling_false(
    monkeypatch,
):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    profile_flags = []

    def fake_train_memory_profile(*args, **kwargs):
        profile_flags.append(
            (
                kwargs.get("worker_warmup"),
                kwargs.get("return_peak", False),
            )
        )
        return _profile_result_with_optional_peak([], kwargs.get("return_peak", False))

    monkeypatch.setattr(
        memopt_pipeline, "_train_memory_profile", fake_train_memory_profile
    )
    monkeypatch.setattr(
        memopt_pipeline, "_train_peak_memory", lambda *args, **kwargs: (100, 100)
    )

    _, summary = memopt.memory_optimization(
        nn.Sequential(SpatialSplitBlock()),
        SpatialSplitBlock,
        dummy_input=(torch.randn(2, 4),),
        profile="memory",
        allow_expensive_profiling=False,
        warmup_in_main_process=False,
        return_summary=True,
    )

    assert profile_flags
    assert all(flag[0] is False for flag in profile_flags)
    assert all(flag[1] is False for flag in profile_flags)
    assert summary.allow_expensive_profiling is False
    assert summary.options["max_split_rounds"] == 1
    assert summary.options["max_candidates_per_round"] == 1


def test_bit_spike_compressor_cpu_round_trip_with_triton_available(monkeypatch):
    monkeypatch.setattr(memopt_compress, "triton", object())
    monkeypatch.setattr(
        memopt_compress, "bit_spike_compress", triton_compress.bit_spike_compress
    )
    monkeypatch.setattr(
        memopt_compress, "bit_spike_decompress", triton_compress.bit_spike_decompress
    )
    spikes = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    compressor = BitSpikeCompressor()

    compressed = compressor.compress(spikes)
    decompressed = compressor.decompress(compressed, spikes.shape)

    assert compressed.device.type == "cpu"
    torch.testing.assert_close(decompressed, spikes)


def test_candidate_entries_treats_zero_budget_as_disabled():
    results = [("a", 1.0), ("b", 2.0)]

    assert memopt_pipeline._candidate_entries(results, None) == results
    assert memopt_pipeline._candidate_entries(results, 0) == []
    assert memopt_pipeline._candidate_entries(results, -1) == []


def test_apply_gc_clones_manual_compressor_instances(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    shared = BooleanSpikeCompressor()
    block1 = TargetBlock()
    block2 = TargetBlock()
    block1.x_compressor = shared
    block2.x_compressor = shared
    net = nn.Sequential(block1, block2)

    optimized = memopt_pipeline.apply_gc(
        net,
        TargetBlock,
        dummy_input=(torch.randint(0, 2, (2, 4), dtype=torch.float32),),
        compress_x=True,
        device="cpu",
    )

    compressor1 = optimized[0].x_compressor
    compressor2 = optimized[1].x_compressor
    assert isinstance(compressor1, BooleanSpikeCompressor)
    assert isinstance(compressor2, BooleanSpikeCompressor)
    assert compressor1 is not compressor2

    compressed1 = compressor1.compress(torch.randint(0, 2, (2, 4), dtype=torch.float16))
    compressed2 = compressor2.compress(torch.randint(0, 2, (2, 4), dtype=torch.float32))

    assert compressor1.s_seq_dtype == torch.float16
    assert compressor2.s_seq_dtype == torch.float32
    torch.testing.assert_close(
        compressor1.decompress(compressed1, (2, 4)),
        compressed1.to(torch.float16),
    )
    torch.testing.assert_close(
        compressor2.decompress(compressed2, (2, 4)),
        compressed2.to(torch.float32),
    )


def test_apply_gc_skips_binary_probe_when_all_targets_have_manual_compressors(
    monkeypatch,
):
    calls = {"count": 0}

    def fake_probe(*args, **kwargs):
        calls["count"] += 1
        return {}

    monkeypatch.setattr(memopt_pipeline, "_probe_binary_inputs", fake_probe)

    block1 = TargetBlock()
    block1.x_compressor = BooleanSpikeCompressor()
    block2 = TargetBlock()
    block2.x_compressor = BitSpikeCompressor()
    net = nn.Sequential(block1, block2)

    optimized = memopt_pipeline.apply_gc(
        net,
        TargetBlock,
        dummy_input=(torch.randint(0, 2, (2, 4), dtype=torch.float32),),
        compress_x=True,
        device="cpu",
    )

    assert calls["count"] == 0
    assert isinstance(optimized[0], GCContainer)
    assert isinstance(optimized[1], GCContainer)
    assert isinstance(optimized[0].x_compressor, BooleanSpikeCompressor)
    assert isinstance(optimized[1].x_compressor, BitSpikeCompressor)


def test_memory_optimization_returns_summary_for_skipped_spatial_stage(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    net, summary = memopt.memory_optimization(
        nn.Sequential(TemporalOnlyBlock()),
        TemporalOnlyBlock,
        dummy_input=(torch.randn(2, 4),),
        compress_x=False,
        level=2,
        warmup_in_main_process=False,
        return_summary=True,
    )

    assert isinstance(net[0], GCContainer)
    assert isinstance(summary, memopt.MemOptSummary)
    assert summary.gc_wrap_count == 1
    assert summary.gc_container_count == 1
    assert summary.tcgc_container_count == 0
    assert "level1_gc" in summary.applied_steps
    assert "level2:no_spatial_candidates" in summary.skipped_steps


def test_memory_optimization_level2_spatially_splits_heavy_gc_container(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    peak_calls = {"count": 0}
    profile_values = iter([
        _result_row("0"),
        [],
    ])
    def fake_train_peak_memory(*args, **kwargs):
        peak_calls["count"] += 1
        return (80, 80) if peak_calls["count"] == 1 else (60, 60)
    monkeypatch.setattr(memopt_pipeline, "_train_peak_memory", fake_train_peak_memory)
    monkeypatch.setattr(
        memopt_pipeline,
        "_train_memory_profile",
        lambda *args, **kwargs: _profile_result_with_optional_peak(
            next(profile_values), kwargs.get("return_peak", False)
        ),
    )

    net = nn.Sequential(SpatialSplitBlock())
    optimized = memopt.memory_optimization(
        net,
        SpatialSplitBlock,
        dummy_input=(torch.randn(2, 4),),
        compress_x=False,
        level=2,
    )

    assert isinstance(optimized[0], nn.Sequential)
    assert len(optimized[0]) == 2
    assert all(isinstance(block, GCContainer) for block in optimized[0])


def test_memory_optimization_level2_skips_unsplittable_hottest_candidate(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    peak_calls = {"count": 0}
    profile_values = iter(
        [
            [("0", 0.0), ("1", 0.0)],
            [],
        ]
    )
    def fake_train_peak_memory(*args, **kwargs):
        peak_calls["count"] += 1
        return (80, 80) if peak_calls["count"] == 1 else (60, 60)
    monkeypatch.setattr(memopt_pipeline, "_train_peak_memory", fake_train_peak_memory)
    monkeypatch.setattr(
        memopt_pipeline,
        "_train_memory_profile",
        lambda *args, **kwargs: _profile_result_with_optional_peak(
            next(profile_values), kwargs.get("return_peak", False)
        ),
    )

    net = nn.Sequential(UnsplittableBlock(), SpatialSplitBlock())
    optimized = memopt.memory_optimization(
        net,
        (UnsplittableBlock, SpatialSplitBlock),
        dummy_input=(torch.randn(2, 4),),
        compress_x=False,
        level=2,
    )

    assert isinstance(optimized[0], GCContainer)
    assert isinstance(optimized[1], nn.Sequential)
    assert all(isinstance(block, GCContainer) for block in optimized[1])


def test_memory_optimization_level2_skips_profiling_when_no_gccontainers(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    peak_calls = {"count": 0}
    profile_calls = {"count": 0}

    monkeypatch.setattr(
        memopt_pipeline,
        "_train_peak_memory",
        lambda *args, **kwargs: peak_calls.__setitem__("count", peak_calls["count"] + 1),
    )
    monkeypatch.setattr(
        memopt_pipeline,
        "_train_memory_profile",
        lambda *args, **kwargs: profile_calls.__setitem__(
            "count", profile_calls["count"] + 1
        ),
    )

    net = nn.Sequential(nn.ReLU())
    optimized = memopt.memory_optimization(
        net,
        TargetBlock,
        dummy_input=(torch.randn(2, 4),),
        compress_x=False,
        level=2,
    )

    assert isinstance(optimized[0], nn.ReLU)
    assert peak_calls["count"] == 0
    assert profile_calls["count"] == 0


def test_memory_optimization_level2_skips_when_no_spatial_split_candidates(
    monkeypatch,
):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    peak_calls = {"count": 0}
    profile_calls = {"count": 0}

    monkeypatch.setattr(
        memopt_pipeline,
        "_train_peak_memory",
        lambda *args, **kwargs: peak_calls.__setitem__("count", peak_calls["count"] + 1),
    )
    monkeypatch.setattr(
        memopt_pipeline,
        "_train_memory_profile",
        lambda *args, **kwargs: profile_calls.__setitem__(
            "count", profile_calls["count"] + 1
        ),
    )

    net = nn.Sequential(TemporalOnlyBlock())
    optimized = memopt.memory_optimization(
        net,
        TemporalOnlyBlock,
        dummy_input=(torch.randn(2, 4),),
        compress_x=False,
        level=2,
        warmup_in_main_process=False,
    )

    assert isinstance(optimized[0], GCContainer)
    assert peak_calls["count"] == 0
    assert profile_calls["count"] == 0


def test_memory_optimization_level3_temporally_splits_heavy_gc_container(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    peak_calls = {"count": 0}
    profile_values = iter([
        _result_row("0"),
        [],
    ])
    def fake_train_peak_memory(*args, **kwargs):
        peak_calls["count"] += 1
        return (80, 80) if peak_calls["count"] == 1 else (60, 60)
    monkeypatch.setattr(memopt_pipeline, "_train_peak_memory", fake_train_peak_memory)
    monkeypatch.setattr(
        memopt_pipeline,
        "_train_memory_profile",
        lambda *args, **kwargs: _profile_result_with_optional_peak(
            next(profile_values), kwargs.get("return_peak", False)
        ),
    )

    net = nn.Sequential(TemporalSplitBlock())
    optimized = memopt.memory_optimization(
        net,
        TemporalSplitBlock,
        dummy_input=(torch.randn(4, 2, 4),),
        compress_x=False,
        level=3,
        temporal_split_factor=2,
    )

    assert isinstance(optimized[0], TCGCContainer)
    assert optimized[0].n_chunk == 2


def test_memory_optimization_level3_skips_when_no_temporal_split_candidates(
    monkeypatch,
):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    peak_calls = {"count": 0}
    profile_calls = {"count": 0}

    monkeypatch.setattr(
        memopt_pipeline,
        "_train_peak_memory",
        lambda *args, **kwargs: peak_calls.__setitem__("count", peak_calls["count"] + 1),
    )
    monkeypatch.setattr(
        memopt_pipeline,
        "_train_memory_profile",
        lambda *args, **kwargs: profile_calls.__setitem__(
            "count", profile_calls["count"] + 1
        ),
    )

    net = nn.Sequential(neuron.PSN(T=4))
    optimized = memopt.memory_optimization(
        net,
        neuron.PSN,
        dummy_input=(torch.randn(4, 2, 4),),
        compress_x=False,
        level=3,
        warmup_in_main_process=False,
    )

    assert isinstance(optimized[0], GCContainer)
    assert peak_calls["count"] == 0
    assert profile_calls["count"] == 0


def test_memory_optimization_level3_respects_split_budgets(monkeypatch):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    peak_calls = {"count": 0}
    profile_calls = {"count": 0}

    def fake_train_peak_memory(*args, **kwargs):
        peak_calls["count"] += 1
        return 100, 100

    def fake_train_memory_profile(*args, **kwargs):
        profile_calls["count"] += 1
        return _profile_result_with_optional_peak(
            [("0", 0.0), ("1", 0.0)], kwargs.get("return_peak", False)
        )

    monkeypatch.setattr(memopt_pipeline, "_train_peak_memory", fake_train_peak_memory)
    monkeypatch.setattr(
        memopt_pipeline, "_train_memory_profile", fake_train_memory_profile
    )

    net = nn.Sequential(TemporalSplitBlock(), TemporalSplitBlock())
    optimized = memopt.memory_optimization(
        net,
        TemporalSplitBlock,
        dummy_input=(torch.randn(4, 2, 4),),
        compress_x=False,
        level=3,
        max_split_rounds=1,
        max_candidates_per_round=1,
    )

    assert isinstance(optimized[0], GCContainer)
    assert isinstance(optimized[1], GCContainer)
    assert profile_calls["count"] == 1
    assert peak_calls["count"] == 2


def test_memory_optimization_level3_retries_blocked_candidates_after_improvement(
    monkeypatch,
):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    peak_calls = {"count": 0}
    profile_values = iter(
        [
            [("0", 0.0), ("1", 0.0)],
            [("0", 0.0), ("1", 0.0)],
            [],
        ]
    )
    temporal_attempts = {"0": 0, "1": 0}

    def fake_temporal_split(block, factor=2):
        if isinstance(block[0], UnsplittableBlock):
            temporal_attempts["0"] += 1
            return None
        temporal_attempts["1"] += 1
        return TCGCContainer(
            block.x_compressor,
            *block,
            n_chunk=getattr(block, "n_chunk", 1) * factor,
            n_seq_inputs=getattr(block, "n_seq_inputs", 1),
            n_outputs=getattr(block, "n_outputs", 1),
        )

    def fake_train_peak_memory(*args, **kwargs):
        peak_calls["count"] += 1
        if peak_calls["count"] == 1:
            return 80, 80
        if peak_calls["count"] == 2:
            return 60, 60
        return 50, 50
    monkeypatch.setattr(memopt_pipeline, "_train_peak_memory", fake_train_peak_memory)
    monkeypatch.setattr(
        memopt_pipeline,
        "_train_memory_profile",
        lambda *args, **kwargs: _profile_result_with_optional_peak(
            next(profile_values), kwargs.get("return_peak", False)
        ),
    )
    monkeypatch.setattr(
        memopt_pipeline, "_temporally_split_gc_container", fake_temporal_split
    )

    net = nn.Sequential(
        UnsplittableBlock(),
        TemporalOnlyBlock(),
    )
    optimized = memopt.memory_optimization(
        net,
        (UnsplittableBlock, TemporalOnlyBlock),
        dummy_input=(torch.randn(4, 2, 4),),
        compress_x=False,
        level=3,
        max_candidates_per_round=2,
    )

    assert isinstance(optimized[0], GCContainer)
    assert isinstance(optimized[1], TCGCContainer)
    assert temporal_attempts["0"] == 2
    assert temporal_attempts["1"] == 2


def test_memory_optimization_level4_unwraps_gc_container_when_memory_allows(
    monkeypatch,
):
    monkeypatch.setattr(memopt_pipeline, "resolve_device", lambda: "cpu")

    peak_calls = {"count": 0}
    profile_values = iter([
        [],
        [],
    ])
    def fake_train_peak_memory(*args, **kwargs):
        peak_calls["count"] += 1
        return (100, 100) if peak_calls["count"] == 1 else (90, 90)
    monkeypatch.setattr(memopt_pipeline, "_train_peak_memory", fake_train_peak_memory)
    monkeypatch.setattr(
        memopt_pipeline,
        "_train_memory_profile",
        lambda *args, **kwargs: _profile_result_with_optional_peak(
            next(profile_values), kwargs.get("return_peak", False)
        ),
    )
    monkeypatch.setattr(
        memopt_pipeline,
        "_inference_time_profile",
        lambda *args, **kwargs: _result_row("0"),
    )

    net = nn.Sequential(TargetBlock())
    optimized = memopt.memory_optimization(
        net,
        TargetBlock,
        dummy_input=(torch.randn(2, 4),),
        compress_x=False,
        level=4,
    )

    assert isinstance(optimized[0], TargetBlock)


def test_input_compressed_gc():
    """Test InputCompressedGC functionality."""
    # Test without gradients
    with torch.no_grad():
        x = torch.randn(5, 3)
        weight = torch.randn(4, 3)
        result = input_compressed_gc(
            simple_forward_fn, NullSpikeCompressor(), x, weight
        )
        expected = torch.matmul(x, weight.t())
        assert torch.allclose(result, expected)
        assert not result.requires_grad

    # Test with gradients
    x = torch.randn(5, 3, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)

    result = input_compressed_gc(simple_forward_fn, NullSpikeCompressor(), x, weight)
    expected = torch.matmul(x, weight.t())

    assert torch.allclose(result, expected)
    assert result.requires_grad

    # Test backward pass
    loss = result.sum()
    loss.backward()

    assert x.grad is not None
    assert weight.grad is not None

    # Test with bias
    bias = torch.randn(4, requires_grad=True)
    result = input_compressed_gc(
        simple_forward_fn, NullSpikeCompressor(), x, weight, bias
    )
    expected = torch.matmul(x, weight.t()) + bias
    assert torch.allclose(result, expected)


def test_to_gc_function():
    """Test to_gc_function decorator and converter."""
    compressor = MockCompressor()

    # Test decorator mode
    @to_gc_function(compressor)
    def decorated_forward(x, weight):
        return torch.matmul(x, weight.t())

    x = torch.randn(5, 3, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)
    result = decorated_forward(x, weight)
    expected = torch.matmul(x, weight.t())
    assert torch.allclose(result, expected)
    assert result.requires_grad

    # Test conversion mode
    def simple_forward(x, weight):
        return torch.matmul(x, weight.t())

    converted_forward = to_gc_function(compressor, simple_forward)
    result = converted_forward(x, weight)
    assert torch.allclose(result, expected)
    assert result.requires_grad


def test_gc_container():
    """Test GCContainer module."""
    compressor = NullSpikeCompressor()
    layer1 = nn.Linear(10, 20)
    layer2 = nn.ReLU()
    container = GCContainer(compressor, layer1, layer2)
    x = torch.randn(3, 10, requires_grad=True)
    result = container(x)
    expected = layer2(layer1(x))
    repr_str = container.extra_repr()
    assert len(container) == 2
    assert container[0] == layer1
    assert container[1] == layer2
    assert isinstance(container.x_compressor, NullSpikeCompressor)
    assert torch.allclose(result, expected)
    assert result.requires_grad
    assert "x_compressor=NullSpikeCompressor" in repr_str

    container_null = GCContainer(None, layer1)
    assert isinstance(container_null.x_compressor, NullSpikeCompressor)

    container_stateful = GCContainer(
        compressor,
        neuron.IFNode(step_mode="m"),
    )
    result = container_stateful(x)
    assert len(container_stateful) == 1
    assert isinstance(container_stateful[0], neuron.IFNode)


def test_tcgc_container():
    """Test TCGCContainer module."""
    compressor = NullSpikeCompressor()
    layer = [nn.Linear(10, 5), nn.ReLU()]
    net = nn.Sequential(*layer)
    container = TCGCContainer(compressor, *layer, n_chunk=4)
    assert len(container) == 2
    assert container.n_chunk == 4

    x_seq = torch.randn(8, 3, 10, requires_grad=True)  # 8 time steps
    result = container(x_seq)
    expected = net(x_seq)
    assert torch.allclose(result, expected)
    assert result.requires_grad

    container_single = TCGCContainer(compressor, *layer, n_chunk=1)
    result_single = container_single(x_seq)
    assert torch.allclose(result_single, expected)

    repr_str = container.extra_repr()
    assert "x_compressor=NullSpikeCompressor" in repr_str
    assert "n_chunk=4" in repr_str


def test_integration():
    """Integration tests for checkpointing functionality."""
    compressor = MockCompressor()

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(50, 100),
                neuron.LIFNode(),
                nn.Linear(100, 50),
                neuron.IFNode(),
                nn.Linear(50, 10),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.layers(x)

    net = SimpleNet()
    gc_net = GCContainer(compressor, *net.layers)
    net = copy.deepcopy(net)

    x = torch.randn(16, 50, requires_grad=True)
    regular_result = net(x)
    gc_result = gc_net(x)
    assert torch.allclose(regular_result, gc_result, atol=1e-6)

    regular_loss = regular_result.sum()
    regular_loss.backward()
    gc_loss = gc_result.sum()
    gc_loss.backward()
    for param, gc_param in zip(net.parameters(), gc_net.parameters()):
        if param.grad is not None and gc_param.grad is not None:
            assert torch.allclose(param.grad, gc_param.grad, atol=1e-5)


if __name__ == "__main__":
    test_thread_local_functions()
    test_autocast_query()
    test_argument_separate_combine()
    test_input_compressed_gc()
    test_to_gc_function()
    test_gc_container()
    test_tcgc_container()
    test_integration()
