# ruff: noqa: F401
import copy
import inspect
import importlib.util
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import TensorDataset

from spikingjelly.activation_based import layer, neuron
from spikingjelly.activation_based.functional import (
    collect_reset_modules,
    reset_collected_modules,
)
from spikingjelly.activation_based.distributed import (
    DistributedFeatureSet,
    DTENSOR_AVAILABLE,
    FSDP2_AVAILABLE,
    PIPELINING_AVAILABLE,
    SNNDistributedPlan,
    SNNDistributedRuntime,
    SNNDistributedTopology,
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE,
    TENSOR_PARALLEL_AVAILABLE,
    TensorShardMemoryModule,
    analyze,
    apply,
    apply_pipeline_stage_memopt,
    build_snn_optimizer,
    make_tensor_shard_memory_module,
    plan,
    recommend_pipeline_memopt_stages,
    recommend_snn_distributed_strategy,
    resolve_tensor_parallel_group_size,
)
from spikingjelly.activation_based.distributed.analysis import (
    SNNDistributedAnalysis as CanonicalSNNDistributedAnalysis,
    analyze_snn_distributed_capability as canonical_analyze_snn_distributed_capability,
)
from spikingjelly.activation_based.distributed.adapters import (
    list_adapters,
    resolve_adapter,
)
from spikingjelly.activation_based.distributed.config import (
    EagerParallelPolicy,
    SNNDistributedConfig as CanonicalSNNDistributedConfig,
)
from spikingjelly.activation_based.distributed.data_parallel import (
    materialize_dtensor_output as canonical_materialize_dtensor_output,
    prepare_snn_data_parallel as canonical_prepare_snn_data_parallel,
    unwrap_parallel_module as canonical_unwrap_parallel_module,
)
from spikingjelly.activation_based.distributed.execution import (
    build_eager_config,
    configure_snn_distributed as canonical_configure_snn_distributed,
)
from spikingjelly.activation_based.distributed.fsdp import (
    FSDP2_AVAILABLE as CANONICAL_FSDP2_AVAILABLE,
    _build_fsdp_mp_policy as canonical_build_fsdp_mp_policy,
    apply_snn_fsdp2 as canonical_apply_snn_fsdp2,
    fully_shard_snn_module as canonical_fully_shard_snn_module,
)
from spikingjelly.activation_based.distributed.metrics import (
    PreparedModelOutput,
    prepare_classification_output,
)
from spikingjelly.activation_based.distributed.mesh import (
    build_device_mesh as canonical_build_device_mesh,
    ensure_distributed_initialized as canonical_ensure_distributed_initialized,
    resolve_data_parallel_partition as canonical_resolve_data_parallel_partition,
    resolve_tensor_parallel_group_size as canonical_resolve_tensor_parallel_group_size,
)
from spikingjelly.activation_based.distributed.optimizer import (
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE as CANONICAL_ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE,
    build_snn_optimizer as canonical_build_snn_optimizer,
)
from spikingjelly.activation_based.distributed import dtensor as distributed_dtensor
from spikingjelly.activation_based.distributed import api as distributed_api
from spikingjelly.activation_based.distributed.planner import (
    SNN_DISTRIBUTED_PREFERENCES as CANONICAL_SNN_DISTRIBUTED_PREFERENCES,
    SNNDistributedRecommendation as CanonicalSNNDistributedRecommendation,
    recommend_snn_distributed_strategy as canonical_recommend_snn_distributed_strategy,
    recommended_pipeline_microbatches as canonical_recommended_pipeline_microbatches,
)
from spikingjelly.activation_based.distributed.pipeline.partition import (
    parse_pipeline_layout as canonical_parse_pipeline_layout,
    resolve_pipeline_schedule_kind as canonical_resolve_pipeline_schedule_kind,
)
from spikingjelly.activation_based.distributed.pipeline.memopt import (
    apply_pipeline_stage_memopt as canonical_apply_pipeline_stage_memopt,
    recommend_pipeline_memopt_stages as canonical_recommend_pipeline_memopt_stages,
)
from spikingjelly.activation_based.distributed.pipeline.cifar10dvs_vgg import (
    _CIFAR10DVSVGGPipelineStage as CanonicalCIFAR10DVSVGGPipelineStage,
    _build_cifar10dvs_vgg_pipeline_module as canonical_build_cifar_pipeline_module,
    configure_cifar10dvs_vgg_pipeline as canonical_configure_cifar_pipeline,
)
from spikingjelly.activation_based.distributed.pipeline.runtime import (
    _MicrobatchResetStage as CanonicalMicrobatchResetStage,
    _build_snn_pipeline_runtime as canonical_build_snn_pipeline_runtime,
    _make_pipeline_outputs_contiguous as canonical_make_pipeline_outputs_contiguous,
    SNNPipelineRuntime as CanonicalSNNPipelineRuntime,
)
from spikingjelly.activation_based.distributed.pipeline.spikformer import (
    _SpikformerPipelineStage as CanonicalSpikformerPipelineStage,
    _build_spikformer_pipeline_module as canonical_build_spikformer_pipeline_module,
    configure_spikformer_pipeline as canonical_configure_spikformer_pipeline,
)
from spikingjelly.activation_based.distributed.tensor_parallel.channel import (
    ChannelShardBatchNorm1d as CanonicalChannelShardBatchNorm1d,
    ChannelShardBatchNorm2d as CanonicalChannelShardBatchNorm2d,
    ChannelShardConv1d as CanonicalChannelShardConv1d,
    ChannelShardConv2d as CanonicalChannelShardConv2d,
)
from spikingjelly.activation_based.distributed.tensor_parallel.cifar10dvs_vgg import (
    parallelize_snn_conv_blocks as canonical_parallelize_snn_conv_blocks,
)
from spikingjelly.activation_based.distributed.tensor_parallel.debug import (
    enable_tp_communication_debug as canonical_enable_tp_communication_debug,
    get_tp_communication_debug_stats as canonical_get_tp_communication_debug_stats,
    reset_tp_communication_debug_stats as canonical_reset_tp_communication_debug_stats,
)
from spikingjelly.activation_based.distributed.tensor_parallel.linear import (
    TENSOR_PARALLEL_AVAILABLE as CANONICAL_TENSOR_PARALLEL_AVAILABLE,
    _is_colwise_local_style as canonical_is_colwise_local_style,
    _iter_named_modules_under_roots as canonical_iter_named_modules_under_roots,
    _make_colwise_parallel as canonical_make_colwise_parallel,
    _normalize_parallel_style as canonical_normalize_parallel_style,
    _replace_module_by_name as canonical_replace_module_by_name,
    auto_build_tensor_parallel_plan as canonical_auto_build_tensor_parallel_plan,
    parallelize_snn_module as canonical_parallelize_snn_module,
    wrap_tp_memory_modules as canonical_wrap_tp_memory_modules,
)
from spikingjelly.activation_based.distributed.tensor_parallel.state import (
    make_tensor_shard_memory_module as canonical_make_tensor_shard_memory_module,
)
from spikingjelly.activation_based.distributed.tensor_parallel.spikformer import (
    parallelize_spikformer_blocks as canonical_parallelize_spikformer_blocks,
    parallelize_spikformer_patch_stem as canonical_parallelize_spikformer_patch_stem,
)
from spikingjelly.activation_based.distributed.dtensor import (
    SNNDistributedConfig,
    analyze_snn_distributed_capability,
    auto_build_tensor_parallel_plan,
    configure_snn_distributed,
    materialize_dtensor_output,
    resolve_data_parallel_partition,
)
from spikingjelly.activation_based.examples.memopt.models import CIFAR10DVSVGG
from spikingjelly.activation_based.memopt.checkpointing import GCContainer
from spikingjelly.activation_based.model.spikformer import spikformer_ti
from spikingjelly.activation_based import functional
from test.activation_based._distributed_test_utils import single_rank_process_group


_TRAIN_DISTRIBUTED_PATH = (
    Path(__file__).resolve().parents[2]
    / "spikingjelly"
    / "activation_based"
    / "examples"
    / "memopt"
    / "train_distributed.py"
)
_TRAIN_DISTRIBUTED_SPEC = importlib.util.spec_from_file_location(
    "train_distributed", _TRAIN_DISTRIBUTED_PATH
)


def _load_train_distributed_module():
    module = importlib.util.module_from_spec(_TRAIN_DISTRIBUTED_SPEC)
    stub_name = "spikingjelly.activation_based.examples.memopt.data_module"
    cleanup_stub = stub_name not in sys.modules
    if cleanup_stub:
        import types

        stub = types.ModuleType(stub_name)

        class _DummyDataModule:  # pragma: no cover - only used to satisfy imports
            pass

        stub.CIFAR10DVSDataModule = _DummyDataModule
        sys.modules[stub_name] = stub
    assert _TRAIN_DISTRIBUTED_SPEC.loader is not None
    try:
        _TRAIN_DISTRIBUTED_SPEC.loader.exec_module(module)
    finally:
        if cleanup_stub:
            sys.modules.pop(stub_name, None)
    return module


class ToyDistributedSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            layer.Linear(8, 16, step_mode="m"),
            neuron.IFNode(step_mode="m"),
            layer.Linear(16, 4, step_mode="m"),
        )

    def forward(self, x):
        return self.features(x)


class _ToyResetCounter(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _ToyNonCallableReset(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset = 1


_DTENSOR_PUBLIC_NAMES = (
    "SNNDistributedConfig",
    "SNNPipelineRuntime",
    "SNNDistributedAnalysis",
    "SNNDistributedRecommendation",
    "TensorShardMemoryModule",
    "make_tensor_shard_memory_module",
    "ChannelShardConv2d",
    "ChannelShardConv1d",
    "ChannelShardBatchNorm2d",
    "ChannelShardBatchNorm1d",
    "parse_pipeline_layout",
    "resolve_pipeline_schedule_kind",
    "recommended_pipeline_microbatches",
    "recommend_snn_distributed_strategy",
    "analyze_snn_distributed_capability",
    "auto_build_tensor_parallel_plan",
    "parallelize_snn_module",
    "prepare_snn_data_parallel",
    "configure_snn_distributed",
    "enable_tp_communication_debug",
    "reset_tp_communication_debug_stats",
    "get_tp_communication_debug_stats",
)


def _reset_net(net: nn.Module):
    for m in net.modules():
        if hasattr(m, "reset"):
            m.reset()


def test_dtensor_all_keeps_legacy_public_surface_importable():
    assert tuple(distributed_dtensor.__all__) == _DTENSOR_PUBLIC_NAMES
    for name in distributed_dtensor.__all__:
        assert getattr(distributed_dtensor, name) is not None


def test_package_root_reexports_keep_object_identity_and_signatures():
    import spikingjelly.activation_based.distributed as distributed

    root_to_canonical = {
        "analyze": distributed_api.analyze,
        "plan": distributed_api.plan,
        "apply": distributed_api.apply,
        "SNNDistributedPlan": SNNDistributedPlan,
        "SNNDistributedRuntime": SNNDistributedRuntime,
        "SNNDistributedTopology": SNNDistributedTopology,
        "SNNDistributedAnalysis": CanonicalSNNDistributedAnalysis,
        "make_tensor_shard_memory_module": canonical_make_tensor_shard_memory_module,
        "build_snn_optimizer": canonical_build_snn_optimizer,
        "build_device_mesh": canonical_build_device_mesh,
        "ensure_distributed_initialized": canonical_ensure_distributed_initialized,
        "recommend_snn_distributed_strategy": canonical_recommend_snn_distributed_strategy,
        "recommended_pipeline_microbatches": canonical_recommended_pipeline_microbatches,
        "resolve_data_parallel_partition": (canonical_resolve_data_parallel_partition),
        "resolve_tensor_parallel_group_size": (
            canonical_resolve_tensor_parallel_group_size
        ),
        "unwrap_parallel_module": canonical_unwrap_parallel_module,
        "enable_tp_communication_debug": canonical_enable_tp_communication_debug,
        "get_tp_communication_debug_stats": canonical_get_tp_communication_debug_stats,
        "reset_tp_communication_debug_stats": (
            canonical_reset_tp_communication_debug_stats
        ),
    }
    for name, canonical in root_to_canonical.items():
        assert getattr(distributed, name) is canonical

    assert distributed_dtensor.SNNDistributedConfig is CanonicalSNNDistributedConfig
    assert (
        distributed_dtensor.configure_snn_distributed
        is canonical_configure_snn_distributed
    )
    assert distributed_dtensor.SNNPipelineRuntime is CanonicalSNNPipelineRuntime
    assert distributed_dtensor.SNNDistributedAnalysis is CanonicalSNNDistributedAnalysis
    assert distributed_dtensor.TensorShardMemoryModule is not None
    assert (
        distributed_dtensor.SNNDistributedRecommendation
        is CanonicalSNNDistributedRecommendation
    )
    assert (
        distributed_dtensor.SNN_DISTRIBUTED_PREFERENCES
        is CANONICAL_SNN_DISTRIBUTED_PREFERENCES
    )
    assert (
        distributed_dtensor.analyze_snn_distributed_capability
        is canonical_analyze_snn_distributed_capability
    )
    assert (
        distributed_dtensor.recommend_snn_distributed_strategy
        is canonical_recommend_snn_distributed_strategy
    )
    assert (
        distributed_dtensor.recommended_pipeline_microbatches
        is canonical_recommended_pipeline_microbatches
    )
    assert distributed_dtensor.parse_pipeline_layout is canonical_parse_pipeline_layout
    assert (
        distributed_dtensor.resolve_pipeline_schedule_kind
        is canonical_resolve_pipeline_schedule_kind
    )
    assert (
        distributed_dtensor.recommend_pipeline_memopt_stages
        is canonical_recommend_pipeline_memopt_stages
    )
    assert (
        distributed_dtensor.apply_pipeline_stage_memopt
        is canonical_apply_pipeline_stage_memopt
    )
    assert (
        distributed_dtensor._make_pipeline_outputs_contiguous
        is canonical_make_pipeline_outputs_contiguous
    )
    assert distributed_dtensor._MicrobatchResetStage is CanonicalMicrobatchResetStage
    assert (
        distributed_dtensor._CIFAR10DVSVGGPipelineStage
        is CanonicalCIFAR10DVSVGGPipelineStage
    )
    assert (
        distributed_dtensor._SpikformerPipelineStage is CanonicalSpikformerPipelineStage
    )
    assert (
        distributed_dtensor._build_snn_pipeline_runtime
        is canonical_build_snn_pipeline_runtime
    )
    assert (
        distributed_dtensor._build_cifar10dvs_vgg_pipeline_module
        is canonical_build_cifar_pipeline_module
    )
    assert (
        distributed_dtensor._build_spikformer_pipeline_module
        is canonical_build_spikformer_pipeline_module
    )
    assert (
        distributed_dtensor.configure_cifar10dvs_vgg_pipeline
        is canonical_configure_cifar_pipeline
    )
    assert (
        distributed_dtensor.configure_spikformer_pipeline
        is canonical_configure_spikformer_pipeline
    )
    assert distributed_dtensor.build_device_mesh is canonical_build_device_mesh
    assert (
        distributed_dtensor.ensure_distributed_initialized
        is canonical_ensure_distributed_initialized
    )
    assert (
        distributed_dtensor.resolve_data_parallel_partition
        is canonical_resolve_data_parallel_partition
    )
    assert (
        distributed_dtensor.resolve_tensor_parallel_group_size
        is canonical_resolve_tensor_parallel_group_size
    )
    assert distributed_dtensor.build_snn_optimizer is canonical_build_snn_optimizer
    assert (
        distributed_dtensor.prepare_snn_data_parallel
        is canonical_prepare_snn_data_parallel
    )
    assert (
        distributed_dtensor.unwrap_parallel_module is canonical_unwrap_parallel_module
    )
    assert (
        distributed_dtensor.materialize_dtensor_output
        is canonical_materialize_dtensor_output
    )
    assert distributed_dtensor.FSDP2_AVAILABLE is CANONICAL_FSDP2_AVAILABLE
    assert (
        distributed_dtensor.ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE
        is CANONICAL_ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE
    )
    assert distributed_dtensor.apply_snn_fsdp2 is canonical_apply_snn_fsdp2
    assert (
        distributed_dtensor.fully_shard_snn_module is canonical_fully_shard_snn_module
    )
    assert distributed_dtensor._build_fsdp_mp_policy is canonical_build_fsdp_mp_policy
    assert (
        distributed_dtensor.TENSOR_PARALLEL_AVAILABLE
        is CANONICAL_TENSOR_PARALLEL_AVAILABLE
    )
    assert distributed_dtensor._make_colwise_parallel is canonical_make_colwise_parallel
    assert (
        distributed_dtensor._normalize_parallel_style
        is canonical_normalize_parallel_style
    )
    assert (
        distributed_dtensor._is_colwise_local_style is canonical_is_colwise_local_style
    )
    assert (
        distributed_dtensor._iter_named_modules_under_roots
        is canonical_iter_named_modules_under_roots
    )
    assert (
        distributed_dtensor._replace_module_by_name is canonical_replace_module_by_name
    )
    assert (
        distributed_dtensor.auto_build_tensor_parallel_plan
        is canonical_auto_build_tensor_parallel_plan
    )
    assert (
        distributed_dtensor.wrap_tp_memory_modules is canonical_wrap_tp_memory_modules
    )
    assert (
        distributed_dtensor.parallelize_snn_module is canonical_parallelize_snn_module
    )
    assert (
        distributed_dtensor.make_tensor_shard_memory_module
        is canonical_make_tensor_shard_memory_module
    )
    assert distributed_dtensor.ChannelShardConv2d is CanonicalChannelShardConv2d
    assert distributed_dtensor.ChannelShardConv1d is CanonicalChannelShardConv1d
    assert (
        distributed_dtensor.ChannelShardBatchNorm2d is CanonicalChannelShardBatchNorm2d
    )
    assert (
        distributed_dtensor.ChannelShardBatchNorm1d is CanonicalChannelShardBatchNorm1d
    )
    assert (
        distributed_dtensor.parallelize_snn_conv_blocks
        is canonical_parallelize_snn_conv_blocks
    )
    assert (
        distributed_dtensor.parallelize_spikformer_blocks
        is canonical_parallelize_spikformer_blocks
    )
    assert (
        distributed_dtensor.parallelize_spikformer_patch_stem
        is canonical_parallelize_spikformer_patch_stem
    )
    assert (
        distributed_dtensor.enable_tp_communication_debug
        is canonical_enable_tp_communication_debug
    )
    assert (
        distributed_dtensor.get_tp_communication_debug_stats
        is canonical_get_tp_communication_debug_stats
    )
    assert (
        distributed_dtensor.reset_tp_communication_debug_stats
        is canonical_reset_tp_communication_debug_stats
    )

    signature_pairs = (
        (distributed.analyze, distributed_api.analyze),
        (distributed.plan, distributed_api.plan),
        (distributed.apply, distributed_api.apply),
        (
            distributed_dtensor.configure_snn_distributed,
            canonical_configure_snn_distributed,
        ),
        (
            distributed_dtensor.recommend_snn_distributed_strategy,
            canonical_recommend_snn_distributed_strategy,
        ),
    )
    for public, canonical in signature_pairs:
        assert inspect.signature(public) == inspect.signature(canonical)


def test_spikformer_pipeline_builder_rejects_4d_input_without_T():
    model = spikformer_ti(
        T=2, img_size_h=32, img_size_w=32, num_classes=4, backend="torch"
    ).eval()
    model.T = None

    with pytest.raises(RuntimeError, match="module.T to be set"):
        canonical_build_spikformer_pipeline_module(
            model,
            num_logical_stages=2,
            example_input=torch.randn(1, 3, 32, 32),
        )


def test_spikformer_pipeline_builder_rejects_invalid_input_rank():
    model = spikformer_ti(
        T=2, img_size_h=32, img_size_w=32, num_classes=4, backend="torch"
    ).eval()

    with pytest.raises(ValueError, match="expected 4D .* or 5D"):
        canonical_build_spikformer_pipeline_module(
            model,
            num_logical_stages=2,
            example_input=torch.randn(3, 32, 32),
        )


def _train_runtime(mode: str = "tp"):
    train_distributed = _load_train_distributed_module()
    runtime = train_distributed.DistributedRuntime(
        mode=mode,
        is_distributed=False,
        rank=0,
        world_size=4,
        local_rank=0,
        device=torch.device("cpu"),
    )
    return train_distributed, runtime


def _train_args(**overrides):
    values = {
        "backend": "torch",
        "batch_size": 1,
        "T": 2,
        "distributed_mode": "tp",
        "memopt_level": 0,
        "memopt_compress_x": False,
        "mesh_shape": None,
        "disable_classifier_tp": False,
        "disable_conv_tp": False,
        "tp_mesh_dim": 0,
        "dp_mesh_dim": None,
        "pp_microbatches": None,
        "pp_schedule": "auto",
        "pp_virtual_stages": 1,
        "pp_layout": None,
        "pp_delay_wgrad": False,
        "pp_memopt_stage_budget_ratio": 0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)
