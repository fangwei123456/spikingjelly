# ruff: noqa: F401
import copy
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

from spikingjelly.activation_based import functional, layer, neuron
from spikingjelly.activation_based.ann2snn.operators import TDLinear
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
    TENSOR_PARALLEL_AVAILABLE,
    TensorShardMemoryModule,
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE,
    make_tensor_shard_memory_module,
    analyze,
    apply,
    apply_pipeline_stage_memopt,
    build_snn_optimizer,
    ensure_distributed_initialized,
    plan,
    recommend_pipeline_memopt_stages,
    recommend_snn_distributed_strategy,
    resolve_tensor_parallel_group_size,
)
from spikingjelly.activation_based.distributed import dtensor as distributed_dtensor
from spikingjelly.activation_based.distributed import api as distributed_api
from spikingjelly.activation_based.distributed.adapters import (
    build_cifar10dvs_vgg_eager_policy,
    build_spikformer_eager_policy,
    list_adapters,
    resolve_adapter,
)
from spikingjelly.activation_based.distributed.analysis import (
    SNNDistributedAnalysis as CanonicalSNNDistributedAnalysis,
    analyze_snn_distributed_capability,
)
from spikingjelly.activation_based.distributed.config import (
    EagerParallelPolicy,
    SNNDistributedConfig,
)
from spikingjelly.activation_based.distributed.data_parallel import (
    materialize_dtensor_output,
)
from spikingjelly.activation_based.distributed.execution import (
    build_eager_config,
    configure_snn_distributed,
)
from spikingjelly.activation_based.distributed.fsdp import apply_snn_fsdp2
from spikingjelly.activation_based.distributed.metrics import (
    PreparedModelOutput,
    prepare_classification_output,
)
from spikingjelly.activation_based.distributed.mesh import (
    resolve_data_parallel_partition,
)
from spikingjelly.activation_based.distributed.pipeline.cifar10dvs_vgg import (
    _CIFAR10DVSVGGPipelineStage,
    _build_cifar10dvs_vgg_pipeline_module,
    configure_cifar10dvs_vgg_pipeline,
)
from spikingjelly.activation_based.distributed.pipeline.partition import (
    parse_pipeline_layout,
    resolve_pipeline_schedule_kind,
)
from spikingjelly.activation_based.distributed.pipeline.runtime import (
    _MicrobatchResetStage,
    _make_pipeline_outputs_contiguous,
    _measure_module_cost,
    SNNPipelineRuntime,
)
from spikingjelly.activation_based.distributed.pipeline.spikformer import (
    _build_spikformer_pipeline_module,
    configure_spikformer_pipeline,
)
from spikingjelly.activation_based.distributed.planner import (
    SNNDistributedRecommendation as CanonicalSNNDistributedRecommendation,
    recommended_pipeline_microbatches,
)
from spikingjelly.activation_based.distributed.tensor_parallel.channel import (
    ChannelShardBatchNorm1d,
    ChannelShardBatchNorm2d,
    ChannelShardConv1d,
    ChannelShardConv2d,
)
from spikingjelly.activation_based.distributed.tensor_parallel.cifar10dvs_vgg import (
    parallelize_snn_conv_blocks,
)
from spikingjelly.activation_based.distributed.tensor_parallel.debug import (
    enable_tp_communication_debug,
    get_tp_communication_debug_stats,
    reset_tp_communication_debug_stats,
)
from spikingjelly.activation_based.distributed.tensor_parallel.linear import (
    _replace_module_by_name,
    auto_build_tensor_parallel_plan,
    parallelize_snn_module,
    wrap_tp_memory_modules,
)
from spikingjelly.activation_based.distributed.tensor_parallel.spikformer import (
    parallelize_spikformer_blocks,
    parallelize_spikformer_patch_stem,
)
from spikingjelly.activation_based.examples.memopt.models import CIFAR10DVSVGG
from spikingjelly.activation_based.memopt.checkpointing import GCContainer
from spikingjelly.activation_based.model.spikformer import spikformer_ti
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


def _reset_net(net: nn.Module):
    for m in net.modules():
        if hasattr(m, "reset"):
            m.reset()


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


__all__ = [name for name in globals() if not name.startswith("__")]
