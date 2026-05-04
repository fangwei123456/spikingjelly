import copy
import importlib.util
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from spikingjelly.activation_based import layer, neuron
from spikingjelly.activation_based.distributed import (
    DTENSOR_AVAILABLE,
    FSDP2_AVAILABLE,
    PIPELINING_AVAILABLE,
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE,
    TENSOR_PARALLEL_AVAILABLE,
    SNNDistributedConfig,
    TensorShardMemoryModule,
    analyze_snn_distributed_capability,
    apply_pipeline_stage_memopt,
    auto_build_tensor_parallel_plan,
    build_snn_optimizer,
    configure_cifar10dvs_vgg_distributed,
    configure_cifar10dvs_vgg_fsdp2,
    configure_spikformer_distributed,
    configure_spikformer_fsdp2,
    configure_snn_distributed,
    materialize_dtensor_output,
    recommend_pipeline_memopt_stages,
    recommend_snn_distributed_strategy,
    resolve_data_parallel_partition,
    resolve_tensor_parallel_group_size,
)
from spikingjelly.activation_based.distributed import dtensor as distributed_dtensor
from spikingjelly.activation_based.examples.memopt.models import CIFAR10DVSVGG
from spikingjelly.activation_based.memopt.checkpointing import GCContainer
from spikingjelly.activation_based.model.spikformer import spikformer_ti
from spikingjelly.activation_based import functional


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


def _reset_net(net: nn.Module):
    for m in net.modules():
        if hasattr(m, "reset"):
            m.reset()


@contextmanager
def _single_rank_process_group():
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            raise RuntimeError(
                "_single_rank_process_group() requires world_size == 1 "
                "when reusing an initialized process group."
            )
        yield
        return

    fd, path = tempfile.mkstemp()
    os.close(fd)
    init_method = "file:///" + path.replace("\\", "/")
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def test_analyze_snn_distributed_capability_finds_state_and_linear_targets():
    model = ToyDistributedSNN()
    analysis = analyze_snn_distributed_capability(model, tensor_parallel_roots=["features"])

    assert "features.1" in analysis.memory_module_names
    assert analysis.tensor_parallel_candidate_names == ("features.0", "features.2")
    assert analysis.unsupported_tensor_parallel_names == ()
    assert any("Stateful neuron modules remain local/replicated" in note for note in analysis.notes)


@pytest.mark.skipif(
    not (DTENSOR_AVAILABLE and TENSOR_PARALLEL_AVAILABLE),
    reason="DTensor tensor-parallel APIs are unavailable in the current PyTorch build.",
)
def test_auto_tensor_parallel_plan_and_forward_match_single_rank():
    with _single_rank_process_group():
        torch.manual_seed(0)
        baseline = ToyDistributedSNN()
        candidate = copy.deepcopy(baseline)

        x = torch.randn(5, 2, 8)
        _reset_net(baseline)
        reference = baseline(x)

        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1,),
            tensor_parallel_roots=["features"],
            auto_tensor_parallel=True,
            enable_data_parallel=False,
        )
        distributed_model, mesh, analysis = configure_snn_distributed(candidate, config)

        assert mesh.ndim == 1
        assert analysis.tensor_parallel_candidate_names == ("features.0", "features.2")

        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))

        torch.testing.assert_close(reference, result)
        assert isinstance(distributed_model.features[1], TensorShardMemoryModule)

        plan = auto_build_tensor_parallel_plan(candidate, tensor_parallel_roots=["features"])
        assert set(plan.keys()) == {"features.0", "features.2"}


@pytest.mark.skipif(
    not DTENSOR_AVAILABLE,
    reason="DTensor DeviceMesh APIs are unavailable in the current PyTorch build.",
)
def test_configure_snn_distributed_supports_data_parallel_only():
    with _single_rank_process_group():
        model = ToyDistributedSNN()
        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1,),
            auto_tensor_parallel=False,
            enable_data_parallel=True,
            dp_mesh_dim=0,
        )
        distributed_model, mesh, _ = configure_snn_distributed(model, config)

        assert isinstance(distributed_model, DistributedDataParallel)
        assert mesh.ndim == 1

        x = torch.randn(3, 2, 8)
        _reset_net(distributed_model.module)
        y = distributed_model(x)
        assert y.shape == (3, 2, 4)


@pytest.mark.skipif(
    not ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE,
    reason="ZeroRedundancyOptimizer is unavailable in the current PyTorch build.",
)
def test_build_snn_optimizer_supports_zero_for_dp():
    with _single_rank_process_group():
        model = ToyDistributedSNN()
        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1,),
            auto_tensor_parallel=False,
            enable_data_parallel=True,
            dp_mesh_dim=0,
        )
        distributed_model, _, _ = configure_snn_distributed(model, config)
        optimizer = build_snn_optimizer(
            distributed_model,
            mode="dp",
            lr=1e-3,
            optimizer_sharding="zero",
        )
        assert "Zero" in type(optimizer).__name__
        x = torch.randn(3, 2, 8)
        y = torch.randn(3, 2, 4)
        _reset_net(distributed_model.module)
        loss = torch.nn.functional.mse_loss(distributed_model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def test_build_snn_optimizer_rejects_zero_outside_dp():
    model = ToyDistributedSNN()
    with pytest.raises(ValueError, match="pure 'dp' mode only"):
        build_snn_optimizer(
            model,
            mode="tp",
            lr=1e-3,
            optimizer_sharding="zero",
        )


@pytest.mark.skipif(
    not DTENSOR_AVAILABLE,
    reason="DTensor DeviceMesh APIs are unavailable in the current PyTorch build.",
)
def test_configure_snn_distributed_rejects_ddp_plus_tp():
    with _single_rank_process_group():
        model = ToyDistributedSNN()
        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1, 1),
            tensor_parallel_roots=["features"],
            auto_tensor_parallel=True,
            enable_data_parallel=True,
            tp_mesh_dim=1,
            dp_mesh_dim=0,
        )
        with pytest.raises(NotImplementedError, match="FSDP2 \\+ TP"):
            configure_snn_distributed(model, config)


@pytest.mark.skipif(
    not DTENSOR_AVAILABLE,
    reason="DTensor DeviceMesh APIs are unavailable in the current PyTorch build.",
)
def test_configure_snn_distributed_rejects_ddp_plus_experimental_conv_tp():
    with _single_rank_process_group():
        model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1, 1),
            auto_tensor_parallel=False,
            experimental_conv_tensor_parallel=True,
            conv_tensor_parallel_roots=["features"],
            enable_data_parallel=True,
            tp_mesh_dim=1,
            dp_mesh_dim=0,
        )
        with pytest.raises(NotImplementedError, match="FSDP2 \\+ TP"):
            configure_snn_distributed(model, config)


@pytest.mark.skipif(
    not DTENSOR_AVAILABLE,
    reason="DTensor DeviceMesh APIs are unavailable in the current PyTorch build.",
)
def test_configure_snn_distributed_requires_dp_mesh_dim_for_multidim_data_parallel():
    with _single_rank_process_group():
        model = ToyDistributedSNN()
        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1, 1),
            auto_tensor_parallel=False,
            enable_data_parallel=True,
            dp_mesh_dim=None,
        )
        with pytest.raises(ValueError, match="dp_mesh_dim must be specified"):
            configure_snn_distributed(model, config)


@pytest.mark.skipif(
    not DTENSOR_AVAILABLE,
    reason="DTensor DeviceMesh APIs are unavailable in the current PyTorch build.",
)
def test_partition_helpers_respect_2d_mesh_coordinates():
    with _single_rank_process_group():
        model = ToyDistributedSNN()
        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1, 1),
            tensor_parallel_roots=["features"],
            auto_tensor_parallel=True,
            enable_data_parallel=False,
            tp_mesh_dim=1,
            dp_mesh_dim=0,
        )
        _, mesh, _ = configure_snn_distributed(model, config)
        assert resolve_data_parallel_partition(mesh, dp_mesh_dim=0, sharded_by_data_parallel=False) == (1, 0)
        assert resolve_data_parallel_partition(mesh, dp_mesh_dim=0, sharded_by_data_parallel=True) == (1, 0)
        assert resolve_tensor_parallel_group_size(mesh, tp_mesh_dim=1, tensor_parallel_enabled=True) == 1


@pytest.mark.skipif(
    not DTENSOR_AVAILABLE,
    reason="DTensor DeviceMesh APIs are unavailable in the current PyTorch build.",
)
def test_configure_snn_distributed_supports_experimental_conv_tp_on_real_snn():
    with _single_rank_process_group():
        torch.manual_seed(0)
        baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch")
        candidate = copy.deepcopy(baseline)
        baseline.eval()
        candidate.eval()

        x = torch.randn(1, 2, 2, 48, 48)
        _reset_net(baseline)
        reference = baseline(x)

        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1,),
            auto_tensor_parallel=True,
            tensor_parallel_roots=["classifier"],
            experimental_conv_tensor_parallel=True,
            conv_tensor_parallel_roots=["features"],
            enable_data_parallel=False,
        )
        distributed_model, _, _ = configure_snn_distributed(candidate, config)

        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))

        torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
        assert "ChannelShardConv2d" in type(distributed_model.features[0].proj_bn[-2]).__name__
        assert isinstance(distributed_model.features[0].neuron, TensorShardMemoryModule)
        assert distributed_model.features[0].neuron.inner.v.shape[1] == distributed_model.features[0].proj_bn[-2].out_channels


@pytest.mark.skipif(
    not DTENSOR_AVAILABLE,
    reason="DTensor DeviceMesh APIs are unavailable in the current PyTorch build.",
)
def test_high_level_cifar10dvs_vgg_helper():
    with _single_rank_process_group():
        model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        _distributed_model, mesh, analysis = configure_cifar10dvs_vgg_distributed(
            model,
            device_type="cpu",
            mesh_shape=(1,),
            enable_data_parallel=False,
        )
        assert mesh.ndim == 1
        assert analysis.tensor_parallel_candidate_names == ("classifier.0",)


@pytest.mark.skipif(
    not FSDP2_AVAILABLE,
    reason="FSDP2 fully_shard is unavailable in the current PyTorch build.",
)
def test_cifar10dvs_vgg_fsdp2_helper_single_rank():
    with _single_rank_process_group():
        torch.manual_seed(0)
        baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        candidate = copy.deepcopy(baseline).eval()
        distributed_model, mesh, _ = configure_cifar10dvs_vgg_fsdp2(
            candidate,
            device_type="cpu",
            mesh_shape=(1,),
            enable_classifier_tensor_parallel=False,
            enable_experimental_conv_tensor_parallel=False,
        )
        x = torch.randn(1, 2, 2, 48, 48)
        _reset_net(baseline)
        reference = baseline(x)
        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))
        torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
        assert mesh.ndim == 1


@pytest.mark.skipif(
    not (FSDP2_AVAILABLE and DTENSOR_AVAILABLE and TENSOR_PARALLEL_AVAILABLE),
    reason="FSDP2/DTensor tensor-parallel APIs are unavailable in the current PyTorch build.",
)
def test_cifar10dvs_vgg_fsdp2_tp_helper_single_rank():
    with _single_rank_process_group():
        torch.manual_seed(0)
        baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        candidate = copy.deepcopy(baseline).eval()
        distributed_model, mesh, _ = configure_cifar10dvs_vgg_fsdp2(
            candidate,
            device_type="cpu",
            mesh_shape=(1, 1),
            enable_classifier_tensor_parallel=True,
            enable_experimental_conv_tensor_parallel=True,
            tp_mesh_dim=1,
            dp_mesh_dim=0,
        )
        x = torch.randn(1, 2, 2, 48, 48)
        _reset_net(baseline)
        reference = baseline(x)
        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))
        torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
        assert mesh.ndim == 2


@pytest.mark.skipif(
    not DTENSOR_AVAILABLE,
    reason="DTensor DeviceMesh APIs are unavailable in the current PyTorch build.",
)
def test_cifar10dvs_vgg_tp_helper_after_memopt_level2_single_rank():
    with _single_rank_process_group():
        torch.manual_seed(0)
        baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        x = torch.randn(1, 2, 2, 48, 48)
        candidate = copy.deepcopy(baseline).eval()
        candidate.features[0] = GCContainer(None, candidate.features[0])
        distributed_model, _, _ = configure_cifar10dvs_vgg_distributed(
            candidate,
            device_type="cpu",
            mesh_shape=(1,),
            enable_data_parallel=False,
        )
        _reset_net(baseline)
        reference = baseline(x)
        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))
        torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not (DTENSOR_AVAILABLE and TENSOR_PARALLEL_AVAILABLE),
    reason="DTensor tensor-parallel APIs are unavailable in the current PyTorch build.",
)
@pytest.mark.skip(reason="TP-aware memopt is validated by multi-rank CUDA benchmarks; single-rank smoke is unstable across runtimes.")
def test_spikformer_tp_plus_memopt_level1_single_rank():
    pass


@pytest.mark.skipif(
    not (FSDP2_AVAILABLE and DTENSOR_AVAILABLE and TENSOR_PARALLEL_AVAILABLE),
    reason="FSDP2/DTensor tensor-parallel APIs are unavailable in the current PyTorch build.",
)
@pytest.mark.skip(reason="TP-aware memopt is validated by multi-rank CUDA benchmarks; single-rank smoke is unstable across runtimes.")
def test_spikformer_fsdp2_tp_plus_memopt_level1_single_rank():
    pass


@pytest.mark.skipif(
    not DTENSOR_AVAILABLE,
    reason="DTensor DeviceMesh APIs are unavailable in the current PyTorch build.",
)
def test_spikformer_tp_helper_after_memopt_level2_single_rank():
    with _single_rank_process_group():
        torch.manual_seed(0)
        baseline = spikformer_ti(T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch").eval()
        x = torch.randn(2, 3, 64, 64)
        candidate = copy.deepcopy(baseline).eval()
        candidate.patch_embed.stages[0] = GCContainer(None, candidate.patch_embed.stages[0])
        candidate.blocks[0] = GCContainer(None, candidate.blocks[0])
        distributed_model, _, _ = configure_spikformer_distributed(
            candidate,
            device_type="cpu",
            mesh_shape=(1,),
            enable_data_parallel=False,
            enable_head_tensor_parallel=True,
        )
        _reset_net(baseline)
        reference = baseline(x)
        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))
        torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not (DTENSOR_AVAILABLE and TENSOR_PARALLEL_AVAILABLE),
    reason="DTensor tensor-parallel APIs are unavailable in the current PyTorch build.",
)
def test_spikformer_head_tp_helper_single_rank():
    with _single_rank_process_group():
        torch.manual_seed(0)
        baseline = spikformer_ti(T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch").eval()
        candidate = copy.deepcopy(baseline).eval()
        distributed_model, mesh, analysis = configure_spikformer_distributed(
            candidate,
            device_type="cpu",
            mesh_shape=(1,),
            enable_data_parallel=False,
            enable_head_tensor_parallel=True,
        )
        x = torch.randn(2, 3, 64, 64)
        _reset_net(baseline)
        reference = baseline(x)
        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))
        torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
        assert mesh.ndim == 1
        assert analysis.tensor_parallel_candidate_names == ("head",)
        assert isinstance(distributed_model.patch_embed.stages[0].neuron, TensorShardMemoryModule)
        assert "ChannelShardConv2d" in type(distributed_model.patch_embed.stages[0].conv_bn.block[0]).__name__


@pytest.mark.skipif(
    not FSDP2_AVAILABLE,
    reason="FSDP2 fully_shard is unavailable in the current PyTorch build.",
)
def test_spikformer_fsdp2_helper_single_rank():
    with _single_rank_process_group():
        torch.manual_seed(0)
        baseline = spikformer_ti(T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch").eval()
        candidate = copy.deepcopy(baseline).eval()
        distributed_model, mesh, _ = configure_spikformer_fsdp2(
            candidate,
            device_type="cpu",
            mesh_shape=(1,),
            enable_head_tensor_parallel=False,
        )
        x = torch.randn(2, 3, 64, 64)
        _reset_net(baseline)
        reference = baseline(x)
        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))
        torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
        assert mesh.ndim == 1


def test_cifar10dvs_vgg_pipeline_module_matches_baseline():
    torch.manual_seed(0)
    baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    x = torch.randn(1, 2, 2, 48, 48)
    pipeline_module = distributed_dtensor._build_cifar10dvs_vgg_pipeline_module(
        copy.deepcopy(baseline),
        num_logical_stages=2,
        example_input=x,
    ).eval()
    reference = baseline(x)
    functional.reset_net(pipeline_module)
    result = pipeline_module(x)
    torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)


def test_spikformer_pipeline_module_matches_baseline():
    torch.manual_seed(0)
    baseline = spikformer_ti(T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch").eval()
    x = torch.randn(2, 3, 64, 64)
    pipeline_module = distributed_dtensor._build_spikformer_pipeline_module(
        copy.deepcopy(baseline),
        num_logical_stages=3,
        example_input=x,
    ).eval()
    functional.reset_net(baseline)
    reference = baseline(x)
    functional.reset_net(pipeline_module)
    result = pipeline_module(x)
    torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not PIPELINING_AVAILABLE,
    reason="torch.distributed.pipelining is unavailable in the current PyTorch build.",
)
def test_cifar10dvs_vgg_pipeline_runtime_supports_interleaved_single_rank():
    with _single_rank_process_group():
        model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        x = torch.randn(2, 2, 2, 48, 48)
        runtime = distributed_dtensor.configure_cifar10dvs_vgg_pipeline(
            copy.deepcopy(model),
            example_input=x,
            device=torch.device("cpu"),
            n_microbatches=2,
            pp_schedule="interleaved",
            pp_virtual_stages=2,
        )
        assert runtime.schedule_kind == "interleaved"
        assert runtime.virtual_pipeline_size == 2
        assert len(runtime.stage_modules) == 2


@pytest.mark.skipif(
    not PIPELINING_AVAILABLE,
    reason="torch.distributed.pipelining is unavailable in the current PyTorch build.",
)
def test_spikformer_pipeline_runtime_supports_zero_bubble_single_rank():
    with _single_rank_process_group():
        model = spikformer_ti(T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch").eval()
        x = torch.randn(2, 3, 64, 64)
        runtime = distributed_dtensor.configure_spikformer_pipeline(
            copy.deepcopy(model),
            example_input=x,
            device=torch.device("cpu"),
            n_microbatches=2,
            pp_schedule="zero_bubble",
            pp_virtual_stages=2,
            pp_delay_wgrad=True,
        )
        assert runtime.schedule_kind == "zero_bubble"
        assert runtime.delayed_wgrad is True
        assert len(runtime.stage_modules) == 2


def test_recommend_pipeline_memopt_stages_prefers_heavy_stages():
    selected = recommend_pipeline_memopt_stages((1.0, 8.0, 3.0, 7.0), stage_budget_ratio=0.5)
    assert selected == (1, 3)


def test_apply_pipeline_stage_memopt_only_wraps_selected_heavy_stage():
    torch.manual_seed(0)
    model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    stage = distributed_dtensor._CIFAR10DVSVGGPipelineStage(
        feature_modules=[copy.deepcopy(model.features[0])],
        classifier=None,
        transpose_input=True,
    ).eval()
    wrapped_stage = distributed_dtensor._MicrobatchResetStage(stage)
    runtime = distributed_dtensor.SNNPipelineRuntime(
        schedule=None,
        stage_module=wrapped_stage,
        stage_modules=(wrapped_stage,),
        local_stage_indices=(0,),
        stage_index=0,
        num_stages=2,
        device=torch.device("cpu"),
        n_microbatches=2,
        model_family="cifar10dvs_vgg",
        split_points=("stages.1",),
        stage_costs=(10.0, 1.0),
        stage_input_example=torch.randn(1, 2, 2, 48, 48),
        stage_input_examples=(torch.randn(1, 2, 2, 48, 48),),
    )
    runtime, optimize_ms, applied = apply_pipeline_stage_memopt(
        runtime,
        memopt_level=1,
        compress_x=False,
        stage_budget_ratio=0.5,
    )
    assert applied is True
    assert optimize_ms >= 0.0
    assert runtime.memopt_selected_stage_indices == (0,)
    assert isinstance(runtime.stage_module.inner.features[0], GCContainer)


def test_apply_pipeline_stage_memopt_supports_legacy_memopt_signature(monkeypatch):
    torch.manual_seed(0)
    model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    stage = distributed_dtensor._CIFAR10DVSVGGPipelineStage(
        feature_modules=[copy.deepcopy(model.features[0])],
        classifier=None,
        transpose_input=True,
    ).eval()
    wrapped_stage = distributed_dtensor._MicrobatchResetStage(stage)
    runtime = distributed_dtensor.SNNPipelineRuntime(
        schedule=None,
        stage_module=wrapped_stage,
        stage_modules=(wrapped_stage,),
        local_stage_indices=(0,),
        stage_index=0,
        num_stages=2,
        device=torch.device("cpu"),
        n_microbatches=2,
        model_family="cifar10dvs_vgg",
        split_points=("stages.1",),
        stage_costs=(10.0, 1.0),
        stage_input_example=torch.randn(1, 2, 2, 48, 48),
        stage_input_examples=(torch.randn(1, 2, 2, 48, 48),),
    )

    import spikingjelly.activation_based.memopt as memopt

    calls = {"count": 0}

    def fake_memory_optimization(module, target_types, dummy_input, compress_x, level, verbose):
        calls["count"] += 1
        return module

    monkeypatch.setattr(memopt, "memory_optimization", fake_memory_optimization)

    runtime, optimize_ms, applied = apply_pipeline_stage_memopt(
        runtime,
        memopt_level=1,
        compress_x=False,
        stage_budget_ratio=0.5,
        use_plan_cache=True,
    )

    assert applied is True
    assert optimize_ms >= 0.0
    assert calls["count"] == 1


def test_recommend_snn_distributed_strategy_speed_prefers_dp_zero():
    recommendation = recommend_snn_distributed_strategy(
        model="spikformer_ti",
        world_size=2,
        prefer="speed",
        batch_size=4,
        zero_redundancy_optimizer_available=True,
        pipelining_available=True,
        fsdp2_available=True,
        tensor_parallel_available=True,
    )
    assert recommendation.mode == "dp"
    assert recommendation.optimizer_sharding == "zero"
    assert recommendation.memopt_level == 0


def test_recommend_snn_distributed_strategy_memory_prefers_fsdp2_tp():
    recommendation = recommend_snn_distributed_strategy(
        model="spikformer_ti",
        world_size=4,
        prefer="memory",
        batch_size=4,
        zero_redundancy_optimizer_available=True,
        pipelining_available=True,
        fsdp2_available=True,
        tensor_parallel_available=True,
    )
    assert recommendation.mode == "fsdp2_tp"
    assert recommendation.memopt_level == 1
    assert recommendation.mesh_shape == (2, 2)


def test_recommend_snn_distributed_strategy_capacity_prefers_pp():
    recommendation = recommend_snn_distributed_strategy(
        model="cifar10dvs_vgg",
        world_size=4,
        prefer="capacity",
        batch_size=8,
        zero_redundancy_optimizer_available=True,
        pipelining_available=True,
        fsdp2_available=True,
        tensor_parallel_available=True,
    )
    assert recommendation.mode == "pp"
    assert recommendation.memopt_level == 1
    assert recommendation.pp_microbatches == 8
    assert recommendation.pp_schedule == "interleaved"
    assert recommendation.pp_virtual_stages == 2
    assert recommendation.pp_delay_wgrad is False


def test_recommend_snn_distributed_strategy_capacity_degrades_virtual_stages_for_small_batch():
    recommendation = recommend_snn_distributed_strategy(
        model="cifar10dvs_vgg",
        world_size=4,
        prefer="capacity",
        batch_size=4,
        zero_redundancy_optimizer_available=True,
        pipelining_available=True,
        fsdp2_available=True,
        tensor_parallel_available=True,
    )
    assert recommendation.mode == "pp"
    assert recommendation.pp_microbatches == 4
    assert recommendation.pp_schedule == "1f1b"
    assert recommendation.pp_virtual_stages == 1


def test_recommend_snn_distributed_strategy_capacity_falls_back_when_batch_too_small_for_pp():
    recommendation = recommend_snn_distributed_strategy(
        model="cifar10dvs_vgg",
        world_size=4,
        prefer="capacity",
        batch_size=2,
        zero_redundancy_optimizer_available=True,
        pipelining_available=True,
        fsdp2_available=True,
        tensor_parallel_available=True,
    )
    assert recommendation.mode == "fsdp2_tp"
    assert recommendation.memopt_level == 1
    assert recommendation.mesh_shape == (2, 2)


def test_recommended_pipeline_microbatches_rejects_too_small_batch():
    with pytest.raises(ValueError, match=r"batch_size .* must be >= num_stages"):
        distributed_dtensor.recommended_pipeline_microbatches(2, 4)


def test_parse_pipeline_layout_validates_counts():
    counts = distributed_dtensor.parse_pipeline_layout("1|2|3", 3, 6)
    assert counts == (1, 2, 3)
    with pytest.raises(ValueError, match="requires 6 units"):
        distributed_dtensor.parse_pipeline_layout("1|2|2", 3, 6)


def test_resolve_pipeline_schedule_kind_rules():
    assert distributed_dtensor.resolve_pipeline_schedule_kind("auto", 1, False) == "1f1b"
    assert distributed_dtensor.resolve_pipeline_schedule_kind("auto", 2, False) == "interleaved"
    assert distributed_dtensor.resolve_pipeline_schedule_kind("auto", 2, True) == "zero_bubble"
    with pytest.raises(ValueError, match="requires pp_virtual_stages >= 2"):
        distributed_dtensor.resolve_pipeline_schedule_kind("interleaved", 1, False)
    with pytest.raises(ValueError, match="does not support pp_virtual_stages=2"):
        distributed_dtensor.resolve_pipeline_schedule_kind("gpipe", 2, False)
    with pytest.raises(ValueError, match="does not support pp_virtual_stages=2"):
        distributed_dtensor.resolve_pipeline_schedule_kind("1f1b", 2, False)


def test_make_pipeline_outputs_contiguous_clones_views():
    base = torch.randn(2, 3, 4)
    view = base.transpose(0, 1)
    out = distributed_dtensor._make_pipeline_outputs_contiguous(view)
    torch.testing.assert_close(out, view)
    assert out.data_ptr() != view.data_ptr()


def test_cifar_pipeline_transposes_on_first_non_empty_stage():
    torch.manual_seed(0)
    baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    example = torch.randn(1, 2, 2, 48, 48)
    pipeline = distributed_dtensor._build_cifar10dvs_vgg_pipeline_module(
        copy.deepcopy(baseline),
        num_logical_stages=2,
        example_input=example,
        layout_counts=(0, len(list(baseline.features.children())) + 1),
    )
    assert pipeline.stages[0].transpose_input is False
    assert pipeline.stages[1].transpose_input is True
    functional.reset_net(baseline)
    reference = baseline(example)
    functional.reset_net(pipeline)
    result = pipeline(example)
    torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)


def test_spikformer_pipeline_attaches_patch_embed_to_first_non_empty_stage():
    torch.manual_seed(0)
    baseline = spikformer_ti(
        T=2,
        img_size_h=64,
        img_size_w=64,
        num_classes=11,
        backend="torch",
    ).eval()
    example = torch.randn(1, 3, 64, 64)
    pipeline = distributed_dtensor._build_spikformer_pipeline_module(
        copy.deepcopy(baseline),
        num_logical_stages=2,
        example_input=example,
        layout_counts=(0, len(baseline.blocks) + 2),
    )
    assert pipeline.stages[0].patch_embed is None
    assert pipeline.stages[1].patch_embed is not None
    functional.reset_net(baseline)
    reference = baseline(example)
    functional.reset_net(pipeline)
    result = pipeline(example)
    torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not (DTENSOR_AVAILABLE and TENSOR_PARALLEL_AVAILABLE),
    reason="DTensor tensor-parallel APIs are unavailable in the current PyTorch build.",
)
def test_spikformer_patch_stem_tp_helper_handles_patch_embed_root():
    with _single_rank_process_group():
        torch.manual_seed(0)
        candidate = spikformer_ti(T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch").eval()
        distributed_model, _, _ = configure_spikformer_distributed(
            candidate,
            device_type="cpu",
            mesh_shape=(1,),
            enable_data_parallel=False,
            enable_head_tensor_parallel=True,
        )
        assert "ChannelShardConv2d" in type(distributed_model.patch_embed.stages[0].conv_bn.block[0]).__name__
        assert isinstance(distributed_model.patch_embed.stages[0].neuron, TensorShardMemoryModule)


@pytest.mark.skipif(
    not (DTENSOR_AVAILABLE and TENSOR_PARALLEL_AVAILABLE),
    reason="DTensor tensor-parallel APIs are unavailable in the current PyTorch build.",
)
def test_spikformer_patch_stem_tp_helper_handles_single_stage_root_colwise():
    with _single_rank_process_group():
        torch.manual_seed(0)
        candidate = spikformer_ti(T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch").eval()
        distributed_model, _, _ = configure_snn_distributed(
            candidate,
            SNNDistributedConfig(
                device_type="cpu",
                mesh_shape=(1,),
                auto_tensor_parallel=False,
                experimental_spikformer_patch_stem_tensor_parallel=True,
                spikformer_patch_stem_tensor_parallel_roots=["patch_embed.stages.0"],
                enable_data_parallel=False,
            ),
        )
        shard_conv = distributed_model.patch_embed.stages[0].conv_bn.block[0]
        assert "ChannelShardConv2d" in type(shard_conv).__name__
        assert shard_conv.mode == "colwise"
        next_conv = distributed_model.patch_embed.stages[1].conv_bn.block[0]
        assert "ChannelShardConv2d" in type(next_conv).__name__
        assert next_conv.mode == "rowwise"


@pytest.mark.skipif(
    not (DTENSOR_AVAILABLE and TENSOR_PARALLEL_AVAILABLE),
    reason="DTensor tensor-parallel APIs are unavailable in the current PyTorch build.",
)
def test_spikformer_patch_stem_tp_helper_rejects_unpaired_isolated_root():
    with _single_rank_process_group():
        torch.manual_seed(0)
        candidate = spikformer_ti(T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch").eval()
        with pytest.raises(ValueError, match="at least two consecutive stem blocks"):
            configure_snn_distributed(
                candidate,
                SNNDistributedConfig(
                    device_type="cpu",
                    mesh_shape=(1,),
                    auto_tensor_parallel=False,
                    experimental_spikformer_patch_stem_tensor_parallel=True,
                    spikformer_patch_stem_tensor_parallel_roots=["patch_embed.stages.3"],
                    enable_data_parallel=False,
                ),
            )


def test_train_distributed_build_model_rejects_tp_without_targets():
    train_distributed = _load_train_distributed_module()
    runtime = train_distributed.DistributedRuntime(
        mode="tp",
        is_distributed=False,
        rank=0,
        world_size=1,
        local_rank=0,
        device=torch.device("cpu"),
    )
    args = SimpleNamespace(
        backend="torch",
        batch_size=1,
        T=2,
        distributed_mode="tp",
        memopt_level=0,
        memopt_compress_x=False,
        mesh_shape=None,
        disable_classifier_tp=True,
        disable_conv_tp=True,
        tp_mesh_dim=0,
        dp_mesh_dim=None,
        pp_microbatches=None,
        pp_schedule="auto",
        pp_virtual_stages=1,
        pp_layout=None,
        pp_delay_wgrad=False,
        pp_memopt_stage_budget_ratio=0.5,
    )
    with pytest.raises(ValueError, match="requires at least one tensor-parallel target"):
        train_distributed.build_model(args, runtime)


def test_train_distributed_build_model_requires_2d_mesh_for_fsdp2_tp():
    train_distributed = _load_train_distributed_module()
    runtime = train_distributed.DistributedRuntime(
        mode="fsdp2_tp",
        is_distributed=False,
        rank=0,
        world_size=1,
        local_rank=0,
        device=torch.device("cpu"),
    )
    args = SimpleNamespace(
        backend="torch",
        batch_size=1,
        T=2,
        distributed_mode="fsdp2_tp",
        memopt_level=0,
        memopt_compress_x=False,
        mesh_shape=[1],
        disable_classifier_tp=False,
        disable_conv_tp=False,
        tp_mesh_dim=0,
        dp_mesh_dim=None,
        pp_microbatches=None,
        pp_schedule="auto",
        pp_virtual_stages=1,
        pp_layout=None,
        pp_delay_wgrad=False,
        pp_memopt_stage_budget_ratio=0.5,
    )
    with pytest.raises(ValueError, match="requires an explicit 2D mesh"):
        train_distributed.build_model(args, runtime)


def test_train_distributed_reduce_classification_output_keeps_batch_major_logits():
    train_distributed = _load_train_distributed_module()
    logits = torch.randn(4, 10)
    labels = torch.tensor([0, 1, 2, 3])
    reduced_logits, reduced_labels = train_distributed.reduce_classification_output(logits, labels)
    torch.testing.assert_close(reduced_logits, logits)
    assert torch.equal(reduced_labels, labels)
    assert reduced_logits.shape == logits.shape
    assert reduced_labels.shape == labels.shape


def test_train_distributed_reduce_classification_output_reduces_time_major_logits():
    train_distributed = _load_train_distributed_module()
    logits = torch.randn(5, 4, 10)
    labels = torch.eye(10)[torch.tensor([0, 1, 2, 3])]
    reduced_logits, reduced_labels = train_distributed.reduce_classification_output(logits, labels)
    torch.testing.assert_close(reduced_logits, logits.mean(dim=0))
    assert torch.equal(reduced_labels, torch.tensor([0, 1, 2, 3]))
    assert reduced_logits.shape == (4, 10)
    assert reduced_labels.shape == (4,)


def test_train_distributed_build_data_uses_shared_sampler_for_pipeline(monkeypatch):
    train_distributed = _load_train_distributed_module()

    class DummyDataModule:
        def __init__(self, data_dir, T, batch_size, num_workers):
            self.train_set = list(range(8))
            self.test_set = list(range(4))

        def prepare_data(self):
            return None

        def setup(self, stage):
            return None

    monkeypatch.setattr(train_distributed, "CIFAR10DVSDataModule", DummyDataModule)
    runtime = SimpleNamespace(is_distributed=True, mode="pp", world_size=2)
    args = SimpleNamespace(
        data_dir="unused",
        T=4,
        batch_size=2,
        num_workers=0,
        dp_mesh_dim=None,
    )

    train_loader, val_loader, train_sampler = train_distributed.build_data(args, runtime, mesh=None)
    assert train_sampler is not None
    assert isinstance(train_loader.sampler, type(train_sampler))
    assert isinstance(val_loader.sampler, type(train_sampler))
    assert train_sampler.num_replicas == 1
    assert train_sampler.rank == 0
    assert train_loader.sampler.num_replicas == train_sampler.num_replicas
    assert train_loader.sampler.rank == train_sampler.rank
    assert val_loader.sampler.num_replicas == train_sampler.num_replicas
    assert val_loader.sampler.rank == train_sampler.rank
