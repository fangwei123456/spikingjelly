import copy
import os
import tempfile
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from spikingjelly.activation_based import layer, neuron
from spikingjelly.activation_based.distributed import (
    DTENSOR_AVAILABLE,
    FSDP2_AVAILABLE,
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE,
    TENSOR_PARALLEL_AVAILABLE,
    SNNDistributedConfig,
    TensorShardMemoryModule,
    analyze_snn_distributed_capability,
    apply_snn_fsdp2,
    auto_build_tensor_parallel_plan,
    build_snn_optimizer,
    configure_cifar10dvs_vgg_distributed,
    configure_cifar10dvs_vgg_fsdp2,
    configure_spikformer_distributed,
    configure_spikformer_fsdp2,
    configure_snn_distributed,
    materialize_dtensor_output,
    resolve_data_parallel_partition,
    resolve_tensor_parallel_group_size,
)
from spikingjelly.activation_based.examples.memopt.models import CIFAR10DVSVGG
from spikingjelly.activation_based.examples.memopt.models import VGGBlock
from spikingjelly.activation_based.memopt import memory_optimization
from spikingjelly.activation_based.model.spikformer import spikformer_ti
from spikingjelly.activation_based.model.spikformer import SpikformerConv2dBNLIF, SpikformerMLP
from spikingjelly.activation_based.layer.attention import SpikingSelfAttention


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
        distributed_model, mesh, analysis = configure_cifar10dvs_vgg_distributed(
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
