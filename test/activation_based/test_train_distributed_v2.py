import pytest
import torch
import torch.nn as nn
from spikingjelly.activation_based import distributed as sjdist
from spikingjelly.activation_based.examples.memopt.models import CIFAR10DVSVGG
from torch.utils.data import TensorDataset
from test.activation_based._distributed_test_utils import single_rank_process_group


@pytest.mark.skipif(
    not (sjdist.DTENSOR_AVAILABLE and sjdist.FSDP2_AVAILABLE),
    reason="DTensor DeviceMesh or FSDP2 APIs are unavailable in the current PyTorch build.",
)
def test_new_distributed_api_supports_manual_training_loop_single_rank():
    with single_rank_process_group():
        torch.manual_seed(0)
        model = CIFAR10DVSVGG(dropout=0.0, backend="torch").to("cpu")
        dataset = TensorDataset(
            torch.randn(4, 2, 2, 48, 48),
            torch.tensor([0, 1, 2, 3]),
        )

        analysis = sjdist.analyze(
            model,
            model_family="cifar10dvs_vgg",
        )
        distributed_plan = sjdist.plan(
            analysis=analysis,
            objective="speed",
            topology={"dp": 1},
            backend="torch",
            batch_size=2,
            model_family="cifar10dvs_vgg",
            mode="fsdp2",
            features=sjdist.DistributedFeatureSet(
                allow_experimental_conv_tp=False,
            ),
        )
        runtime = sjdist.apply(
            model=model,
            plan=distributed_plan,
            device_type="cpu",
        )
        assert runtime.plan is not None
        assert runtime.plan.mode == "fsdp2"

        optimizer = runtime.build_optimizer(
            optimizer_cls=torch.optim.SGD,
            lr=1e-3,
        )
        loader = runtime.prepare_dataloader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )
        criterion = nn.CrossEntropyLoss()

        first_param_before = next(runtime.model.parameters()).detach().clone()
        last_loss = None

        runtime.model.train()
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = runtime.model(images.float())
            logits, labels = runtime.prepare_classification_output(logits, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            runtime.reset_state()
            last_loss = loss

        assert last_loss is not None
        assert torch.isfinite(last_loss)
        first_param_after = next(runtime.model.parameters()).detach()
        assert not torch.equal(first_param_before, first_param_after)


# ruff: noqa: F401,F403,F405
from test.activation_based.test_distributed_dtensor import *
from test.activation_based.test_distributed_dtensor import (
    _ToyNonCallableReset,
    _ToyResetCounter,
    _load_train_distributed_module,
    _reset_net,
    _train_args,
    _train_runtime,
)


@pytest.mark.parametrize(
    ("mode", "mesh_shape", "overrides", "expected"),
    (
        (
            "dp",
            None,
            {},
            {
                "mesh_shape": (4,),
                "enable_data_parallel": True,
                "enable_fsdp2": False,
                "auto_tensor_parallel": False,
                "tensor_parallel_roots": None,
                "experimental_conv_tensor_parallel": False,
                "conv_tensor_parallel_roots": None,
                "fsdp_shard_roots": None,
                "fsdp_shard_module_root": True,
                "tp_mesh_dim": 0,
                "dp_mesh_dim": 0,
            },
        ),
        (
            "tp",
            None,
            {},
            {
                "mesh_shape": (4,),
                "enable_data_parallel": False,
                "enable_fsdp2": False,
                "auto_tensor_parallel": True,
                "tensor_parallel_roots": ["classifier"],
                "experimental_conv_tensor_parallel": True,
                "conv_tensor_parallel_roots": ["features"],
                "fsdp_shard_roots": None,
                "fsdp_shard_module_root": True,
                "tp_mesh_dim": 0,
                "dp_mesh_dim": None,
            },
        ),
        (
            "tp",
            [4],
            {"disable_classifier_tp": True},
            {
                "mesh_shape": (4,),
                "enable_data_parallel": False,
                "enable_fsdp2": False,
                "auto_tensor_parallel": False,
                "tensor_parallel_roots": None,
                "experimental_conv_tensor_parallel": True,
                "conv_tensor_parallel_roots": ["features"],
                "fsdp_shard_roots": None,
                "fsdp_shard_module_root": True,
                "tp_mesh_dim": 0,
                "dp_mesh_dim": None,
            },
        ),
        (
            "fsdp2",
            None,
            {},
            {
                "mesh_shape": (4,),
                "enable_data_parallel": False,
                "enable_fsdp2": True,
                "auto_tensor_parallel": False,
                "tensor_parallel_roots": None,
                "experimental_conv_tensor_parallel": False,
                "conv_tensor_parallel_roots": None,
                "fsdp_shard_roots": ["features", "classifier"],
                "fsdp_shard_module_root": True,
                "tp_mesh_dim": 0,
                "dp_mesh_dim": 0,
            },
        ),
        (
            "fsdp2_tp",
            [2, 2],
            {},
            {
                "mesh_shape": (2, 2),
                "enable_data_parallel": False,
                "enable_fsdp2": True,
                "auto_tensor_parallel": True,
                "tensor_parallel_roots": ["classifier"],
                "experimental_conv_tensor_parallel": True,
                "conv_tensor_parallel_roots": ["features"],
                "fsdp_shard_roots": ["features"],
                "fsdp_shard_module_root": False,
                "tp_mesh_dim": 1,
                "dp_mesh_dim": 0,
            },
        ),
        (
            "fsdp2_tp",
            [2, 2],
            {"disable_conv_tp": True, "tp_mesh_dim": 1, "dp_mesh_dim": 0},
            {
                "mesh_shape": (2, 2),
                "enable_data_parallel": False,
                "enable_fsdp2": True,
                "auto_tensor_parallel": True,
                "tensor_parallel_roots": ["classifier"],
                "experimental_conv_tensor_parallel": False,
                "conv_tensor_parallel_roots": None,
                "fsdp_shard_roots": ["features"],
                "fsdp_shard_module_root": False,
                "tp_mesh_dim": 1,
                "dp_mesh_dim": 0,
            },
        ),
    ),
)
def test_train_distributed_build_model_config_matrix(
    monkeypatch, mode, mesh_shape, overrides, expected
):
    train_distributed, runtime = _train_runtime(mode)
    captured = {}

    def _fake_configure(model, config):
        captured["config"] = config
        return model, None, None

    monkeypatch.setattr(train_distributed, "configure_snn_distributed", _fake_configure)
    args = _train_args(
        distributed_mode=mode,
        mesh_shape=mesh_shape,
        **overrides,
    )
    train_distributed.build_model(args, runtime)

    config = captured["config"]
    assert config.device_type == "cpu"
    for field, value in expected.items():
        assert getattr(config, field) == value

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
    with pytest.raises(
        ValueError, match="requires at least one tensor-parallel target"
    ):
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
    reduced_logits, reduced_labels = train_distributed.reduce_classification_output(
        logits, labels
    )
    torch.testing.assert_close(reduced_logits, logits)
    assert torch.equal(reduced_labels, labels)
    assert reduced_logits.shape == logits.shape
    assert reduced_labels.shape == labels.shape

def test_train_distributed_reduce_classification_output_reduces_time_major_logits():
    train_distributed = _load_train_distributed_module()
    logits = torch.randn(5, 4, 10)
    labels = torch.eye(10)[torch.tensor([0, 1, 2, 3])]
    reduced_logits, reduced_labels = train_distributed.reduce_classification_output(
        logits, labels
    )
    torch.testing.assert_close(reduced_logits, logits.mean(dim=0))
    assert torch.equal(reduced_labels, torch.tensor([0, 1, 2, 3]))
    assert reduced_logits.shape == (4, 10)
    assert reduced_labels.shape == (4,)

def test_train_distributed_prepare_classification_output_matches_reduce_for_local_tensor():
    train_distributed = _load_train_distributed_module()
    logits = torch.randn(5, 4, 10)
    labels = torch.eye(10)[torch.tensor([0, 1, 2, 3])]
    prepared_logits, prepared_labels = train_distributed.prepare_classification_output(
        logits, labels
    )
    torch.testing.assert_close(prepared_logits, logits.mean(dim=0))
    assert torch.equal(prepared_labels, torch.tensor([0, 1, 2, 3]))

def test_train_distributed_forward_loss_uses_normalized_singleton_labels():
    train_distributed = _load_train_distributed_module()
    model = nn.Linear(3, 4)
    criterion = nn.CrossEntropyLoss()
    images = torch.randn(2, 3)
    labels = torch.tensor([[1], [3]])
    logits, normalized_labels, loss = train_distributed.forward_loss(
        model,
        criterion,
        images,
        labels,
    )
    assert logits.shape == (2, 4)
    assert torch.equal(normalized_labels, torch.tensor([1, 3]))
    assert torch.is_tensor(loss)

def test_train_distributed_reduce_classification_output_normalizes_singleton_labels():
    train_distributed = _load_train_distributed_module()
    logits = torch.randn(2, 4)
    labels = torch.tensor([[1], [3]])
    reduced_logits, reduced_labels = train_distributed.reduce_classification_output(
        logits, labels
    )
    torch.testing.assert_close(reduced_logits, logits)
    assert torch.equal(reduced_labels, torch.tensor([1, 3]))

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

    train_loader, val_loader, train_sampler = train_distributed.build_data(
        args, runtime, mesh=None
    )
    assert train_sampler is not None
    assert isinstance(train_loader.sampler, type(train_sampler))
    assert isinstance(val_loader.sampler, type(train_sampler))
    assert train_sampler.num_replicas == 1
    assert train_sampler.rank == 0
    assert train_loader.sampler.num_replicas == train_sampler.num_replicas
    assert train_loader.sampler.rank == train_sampler.rank
    assert val_loader.sampler.num_replicas == train_sampler.num_replicas
    assert val_loader.sampler.rank == train_sampler.rank

def test_train_distributed_setup_runtime_normalizes_local_auto(monkeypatch):
    train_distributed = _load_train_distributed_module()
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    args = SimpleNamespace(distributed_mode="auto")
    runtime = train_distributed.setup_runtime(args)
    assert runtime.mode == "none"
    assert runtime.is_distributed is False
