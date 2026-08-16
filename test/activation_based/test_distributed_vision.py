from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from spikingjelly.activation_based import layer
from spikingjelly.activation_based.distributed import vision
from spikingjelly.activation_based.distributed.tensor_parallel import (
    ChannelShardBatchNorm2d,
)
from spikingjelly.activation_based.distributed.vision import training
from spikingjelly.activation_based.distributed.vision.spikformer import (
    _pipeline_stage,
)
from spikingjelly.activation_based.model.spikformer import SpikformerBlock, spikformer_s


def test_vision_training_config_json_round_trip():
    config = vision.TrainingConfig(
        model=vision.SpikformerConfig(
            time_steps=6,
            num_classes=11,
            image_height=48,
            image_width=64,
        ),
        dataset_builder="package.datasets.build",
        dataset_kwargs={"root": Path("images")},
        tensor_parallel_size=2,
        data_parallel="fsdp2",
        checkpoint_dir=Path("checkpoints"),
        checkpoint_interval=5,
    )

    restored = vision.TrainingConfig.from_dict(config.as_dict())

    assert restored == config
    assert restored.model.get_builder_cls().__name__ == "SpikformerBuilder"


def test_vision_training_config_rejects_invalid_parallel_values():
    model = vision.SEWResNet34Config()

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        vision.TrainingConfig(
            model=model,
            dataset_builder="package.datasets.build",
            tensor_parallel_size=0,
        )
    with pytest.raises(ValueError, match="checkpoint_dir"):
        vision.TrainingConfig(
            model=model,
            dataset_builder="package.datasets.build",
            checkpoint_interval=1,
        )
    with pytest.raises(ValueError, match="pipeline_microbatches"):
        vision.TrainingConfig(
            model=model,
            dataset_builder="package.datasets.build",
            batch_size=10,
            pipeline_microbatches=4,
        )
    with pytest.raises(ValueError, match="timing_warmup_steps"):
        vision.TrainingConfig(
            model=model,
            dataset_builder="package.datasets.build",
            max_steps=10,
            timing_warmup_steps=10,
        )
    with pytest.raises(ValueError, match="in_channels=3"):
        vision.SEWResNet34Config(in_channels=1)


def test_vision_training_config_rejects_unknown_serialized_fields():
    data = vision.TrainingConfig(
        model=vision.SEWResNet34Config(),
        dataset_builder="package.datasets.build",
    ).as_dict()
    data["unknown"] = True

    with pytest.raises(TypeError, match="unknown"):
        vision.TrainingConfig.from_dict(data)


def test_vision_training_rejects_empty_datasets(monkeypatch):
    empty = TensorDataset(torch.empty(0), torch.empty(0, dtype=torch.long))
    monkeypatch.setattr(
        training, "_import_object", lambda _path: lambda: (empty, empty)
    )
    config = vision.TrainingConfig(
        model=vision.SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        workers=0,
    )

    with pytest.raises(ValueError, match="non-empty"):
        training._build_loaders(config, dp_size=1, dp_rank=0)


def test_vision_pipeline_drops_ragged_batches(monkeypatch):
    dataset = TensorDataset(torch.zeros(3, 3, 4, 4), torch.zeros(3, dtype=torch.long))
    monkeypatch.setattr(
        training, "_import_object", lambda _path: lambda: (dataset, dataset)
    )
    config = vision.TrainingConfig(
        model=vision.SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        batch_size=2,
        workers=0,
        pipeline_parallel_size=2,
    )

    train_loader, validation_loader, _, _ = training._build_loaders(
        config, dp_size=1, dp_rank=0
    )

    assert len(train_loader) == len(validation_loader) == 1


def test_spikformer_pipeline_rejects_ragged_patch_grid():
    config = vision.SpikformerConfig(image_height=33, image_width=32)
    builder = config.get_builder_cls()(config)

    with pytest.raises(ValueError, match="divisible by 16"):
        builder.build(
            process_group=None,
            pipeline_rank=0,
            pipeline_size=2,
            pipeline_microbatches=1,
            device=torch.device("cpu"),
            micro_batch_size=2,
            memopt_level=0,
            memopt_compress_inputs=False,
        )


def test_vision_checkpoint_restores_rng(tmp_path, monkeypatch):
    from torch.distributed.checkpoint import state_dict as dcp_state_dict

    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    config = vision.TrainingConfig(
        model=vision.SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        workers=0,
    )
    cpu_rng = torch.tensor([3], dtype=torch.uint8)
    cuda_rng = torch.tensor([7], dtype=torch.uint8)
    restored = {}
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch, "get_rng_state", lambda: cpu_rng)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda_rng)
    monkeypatch.setattr(
        torch, "set_rng_state", lambda state: restored.setdefault("torch", state)
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state: restored.setdefault("cuda", state),
    )
    monkeypatch.setattr(
        dcp_state_dict,
        "set_state_dict",
        lambda *_args, **_kwargs: None,
    )

    checkpoint = tmp_path / "checkpoint"
    training._save_checkpoint(
        checkpoint,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        step=2,
        epoch=1,
        batch_in_epoch=3,
        tp_rank=0,
        pp_rank=0,
        dp_rank=0,
    )
    progress = training._load_checkpoint(
        checkpoint,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        tp_rank=0,
        pp_rank=0,
    )

    assert progress == (2, 1, 3)
    assert torch.equal(restored["torch"], cpu_rng)
    assert torch.equal(restored["cuda"], cuda_rng)


def test_vision_checkpoint_broadcasts_rank_zero_creation_failure(tmp_path, monkeypatch):
    parent = tmp_path / "not-a-directory"
    parent.write_text("occupied", encoding="utf-8")
    broadcasts = []
    barrier_called = False
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, **_kwargs: broadcasts.append(tensor.item()),
    )

    def barrier():
        nonlocal barrier_called
        barrier_called = True

    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = vision.TrainingConfig(
        model=vision.SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        workers=0,
    )

    with pytest.raises(OSError):
        training._save_checkpoint(
            parent / "checkpoint",
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            step=1,
            epoch=0,
            batch_in_epoch=1,
            tp_rank=0,
            pp_rank=0,
            dp_rank=0,
        )

    assert broadcasts == [0, 1]
    assert not barrier_called


def test_spikformer_pipeline_keeps_every_transformer_block():
    model = spikformer_s(img_size_h=32, img_size_w=32)

    stages = [_pipeline_stage(model, rank, 2) for rank in range(2)]

    assert sum(
        isinstance(module, SpikformerBlock)
        for stage in stages
        for module in stage.modules()
    ) == len(model.blocks)


def test_fsdp2_keeps_batch_norm_in_full_precision(monkeypatch):
    import torch.distributed.fsdp as fsdp

    calls = []
    monkeypatch.setattr(
        fsdp,
        "fully_shard",
        lambda module, **kwargs: calls.append((module, kwargs)),
    )
    model = nn.Sequential(
        nn.BatchNorm2d(3),
        ChannelShardBatchNorm2d(layer.BatchNorm2d(4), None),
        nn.Conv2d(4, 5, 1),
    )
    config = vision.TrainingConfig(
        model=vision.SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        data_parallel="fsdp2",
        precision="bf16",
    )

    training._wrap_data_parallel(
        model,
        config=config,
        device=torch.device("cuda", 0),
        dp_size=2,
        dp_group=None,
        dp_mesh=object(),
        fsdp_roots=(),
    )

    assert [call[0] for call in calls[:2]] == [model[0], model[1]]
    batch_norm_policy = calls[1][1]["mp_policy"]
    assert calls[0][0] is model[0]
    assert batch_norm_policy.param_dtype is None
    assert batch_norm_policy.output_dtype is torch.bfloat16
    assert calls[-1][0] is model
    assert calls[-1][1]["mp_policy"].param_dtype is torch.bfloat16
