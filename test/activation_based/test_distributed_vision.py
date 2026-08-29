import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import torch.distributed.device_mesh as device_mesh
import torch.nn as nn
from torch.utils.data import TensorDataset

from spikingjelly.activation_based import functional, layer
from spikingjelly.activation_based.distributed import vision
from spikingjelly.activation_based.distributed.vision import config as vision_config
from spikingjelly.activation_based.distributed.vision import execution, inference
from spikingjelly.activation_based.distributed.tensor_parallel import (
    ChannelShardBatchNorm2d,
)
from spikingjelly.activation_based.distributed.vision import training
from spikingjelly.activation_based.model.sew_resnet import (
    SEWResNet34Config,
    _pipeline_stage as sew_pipeline_stage,
)
from spikingjelly.activation_based.model.sew_resnet import BasicBlock
from spikingjelly.activation_based.model.spikformer import (
    SpikformerCIFAR10Config,
    SpikformerBuilder,
    SpikformerConfig,
    _pipeline_stage,
    SpikformerBlock,
    spikformer_cifar10,
    spikformer_s,
)
from spikingjelly.activation_based.precision import PrecisionConfig


def test_vision_training_config_json_round_trip():
    config = vision.TrainingConfig(
        model=SEWResNet34Config(
            time_steps=6,
            num_classes=11,
            step_mode="s",
            image_size=48,
        ),
        dataset_builder="package.datasets.build",
        dataset_kwargs={"root": Path("images")},
        input_layout="NTCHW",
        loss_function="package.losses.focal_loss",
        loss_kwargs={"gamma": 2.0},
        mixup_alpha=0.5,
        tensor_parallel_size=2,
        data_parallel="fsdp2",
        checkpoint_dir=Path("checkpoints"),
        checkpoint_interval=5,
    )

    restored = vision.TrainingConfig.from_dict(config.as_dict())

    assert restored == config
    assert restored.model.get_builder_cls().__name__ == "SEWResNet34Builder"

    cifar = vision.TrainingConfig(
        model=SpikformerCIFAR10Config(),
        dataset_builder="package.datasets.build",
    )
    assert vision.TrainingConfig.from_dict(cifar.as_dict()) == cifar


def test_vision_model_config_targets_load_in_a_fresh_process():
    script = (
        "import json,sys; "
        "from spikingjelly.activation_based.distributed.vision.config import ModelConfig; "
        "config=ModelConfig.from_dict(json.loads(sys.argv[1])); "
        "assert type(config).__module__ == "
        "'spikingjelly.activation_based.model.sew_resnet'"
    )
    values = {
        "time_steps": 2,
        "num_classes": 3,
        "step_mode": "m",
        "image_size": 32,
    }
    for target in (
        "spikingjelly.activation_based.model.sew_resnet.SEWResNet34Config",
        "spikingjelly.activation_based.distributed.vision.sew_resnet.SEWResNet34Config",
    ):
        completed = subprocess.run(
            [sys.executable, "-c", script, json.dumps({"_target_": target, **values})],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_vision_config_does_not_auto_import_external_targets(monkeypatch):
    imported = []
    monkeypatch.setattr(
        vision_config.importlib,
        "import_module",
        lambda name: imported.append(name),
    )

    with pytest.raises(ValueError, match="Unsupported config target"):
        vision_config.ModelConfig.from_dict(
            {"_target_": "external_package.model.CustomConfig"}
        )

    assert imported == []


def test_vision_evaluation_config_and_artifact_round_trip(tmp_path):
    config = vision.EvaluationConfig(
        artifact=tmp_path / "model.pt",
        dataset_builder="package.datasets.build",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        pipeline_microbatches=2,
        batch_size=4,
        data_parallel="fsdp2",
    )
    assert config.data_parallel == "fsdp2"

    model_config = SEWResNet34Config(time_steps=2, num_classes=3, image_size=32)
    model, _, _, _ = model_config.get_builder_cls()(model_config).build(
        process_group=None,
        memopt_process_group=None,
        pipeline_rank=0,
        pipeline_size=1,
        pipeline_microbatches=1,
        device=torch.device("cpu"),
        micro_batch_size=1,
        memopt_level=0,
        memopt_compress_inputs=False,
        memopt_checkpoint_budget="memory",
    )
    torch.save(
        {
            "schema_version": 1,
            "model_config": model_config.as_dict(),
            "state_dict": model.state_dict(),
            "source": {"checkpoint": "checkpoint"},
        },
        config.artifact,
    )

    restored_config, restored_state, source = vision.load_inference_artifact(
        config.artifact
    )

    assert restored_config == model_config
    assert restored_state.keys() == model.state_dict().keys()
    assert source == {"checkpoint": "checkpoint"}


def test_vision_prediction_writes_only_ordered_outputs(tmp_path):
    shard_paths = [tmp_path / "rank-0.h5", tmp_path / "rank-1.h5"]
    for path, indices, logits in (
        (shard_paths[0], [2, 0], [[2.0, 3.0], [0.0, 1.0]]),
        (shard_paths[1], [1], [[1.0, 2.0]]),
    ):
        handle = inference._open_prediction_shard(path, num_classes=2)
        inference._append_predictions(
            handle,
            torch.tensor(indices),
            torch.tensor(logits),
        )
        handle.close()

    output = tmp_path / "predictions.h5"
    inference._merge_prediction_shards(
        output,
        shard_paths,
        dataset_size=3,
        num_classes=2,
        attributes={},
    )

    with h5py.File(output, "r") as predictions:
        assert set(predictions) == {"index", "logits"}
        np.testing.assert_array_equal(predictions["index"][:], [0, 1, 2])
        np.testing.assert_array_equal(
            predictions["logits"][:],
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
        )


def test_vision_prediction_merge_cleans_failed_temporary_file(tmp_path):
    shard = tmp_path / "rank-0.h5"
    handle = inference._open_prediction_shard(shard, num_classes=2)
    inference._append_predictions(
        handle,
        torch.tensor([0, 0]),
        torch.tensor([[0.0, 1.0], [1.0, 2.0]]),
    )
    handle.close()
    output = tmp_path / "predictions.h5"

    with pytest.raises(ValueError, match="duplicate"):
        inference._merge_prediction_shards(
            output,
            [shard],
            dataset_size=1,
            num_classes=2,
            attributes={},
        )

    assert not output.with_name(".predictions.h5.tmp").exists()


def test_vision_predict_returns_no_metrics(monkeypatch, tmp_path):
    config = vision.PredictionConfig(
        artifact=tmp_path / "model.pt",
        dataset_builder="package.datasets.build",
    )
    monkeypatch.setattr(
        inference, "_run_classification", lambda *_args, **_kwargs: None
    )

    assert inference.predict_classification(config, tmp_path / "output.h5") is None


@pytest.mark.parametrize(
    ("match", "kwargs"),
    [
        ("batch_size", {"batch_size": 0}),
        ("dataset_builder", {"dataset_builder": "dataset"}),
        ("data_parallel", {"data_parallel": "ddp"}),
        ("compile", {"compile": True, "pipeline_parallel_size": 2}),
    ],
)
def test_vision_prediction_config_rejects_invalid_values(match, kwargs):
    arguments = {
        "artifact": Path("model.pt"),
        "dataset_builder": "package.datasets.build",
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        vision.PredictionConfig(**arguments)


def test_vision_evaluation_owns_loss_configuration():
    prediction = vision.PredictionConfig(
        artifact=Path("model.pt"),
        dataset_builder="package.datasets.build",
    )
    assert not hasattr(prediction, "loss_function")
    with pytest.raises(ValueError, match="loss_function"):
        vision.EvaluationConfig(
            artifact=Path("model.pt"),
            dataset_builder="package.datasets.build",
            loss_function="cross_entropy",
        )
    with pytest.raises(ValueError, match="timing_warmup_batches"):
        vision.EvaluationConfig(
            artifact=Path("model.pt"),
            dataset_builder="package.datasets.build",
            timing_warmup_batches=-1,
        )


@pytest.mark.parametrize(
    ("match", "kwargs"),
    [
        ("tensor_parallel_size", {"tensor_parallel_size": 0}),
        ("checkpoint_dir", {"checkpoint_interval": 1}),
        ("pipeline_microbatches", {"batch_size": 10, "pipeline_microbatches": 4}),
        ("timing_warmup_steps", {"max_steps": 10, "timing_warmup_steps": 10}),
        ("loss_function", {"loss_function": "cross_entropy"}),
        ("input_layout", {"input_layout": "NHWC"}),
        ("mixup_alpha", {"mixup_alpha": -0.1}),
        (
            "step_mode='m'",
            {
                "model": SEWResNet34Config(step_mode="s"),
                "pipeline_parallel_size": 2,
            },
        ),
        (
            "memopt",
            {"model": SEWResNet34Config(step_mode="s"), "memopt_level": 1},
        ),
    ],
)
def test_vision_training_config_rejects_invalid_values(match, kwargs):
    kwargs = dict(kwargs)
    model = kwargs.pop("model", SEWResNet34Config())

    with pytest.raises(ValueError, match=match):
        vision.TrainingConfig(
            model=model,
            dataset_builder="package.datasets.build",
            **kwargs,
        )


def test_vision_model_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="in_channels=3"):
        SEWResNet34Config(in_channels=1)
    with pytest.raises(ValueError, match="step_mode"):
        SEWResNet34Config(step_mode="invalid")
    with pytest.raises(ValueError, match="Spikformer requires step_mode='m'"):
        SpikformerConfig(step_mode="s")


def test_vision_artifact_tensor_sharding_round_trip():
    sew_builder = SEWResNet34Config().get_builder_cls()(SEWResNet34Config())
    reference = torch.arange(32).reshape(8, 4)
    targets = [torch.empty(4, 4), torch.empty(4, 4)]
    shards = [
        sew_builder._shard_tensor_parallel_tensor("weight", reference, target, rank, 2)
        for rank, target in enumerate(targets)
    ]
    assert torch.equal(
        sew_builder._merge_tensor_parallel_shards("weight", shards, reference),
        reference,
    )

    spikformer_builder = SpikformerBuilder(SpikformerConfig())
    qkv = torch.arange(24 * 4).reshape(24, 4)
    qkv_targets = [torch.empty(12, 4), torch.empty(12, 4)]
    qkv_shards = [
        spikformer_builder._shard_tensor_parallel_tensor(
            "blocks.0.attn.qkv_conv_bn.0.weight", qkv, target, rank, 2
        )
        for rank, target in enumerate(qkv_targets)
    ]
    assert torch.equal(
        spikformer_builder._merge_tensor_parallel_shards(
            "blocks.0.attn.qkv_conv_bn.0.weight", qkv_shards, qkv
        ),
        qkv,
    )


def test_sew_resnet_memopt_preserves_add_residual_results():
    config = SEWResNet34Config(time_steps=1, num_classes=3, image_size=16)
    build_kwargs = {
        "process_group": None,
        "memopt_process_group": None,
        "pipeline_rank": 0,
        "pipeline_size": 1,
        "pipeline_microbatches": 1,
        "device": torch.device("cpu"),
        "micro_batch_size": 2,
        "memopt_checkpoint_budget": "memory",
    }
    torch.manual_seed(7)
    baseline, *_ = config.get_builder_cls()(config).build(
        **build_kwargs, memopt_level=0, memopt_compress_inputs=False
    )
    torch.manual_seed(7)
    candidate, *_ = config.get_builder_cls()(config).build(
        **build_kwargs, memopt_level=1, memopt_compress_inputs=True
    )

    x0 = torch.randn(1, 2, 3, 16, 16, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    y0 = baseline(x0)
    y0.square().mean().backward()
    y1 = candidate(x1)
    y1.square().mean().backward()

    torch.testing.assert_close(y1, y0)
    torch.testing.assert_close(x1.grad, x0.grad)
    for parameter0, parameter1 in zip(
        baseline.parameters(), candidate.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter1.grad, parameter0.grad)


def test_vision_classification_loss_uses_custom_function_and_requires_scalar():
    logits = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    targets = torch.tensor([0, 1])
    config = vision.TrainingConfig(
        model=SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        loss_kwargs={"label_smoothing": 0.2},
    )
    loss_function = training._build_loss_function(config)

    assert torch.equal(
        execution._classification_loss(logits, targets, loss_function),
        nn.functional.cross_entropy(logits, targets, label_smoothing=0.2),
    )

    with pytest.raises(TypeError, match="torch.Tensor"):
        execution._classification_loss(logits, targets, lambda *_args: 0.0)
    with pytest.raises(ValueError, match="scalar"):
        execution._classification_loss(logits, targets, lambda output, _labels: output)


def test_vision_classification_forward_respects_step_mode():
    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def forward(self, x):
            self.shapes.append(tuple(x.shape))
            return x.mean(dim=(-2, -1))

    images = torch.randn(2, 4, 3, 3)
    single_step = Recorder()
    multi_step = Recorder()

    single_logits = execution._forward_classification(
        single_step, images, 3, "s", "NCHW"
    )
    multi_logits = execution._forward_classification(multi_step, images, 3, "m", "NCHW")

    expected = images.mean(dim=(-2, -1))
    torch.testing.assert_close(single_logits, expected)
    torch.testing.assert_close(multi_logits, expected)
    assert single_step.shapes == [(2, 4, 3, 3)] * 3
    assert multi_step.shapes == [(3, 2, 4, 3, 3)]


def test_vision_classification_sequence_uses_declared_layout():
    temporal = torch.randn(2, 3, 4, 5, 5)

    time_first = execution._classification_sequence(temporal, 3, "NTCHW")
    batch_first = execution._classification_sequence(
        temporal, 3, "NTCHW", batch_first=True
    )

    torch.testing.assert_close(time_first, temporal.transpose(0, 1))
    torch.testing.assert_close(batch_first, temporal)
    with pytest.raises(ValueError, match="model.time_steps"):
        execution._classification_sequence(temporal, 4, "NTCHW")
    with pytest.raises(ValueError, match="NCHW"):
        execution._classification_sequence(temporal, 3, "NCHW")


def test_pipeline_expands_static_input_per_microbatch():
    class Recorder(nn.Module):
        def forward(self, value):
            self.shape = tuple(value.shape)
            return value.mean(dim=(-2, -1))

    recorder = Recorder()
    images = torch.randn(2, 3, 5, 5)

    pipeline = inference._ForwardPipeline(
        recorder,
        process_group=None,
        pipeline_rank=0,
        pipeline_size=1,
        microbatches=1,
        input_shape=(2, 3, 5, 5),
        communication_dtype=torch.float32,
        device=torch.device("cpu"),
        time_steps=4,
    )
    output = pipeline.step(images)

    assert recorder.shape == (4, 2, 3, 5, 5)
    assert output.shape == (2, 3)


def test_forward_pipeline_merges_semantic_microbatches():
    class Classifier(nn.Module):
        def forward(self, value):
            return value.mean(dim=(-2, -1))

    pipeline = inference._ForwardPipeline(
        Classifier(),
        process_group=None,
        pipeline_rank=0,
        pipeline_size=1,
        microbatches=2,
        input_shape=(2, 3, 5, 5),
        communication_dtype=torch.float32,
        device=torch.device("cpu"),
        time_steps=4,
    )
    images = torch.randn(4, 3, 5, 5)

    output = pipeline.step(images)

    torch.testing.assert_close(output, images.mean(dim=(-2, -1)))


def test_forward_pipeline_sends_declared_dtype(monkeypatch):
    sent = []
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda _group, rank: rank)
    monkeypatch.setattr(
        torch.distributed, "send", lambda value, **_kwargs: sent.append(value)
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda **_kwargs: None)
    pipeline = inference._ForwardPipeline(
        nn.Identity(),
        process_group=None,
        pipeline_rank=0,
        pipeline_size=2,
        microbatches=1,
        input_shape=(2, 3),
        communication_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    pipeline.step(torch.ones(2, 3))

    assert sent[0].dtype == torch.bfloat16


def test_vision_inference_preserves_early_configuration_error(monkeypatch):
    config = vision.EvaluationConfig(
        artifact=Path("artifact.pt"),
        dataset_builder="package.datasets.build",
        pipeline_parallel_size=2,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 3)

    with pytest.raises(ValueError, match="world_size"):
        inference._run_classification(config, mode="evaluate")


def test_vision_inference_rejects_single_step_pipeline_artifact(monkeypatch):
    config = vision.EvaluationConfig(
        artifact=Path("artifact.pt"),
        dataset_builder="package.datasets.build",
        pipeline_parallel_size=2,
    )

    class Mesh:
        def __getitem__(self, _name):
            return self

        def get_group(self):
            return object()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        device_mesh, "init_device_mesh", lambda *_args, **_kwargs: Mesh()
    )
    monkeypatch.setattr(
        inference,
        "load_inference_artifact",
        lambda _path: (SimpleNamespace(step_mode="s"), {}, {}),
    )

    with pytest.raises(ValueError, match="step_mode='m'"):
        inference._run_classification(config, mode="evaluate")


def test_vision_rank_zero_error_is_broadcast(monkeypatch):
    broadcasts = []
    monkeypatch.setattr(
        inference.dist,
        "broadcast",
        lambda tensor, **_kwargs: broadcasts.append(tensor.item()),
    )

    with pytest.raises(OSError, match="merge failed"):
        inference._sync_rank_zero_error(
            OSError("merge failed"), torch.device("cpu"), "remote failure"
        )

    assert broadcasts == [1]

    monkeypatch.setattr(
        inference.dist, "broadcast", lambda tensor, **_kwargs: tensor.fill_(1)
    )
    with pytest.raises(RuntimeError, match="remote failure"):
        inference._sync_rank_zero_error(None, torch.device("cpu"), "remote failure")


def test_vision_inference_rejects_wrong_config_before_runtime():
    prediction = vision.PredictionConfig(
        artifact=Path("artifact.pt"), dataset_builder="package.datasets.build"
    )
    evaluation = vision.EvaluationConfig(
        artifact=Path("artifact.pt"), dataset_builder="package.datasets.build"
    )

    with pytest.raises(TypeError, match="EvaluationConfig"):
        vision.evaluate_classification(prediction)
    with pytest.raises(TypeError, match="PredictionConfig"):
        vision.predict_classification(evaluation, Path("predictions.h5"))


def test_vision_broadcasts_data_parallel_buffers(monkeypatch):
    model = nn.BatchNorm2d(3)
    process_group = object()
    calls = []
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda group, rank: 7)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, **kwargs: calls.append((id(tensor), kwargs)),
    )

    training._broadcast_data_parallel_buffers(model, process_group)

    assert [call[0] for call in calls] == [id(buffer) for buffer in model.buffers()]
    assert all(call[1] == {"src": 7, "group": process_group} for call in calls)


def test_set_step_mode_preserves_seq_to_ann_children():
    container = layer.SeqToANNContainer(
        layer.Conv2d(2, 3, kernel_size=1, step_mode="s")
    )

    functional.set_step_mode(container, "m")

    assert container[0].step_mode == "s"


def test_vision_training_config_rejects_unknown_serialized_fields():
    data = vision.TrainingConfig(
        model=SEWResNet34Config(),
        dataset_builder="package.datasets.build",
    ).as_dict()
    data["unknown"] = True

    with pytest.raises(TypeError, match="unknown"):
        vision.TrainingConfig.from_dict(data)


def test_vision_training_config_rejects_non_config_target():
    with pytest.raises(ValueError, match="Unsupported config target"):
        vision.TrainingConfig.from_dict(
            {"_target_": "pathlib.Path", "pathsegments": ["unexpected"]}
        )


def test_vision_training_rejects_empty_datasets(monkeypatch):
    empty = TensorDataset(torch.empty(0), torch.empty(0, dtype=torch.long))
    monkeypatch.setattr(
        training, "_import_object", lambda _path: lambda: (empty, empty)
    )
    config = vision.TrainingConfig(
        model=SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        workers=0,
    )

    with pytest.raises(ValueError, match="non-empty"):
        training._build_loaders(config, dp_size=1, dp_rank=0)


def test_vision_pipeline_drops_ragged_batches(monkeypatch):
    train_dataset = TensorDataset(
        torch.zeros(3, 3, 4, 4), torch.zeros(3, dtype=torch.long)
    )
    validation_dataset = TensorDataset(
        torch.zeros(4, 3, 4, 4), torch.zeros(4, dtype=torch.long)
    )
    monkeypatch.setattr(
        training,
        "_import_object",
        lambda _path: lambda: (train_dataset, validation_dataset),
    )
    config = vision.TrainingConfig(
        model=SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        batch_size=2,
        workers=0,
        pipeline_parallel_size=2,
    )

    train_loader, validation_loader, _, _ = training._build_loaders(
        config, dp_size=1, dp_rank=0
    )

    assert len(train_loader) == 1
    assert len(validation_loader) == 2


def test_vision_pipeline_rejects_ragged_validation_dataset(monkeypatch):
    dataset = TensorDataset(torch.zeros(3, 3, 4, 4), torch.zeros(3, dtype=torch.long))
    monkeypatch.setattr(
        training, "_import_object", lambda _path: lambda: (dataset, dataset)
    )
    config = vision.TrainingConfig(
        model=SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        batch_size=2,
        workers=0,
        pipeline_parallel_size=2,
    )

    with pytest.raises(ValueError, match="validation dataset size"):
        training._build_loaders(config, dp_size=1, dp_rank=0)


def test_vision_data_parallel_rejects_padded_validation_dataset(monkeypatch):
    dataset = TensorDataset(torch.zeros(3, 3, 4, 4), torch.zeros(3, dtype=torch.long))
    monkeypatch.setattr(
        training, "_import_object", lambda _path: lambda: (dataset, dataset)
    )
    config = vision.TrainingConfig(
        model=SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        workers=0,
    )

    with pytest.raises(ValueError, match="validation dataset size"):
        training._build_loaders(config, dp_size=2, dp_rank=0)


def test_spikformer_pipeline_rejects_ragged_patch_grid():
    config = SpikformerConfig(image_height=33, image_width=32)
    builder = config.get_builder_cls()(config)

    with pytest.raises(ValueError, match="divisible by 16"):
        builder.build(
            process_group=None,
            memopt_process_group=None,
            pipeline_rank=0,
            pipeline_size=2,
            pipeline_microbatches=1,
            device=torch.device("cpu"),
            micro_batch_size=2,
            memopt_level=0,
            memopt_compress_inputs=False,
            memopt_checkpoint_budget="memory",
        )


def test_sew_resnet34_single_step_matches_multi_step():
    config = SEWResNet34Config(
        time_steps=2,
        num_classes=5,
        step_mode="m",
        image_size=32,
    )
    model, _, _, _ = config.get_builder_cls()(config).build(
        process_group=None,
        memopt_process_group=None,
        pipeline_rank=0,
        pipeline_size=1,
        pipeline_microbatches=1,
        device=torch.device("cpu"),
        micro_batch_size=2,
        memopt_level=0,
        memopt_compress_inputs=False,
        memopt_checkpoint_budget="memory",
    )
    model.eval()
    images = torch.randn(2, 3, 32, 32)
    sequence = images.unsqueeze(0).expand(2, *images.shape).contiguous()

    functional.set_step_mode(model, "m")
    functional.reset_net(model)
    multi_step = model(sequence)
    functional.set_step_mode(model, "s")
    functional.reset_net(model)
    single_step = torch.stack([model(x) for x in sequence])

    torch.testing.assert_close(single_step, multi_step)


@pytest.mark.parametrize("legacy_precision", [False, True])
def test_vision_checkpoint_restores_rng(tmp_path, monkeypatch, legacy_precision):
    from torch.distributed.checkpoint import state_dict as dcp_state_dict

    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    config = vision.TrainingConfig(
        model=SEWResNet34Config(),
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
    if legacy_precision:
        recipe_path = checkpoint / "config.json"
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        recipe["precision"] = "bf16"
        recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
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
        model=SEWResNet34Config(),
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

    stages = [_pipeline_stage(model, rank, 4) for rank in range(4)]
    block_counts = [
        sum(isinstance(module, SpikformerBlock) for module in stage.modules())
        for stage in stages
    ]

    assert block_counts == [0, 2, 2, 2]


def test_four_block_spikformer_rejects_pipeline_size_four():
    with pytest.raises(ValueError, match="4-block Spikformer"):
        _pipeline_stage(spikformer_cifar10(), 0, 4)


def test_sew_pipeline_downsamples_before_stage_boundaries():
    config = SEWResNet34Config(time_steps=2, image_size=224)
    builder = config.get_builder_cls()(config)
    model = builder._build_canonical_model()
    stages = [sew_pipeline_stage(model, rank, 4) for rank in range(4)]

    assert (
        sum(
            isinstance(module, BasicBlock)
            for stage in stages
            for module in stage.modules()
        )
        == 16
    )

    expected_shapes = (
        (2, 2, 128, 28, 28),
        (2, 2, 256, 14, 14),
        (2, 2, 512, 7, 7),
        (2, 2, 1000),
    )
    for rank, expected_output_shape in enumerate(expected_shapes):
        _, _, input_shape, output_shape = builder.build(
            process_group=None,
            memopt_process_group=None,
            pipeline_rank=rank,
            pipeline_size=4,
            pipeline_microbatches=2,
            device=torch.device("cpu"),
            micro_batch_size=4,
            memopt_level=0,
            memopt_compress_inputs=False,
            memopt_checkpoint_budget="memory",
        )
        assert output_shape == expected_output_shape
        if rank:
            assert input_shape == expected_shapes[rank - 1]


def test_spikformer_cifar10_pipeline_memopt_uses_8_by_8_tokens():
    config = SpikformerCIFAR10Config(time_steps=2)
    builder = config.get_builder_cls()(config)

    assert config.num_classes == 10

    _, _, input_shape, output_shape = builder.build(
        process_group=None,
        memopt_process_group=None,
        pipeline_rank=0,
        pipeline_size=2,
        pipeline_microbatches=2,
        device=torch.device("cpu"),
        micro_batch_size=4,
        memopt_level=1,
        memopt_compress_inputs=False,
        memopt_checkpoint_budget="memory",
    )

    assert input_shape == (2, 2, 3, 32, 32)
    assert output_shape == (2, 2, 384, 8, 8)


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
        model=SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        data_parallel="fsdp2",
        precision=PrecisionConfig(mode="bf16"),
    )

    execution._wrap_data_parallel(
        model,
        data_parallel=config.data_parallel,
        pipeline_parallel_size=config.pipeline_parallel_size,
        step_mode=config.model.step_mode,
        precision=config.precision,
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


def test_training_config_round_trips_precision_config():
    config = vision.TrainingConfig(
        model=SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        precision=PrecisionConfig(
            mode="fp8",
            fp8_recipe="delayed",
            triton_storage="float8_e4m3fn",
            triton_fwd="bf16",
            triton_bwd="fp16",
        ),
    )
    assert vision.TrainingConfig.from_dict(config.as_dict()) == config


def test_training_config_rejects_experimental_precision_outside_ddp():
    with pytest.raises(ValueError, match="requires DDP"):
        vision.TrainingConfig(
            model=SEWResNet34Config(),
            dataset_builder="package.datasets.build",
            data_parallel="fsdp2",
            precision=PrecisionConfig(mode="fp8"),
        )
