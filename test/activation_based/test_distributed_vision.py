import json
import os
import subprocess
import sys
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import torch.distributed.device_mesh as device_mesh
import torch.nn as nn
from torch.utils.data import TensorDataset

from spikingjelly.activation_based import base as activation_base
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based._cuda_graph import validate_cuda_graph_model
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


class RegisteredTestNeuron(neuron.IFNode):
    pass


class RegisteredTestSurrogate(surrogate.Rect):
    def __init__(self, alpha=1.0, options=None):
        super().__init__(alpha=alpha)
        self.options = options


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


def test_neuron_config_round_trips_and_builds_registered_classes(monkeypatch):
    monkeypatch.setattr(vision_config, "_NEURON_REGISTRY", {})
    monkeypatch.setattr(vision_config, "_SURROGATE_REGISTRY", {})
    neuron_config = vision.NeuronConfig(
        class_path=f"{RegisteredTestNeuron.__module__}.{RegisteredTestNeuron.__qualname__}",
        kwargs={"v_threshold": 0.7},
        surrogate=f"{RegisteredTestSurrogate.__module__}.{RegisteredTestSurrogate.__qualname__}",
        surrogate_kwargs={"alpha": 1.5, "options": {"values": [1, {"_path_": "data"}]}},
    )
    config = vision.TrainingConfig(
        model=SpikformerCIFAR10Config(neuron_config=neuron_config),
        dataset_builder="package.datasets.build",
    )
    serialized = json.loads(json.dumps(config.as_dict(), allow_nan=False))
    restored = vision.TrainingConfig.from_dict(serialized)
    assert restored == config
    assert training._recipe(restored) == training._recipe(config)

    imported = []
    original_import_module = vision_config.importlib.import_module

    def record_import(module_name):
        imported.append(module_name)
        return original_import_module(module_name)

    monkeypatch.setattr(vision_config.importlib, "import_module", record_import)
    with pytest.raises(ValueError, match="Unregistered neuron class"):
        config.model.get_builder_cls()(config.model)._build_canonical_model()
    assert RegisteredTestNeuron.__module__ not in imported

    vision.register_neuron_class(RegisteredTestNeuron)
    imported.clear()
    with pytest.raises(ValueError, match="Unregistered surrogate class"):
        config.model.get_builder_cls()(config.model)._build_canonical_model()
    assert RegisteredTestSurrogate.__module__ not in imported
    vision.register_surrogate_class(RegisteredTestSurrogate)
    model = config.model.get_builder_cls()(config.model)._build_canonical_model()
    nodes = [
        module for module in model.modules() if isinstance(module, neuron.BaseNode)
    ]
    assert nodes
    assert all(type(node) is RegisteredTestNeuron for node in nodes)
    assert all(node.v_threshold == 0.7 for node in nodes)
    assert all(
        type(node.surrogate_function) is RegisteredTestSurrogate for node in nodes
    )
    assert all(node.surrogate_function.alpha == 1.5 for node in nodes)
    assert all(
        node.surrogate_function.options == neuron_config.surrogate_kwargs["options"]
        for node in nodes
    )


@pytest.mark.parametrize(
    ("register", "cls"),
    [
        (vision.register_neuron_class, RegisteredTestSurrogate),
        (vision.register_surrogate_class, RegisteredTestNeuron),
        (vision.register_neuron_class, None),
    ],
)
def test_neuron_registration_rejects_invalid_classes(register, cls):
    with pytest.raises(TypeError, match="cls must inherit"):
        register(cls)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"value": (1, 2)}, TypeError, "JSON-native"),
        ({"value": torch.tensor(1)}, TypeError, "JSON-native"),
        ({"value": float("inf")}, ValueError, "finite"),
        ({1: "value"}, TypeError, "keys must be strings"),
    ],
)
def test_neuron_config_rejects_non_json_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        vision.NeuronConfig(
            class_path="spikingjelly.activation_based.neuron.IFNode",
            kwargs=kwargs,
        )


def test_neuron_config_revalidates_mutated_kwargs_at_use_boundaries():
    config = SpikformerCIFAR10Config(
        neuron_config=vision.NeuronConfig(
            class_path="spikingjelly.activation_based.neuron.integrate_and_fire.IFNode"
        )
    )
    config.neuron_config.kwargs["value"] = object()

    with pytest.raises(TypeError, match="JSON-native"):
        config.as_dict()
    with pytest.raises(TypeError, match="JSON-native"):
        config.get_builder_cls()(config)._build_canonical_model()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two CUDA devices are required",
)
def test_nondefault_neuron_survives_distributed_resume_and_artifact_round_trip(
    tmp_path,
):
    script = tmp_path / "neuron_round_trip.py"
    script.write_text(
        """
from dataclasses import replace
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.distributed import vision
from spikingjelly.activation_based.distributed.vision.config import TrainingConfig
from spikingjelly.activation_based.distributed.vision.inference import (
    _load_checkpoint_model,
    export_inference_artifact,
    load_inference_artifact,
)
from spikingjelly.activation_based.model.spikformer import SpikformerCIFAR10Config
from spikingjelly.activation_based.precision import PrecisionConfig

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def build_tiny_datasets(count):
    images = torch.arange(count * 3 * 32 * 32, dtype=torch.float32).reshape(
        count, 3, 32, 32
    ) / (count * 3 * 32 * 32)
    targets = torch.arange(count, dtype=torch.long) % 2
    return TensorDataset(images, targets), TensorDataset(images[:2], targets[:2])


phase = sys.argv[1]
checkpoint_root = Path(sys.argv[2])
artifact_path = Path(sys.argv[3])
neuron_config = vision.NeuronConfig(
    class_path="spikingjelly.activation_based.neuron.integrate_and_fire.IFNode",
    kwargs={"v_threshold": 0.7},
    surrogate="spikingjelly.activation_based.surrogate.Rect",
    surrogate_kwargs={"alpha": 1.5},
)
config = TrainingConfig(
    model=SpikformerCIFAR10Config(
        time_steps=1,
        num_classes=2,
        neuron_config=neuron_config,
    ),
    dataset_builder="__main__.build_tiny_datasets",
    dataset_kwargs={"count": 4},
    input_layout="NCHW",
    epochs=1,
    batch_size=1,
    workers=0,
    tensor_parallel_size=1,
    data_parallel="ddp",
    precision=PrecisionConfig(mode="fp32"),
    max_steps=1,
    timing_warmup_steps=0,
    checkpoint_dir=checkpoint_root,
    checkpoint_interval=1,
)
if phase == "train":
    vision.train_classification(config)
    sys.exit(0)

if phase == "resume":
    checkpoint = checkpoint_root / "step_00000001"
    resumed = TrainingConfig.from_dict(config.as_dict())
    resumed = replace(resumed, max_steps=2, resume=checkpoint)
    vision.train_classification(resumed)
    sys.exit(0)

checkpoint = checkpoint_root / "step_00000002"
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
images = torch.linspace(0.0, 1.0, 3 * 32 * 32, device=device).reshape(1, 3, 32, 32)
reference_path = checkpoint_root / "reference_logits.pt"
if phase == "export":
    model = config.model.get_builder_cls()(config.model)._build_canonical_model()
    model.to(device).eval()
    _load_checkpoint_model(checkpoint, model, pipeline_rank=0, tensor_rank=0)
    functional.reset_net(model)
    with torch.no_grad():
        torch.save(model(images).cpu(), reference_path)
    export_inference_artifact(checkpoint, artifact_path)
    sys.exit(0)

model_config, state_dict, _ = load_inference_artifact(artifact_path)
assert model_config == config.model
builder = model_config.get_builder_cls()(model_config)
model, _, _, _ = builder.build_for_inference(
    state_dict,
    process_group=None,
    pipeline_rank=0,
    pipeline_size=1,
    pipeline_microbatches=1,
    device=device,
    micro_batch_size=1,
)
model.eval()
functional.reset_net(model)
with torch.no_grad():
    logits = model(images)
nodes = [module for module in model.modules() if isinstance(module, neuron.BaseNode)]
assert nodes and all(type(node) is neuron.IFNode for node in nodes)
assert all(node.v_threshold == 0.7 for node in nodes)
assert all(isinstance(node.surrogate_function, surrogate.Rect) for node in nodes)
assert all(node.surrogate_function.alpha == 1.5 for node in nodes)
assert logits.shape == (1, 1, 2) and torch.isfinite(logits).all()
torch.testing.assert_close(
    logits.cpu(), torch.load(reference_path, weights_only=True), rtol=0, atol=0
)
if int(os.environ["RANK"]) == 0:
    print("NEURON_ROUND_TRIP_OK", logits.cpu().tolist())
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    checkpoint_dir = tmp_path / "checkpoints"
    artifact = tmp_path / "model.pt"
    for process_count, phase in (
        (2, "train"),
        (2, "resume"),
        (1, "export"),
        (1, "infer"),
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={process_count}",
                str(script),
                phase,
                str(checkpoint_dir),
                str(artifact),
            ],
            capture_output=True,
            text=True,
            env=environment,
            timeout=900,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        if phase == "infer":
            assert "NEURON_ROUND_TRIP_OK" in result.stdout
    assert artifact.is_file()


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


def test_neuron_config_kwargs_do_not_resolve_nested_targets(monkeypatch):
    imported = []
    monkeypatch.setattr(
        vision_config.importlib,
        "import_module",
        lambda name: imported.append(name),
    )
    data = SpikformerCIFAR10Config(
        neuron_config=vision.NeuronConfig(
            class_path="spikingjelly.activation_based.neuron.IFNode",
            kwargs={"options": {"_target_": "external_package.payload.Code"}},
        )
    ).as_dict()

    restored = vision.ModelConfig.from_dict(data)

    assert restored.neuron_config.kwargs == {
        "options": {"_target_": "external_package.payload.Code"}
    }
    assert imported == []


@pytest.mark.parametrize("field", ["class_path", "surrogate"])
def test_neuron_config_rejects_structured_class_paths_without_importing(
    field, monkeypatch
):
    data = SpikformerCIFAR10Config(
        neuron_config=vision.NeuronConfig(
            class_path="spikingjelly.activation_based.neuron.integrate_and_fire.IFNode"
        )
    ).as_dict()
    data["neuron_config"][field] = {
        "_target_": "spikingjelly.activation_based.model.external.Payload"
    }
    imported = []
    monkeypatch.setattr(vision_config.importlib, "import_module", imported.append)

    with pytest.raises(TypeError, match=f"{field} must be a string"):
        vision.ModelConfig.from_dict(data)
    assert imported == []


@pytest.mark.parametrize(
    "model_config",
    [
        SEWResNet34Config(time_steps=1, num_classes=2, image_size=32),
        SpikformerConfig(time_steps=1, num_classes=2, image_height=32, image_width=32),
        SpikformerCIFAR10Config(time_steps=1, num_classes=2),
    ],
)
def test_builtin_neuron_config_builders_restore_model_weights(model_config):
    neuron_config = vision.NeuronConfig(
        class_path=f"{neuron.IFNode.__module__}.{neuron.IFNode.__qualname__}",
        kwargs={"v_threshold": 0.7},
        surrogate=f"{surrogate.Rect.__module__}.{surrogate.Rect.__qualname__}",
        surrogate_kwargs={"alpha": 1.5},
    )
    config = replace(model_config, neuron_config=neuron_config)
    restored = vision.ModelConfig.from_dict(
        json.loads(json.dumps(config.as_dict(), allow_nan=False))
    )

    torch.manual_seed(7)
    source = restored.get_builder_cls()(restored)._build_canonical_model()
    target = restored.get_builder_cls()(restored)._build_canonical_model()
    target.load_state_dict(source.state_dict(), strict=True)
    target_state = target.state_dict()
    assert all(
        torch.equal(source_value, target_state[name])
        for name, source_value in source.state_dict().items()
    )
    nodes = [
        module for module in target.modules() if isinstance(module, neuron.BaseNode)
    ]
    assert nodes and all(type(node) is neuron.IFNode for node in nodes)
    assert all(node.v_threshold == 0.7 for node in nodes)
    assert all(node.surrogate_function.alpha == 1.5 for node in nodes)


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
            "schema_version": inference._ARTIFACT_SCHEMA_VERSION,
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


def test_vision_artifact_rejects_v1_schema(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_config": {},
            "state_dict": {"weight": torch.ones(1)},
            "source": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="expected 2.*Re-export"):
        inference.load_inference_artifact(path)


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


def test_vision_valid_indices_filter_padding_and_missing_targets():
    valid = torch.tensor([True, False, True, True])
    has_target = torch.tensor([True, True, False, True])

    assert torch.equal(inference._valid_indices(valid), torch.tensor([0, 2, 3]))
    assert torch.equal(
        inference._valid_indices(valid, has_target), torch.tensor([0, 3])
    )
    assert torch.equal(valid, torch.tensor([True, False, True, True]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vision_prediction_writer_preserves_submission_order(tmp_path):
    output = tmp_path / "predictions.h5"
    writer = inference._PredictionWriter(
        output,
        device=torch.device("cuda"),
        batch_size=2,
        num_classes=2,
        reuse_output=False,
    )

    writer.submit(torch.tensor([2, 0]), torch.tensor([[2.0, 3.0], [0.0, 1.0]]).cuda())
    writer.submit(torch.tensor([1]), torch.tensor([[1.0, 2.0]]).cuda())
    writer.close()

    with h5py.File(output, "r") as predictions:
        np.testing.assert_array_equal(predictions["index"][:], [2, 0, 1])
        np.testing.assert_array_equal(
            predictions["logits"][:],
            [[2.0, 3.0], [0.0, 1.0], [1.0, 2.0]],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vision_prediction_writer_defers_write_errors_until_close(
    tmp_path, monkeypatch
):
    def fail(*_args):
        raise RuntimeError("write failed")

    writer = inference._PredictionWriter(
        tmp_path / "predictions.h5",
        device=torch.device("cuda"),
        batch_size=1,
        num_classes=2,
        reuse_output=False,
    )
    monkeypatch.setattr(inference, "_append_predictions", fail)

    for index in range(3):
        writer.submit(torch.tensor([index]), torch.ones(1, 2, device="cuda"))

    with pytest.raises(RuntimeError, match="write failed"):
        writer.close()


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


@pytest.mark.parametrize(
    "model_config",
    [
        SEWResNet34Config(image_size=32, num_classes=3, neuron_backend="triton"),
        SpikformerConfig(
            image_height=32,
            image_width=32,
            num_classes=3,
            neuron_backend="triton",
        ),
    ],
)
def test_vision_builder_sets_step_mode_before_triton_backend(monkeypatch, model_config):
    monkeypatch.setattr(activation_base, "check_backend_library", lambda _backend: None)

    model = model_config.get_builder_cls()(model_config)._build_canonical_model()
    nodes = [
        module for module in model.modules() if isinstance(module, neuron.BaseNode)
    ]

    assert nodes
    assert all(module.step_mode == "m" for module in nodes)
    assert all(module.backend == "triton" for module in nodes)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [("backend", "cupy", "CuPy"), ("store_v_seq", True, "store_v_seq")],
)
def test_vision_cuda_graph_rejects_unsafe_model_state(attribute, value, message):
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    setattr(model[1], attribute, value)

    with pytest.raises(ValueError, match=message):
        validate_cuda_graph_model(model)


@pytest.mark.parametrize(
    ("match", "kwargs"),
    [
        ("batch_size", {"batch_size": 0}),
        ("dataset_builder", {"dataset_builder": "dataset"}),
        ("data_parallel", {"data_parallel": "ddp"}),
        ("execution_mode", {"execution_mode": "unknown"}),
        (
            "compile",
            {"execution_mode": "compile", "pipeline_parallel_size": 2},
        ),
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
        ("execution_mode", {"execution_mode": "unknown"}),
        (
            "single-rank",
            {"execution_mode": "cuda_graph", "tensor_parallel_size": 2},
        ),
        (
            "memopt",
            {"execution_mode": "cuda_graph", "memopt_level": 1},
        ),
        ("positive", {"cuda_graph_warmup_steps": 0}),
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


def test_vision_distributed_error_is_reduced(monkeypatch):
    reductions = []
    monkeypatch.setattr(
        inference.dist,
        "all_reduce",
        lambda tensor, **_kwargs: reductions.append(tensor.item()),
    )

    with pytest.raises(OSError, match="merge failed"):
        inference._sync_error(
            OSError("merge failed"), torch.device("cpu"), "remote failure"
        )

    assert reductions == [1]

    monkeypatch.setattr(
        inference.dist, "all_reduce", lambda tensor, **_kwargs: tensor.fill_(1)
    )
    with pytest.raises(RuntimeError, match="remote failure"):
        inference._sync_error(None, torch.device("cpu"), "remote failure")


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
    future = training._save_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
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
    future.result()
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


def test_legacy_checkpoint_precision_does_not_inherit_new_triton_fields(tmp_path):
    config = vision.TrainingConfig(
        model=SEWResNet34Config(),
        dataset_builder="package.datasets.build",
        precision=PrecisionConfig(
            mode="bf16",
            triton_storage="bf16",
            triton_fwd="bf16",
        ),
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    recipe = training._recipe(config)
    recipe["precision"] = "bf16"
    (checkpoint / "config.json").write_text(json.dumps(recipe), encoding="utf-8")
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="configuration does not match"):
        training._load_checkpoint(
            checkpoint,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            tp_rank=0,
            pp_rank=0,
        )


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
            device=torch.device("cpu"),
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


def test_vision_async_checkpoint_propagates_write_failure(monkeypatch):
    future = Future()
    future.set_exception(RuntimeError("write failed"))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _tensor: None)

    with pytest.raises(RuntimeError, match="write failed"):
        training._finish_checkpoint(future, torch.device("cpu"))


def test_vision_async_checkpoint_does_not_collect_interrupts(monkeypatch):
    future = Future()
    future.set_exception(KeyboardInterrupt())
    all_reduce_called = False

    def all_reduce(_tensor):
        nonlocal all_reduce_called
        all_reduce_called = True

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    with pytest.raises(KeyboardInterrupt):
        training._finish_checkpoint(future, torch.device("cpu"))

    assert not all_reduce_called


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
