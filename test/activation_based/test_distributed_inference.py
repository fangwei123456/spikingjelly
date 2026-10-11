import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from spikingjelly.activation_based.distributed.llm import (
    EvaluationConfig,
    ModelConfig,
    SGLangExportStage,
    SGLangEngineConfig,
    generate,
    open_sglang_engine,
)
from spikingjelly.activation_based.distributed.llm.inference import (
    _EvaluationDataset,
    _perplexity,
)
from spikingjelly.activation_based.distributed.llm.temporal import _reduce_time_batch
from spikingjelly.activation_based.distributed.llm.sglang_export import (
    _copy_tokenizer,
    _validate_model_parallel_topology,
    _write_tensor_shards,
    export_sglang_artifact,
)
from spikingjelly.activation_based.distributed.llm import sglang as sglang_runtime
from spikingjelly.activation_based.distributed.llm.sglang import _validate_artifact


def test_inference_temporal_logit_reduction():
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    assert torch.equal(
        _reduce_time_batch(logits, 2, "sum"),
        torch.tensor([[6.0, 8.0], [10.0, 12.0]]),
    )
    assert torch.equal(
        _reduce_time_batch(logits, 2, "mean"),
        torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
    )
    with pytest.raises(ValueError, match="reduction"):
        _reduce_time_batch(logits, 2, "max")


def test_perplexity_overflows_only_when_math_exp_does():
    assert _perplexity(100.0) == math.exp(100.0)
    assert math.isinf(_perplexity(1000.0))


def test_mcore_generate_points_low_level_callers_to_generate_mcore():
    with pytest.raises(TypeError, match="use generate_mcore"):
        generate(object(), torch.ones((1, 2), dtype=torch.long))


def test_evaluation_config_and_padding_mask():
    model = ModelConfig(
        transformer=object(), vocab_size=16, max_sequence_length=4, time_steps=2
    )
    config = EvaluationConfig(
        model=model,
        checkpoint=Path("checkpoint"),
        dataset_builder="package.dataset.build",
        sequence_length=4,
        micro_batch_size=2,
    )
    assert config.model is model

    with pytest.raises(ValueError, match="timing_warmup_batches"):
        EvaluationConfig(
            model=model,
            checkpoint=Path("checkpoint"),
            dataset_builder="package.dataset.build",
            sequence_length=4,
            micro_batch_size=2,
            timing_warmup_batches=-1,
        )
    with pytest.raises(ValueError, match="pipeline_microbatches"):
        EvaluationConfig(
            model=model,
            checkpoint=Path("checkpoint"),
            dataset_builder="package.dataset.build",
            sequence_length=4,
            micro_batch_size=2,
            pipeline_microbatches=0,
        )

    class Tokens(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {
                "input_ids": torch.arange(4),
                "labels": torch.arange(4),
            }

    padded = _EvaluationDataset(Tokens(), padded_size=2, sequence_length=4)

    assert torch.equal(padded[0]["loss_mask"], torch.ones(4))
    assert torch.equal(padded[1]["loss_mask"], torch.zeros(4))


def _sglang_artifact(tmp_path, *, schema=2):
    artifact = tmp_path / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["TestForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    (artifact / "spikingjelly_sglang.json").write_text(
        json.dumps(
            {
                "schema_version": schema,
                "dtype": "bfloat16",
                "recipe_name": "test",
            }
        ),
        encoding="utf-8",
    )
    (artifact / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    (artifact / "model.safetensors").write_bytes(b"test")
    return artifact


def test_sglang_engine_rejects_invalid_artifact_and_package(tmp_path):
    artifact = _sglang_artifact(tmp_path, schema=99)

    with pytest.raises(ValueError, match="Unsupported"):
        with open_sglang_engine(
            SGLangEngineConfig(
                artifact=artifact,
                external_model_package=__name__,
            )
        ):
            pass

    manifest_path = artifact / "spikingjelly_sglang.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest.pop("recipe_name")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="recipe_name"):
        _validate_artifact(
            SGLangEngineConfig(
                artifact=artifact,
                external_model_package=__name__,
            )
        )

    with pytest.raises(ValueError, match="external_model_package"):
        SGLangEngineConfig(artifact=artifact, external_model_package="")

    artifact = _sglang_artifact(tmp_path / "valid")
    with pytest.raises(ImportError, match="unavailable"):
        with open_sglang_engine(
            SGLangEngineConfig(
                artifact=artifact,
                external_model_package="package_that_does_not_exist",
            )
        ):
            pass


def test_sglang_engine_manages_lifecycle_and_environment(tmp_path, monkeypatch):
    artifact = _sglang_artifact(tmp_path)
    observed = []

    class Engine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            observed.append(os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"])

        def shutdown(self):
            observed.append("shutdown")

    monkeypatch.setitem(
        sys.modules,
        "sglang",
        SimpleNamespace(Engine=Engine),
    )
    monkeypatch.setattr(sglang_runtime, "_external_model_package", None)
    monkeypatch.setenv("SGLANG_EXTERNAL_MODEL_PACKAGE", "original")

    with open_sglang_engine(
        SGLangEngineConfig(
            artifact=artifact,
            external_model_package=__name__,
        )
    ) as engine:
        assert engine.kwargs["disable_cuda_graph"]
        assert engine.kwargs["disable_prefill_cuda_graph"]
        assert engine.kwargs["disable_decode_cuda_graph"]

    assert observed == [
        __name__,
        "shutdown",
    ]
    assert os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] == "original"
    with pytest.raises(ValueError, match="cannot change"):
        with open_sglang_engine(
            SGLangEngineConfig(
                artifact=artifact, external_model_package="another.package"
            )
        ):
            pass


def test_sglang_export_reconstructs_tp_tensor_layouts():
    class Reader:
        def __init__(self, values):
            self.values = values

        def get_tensor(self, name):
            return self.values[name]

        def keys(self):
            return self.values.keys()

    readers = {
        0: (
            Reader(
                {"replicated": torch.tensor([3]), "gated": torch.tensor([[0], [10]])}
            ),
            Reader(
                {"replicated": torch.tensor([3]), "gated": torch.tensor([[1], [11]])}
            ),
        ),
        1: (Reader({"other": torch.tensor([5])}), Reader({"other": torch.tensor([5])})),
    }
    stage = SGLangExportStage(
        pipeline_rank=0,
        is_first=True,
        is_last=False,
        layer_offset=0,
        local_layer_count=2,
        readers=readers.__getitem__,
    )

    assert stage.tensor_names() == ("replicated", "gated")
    assert torch.equal(stage.merge_tensor("replicated"), torch.tensor([3]))
    assert torch.equal(stage.merge_tensor("other", pipeline_rank=1), torch.tensor([5]))
    readers[0][1].values["replicated"] = torch.tensor([4])
    with pytest.raises(ValueError, match="differs across ranks"):
        stage.merge_tensor("replicated")


def test_sglang_export_rejects_expert_parallelism_before_runtime(tmp_path, monkeypatch):
    transformer = SimpleNamespace(
        bf16=True,
        params_dtype=torch.bfloat16,
        expert_model_parallel_size=2,
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(ValueError, match="expert parallelism"):
        export_sglang_artifact(
            transformer,
            lambda *_args: None,
            checkpoint,
            tmp_path / "artifact",
            artifact_config={"architectures": ["TestForCausalLM"]},
            stage_tensors=lambda _stage: (),
        )


def test_sglang_export_rejects_mismatched_preinitialized_topology():
    parallel_state = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 4,
        get_pipeline_model_parallel_world_size=lambda: 1,
        get_context_parallel_world_size=lambda: 1,
    )
    transformer = SimpleNamespace(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
    )

    with pytest.raises(ValueError, match="source TP/PP/CP"):
        _validate_model_parallel_topology(transformer, parallel_state)


def test_sglang_export_copies_generic_tokenizer_assets_and_counts_parameters(
    tmp_path, monkeypatch
):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.model").write_text("model", encoding="utf-8")
    (tokenizer / "custom_tokenizer.py").write_text("class Tokenizer: pass\n")
    (tokenizer / "config.json").write_text("model config", encoding="utf-8")
    (tokenizer / "model.safetensors").write_bytes(b"weights")
    output = tmp_path / "artifact"
    output.mkdir()
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "safetensors.torch",
        SimpleNamespace(save_file=lambda _tensors, path: path.write_bytes(b"weights")),
    )

    _copy_tokenizer(tokenizer, output)
    weight_map, parameter_count = _write_tensor_shards(
        iter(
            (
                ("a", torch.ones(2, 3, dtype=torch.bfloat16)),
                ("b", torch.ones(4, dtype=torch.bfloat16)),
            )
        ),
        output,
        "test",
        1024,
    )

    assert (output / "tokenizer.model").read_text(encoding="utf-8") == "model"
    assert (output / "custom_tokenizer.py").is_file()
    assert not (output / "config.json").exists()
    assert not (output / "model.safetensors").exists()
    assert set(weight_map) == {"a", "b"}
    assert parameter_count == 10

    with pytest.raises(ValueError, match="torch.bfloat16"):
        _write_tensor_shards(iter((("fp32", torch.ones(1)),)), output, "invalid", 1024)

    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        _copy_tokenizer(tokenizer_file, output)
