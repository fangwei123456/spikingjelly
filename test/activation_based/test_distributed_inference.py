import asyncio
import contextlib
import json
import math
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from benchmark.snn_llm import sglang_benchmark, spikelm
from benchmark.snn_llm.spikelm import SpikeLMConfig
from benchmark.snn_llm.qwen2 import _gated_tensor, _reorder_qkv
from benchmark.snn_llm.sglang_benchmark import _prompts, _run_requests
from benchmark import vision_inference
from benchmark.vision_inference import build_synthetic_dataset
from spikingjelly.activation_based.distributed.llm import (
    EvaluationConfig,
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
    _write_tensor_shards,
)
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


def test_vision_inference_synthetic_dataset_follows_input_layout():
    dataset = build_synthetic_dataset(2, 10, 8, 4, "NTCHW")

    assert dataset[0][0].shape == (4, 3, 8, 8)


def test_evaluation_config_and_padding_mask():
    model = SpikeLMConfig(
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
                external_model_package="benchmark.snn_llm.sglang_models",
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
                external_model_package="benchmark.snn_llm.sglang_models",
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
    monkeypatch.setenv("SGLANG_EXTERNAL_MODEL_PACKAGE", "original")

    with open_sglang_engine(
        SGLangEngineConfig(
            artifact=artifact,
            external_model_package="benchmark.snn_llm.sglang_models",
        )
    ) as engine:
        assert engine.kwargs["disable_cuda_graph"]
        assert engine.kwargs["disable_prefill_cuda_graph"]
        assert engine.kwargs["disable_decode_cuda_graph"]

    assert observed == [
        "benchmark.snn_llm.sglang_models",
        "shutdown",
    ]
    assert os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] == "original"


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
    assert torch.equal(
        _gated_tensor(stage, "gated"), torch.tensor([[0], [1], [10], [11]])
    )
    assert torch.equal(
        _reorder_qkv(torch.arange(8), heads=4, kv_heads=2),
        torch.tensor([0, 1, 4, 5, 2, 6, 3, 7]),
    )

    readers[0][1].values["replicated"] = torch.tensor([4])
    with pytest.raises(ValueError, match="differs across ranks"):
        stage.merge_tensor("replicated")


def test_sglang_benchmark_builds_variable_prompts_with_shared_prefix():
    prompts = _prompts(
        count=4,
        input_length=16,
        shared_prefix_length=4,
        vocab_size=32,
        seed=7,
    )

    assert [len(prompt) for prompt in prompts] == [16, 15, 16, 15]
    assert all(prompt[:4] == prompts[0][:4] for prompt in prompts)

    full_prefix = _prompts(
        count=4,
        input_length=16,
        shared_prefix_length=16,
        vocab_size=32,
        seed=7,
    )
    assert [len(prompt) for prompt in full_prefix] == [16, 16, 16, 16]


def test_sglang_benchmark_validates_prompt_array_shape(tmp_path, monkeypatch):
    import numpy as np

    path = tmp_path / "prompts.npy"
    np.save(path, np.ones((2, 4), dtype=np.int64))
    assert sglang_benchmark._load_prompts(path, 2, 4) == [[1] * 4, [1] * 4]

    with pytest.raises(ValueError, match="shape"):
        sglang_benchmark._load_prompts(path, 3, 4)
    with pytest.raises(ValueError, match="shape"):
        sglang_benchmark._load_prompts(path, 2, 5)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sglang_benchmark",
            "--artifact",
            "artifact",
            "--prompts-npy",
            str(path),
            "--input-length",
            "4",
            "--shared-prefix-length",
            "5",
        ],
    )
    with pytest.raises(SystemExit):
        sglang_benchmark.main()


def test_sglang_benchmark_flushes_cache_before_each_repeat(
    tmp_path, monkeypatch, capsys
):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text('{"vocab_size": 16}', encoding="utf-8")
    flushes = []

    class Engine:
        def __init__(self):
            self.loop = asyncio.new_event_loop()

        def flush_cache(self):
            flushes.append(None)

        async def async_generate(self, **_kwargs):
            async def stream():
                yield {"output_ids": [1], "meta_info": {"completion_tokens": 1}}

            return stream()

    @contextlib.contextmanager
    def open_engine(_config):
        engine = Engine()
        try:
            yield engine
        finally:
            engine.loop.close()

    monkeypatch.setattr(sglang_benchmark.llm, "open_sglang_engine", open_engine)
    monkeypatch.setattr(
        sglang_benchmark, "_gpu_memory", lambda stop, _peaks, _errors: stop.wait()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sglang_benchmark",
            "--artifact",
            str(artifact),
            "--requests",
            "1",
            "--input-length",
            "1",
            "--output-length",
            "1",
            "--repeats",
            "2",
        ],
    )

    sglang_benchmark.main()

    assert len(flushes) == 2
    capsys.readouterr()


def test_spikelm_sglang_export_supports_tied_embeddings(monkeypatch):
    transformer = SimpleNamespace(
        num_attention_heads=4,
        hidden_size=16,
        num_layers=0,
        ffn_hidden_size=32,
        layernorm_epsilon=1e-5,
    )
    config = SpikeLMConfig(
        transformer=transformer,
        vocab_size=16,
        max_sequence_length=8,
        time_steps=2,
        share_embeddings_and_output_weights=True,
    )
    embedding = torch.arange(8)

    def merge_tensor(name, **_kwargs):
        if name == "output_layer.weight":
            raise AssertionError("tied export must reuse the embedding")
        return {
            "embedding.word_embeddings.weight": embedding,
            "decoder.final_layernorm.weight": torch.ones(1),
            "decoder.final_layernorm.bias": torch.zeros(1),
        }[name]

    stage = SimpleNamespace(
        is_first=True,
        is_last=True,
        layer_offset=0,
        local_layer_count=0,
        tensor_names=lambda: (),
        merge_tensor=merge_tensor,
    )
    tensors = dict(spikelm._sglang_tensors(config, stage))
    assert tensors["lm_head.weight"] is embedding

    captured = {}
    monkeypatch.setattr(
        spikelm,
        "export_sglang_artifact",
        lambda *_args, **kwargs: captured.update(kwargs),
    )
    spikelm.export_sglang(config, object(), Path("checkpoint"), Path("artifact"))
    assert captured["artifact_config"]["tie_word_embeddings"] is True


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
        iter((("a", torch.ones(2, 3)), ("b", torch.ones(4)))),
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


def test_sglang_benchmark_rejects_failed_gpu_memory_poll(monkeypatch):
    monkeypatch.setattr(
        sglang_benchmark.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="nvidia-smi failed"
        ),
    )
    errors = []

    sglang_benchmark._gpu_memory(threading.Event(), [0], errors)

    assert errors == ["nvidia-smi failed"]


def test_sglang_benchmark_rejects_gpu_memory_poll_timeout(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise sglang_benchmark.subprocess.TimeoutExpired("nvidia-smi", 5)

    monkeypatch.setattr(sglang_benchmark.subprocess, "run", timeout)
    errors = []

    sglang_benchmark._gpu_memory(threading.Event(), [0], errors)

    assert errors


def test_sglang_benchmark_samples_gpu_memory_when_already_stopped(monkeypatch):
    monkeypatch.setattr(
        sglang_benchmark.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="123\n", stderr=""
        ),
    )
    stop = threading.Event()
    stop.set()
    peaks = [0]
    errors = []

    sglang_benchmark._gpu_memory(stop, peaks, errors)

    assert peaks == [123]
    assert errors == []


def test_sglang_benchmark_merges_incremental_stream_tokens():
    class Engine:
        async def async_generate(self, **_kwargs):
            async def stream():
                yield {
                    "output_ids": [1, 2],
                    "meta_info": {"completion_tokens": 2},
                }
                yield {
                    "output_ids": [3, 4],
                    "meta_info": {"completion_tokens": 4},
                }

            return stream()

    requests, _ = asyncio.run(_run_requests(Engine(), [[7, 8]], 4))

    assert requests[0]["tokens"] == 4
    assert requests[0]["output_ids"] == [1, 2, 3, 4]


def test_sglang_benchmark_rejects_stream_chunks_without_token_counts():
    class Engine:
        async def async_generate(self, **_kwargs):
            async def stream():
                yield {"output_ids": [1]}
                yield {"output_ids": [2]}

            return stream()

    with pytest.raises(RuntimeError, match="completion_tokens"):
        asyncio.run(_run_requests(Engine(), [[7, 8]], 2))


def test_vision_benchmark_subset_indices_span_the_dataset():
    assert vision_inference._spread_indices(10, 4) == [0, 3, 6, 9]
