import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from benchmark.snn_llm import qwen2


class _Linear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _RadixAttention(nn.Module):
    def __init__(self, heads, head_dim, scale, *, num_kv_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.num_kv_heads = num_kv_heads


def _load_qwen_runtime(monkeypatch):
    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    for name in ("sglang", "sglang.srt", "sglang.srt.layers"):
        value = module(name)
        value.__path__ = []

    module("sglang.srt.distributed", get_pp_group=lambda: None)
    module(
        "sglang.srt.layers.linear",
        MergedColumnParallelLinear=_Linear,
        QKVParallelLinear=_Linear,
        RowParallelLinear=_Linear,
    )
    module("sglang.srt.layers.logits_processor", LogitsProcessor=_Linear)
    module("sglang.srt.layers.quantization.base_config", QuantizationConfig=object)
    module("sglang.srt.layers.radix_attention", RadixAttention=_RadixAttention)
    module("sglang.srt.layers.rotary_embedding", get_rope=lambda *args, **kwargs: None)
    module("sglang.srt.layers.utils", PPMissingLayer=_Linear)
    module(
        "sglang.srt.layers.vocab_parallel_embedding",
        ParallelLMHead=_Linear,
        VocabParallelEmbedding=_Linear,
    )
    module(
        "sglang.srt.model_executor.forward_batch_info",
        ForwardBatch=object,
        PPProxyTensors=dict,
    )
    module(
        "sglang.srt.runtime_context",
        get_parallel=lambda: SimpleNamespace(tp_size=1, tp_rank=0),
    )
    module(
        "sglang.srt.utils",
        add_prefix=lambda name, prefix: f"{prefix}.{name}" if prefix else name,
        make_layers=lambda *args, **kwargs: (nn.ModuleList(), 0, 0),
    )

    package_name = "benchmark.snn_llm.sglang_models"
    package = module(package_name, _load_stage_weights=lambda *args: None)
    package.__path__ = [str(Path(__file__).parents[1] / "snn_llm" / "sglang_models")]
    name = f"{package_name}.qwen2"
    path = Path(__file__).parents[1] / "snn_llm" / "sglang_models" / "qwen2.py"
    spec = importlib.util.spec_from_file_location(name, path)
    runtime = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, runtime)
    spec.loader.exec_module(runtime)
    return runtime


def _config(**overrides):
    values = {
        "hidden_size": 4,
        "snn_time_steps": 3,
        "snn_num_attention_heads": 4,
        "snn_num_key_value_heads": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 1,
        "attention_bias": False,
        "max_position_embeddings": 16,
        "rope_parameters": {"rope_theta": 10000.0},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen_attention_uses_only_physical_base_heads(monkeypatch):
    runtime = _load_qwen_runtime(monkeypatch)
    runtime.get_parallel = lambda: SimpleNamespace(tp_size=2, tp_rank=0)

    attention = runtime._QwenAttention(_config(), 0, None, "model.layers.0.attn")

    assert attention.attention.heads == 2
    assert attention.attention.num_kv_heads == 1

    with pytest.raises(ValueError, match="re-export"):
        runtime._QwenAttention(
            _config(num_attention_heads=12, num_key_value_heads=6),
            0,
            None,
            "model.layers.0.attn",
        )


@pytest.mark.parametrize("time_steps", [1, 4])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_qwen_attention_matches_temporal_head_reference(
    monkeypatch, time_steps, tp_size
):
    runtime = _load_qwen_runtime(monkeypatch)
    runtime.get_parallel = lambda: SimpleNamespace(tp_size=tp_size, tp_rank=0)
    attention = runtime._QwenAttention(
        _config(snn_time_steps=time_steps), 0, None, "model.layers.0.attn"
    )
    torch.manual_seed(1)

    class Linear(nn.Linear):
        def forward(self, hidden):
            return super().forward(hidden), None

    class Rotary(nn.Module):
        def forward(self, positions, query, key):
            return query, key

    class Recorder(nn.Module):
        def forward(self, query, key, value, forward_batch):
            self.inputs = query, key, value
            return sdpa(
                query, key, value, attention.local_heads, attention.local_kv_heads
            )

    def sdpa(query, key, value, heads, kv_heads):
        query = query.view(-1, heads, 1).transpose(0, 1)
        key = key.view(-1, kv_heads, 1).transpose(0, 1)
        value = value.view(-1, kv_heads, 1).transpose(0, 1)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query,
            key.repeat_interleave(heads // kv_heads, dim=0),
            value.repeat_interleave(heads // kv_heads, dim=0),
            is_causal=True,
        )
        return attended.transpose(0, 1).flatten(1)

    attention.qkv = Linear(4, attention.q_size + 2 * attention.kv_size)
    attention.rotary = Rotary()
    attention.attention = Recorder()
    attention.proj = Linear(attention.q_size, 4, bias=False)
    attention.query_scale.fill_(0.125)
    attention.key_scale.fill_(0.125)
    attention.value_scale.fill_(0.125)
    hidden = torch.randn(5, time_steps, 4)

    with torch.no_grad():
        output = attention(torch.arange(5), hidden, SimpleNamespace())
        qkv, _ = attention.qkv(hidden.flatten(0, 1))
        query, key, value = qkv.split(
            (attention.q_size, attention.kv_size, attention.kv_size), dim=-1
        )
        encoded = [
            runtime._encode(
                tensor.reshape(5, time_steps, -1), scale.chunk(tp_size)[0], mean=False
            )
            for tensor, scale in (
                (query, attention.query_scale),
                (key, attention.key_scale),
                (value, attention.value_scale),
            )
        ]
        expected = sdpa(
            *(tensor.flatten(1) for tensor in encoded),
            time_steps * attention.local_heads,
            time_steps * attention.local_kv_heads,
        )
        expected, _ = attention.proj(expected.reshape(5 * time_steps, -1))
        expected = expected.reshape_as(output)

    for tensor, heads in zip(
        attention.attention.inputs,
        (attention.local_heads, attention.local_kv_heads, attention.local_kv_heads),
    ):
        assert tensor.shape == (5, heads) and tensor.is_contiguous()
    torch.testing.assert_close(output, expected)
    assert torch.count_nonzero(output[:, 0]) > 0
    assert torch.count_nonzero(output[:, 1:]) == 0


def test_qwen_sglang_export_uses_physical_base_heads(monkeypatch):
    source_config = SimpleNamespace(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        attention_bias=True,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    captured = {}

    transformers = ModuleType("transformers")
    transformers.AutoConfig = SimpleNamespace(from_pretrained=lambda _: source_config)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        qwen2,
        "export_sglang_artifact",
        lambda *args, **kwargs: captured.update(kwargs["artifact_config"]),
    )

    qwen2.export_sglang(
        SimpleNamespace(source_path=Path("source"), time_steps=4, transformer=object()),
        object(),
        Path("checkpoint"),
        Path("artifact"),
    )

    assert captured["num_attention_heads"] == 4
    assert captured["num_key_value_heads"] == 2
    assert captured["snn_num_attention_heads"] == 4
    assert captured["snn_num_key_value_heads"] == 2
    assert captured["snn_time_steps"] == 4
