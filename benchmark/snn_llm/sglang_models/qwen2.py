from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, make_layers

from . import _load_stage_weights


def _qcfs(value: torch.Tensor, scale: torch.Tensor, time_steps: int) -> torch.Tensor:
    scale = scale.to(value)
    shape = [1] * value.ndim
    shape[-1] = scale.numel()
    scale = scale.reshape(shape)
    positive = torch.round(torch.clamp(torch.relu(value) / scale, 0, time_steps))
    negative = torch.round(torch.clamp(torch.relu(-value) / scale, 0, time_steps))
    return (positive - negative) * scale


def _encode(hidden: torch.Tensor, scale: torch.Tensor, *, mean: bool) -> torch.Tensor:
    dense = hidden.mean(1) if mean else hidden.sum(1)
    output = torch.zeros_like(hidden)
    output[:, 0] = _qcfs(dense, scale, hidden.shape[1])
    return output


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        variance = hidden.float().square().mean(dim=-1, keepdim=True)
        return hidden * torch.rsqrt(variance + self.eps).to(hidden) * self.weight


class _QwenAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.time_steps = config.snn_time_steps
        self.total_heads = config.snn_num_attention_heads
        self.total_kv_heads = config.snn_num_key_value_heads
        self.head_dim = config.head_dim
        tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        if self.total_heads % tp_size or self.total_kv_heads % tp_size:
            raise ValueError("Qwen2 SNN heads must be divisible by TP size.")
        self.local_heads = self.total_heads // tp_size
        self.local_kv_heads = self.total_kv_heads // tp_size
        self.q_size = self.local_heads * self.head_dim
        self.kv_size = self.local_kv_heads * self.head_dim
        self.qkv = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_heads,
            self.total_kv_heads,
            bias=bool(config.attention_bias),
            quant_config=quant_config,
            prefix=add_prefix("qkv", prefix),
        )
        self.proj = RowParallelLinear(
            self.total_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("proj", prefix),
        )
        self.rotary = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=float(config.rope_parameters["rope_theta"]),
        )
        self.attention = RadixAttention(
            self.time_steps * self.local_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.time_steps * self.local_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attention", prefix),
        )
        self.register_buffer(
            "query_scale", torch.ones(self.total_heads * self.head_dim)
        )
        self.register_buffer(
            "key_scale", torch.ones(self.total_kv_heads * self.head_dim)
        )
        self.register_buffer(
            "value_scale", torch.ones(self.total_kv_heads * self.head_dim)
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        token_count = hidden.shape[0]
        flat = hidden.reshape(token_count * self.time_steps, self.hidden_size)
        qkv, _ = self.qkv(flat)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        temporal_positions = positions.repeat_interleave(self.time_steps)
        q, k = self.rotary(temporal_positions, q, k)
        q = _encode(
            q.reshape(token_count, self.time_steps, self.q_size),
            self.query_scale.chunk(self.tp_size)[self.tp_rank],
            mean=False,
        )
        k = _encode(
            k.reshape(token_count, self.time_steps, self.kv_size),
            self.key_scale.chunk(self.tp_size)[self.tp_rank],
            mean=False,
        )
        v = _encode(
            v.reshape(token_count, self.time_steps, self.kv_size),
            self.value_scale.chunk(self.tp_size)[self.tp_rank],
            mean=False,
        )

        def pack(value: torch.Tensor) -> torch.Tensor:
            return value.flatten(1)

        output = self.attention(pack(q), pack(k), pack(v), forward_batch)
        output = output.reshape(token_count * self.time_steps, self.q_size)
        output, _ = self.proj(output)
        return output.reshape(token_count, self.time_steps, self.hidden_size)


class _QwenMLP(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.time_steps = config.snn_time_steps
        tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.gate_up = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up", prefix),
        )
        self.down = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down", prefix),
        )
        self.register_buffer("scale", torch.ones(config.intermediate_size))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        token_count = hidden.shape[0]
        gate_up, _ = self.gate_up(hidden.reshape(-1, self.hidden_size))
        gate, up = gate_up.chunk(2, dim=-1)
        activated = (F.silu(gate) * up).reshape(token_count, self.time_steps, -1)
        activated = _encode(
            activated,
            self.scale.chunk(self.tp_size)[self.tp_rank],
            mean=False,
        )
        output, _ = self.down(activated.flatten(0, 1))
        return output.reshape(token_count, self.time_steps, self.hidden_size)


class _QwenLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = _QwenAttention(
            config, layer_id, quant_config, add_prefix("attn", prefix)
        )
        self.mlp_norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = _QwenMLP(config, layer_id, quant_config, add_prefix("mlp", prefix))
        self.register_buffer("input_scale", torch.ones(config.hidden_size))

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        normalized = self.input_norm(
            _encode(hidden, self.input_scale, mean=True)
            if self.layer_id == 0
            else hidden
        )
        hidden = hidden + self.attn(positions, normalized, forward_batch)
        return hidden + self.mlp(self.mlp_norm(hidden))


class _QwenModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.time_steps = config.snn_time_steps
        self.embedding = (
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("embedding", prefix),
            )
            if self.pp_group.is_first_rank
            else PPMissingLayer()
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: _QwenLayer(
                config,
                idx,
                quant_config,
                prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        self.final_norm = (
            _RMSNorm(config.hidden_size, config.rms_norm_eps)
            if self.pp_group.is_last_rank
            else PPMissingLayer()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor | PPProxyTensors:
        if self.pp_group.is_first_rank:
            hidden = self.embedding(input_ids)[:, None].expand(-1, self.time_steps, -1)
        else:
            hidden = pp_proxy_tensors["hidden_states"]
        for index in range(self.start_layer, self.end_layer):
            hidden = self.layers[index](positions, hidden, forward_batch)
        if not self.pp_group.is_last_rank:
            return PPProxyTensors({"hidden_states": hidden})
        return self.final_norm(hidden).sum(dim=1)


class SpikingJellyQwen2ForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.model = _QwenModel(config, quant_config, add_prefix("model", prefix))
        self.lm_head = (
            ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
            if self.pp_group.is_last_rank
            else PPMissingLayer()
        )
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden = self.model(input_ids, positions, forward_batch, pp_proxy_tensors)
        if not self.pp_group.is_last_rank:
            return hidden
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        _load_stage_weights(self, weights, "Qwen2")


EntryClass = SpikingJellyQwen2ForCausalLM

__all__ = ["SpikingJellyQwen2ForCausalLM"]
