"""SGLang runtime model for the benchmark SpikeLM recipe."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
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


class _TemporalSpikingLayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        time_steps: int,
        decay: float,
        amplitude: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        self.time_steps = time_steps
        self.decay = decay
        self.register_buffer("amplitude", torch.full((time_steps,), amplitude))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(hidden)
        voltage = torch.zeros_like(hidden[:, 0])
        previous_spike = torch.zeros_like(voltage)
        amplitudes = self.amplitude.to(hidden)
        spikes = []
        for step in range(self.time_steps):
            amplitude = amplitudes[step]
            if step == 0:
                voltage = voltage + hidden[:, step]
            else:
                voltage = (
                    voltage * self.decay * (amplitudes[step - 1] - previous_spike)
                    + hidden[:, step]
                )
            previous_spike = (voltage / amplitude).clamp(-1.0, 1.0).round() * amplitude
            spikes.append(previous_spike)
        return torch.stack(spikes, dim=1)


class _TemporalAttention(nn.Module):
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
        tp_size = get_parallel().tp_size
        if self.total_heads % tp_size:
            raise ValueError("SpikeLM attention heads must be divisible by TP size.")
        self.local_heads = self.total_heads // tp_size
        self.head_dim = self.hidden_size // self.total_heads
        self.local_hidden_size = self.local_heads * self.head_dim
        self.qkv = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv", prefix),
        )
        self.proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("proj", prefix),
        )
        self.rotary = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.attention = RadixAttention(
            self.time_steps * self.local_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.time_steps * self.local_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attention", prefix),
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
        q, k, v = qkv.chunk(3, dim=-1)
        temporal_positions = positions.repeat_interleave(self.time_steps)
        q, k = self.rotary(temporal_positions, q, k)

        packed_size = self.time_steps * self.local_heads * self.head_dim
        output = self.attention(
            q.reshape(token_count, packed_size),
            k.reshape(token_count, packed_size),
            v.reshape(token_count, packed_size),
            forward_batch,
        )
        output = output.reshape(token_count * self.time_steps, self.local_hidden_size)
        output, _ = self.proj(output)
        return output.reshape(token_count, self.time_steps, self.hidden_size)


class _TemporalMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.time_steps = config.snn_time_steps
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = hidden.shape
        hidden, _ = self.fc1(hidden.reshape(-1, self.hidden_size))
        hidden = F.gelu(hidden)
        hidden, _ = self.fc2(hidden)
        return hidden.reshape(shape)


class _SpikeLMLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        norm = dict(
            hidden_size=config.hidden_size,
            eps=config.layer_norm_epsilon,
            time_steps=config.snn_time_steps,
            decay=config.snn_spike_decay,
            amplitude=config.snn_spike_amplitude,
        )
        self.attn_norm = _TemporalSpikingLayerNorm(**norm)
        self.attn = _TemporalAttention(
            config, layer_id, quant_config, add_prefix("attn", prefix)
        )
        self.mlp_norm = _TemporalSpikingLayerNorm(**norm)
        self.mlp = _TemporalMLP(config, quant_config, add_prefix("mlp", prefix))

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden = hidden + self.attn(positions, self.attn_norm(hidden), forward_batch)
        return hidden + self.mlp(self.mlp_norm(hidden))


class _SpikeLMModel(nn.Module):
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
            lambda idx, prefix: _SpikeLMLayer(
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
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
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
        return self.final_norm(hidden).mean(dim=1)


class SpikingJellySpikeLMForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.model = _SpikeLMModel(config, quant_config, add_prefix("model", prefix))
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
        _load_stage_weights(self, weights, "SpikeLM")


EntryClass = SpikingJellySpikeLMForCausalLM

__all__ = ["SpikingJellySpikeLMForCausalLM"]
