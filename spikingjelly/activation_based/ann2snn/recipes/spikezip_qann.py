from __future__ import annotations

import copy
import math
import types
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from spikingjelly.activation_based import base
from spikingjelly.activation_based.ann2snn.operators import (
    TDConv2d,
    TDLayerNorm,
    TDLinear,
    TDSoftmax,
)
from spikingjelly.activation_based.ann2snn.recipes.base import ModuleConversionRecipe
from spikingjelly.activation_based.neuron import STBIFNeuron

if TYPE_CHECKING:
    from spikingjelly.activation_based.ann2snn.converter import ModuleConverter


__all__ = [
    "STBIFNeuron",
    "SpikeZIPConv2d",
    "SpikeZIPEmbedding",
    "SpikeZIPLayerNorm",
    "SpikeZIPLinear",
    "SpikeZIPRobertaSelfAttention",
    "SpikeZIPSoftmax",
    "SpikeZIPTFQANNRecipe",
    "SpikeZIPViTSelfAttention",
]


class SpikeZIPLinear(TDLinear):
    def __init__(
        self,
        linear: nn.Linear,
        level: int,
        bias_steps: Optional[int] = None,
        step_mode: str = "s",
    ) -> None:
        super().__init__(
            linear.in_features,
            linear.out_features,
            bias=False,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            step_mode=step_mode,
        )
        with torch.no_grad():
            self.weight.copy_(linear.weight)
        if linear.bias is None:
            self.register_parameter("spikezip_bias", None)
        else:
            self.spikezip_bias = nn.Parameter(linear.bias.detach().clone())
        self.level = int(level)
        self.bias_steps = self.level if bias_steps is None else int(bias_steps)
        if self.bias_steps <= 0:
            raise ValueError("bias_steps must be positive.")
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.realize_time = self.bias_steps
        self.is_work = False

    def _release_bias(self, y: torch.Tensor) -> torch.Tensor:
        if self.spikezip_bias is not None:
            if self.realize_time > 0:
                self.realize_time -= 1
                self.is_work = True
                bias = self.spikezip_bias.to(device=y.device, dtype=y.dtype)
                y = y + bias / self.bias_steps
        return y

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().single_step_forward(x)
        self.is_work = not bool((x == 0).all())
        return self._release_bias(y)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        y_seq = super().multi_step_forward(x_seq)
        active = bool((x_seq != 0).any())
        bias_steps = min(x_seq.shape[0], self.realize_time)
        if self.spikezip_bias is not None and bias_steps > 0:
            bias = self.spikezip_bias.to(device=x_seq.device, dtype=x_seq.dtype)
            view_shape = (1,) * (y_seq.dim() - 1) + (bias.numel(),)
            y_seq[:bias_steps] = (
                y_seq[:bias_steps] + bias.view(view_shape) / self.bias_steps
            )
            self.realize_time -= bias_steps
        self.is_work = active or bias_steps > 0
        return y_seq


class SpikeZIPConv2d(TDConv2d):
    def __init__(
        self,
        conv: nn.Conv2d,
        level: int,
        bias_steps: Optional[int] = None,
        step_mode: str = "s",
    ) -> None:
        super().__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=False,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
            step_mode=step_mode,
        )
        with torch.no_grad():
            self.weight.copy_(conv.weight)
        if conv.bias is None:
            self.register_parameter("spikezip_bias", None)
        else:
            self.spikezip_bias = nn.Parameter(conv.bias.detach().clone())
        self.level = int(level)
        self.bias_steps = self.level if bias_steps is None else int(bias_steps)
        if self.bias_steps <= 0:
            raise ValueError("bias_steps must be positive.")
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.realize_time = self.bias_steps
        self.is_work = False

    def _release_bias(self, y: torch.Tensor) -> torch.Tensor:
        if self.spikezip_bias is not None:
            if self.realize_time > 0:
                self.realize_time -= 1
                self.is_work = True
                bias = self.spikezip_bias.to(device=y.device, dtype=y.dtype)
                y = y + bias.view(1, -1, 1, 1) / self.bias_steps
        return y

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().single_step_forward(x)
        self.is_work = not bool((x == 0).all())
        return self._release_bias(y)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        y_seq = super().multi_step_forward(x_seq)
        active = bool((x_seq != 0).any())
        bias_steps = min(x_seq.shape[0], self.realize_time)
        if self.spikezip_bias is not None and bias_steps > 0:
            bias = self.spikezip_bias.to(device=x_seq.device, dtype=x_seq.dtype)
            y_seq[:bias_steps] = (
                y_seq[:bias_steps] + bias.view(1, 1, -1, 1, 1) / self.bias_steps
            )
            self.realize_time -= bias_steps
        self.is_work = active or bias_steps > 0
        return y_seq


class SpikeZIPEmbedding(base.MemoryModule):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.step_mode = "s"
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.shape = None

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.t == 0:
            y = self.embedding(x)
            self.shape = y.shape
            self.t += 1
            return y
        if self.shape is None:
            raise RuntimeError("SpikeZIPEmbedding has no cached output shape.")
        return torch.zeros(
            self.shape, device=x.device, dtype=self.embedding.weight.dtype
        )

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.shape[0] == 0:
            raise ValueError("SpikeZIPEmbedding expects a non-empty sequence.")
        y0 = self.embedding(x_seq[0])
        y_seq = torch.zeros(
            x_seq.shape[0],
            *y0.shape,
            device=y0.device,
            dtype=y0.dtype,
        )
        y_seq[0] = y0
        self.shape = y0.shape
        self.t = x_seq.shape[0]
        return y_seq


class SpikeZIPLayerNorm(TDLayerNorm):
    def __init__(self, source: nn.LayerNorm) -> None:
        super().__init__(
            source.normalized_shape,
            eps=source.eps,
            elementwise_affine=source.elementwise_affine,
            bias=source.bias is not None,
            device=(source.weight.device if source.weight is not None else None),
            dtype=(source.weight.dtype if source.weight is not None else None),
            step_mode="s",
        )
        self.load_state_dict(source.state_dict())


class SpikeZIPSoftmax(TDSoftmax):
    def __init__(self, dim: int = -1) -> None:
        super().__init__(dim=dim, step_mode="s")


def _spikezip_matmul_delta(a_t, b_t, a_sum, b_sum, transpose_b: bool = False):
    b_t_arg = b_t.transpose(-2, -1) if transpose_b else b_t
    b_sum_arg = b_sum.transpose(-2, -1) if transpose_b else b_sum
    return a_sum @ b_t_arg + a_t @ b_sum_arg - a_t @ b_t_arg


def _temporal_difference(y_cum: torch.Tensor) -> torch.Tensor:
    y_seq = torch.empty_like(y_cum)
    y_seq[0] = y_cum[0]
    y_seq[1:] = y_cum[1:] - y_cum[:-1]
    return y_seq


def _spikezip_matmul_sequence_delta(
    a_seq: torch.Tensor,
    b_seq: torch.Tensor,
    transpose_b: bool = False,
) -> torch.Tensor:
    a_cum = a_seq.cumsum(dim=0)
    b_cum = b_seq.cumsum(dim=0)
    if transpose_b:
        b_cum = b_cum.transpose(-2, -1)
    return _temporal_difference(a_cum @ b_cum)


def _step_modes(module: nn.Module) -> dict[nn.Module, str]:
    return {
        child: child.step_mode
        for child in module.modules()
        if child is not module and hasattr(child, "step_mode")
    }


def _restore_step_modes(step_modes: dict[nn.Module, str]) -> None:
    for module, step_mode in step_modes.items():
        module.step_mode = step_mode


class SpikeZIPRobertaSelfAttention(base.MemoryModule):
    def __init__(self, source: nn.Module, level: int) -> None:
        super().__init__()
        self.num_attention_heads = int(source.num_attention_heads)
        self.attention_head_size = int(source.attention_head_size)
        self.all_head_size = int(
            getattr(
                source,
                "all_head_size",
                self.num_attention_heads * self.attention_head_size,
            )
        )
        self.query = SpikeZIPLinear(source.query, level)
        self.key = SpikeZIPLinear(source.key, level)
        self.value = SpikeZIPLinear(source.value, level)
        self.query_if = STBIFNeuron.from_quantizer(source.query_quan)
        self.key_if = STBIFNeuron.from_quantizer(source.key_quan)
        self.value_if = STBIFNeuron.from_quantizer(source.value_quan)
        self.attn_if = STBIFNeuron.from_quantizer(source.attn_quan)
        self.after_attn_if = STBIFNeuron.from_quantizer(source.after_attn_quan)
        self.softmax = SpikeZIPSoftmax(dim=-1)
        self.dropout = copy.deepcopy(getattr(source, "dropout", nn.Identity()))
        self.step_mode = "s"
        self.t = 0
        self.position_embedding_type = getattr(
            source,
            "position_embedding_type",
            "absolute",
        )
        self.is_decoder = bool(getattr(source, "is_decoder", False))
        if self.position_embedding_type != "absolute":
            raise ValueError(
                "SpikeZIPTFQANNRecipe v1 supports absolute position attention only."
            )

    def reset(self) -> None:
        for module in (
            self.query,
            self.key,
            self.value,
            self.query_if,
            self.key_if,
            self.value_if,
            self.attn_if,
            self.after_attn_if,
            self.softmax,
        ):
            module.reset()
        self.t = 0

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        shape = (*x.size()[:-1], self.num_attention_heads, self.attention_head_size)
        return x.view(shape).permute(0, 2, 1, 3)

    def transpose_sequence_for_scores(self, x_seq: torch.Tensor) -> torch.Tensor:
        shape = (
            *x_seq.size()[:-1],
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x_seq.view(shape).permute(0, 1, 3, 2, 4)

    def single_step_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
    ):
        if encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise ValueError(
                "SpikeZIPTFQANNRecipe v1 does not support cross-attention."
            )
        if past_key_value is not None or self.is_decoder:
            raise ValueError("SpikeZIPTFQANNRecipe v1 does not support decoder cache.")
        if head_mask is not None:
            raise ValueError("SpikeZIPTFQANNRecipe v1 does not support head_mask.")
        query_layer = self.transpose_for_scores(
            self.query_if(self.query(hidden_states))
        )
        key_layer = self.transpose_for_scores(self.key_if(self.key(hidden_states)))
        value_layer = self.transpose_for_scores(
            self.value_if(self.value(hidden_states))
        )

        q_sum = self.transpose_for_scores(self.query_if.accumulated)
        k_sum = self.transpose_for_scores(self.key_if.accumulated)
        scores = _spikezip_matmul_delta(query_layer, key_layer, q_sum, k_sum, True)
        scores = scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None and self.t == 0:
            scores = scores + attention_mask
        attention_probs = self.softmax(scores)
        attention_probs = self.dropout(attention_probs)
        attention_probs = self.attn_if(attention_probs)
        context = _spikezip_matmul_delta(
            attention_probs,
            value_layer,
            self.attn_if.accumulated,
            self.transpose_for_scores(self.value_if.accumulated),
        )
        context = self.after_attn_if(context)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:-2], self.all_head_size)
        self.t += 1
        return (context, attention_probs) if output_attentions else (context,)

    def multi_step_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
    ):
        if encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise ValueError(
                "SpikeZIPTFQANNRecipe v1 does not support cross-attention."
            )
        if past_key_value is not None or self.is_decoder:
            raise ValueError("SpikeZIPTFQANNRecipe v1 does not support decoder cache.")
        if head_mask is not None:
            raise ValueError("SpikeZIPTFQANNRecipe v1 does not support head_mask.")
        previous_step_modes = _step_modes(self)
        try:
            for module in previous_step_modes:
                module.step_mode = "m"
            query_layer = self.transpose_sequence_for_scores(
                self.query_if(self.query(hidden_states))
            )
            key_layer = self.transpose_sequence_for_scores(
                self.key_if(self.key(hidden_states))
            )
            value_layer = self.transpose_sequence_for_scores(
                self.value_if(self.value(hidden_states))
            )
            scores = _spikezip_matmul_sequence_delta(
                query_layer,
                key_layer,
                transpose_b=True,
            )
            scores = scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                scores = scores.clone()
                scores[0] = scores[0] + attention_mask
            attention_probs = self.softmax(scores)
            attention_probs = self.dropout(attention_probs)
            attention_probs = self.attn_if(attention_probs)
            context_seq = _spikezip_matmul_sequence_delta(
                attention_probs,
                value_layer,
            )
            context_seq = self.after_attn_if(context_seq)
            context_seq = context_seq.permute(0, 1, 3, 2, 4).contiguous()
            context_seq = context_seq.view(
                *context_seq.size()[:-2],
                self.all_head_size,
            )
            if output_attentions:
                return context_seq, attention_probs
            return (context_seq,)
        finally:
            _restore_step_modes(previous_step_modes)


class SpikeZIPViTSelfAttention(base.MemoryModule):
    def __init__(self, source: nn.Module, level: int) -> None:
        super().__init__()
        self.num_heads = int(source.num_heads)
        self.head_dim = int(source.head_dim)
        self.scale = float(getattr(source, "scale", self.head_dim**-0.5))
        self.is_softmax = bool(getattr(source, "is_softmax", True))
        self.qkv = SpikeZIPLinear(source.qkv, level, bias_steps=1)
        self.proj = SpikeZIPLinear(source.proj, level, bias_steps=1)
        self.q_if = STBIFNeuron.from_quantizer(source.quan_q)
        self.k_if = STBIFNeuron.from_quantizer(source.quan_k)
        self.v_if = STBIFNeuron.from_quantizer(source.quan_v)
        self.attn_if = STBIFNeuron.from_quantizer(source.attn_quan)
        self.after_attn_if = STBIFNeuron.from_quantizer(source.after_attn_quan)
        self.proj_if = STBIFNeuron.from_quantizer(source.quan_proj)
        self.attn_drop = copy.deepcopy(getattr(source, "attn_drop", nn.Identity()))
        self.proj_drop = copy.deepcopy(getattr(source, "proj_drop", nn.Identity()))
        self.step_mode = "s"
        if self.is_softmax:
            self.softmax = SpikeZIPSoftmax(dim=-1)

    def reset(self) -> None:
        modules = [
            self.qkv,
            self.proj,
            self.q_if,
            self.k_if,
            self.v_if,
            self.attn_if,
            self.after_attn_if,
            self.proj_if,
        ]
        if self.is_softmax:
            modules.append(self.softmax)
        for module in modules:
            module.reset()

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        query = self.q_if(query)
        key = self.k_if(key)
        value = self.v_if(value)

        query = query * self.scale
        q_sum = self.q_if.accumulated * self.scale
        k_sum = self.k_if.accumulated
        attention = _spikezip_matmul_delta(query, key, q_sum, k_sum, True)

        if self.is_softmax:
            attention = self.softmax(attention)
            attention = self.attn_if(attention)
            attention_sum = self.attn_if.accumulated
        else:
            attention = self.attn_if(attention) / seq_len
            attention_sum = self.attn_if.accumulated / seq_len

        attention = self.attn_drop(attention)
        context = _spikezip_matmul_delta(
            attention,
            value,
            attention_sum,
            self.v_if.accumulated,
        )
        context = self.after_attn_if(context)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, channels)
        context = self.proj(context)
        context = self.proj_drop(context)
        return self.proj_if(context)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        previous_step_modes = _step_modes(self)
        try:
            for module in previous_step_modes:
                module.step_mode = "m"
            batch_size, seq_len, channels = x_seq.shape[1:]
            qkv = self.qkv(x_seq).reshape(
                x_seq.shape[0],
                batch_size,
                seq_len,
                3,
                self.num_heads,
                self.head_dim,
            )
            qkv = qkv.permute(3, 0, 1, 4, 2, 5)
            query, key, value = qkv.unbind(0)

            query = self.q_if(query)
            key = self.k_if(key)
            value = self.v_if(value)

            attention = _spikezip_matmul_sequence_delta(
                query * self.scale,
                key,
                transpose_b=True,
            )
            if self.is_softmax:
                attention = self.softmax(attention)
                attention = self.attn_if(attention)
                attention_for_context = attention
            else:
                attention = self.attn_if(attention) / seq_len
                attention_for_context = attention

            attention_for_context = self.attn_drop(attention_for_context)
            context = _spikezip_matmul_sequence_delta(
                attention_for_context,
                value,
            )
            context = self.after_attn_if(context)
            context = context.transpose(2, 3).reshape(
                x_seq.shape[0],
                batch_size,
                seq_len,
                channels,
            )
            context = self.proj(context)
            context = self.proj_drop(context)
            return self.proj_if(context)
        finally:
            _restore_step_modes(previous_step_modes)


def _is_roberta_qattention(module: nn.Module) -> bool:
    return all(
        hasattr(module, name)
        for name in (
            "query",
            "query_quan",
            "key",
            "key_quan",
            "value",
            "value_quan",
            "attn_quan",
            "after_attn_quan",
            "num_attention_heads",
            "attention_head_size",
            "all_head_size",
            "dropout",
        )
    )


def _is_vit_qattention(module: nn.Module) -> bool:
    return all(
        hasattr(module, name)
        for name in (
            "qkv",
            "quan_q",
            "quan_k",
            "quan_v",
            "attn_quan",
            "after_attn_quan",
            "proj",
            "quan_proj",
            "num_heads",
            "head_dim",
            "scale",
            "attn_drop",
            "proj_drop",
        )
    )


def _spikezip_vit_patch_embed(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    patch_embed = model.patch_embed
    used_proj = False
    input_dim = x.dim()
    if hasattr(patch_embed, "proj"):
        x = patch_embed.proj(x)
        used_proj = True
    else:
        x = patch_embed(x)
        if x.dim() == input_dim - 1:
            return x
    if x.dim() == 4:
        x = x.flatten(2).transpose(1, 2)
    elif x.dim() == 5:
        x = x.flatten(3).transpose(2, 3)
    else:
        raise ValueError("SpikeZIP ViT patch_embed expects [B,C,H,W] or [T,B,C,H,W].")
    if used_proj and hasattr(patch_embed, "norm"):
        x = patch_embed.norm(x)
    return x


def _spikezip_vit_token_sequence(
    model: nn.Module,
    time_steps: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cls = model.cls_token.expand(batch_size, -1, -1)
    cls_seq = torch.zeros(
        time_steps,
        *cls.shape,
        device=cls.device,
        dtype=cls.dtype,
    )
    pos_seq = torch.zeros(
        time_steps,
        *model.pos_embed.shape,
        device=model.pos_embed.device,
        dtype=model.pos_embed.dtype,
    )
    cls_seq[0] = cls
    pos_seq[0] = model.pos_embed
    return cls_seq, pos_seq


def _spikezip_vit_single_tokens(
    model: nn.Module,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = getattr(model, "_spikezip_t", 0)
    if t == 0:
        cls = model.cls_token.expand(batch_size, -1, -1)
        pos = model.pos_embed
    else:
        cls = torch.zeros(
            batch_size,
            *model.cls_token.shape[1:],
            device=model.cls_token.device,
            dtype=model.cls_token.dtype,
        )
        pos = torch.zeros_like(model.pos_embed)
    model._spikezip_t = t + 1
    return cls, pos


def _spikezip_vit_forward_features(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        hidden = _spikezip_vit_patch_embed(self, x)
        cls, pos = _spikezip_vit_single_tokens(self, hidden.shape[0])
        hidden = torch.cat((cls, hidden), dim=1)
        pos_drop = getattr(self, "pos_drop", None)
        hidden = hidden + pos
        if pos_drop is not None:
            hidden = pos_drop(hidden)
        if hasattr(self, "blocks"):
            hidden = self.blocks(hidden)
            hidden = self.norm(hidden)
            return hidden[:, 0]
        if hasattr(self, "attn"):
            hidden = hidden + self.attn(self.norm(hidden))
            return self.norm(hidden)[:, 0]
        raise ValueError("SpikeZIP ViT model expects blocks or attn.")
    if x.dim() == 5:
        hidden = _spikezip_vit_patch_embed(self, x)
        cls_seq, pos_seq = _spikezip_vit_token_sequence(
            self,
            hidden.shape[0],
            hidden.shape[1],
        )
        hidden = torch.cat((cls_seq, hidden), dim=2)
        pos_drop = getattr(self, "pos_drop", None)
        hidden = hidden + pos_seq
        if pos_drop is not None:
            hidden = pos_drop(hidden)
        if hasattr(self, "blocks"):
            hidden = self.blocks(hidden)
            hidden = self.norm(hidden)
            return hidden[:, :, 0]
        if hasattr(self, "attn"):
            hidden = hidden + self.attn(self.norm(hidden))
            return self.norm(hidden)[:, :, 0]
        raise ValueError("SpikeZIP ViT model expects blocks or attn.")
    raise ValueError("SpikeZIP ViT forward expects [B,C,H,W] or [T,B,C,H,W].")


def _spikezip_vit_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.forward_features(x))


def _patch_spikezip_vit_forward(model: nn.Module) -> None:
    if not all(
        hasattr(model, name)
        for name in ("patch_embed", "cls_token", "pos_embed", "head")
    ):
        return
    model._spikezip_t = 0
    original_reset = getattr(model, "reset", None)

    def _spikezip_vit_reset(self: nn.Module) -> None:
        self._spikezip_t = 0
        if original_reset is not None:
            original_reset()

    model.reset = types.MethodType(_spikezip_vit_reset, model)
    model.forward_features = types.MethodType(_spikezip_vit_forward_features, model)
    model.forward = types.MethodType(_spikezip_vit_forward, model)


def _validate_spikezip_vit_model(model: nn.Module) -> None:
    has_top_level_vit = all(
        hasattr(model, name)
        for name in ("patch_embed", "cls_token", "pos_embed", "head")
    )
    if not has_top_level_vit:
        return
    if not hasattr(model, "norm"):
        raise ValueError("SpikeZIP ViT model expects a top-level norm module.")
    if not (hasattr(model, "blocks") or hasattr(model, "attn")):
        raise ValueError("SpikeZIP ViT model expects top-level blocks or attn.")


class SpikeZIPTFQANNRecipe(ModuleConversionRecipe):
    def __init__(
        self,
        time_steps: int = 200,
        model_family: str = "roberta",
        strict: bool = True,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <SpikeZIPTFQANNRecipe.__init__-cn>` | :ref:`English <SpikeZIPTFQANNRecipe.__init__-en>`

        ----

        .. _SpikeZIPTFQANNRecipe.__init__-cn:

        * **中文**

        SpikeZIP-TF QANN-to-SNN recipe。该 recipe 不执行 ANN 量化或后训练，
        只把已经兼容 SpikeZIP 的 QANN 原位替换为透明 SNN module。当前版本支持
        ``"roberta"`` 与 ``"vit"`` 两类已量化模型。RoBERTa-style self-attention
        module 需要暴露 ``query``、``key``、``value`` linear layers，
        ``num_attention_heads``、``attention_head_size``、``all_head_size``、
        ``dropout``，以及带有 ``s``、``sym``、``pos_max``、``neg_min``、
        ``level`` 属性的 ``query_quan``、``key_quan``、``value_quan``、
        ``attn_quan``、``after_attn_quan`` quantizers。ViT-style attention
        module 需要暴露 ``qkv``、``proj``、``quan_q``、``quan_k``、
        ``quan_v``、``attn_quan``、``after_attn_quan``、``quan_proj``、
        ``num_heads``、``head_dim``、``scale``、``attn_drop`` 与
        ``proj_drop``。RoBERTa attention mask 仅在第一个时间步加入，随后由
        temporal-difference softmax 的累计状态传播该静态 mask 影响。
        若 quantizer 未显式暴露 ``level``，则按 ``pos_max - neg_min + 1`` 推断。

        :param time_steps: 记录在转换后模型上的时间步元数据。用户仍需显式构造
            单步循环或多步输入序列；该参数不在 recipe 内部编码输入或控制
            ``step_mode``。建议不小于 QANN 的量化 ``level``，否则部分 bias 或
            残余电荷可能无法完全释放。
        :type time_steps: int
        :param model_family: 模型族。支持 ``"roberta"`` 或 ``"vit"``。
        :type model_family: str
        :param strict: 必须为 ``True``。保留该参数用于未来显式放宽支持边界。
        :type strict: bool

        ----

        .. _SpikeZIPTFQANNRecipe.__init__-en:

        * **English**

        SpikeZIP-TF QANN-to-SNN recipe. This recipe does not quantize or
        post-train an ANN; it only converts an already SpikeZIP-compatible QANN
        into transparent SNN modules. The current version supports ``"roberta"`` and
        ``"vit"`` quantized models. RoBERTa-style self-attention modules must
        expose ``query``, ``key`` and ``value`` linear layers,
        ``num_attention_heads``, ``attention_head_size``, ``all_head_size`` and
        ``dropout``, plus ``query_quan``, ``key_quan``, ``value_quan``,
        ``attn_quan`` and ``after_attn_quan`` quantizers with ``s``, ``sym``,
        ``pos_max``, ``neg_min`` and ``level`` attributes. ViT-style attention
        modules must expose ``qkv``, ``proj``, ``quan_q``, ``quan_k``,
        ``quan_v``, ``attn_quan``, ``after_attn_quan``, ``quan_proj``,
        ``num_heads``, ``head_dim``, ``scale``, ``attn_drop`` and
        ``proj_drop``. RoBERTa attention masks are added only at the first
        timestep; the temporal-difference softmax state carries the static mask
        effect afterwards. If a quantizer does not expose ``level`` explicitly,
        the recipe infers it as ``pos_max - neg_min + 1``.

        :param time_steps: Timestep metadata stored on the converted model.
            Users still explicitly construct single-step loops or multi-step
            input sequences; the recipe does not encode inputs or control
            ``step_mode`` internally. It should be no smaller than the QANN
            quantization ``level``; otherwise some bias terms or residual
            membrane charge may not be fully emitted.
        :type time_steps: int
        :param model_family: Model family. Supported values are ``"roberta"``
            and ``"vit"``.
        :type model_family: str
        :param strict: Must be ``True``. The parameter is reserved for future
            explicit boundary relaxation.
        :type strict: bool
        """
        self.time_steps = time_steps
        self.model_family = model_family
        self.strict = strict

    def validate(self, converter: "ModuleConverter") -> None:
        if (
            not isinstance(self.time_steps, int)
            or isinstance(self.time_steps, bool)
            or self.time_steps <= 0
        ):
            raise ValueError("time_steps must be a positive integer.")
        if self.model_family not in ("roberta", "vit"):
            raise ValueError(
                "SpikeZIPTFQANNRecipe supports model_family='roberta' or 'vit'."
            )
        if self.strict is not True:
            raise ValueError("SpikeZIPTFQANNRecipe requires strict=True.")

    def convert_module(self, converter: "ModuleConverter", ann: nn.Module) -> nn.Module:
        model = copy.deepcopy(ann).eval()
        self._replace_weight(model)
        model.ann2snn_recipe = "spikezip_tf_qann"
        model.time_steps = self.time_steps
        model.model_family = self.model_family
        if self.model_family == "vit":
            _validate_spikezip_vit_model(model)
            _patch_spikezip_vit_forward(model)
        return model

    def _replace_weight(self, model: nn.Module) -> None:
        for name, child in list(model.named_children()):
            replacement = None
            if self.model_family == "roberta" and _is_roberta_qattention(child):
                replacement = SpikeZIPRobertaSelfAttention(
                    child,
                    self._level_from_qann(child),
                )
            elif self.model_family == "vit" and _is_vit_qattention(child):
                replacement = SpikeZIPViTSelfAttention(
                    child,
                    self._level_from_qann(child),
                )
            elif isinstance(child, nn.Embedding):
                replacement = SpikeZIPEmbedding(child)
            elif isinstance(child, nn.Conv2d):
                replacement = SpikeZIPConv2d(
                    child,
                    self._level_from_model(model),
                    bias_steps=1 if self.model_family == "vit" else None,
                )
            elif isinstance(child, nn.Linear):
                replacement = SpikeZIPLinear(
                    child,
                    self._level_from_model(model),
                    bias_steps=1 if self.model_family == "vit" else None,
                )
            elif isinstance(child, nn.LayerNorm):
                replacement = SpikeZIPLayerNorm(child)
            elif self._is_quantizer(child):
                replacement = STBIFNeuron.from_quantizer(child)
            elif isinstance(child, nn.ReLU):
                replacement = nn.Identity()

            if replacement is None:
                self._replace_weight(child)
            else:
                setattr(model, name, replacement)

    @staticmethod
    def _is_quantizer(module: nn.Module) -> bool:
        return all(hasattr(module, name) for name in ("s", "sym", "pos_max", "neg_min"))

    @staticmethod
    def _level_from_qann(module: nn.Module) -> int:
        for name in (
            "query_quan",
            "key_quan",
            "value_quan",
            "quan_q",
            "quan_k",
            "quan_v",
        ):
            quantizer = getattr(module, name, None)
            if quantizer is None:
                continue
            if hasattr(quantizer, "level"):
                return int(quantizer.level)
            if hasattr(quantizer, "pos_max") and hasattr(quantizer, "neg_min"):
                return int(float(quantizer.pos_max) - float(quantizer.neg_min) + 1)
        raise ValueError("SpikeZIP QANN attention must expose quantizer level.")

    def _level_from_model(self, module: nn.Module) -> int:
        for child in module.modules():
            if not self._is_quantizer(child):
                continue
            if hasattr(child, "level"):
                return int(child.level)
            return int(float(child.pos_max) - float(child.neg_min) + 1)
        return self.time_steps
