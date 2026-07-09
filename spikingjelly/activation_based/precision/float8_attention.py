from __future__ import annotations

import math

import torch
import torch.nn as nn


def _import_te_pytorch():
    import transformer_engine.pytorch as te

    return te


class TransformerEngineDotProductAttentionAdapter(nn.Module):
    """Adapter for standard PyTorch SDPA tensors backed by TE DotProductAttention.

    The public input/output layout is PyTorch's common ``[B, H, S, D]`` layout.
    Internally the adapter calls TE with explicit ``bshd`` layout tensors.
    """

    def __init__(
        self,
        num_attention_heads: int,
        head_dim: int,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        te = _import_te_pytorch()
        DotProductAttention = te.DotProductAttention
        try:
            self.wrapped = DotProductAttention(
                num_attention_heads=num_attention_heads,
                kv_channels=head_dim,
                attention_dropout=attention_dropout,
                attn_mask_type="no_mask",
            )
        except TypeError:
            self.wrapped = DotProductAttention(
                num_attention_heads,
                head_dim,
                attention_dropout=attention_dropout,
            )
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.attention_dropout = attention_dropout
        self.qkv_layout = "bshd"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        if attn_mask is not None:
            raise ValueError("fp8-te SDPA adapter v1 only supports attn_mask=None.")
        if is_causal:
            raise ValueError("fp8-te SDPA adapter v1 does not support causal masks.")
        if not math.isclose(dropout_p, self.attention_dropout):
            raise ValueError("fp8-te SDPA adapter v1 requires fixed adapter dropout.")
        if scale is not None:
            raise ValueError("fp8-te SDPA adapter v1 does not support custom scale.")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("expected query/key/value with shape [B, H, S, D].")
        if query.shape[1] != self.num_attention_heads or query.shape[-1] != self.head_dim:
            raise ValueError(
                "query shape does not match adapter num_attention_heads/head_dim."
            )

        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        output = self._call_te_attention(q, k, v)
        if isinstance(output, tuple):
            output = output[0]
        return output.transpose(1, 2).contiguous()

    def _call_te_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        try:
            return self.wrapped(
                query,
                key,
                value,
                attention_mask=None,
                qkv_format=self.qkv_layout,
                attn_mask_type="no_mask",
            )
        except TypeError:
            try:
                return self.wrapped(
                    query,
                    key,
                    value,
                    qkv_format=self.qkv_layout,
                )
            except TypeError:
                return self.wrapped(query, key, value)


__all__ = ["TransformerEngineDotProductAttentionAdapter"]
