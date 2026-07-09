from __future__ import annotations

import inspect
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
        self._te_forward_kwargs = self._resolve_forward_kwargs()

    def _resolve_forward_kwargs(self) -> dict:
        full_kwargs = {
            "attention_mask": None,
            "qkv_format": self.qkv_layout,
            "attn_mask_type": "no_mask",
        }
        try:
            signature = inspect.signature(self.wrapped.forward)
        except (TypeError, ValueError):
            return full_kwargs
        parameters = signature.parameters
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_var_kwargs or {
            "attention_mask",
            "qkv_format",
            "attn_mask_type",
        }.issubset(parameters):
            return full_kwargs
        if "qkv_format" in parameters:
            return {"qkv_format": self.qkv_layout}
        return {}

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
        if self.training:
            if not math.isclose(dropout_p, self.attention_dropout):
                raise ValueError(
                    "fp8-te SDPA adapter v1 requires fixed adapter dropout "
                    "during training."
                )
        elif not math.isclose(dropout_p, 0.0):
            raise ValueError(
                "fp8-te SDPA adapter v1 requires dropout_p=0.0 during evaluation."
            )
        if scale is not None:
            raise ValueError("fp8-te SDPA adapter v1 does not support custom scale.")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("expected query/key/value with shape [B, H, S, D].")
        if (
            query.shape[1] != self.num_attention_heads
            or query.shape[-1] != self.head_dim
        ):
            raise ValueError(
                "query shape does not match adapter num_attention_heads/head_dim."
            )
        if key.shape[1] != self.num_attention_heads or key.shape[-1] != self.head_dim:
            raise ValueError(
                "key shape does not match adapter num_attention_heads/head_dim."
            )
        if (
            value.shape[1] != self.num_attention_heads
            or value.shape[-1] != self.head_dim
        ):
            raise ValueError(
                "value shape does not match adapter num_attention_heads/head_dim."
            )
        if not (query.shape[0] == key.shape[0] == value.shape[0]):
            raise ValueError("query/key/value must have the same batch size.")
        if key.shape[2] != value.shape[2]:
            raise ValueError("key/value must have the same sequence length.")

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
        return self.wrapped(query, key, value, **self._te_forward_kwargs)


__all__ = ["TransformerEngineDotProductAttentionAdapter"]
