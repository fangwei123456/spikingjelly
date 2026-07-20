from __future__ import annotations

import inspect
import math
import warnings

import torch
import torch.nn as nn


def _import_te_pytorch():
    import transformer_engine.pytorch as te

    return te


class TransformerEngineDotProductAttentionAdapter(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        head_dim: int,
        attention_dropout: float = 0.0,
    ) -> None:
        r"""
        **API Language** - 中文 | English

        中文
        ----

        将 PyTorch SDPA 常用的 ``[B, H, S, D]`` 输入输出布局适配到
        Transformer Engine ``DotProductAttention`` 的 ``bshd`` 布局。兼容 TE
        返回 ``[B, S, H, D]`` 或展平的 ``[B, S, H * D]`` 张量。
        交叉注意力允许查询序列长度与键值序列长度不同，但键和值的序列长度
        必须相同。

        当前仅支持无掩码注意力：``attn_mask`` 必须为 ``None``，
        ``is_causal`` 必须为 ``False``，``scale`` 必须为 ``None``。训练时
        ``dropout_p`` 必须等于固定的 ``attention_dropout``；推理时必须为
        ``0.0``。

        :param num_attention_heads: 注意力头数，必须与输入的 ``H`` 一致。
        :type num_attention_heads: int
        :param head_dim: 每个注意力头的维度，必须与输入的 ``D`` 一致。
        :type head_dim: int
        :param attention_dropout: 训练时使用的固定注意力 dropout 概率。
        :type attention_dropout: float

        English
        -------

        Adapts PyTorch SDPA's common ``[B, H, S, D]`` input/output layout to
        Transformer Engine ``DotProductAttention`` with the ``bshd`` layout. TE
        outputs in either ``[B, S, H, D]`` or flattened ``[B, S, H * D]`` form
        are supported. Cross-attention may use a different query sequence length,
        but the key and value sequence lengths must match.

        This version supports unmasked attention only: ``attn_mask`` must be
        ``None``, ``is_causal`` must be ``False``, and ``scale`` must be
        ``None``. During training, ``dropout_p`` must equal the fixed
        ``attention_dropout``; during evaluation it must be ``0.0``.

        :param num_attention_heads: Number of attention heads; must match input
            dimension ``H``.
        :type num_attention_heads: int
        :param head_dim: Per-head dimension; must match input dimension ``D``.
        :type head_dim: int
        :param attention_dropout: Fixed attention dropout probability for
            training.
        :type attention_dropout: float
        """
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
            try:
                self.wrapped.attn_mask_type = "no_mask"
            except AttributeError:
                warnings.warn(
                    "fp8-te SDPA adapter could not force "
                    "attn_mask_type='no_mask' on the legacy TE "
                    "DotProductAttention; downstream behavior may diverge "
                    "from the v1 contract.",
                    RuntimeWarning,
                    stacklevel=2,
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
        warnings.warn(
            "fp8-te SDPA adapter v1 could not detect "
            "qkv_format/attention_mask/attn_mask_type on the wrapped TE "
            "DotProductAttention.forward; the adapter will rely on TE "
            "defaults, which may break the v1 contract.",
            RuntimeWarning,
            stacklevel=2,
        )
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
            raise ValueError(
                "expected query/key/value with shape [B, H, S, D], but got "
                f"query.shape={tuple(query.shape)}, "
                f"key.shape={tuple(key.shape)}, "
                f"value.shape={tuple(value.shape)}."
            )
        if (
            query.shape[1] != self.num_attention_heads
            or query.shape[-1] != self.head_dim
        ):
            raise ValueError(
                "query shape does not match adapter "
                f"num_attention_heads={self.num_attention_heads} and "
                f"head_dim={self.head_dim}; got query.shape={tuple(query.shape)}."
            )
        if key.shape[1] != self.num_attention_heads or key.shape[-1] != self.head_dim:
            raise ValueError(
                "key shape does not match adapter "
                f"num_attention_heads={self.num_attention_heads} and "
                f"head_dim={self.head_dim}; got key.shape={tuple(key.shape)}."
            )
        if (
            value.shape[1] != self.num_attention_heads
            or value.shape[-1] != self.head_dim
        ):
            raise ValueError(
                "value shape does not match adapter "
                f"num_attention_heads={self.num_attention_heads} and "
                f"head_dim={self.head_dim}; got value.shape={tuple(value.shape)}."
            )
        if not (query.shape[0] == key.shape[0] == value.shape[0]):
            raise ValueError(
                "query/key/value must have the same batch size, but got "
                f"query.shape={tuple(query.shape)}, "
                f"key.shape={tuple(key.shape)}, "
                f"value.shape={tuple(value.shape)}."
            )
        if key.shape[2] != value.shape[2]:
            raise ValueError(
                "key/value must have the same sequence length, but got "
                f"key.shape={tuple(key.shape)} and value.shape={tuple(value.shape)}."
            )

        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        output = self._call_te_attention(q, k, v)
        if isinstance(output, tuple):
            output = output[0]
        expected_shape = (
            query.shape[0],
            query.shape[2],
            self.num_attention_heads,
            self.head_dim,
        )
        if output.ndim == 3 and output.shape == (
            expected_shape[0],
            expected_shape[1],
            expected_shape[2] * expected_shape[3],
        ):
            output = output.reshape(expected_shape)
        elif output.shape != expected_shape:
            raise RuntimeError(
                "Transformer Engine DotProductAttention returned an unsupported "
                f"shape {tuple(output.shape)}; expected {expected_shape} or "
                f"{expected_shape[:2] + (expected_shape[2] * expected_shape[3],)}."
            )
        return output.transpose(1, 2).contiguous()

    def _call_te_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor | tuple:
        return self.wrapped(query, key, value, **self._te_forward_kwargs)


__all__ = ["TransformerEngineDotProductAttentionAdapter"]
