from __future__ import annotations

from copy import deepcopy
import math
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based.ann2snn.recipes.base import ConversionRecipe

if TYPE_CHECKING:
    from spikingjelly.activation_based.ann2snn.converter import Converter


__all__ = [
    "SpikeZIPTFQANNRecipe",
    "STBIFNeuron",
    "SpikeZIPLinear",
    "SpikeZIPEmbedding",
    "SpikeZIPLayerNorm",
    "SpikeZIPSoftmax",
    "SpikeZIPRobertaSelfAttention",
]


def _as_scalar_tensor(value, reference: Optional[torch.Tensor] = None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.tensor(value)
    if reference is not None:
        tensor = tensor.to(device=reference.device, dtype=reference.dtype)
    return tensor.detach().clone()


def _quantizer_attr(module: nn.Module, name: str):
    if not hasattr(module, name):
        raise ValueError(
            f"SpikeZIPTFQANNRecipe requires quantizer {module!r} to expose {name!r}."
        )
    return getattr(module, name)


def _copy_linear(source: nn.Linear) -> nn.Linear:
    target = nn.Linear(
        source.in_features,
        source.out_features,
        bias=source.bias is not None,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    target.load_state_dict(source.state_dict())
    target.train(source.training)
    return target


class STBIFNeuron(nn.Module):
    def __init__(
        self,
        q_threshold,
        level: int,
        sym: bool = False,
        pos_max=None,
        neg_min=None,
    ) -> None:
        super().__init__()
        self.register_buffer("q_threshold", _as_scalar_tensor(q_threshold).float())
        self.register_buffer(
            "pos_max",
            _as_scalar_tensor(
                (
                    level // 2 - 1
                    if sym
                    else level - 1
                )
                if pos_max is None
                else pos_max
            ).float(),
        )
        self.register_buffer(
            "neg_min",
            _as_scalar_tensor(
                (-level // 2 if sym else 0) if neg_min is None else neg_min
            ).float(),
        )
        self.level = int(level)
        self.sym = bool(sym)
        self.reset()

    @classmethod
    def from_quantizer(cls, quantizer: nn.Module) -> "STBIFNeuron":
        scale = _quantizer_attr(quantizer, "s")
        sym = bool(_quantizer_attr(quantizer, "sym"))
        pos_max = _quantizer_attr(quantizer, "pos_max")
        neg_min = _quantizer_attr(quantizer, "neg_min")
        default_level = int(_as_scalar_tensor(pos_max).item()) + 1
        level = int(getattr(quantizer, "level", default_level))
        return cls(scale, level=level, sym=sym, pos_max=pos_max, neg_min=neg_min)

    def reset(self) -> None:
        self.q = None
        self.acc_q = None
        self.cur_output = None
        self.is_work = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_threshold = self.q_threshold.to(device=x.device, dtype=x.dtype)
        normalized = x / q_threshold
        if self.cur_output is None or self.cur_output.shape != normalized.shape:
            self.cur_output = torch.zeros_like(normalized)
            self.acc_q = torch.zeros_like(normalized)
            self.q = torch.zeros_like(normalized) + 0.5

        self.q = self.q + normalized.detach()
        self.acc_q = torch.round(self.acc_q)
        pos_max = self.pos_max.to(device=x.device, dtype=x.dtype)
        neg_min = self.neg_min.to(device=x.device, dtype=x.dtype)
        spike_position = (self.q - 1 >= 0) & (self.acc_q < pos_max)
        neg_spike_position = (self.q < 0) & (self.acc_q > neg_min)

        self.cur_output.zero_()
        self.cur_output[spike_position] = 1
        self.cur_output[neg_spike_position] = -1
        self.acc_q = self.acc_q + self.cur_output
        self.q[spike_position] = self.q[spike_position] - 1
        self.q[neg_spike_position] = self.q[neg_spike_position] + 1
        self.is_work = not (
            bool((normalized == 0).all()) and bool((self.cur_output == 0).all())
        )
        return self.cur_output * q_threshold

    @property
    def accumulated(self) -> torch.Tensor:
        if self.acc_q is None:
            raise RuntimeError("STBIFNeuron has no accumulated state before forward.")
        return self.acc_q * self.q_threshold.to(
            device=self.acc_q.device,
            dtype=self.acc_q.dtype,
        )


class SpikeZIPLinear(nn.Module):
    def __init__(self, linear: nn.Linear, level: int) -> None:
        super().__init__()
        self.linear = _copy_linear(linear)
        self.level = int(level)
        self.reset()

    def reset(self) -> None:
        self.zero_output = None
        self.realize_time = self.level
        self.is_work = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_shape = (*x.shape[:-1], self.linear.out_features)
        if self.zero_output is None or self.zero_output.shape != output_shape:
            self.zero_output = torch.zeros(
                output_shape,
                device=x.device,
                dtype=x.dtype,
            )
        if bool((x == 0).all()):
            self.is_work = False
            if self.linear.bias is not None and self.realize_time > 0:
                self.realize_time -= 1
                self.is_work = True
                return self.zero_output + self.linear.bias / self.level
            return self.zero_output
        y = self.linear(x)
        if self.linear.bias is not None:
            y = y - self.linear.bias
            if self.realize_time > 0:
                self.realize_time -= 1
                y = y + self.linear.bias / self.level
        self.is_work = True
        return y


class SpikeZIPEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.t == 0:
            y = self.embedding(x)
            self.shape = y.shape
            self.t += 1
            return y
        if self.shape is None:
            raise RuntimeError("SpikeZIPEmbedding has no cached output shape.")
        return torch.zeros(self.shape, device=x.device, dtype=self.embedding.weight.dtype)


class SpikeZIPLayerNorm(nn.Module):
    def __init__(self, source: nn.LayerNorm) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(
            source.normalized_shape,
            eps=source.eps,
            elementwise_affine=source.elementwise_affine,
            bias=source.bias is not None,
            device=(source.weight.device if source.weight is not None else None),
            dtype=(source.weight.dtype if source.weight is not None else None),
        )
        self.layernorm.load_state_dict(source.state_dict())
        self.reset()

    def reset(self) -> None:
        self.x_cum = None
        self.y_pre = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x_cum = x if self.x_cum is None else self.x_cum + x
        y = self.layernorm(self.x_cum)
        y_pre = 0.0 if self.y_pre is None else self.y_pre.detach().clone()
        self.y_pre = y
        return y - y_pre


class SpikeZIPSoftmax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
        self.reset()

    def reset(self) -> None:
        self.x_cum = None
        self.y_pre = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x_cum = x if self.x_cum is None else self.x_cum + x
        y = F.softmax(self.x_cum, dim=self.dim)
        y_pre = 0.0 if self.y_pre is None else self.y_pre.detach().clone()
        self.y_pre = y
        return y - y_pre


def _spikezip_matmul_delta(a_t, b_t, a_sum, b_sum, transpose_b: bool = False):
    b_t_arg = b_t.transpose(-2, -1) if transpose_b else b_t
    b_sum_arg = b_sum.transpose(-2, -1) if transpose_b else b_sum
    return a_sum @ b_t_arg + a_t @ b_sum_arg - a_t @ b_t_arg


class SpikeZIPRobertaSelfAttention(nn.Module):
    def __init__(self, source: nn.Module, level: int) -> None:
        super().__init__()
        self.num_attention_heads = int(source.num_attention_heads)
        self.attention_head_size = int(source.attention_head_size)
        self.all_head_size = int(source.all_head_size)
        self.query = SpikeZIPLinear(source.query, level)
        self.key = SpikeZIPLinear(source.key, level)
        self.value = SpikeZIPLinear(source.value, level)
        self.query_if = STBIFNeuron.from_quantizer(source.query_quan)
        self.key_if = STBIFNeuron.from_quantizer(source.key_quan)
        self.value_if = STBIFNeuron.from_quantizer(source.value_quan)
        self.attn_if = STBIFNeuron.from_quantizer(source.attn_quan)
        self.after_attn_if = STBIFNeuron.from_quantizer(source.after_attn_quan)
        self.softmax = SpikeZIPSoftmax(dim=-1)
        self.dropout = deepcopy(source.dropout)
        self.position_embedding_type = getattr(
            source,
            "position_embedding_type",
            "absolute",
        )
        self.is_decoder = bool(getattr(source, "is_decoder", False))
        if self.position_embedding_type != "absolute":
            raise ValueError("SpikeZIPTFQANNRecipe v1 supports absolute position attention only.")

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

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(shape).permute(0, 2, 1, 3)

    def forward(
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
            raise ValueError("SpikeZIPTFQANNRecipe v1 does not support cross-attention.")
        if past_key_value is not None or self.is_decoder:
            raise ValueError("SpikeZIPTFQANNRecipe v1 does not support decoder cache.")
        query_layer = self.transpose_for_scores(self.query_if(self.query(hidden_states)))
        key_layer = self.transpose_for_scores(self.key_if(self.key(hidden_states)))
        value_layer = self.transpose_for_scores(self.value_if(self.value(hidden_states)))

        q_sum = self.transpose_for_scores(self.query_if.accumulated)
        k_sum = self.transpose_for_scores(self.key_if.accumulated)
        scores = _spikezip_matmul_delta(query_layer, key_layer, q_sum, k_sum, True)
        scores = scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            scores = scores + attention_mask
        attention_probs = self.softmax(scores)
        attention_probs = self.dropout(attention_probs)
        attention_probs = self.attn_if(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context = _spikezip_matmul_delta(
            attention_probs,
            value_layer,
            self.attn_if.accumulated,
            self.transpose_for_scores(self.value_if.accumulated),
        )
        context = self.after_attn_if(context)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        return (context, attention_probs) if output_attentions else (context,)


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
        )
    )


def _reset_spikezip_modules(module: nn.Module) -> None:
    for child in module.modules():
        reset = getattr(child, "reset", None)
        if child is not module and callable(reset):
            reset()


class SpikeZIPSNNWrapper(nn.Module):
    def __init__(self, model: nn.Module, time_steps: int, encoding: str) -> None:
        super().__init__()
        self.model = model
        self.time_steps = time_steps
        self.encoding = encoding
        self.ann2snn_recipe = "spikezip_tf_qann"

    def reset(self) -> None:
        _reset_spikezip_modules(self.model)

    def forward(self, *args, return_sequences: bool = False, **kwargs):
        self.reset()
        outputs = []
        accumulated = None
        for step in range(self.time_steps):
            step_args = []
            for arg in args:
                if torch.is_tensor(arg) and arg.is_floating_point():
                    if self.encoding == "analog":
                        step_args.append(arg if step == 0 else torch.zeros_like(arg))
                    else:
                        step_args.append(arg / self.time_steps)
                else:
                    step_args.append(arg)
            output = self.model(*step_args, **kwargs)
            logits = output[0] if isinstance(output, tuple) else output
            accumulated = logits if accumulated is None else accumulated + logits
            outputs.append(accumulated)
        if return_sequences:
            return accumulated, torch.stack(outputs, dim=0)
        return accumulated


class SpikeZIPTFQANNRecipe(ConversionRecipe):
    def __init__(
        self,
        time_steps: int = 200,
        model_family: str = "roberta",
        encoding: str = "analog",
        strict: bool = True,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <SpikeZIPTFQANNRecipe.__init__-cn>` | :ref:`English <SpikeZIPTFQANNRecipe.__init__-en>`

        ----

        .. _SpikeZIPTFQANNRecipe.__init__-cn:

        * **中文**

        SpikeZIP-TF QANN-to-SNN recipe。该 recipe 不执行 ANN 量化或后训练，
        只把已经兼容 SpikeZIP 的 QANN 转换为 SNN wrapper。当前版本支持
        RoBERTa-style self-attention module，要求对应 module 暴露 ``query``、
        ``key``、``value`` linear layers，以及带有 ``s``、``sym``、
        ``pos_max``、``neg_min``、``level`` 属性的 ``query_quan``、
        ``key_quan``、``value_quan``、``attn_quan``、``after_attn_quan``
        quantizers。

        :param time_steps: SNN wrapper 执行的时间步数。应不小于 QANN 的量化
            ``level``，否则部分 bias 或残余电荷可能无法完全释放。
        :type time_steps: int
        :param model_family: 模型族。当前仅支持 ``"roberta"``。
        :type model_family: str
        :param encoding: 浮点输入的时间编码方式。``"analog"`` 表示首个时间步
            输入原值、后续输入零；``"rate"`` 表示每个时间步输入 ``x / time_steps``。
            整数 token 与 keyword masks 保持静态。
        :type encoding: str
        :param strict: 必须为 ``True``。保留该参数用于未来显式放宽支持边界。
        :type strict: bool

        ----

        .. _SpikeZIPTFQANNRecipe.__init__-en:

        * **English**

        SpikeZIP-TF QANN-to-SNN recipe. This recipe does not quantize or
        post-train an ANN; it only converts an already SpikeZIP-compatible QANN
        into an SNN wrapper. The current version supports RoBERTa-style
        self-attention modules that expose ``query``, ``key`` and ``value``
        linear layers, plus ``query_quan``, ``key_quan``, ``value_quan``,
        ``attn_quan`` and ``after_attn_quan`` quantizers with ``s``, ``sym``,
        ``pos_max``, ``neg_min`` and ``level`` attributes.

        :param time_steps: Number of timesteps executed by the SNN wrapper. It
            should be no smaller than the QANN quantization ``level``; otherwise
            some bias terms or residual membrane charge may not be fully emitted.
        :type time_steps: int
        :param model_family: Model family. Currently only ``"roberta"`` is
            supported.
        :type model_family: str
        :param encoding: Temporal encoding for floating-point inputs. ``"analog"``
            feeds the original value at the first timestep and zeros afterwards;
            ``"rate"`` feeds ``x / time_steps`` at every timestep. Integer tokens
            and keyword masks stay static.
        :type encoding: str
        :param strict: Must be ``True``. The parameter is reserved for future
            explicit boundary relaxation.
        :type strict: bool
        """
        self.time_steps = time_steps
        self.model_family = model_family
        self.encoding = encoding
        self.strict = strict

    def requires_fx_trace(self) -> bool:
        return False

    def validate(self, converter: "Converter") -> None:
        if (
            not isinstance(self.time_steps, int)
            or isinstance(self.time_steps, bool)
            or self.time_steps <= 0
        ):
            raise ValueError("time_steps must be a positive integer.")
        if self.model_family != "roberta":
            raise ValueError(
                "SpikeZIPTFQANNRecipe v1 supports model_family='roberta' only."
            )
        if self.encoding not in ("analog", "rate"):
            raise ValueError("encoding must be 'analog' or 'rate'.")
        if self.strict is not True:
            raise ValueError("SpikeZIPTFQANNRecipe requires strict=True.")

    def before_trace(self, converter: "Converter", ann: nn.Module) -> nn.Module:
        model = deepcopy(ann).eval()
        self._replace_weight(model)
        return SpikeZIPSNNWrapper(model, self.time_steps, self.encoding)

    def finalize(self, converter: "Converter", fx_model: nn.Module) -> nn.Module:
        return fx_model

    def _replace_weight(self, model: nn.Module) -> None:
        for name, child in list(model.named_children()):
            replacement = None
            if _is_roberta_qattention(child):
                replacement = SpikeZIPRobertaSelfAttention(
                    child,
                    self._level_from_qann(child),
                )
            elif isinstance(child, nn.Embedding):
                replacement = SpikeZIPEmbedding(child)
            elif isinstance(child, nn.Linear):
                replacement = SpikeZIPLinear(child, self._level_from_model(model))
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
        for name in ("query_quan", "key_quan", "value_quan"):
            quantizer = getattr(module, name, None)
            if quantizer is not None and hasattr(quantizer, "level"):
                return int(quantizer.level)
        raise ValueError("SpikeZIP QANN attention must expose quantizer level.")

    def _level_from_model(self, module: nn.Module) -> int:
        for child in module.modules():
            if self._is_quantizer(child) and hasattr(child, "level"):
                return int(child.level)
        return self.time_steps
