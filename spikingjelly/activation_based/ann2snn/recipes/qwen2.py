from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based.ann2snn.operators import (
    SNNElementWiseProduct,
    TDLinear,
    TDRMSNorm,
    TDScaledDotProductAttention,
    TDSiLU,
)
from spikingjelly.activation_based.ann2snn.qcfs import SignedQCFSSequenceEncoder

from .base import ModuleConversionRecipe

if TYPE_CHECKING:
    from ..converter import ModuleConverter

__all__ = [
    "Qwen2SNNCalibration",
    "Qwen2SNNConfig",
    "Qwen2SNNModel",
    "Qwen2SNNRecipe",
    "calibrate_qwen2_snn",
]


@dataclass(frozen=True)
class Qwen2SNNConfig:
    time_steps: int = 32
    calibration_levels: int = 16
    calibration_quantile: float = 1.0
    calibration_reservoir_size: int = 4096
    calibration_seed: int = 20260719
    neuron_backend: str = "torch"

    def __post_init__(self) -> None:
        if not isinstance(self.time_steps, int) or isinstance(self.time_steps, bool):
            raise TypeError("time_steps must be an integer.")
        if self.time_steps <= 0:
            raise ValueError("time_steps must be positive.")
        if not isinstance(self.calibration_levels, int) or isinstance(
            self.calibration_levels, bool
        ):
            raise TypeError("calibration_levels must be an integer.")
        if not 0 < self.calibration_levels <= self.time_steps:
            raise ValueError("calibration_levels must lie in [1, time_steps].")
        if not 0.0 < float(self.calibration_quantile) <= 1.0:
            raise ValueError("calibration_quantile must lie in (0, 1].")
        if not isinstance(self.calibration_reservoir_size, int) or isinstance(
            self.calibration_reservoir_size, bool
        ):
            raise TypeError("calibration_reservoir_size must be an integer.")
        if self.calibration_reservoir_size <= 0:
            raise ValueError("calibration_reservoir_size must be positive.")
        if not isinstance(self.calibration_seed, int) or isinstance(
            self.calibration_seed, bool
        ):
            raise TypeError("calibration_seed must be an integer.")
        if self.neuron_backend not in ("torch", "triton"):
            raise ValueError("neuron_backend must be 'torch' or 'triton'.")


Qwen2SNNConfig.__init__.__doc__ = r"""
**API Language** - :ref:`中文 <Qwen2SNNConfig-cn>` | :ref:`English <Qwen2SNNConfig-en>`

----

.. _Qwen2SNNConfig-cn:

* **中文**

Qwen2 后训练 SNN 转换配置。``calibration_levels`` 决定逐通道 QCFS 步长，
``time_steps`` 决定输出时间维长度及可表示计数范围。

:param time_steps: 正整数时间步数。
:type time_steps: int
:param calibration_levels: 正整数校准等级，不得超过 ``time_steps``。
:type calibration_levels: int
:param calibration_quantile: 逐通道绝对激活校准分位点，位于 ``(0, 1]``。
:type calibration_quantile: float
:param calibration_reservoir_size: 分位点校准保留的确定性 token 样本数。
:type calibration_reservoir_size: int
:param calibration_seed: reservoir priority sampling 的随机种子。
:type calibration_seed: int
:param neuron_backend: ``"torch"`` 或 ``"triton"``。
:type neuron_backend: str
:raises TypeError: ``time_steps``、``calibration_levels``、reservoir size 或 seed
    不是非布尔整数。
:raises ValueError: 数值范围无效，或 neuron backend 不受支持。

----

.. _Qwen2SNNConfig-en:

* **English**

Configuration for post-training Qwen2 SNN conversion. ``calibration_levels``
defines the per-channel QCFS step, while ``time_steps`` defines the explicit
temporal length and representable spike-count range.

:param time_steps: Positive number of time steps.
:type time_steps: int
:param calibration_levels: Positive calibration level count no greater than
    ``time_steps``.
:type calibration_levels: int
:param calibration_quantile: Per-channel absolute-activation quantile in
    ``(0, 1]``.
:type calibration_quantile: float
:param calibration_reservoir_size: Number of deterministic token samples kept
    for quantile calibration.
:type calibration_reservoir_size: int
:param calibration_seed: Random seed for reservoir priority sampling.
:type calibration_seed: int
:param neuron_backend: ``"torch"`` or ``"triton"``.
:type neuron_backend: str
:raises TypeError: If a time-step, level, reservoir-size, or seed argument is not
    a non-boolean integer.
:raises ValueError: If a numeric range is invalid or the neuron backend is not
    supported.
"""


@dataclass(frozen=True)
class Qwen2SNNCalibration:
    input_scale: torch.Tensor
    layer_scales: Tuple[Mapping[str, torch.Tensor], ...]
    time_steps: int
    calibration_levels: int
    calibration_quantile: float
    calibration_reservoir_size: int
    calibration_seed: int
    valid_token_count: int

    def state_dict(self) -> Dict[str, object]:
        r"""
        **API Language** - :ref:`中文 <Qwen2SNNCalibration.state_dict-cn>` | :ref:`English <Qwen2SNNCalibration.state_dict-en>`

        ----

        .. _Qwen2SNNCalibration.state_dict-cn:

        * **中文**

        返回仅包含张量与基础 Python 值的校准状态，可用 ``torch.save`` 保存并以
        ``weights_only=True`` 加载。

        :return: 校准状态 mapping。
        :rtype: Dict[str, object]

        ----

        .. _Qwen2SNNCalibration.state_dict-en:

        * **English**

        Return a calibration state containing only tensors and basic Python values.
        It can be saved with ``torch.save`` and loaded with ``weights_only=True``.

        :return: Calibration state mapping.
        :rtype: Dict[str, object]
        """
        tensors: Dict[str, torch.Tensor] = {"input": self.input_scale.detach().cpu()}
        for index, group in enumerate(self.layer_scales):
            for name, value in group.items():
                tensors[f"layer.{index}.{name}"] = value.detach().cpu()
        return {
            "time_steps": self.time_steps,
            "calibration_levels": self.calibration_levels,
            "calibration_quantile": self.calibration_quantile,
            "calibration_reservoir_size": self.calibration_reservoir_size,
            "calibration_seed": self.calibration_seed,
            "valid_token_count": self.valid_token_count,
            "tensors": tensors,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "Qwen2SNNCalibration":
        r"""
        **API Language** - :ref:`中文 <Qwen2SNNCalibration.from_state_dict-cn>` | :ref:`English <Qwen2SNNCalibration.from_state_dict-en>`

        ----

        .. _Qwen2SNNCalibration.from_state_dict-cn:

        * **中文**

        从 :meth:`state_dict` 返回的 mapping 恢复校准对象，并校验张量键、层索引和
        每层 Q/K/V/MLP scale 集合。

        :param state: 序列化校准状态。
        :type state: Mapping[str, object]
        :return: 恢复的校准对象。
        :rtype: Qwen2SNNCalibration
        :raises ValueError: 状态缺少必要张量、张量键无效、层索引不连续或每层
            scale 集合不完整。
        :raises KeyError: 状态缺少必要元数据。

        ----

        .. _Qwen2SNNCalibration.from_state_dict-en:

        * **English**

        Rebuild calibration from :meth:`state_dict` output while validating tensor
        keys, contiguous layer indices, and each layer's Q/K/V/MLP scale set.

        :param state: Serialized calibration state.
        :type state: Mapping[str, object]
        :return: Restored calibration object.
        :rtype: Qwen2SNNCalibration
        :raises ValueError: If required tensors are missing, tensor keys are invalid,
            layer indices are not contiguous, or a layer scale set is incomplete.
        :raises KeyError: If required metadata is missing.
        """
        tensors = state.get("tensors")
        if not isinstance(tensors, Mapping) or not isinstance(
            tensors.get("input"), torch.Tensor
        ):
            raise ValueError("Calibration state must contain tensor key 'input'.")
        layers: Dict[int, Dict[str, torch.Tensor]] = {}
        for key, value in tensors.items():
            if key == "input":
                continue
            parts = str(key).split(".")
            if (
                len(parts) != 3
                or parts[0] != "layer"
                or not isinstance(value, torch.Tensor)
            ):
                raise ValueError(f"Invalid calibration tensor key {key!r}.")
            layers.setdefault(int(parts[1]), {})[parts[2]] = value
        indices = sorted(layers)
        if indices != list(range(len(indices))):
            raise ValueError("Calibration layer indices must be contiguous from zero.")
        required = {"query", "key", "value", "mlp"}
        if any(set(layers[index]) != required for index in indices):
            raise ValueError(
                "Each calibration layer requires query/key/value/mlp scales."
            )
        return cls(
            input_scale=tensors["input"],
            layer_scales=tuple(layers[index] for index in indices),
            time_steps=int(state["time_steps"]),
            calibration_levels=int(state["calibration_levels"]),
            calibration_quantile=float(state["calibration_quantile"]),
            calibration_reservoir_size=int(state["calibration_reservoir_size"]),
            calibration_seed=int(state["calibration_seed"]),
            valid_token_count=int(state["valid_token_count"]),
        )


Qwen2SNNCalibration.__init__.__doc__ = r"""
**API Language** - :ref:`中文 <Qwen2SNNCalibration-cn>` | :ref:`English <Qwen2SNNCalibration-en>`

----

.. _Qwen2SNNCalibration-cn:

* **中文**

Qwen2 离线多步 SNN 转换的不可变校准产物。``input_scale`` 的最后一维对应
embedding hidden channels；``layer_scales`` 按 decoder 顺序保存每层
``query``、``key``、``value`` 和 ``mlp`` 的逐通道 scale。元数据必须与转换时的
:class:`Qwen2SNNConfig` 一致。

:param input_scale: 输入 embedding 的正逐通道 scale，形状为 ``[hidden_size]``。
:type input_scale: torch.Tensor
:param layer_scales: 每个 decoder 的 Q/K/V/MLP scale mapping。
:type layer_scales: Tuple[Mapping[str, torch.Tensor], ...]
:param time_steps: 校准目标的时间步数。
:type time_steps: int
:param calibration_levels: 生成 scale 时使用的量化等级。
:type calibration_levels: int
:param calibration_quantile: 逐通道绝对激活分位点。
:type calibration_quantile: float
:param calibration_reservoir_size: 分位点 observer 的样本容量。
:type calibration_reservoir_size: int
:param calibration_seed: 确定性 priority sampling 随机种子。
:type calibration_seed: int
:param valid_token_count: 校准实际观察的非 padding token 数。
:type valid_token_count: int

----

.. _Qwen2SNNCalibration-en:

* **English**

Immutable calibration artifact for offline multi-step Qwen2 SNN conversion.
The final dimension of ``input_scale`` corresponds to embedding hidden
channels. ``layer_scales`` follows decoder order and stores per-channel scales
for ``query``, ``key``, ``value``, and ``mlp``. Its metadata must match the
:class:`Qwen2SNNConfig` used for conversion.

:param input_scale: Positive per-channel input-embedding scale with shape
    ``[hidden_size]``.
:type input_scale: torch.Tensor
:param layer_scales: Q/K/V/MLP scale mapping for every decoder.
:type layer_scales: Tuple[Mapping[str, torch.Tensor], ...]
:param time_steps: Time-step count targeted by calibration.
:type time_steps: int
:param calibration_levels: Quantization level count used to derive scales.
:type calibration_levels: int
:param calibration_quantile: Per-channel absolute-activation quantile.
:type calibration_quantile: float
:param calibration_reservoir_size: Sample capacity of the quantile observer.
:type calibration_reservoir_size: int
:param calibration_seed: Seed for deterministic priority sampling.
:type calibration_seed: int
:param valid_token_count: Number of non-padding tokens observed by calibration.
:type valid_token_count: int
"""


class _ChannelObserver:
    def __init__(self, quantile: float, reservoir_size: int, seed: int) -> None:
        self.quantile = quantile
        self.reservoir_size = reservoir_size
        self._samples: Optional[torch.Tensor] = None
        self._priorities: Optional[torch.Tensor] = None
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self.valid_token_count = 0

    def update(self, activation: torch.Tensor, mask: torch.Tensor) -> None:
        valid = activation.detach()[mask.to(torch.bool)]
        if valid.numel() == 0:
            raise ValueError("Qwen2 calibration requires at least one valid token.")
        self.valid_token_count += int(valid.shape[0])
        if self.quantile == 1.0:
            value = valid.abs().amax(0, keepdim=True)
            self._samples = (
                value if self._samples is None else torch.maximum(self._samples, value)
            )
        else:
            priorities = torch.rand(valid.shape[0], generator=self._generator)
            keep = min(self.reservoir_size, int(valid.shape[0]))
            selected_priorities, selected_indices = priorities.topk(keep)
            selected = (
                valid.index_select(0, selected_indices.to(valid.device))
                .abs()
                .to("cpu", torch.float32)
            )
            if self._samples is not None:
                selected = torch.cat((self._samples, selected), dim=0)
                selected_priorities = torch.cat(
                    (self._priorities, selected_priorities), dim=0
                )
            keep = min(self.reservoir_size, int(selected.shape[0]))
            self._priorities, indices = selected_priorities.topk(keep)
            self._samples = selected.index_select(0, indices)

    def scale(self, levels: int) -> torch.Tensor:
        if self._samples is None:
            raise RuntimeError("Qwen2 calibration observer received no values.")
        if self.quantile == 1.0:
            bound = self._samples.squeeze(0)
        else:
            bound = torch.quantile(self._samples, self.quantile, dim=0)
        return bound.clamp_min(1e-6) / levels


def _qwen_layers(model: nn.Module) -> Sequence[nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, (nn.ModuleList, list, tuple)):
        raise TypeError("Qwen2 model must expose model.layers.")
    return layers


def _validate_qwen2(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "qwen2":
        raise TypeError("Qwen2SNNRecipe requires a Hugging Face Qwen2 causal LM.")
    if bool(getattr(config, "use_sliding_window", False)):
        raise ValueError("Qwen2 sliding-window attention is not supported.")
    if bool(getattr(config, "use_mrope", False)):
        raise ValueError("Qwen2 MRoPE is not supported.")
    if getattr(config, "hidden_act", "silu") != "silu":
        raise ValueError("Qwen2SNNRecipe requires hidden_act='silu'.")
    if model.training:
        raise ValueError("Qwen2SNNRecipe is inference-only; call model.eval().")
    if len(_qwen_layers(model)) != int(config.num_hidden_layers):
        raise ValueError("Qwen2 config and concrete decoder layer count disagree.")


def _rotary(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        half = value.shape[-1] // 2
        return torch.cat((-value[..., half:], value[..., :half]), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


def _repeat_kv(value: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return value
    time, batch, heads, sequence, width = value.shape
    return (
        value[:, :, :, None]
        .expand(time, batch, heads, groups, sequence, width)
        .reshape(time, batch, heads * groups, sequence, width)
    )


def _causal_mask(
    mask: torch.Tensor, query_length: int, past_length: int
) -> torch.Tensor:
    key_length = past_length + query_length
    if mask.dim() != 2 or mask.shape[1] != key_length:
        raise ValueError("attention_mask length must equal past_length + query_length.")
    query = torch.arange(past_length, key_length, device=mask.device).unsqueeze(1)
    key = torch.arange(key_length, device=mask.device).unsqueeze(0)
    return (key <= query)[None, None] & mask[:, None, None, :].to(torch.bool)


def _copy_linear(source: nn.Linear) -> TDLinear:
    target = TDLinear(
        source.in_features,
        source.out_features,
        bias=source.bias is not None,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        target.weight.copy_(source.weight)
        if source.bias is not None:
            target.bias.copy_(source.bias)
    target.weight.requires_grad = source.weight.requires_grad
    if source.bias is not None:
        target.bias.requires_grad = source.bias.requires_grad
    return target.train(source.training)


def _copy_embedding(source: nn.Embedding) -> nn.Embedding:
    target = nn.Embedding(
        source.num_embeddings,
        source.embedding_dim,
        padding_idx=source.padding_idx,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        target.weight.copy_(source.weight)
    target.weight.requires_grad = source.weight.requires_grad
    return target.train(source.training)


def _copy_norm(source: nn.Module) -> TDRMSNorm:
    weight = getattr(source, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("Qwen2 RMSNorm must expose tensor weight.")
    eps = getattr(source, "variance_epsilon", getattr(source, "eps", None))
    target = TDRMSNorm(weight.shape, eps=eps, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        target.weight.copy_(weight)
    target.weight.requires_grad = weight.requires_grad
    return target.train(source.training)


class _Qwen2Cache:
    def __init__(self, layer_count: int) -> None:
        self.keys: list[Optional[torch.Tensor]] = [None] * layer_count
        self.values: list[Optional[torch.Tensor]] = [None] * layer_count

    def get(self, index: int):
        return self.keys[index], self.values[index]

    def append(self, index: int, key: torch.Tensor, value: torch.Tensor) -> None:
        old_key, old_value = self.get(index)
        if old_key is not None:
            key = torch.cat((old_key, key), dim=2)
            value = torch.cat((old_value, value), dim=2)
        self.keys[index] = key.detach()
        self.values[index] = value.detach()

    def get_seq_length(self) -> int:
        return 0 if self.keys[0] is None else int(self.keys[0].shape[2])

    def reorder_cache(self, indices: torch.Tensor) -> None:
        self.keys = [None if value is None else value[indices] for value in self.keys]
        self.values = [
            None if value is None else value[indices] for value in self.values
        ]

    def storage_summary(self) -> Dict[str, object]:
        tensors = [
            value
            for pair in zip(self.keys, self.values, strict=True)
            for value in pair
            if value is not None
        ]
        logical_bytes = sum(value.numel() * value.element_size() for value in tensors)
        storages: Dict[tuple[str, int], int] = {}
        for value in tensors:
            storage = value.untyped_storage()
            storages.setdefault(
                (str(value.device), storage.data_ptr()), storage.nbytes()
            )
        return {
            "per_layer_key_value_shape": [
                {
                    "key": None if key is None else list(key.shape),
                    "value": None if value is None else list(value.shape),
                }
                for key, value in zip(self.keys, self.values, strict=True)
            ],
            "stored_cache_logical_bytes": logical_bytes,
            "stored_cache_unique_storage_bytes": sum(storages.values()),
            "theoretical_cache_bytes": logical_bytes,
            "temporal_prefix_materialization_bytes": 0,
        }


def _exact_sequence(value: torch.Tensor, time_steps: int) -> torch.Tensor:
    zeros = torch.zeros_like(value).unsqueeze(0).expand(time_steps - 1, *value.shape)
    return torch.cat((value.unsqueeze(0), zeros), dim=0)


class _Qwen2Decoder(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        config: object,
        scales: Mapping[str, torch.Tensor],
        conversion: Qwen2SNNConfig,
        index: int,
    ) -> None:
        super().__init__()
        attention = source.self_attn
        mlp = source.mlp
        self.index = index
        self.time_steps = conversion.time_steps
        self.heads = int(config.num_attention_heads)
        self.kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(attention.head_dim)
        self.kv_groups = int(attention.num_key_value_groups)
        self.input_layernorm = _copy_norm(source.input_layernorm)
        self.post_attention_layernorm = _copy_norm(source.post_attention_layernorm)
        self.q_proj = _copy_linear(attention.q_proj)
        self.k_proj = _copy_linear(attention.k_proj)
        self.v_proj = _copy_linear(attention.v_proj)
        self.o_proj = _copy_linear(attention.o_proj)
        self.sdpa = TDScaledDotProductAttention(scale=float(attention.scaling))
        self.gate_proj = _copy_linear(mlp.gate_proj)
        self.up_proj = _copy_linear(mlp.up_proj)
        self.down_proj = _copy_linear(mlp.down_proj)
        self.act = TDSiLU()
        self.product = SNNElementWiseProduct()
        self.encoders = nn.ModuleDict(
            {
                name: SignedQCFSSequenceEncoder(
                    scale,
                    conversion.time_steps,
                    neuron_backend=conversion.neuron_backend,
                    name=f"layer.{index}.{name}",
                )
                for name, scale in scales.items()
            }
        )

    def _encode(
        self, name: str, sequence: torch.Tensor, mode: str, mask: torch.Tensor
    ) -> torch.Tensor:
        dense = sequence.sum(0)
        if mode == "exact_td":
            return _exact_sequence(dense, self.time_steps)
        if mode != "signed_if":
            raise ValueError(f"Unsupported encoding_mode={mode!r}.")
        return self.encoders[name].encode(dense, mask)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        value_mask: torch.Tensor,
        encoding_mode: str,
        cache: Optional[_Qwen2Cache],
    ) -> torch.Tensor:
        residual = hidden
        normalized = self.input_layernorm(hidden)
        time, batch, sequence, _ = normalized.shape
        query = (
            self.q_proj(normalized)
            .view(time, batch, sequence, self.heads, self.head_dim)
            .transpose(2, 3)
        )
        key = (
            self.k_proj(normalized)
            .view(time, batch, sequence, self.kv_heads, self.head_dim)
            .transpose(2, 3)
        )
        value = (
            self.v_proj(normalized)
            .view(time, batch, sequence, self.kv_heads, self.head_dim)
            .transpose(2, 3)
        )
        query, key = _rotary(query, key, cos, sin)
        query = (
            self._encode(
                "query",
                query.transpose(2, 3).reshape(time, batch, sequence, -1),
                encoding_mode,
                value_mask,
            )
            .view(time, batch, sequence, self.heads, self.head_dim)
            .transpose(2, 3)
        )
        key = (
            self._encode(
                "key",
                key.transpose(2, 3).reshape(time, batch, sequence, -1),
                encoding_mode,
                value_mask,
            )
            .view(time, batch, sequence, self.kv_heads, self.head_dim)
            .transpose(2, 3)
        )
        value = (
            self._encode(
                "value",
                value.transpose(2, 3).reshape(time, batch, sequence, -1),
                encoding_mode,
                value_mask,
            )
            .view(time, batch, sequence, self.kv_heads, self.head_dim)
            .transpose(2, 3)
        )
        past_key = past_value = None
        if cache is not None:
            past_key, past_value = cache.get(self.index)
            cache.append(self.index, key.sum(0), value.sum(0))
        query_dtype = query.dtype
        with torch.amp.autocast(query.device.type, enabled=False):
            query = query.float()
            key = _repeat_kv(key.float(), self.kv_groups)
            value = _repeat_kv(value.float(), self.kv_groups)
            if past_key is None:
                attended = self.sdpa(query, key, value, attention_mask)
            else:
                past_key = past_key.float().repeat_interleave(self.kv_groups, dim=1)
                past_value = past_value.float().repeat_interleave(self.kv_groups, dim=1)
                zero_key = torch.zeros_like(past_key)
                zero_value = torch.zeros_like(past_value)
                attended = torch.stack(
                    [
                        self.sdpa.single_step_forward(
                            query[step],
                            torch.cat(
                                (past_key if step == 0 else zero_key, key[step]), dim=2
                            ),
                            torch.cat(
                                (past_value if step == 0 else zero_value, value[step]),
                                dim=2,
                            ),
                            attention_mask,
                        )
                        for step in range(time)
                    ]
                )
        attended = (
            attended.to(query_dtype).transpose(2, 3).reshape(time, batch, sequence, -1)
        )
        hidden = residual + self.o_proj(attended)
        residual = hidden
        normalized = self.post_attention_layernorm(hidden)
        intermediate = self.product(
            self.act(self.gate_proj(normalized)), self.up_proj(normalized)
        )
        intermediate = self._encode("mlp", intermediate, encoding_mode, value_mask)
        return residual + self.down_proj(intermediate)


class Qwen2SNNModel(nn.Module):
    temporal_layout = "[T,B,S,H]"
    execution_schedule = "layerwise_offline_multistep"
    online_inference = False

    def __init__(
        self,
        source: nn.Module,
        calibration: Qwen2SNNCalibration,
        conversion: Qwen2SNNConfig,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <Qwen2SNNModel-cn>` | :ref:`English <Qwen2SNNModel-en>`

        ----

        .. _Qwen2SNNModel-cn:

        * **中文**

        Qwen2 的只读离线多步 SNN 推理模型。通常应通过
        :class:`Qwen2SNNRecipe` 和 :class:`ModuleConverter` 创建。模型冻结全部
        参数；``signed_if`` 路径在每个转换阶段显式构造 ``[T,B,S,H]`` 时间序列，
        ``exact_td`` 仅供 TD 等价性验证。调用 :meth:`forward` 时必须处于
        :func:`torch.inference_mode` 或 no-grad 上下文。

        :param source: 已进入 evaluation mode 的 Hugging Face Qwen2 causal LM。
        :type source: torch.nn.Module
        :param calibration: 与 source 结构匹配的校准尺度。
        :type calibration: Qwen2SNNCalibration
        :param conversion: 转换配置。
        :type conversion: Qwen2SNNConfig

        ----

        .. _Qwen2SNNModel-en:

        * **English**

        Read-only, offline multi-step SNN inference model for Qwen2. Normally create
        it through :class:`Qwen2SNNRecipe` and :class:`ModuleConverter`. All
        parameters are frozen. The ``signed_if`` path constructs an explicit
        ``[T,B,S,H]`` sequence at every converted stage; ``exact_td`` is only for TD
        equivalence checks. Call :meth:`forward` under :func:`torch.inference_mode`
        or another no-grad context.

        :param source: Evaluation-mode Hugging Face Qwen2 causal LM.
        :type source: torch.nn.Module
        :param calibration: Calibration scales matching the source structure.
        :type calibration: Qwen2SNNCalibration
        :param conversion: Conversion configuration.
        :type conversion: Qwen2SNNConfig
        """
        super().__init__()
        inner = source.model
        self.config = source.config
        self.time_steps = conversion.time_steps
        self.embed_tokens = _copy_embedding(inner.embed_tokens)
        self.rotary_emb = inner.rotary_emb
        self.layers = nn.ModuleList(
            [
                _Qwen2Decoder(
                    layer,
                    self.config,
                    calibration.layer_scales[index],
                    conversion,
                    index,
                )
                for index, layer in enumerate(inner.layers)
            ]
        )
        self.norm = _copy_norm(inner.norm)
        self.lm_head = _copy_linear(source.lm_head)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if bool(getattr(self.config, "tie_word_embeddings", False)):
            self.lm_head.weight = self.embed_tokens.weight
        self.input_encoder = SignedQCFSSequenceEncoder(
            calibration.input_scale,
            conversion.time_steps,
            neuron_backend=conversion.neuron_backend,
            name="model.input",
        )

    def signed_encoders(self) -> Tuple[SignedQCFSSequenceEncoder, ...]:
        r"""
        **API Language** - 中文 | English

        **中文：** 按执行顺序返回输入及各 decoder 的 signed QCFS encoder。

        :return: encoder 元组。
        :rtype: Tuple[SignedQCFSSequenceEncoder, ...]

        **English:** Return input and decoder signed QCFS encoders in execution order.

        :return: Tuple of encoders.
        :rtype: Tuple[SignedQCFSSequenceEncoder, ...]
        """
        result = [self.input_encoder]
        for layer in self.layers:
            result.extend(layer.encoders.values())
        return tuple(result)

    def set_collect_statistics(self, enabled: bool) -> None:
        r"""
        **API Language** - 中文 | English

        **中文：** 为全部 signed encoder 开启或关闭推理统计。关闭统计不改变脉冲
        或膜电位语义。

        :param enabled: 是否采集统计。
        :type enabled: bool

        **English:** Enable or disable inference statistics for every signed encoder.
        Disabling statistics does not alter spike or membrane semantics.

        :param enabled: Whether to collect statistics.
        :type enabled: bool
        """
        for encoder in self.signed_encoders():
            encoder.collect_statistics = bool(enabled)

    def encoder_statistics(self) -> Tuple[Dict[str, object], ...]:
        r"""
        **API Language** - 中文 | English

        **中文：** 返回全部 signed encoder 最近一次执行的统计。

        :return: 与 :meth:`signed_encoders` 顺序一致的统计元组。
        :rtype: Tuple[Dict[str, object], ...]

        **English:** Return statistics from the most recent execution of every signed
        encoder.

        :return: Statistics ordered like :meth:`signed_encoders`.
        :rtype: Tuple[Dict[str, object], ...]
        """
        return tuple(encoder.statistics() for encoder in self.signed_encoders())

    def structure_summary(self) -> Dict[str, object]:
        r"""
        **API Language** - 中文 | English

        **中文：** 返回 decoder、TD operator、signed encoder、神经元 backend 和
        tied embedding 的结构计数。

        :return: 机器可读的转换结构摘要。
        :rtype: Dict[str, object]

        **English:** Return decoder, TD-operator, signed-encoder, neuron-backend, and
        tied-embedding structure counts.

        :return: Machine-readable converted-structure summary.
        :rtype: Dict[str, object]
        """
        modules = tuple(self.modules())
        encoders = self.signed_encoders()
        return {
            "converted_decoder_count": len(self.layers),
            "td_rms_norm_count": sum(isinstance(value, TDRMSNorm) for value in modules),
            "td_linear_count": sum(isinstance(value, TDLinear) for value in modules),
            "td_silu_count": sum(isinstance(value, TDSiLU) for value in modules),
            "td_elementwise_product_count": sum(
                isinstance(value, SNNElementWiseProduct) for value in modules
            ),
            "td_sdpa_count": sum(
                isinstance(value, TDScaledDotProductAttention) for value in modules
            ),
            "signed_if_encoder_count": len(encoders),
            "activation_aware_if_node_count": 2 * len(encoders),
            "activation_aware_if_backends": sorted(
                {encoder.positive_neuron.backend for encoder in encoders}
            ),
            "activation_aware_if_step_modes": sorted(
                {encoder.positive_neuron.step_mode for encoder in encoders}
            ),
            "lm_head_tied_to_embedding": self.lm_head.weight
            is self.embed_tokens.weight,
        }

    @property
    def device(self) -> torch.device:
        r"""
        **API Language** - 中文 | English

        **中文：** 返回 input embedding 参数所在设备。

        :return: 模型设备。
        :rtype: torch.device

        **English:** Return the device of the input-embedding parameter.

        :return: Model device.
        :rtype: torch.device
        """
        return self.embed_tokens.weight.device

    def get_input_embeddings(self) -> nn.Embedding:
        r"""
        **API Language** - 中文 | English

        **中文：** 返回 Hugging Face 兼容的 token embedding 模块。

        :return: 输入 embedding。
        :rtype: torch.nn.Embedding

        **English:** Return the Hugging Face-compatible token embedding module.

        :return: Input embedding.
        :rtype: torch.nn.Embedding
        """
        return self.embed_tokens

    def get_output_embeddings(self) -> TDLinear:
        r"""
        **API Language** - 中文 | English

        **中文：** 返回 Hugging Face 兼容的 tied 或 untied TDLinear LM head。

        :return: 输出 LM head。
        :rtype: TDLinear

        **English:** Return the Hugging Face-compatible tied or untied TDLinear LM
        head.

        :return: Output LM head.
        :rtype: TDLinear
        """
        return self.lm_head

    def tie_weights(self) -> None:
        r"""
        **API Language** - 中文 | English

        **中文：** 当 Qwen2 config 要求 tied embeddings 时，将 LM head weight
        重新绑定到 input embedding。该方法兼容 Hugging Face evaluator。

        **English:** Rebind the LM-head weight to the input embedding when requested
        by the Qwen2 configuration. This method provides Hugging Face evaluator
        compatibility.
        """
        if bool(getattr(self.config, "tie_word_embeddings", False)):
            self.lm_head.weight = self.embed_tokens.weight

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        num_beams: int = 1,
        **_: object,
    ) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <Qwen2SNNModel.generate-cn>` | :ref:`English <Qwen2SNNModel.generate-en>`

        ----

        .. _Qwen2SNNModel.generate-cn:

        * **中文**

        使用 KV cache 执行确定性 greedy 解码。每个 token chunk 前重置神经元状态，
        但保留累计 KV cache；方法内部启用 :func:`torch.inference_mode`。

        :param input_ids: 输入 token，形状 ``[B,S]``。
        :type input_ids: torch.Tensor
        :param attention_mask: 可选的 ``[B,S]`` attention mask。
        :type attention_mask: Optional[torch.Tensor]
        :param max_new_tokens: 非负的新 token 数。
        :type max_new_tokens: int
        :param do_sample: 必须为 ``False``。
        :type do_sample: bool
        :param num_beams: 必须为 ``1``。
        :type num_beams: int
        :return: 形状 ``[B,S+max_new_tokens]`` 的 token IDs。
        :rtype: torch.Tensor
        :raises ValueError: 请求 sampling、beam search 或负 token 数。

        ----

        .. _Qwen2SNNModel.generate-en:

        * **English**

        Perform deterministic greedy decoding with a KV cache. Neuron state is reset
        before each token chunk while the accumulated KV cache is retained. The method
        enables :func:`torch.inference_mode` internally.

        :param input_ids: Input tokens with shape ``[B,S]``.
        :type input_ids: torch.Tensor
        :param attention_mask: Optional ``[B,S]`` attention mask.
        :type attention_mask: Optional[torch.Tensor]
        :param max_new_tokens: Non-negative number of new tokens.
        :type max_new_tokens: int
        :param do_sample: Must be ``False``.
        :type do_sample: bool
        :param num_beams: Must be ``1``.
        :type num_beams: int
        :return: Token IDs with shape ``[B,S+max_new_tokens]``.
        :rtype: torch.Tensor
        :raises ValueError: If sampling, beam search, or a negative token count is
            requested.
        """
        if do_sample or num_beams != 1:
            raise ValueError(
                "Converted Qwen2 supports deterministic greedy generation only."
            )
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        with torch.inference_mode():
            generated = input_ids
            mask = (
                torch.ones_like(input_ids) if attention_mask is None else attention_mask
            )
            cache = None
            for step in range(max_new_tokens):
                current = generated if step == 0 else generated[:, -1:]
                for module in self.modules():
                    reset = getattr(module, "reset", None)
                    if callable(reset):
                        reset()
                output = self(
                    input_ids=current,
                    attention_mask=mask,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
                token = output.logits[:, -1].argmax(-1, keepdim=True)
                generated = torch.cat((generated, token), dim=1)
                mask = torch.cat((mask, torch.ones_like(token)), dim=1)
            return generated

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        position_ids: Optional[torch.Tensor] = None,
        encoding_mode: Optional[str] = None,
        past_key_values: Optional[_Qwen2Cache] = None,
        use_cache: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        r"""
        **API Language** - :ref:`中文 <Qwen2SNNModel.forward-cn>` | :ref:`English <Qwen2SNNModel.forward-en>`

        ----

        .. _Qwen2SNNModel.forward-cn:

        * **中文**

        执行 Qwen2 离线多步推理。``signed_if`` 使用真实 activation-aware IF
        时间序列；``exact_td`` 仅用于数值 reference。返回对象包含 ``logits`` 和
        ``past_key_values``。

        :param input_ids: 输入 token，形状 ``[B,S]``。
        :type input_ids: torch.Tensor
        :param attention_mask: 可选 ``[B,past+S]`` mask；省略时视为全有效。
        :type attention_mask: Optional[torch.Tensor]
        :param position_ids: 可选 ``[B,S]`` RoPE 位置；省略时从 attention mask
            计算，以正确处理左 padding 和 cache continuation。
        :type position_ids: Optional[torch.Tensor]
        :param encoding_mode: ``"signed_if"``（默认）或 ``"exact_td"``。
        :type encoding_mode: Optional[str]
        :param past_key_values: 此模型上次返回的私有 cache 对象。
        :type past_key_values: Optional[_Qwen2Cache]
        :param use_cache: 是否返回并更新 KV cache。
        :type use_cache: bool
        :return: 含 ``logits`` 与 ``past_key_values`` 的 namespace。
        :rtype: types.SimpleNamespace
        :raises RuntimeError: 模型处于 training mode 或启用了 autograd。
        :raises ValueError: ``encoding_mode``、position shape 或 cache 参数组合无效。

        ----

        .. _Qwen2SNNModel.forward-en:

        * **English**

        Run offline multi-step Qwen2 inference. ``signed_if`` uses actual
        activation-aware IF temporal sequences; ``exact_td`` is only a numerical
        reference. The result contains ``logits`` and ``past_key_values``.

        :param input_ids: Input tokens with shape ``[B,S]``.
        :type input_ids: torch.Tensor
        :param attention_mask: Optional ``[B,past+S]`` mask; omitted means all valid.
        :type attention_mask: Optional[torch.Tensor]
        :param position_ids: Optional ``[B,S]`` RoPE positions. When omitted they
            are derived from the attention mask for left padding and cache
            continuation.
        :type position_ids: Optional[torch.Tensor]
        :param encoding_mode: ``"signed_if"`` (default) or ``"exact_td"``.
        :type encoding_mode: Optional[str]
        :param past_key_values: Private cache object returned by a previous call.
        :type past_key_values: Optional[_Qwen2Cache]
        :param use_cache: Whether to return and update the KV cache.
        :type use_cache: bool
        :return: Namespace containing ``logits`` and ``past_key_values``.
        :rtype: types.SimpleNamespace
        :raises RuntimeError: If the model is training or autograd is enabled.
        :raises ValueError: If ``encoding_mode``, the position shape, or the cache
            argument combination is invalid.
        """
        if self.training:
            raise RuntimeError("Converted Qwen2 SNN is inference-only; call eval().")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Converted Qwen2 SNN does not support autograd; use inference_mode()."
            )
        encoding_mode = encoding_mode or "signed_if"
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True.")
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        if use_cache and past_key_values is None:
            past_key_values = _Qwen2Cache(len(self.layers))
        embeddings = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = attention_mask.to(torch.long).cumsum(-1) - 1
            position_ids.masked_fill_(~attention_mask.to(torch.bool), 0)
            position_ids = position_ids[:, -input_ids.shape[1] :]
        elif tuple(position_ids.shape) != tuple(input_ids.shape):
            raise ValueError("position_ids shape must match input_ids shape.")
        position_ids = position_ids.to(device=input_ids.device, dtype=torch.long)
        cos, sin = self.rotary_emb(embeddings, position_ids)
        causal = _causal_mask(attention_mask, input_ids.shape[1], past_length)
        value_mask = attention_mask[:, -input_ids.shape[1] :]
        if encoding_mode == "exact_td":
            hidden = _exact_sequence(embeddings, self.time_steps)
        elif encoding_mode == "signed_if":
            hidden = self.input_encoder.encode(embeddings, value_mask)
        else:
            raise ValueError(f"Unsupported encoding_mode={encoding_mode!r}.")
        for layer in self.layers:
            hidden = layer(
                hidden,
                cos=cos,
                sin=sin,
                attention_mask=causal,
                value_mask=value_mask,
                encoding_mode=encoding_mode,
                cache=past_key_values if use_cache else None,
            )
        hidden = self.norm(hidden)
        with torch.amp.autocast(hidden.device.type, enabled=False):
            logits = F.linear(
                hidden.sum(0).float(),
                self.lm_head.weight.float(),
                None if self.lm_head.bias is None else self.lm_head.bias.float(),
            )
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def calibrate_qwen2_snn(
    model: nn.Module,
    calibration_batches: Iterable[Mapping[str, torch.Tensor]],
    config: Qwen2SNNConfig,
) -> Qwen2SNNCalibration:
    r"""
    **API Language** - :ref:`中文 <calibrate_qwen2_snn-cn>` | :ref:`English <calibrate_qwen2_snn-en>`

    ----

    .. _calibrate_qwen2_snn-cn:

    * **中文**

    在不保存完整 activation 的情况下，通过 hooks 收集 Qwen2 input、Q/K/V 和
    MLP 中间激活并计算逐通道 signed QCFS scale。调用者负责 tokenizer、数据加载
    和 batch 设备放置。

    :param model: evaluation-mode Hugging Face Qwen2 causal LM。
    :type model: torch.nn.Module
    :param calibration_batches: ``input_ids`` 和 ``attention_mask`` 张量 mapping 的
        iterable。
    :type calibration_batches: Iterable[Mapping[str, torch.Tensor]]
    :param config: Qwen2 SNN 转换配置。
    :type config: Qwen2SNNConfig
    :return: 可复用的逐通道校准尺度。
    :rtype: Qwen2SNNCalibration
    :raises TypeError: 模型不是受支持的 Qwen2 causal LM，或 batch 缺少张量字段。
    :raises ValueError: 模型配置不受支持、模型不在 evaluation mode、校准数据为空
        或无法产生有限正 scale。
    :raises RuntimeError: hook 执行时没有活动的 attention mask。

    ----

    .. _calibrate_qwen2_snn-en:

    * **English**

    Collect Qwen2 input, Q/K/V, and MLP intermediate activations through hooks
    without retaining complete activations, then compute per-channel signed QCFS
    scales. The caller owns tokenization, data loading, and batch device placement.

    :param model: Evaluation-mode Hugging Face Qwen2 causal LM.
    :type model: torch.nn.Module
    :param calibration_batches: Iterable of ``input_ids`` and ``attention_mask`` mappings.
    :type calibration_batches: Iterable[Mapping[str, torch.Tensor]]
    :param config: Qwen2 SNN conversion configuration.
    :type config: Qwen2SNNConfig
    :return: Reusable calibration scales.
    :rtype: Qwen2SNNCalibration
    :raises TypeError: If the model is not a supported Qwen2 causal LM or a batch
        lacks tensor fields.
    :raises ValueError: If the model configuration is unsupported, the model is not
        in evaluation mode, calibration data is empty, or no finite positive scale
        can be produced.
    :raises RuntimeError: If a hook executes without an active attention mask.
    """
    _validate_qwen2(model)
    layers = list(_qwen_layers(model))
    input_observer = _ChannelObserver(
        config.calibration_quantile,
        config.calibration_reservoir_size,
        config.calibration_seed,
    )
    layer_observers = [
        {
            name: _ChannelObserver(
                config.calibration_quantile,
                config.calibration_reservoir_size,
                config.calibration_seed,
            )
            for name in ("query", "key", "value", "mlp")
        }
        for _ in layers
    ]
    active_mask: Optional[torch.Tensor] = None
    handles = []

    def capture_embedding(_module, _inputs, output):
        if active_mask is None:
            raise RuntimeError("Calibration mask is not active.")
        input_observer.update(output, active_mask)

    handles.append(model.model.embed_tokens.register_forward_hook(capture_embedding))
    for index, layer in enumerate(layers):

        def capture_attention(module, inputs, kwargs, layer_index=index):
            if active_mask is None:
                raise RuntimeError("Calibration mask is not active.")
            hidden = inputs[0] if inputs else kwargs["hidden_states"]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(inputs) > 1:
                position_embeddings = inputs[1]
            batch, sequence, _ = hidden.shape
            head_dim = int(module.head_dim)
            heads = int(module.config.num_attention_heads)
            kv_heads = int(module.config.num_key_value_heads)
            query = (
                module.q_proj(hidden)
                .view(batch, sequence, heads, head_dim)
                .transpose(1, 2)
            )
            key = (
                module.k_proj(hidden)
                .view(batch, sequence, kv_heads, head_dim)
                .transpose(1, 2)
            )
            value = (
                module.v_proj(hidden)
                .view(batch, sequence, kv_heads, head_dim)
                .transpose(1, 2)
            )
            cos, sin = position_embeddings
            half = query.shape[-1] // 2
            rotate_query = torch.cat((-query[..., half:], query[..., :half]), -1)
            rotate_key = torch.cat((-key[..., half:], key[..., :half]), -1)
            query = query * cos.unsqueeze(1) + rotate_query * sin.unsqueeze(1)
            key = key * cos.unsqueeze(1) + rotate_key * sin.unsqueeze(1)
            observers = layer_observers[layer_index]
            observers["query"].update(
                query.transpose(1, 2).reshape(batch, sequence, -1), active_mask
            )
            observers["key"].update(
                key.transpose(1, 2).reshape(batch, sequence, -1), active_mask
            )
            observers["value"].update(
                value.transpose(1, 2).reshape(batch, sequence, -1), active_mask
            )

        def capture_mlp(module, inputs, layer_index=index):
            if active_mask is None:
                raise RuntimeError("Calibration mask is not active.")
            hidden = inputs[0]
            intermediate = module.act_fn(module.gate_proj(hidden)) * module.up_proj(
                hidden
            )
            layer_observers[layer_index]["mlp"].update(intermediate, active_mask)

        handles.append(
            layer.self_attn.register_forward_pre_hook(
                capture_attention, with_kwargs=True
            )
        )
        handles.append(layer.mlp.register_forward_pre_hook(capture_mlp))
    try:
        with torch.no_grad():
            for batch in calibration_batches:
                input_ids = batch.get("input_ids")
                mask = batch.get("attention_mask")
                if not isinstance(input_ids, torch.Tensor) or not isinstance(
                    mask, torch.Tensor
                ):
                    raise TypeError(
                        "Calibration batches require tensor input_ids and attention_mask."
                    )
                active_mask = mask
                model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    use_cache=False,
                    return_dict=True,
                )
    finally:
        for handle in handles:
            handle.remove()
    return Qwen2SNNCalibration(
        input_scale=input_observer.scale(config.calibration_levels),
        layer_scales=tuple(
            {
                name: observer.scale(config.calibration_levels)
                for name, observer in group.items()
            }
            for group in layer_observers
        ),
        time_steps=config.time_steps,
        calibration_levels=config.calibration_levels,
        calibration_quantile=config.calibration_quantile,
        calibration_reservoir_size=config.calibration_reservoir_size,
        calibration_seed=config.calibration_seed,
        valid_token_count=input_observer.valid_token_count,
    )


class Qwen2SNNRecipe(ModuleConversionRecipe):
    def __init__(
        self, calibration: Qwen2SNNCalibration, config: Qwen2SNNConfig
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <Qwen2SNNRecipe-cn>` | :ref:`English <Qwen2SNNRecipe-en>`

        ----

        .. _Qwen2SNNRecipe-cn:

        * **中文**

        将 evaluation-mode Hugging Face Qwen2 causal LM 转换为保留
        ``[T,B,S,H]`` 时间维的离线多步 TD/SNN hybrid。通过
        :class:`~spikingjelly.activation_based.ann2snn.ModuleConverter` 使用。

        :param calibration: 与模型层数和配置匹配的校准尺度。
        :type calibration: Qwen2SNNCalibration
        :param config: 转换配置。
        :type config: Qwen2SNNConfig

        ----

        .. _Qwen2SNNRecipe-en:

        * **English**

        Convert an evaluation-mode Hugging Face Qwen2 causal LM into an offline
        multi-step TD/SNN hybrid retaining a ``[T,B,S,H]`` temporal layout. Use
        this recipe through
        :class:`~spikingjelly.activation_based.ann2snn.ModuleConverter`.

        :param calibration: Calibration scales matching the model structure.
        :type calibration: Qwen2SNNCalibration
        :param config: Conversion configuration.
        :type config: Qwen2SNNConfig
        """
        self.calibration = calibration
        self.config = config

    def validate(self, converter: "ModuleConverter") -> None:
        r"""
        **API Language** - :ref:`中文 <Qwen2SNNRecipe.validate-cn>` | :ref:`English <Qwen2SNNRecipe.validate-en>`

        ----

        .. _Qwen2SNNRecipe.validate-cn:

        * **中文**

        校验校准产物与转换配置的时间步、量化等级、分位点和采样参数一致。

        :param converter: 执行此 recipe 的 module converter。
        :type converter: ModuleConverter
        :raises ValueError: 校准产物与配置不一致。

        ----

        .. _Qwen2SNNRecipe.validate-en:

        * **English**

        Validate that the calibration artifact matches the temporal, quantization,
        quantile, and sampling settings in the conversion configuration.

        :param converter: Module converter executing this recipe.
        :type converter: ModuleConverter
        :raises ValueError: If the calibration artifact and configuration differ.
        """
        del converter
        for name in (
            "time_steps",
            "calibration_levels",
            "calibration_quantile",
            "calibration_reservoir_size",
            "calibration_seed",
        ):
            if getattr(self.calibration, name) != getattr(self.config, name):
                raise ValueError(f"Calibration {name} does not match config.")

    def convert_module(
        self, converter: "ModuleConverter", ann: nn.Module
    ) -> Qwen2SNNModel:
        r"""
        **API Language** - :ref:`中文 <Qwen2SNNRecipe.convert_module-cn>` | :ref:`English <Qwen2SNNRecipe.convert_module-en>`

        ----

        .. _Qwen2SNNRecipe.convert_module-cn:

        * **中文**

        将受支持的 evaluation-mode Qwen2 causal LM 转换为只读离线多步 SNN。

        :param converter: 执行此 recipe 的 module converter。
        :type converter: ModuleConverter
        :param ann: 待转换的 Hugging Face Qwen2 causal LM。
        :type ann: torch.nn.Module
        :return: evaluation mode 的转换模型。
        :rtype: Qwen2SNNModel
        :raises TypeError: ``ann`` 不是受支持的 Qwen2 causal LM。
        :raises ValueError: 模型结构、运行模式或校准层数不受支持。

        ----

        .. _Qwen2SNNRecipe.convert_module-en:

        * **English**

        Convert a supported evaluation-mode Qwen2 causal LM into a read-only,
        offline multi-step SNN.

        :param converter: Module converter executing this recipe.
        :type converter: ModuleConverter
        :param ann: Hugging Face Qwen2 causal LM to convert.
        :type ann: torch.nn.Module
        :return: Converted model in evaluation mode.
        :rtype: Qwen2SNNModel
        :raises TypeError: If ``ann`` is not a supported Qwen2 causal LM.
        :raises ValueError: If the model structure, mode, or calibration layer count
            is unsupported.
        """
        del converter
        _validate_qwen2(ann)
        if len(_qwen_layers(ann)) != len(self.calibration.layer_scales):
            raise ValueError("Calibration layer count does not match Qwen2 model.")
        return Qwen2SNNModel(ann, self.calibration, self.config).eval()
