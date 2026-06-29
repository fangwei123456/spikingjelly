from __future__ import annotations

import copy
import math
import operator
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fx
from tqdm import tqdm

from spikingjelly.activation_based.ann2snn.recipes.base import ConversionRecipe

if TYPE_CHECKING:
    from spikingjelly.activation_based.ann2snn.converter import Converter


__all__ = [
    "STATransformerRecipe",
]


def _clone_parameter(parameter: nn.Parameter) -> nn.Parameter:
    return nn.Parameter(
        parameter.detach().clone(),
        requires_grad=parameter.requires_grad,
    )


class _STAThresholdObserver:
    def __init__(
        self,
        time_steps: int,
        channel_dim: int,
        mode: str,
        momentum: float,
        eps: float,
    ) -> None:
        self.time_steps = time_steps
        self.channel_dim = channel_dim
        self.mode = mode
        self.momentum = momentum
        self.eps = eps
        self.threshold: Optional[torch.Tensor] = None
        self.num_batches_tracked = 0

    @staticmethod
    def _normalize_channel_dim(x: torch.Tensor, channel_dim: int) -> int:
        if x.dim() < 2:
            raise ValueError("STA calibration requires activations with rank >= 2.")
        if channel_dim < 0:
            channel_dim += x.dim()
        if channel_dim < 0 or channel_dim >= x.dim():
            raise ValueError("channel_dim is out of range.")
        return channel_dim

    def __call__(self, output: torch.Tensor) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError("STA observer output must be a tensor.")
        with torch.no_grad():
            threshold = self._compute_threshold(output.detach())
            if self.threshold is None:
                self.threshold = threshold
            else:
                self.threshold = (
                    (1 - self.momentum)
                    * self.threshold.to(device=threshold.device, dtype=threshold.dtype)
                    + self.momentum * threshold
                ).detach()
            self.num_batches_tracked += 1

    def _compute_threshold(self, x: torch.Tensor) -> torch.Tensor:
        channel_dim = self._normalize_channel_dim(x, self.channel_dim)
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(torch.float32)
        values = x.movedim(channel_dim, 0).reshape(x.shape[channel_dim], -1)
        max_abs = values.abs().max(dim=1).values
        max_abs = torch.clamp(max_abs, min=self.eps)
        if self.mode == "max":
            return (max_abs / self.time_steps).detach()
        if self.mode != "mse":
            raise ValueError("STA threshold mode must be 'mse' or 'max'.")

        best_threshold = max_abs
        best_error = torch.full_like(max_abs, float("inf"))
        for factor in torch.linspace(
            0.125,
            2.0,
            steps=31,
        ).tolist():
            threshold = torch.clamp(max_abs * factor, min=self.eps)
            step_threshold = threshold[:, None] / self.time_steps
            quantized = torch.trunc(values / step_threshold) * step_threshold
            error = (values - quantized).square().mean(dim=1)
            update = error < best_error
            best_threshold = torch.where(update, threshold, best_threshold)
            best_error = torch.where(update, error, best_error)
        return (best_threshold / self.time_steps).detach()

    def compute_threshold(self) -> torch.Tensor:
        if self.threshold is None:
            raise ValueError("No STA calibration activations have been recorded.")
        if not torch.isfinite(self.threshold).all() or (self.threshold <= 0).any():
            raise ValueError("STA thresholds must be finite positive values.")
        return self.threshold.detach()


class _STATimeStepMixin:
    def _reset_sta_state(self) -> None:
        self.t = 0


class _STASpikeMixin(_STATimeStepMixin):
    def _reset_sta_state(self) -> None:
        super()._reset_sta_state()
        self.mem = None

    @staticmethod
    def _broadcast_threshold(
        threshold: torch.Tensor, output: torch.Tensor, channel_dim: int
    ) -> torch.Tensor:
        if channel_dim < 0:
            channel_dim += output.dim()
        shape = [1] * output.dim()
        shape[channel_dim] = threshold.numel()
        return threshold.to(device=output.device, dtype=output.dtype).view(shape)

    def _spike(self, analog: torch.Tensor, channel_dim: int) -> torch.Tensor:
        threshold = self._broadcast_threshold(self.v_threshold, analog, channel_dim)
        if (
            self.mem is None
            or self.mem.shape != analog.shape
            or self.mem.device != analog.device
            or self.mem.dtype != analog.dtype
        ):
            self.mem = torch.zeros_like(analog)
        self.mem = self.mem + analog
        spike_count = torch.trunc(self.mem / threshold)
        spike = spike_count * threshold
        self.mem = self.mem - spike
        return spike


class _STASpikeLinear(nn.Module, _STASpikeMixin):
    def __init__(self, source: nn.Linear, threshold: torch.Tensor) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = _clone_parameter(source.weight)
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = _clone_parameter(source.bias)
        self.register_buffer("v_threshold", threshold.detach().clone())
        self.mem: Optional[torch.Tensor] = None
        self.t = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.t == 0 else None
        self.t += 1
        return self._spike(F.linear(x, self.weight, bias), channel_dim=-1)


class _STASpikeConv2d(nn.Module, _STASpikeMixin):
    def __init__(self, source: nn.Conv2d, threshold: torch.Tensor) -> None:
        super().__init__()
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.padding_mode = source.padding_mode
        self._sta_reversed_padding_repeated_twice = tuple(
            source._reversed_padding_repeated_twice
        )
        self.weight = _clone_parameter(source.weight)
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = _clone_parameter(source.bias)
        self.register_buffer("v_threshold", threshold.detach().clone())
        self.mem: Optional[torch.Tensor] = None
        self.t = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.t == 0 else None
        self.t += 1
        if self.padding_mode != "zeros":
            x = F.pad(
                x,
                self._sta_reversed_padding_repeated_twice,
                mode=self.padding_mode,
            )
            analog = F.conv2d(
                x,
                self.weight,
                bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        else:
            analog = F.conv2d(
                x,
                self.weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return self._spike(analog, channel_dim=1)


class _STAActivationSpikeEncoder(nn.Module, _STASpikeMixin):
    def __init__(self, threshold: torch.Tensor, channel_dim: int = -1) -> None:
        super().__init__()
        self.channel_dim = channel_dim
        self.register_buffer("v_threshold", threshold.detach().clone())
        self.mem: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._spike(x, channel_dim=self.channel_dim)


class _STAAnalogLinear(nn.Module, _STATimeStepMixin):
    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = _clone_parameter(source.weight)
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = _clone_parameter(source.bias)
        self.t = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.t == 0 else None
        self.t += 1
        return F.linear(x, self.weight, bias)


class _STAAnalogConv2d(nn.Module, _STATimeStepMixin):
    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__()
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.padding_mode = source.padding_mode
        self._sta_reversed_padding_repeated_twice = tuple(
            source._reversed_padding_repeated_twice
        )
        self.weight = _clone_parameter(source.weight)
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = _clone_parameter(source.bias)
        self.t = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.t == 0 else None
        self.t += 1
        if self.padding_mode != "zeros":
            x = F.pad(
                x,
                self._sta_reversed_padding_repeated_twice,
                mode=self.padding_mode,
            )
            return F.conv2d(
                x,
                self.weight,
                bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            x,
            self.weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _STAOnlineLayerNorm(nn.Module):
    def __init__(
        self,
        source: nn.LayerNorm,
        encoder_threshold: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = source.normalized_shape
        self.eps = source.eps
        self.elementwise_affine = source.elementwise_affine
        if source.weight is None:
            self.register_parameter("weight", None)
        else:
            self.weight = _clone_parameter(source.weight)
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = _clone_parameter(source.bias)
        self.mem: Optional[torch.Tensor] = None
        self.prev_output: Optional[torch.Tensor] = None
        self.encoder = (
            None
            if encoder_threshold is None
            else _STAActivationSpikeEncoder(encoder_threshold, channel_dim=-1)
        )

    def _reset_sta_state(self) -> None:
        self.mem = None
        self.prev_output = None
        if self.encoder is not None:
            self.encoder._reset_sta_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.mem is None
            or self.mem.shape != x.shape
            or self.mem.device != x.device
            or self.mem.dtype != x.dtype
        ):
            self.mem = torch.zeros_like(x)
            self.prev_output = torch.zeros_like(x)
        self.mem = self.mem + x
        output = F.layer_norm(
            self.mem,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
        delta = output - self.prev_output
        self.prev_output = output
        if self.encoder is not None:
            delta = self.encoder(delta)
        return delta


class _STAOnlineGELU(nn.Module):
    def __init__(
        self,
        source: nn.GELU,
        encoder_threshold: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.approximate = getattr(source, "approximate", "none")
        self.mem: Optional[torch.Tensor] = None
        self.prev_output: Optional[torch.Tensor] = None
        self.encoder = (
            None
            if encoder_threshold is None
            else _STAActivationSpikeEncoder(encoder_threshold, channel_dim=-1)
        )

    def _reset_sta_state(self) -> None:
        self.mem = None
        self.prev_output = None
        if self.encoder is not None:
            self.encoder._reset_sta_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.mem is None
            or self.mem.shape != x.shape
            or self.mem.device != x.device
            or self.mem.dtype != x.dtype
        ):
            self.mem = torch.zeros_like(x)
            self.prev_output = torch.zeros_like(x)
        self.mem = self.mem + x
        output = F.gelu(self.mem, approximate=self.approximate)
        delta = output - self.prev_output
        self.prev_output = output
        if self.encoder is not None:
            delta = self.encoder(delta)
        return delta


class _STAOnlineMultiheadAttention(nn.Module):
    def __init__(
        self,
        source: nn.MultiheadAttention,
        encoder_threshold: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        # MultiheadAttention carries fused projection parameters and option
        # fields; deepcopy preserves that full internal layout.
        self.attn = copy.deepcopy(source)
        self.q_mem: Optional[torch.Tensor] = None
        self.k_mem: Optional[torch.Tensor] = None
        self.v_mem: Optional[torch.Tensor] = None
        self.prev_output: Optional[torch.Tensor] = None
        self.prev_weights: Optional[torch.Tensor] = None
        self.encoder = (
            None
            if encoder_threshold is None
            else _STAActivationSpikeEncoder(encoder_threshold, channel_dim=-1)
        )

    def _reset_sta_state(self) -> None:
        self.q_mem = None
        self.k_mem = None
        self.v_mem = None
        self.prev_output = None
        self.prev_weights = None
        if self.encoder is not None:
            self.encoder._reset_sta_state()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (
            self.q_mem is None
            or self.k_mem is None
            or self.v_mem is None
            or self.q_mem.shape != query.shape
            or self.k_mem.shape != key.shape
            or self.v_mem.shape != value.shape
            or self.q_mem.device != query.device
            or self.k_mem.device != key.device
            or self.v_mem.device != value.device
            or self.q_mem.dtype != query.dtype
            or self.k_mem.dtype != key.dtype
            or self.v_mem.dtype != value.dtype
        ):
            self.q_mem = torch.zeros_like(query)
            self.k_mem = torch.zeros_like(key)
            self.v_mem = torch.zeros_like(value)
            self.prev_output = None
            self.prev_weights = None
        self.q_mem = self.q_mem + query
        self.k_mem = self.k_mem + key
        self.v_mem = self.v_mem + value
        output, weights = self.attn(
            self.q_mem,
            self.k_mem,
            self.v_mem,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if (
            self.prev_output is None
            or self.prev_output.shape != output.shape
            or self.prev_output.device != output.device
            or self.prev_output.dtype != output.dtype
        ):
            self.prev_output = torch.zeros_like(output)
        delta = output - self.prev_output
        self.prev_output = output
        if self.encoder is not None:
            delta = self.encoder(delta)
        if weights is None:
            weights_delta = None
            self.prev_weights = None
        else:
            if (
                self.prev_weights is None
                or self.prev_weights.shape != weights.shape
                or self.prev_weights.device != weights.device
                or self.prev_weights.dtype != weights.dtype
            ):
                self.prev_weights = torch.zeros_like(weights)
            weights_delta = weights - self.prev_weights
            self.prev_weights = weights
        return delta, weights_delta


class _STAConvertedModel(nn.Module):
    _STATIC_TENSOR_KWARGS = frozenset(
        {
            "attention_mask",
            "attn_mask",
            "causal_mask",
            "decoder_attention_mask",
            "encoder_attention_mask",
            "key_padding_mask",
            "mask",
            "padding_mask",
        }
    )

    def __init__(
        self,
        model: nn.Module,
        time_steps: int,
        static_arg_indices: Optional[frozenset[int]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.time_steps = time_steps
        self.static_arg_indices = static_arg_indices or frozenset()

    def reset(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            reset = getattr(module, "_reset_sta_state", None)
            if reset is not None:
                reset()

    @staticmethod
    def _rebuild_container(source: Any, items: Any) -> Any:
        if isinstance(source, tuple) and hasattr(source, "_fields"):
            return type(source)(*items)
        if type(source) is tuple:
            return tuple(items)
        if type(source) is list:
            return list(items)
        if type(source) is dict:
            return dict(items)
        return type(source)(items)

    @staticmethod
    def _add_outputs(a: Any, b: Any) -> Any:
        if torch.is_tensor(a) and torch.is_tensor(b):
            return a + b
        if isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) != len(b):
                raise TypeError("STA converted model output tuple lengths must match.")
            return _STAConvertedModel._rebuild_container(
                a,
                (_STAConvertedModel._add_outputs(x, y) for x, y in zip(a, b)),
            )
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise TypeError("STA converted model output list lengths must match.")
            return _STAConvertedModel._rebuild_container(
                a,
                (_STAConvertedModel._add_outputs(x, y) for x, y in zip(a, b)),
            )
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                raise TypeError("STA converted model output dict keys must match.")
            return _STAConvertedModel._rebuild_container(
                a,
                {key: _STAConvertedModel._add_outputs(a[key], b[key]) for key in a},
            )
        if a is None and b is None:
            return None
        raise TypeError("STA converted model can only accumulate tensor outputs.")

    @staticmethod
    def _zero_tensors(
        value: Any,
        preserve_tensor: bool = False,
        static_arg_indices: Optional[frozenset[int]] = None,
    ) -> Any:
        if torch.is_tensor(value):
            if preserve_tensor:
                return value
            if not value.is_floating_point():
                raise TypeError(
                    "STA converted model cannot create zero-input timesteps for "
                    "non-floating data tensors. Pass control masks as named static "
                    "kwargs, or convert data inputs to floating tensors before "
                    "conversion."
                )
            return torch.zeros_like(value)
        if isinstance(value, tuple):
            return _STAConvertedModel._rebuild_container(
                value,
                (
                    _STAConvertedModel._zero_tensors(
                        x,
                        preserve_tensor=preserve_tensor
                        or (static_arg_indices is not None and i in static_arg_indices),
                    )
                    for i, x in enumerate(value)
                ),
            )
        if isinstance(value, list):
            return _STAConvertedModel._rebuild_container(
                value,
                (
                    _STAConvertedModel._zero_tensors(
                        x,
                        preserve_tensor=preserve_tensor
                        or (static_arg_indices is not None and i in static_arg_indices),
                    )
                    for i, x in enumerate(value)
                ),
            )
        if isinstance(value, dict):
            return _STAConvertedModel._rebuild_container(
                value,
                {
                    key: _STAConvertedModel._zero_tensors(
                        x,
                        preserve_tensor=preserve_tensor
                        or key in _STAConvertedModel._STATIC_TENSOR_KWARGS,
                    )
                    for key, x in value.items()
                },
            )
        return value

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.reset()
        output = None
        zero_args = self._zero_tensors(
            args,
            static_arg_indices=self.static_arg_indices,
        )
        zero_kwargs = self._zero_tensors(kwargs)
        for step in range(self.time_steps):
            step_args = args if step == 0 else zero_args
            step_kwargs = kwargs if step == 0 else zero_kwargs
            step_output = self.model(*step_args, **step_kwargs)
            output = (
                step_output
                if output is None
                else self._add_outputs(output, step_output)
            )
        return output


class _STAFirstStepConstant(nn.Module, _STATimeStepMixin):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        if isinstance(value, nn.Parameter):
            self.value = nn.Parameter(
                value.detach().clone(), requires_grad=value.requires_grad
            )
        else:
            self.register_buffer("value", value.detach().clone())
        self._zero_value: Optional[torch.Tensor] = None
        self.t = 0

    def _reset_sta_state(self) -> None:
        self.t = 0

    def forward(self) -> torch.Tensor:
        if self.t == 0:
            output = self.value
        else:
            if (
                self._zero_value is None
                or self._zero_value.shape != self.value.shape
                or self._zero_value.device != self.value.device
                or self._zero_value.dtype != self.value.dtype
            ):
                self._zero_value = torch.zeros_like(self.value)
            output = self._zero_value
        self.t += 1
        return output


class STATransformerRecipe(ConversionRecipe):
    def __init__(
        self,
        dataloader: Optional[Iterable] = None,
        time_steps: int = 32,
        mode: str = "equivalent",
        threshold_mode: str = "mse",
        threshold_scale: float = 1.0,
        spike_linear: Optional[bool] = None,
        spike_conv2d: Optional[bool] = None,
        spike_classifier: bool = False,
        momentum: float = 0.1,
        num_calibration_batches: Optional[int] = None,
        show_progress: bool = False,
        eps: float = 1e-6,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <STATransformerRecipe.__init__-cn>` | :ref:`English <STATransformerRecipe.__init__-en>`

        ----

        .. _STATransformerRecipe.__init__-cn:

        * **中文**

        实现基于 Spatio-Temporal Approximation (STA) [#sta_transformer]_ 思路的
        training-free Transformer 转换 recipe。该 recipe 将普通 ANN 输入编码
        为第 0 步真实输入和后续 0 输入，并在模型内部执行 ``time_steps`` 次
        循环后返回累计输出。
        转换过程中会以 ``eval`` 模式 trace 和校准原 ANN；``Converter`` 会在
        转换结束后恢复原 ANN 的 ``training`` 标志。

        ``mode="equivalent"`` 是默认的在线累计-差分基线：Linear、Conv2d、
        LayerNorm、GELU、``MultiheadAttention`` 和 FX tensor 常量都按时间步
        保持与原 ANN 的累计输出等价。该模式用于建立 Transformer 图形态和
        模型级接受基线，不宣称 fully spike-driven。

        ``mode="spiking_encoder"`` 会在 LayerNorm、GELU 和
        ``MultiheadAttention`` 输出侧插入校准后的有状态 spike encoder，同时
        保持主干 affine 在线等价。``mode="spiking_affine"`` 会为选中的
        Linear/Conv2d 统计阈值并替换为有状态 bipolar IF / burst affine 模块；
        LayerNorm、GELU 和 ``MultiheadAttention`` 仍使用在线累计-差分浮点
        模块。它是 STA affine spiking 路径，不修改 rate-coding recipe，也
        不依赖 tutorial。``time_steps`` 参与阈值搜索和最终推理循环，因而是
        转换 recipe 的一部分，而不仅是外部评估循环参数。

        :param dataloader: 校准数据加载器。每个 batch 可为单输入 tensor、
            ``(input, target)`` 风格的 tuple/list，或传递给模型的 kwargs
            dict。``mode="equivalent"`` 默认不执行校准，可传 ``None``；显式
            启用 ``spike_linear`` 或 ``spike_conv2d`` 时也需要提供。
        :type dataloader: Optional[Iterable]
        :param time_steps: STA 内部推理时间步数，也用于阈值搜索。
        :type time_steps: int
        :param mode: 转换模式，支持 ``"equivalent"``、
            ``"spiking_encoder"`` 和 ``"spiking_affine"``。
        :type mode: str
        :param threshold_mode: 阈值统计模式，支持 ``"mse"`` 和 ``"max"``。
        :type threshold_mode: str
        :param threshold_scale: 校准阈值的正数缩放因子。
        :type threshold_scale: float
        :param spike_linear: 是否替换 Linear 为 spiking affine。若为 ``None``，
            在 ``mode="spiking_affine"`` 时启用。
        :type spike_linear: Optional[bool]
        :param spike_conv2d: 是否替换 Conv2d 为 spiking affine。若为 ``None``，
            默认不启用。
        :type spike_conv2d: Optional[bool]
        :param spike_classifier: ``spike_linear=True`` 时是否也转换分类头。
        :type spike_classifier: bool
        :param momentum: 阈值 observer 的动量。
        :type momentum: float
        :param num_calibration_batches: 最多使用的校准 batch 数；``None`` 表示
            使用整个 dataloader。
        :type num_calibration_batches: Optional[int]
        :param show_progress: 是否显示校准进度条。
        :type show_progress: bool
        :param eps: 阈值数值下界。
        :type eps: float
        :raises ValueError: 当校验发现不支持的转换模式、阈值模式、非正
            time step、非正缩放因子、非法动量、非法校准 batch 上限，或布尔
            选项类型错误时抛出。

        .. [#sta_transformer] Y. Jiang, K. Hu, T. Zhang, H. Gao, Y. Liu,
           Y. Fang, and F. Chen, "Spatio-Temporal Approximation: A
           Training-Free SNN Conversion for Transformers," ICLR 2024.
           https://openreview.net/forum?id=XrunSYwoLr

        ----

        .. _STATransformerRecipe.__init__-en:

        * **English**

        Implement a training-free Transformer conversion recipe based on
        Spatio-Temporal Approximation (STA) [#sta_transformer]_. The recipe
        encodes ordinary ANN
        inputs as one real input step followed by zero-input steps, runs
        ``time_steps`` internal iterations, and returns the cumulative output.
        Conversion traces and calibrates the original ANN in ``eval`` mode;
        ``Converter`` restores the original ANN ``training`` flags after
        conversion finishes.

        ``mode="equivalent"`` is the default online cumulative-difference
        baseline: Linear, Conv2d, LayerNorm, GELU, ``MultiheadAttention`` and
        FX tensor constants preserve the original ANN cumulative output across
        time. This mode establishes the Transformer graph shape and model-level
        acceptance baseline; it does not claim to be fully spike-driven.

        ``mode="spiking_encoder"`` inserts calibrated stateful spike encoders
        after LayerNorm, GELU and ``MultiheadAttention`` outputs while keeping
        main affine projections online-equivalent. ``mode="spiking_affine"``
        calibrates thresholds for selected Linear/Conv2d modules and replaces
        them with stateful bipolar IF / burst affine modules; LayerNorm, GELU
        and ``MultiheadAttention`` remain online cumulative-difference
        floating-point modules. It is the STA affine spiking path, separate
        from the rate-coding recipe and from tutorials. ``time_steps`` is used
        by threshold search and by the final inference loop, so it belongs to
        the conversion recipe rather than only to an external evaluation loop.

        :param dataloader: Calibration dataloader. Each batch can be a
            single-input tensor, a ``(input, target)``-style tuple/list, or a
            kwargs dict passed to the model. ``mode="equivalent"`` skips
            calibration by default and can use ``None``; explicitly enabling
            ``spike_linear`` or ``spike_conv2d`` still requires a dataloader.
        :type dataloader: Optional[Iterable]
        :param time_steps: Number of STA internal inference timesteps. It is
            also used by threshold search.
        :type time_steps: int
        :param mode: Conversion mode. Supported values are ``"equivalent"``,
            ``"spiking_encoder"`` and ``"spiking_affine"``.
        :type mode: str
        :param threshold_mode: Threshold statistics mode. Supported values are
            ``"mse"`` and ``"max"``.
        :type threshold_mode: str
        :param threshold_scale: Positive scale factor applied to calibrated
            thresholds.
        :type threshold_scale: float
        :param spike_linear: Whether to replace Linear with spiking affine
            modules. If ``None``, enable it for ``mode="spiking_affine"``.
        :type spike_linear: Optional[bool]
        :param spike_conv2d: Whether to replace Conv2d with spiking affine
            modules. If ``None``, disable it by default.
        :type spike_conv2d: Optional[bool]
        :param spike_classifier: Whether to convert classifier heads when
            ``spike_linear=True``.
        :type spike_classifier: bool
        :param momentum: Momentum used by threshold observers.
        :type momentum: float
        :param num_calibration_batches: Maximum number of calibration batches;
            ``None`` uses the full dataloader.
        :type num_calibration_batches: Optional[int]
        :param show_progress: Whether to show a calibration progress bar.
        :type show_progress: bool
        :param eps: Numeric lower bound for thresholds.
        :type eps: float
        :raises ValueError: If validation finds an unsupported mode, threshold
            mode, non-positive timestep count, non-positive scale, invalid
            momentum, invalid calibration batch limit, or invalid type for a
            boolean option.
        """
        self.dataloader = dataloader
        self.time_steps = time_steps
        self.mode = mode
        self.threshold_mode = (
            threshold_mode.lower()
            if isinstance(threshold_mode, str)
            else threshold_mode
        )
        self.threshold_scale = threshold_scale
        if spike_linear is None:
            spike_linear = mode == "spiking_affine"
        if spike_conv2d is None:
            spike_conv2d = False
        self.spike_linear = spike_linear
        self.spike_conv2d = spike_conv2d
        self.spike_classifier = spike_classifier
        self.momentum = momentum
        self.num_calibration_batches = num_calibration_batches
        self.show_progress = show_progress
        self.eps = eps
        self._observers: Dict[str, _STAThresholdObserver] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    def validate(self, converter: "Converter") -> None:
        needs_calibration = (
            self.mode == "spiking_encoder" or self.spike_linear or self.spike_conv2d
        )
        if needs_calibration and self.dataloader is None:
            raise ValueError(
                "STATransformerRecipe requires a dataloader when calibration "
                "is enabled. "
                "Pass dataloader to STATransformerRecipe."
            )
        if (
            not isinstance(self.time_steps, int)
            or isinstance(self.time_steps, bool)
            or self.time_steps <= 0
        ):
            raise ValueError("time_steps must be a positive integer.")
        if not isinstance(self.mode, str):
            raise ValueError("mode must be str.")
        if self.mode not in ("equivalent", "spiking_encoder", "spiking_affine"):
            raise ValueError(
                "mode must be 'equivalent', 'spiking_encoder' or 'spiking_affine'."
            )
        if not isinstance(self.threshold_mode, str):
            raise ValueError("threshold_mode must be str.")
        if self.threshold_mode not in ("mse", "max"):
            raise ValueError("threshold_mode must be 'mse' or 'max'.")
        if not isinstance(self.threshold_scale, (int, float)) or isinstance(
            self.threshold_scale, bool
        ):
            raise ValueError("threshold_scale must be a positive number.")
        if self.threshold_scale <= 0 or not math.isfinite(self.threshold_scale):
            raise ValueError("threshold_scale must be positive.")
        if not isinstance(self.spike_linear, bool):
            raise ValueError("spike_linear must be bool.")
        if not isinstance(self.spike_conv2d, bool):
            raise ValueError("spike_conv2d must be bool.")
        if not isinstance(self.spike_classifier, bool):
            raise ValueError("spike_classifier must be bool.")
        if not isinstance(self.momentum, (int, float)) or isinstance(
            self.momentum, bool
        ):
            raise ValueError("momentum must be a number in [0, 1].")
        if not (0.0 <= self.momentum <= 1.0):
            raise ValueError("momentum must be in [0, 1].")
        if self.num_calibration_batches is not None and (
            not isinstance(self.num_calibration_batches, int)
            or isinstance(self.num_calibration_batches, bool)
            or self.num_calibration_batches <= 0
        ):
            raise ValueError("num_calibration_batches must be positive or None.")
        if not isinstance(self.show_progress, bool):
            raise ValueError("show_progress must be bool.")
        if (
            not isinstance(self.eps, (int, float))
            or isinstance(self.eps, bool)
            or self.eps <= 0
            or not math.isfinite(self.eps)
        ):
            raise ValueError("eps must be a positive number.")

    def before_trace(self, converter: "Converter", ann: nn.Module) -> nn.Module:
        ann.eval()
        return ann

    def insert_observers(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        self._remove_hooks()
        self._observers = {}
        modules = dict(fx_model.named_modules())
        for node in fx_model.graph.nodes:
            if node.op != "call_module":
                continue
            if not isinstance(node.target, str):
                continue
            module = modules.get(node.target)
            if isinstance(module, nn.Linear):
                if not self._should_spike_linear(node.target):
                    continue
                observer = _STAThresholdObserver(
                    self.time_steps,
                    channel_dim=-1,
                    mode=self.threshold_mode,
                    momentum=self.momentum,
                    eps=self.eps,
                )
            elif isinstance(module, nn.Conv2d):
                if not self.spike_conv2d:
                    continue
                observer = _STAThresholdObserver(
                    self.time_steps,
                    channel_dim=1,
                    mode=self.threshold_mode,
                    momentum=self.momentum,
                    eps=self.eps,
                )
            elif self.mode == "spiking_encoder" and isinstance(
                module, (nn.LayerNorm, nn.GELU, nn.MultiheadAttention)
            ):
                observer = _STAThresholdObserver(
                    self.time_steps,
                    channel_dim=-1,
                    mode=self.threshold_mode,
                    momentum=self.momentum,
                    eps=self.eps,
                )
            else:
                continue
            self._observers[node.target] = observer
            handle = module.register_forward_hook(
                lambda _module, _inputs, output, obs=observer: obs(output)
            )
            self._hook_handles.append(handle)
        return fx_model

    def calibrate(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        if not self._observers:
            return fx_model
        fx_model.eval()
        iterator = self.dataloader
        if self.show_progress:
            iterator = tqdm(iterator, desc="STA calibration")
        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(iterator):
                    if (
                        self.num_calibration_batches is not None
                        and batch_index >= self.num_calibration_batches
                    ):
                        break
                    args, kwargs = self._batch_to_args(batch, converter.device)
                    fx_model(*args, **kwargs)
        finally:
            self._remove_hooks()
        return fx_model

    def replace(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        self._remove_hooks()
        modules = dict(fx_model.named_modules())
        for node in fx_model.graph.nodes:
            if node.op != "call_module" or not isinstance(node.target, str):
                continue
            module = modules.get(node.target)
            if isinstance(module, nn.Linear):
                observer = self._observers.get(node.target)
                threshold = self._compute_scaled_threshold(observer)
                if threshold is None:
                    replacement = _STAAnalogLinear(module)
                else:
                    replacement = _STASpikeLinear(module, threshold)
            elif isinstance(module, nn.Conv2d):
                observer = self._observers.get(node.target)
                threshold = self._compute_scaled_threshold(observer)
                if threshold is None:
                    replacement = _STAAnalogConv2d(module)
                else:
                    replacement = _STASpikeConv2d(module, threshold)
            else:
                continue
            replacement.train(module.training)
            self._replace_submodule(fx_model, node.target, replacement)
            modules[node.target] = replacement
        self._wrap_time_constants(fx_model)
        modules = dict(fx_model.named_modules())
        for node in fx_model.graph.nodes:
            if node.op != "call_module" or not isinstance(node.target, str):
                continue
            module = modules.get(node.target)
            if isinstance(module, nn.LayerNorm):
                observer = self._observers.get(node.target)
                threshold = self._compute_scaled_threshold(observer)
                replacement = _STAOnlineLayerNorm(module, threshold)
            elif isinstance(module, nn.GELU):
                observer = self._observers.get(node.target)
                threshold = self._compute_scaled_threshold(observer)
                replacement = _STAOnlineGELU(module, threshold)
            elif isinstance(module, nn.MultiheadAttention):
                observer = self._observers.get(node.target)
                threshold = self._compute_scaled_threshold(observer)
                replacement = _STAOnlineMultiheadAttention(module, threshold)
            else:
                continue
            replacement.train(module.training)
            self._replace_submodule(fx_model, node.target, replacement)
            modules[node.target] = replacement
        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()
        fx_model.recompile()
        return fx_model

    def finalize(self, converter: "Converter", fx_model: fx.GraphModule) -> nn.Module:
        converted = _STAConvertedModel(
            fx_model,
            time_steps=self.time_steps,
            static_arg_indices=self._find_static_placeholder_indices(fx_model),
        )
        converted.train(fx_model.training)
        return converted

    @staticmethod
    def _batch_to_args(
        batch: Any, device: torch.device
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        if isinstance(batch, dict):
            kwargs = {
                key: STATransformerRecipe._to_device(value, device)
                for key, value in batch.items()
            }
            return (), kwargs
        if isinstance(batch, (tuple, list)):
            if len(batch) == 0:
                raise ValueError("Calibration batch must not be empty.")
            if isinstance(batch[0], (tuple, list)):
                args = tuple(
                    STATransformerRecipe._to_device(value, device) for value in batch[0]
                )
                kwargs = (
                    STATransformerRecipe._to_device(batch[1], device)
                    if len(batch) > 1 and isinstance(batch[1], dict)
                    else {}
                )
                return args, kwargs
            return (STATransformerRecipe._to_device(batch[0], device),), {}
        return (STATransformerRecipe._to_device(batch, device),), {}

    @staticmethod
    def _to_device(value: Any, device: torch.device) -> Any:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(STATransformerRecipe._to_device(v, device) for v in value)
        if isinstance(value, list):
            return [STATransformerRecipe._to_device(v, device) for v in value]
        if isinstance(value, dict):
            return {
                key: STATransformerRecipe._to_device(v, device)
                for key, v in value.items()
            }
        return value

    @staticmethod
    def _replace_submodule(
        fx_model: torch.fx.GraphModule, target: str, module: nn.Module
    ) -> None:
        parent_name, _, child_name = target.rpartition(".")
        parent = fx_model.get_submodule(parent_name) if parent_name else fx_model
        setattr(parent, child_name, module)

    @staticmethod
    def _get_attr_value(fx_model: fx.GraphModule, target: str) -> Any:
        value: Any = fx_model
        for atom in target.split("."):
            value = getattr(value, atom)
        return value

    def _wrap_time_constants(self, fx_model: fx.GraphModule) -> None:
        modules = dict(fx_model.named_modules())
        existing_modules = set(modules.keys())
        constant_index = 0
        for node in list(fx_model.graph.nodes):
            if node.op != "get_attr" or not isinstance(node.target, str):
                continue
            value = self._get_attr_value(fx_model, node.target)
            if not torch.is_tensor(value):
                continue
            if not value.is_floating_point():
                continue
            if self._is_static_control_attr(node, modules):
                continue
            if self._is_functional_parameter_attr(node):
                continue
            target = f"sta_time_constant_{constant_index}"
            while target in existing_modules:
                constant_index += 1
                target = f"sta_time_constant_{constant_index}"
            constant_index += 1
            fx_model.add_submodule(target, _STAFirstStepConstant(value))
            existing_modules.add(target)
            with fx_model.graph.inserting_after(node):
                constant_node = fx_model.graph.call_module(target, args=())
            for user in list(node.users):
                user.replace_input_with(node, constant_node)
            fx_model.graph.erase_node(node)

    @staticmethod
    def _is_static_control_attr(node: fx.Node, modules: Dict[str, nn.Module]) -> bool:
        for user in node.users:
            if STATransformerRecipe._is_static_control_arg(user, node, modules):
                return True
            for key, value in user.kwargs.items():
                if key in _STAConvertedModel._STATIC_TENSOR_KWARGS and value is node:
                    return True
        return False

    @staticmethod
    def _is_static_control_arg(
        user: fx.Node,
        node: fx.Node,
        modules: Optional[Dict[str, nn.Module]] = None,
    ) -> bool:
        args = tuple(user.args)
        if user.op == "call_module":
            target = getattr(user, "target", None)
            if not isinstance(target, str) or modules is None:
                return False
            module = modules.get(target)
            if not isinstance(
                module, (nn.MultiheadAttention, _STAOnlineMultiheadAttention)
            ):
                return False
            return (
                (len(args) > 3 and args[3] is node)
                or (len(args) > 5 and args[5] is node)
                or user.kwargs.get("key_padding_mask") is node
                or user.kwargs.get("attn_mask") is node
            )
        if user.op != "call_function":
            return False
        if user.target is F.scaled_dot_product_attention:
            return len(args) > 3 and args[3] is node
        return False

    @staticmethod
    def _find_static_placeholder_indices(fx_model: fx.GraphModule) -> frozenset[int]:
        indices = set()
        modules = dict(fx_model.named_modules())
        placeholder_index = 0
        for node in fx_model.graph.nodes:
            if node.op != "placeholder":
                continue
            for user in node.users:
                if STATransformerRecipe._is_static_control_arg(user, node, modules):
                    indices.add(placeholder_index)
                    break
                if any(
                    key in _STAConvertedModel._STATIC_TENSOR_KWARGS and value is node
                    for key, value in user.kwargs.items()
                ):
                    indices.add(placeholder_index)
                    break
            placeholder_index += 1
        return frozenset(indices)

    @staticmethod
    def _is_functional_parameter_attr(node: fx.Node) -> bool:
        weighted_function_targets = {
            F.linear,
            F.conv1d,
            F.conv2d,
            F.conv3d,
        }
        matmul_targets = {
            torch.matmul,
            operator.matmul,
        }
        for user in node.users:
            if user.op != "call_function":
                continue
            args = tuple(user.args)
            if user.target in weighted_function_targets:
                if len(args) > 1 and node is args[1]:
                    return True
                if user.kwargs.get("weight") is node:
                    return True
            elif user.target in matmul_targets:
                if node in args:
                    return True
                if user.kwargs.get("input") is node or user.kwargs.get("other") is node:
                    return True
        return False

    def _should_spike_linear(self, target: str) -> bool:
        if not self.spike_linear:
            return False
        if self.spike_classifier:
            return True
        parts = target.split(".")
        return not any(part in {"head", "heads", "classifier"} for part in parts)

    def _compute_scaled_threshold(
        self, observer: Optional[_STAThresholdObserver]
    ) -> Optional[torch.Tensor]:
        if observer is None:
            return None
        if observer.threshold is None:
            raise RuntimeError(
                "STA threshold observer did not collect calibration data."
            )
        return observer.compute_threshold() * self.threshold_scale

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
