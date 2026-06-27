import math
from typing import Optional

import torch
import torch.nn as nn


__all__ = ["VoltageHook", "VoltageScaler", "ChannelVoltageScaler"]


def _safe_quantile(
    x: torch.Tensor,
    quantile: float,
    dim: Optional[int] = None,
    max_elements: int = 1048576,
) -> torch.Tensor:
    """Approximate ``torch.quantile`` with bounded calibration memory.

    Large activation tensors can make ``torch.quantile`` allocate enough
    temporary memory to dominate ANN2SNN calibration. This helper limits each
    reduction to at most ``max_elements`` evenly spaced values, then applies
    kth-value interpolation. The result is deterministic but approximate when
    subsampling is used.
    """
    if not (0.0 <= quantile <= 1.0):
        raise ValueError("quantile must be in [0, 1].")
    if x.numel() == 0:
        raise ValueError("quantile input must not be empty.")
    if max_elements <= 0:
        raise ValueError("max_elements must be positive.")

    if dim is None:
        values = x.reshape(-1)
        if values.numel() > max_elements:
            stride = math.ceil(values.numel() / max_elements)
            values = values[::stride][:max_elements].contiguous()
        rank = quantile * (values.numel() - 1)
        lower_idx = int(math.floor(rank))
        upper_idx = int(math.ceil(rank))
        lower = values.kthvalue(lower_idx + 1).values
        if lower_idx == upper_idx:
            return lower
        upper = values.kthvalue(upper_idx + 1).values
        return lower + (upper - lower) * (rank - lower_idx)

    if dim < 0:
        dim += x.dim()
    if dim < 0 or dim >= x.dim():
        raise ValueError("dim is out of range.")
    values = x.movedim(dim, -1)
    original_shape = values.shape[:-1]
    values = values.reshape(-1, values.shape[-1]).contiguous()
    if values.shape[-1] > max_elements:
        stride = math.ceil(values.shape[-1] / max_elements)
        values = values[:, ::stride][:, :max_elements].contiguous()
    rank = quantile * (values.shape[-1] - 1)
    lower_idx = int(math.floor(rank))
    upper_idx = int(math.ceil(rank))
    lower = values.kthvalue(lower_idx + 1, dim=1).values
    if lower_idx == upper_idx:
        return lower.reshape(original_shape)
    upper = values.kthvalue(upper_idx + 1, dim=1).values
    result = lower + (upper - lower) * (rank - lower_idx)
    return result.reshape(original_shape)


class VoltageHook(nn.Module):
    def __init__(self, scale=1.0, momentum=0.1, mode="Max"):
        r"""
        **API Language** - :ref:`中文 <VoltageHook.__init__-cn>` | :ref:`English <VoltageHook.__init__-en>`

        ----

        .. _VoltageHook.__init__-cn:

        * **中文**

        :class:`VoltageHook` 的构造函数。

        :param scale: 缩放初始值
        :type scale: float
        :param momentum: 动量值
        :type momentum: float
        :param mode: 模式。``"Max"`` 表示记录ANN激活最大值；``"99.9%"`` 表示记录99.9%分位点；
            0-1 的 float 表示记录激活最大值的对应倍数
        :type mode: str, float

        ----

        .. _VoltageHook.__init__-en:

        * **English**

        Constructor of :class:`VoltageHook`.

        :param scale: initial scaling value
        :type scale: float
        :param momentum: momentum value
        :type momentum: float
        :param mode: Mode. ``"Max"`` means recording the maximum value of ANN activation;
            ``"99.9%"`` means recording the 99.9% percentile; a float of 0-1 means
            recording the corresponding multiple of the maximum value
        :type mode: str, float
        """
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))
        self.mode = mode
        self.num_batches_tracked = 0
        self.momentum = momentum

    def forward(self, x):
        r"""
        **API Language** - :ref:`中文 <VoltageHook.forward-cn>` | :ref:`English <VoltageHook.forward-en>`

        ----

        .. _VoltageHook.forward-cn:

        * **中文**

        前向传播函数。不对输入张量做任何处理，只是抓取ReLU的激活值用于确定ANN激活范围。

        :param x: 输入张量
        :type x: torch.Tensor
        :return: 原输入张量
        :rtype: torch.Tensor

        ----

        .. _VoltageHook.forward-en:

        * **English**

        Forward function. It doesn't process input tensors, but hooks the activation
        values of ReLU to determine ANN activation ranges.

        :param x: input tensor
        :type x: torch.Tensor
        :return: original input tensor
        :rtype: torch.Tensor
        """
        err_msg = "You have used a non-defined VoltageScale Method."
        if isinstance(self.mode, str):
            if not self.mode:
                raise NotImplementedError(err_msg)
            if self.mode[-1] == "%":
                try:
                    quantile = float(self.mode[:-1]) / 100.0
                    if not (0.0 <= quantile <= 1.0):
                        raise NotImplementedError(err_msg)
                    quantile_input = x.detach()
                    if quantile_input.dtype in [torch.float16, torch.bfloat16]:
                        quantile_input = quantile_input.to(torch.float32)
                    s_t = _safe_quantile(quantile_input, quantile).to(x.dtype)
                except (ValueError, RuntimeError) as exc:
                    raise NotImplementedError(err_msg) from exc
            elif self.mode.lower() in ["max"]:
                s_t = x.max().detach()
            else:
                raise NotImplementedError(err_msg)
        elif (
            isinstance(self.mode, (int, float))
            and not isinstance(self.mode, bool)
            and self.mode <= 1
            and self.mode > 0
        ):
            s_t = x.max().detach() * self.mode
        else:
            raise NotImplementedError(err_msg)

        if self.num_batches_tracked == 0:
            self.scale = s_t
        else:
            self.scale = (1 - self.momentum) * self.scale + self.momentum * s_t
        self.num_batches_tracked += x.shape[0]
        return x


class VoltageScaler(nn.Module):
    def __init__(self, scale=1.0):
        r"""
        **API Language** - :ref:`中文 <VoltageScaler.__init__-cn>` | :ref:`English <VoltageScaler.__init__-en>`

        ----

        .. _VoltageScaler.__init__-cn:

        * **中文**

        :class:`VoltageScaler` 的构造函数。用于SNN推理中缩放电流。

        :param scale: 缩放值
        :type scale: float

        ----

        .. _VoltageScaler.__init__-en:

        * **English**

        Constructor of :class:`VoltageScaler`. Used for scaling current in SNN inference.

        :param scale: scaling value
        :type scale: float
        """
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x):
        r"""
        **API Language** - :ref:`中文 <VoltageScaler.forward-cn>` | :ref:`English <VoltageScaler.forward-en>`

        ----

        .. _VoltageScaler.forward-cn:

        * **中文**

        前向传播函数。对输入电流进行缩放。

        :param x: 输入张量，亦即输入电流
        :type x: torch.Tensor
        :return: 缩放后的电流
        :rtype: torch.Tensor

        ----

        .. _VoltageScaler.forward-en:

        * **English**

        Forward function. Scales the input current.

        :param x: input tensor, or input current
        :type x: torch.Tensor
        :return: current after scaling
        :rtype: torch.Tensor
        """
        return x * self.scale

    def extra_repr(self):
        return "%f" % self.scale.item()


class ChannelVoltageScaler(nn.Module):
    def __init__(self, scale=1.0, channel_dim: int = 1):
        r"""
        **API Language** - :ref:`中文 <ChannelVoltageScaler.__init__-cn>` | :ref:`English <ChannelVoltageScaler.__init__-en>`

        ----

        .. _ChannelVoltageScaler.__init__-cn:

        * **中文**

        按通道缩放输入电流。``scale`` 可以是标量或 1D 张量；当为 1D 张量时，
        会沿 ``channel_dim`` 广播到输入张量。该模块用于需要 channel-wise
        阈值/尺度的 ANN2SNN 转换 recipe。

        :param scale: 缩放值，必须为有限正标量或有限正 1D 张量。
        :type scale: float or torch.Tensor
        :param channel_dim: ``scale`` 对应的输入通道维。
        :type channel_dim: int
        :raises ValueError: 当 ``scale`` 或 ``channel_dim`` 非法时抛出。

        ----

        .. _ChannelVoltageScaler.__init__-en:

        * **English**

        Scale input current channel-wise. ``scale`` can be a scalar or a 1D
        tensor; a 1D tensor is broadcast to the input tensor along
        ``channel_dim``. This module is used by ANN2SNN recipes that need
        channel-wise thresholds or scales.

        :param scale: Scaling value. Must be a finite positive scalar or finite
            positive 1D tensor.
        :type scale: float or torch.Tensor
        :param channel_dim: Input channel dimension corresponding to ``scale``.
        :type channel_dim: int
        :raises ValueError: If ``scale`` or ``channel_dim`` is invalid.
        """
        super().__init__()
        if not isinstance(channel_dim, int):
            raise ValueError("channel_dim must be int.")
        scale_tensor = torch.as_tensor(scale).detach().clone()
        if scale_tensor.dim() > 1:
            raise ValueError("scale must be a scalar or a 1D tensor.")
        if scale_tensor.numel() == 0:
            raise ValueError("scale must not be empty.")
        if not torch.isfinite(scale_tensor).all() or (scale_tensor <= 0).any():
            raise ValueError("scale must contain finite positive values.")
        self.register_buffer("scale", scale_tensor)
        self.channel_dim = channel_dim

    def _view_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale.dim() == 0:
            return self.scale
        channel_dim = self.channel_dim
        if channel_dim < 0:
            channel_dim += x.dim()
        if channel_dim < 0 or channel_dim >= x.dim():
            raise ValueError("channel_dim is out of range for input.")
        if x.shape[channel_dim] != self.scale.numel():
            raise ValueError(
                "Input channel dimension does not match scale length: "
                f"got {x.shape[channel_dim]} and {self.scale.numel()}."
            )
        shape = [1] * x.dim()
        shape[channel_dim] = self.scale.numel()
        return self.scale.reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <ChannelVoltageScaler.forward-cn>` | :ref:`English <ChannelVoltageScaler.forward-en>`

        ----

        .. _ChannelVoltageScaler.forward-cn:

        * **中文**

        按通道缩放输入张量。

        :param x: 输入张量。
        :type x: torch.Tensor
        :return: 缩放后的张量。
        :rtype: torch.Tensor

        ----

        .. _ChannelVoltageScaler.forward-en:

        * **English**

        Scale the input tensor channel-wise.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Scaled tensor.
        :rtype: torch.Tensor
        """
        return x * self._view_scale(x).to(dtype=x.dtype, device=x.device)

    def extra_repr(self):
        return f"scale_shape={tuple(self.scale.shape)}, channel_dim={self.channel_dim}"
