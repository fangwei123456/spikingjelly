from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from .. import functional

__all__ = ["SpikeLinear", "SpikeConv1d", "SpikeConv2d", "SpikeConv3d"]


class SpikeLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-layer-SpikeLinear-cn>` | :ref:`English <registered-layer-SpikeLinear-en>`

        ----

        .. _registered-layer-SpikeLinear-cn:

        * **中文**

        面向二值 0/1 输入的已注册 SpikeLinear。参数、形状和设备语义与对应 PyTorch 层一致。前向调用 ops，反向保存 bool 或位压缩输入，支持一阶和重复反向；无时间状态。不需要在导入时编译原生扩展。

        :param in_features: 输入末维特征数，非负。
        :type in_features: int
        :param out_features: 输出末维特征数，非负。
        :type out_features: int
        :param bias: 是否创建可学习偏置。 默认 ``True``.
        :type bias: bool
        :param device: 参数设备；None 使用 PyTorch 默认设备。 默认 ``None``.
        :type device: Optional[torch.device]
        :param dtype: 参数 dtype；None 使用 PyTorch 默认浮点类型。 默认 ``None``.
        :type dtype: Optional[torch.dtype]
        :raises ValueError: 层参数不满足 PyTorch 的约束。

        ----

        .. _registered-layer-SpikeLinear-en:

        * **English**

        Registered SpikeLinear for binary 0/1 inputs. Parameter, shape, and device semantics match the corresponding PyTorch layer. Forward uses ops and backward saves bool/packed inputs, supporting first-order and repeated backward. There is no temporal state or import-time native compilation.

        :param in_features: Nonnegative input feature count.
        :type in_features: int
        :param out_features: Nonnegative output feature count.
        :type out_features: int
        :param bias: Create a learnable bias. Default: ``True``.
        :type bias: bool
        :param device: Parameter device; None uses the PyTorch default. Default: ``None``.
        :type device: Optional[torch.device]
        :param dtype: Parameter dtype; None uses the PyTorch default floating dtype. Default: ``None``.
        :type dtype: Optional[torch.dtype]
        :raises ValueError: Layer arguments violate PyTorch constraints.
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, spike: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <registered-SpikeLinear-forward-cn>` | :ref:`English <registered-SpikeLinear-forward-en>`

        ----

        .. _registered-SpikeLinear-forward-cn:

        * **中文**

        对二值输入执行注册 Linear，不保存时间状态。

        :param spike: CPU/CUDA 二值浮点输入 ``[..., in_features]``；dtype 遵循 PyTorch Linear。
        :type spike: torch.Tensor
        :return: ``[..., out_features]`` 输出，dtype/device 遵循 PyTorch Linear。
        :rtype: torch.Tensor
        :raises RuntimeError: shape、dtype 或 device 不满足底层运算约束。

        ----

        .. _registered-SpikeLinear-forward-en:

        * **English**

        Apply registered Linear to binary input without temporal state.

        :param spike: CPU/CUDA binary floating input ``[..., in_features]``;
            dtype follows PyTorch Linear constraints.
        :type spike: torch.Tensor
        :return: ``[..., out_features]`` output with PyTorch Linear dtype/device semantics.
        :rtype: torch.Tensor
        :raises RuntimeError: Shape, dtype, or device violates the underlying operator constraints.
        """
        return functional.spike_linear(spike, self.weight, self.bias)


class SpikeConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        stride: Union[int, tuple[int, ...]] = 1,
        padding: Union[str, int, tuple[int, ...]] = 0,
        dilation: Union[int, tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-layer-SpikeConv1d-cn>` | :ref:`English <registered-layer-SpikeConv1d-en>`

        ----

        .. _registered-layer-SpikeConv1d-cn:

        * **中文**

        面向二值 0/1 输入的已注册 SpikeConv1d。参数、形状和设备语义与对应 PyTorch 层一致。前向调用 ops，反向保存 bool 或位压缩输入，支持一阶和重复反向；无时间状态。不需要在导入时编译原生扩展。

        :param in_channels: 正输入通道数。
        :type in_channels: int
        :param out_channels: 正输出通道数。
        :type out_channels: int
        :param kernel_size: 各空间维正卷积核尺寸。
        :type kernel_size: Union[int, tuple[int, ...]]
        :param stride: 各空间维正步幅。 默认 ``1``.
        :type stride: Union[int, tuple[int, ...]]
        :param padding: 非负填充或 same/valid。 默认 ``0``.
        :type padding: Union[str, int, tuple[int, ...]]
        :param dilation: 正膨胀率。 默认 ``1``.
        :type dilation: Union[int, tuple[int, ...]]
        :param groups: 正分组数，整除输入及输出通道数。 默认 ``1``.
        :type groups: int
        :param bias: 是否创建可学习偏置。 默认 ``True``.
        :type bias: bool
        :param padding_mode: zeros/reflect/replicate/circular，遵循 PyTorch。 默认 ``"zeros"``.
        :type padding_mode: str
        :param device: 参数设备；None 使用 PyTorch 默认设备。 默认 ``None``.
        :type device: Optional[torch.device]
        :param dtype: 参数 dtype；None 使用 PyTorch 默认浮点类型。 默认 ``None``.
        :type dtype: Optional[torch.dtype]
        :raises ValueError: 层参数不满足 PyTorch 的约束。

        ----

        .. _registered-layer-SpikeConv1d-en:

        * **English**

        Registered SpikeConv1d for binary 0/1 inputs. Parameter, shape, and device semantics match the corresponding PyTorch layer. Forward uses ops and backward saves bool/packed inputs, supporting first-order and repeated backward. There is no temporal state or import-time native compilation.

        :param in_channels: Positive input channel count.
        :type in_channels: int
        :param out_channels: Positive output channel count.
        :type out_channels: int
        :param kernel_size: Positive kernel size for each spatial dimension.
        :type kernel_size: Union[int, tuple[int, ...]]
        :param stride: Positive spatial stride. Default: ``1``.
        :type stride: Union[int, tuple[int, ...]]
        :param padding: Nonnegative padding or same/valid. Default: ``0``.
        :type padding: Union[str, int, tuple[int, ...]]
        :param dilation: Positive dilation. Default: ``1``.
        :type dilation: Union[int, tuple[int, ...]]
        :param groups: Positive group count dividing input/output channels. Default: ``1``.
        :type groups: int
        :param bias: Create a learnable bias. Default: ``True``.
        :type bias: bool
        :param padding_mode: zeros/reflect/replicate/circular, following PyTorch. Default: ``"zeros"``.
        :type padding_mode: str
        :param device: Parameter device; None uses the PyTorch default. Default: ``None``.
        :type device: Optional[torch.device]
        :param dtype: Parameter dtype; None uses the PyTorch default floating dtype. Default: ``None``.
        :type dtype: Optional[torch.dtype]
        :raises ValueError: Layer arguments violate PyTorch constraints.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def _conv_forward(
        self, spike: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        padding = self.padding
        if self.padding_mode != "zeros":
            spike = F.pad(
                spike, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
            padding = (0,) * 1
        return functional.spike_conv1d(
            spike, weight, bias, self.stride, padding, self.dilation, self.groups
        )


class SpikeConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        stride: Union[int, tuple[int, ...]] = 1,
        padding: Union[str, int, tuple[int, ...]] = 0,
        dilation: Union[int, tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-layer-SpikeConv2d-cn>` | :ref:`English <registered-layer-SpikeConv2d-en>`

        ----

        .. _registered-layer-SpikeConv2d-cn:

        * **中文**

        面向二值 0/1 输入的已注册 SpikeConv2d。参数、形状和设备语义与对应 PyTorch 层一致。前向调用 ops，反向保存 bool 或位压缩输入，支持一阶和重复反向；无时间状态。不需要在导入时编译原生扩展。

        :param in_channels: 正输入通道数。
        :type in_channels: int
        :param out_channels: 正输出通道数。
        :type out_channels: int
        :param kernel_size: 各空间维正卷积核尺寸。
        :type kernel_size: Union[int, tuple[int, ...]]
        :param stride: 各空间维正步幅。 默认 ``1``.
        :type stride: Union[int, tuple[int, ...]]
        :param padding: 非负填充或 same/valid。 默认 ``0``.
        :type padding: Union[str, int, tuple[int, ...]]
        :param dilation: 正膨胀率。 默认 ``1``.
        :type dilation: Union[int, tuple[int, ...]]
        :param groups: 正分组数，整除输入及输出通道数。 默认 ``1``.
        :type groups: int
        :param bias: 是否创建可学习偏置。 默认 ``True``.
        :type bias: bool
        :param padding_mode: zeros/reflect/replicate/circular，遵循 PyTorch。 默认 ``"zeros"``.
        :type padding_mode: str
        :param device: 参数设备；None 使用 PyTorch 默认设备。 默认 ``None``.
        :type device: Optional[torch.device]
        :param dtype: 参数 dtype；None 使用 PyTorch 默认浮点类型。 默认 ``None``.
        :type dtype: Optional[torch.dtype]
        :raises ValueError: 层参数不满足 PyTorch 的约束。

        ----

        .. _registered-layer-SpikeConv2d-en:

        * **English**

        Registered SpikeConv2d for binary 0/1 inputs. Parameter, shape, and device semantics match the corresponding PyTorch layer. Forward uses ops and backward saves bool/packed inputs, supporting first-order and repeated backward. There is no temporal state or import-time native compilation.

        :param in_channels: Positive input channel count.
        :type in_channels: int
        :param out_channels: Positive output channel count.
        :type out_channels: int
        :param kernel_size: Positive kernel size for each spatial dimension.
        :type kernel_size: Union[int, tuple[int, ...]]
        :param stride: Positive spatial stride. Default: ``1``.
        :type stride: Union[int, tuple[int, ...]]
        :param padding: Nonnegative padding or same/valid. Default: ``0``.
        :type padding: Union[str, int, tuple[int, ...]]
        :param dilation: Positive dilation. Default: ``1``.
        :type dilation: Union[int, tuple[int, ...]]
        :param groups: Positive group count dividing input/output channels. Default: ``1``.
        :type groups: int
        :param bias: Create a learnable bias. Default: ``True``.
        :type bias: bool
        :param padding_mode: zeros/reflect/replicate/circular, following PyTorch. Default: ``"zeros"``.
        :type padding_mode: str
        :param device: Parameter device; None uses the PyTorch default. Default: ``None``.
        :type device: Optional[torch.device]
        :param dtype: Parameter dtype; None uses the PyTorch default floating dtype. Default: ``None``.
        :type dtype: Optional[torch.dtype]
        :raises ValueError: Layer arguments violate PyTorch constraints.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def _conv_forward(
        self, spike: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        padding = self.padding
        if self.padding_mode != "zeros":
            spike = F.pad(
                spike, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
            padding = (0,) * 2
        return functional.spike_conv2d(
            spike, weight, bias, self.stride, padding, self.dilation, self.groups
        )


class SpikeConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        stride: Union[int, tuple[int, ...]] = 1,
        padding: Union[str, int, tuple[int, ...]] = 0,
        dilation: Union[int, tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-layer-SpikeConv3d-cn>` | :ref:`English <registered-layer-SpikeConv3d-en>`

        ----

        .. _registered-layer-SpikeConv3d-cn:

        * **中文**

        面向二值 0/1 输入的已注册 SpikeConv3d。参数、形状和设备语义与对应 PyTorch 层一致。前向调用 ops，反向保存 bool 或位压缩输入，支持一阶和重复反向；无时间状态。不需要在导入时编译原生扩展。

        :param in_channels: 正输入通道数。
        :type in_channels: int
        :param out_channels: 正输出通道数。
        :type out_channels: int
        :param kernel_size: 各空间维正卷积核尺寸。
        :type kernel_size: Union[int, tuple[int, ...]]
        :param stride: 各空间维正步幅。 默认 ``1``.
        :type stride: Union[int, tuple[int, ...]]
        :param padding: 非负填充或 same/valid。 默认 ``0``.
        :type padding: Union[str, int, tuple[int, ...]]
        :param dilation: 正膨胀率。 默认 ``1``.
        :type dilation: Union[int, tuple[int, ...]]
        :param groups: 正分组数，整除输入及输出通道数。 默认 ``1``.
        :type groups: int
        :param bias: 是否创建可学习偏置。 默认 ``True``.
        :type bias: bool
        :param padding_mode: zeros/reflect/replicate/circular，遵循 PyTorch。 默认 ``"zeros"``.
        :type padding_mode: str
        :param device: 参数设备；None 使用 PyTorch 默认设备。 默认 ``None``.
        :type device: Optional[torch.device]
        :param dtype: 参数 dtype；None 使用 PyTorch 默认浮点类型。 默认 ``None``.
        :type dtype: Optional[torch.dtype]
        :raises ValueError: 层参数不满足 PyTorch 的约束。

        ----

        .. _registered-layer-SpikeConv3d-en:

        * **English**

        Registered SpikeConv3d for binary 0/1 inputs. Parameter, shape, and device semantics match the corresponding PyTorch layer. Forward uses ops and backward saves bool/packed inputs, supporting first-order and repeated backward. There is no temporal state or import-time native compilation.

        :param in_channels: Positive input channel count.
        :type in_channels: int
        :param out_channels: Positive output channel count.
        :type out_channels: int
        :param kernel_size: Positive kernel size for each spatial dimension.
        :type kernel_size: Union[int, tuple[int, ...]]
        :param stride: Positive spatial stride. Default: ``1``.
        :type stride: Union[int, tuple[int, ...]]
        :param padding: Nonnegative padding or same/valid. Default: ``0``.
        :type padding: Union[str, int, tuple[int, ...]]
        :param dilation: Positive dilation. Default: ``1``.
        :type dilation: Union[int, tuple[int, ...]]
        :param groups: Positive group count dividing input/output channels. Default: ``1``.
        :type groups: int
        :param bias: Create a learnable bias. Default: ``True``.
        :type bias: bool
        :param padding_mode: zeros/reflect/replicate/circular, following PyTorch. Default: ``"zeros"``.
        :type padding_mode: str
        :param device: Parameter device; None uses the PyTorch default. Default: ``None``.
        :type device: Optional[torch.device]
        :param dtype: Parameter dtype; None uses the PyTorch default floating dtype. Default: ``None``.
        :type dtype: Optional[torch.dtype]
        :raises ValueError: Layer arguments violate PyTorch constraints.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def _conv_forward(
        self, spike: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        padding = self.padding
        if self.padding_mode != "zeros":
            spike = F.pad(
                spike, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
            padding = (0,) * 3
        return functional.spike_conv3d(
            spike, weight, bias, self.stride, padding, self.dilation, self.groups
        )
