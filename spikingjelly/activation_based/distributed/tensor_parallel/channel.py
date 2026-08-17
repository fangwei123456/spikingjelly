from __future__ import annotations

from typing import Any, Literal, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ... import base
from ...memopt import in_gc_1st_forward

__all__ = [
    "ChannelShardBatchNorm1d",
    "ChannelShardBatchNorm2d",
    "ChannelShardConv1d",
    "ChannelShardConv2d",
]


def _require_even_shard(total: int, world_size: int, name: str) -> None:
    if total % world_size:
        raise ValueError(
            f"{name}={total} must be divisible by tensor-parallel world_size={world_size}."
        )


def _shard_range(total: int, rank: int, world_size: int) -> tuple[int, int]:
    shard = total // world_size
    start = shard * rank
    return start, start + shard


class _ColwiseBackwardAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, process_group):
        ctx.process_group = process_group
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, group=ctx.process_group)
        return grad_output, None


class _ChannelShardConv(nn.Module, base.StepModule):
    _conv = None
    _input_shape = ""
    _spatial_dims = 0
    _zero_padding = ()

    def __init__(
        self,
        source: nn.Module,
        process_group: Optional[Any],
        mode: Literal["colwise", "rowwise"],
    ) -> None:
        super().__init__()
        if source.groups != 1:
            raise NotImplementedError(f"{type(self).__name__} only supports groups=1.")
        if mode not in ("colwise", "rowwise"):
            raise ValueError(f"Unsupported {type(self).__name__} mode '{mode}'.")

        self.mode = mode
        self.process_group = process_group
        rank = dist.get_rank(process_group) if process_group is not None else 0
        self.world_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        self.step_mode = getattr(source, "step_mode", "s")
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.padding_mode = source.padding_mode
        self._reversed_padding_repeated_twice = tuple(
            source._reversed_padding_repeated_twice
        )
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size

        if mode == "colwise":
            _require_even_shard(source.out_channels, self.world_size, "out_channels")
            start, end = _shard_range(source.out_channels, rank, self.world_size)
            weight = source.weight.detach()[start:end].clone()
            bias = (
                source.bias.detach()[start:end].clone()
                if source.bias is not None
                else None
            )
            self.register_parameter(
                "weight",
                nn.Parameter(weight, requires_grad=source.weight.requires_grad),
            )
            self.register_parameter(
                "bias",
                nn.Parameter(bias, requires_grad=source.bias.requires_grad)
                if bias is not None
                else None,
            )
        else:
            _require_even_shard(source.in_channels, self.world_size, "in_channels")
            start, end = _shard_range(source.in_channels, rank, self.world_size)
            weight = source.weight.detach()[:, start:end].clone()
            bias = source.bias.detach().clone() if source.bias is not None else None
            self.register_parameter(
                "weight",
                nn.Parameter(weight, requires_grad=source.weight.requires_grad),
            )
            self.register_parameter(
                "bias",
                nn.Parameter(bias, requires_grad=source.bias.requires_grad)
                if bias is not None
                else None,
            )

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, step_mode={self.step_mode}, "
            f"in_channels={self.in_channels}, out_channels={self.out_channels}"
        )

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = self._zero_padding
        else:
            padding = self.padding

        if self.mode == "colwise":
            if self.world_size > 1:
                x = _ColwiseBackwardAllReduce.apply(x, self.process_group)
            return self._conv(
                x,
                self.weight,
                self.bias,
                self.stride,
                padding,
                self.dilation,
                self.groups,
            )

        y = self._conv(
            x,
            self.weight,
            None,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )
        if self.world_size > 1:
            dist.all_reduce(y, group=self.process_group)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, *(1,) * self._spatial_dims)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self._conv_forward(x)
        if self.step_mode != "m":
            raise ValueError(f"Unsupported step_mode='{self.step_mode}'.")
        if x.dim() != self._spatial_dims + 3:
            raise ValueError(
                f"expected x with shape {self._input_shape}, but got {x.shape}!"
            )
        y = self._conv_forward(x.flatten(0, 1))
        return y.view(*x.shape[:2], *y.shape[1:])


class ChannelShardConv2d(_ChannelShardConv):
    _conv = staticmethod(F.conv2d)
    _input_shape = "[T, N, C, H, W]"
    _spatial_dims = 2
    _zero_padding = (0, 0)

    def __init__(
        self,
        source: nn.Module,
        process_group: Optional[Any],
        mode: Literal["colwise", "rowwise"],
    ) -> None:
        r"""**API Language** - :ref:`中文 <ChannelShardConv2d-cn>` | :ref:`English <ChannelShardConv2d-en>`

        ----

        .. _ChannelShardConv2d-cn:

        * **中文**

        构造二维通道分片卷积。

        :param source: 源 Conv2d 模块。
        :type source: nn.Module
        :param process_group: 分布式进程组。
        :type process_group: Any
        :param mode: ``"colwise"`` 或 ``"rowwise"``。
        :type mode: str

        ----

        .. _ChannelShardConv2d-en:

        * **English**

        Build a channel-sharded 2D convolution.

        :param source: Source Conv2d module.
        :type source: nn.Module
        :param process_group: Distributed process group.
        :type process_group: Any
        :param mode: ``"colwise"`` or ``"rowwise"``.
        :type mode: str
        """
        super().__init__(source, process_group, mode)


class ChannelShardConv1d(_ChannelShardConv):
    _conv = staticmethod(F.conv1d)
    _input_shape = "[T, N, C, L]"
    _spatial_dims = 1
    _zero_padding = (0,)

    def __init__(
        self,
        source: nn.Module,
        process_group: Optional[Any],
        mode: Literal["colwise", "rowwise"],
    ) -> None:
        r"""**API Language** - :ref:`中文 <ChannelShardConv1d-cn>` | :ref:`English <ChannelShardConv1d-en>`

        ----

        .. _ChannelShardConv1d-cn:

        * **中文**

        构造一维通道分片卷积。

        :param source: 源 Conv1d 模块。
        :type source: nn.Module
        :param process_group: 分布式进程组。
        :type process_group: Any
        :param mode: ``"colwise"`` 或 ``"rowwise"``。
        :type mode: str

        ----

        .. _ChannelShardConv1d-en:

        * **English**

        Build a channel-sharded 1D convolution.

        :param source: Source Conv1d module.
        :type source: nn.Module
        :param process_group: Distributed process group.
        :type process_group: Any
        :param mode: ``"colwise"`` or ``"rowwise"``.
        :type mode: str
        """
        super().__init__(source, process_group, mode)


class _ChannelShardBatchNorm(nn.Module, base.StepModule):
    _input_shape = ""
    _spatial_dims = 0

    def __init__(self, source: nn.Module, process_group: Optional[Any]) -> None:
        super().__init__()
        self.process_group = process_group
        rank = dist.get_rank(process_group) if process_group is not None else 0
        self.world_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        self.step_mode = getattr(source, "step_mode", "s")
        self.eps = source.eps
        self.momentum = source.momentum
        self.affine = source.affine
        self.track_running_stats = source.track_running_stats
        self.num_features = source.num_features

        _require_even_shard(source.num_features, self.world_size, "num_features")
        start, end = _shard_range(source.num_features, rank, self.world_size)

        if self.affine:
            self.register_parameter(
                "weight",
                nn.Parameter(
                    source.weight.detach()[start:end].clone(),
                    requires_grad=source.weight.requires_grad,
                ),
            )
            self.register_parameter(
                "bias",
                nn.Parameter(
                    source.bias.detach()[start:end].clone(),
                    requires_grad=source.bias.requires_grad,
                ),
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer(
                "running_mean", source.running_mean.detach()[start:end].clone()
            )
            self.register_buffer(
                "running_var", source.running_var.detach()[start:end].clone()
            )
            self.register_buffer(
                "num_batches_tracked", source.num_batches_tracked.detach().clone()
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        self.training = source.training

    def extra_repr(self) -> str:
        return f"step_mode={self.step_mode}, num_features={self.num_features}"

    def _batch_norm(self, x: torch.Tensor) -> torch.Tensor:
        exponential_average_factor = self.momentum
        checkpoint_forward = in_gc_1st_forward()
        if self.training and self.track_running_stats and not checkpoint_forward:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(
                    self.num_batches_tracked.item()
                )
        return F.batch_norm(
            x,
            None if checkpoint_forward else self.running_mean,
            None if checkpoint_forward else self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self._batch_norm(x)
        if self.step_mode != "m":
            raise ValueError(f"Unsupported step_mode='{self.step_mode}'.")
        if x.dim() != self._spatial_dims + 3:
            raise ValueError(
                f"expected x with shape {self._input_shape}, but got {x.shape}!"
            )
        y = self._batch_norm(x.flatten(0, 1))
        return y.view(*x.shape[:2], *y.shape[1:])


class ChannelShardBatchNorm2d(_ChannelShardBatchNorm):
    _input_shape = "[T, N, C, H, W]"
    _spatial_dims = 2

    def __init__(self, source: nn.Module, process_group: Optional[Any]) -> None:
        r"""**API Language** - :ref:`中文 <ChannelShardBatchNorm2d-cn>` | :ref:`English <ChannelShardBatchNorm2d-en>`

        ----

        .. _ChannelShardBatchNorm2d-cn:

        * **中文**

        构造二维通道分片批归一化。

        :param source: 源 BatchNorm2d 模块。
        :type source: nn.Module
        :param process_group: 分布式进程组。
        :type process_group: Any

        ----

        .. _ChannelShardBatchNorm2d-en:

        * **English**

        Build channel-sharded 2D batch normalization.

        :param source: Source BatchNorm2d module.
        :type source: nn.Module
        :param process_group: Distributed process group.
        :type process_group: Any
        """
        super().__init__(source, process_group)


class ChannelShardBatchNorm1d(_ChannelShardBatchNorm):
    _input_shape = "[T, N, C, L]"
    _spatial_dims = 1

    def __init__(self, source: nn.Module, process_group: Optional[Any]) -> None:
        r"""**API Language** - :ref:`中文 <ChannelShardBatchNorm1d-cn>` | :ref:`English <ChannelShardBatchNorm1d-en>`

        ----

        .. _ChannelShardBatchNorm1d-cn:

        * **中文**

        构造一维通道分片批归一化。

        :param source: 源 BatchNorm1d 模块。
        :type source: nn.Module
        :param process_group: 分布式进程组。
        :type process_group: Any

        ----

        .. _ChannelShardBatchNorm1d-en:

        * **English**

        Build channel-sharded 1D batch normalization.

        :param source: Source BatchNorm1d module.
        :type source: nn.Module
        :param process_group: Distributed process group.
        :type process_group: Any
        """
        super().__init__(source, process_group)
