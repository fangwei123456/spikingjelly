import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based.distributed.tensor_parallel.debug import (
    _record_tp_all_reduce,
)
from spikingjelly.activation_based.distributed.tensor_parallel.state import (
    _require_even_shard,
    _shard_range,
)


def _reversed_padding_repeated_twice(source: nn.Module) -> tuple[int, ...]:
    padding = getattr(source, "_reversed_padding_repeated_twice", None)
    if padding is not None:
        return tuple(padding)

    padding = source.padding
    if isinstance(padding, str):
        if padding == "valid":
            return tuple(0 for _ in range(2 * len(source.kernel_size)))
        if padding != "same":
            raise ValueError(f"Unsupported padding string '{padding}'.")
        total_padding = [
            dilation * (kernel - 1)
            for dilation, kernel in zip(
                source.dilation, source.kernel_size, strict=True
            )
        ]
        return tuple(
            value
            for total in reversed(total_padding)
            for value in (total // 2, total - total // 2)
        )
    if isinstance(padding, int):
        padding = (padding,)
    return tuple(value for pad in reversed(padding) for value in (pad, pad))


class _ColwiseBackwardAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, process_group, world_size: int):
        ctx.process_group = process_group
        ctx.world_size = int(world_size)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.world_size > 1 and dist.is_available() and dist.is_initialized():
            _record_tp_all_reduce(grad_output)
            dist.all_reduce(grad_output, group=ctx.process_group)
        return grad_output, None, None


class ChannelShardConv2d(nn.Module):
    def __init__(self, source: nn.Module, process_group, mode: str):
        """
        **API Language** - :ref:`中文 <ChannelShardConv2d-cn>` | :ref:`English <ChannelShardConv2d-en>`

        ----

        .. _ChannelShardConv2d-cn:

        * **中文**

        支持通道分片的二维卷积层。

        :param source: 源 Conv2d 模块
        :type source: nn.Module
        :param process_group: 分布式进程组
        :type process_group: Any
        :param mode: 分片模式
        :type mode: str

        ----

        .. _ChannelShardConv2d-en:

        * **English**

        2D conv layer with channel sharding support.

        :param source: Source Conv2d module
        :type source: nn.Module
        :param process_group: Distributed process group
        :type process_group: Any
        :param mode: Sharding mode
        :type mode: str
        """
        super().__init__()
        if source.groups != 1:
            raise NotImplementedError("ChannelShardConv2d only supports groups=1.")

        self.mode = mode
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if process_group is not None else 0
        self.world_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        self.step_mode = getattr(source, "step_mode", "s")
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.padding_mode = source.padding_mode
        self._reversed_padding_repeated_twice = _reversed_padding_repeated_twice(source)

        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size

        if mode == "colwise":
            _require_even_shard(source.out_channels, self.world_size, "out_channels")
            start, end = _shard_range(source.out_channels, self.rank, self.world_size)
            weight = source.weight.detach()[start:end].clone()
            bias = (
                source.bias.detach()[start:end].clone()
                if source.bias is not None
                else None
            )
            self.local_out_channels = end - start
            self.register_parameter("weight", nn.Parameter(weight))
            self.register_parameter(
                "bias", nn.Parameter(bias) if bias is not None else None
            )
        elif mode == "rowwise":
            _require_even_shard(source.in_channels, self.world_size, "in_channels")
            start, end = _shard_range(source.in_channels, self.rank, self.world_size)
            weight = source.weight.detach()[:, start:end].clone()
            bias = source.bias.detach().clone() if source.bias is not None else None
            self.local_in_channels = end - start
            self.register_parameter("weight", nn.Parameter(weight))
            self.register_parameter(
                "bias", nn.Parameter(bias) if bias is not None else None
            )
        else:
            raise ValueError(f"Unsupported ChannelShardConv2d mode '{mode}'.")

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, step_mode={self.step_mode}, "
            f"in_channels={self.in_channels}, out_channels={self.out_channels}"
        )

    def _conv2d(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding

        if self.mode == "colwise":
            if self.world_size > 1:
                x = _ColwiseBackwardAllReduce.apply(
                    x, self.process_group, self.world_size
                )
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                padding,
                self.dilation,
                self.groups,
            )

        y = F.conv2d(
            x,
            self.weight,
            None,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )
        if self.world_size > 1:
            _record_tp_all_reduce(y)
            dist.all_reduce(y, group=self.process_group)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self._conv2d(x)

        if self.step_mode != "m":
            raise ValueError(f"Unsupported step_mode='{self.step_mode}'.")
        if x.dim() != 5:
            raise ValueError(
                f"expected x with shape [T, N, C, H, W], but got {x.shape}!"
            )

        y_shape = [x.shape[0], x.shape[1]]
        y = self._conv2d(x.flatten(0, 1))
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)


class ChannelShardConv1d(nn.Module):
    def __init__(self, source: nn.Module, process_group, mode: str):
        """
        **API Language** - :ref:`中文 <ChannelShardConv1d-cn>` | :ref:`English <ChannelShardConv1d-en>`

        ----

        .. _ChannelShardConv1d-cn:

        * **中文**

        支持通道分片的一维卷积层。

        :param source: 源 Conv1d 模块
        :type source: nn.Module
        :param process_group: 分布式进程组
        :type process_group: Any
        :param mode: 分片模式
        :type mode: str

        ----

        .. _ChannelShardConv1d-en:

        * **English**

        1D conv layer with channel sharding support.

        :param source: Source Conv1d module
        :type source: nn.Module
        :param process_group: Distributed process group
        :type process_group: Any
        :param mode: Sharding mode
        :type mode: str
        """
        super().__init__()
        if source.groups != 1:
            raise NotImplementedError("ChannelShardConv1d only supports groups=1.")

        self.mode = mode
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if process_group is not None else 0
        self.world_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        self.step_mode = getattr(source, "step_mode", "s")
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.padding_mode = source.padding_mode
        self._reversed_padding_repeated_twice = _reversed_padding_repeated_twice(source)
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size

        if mode == "colwise":
            _require_even_shard(source.out_channels, self.world_size, "out_channels")
            start, end = _shard_range(source.out_channels, self.rank, self.world_size)
            weight = source.weight.detach()[start:end].clone()
            bias = (
                source.bias.detach()[start:end].clone()
                if source.bias is not None
                else None
            )
            self.local_out_channels = end - start
            self.register_parameter("weight", nn.Parameter(weight))
            self.register_parameter(
                "bias", nn.Parameter(bias) if bias is not None else None
            )
        elif mode == "rowwise":
            _require_even_shard(source.in_channels, self.world_size, "in_channels")
            start, end = _shard_range(source.in_channels, self.rank, self.world_size)
            weight = source.weight.detach()[:, start:end].clone()
            bias = source.bias.detach().clone() if source.bias is not None else None
            self.local_in_channels = end - start
            self.register_parameter("weight", nn.Parameter(weight))
            self.register_parameter(
                "bias", nn.Parameter(bias) if bias is not None else None
            )
        else:
            raise ValueError(f"Unsupported ChannelShardConv1d mode '{mode}'.")

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, step_mode={self.step_mode}, in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}"
        )

    def _conv1d(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0,)
        else:
            padding = self.padding

        if self.mode == "colwise":
            if self.world_size > 1:
                x = _ColwiseBackwardAllReduce.apply(
                    x, self.process_group, self.world_size
                )
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                padding,
                self.dilation,
                self.groups,
            )

        y = F.conv1d(
            x,
            self.weight,
            None,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )
        if self.world_size > 1:
            _record_tp_all_reduce(y)
            dist.all_reduce(y, group=self.process_group)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)
        return y

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self._conv1d(x)

        if self.step_mode != "m":
            raise ValueError(f"Unsupported step_mode='{self.step_mode}'.")
        if x.dim() != 4:
            raise ValueError(f"expected x with shape [T, N, C, L], but got {x.shape}!")

        y_shape = [x.shape[0], x.shape[1]]
        y = self._conv1d(x.flatten(0, 1))
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)


class ChannelShardBatchNorm2d(nn.Module):
    def __init__(self, source: nn.Module, process_group):
        """
        **API Language** - :ref:`中文 <ChannelShardBatchNorm2d-cn>` | :ref:`English <ChannelShardBatchNorm2d-en>`

        ----

        .. _ChannelShardBatchNorm2d-cn:

        * **中文**

        支持通道分片的二维批归一化层。

        :param source: 源 BatchNorm2d 模块
        :type source: nn.Module
        :param process_group: 分布式进程组
        :type process_group: Any

        ----

        .. _ChannelShardBatchNorm2d-en:

        * **English**

        2D batch norm layer with channel sharding.

        :param source: Source BatchNorm2d module
        :type source: nn.Module
        :param process_group: Distributed process group
        :type process_group: Any
        """
        super().__init__()
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if process_group is not None else 0
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
        start, end = _shard_range(source.num_features, self.rank, self.world_size)

        if self.affine:
            self.register_parameter(
                "weight", nn.Parameter(source.weight.detach()[start:end].clone())
            )
            self.register_parameter(
                "bias", nn.Parameter(source.bias.detach()[start:end].clone())
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
            num_batches_tracked = getattr(source, "num_batches_tracked", None)
            if num_batches_tracked is not None:
                self.register_buffer(
                    "num_batches_tracked", num_batches_tracked.detach().clone()
                )
            else:
                self.register_buffer("num_batches_tracked", None)
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        self.training = source.training

    def extra_repr(self) -> str:
        return f"step_mode={self.step_mode}, num_features={self.num_features}"

    def _batch_norm(self, x: torch.Tensor) -> torch.Tensor:
        exponential_average_factor = self.momentum
        if (
            self.training
            and self.track_running_stats
            and self.num_batches_tracked is not None
        ):
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(
                    self.num_batches_tracked.item()
                )
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self._batch_norm(x)

        if self.step_mode != "m":
            raise ValueError(f"Unsupported step_mode='{self.step_mode}'.")
        if x.dim() != 5:
            raise ValueError(
                f"expected x with shape [T, N, C, H, W], but got {x.shape}!"
            )
        y_shape = [x.shape[0], x.shape[1]]
        y = self._batch_norm(x.flatten(0, 1))
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)


class ChannelShardBatchNorm1d(nn.Module):
    def __init__(self, source: nn.Module, process_group):
        """
        **API Language** - :ref:`中文 <ChannelShardBatchNorm1d-cn>` | :ref:`English <ChannelShardBatchNorm1d-en>`

        ----

        .. _ChannelShardBatchNorm1d-cn:

        * **中文**

        支持通道分片的一维批归一化层。

        :param source: 源 BatchNorm1d 模块
        :type source: nn.Module
        :param process_group: 分布式进程组
        :type process_group: Any

        ----

        .. _ChannelShardBatchNorm1d-en:

        * **English**

        1D batch norm layer with channel sharding.

        :param source: Source BatchNorm1d module
        :type source: nn.Module
        :param process_group: Distributed process group
        :type process_group: Any
        """
        super().__init__()
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if process_group is not None else 0
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
        start, end = _shard_range(source.num_features, self.rank, self.world_size)

        if self.affine:
            self.register_parameter(
                "weight", nn.Parameter(source.weight.detach()[start:end].clone())
            )
            self.register_parameter(
                "bias", nn.Parameter(source.bias.detach()[start:end].clone())
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
            num_batches_tracked = getattr(source, "num_batches_tracked", None)
            if num_batches_tracked is not None:
                self.register_buffer(
                    "num_batches_tracked", num_batches_tracked.detach().clone()
                )
            else:
                self.register_buffer("num_batches_tracked", None)
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        self.training = source.training

    def extra_repr(self) -> str:
        return f"step_mode={self.step_mode}, num_features={self.num_features}"

    def _batch_norm(self, x: torch.Tensor) -> torch.Tensor:
        exponential_average_factor = self.momentum
        if (
            self.training
            and self.track_running_stats
            and self.num_batches_tracked is not None
        ):
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(
                    self.num_batches_tracked.item()
                )
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self._batch_norm(x)

        if self.step_mode != "m":
            raise ValueError(f"Unsupported step_mode='{self.step_mode}'.")
        if x.dim() != 4:
            raise ValueError(f"expected x with shape [T, N, C, L], but got {x.shape}!")
        y_shape = [x.shape[0], x.shape[1]]
        y = self._batch_norm(x.flatten(0, 1))
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)
