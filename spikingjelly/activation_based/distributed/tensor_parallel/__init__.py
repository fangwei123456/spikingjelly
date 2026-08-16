r"""
**API Language** - 中文 | English

**中文：** 提供 PyTorch 原生并行尚未覆盖的 SNN 通道分片卷积和归一化。

**English:** Channel-sharded convolution and normalization for SNN modules not
covered by native PyTorch parallelism.
"""

from .channel import (
    ChannelShardBatchNorm1d,
    ChannelShardBatchNorm2d,
    ChannelShardConv1d,
    ChannelShardConv2d,
)

__all__ = [
    "ChannelShardBatchNorm1d",
    "ChannelShardBatchNorm2d",
    "ChannelShardConv1d",
    "ChannelShardConv2d",
]
