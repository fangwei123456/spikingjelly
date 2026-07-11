from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from spikingjelly.activation_based import base


def _require_even_shard(total: int, world_size: int, name: str):
    if total % world_size != 0:
        raise ValueError(
            f"{name}={total} must be divisible by tensor-parallel world_size={world_size}."
        )


def _shard_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    shard = total // world_size
    start = shard * rank
    return start, start + shard


class TensorShardMemoryModule(base.MemoryModule):
    def __init__(
        self,
        source: base.MemoryModule,
        shard_dim: int,
        logical_dim_size: Optional[int] = None,
        process_group=None,
    ):
        """
        **API Language** - :ref:`中文 <TensorShardMemoryModule-cn>` | :ref:`English <TensorShardMemoryModule-en>`

        ----

        .. _TensorShardMemoryModule-cn:

        * **中文**

        支持张量并行分片的内存模块基类。

        :param source: 源 MemoryModule
        :type source: base.MemoryModule
        :param shard_dim: 切分维度
        :type shard_dim: int
        :param logical_dim_size: 逻辑维度大小（每一维的大小），用于验证分片正确性
        :type logical_dim_size: Optional[int]
        :param process_group: 分布式进程组
        :type process_group: Any

        ----

        .. _TensorShardMemoryModule-en:

        * **English**

        Base memory module supporting tensor parallel sharding.

        :param source: Source MemoryModule
        :type source: base.MemoryModule
        :param shard_dim: Dimension along which to shard
        :type shard_dim: int
        :param logical_dim_size: Logical dimension size, used to validate sharding
        :type logical_dim_size: Optional[int]
        :param process_group: Distributed process group
        :type process_group: Any
        """
        super().__init__()
        self.inner = copy.deepcopy(source)
        self.step_mode = getattr(self.inner, "step_mode", "s")
        self.shard_dim = 1 if shard_dim == 2 and self.step_mode == "s" else shard_dim
        self.logical_dim_size = logical_dim_size
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if process_group is not None else 0
        self.world_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        self.expected_local_dim_size = None
        if logical_dim_size is not None:
            _require_even_shard(logical_dim_size, self.world_size, "logical_dim_size")
            start, end = _shard_range(logical_dim_size, self.rank, self.world_size)
            self.expected_local_dim_size = end - start
        if hasattr(self.inner, "backend"):
            self.backend = self.inner.backend

    @property
    def supported_backends(self):
        supported = getattr(self.inner, "supported_backends", None)
        if supported is None:
            return ("torch",)
        return supported

    @property
    def store_v_seq(self):
        return getattr(self.inner, "store_v_seq", False)

    def reset(self):
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def extra_repr(self) -> str:
        return (
            f"shard_dim={self.shard_dim}, logical_dim_size={self.logical_dim_size}, "
            f"world_size={self.world_size}"
        )

    def forward(self, x: torch.Tensor):
        shard_dim = self.shard_dim if self.shard_dim >= 0 else x.dim() + self.shard_dim
        if shard_dim < 0 or shard_dim >= x.dim():
            raise ValueError(
                f"shard_dim={shard_dim} (from self.shard_dim={self.shard_dim}) "
                f"is invalid for input with shape {tuple(x.shape)}."
            )
        if (
            self.expected_local_dim_size is not None
            and x.shape[shard_dim] != self.expected_local_dim_size
        ):
            raise ValueError(
                f"Expected local shard size {self.expected_local_dim_size} on dim {shard_dim}, "
                f"but got input shape {tuple(x.shape)}."
            )
        return self.inner(x)
