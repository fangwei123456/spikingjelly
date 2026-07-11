from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch.distributed as dist

from spikingjelly.activation_based import base


_TENSOR_SHARD_VALIDATOR_ATTR = "_tensor_shard_input_validator"


def _require_even_shard(total: int, world_size: int, name: str):
    if total % world_size != 0:
        raise ValueError(
            f"{name}={total} must be divisible by tensor-parallel world_size={world_size}."
        )


def _shard_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    shard = total // world_size
    start = shard * rank
    return start, start + shard


@dataclass(frozen=True)
class _TensorShardInputValidator:
    shard_dim: int
    logical_dim_size: Optional[int]
    rank: int
    world_size: int
    expected_local_dim_size: Optional[int]

    def __call__(
        self,
        _module: base.MemoryModule,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if args:
            x = args[0]
        elif "x" in kwargs:
            x = kwargs["x"]
        else:
            raise TypeError(
                "Shard validation requires the input tensor as the first positional "
                "argument or the 'x' keyword argument."
            )
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


def _has_tensor_shard_input_validator(module: base.MemoryModule) -> bool:
    validator = getattr(module, _TENSOR_SHARD_VALIDATOR_ATTR, None)
    return isinstance(validator, _TensorShardInputValidator) and any(
        hook == validator for hook in module._forward_pre_hooks.values()
    )


def make_tensor_shard_memory_module(
    source: base.MemoryModule,
    shard_dim: int,
    logical_dim_size: Optional[int] = None,
    process_group: Optional[Any] = None,
) -> base.MemoryModule:
    r"""
    **API Language** - :ref:`中文 <make_tensor_shard_memory_module-cn>` | :ref:`English <make_tensor_shard_memory_module-en>`

    ----

    .. _make_tensor_shard_memory_module-cn:

    * **中文**

    返回 ``source`` 的深拷贝，并通过前向传播预钩子验证输入张量的局部分片
    维度。返回值保留 ``source`` 的具体类型、参数和记忆状态接口。

    输入张量必须作为首个位置参数或 ``x`` 关键字参数传入。必须使用
    ``module(...)`` 调用返回的模块；直接调用 ``module.forward(...)`` 会绕过
    PyTorch 的前向传播钩子及分片验证。若 ``source`` 已包含该验证钩子，则
    原样返回 ``source``。

    :param source: 待复制并添加分片验证的有状态模块
    :type source: base.MemoryModule
    :param shard_dim: 输入张量中局部分片所在的维度
    :type shard_dim: int
    :param logical_dim_size: 分片前对应逻辑维度的大小；为 ``None`` 时不验证局部大小
    :type logical_dim_size: Optional[int]
    :param process_group: 用于计算局部大小的张量并行进程组；为 ``None`` 时按单进程处理
    :type process_group: Optional[Any]
    :return: 带局部分片输入验证钩子的有状态模块
    :rtype: base.MemoryModule
    :raises TypeError: 前向调用未通过首个位置参数或 ``x`` 关键字参数提供输入张量
    :raises ValueError: 逻辑维度无法被进程数整除，或前向输入的分片维度或大小无效

    ----

    .. _make_tensor_shard_memory_module-en:

    * **English**

    Return a deep copy of ``source`` with a forward pre-hook that validates the
    local-shard dimension of its input tensor. The returned module preserves the
    concrete type, parameters, and memory-state interface of ``source``.

    Pass the input tensor as the first positional argument or the ``x`` keyword
    argument. Invoke the returned module through ``module(...)``. Calling
    ``module.forward(...)`` directly bypasses PyTorch forward hooks and shard
    validation. If ``source`` already has this validation hook, ``source`` is
    returned unchanged.

    :param source: Stateful module to copy and equip with shard validation
    :type source: base.MemoryModule
    :param shard_dim: Input dimension containing the local shard
    :type shard_dim: int
    :param logical_dim_size: Corresponding logical dimension size before sharding;
        ``None`` disables local-size validation
    :type logical_dim_size: Optional[int]
    :param process_group: Tensor-parallel process group used to compute the local
        size; ``None`` uses single-process semantics
    :type process_group: Optional[Any]
    :return: Stateful module with local-shard input validation
    :rtype: base.MemoryModule
    :raises TypeError: If the forward call does not provide an input tensor as
        the first positional argument or the ``x`` keyword argument
    :raises ValueError: If the logical dimension is not evenly shardable or the
        forward input has an invalid shard dimension or local size
    """
    if _has_tensor_shard_input_validator(source):
        return source

    module = copy.deepcopy(source)
    step_mode = getattr(module, "step_mode", "s")
    effective_shard_dim = 1 if shard_dim == 2 and step_mode == "s" else shard_dim
    rank = dist.get_rank(process_group) if process_group is not None else 0
    world_size = dist.get_world_size(process_group) if process_group is not None else 1
    expected_local_dim_size = None
    if logical_dim_size is not None:
        _require_even_shard(logical_dim_size, world_size, "logical_dim_size")
        start, end = _shard_range(logical_dim_size, rank, world_size)
        expected_local_dim_size = end - start

    validator = _TensorShardInputValidator(
        shard_dim=effective_shard_dim,
        logical_dim_size=logical_dim_size,
        rank=rank,
        world_size=world_size,
        expected_local_dim_size=expected_local_dim_size,
    )
    module.register_forward_pre_hook(validator, prepend=True, with_kwargs=True)
    setattr(module, _TENSOR_SHARD_VALIDATOR_ATTR, validator)
    return module


def TensorShardMemoryModule(
    source: base.MemoryModule,
    shard_dim: int,
    logical_dim_size: Optional[int] = None,
    process_group: Optional[Any] = None,
) -> base.MemoryModule:
    """Deprecated callable alias for :func:`make_tensor_shard_memory_module`."""
    warnings.warn(
        "TensorShardMemoryModule is deprecated and will be removed in a future "
        "version. Use make_tensor_shard_memory_module instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return make_tensor_shard_memory_module(
        source=source,
        shard_dim=shard_dim,
        logical_dim_size=logical_dim_size,
        process_group=process_group,
    )
