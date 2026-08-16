"""Temporal layout operations shared by MCore SNN language models."""

from typing import Any

import torch

from ... import base


def pack_time_batch(hidden: torch.Tensor) -> torch.Tensor:
    r"""
    **API Language** - 中文 | English

    **中文**

    将 ``[T, B, S, H]`` 张量转换为 MCore 使用的 ``[S, T*B, H]`` 张量。
    时间步仅被并入 batch 维，不会并入 token 维。

    :param hidden: 四维时间优先张量。
    :type hidden: torch.Tensor
    :return: MCore tensor envelope。
    :rtype: torch.Tensor
    :raises ValueError: ``hidden`` 不是四维张量。

    **English**

    Convert a ``[T, B, S, H]`` tensor to MCore's ``[S, T*B, H]``
    envelope. Time steps are folded only into the batch dimension, never the
    token dimension.

    :param hidden: Four-dimensional time-major tensor.
    :type hidden: torch.Tensor
    :return: MCore tensor envelope.
    :rtype: torch.Tensor
    :raises ValueError: If ``hidden`` is not four-dimensional.
    """
    if hidden.ndim != 4:
        raise ValueError(f"expected [T, B, S, H], got shape {tuple(hidden.shape)}")
    time_steps, batch_size, sequence_length, hidden_size = hidden.shape
    return (
        hidden.permute(2, 0, 1, 3)
        .reshape(sequence_length, time_steps * batch_size, hidden_size)
        .contiguous()
    )


def unpack_time_batch(hidden: torch.Tensor, time_steps: int) -> torch.Tensor:
    r"""
    **API Language** - 中文 | English

    **中文**

    将 ``[S, T*B, H]`` MCore tensor envelope 还原为 ``[T, B, S, H]``。

    :param hidden: 三维 MCore tensor envelope。
    :type hidden: torch.Tensor
    :param time_steps: SNN simulation-step 数量 ``T``，必须为正数并整除第二维。
    :type time_steps: int
    :return: 时间优先张量。
    :rtype: torch.Tensor
    :raises ValueError: 输入不是三维、``time_steps`` 非正或不能整除第二维。

    **English**

    Restore a ``[S, T*B, H]`` MCore tensor envelope to ``[T, B, S, H]``.

    :param hidden: Three-dimensional MCore tensor envelope.
    :type hidden: torch.Tensor
    :param time_steps: Positive SNN simulation-step count ``T`` that divides
        the second dimension.
    :type time_steps: int
    :return: Time-major tensor.
    :rtype: torch.Tensor
    :raises ValueError: If the input is not three-dimensional or its folded
        batch dimension is incompatible with ``time_steps``.
    """
    if hidden.ndim != 3:
        raise ValueError(f"expected [S, T*B, H], got shape {tuple(hidden.shape)}")
    if time_steps <= 0 or hidden.shape[1] % time_steps:
        raise ValueError(
            f"time_steps={time_steps} must be positive and divide folded batch "
            f"size {hidden.shape[1]}"
        )
    sequence_length, folded_batch_size, hidden_size = hidden.shape
    batch_size = folded_batch_size // time_steps
    return hidden.reshape(sequence_length, time_steps, batch_size, hidden_size).permute(
        1, 2, 0, 3
    )


def _reduce_time_batch(
    value: torch.Tensor, time_steps: int, reduction: str
) -> torch.Tensor:
    if value.shape[0] % time_steps:
        raise ValueError("Batch dimension must be divisible by time_steps.")
    temporal = value.reshape(time_steps, value.shape[0] // time_steps, *value.shape[1:])
    if reduction == "sum":
        return temporal.sum(0)
    if reduction == "mean":
        return temporal.mean(0)
    raise ValueError(f"Unsupported temporal reduction: {reduction!r}.")


def run_functional_sequence(
    module: base.MemoryModule,
    inputs: tuple[torch.Tensor, ...],
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    r"""
    **API Language** - 中文 | English

    **中文**

    从 ``module`` 自身注册 memory 的 reset value 执行 functional forward，
    返回输出并丢弃最终状态。该函数不会读写模块的当前 memory，适用于每个
    microbatch 都从零状态开始的完整 ``T`` 窗口。

    :param module: 直接拥有待执行状态的 ``MemoryModule``。
    :type module: spikingjelly.activation_based.base.MemoryModule
    :param inputs: 传给 ``module.functional_forward`` 的输入元组。
    :type inputs: tuple[torch.Tensor, ...]
    :param kwargs: 传给 ``module.functional_forward`` 的关键字参数。
    :type kwargs: Any
    :return: functional forward 的输出元组。
    :rtype: tuple[torch.Tensor, ...]

    **English**

    Run functional forward from the reset values registered by ``module`` and
    discard the final states. The current module memories are neither read nor
    written, matching complete ``T`` windows that restart at every microbatch.

    :param module: ``MemoryModule`` that directly owns the executed states.
    :type module: spikingjelly.activation_based.base.MemoryModule
    :param inputs: Input tuple passed to ``module.functional_forward``.
    :type inputs: tuple[torch.Tensor, ...]
    :param kwargs: Keyword arguments passed to ``module.functional_forward``.
    :type kwargs: Any
    :return: Output tuple from functional forward.
    :rtype: tuple[torch.Tensor, ...]
    """
    reset_states = tuple(
        module.get_reset_value(name) for name, _ in module.named_memories()
    )
    outputs, _ = module.functional_forward(inputs, reset_states, **kwargs)
    return outputs
