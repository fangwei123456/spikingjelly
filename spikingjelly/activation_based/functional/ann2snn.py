from __future__ import annotations

import torch


__all__ = [
    "spikezip_bias_step",
    "spikezip_bias_multi_step",
]


def spikezip_bias_step(
    y: torch.Tensor,
    bias: torch.Tensor | None,
    remaining_steps: int,
    bias_steps: int,
) -> tuple[torch.Tensor, int, bool]:
    r"""
    **API Language** - :ref:`中文 <spikezip_bias_step-cn>` | :ref:`English <spikezip_bias_step-en>`

    ----

    .. _spikezip_bias_step-cn:

    * **中文**

    执行 SpikeZIP bias 的单步释放。函数接收当前输出 ``y``、显式剩余释放步数
    ``remaining_steps`` 和总释放步数 ``bias_steps``；
    若仍需释放 bias，则返回 ``y + bias / bias_steps``、递减后的时间和
    ``True``。否则返回原输出、原时间和 ``False``。

    函数不读取或写入 ``MemoryModule`` memory，不管理 ``step_mode``、
    ``training/eval`` 或子模块状态。

    :param y: 当前输出 tensor
    :type y: torch.Tensor
    :param bias: 可选 bias tensor
    :type bias: torch.Tensor or None
    :param remaining_steps: 剩余 bias 释放步数
    :type remaining_steps: int
    :param bias_steps: bias 总释放步数
    :type bias_steps: int
    :return: ``(y_next, remaining_steps_next, released)``
    :rtype: Tuple[torch.Tensor, int, bool]

    ----

    .. _spikezip_bias_step-en:

    * **English**

    Run one SpikeZIP bias-release step. The function receives the current output
    ``y``, explicit ``remaining_steps``, and total ``bias_steps``. If a bias step is
    still pending, it returns ``y + bias / bias_steps``, the decremented time and
    ``True``. Otherwise it returns the original output, original time and
    ``False``.

    The function does not read or write ``MemoryModule`` memory and does not
    manage ``step_mode``, ``training/eval``, or child-module state.

    :param y: Current output tensor
    :type y: torch.Tensor
    :param bias: Optional bias tensor
    :type bias: torch.Tensor or None
    :param remaining_steps: Remaining bias release steps
    :type remaining_steps: int
    :param bias_steps: Total bias release steps
    :type bias_steps: int
    :return: ``(y_next, remaining_steps_next, released)``
    :rtype: Tuple[torch.Tensor, int, bool]
    """
    if bias is None or remaining_steps <= 0:
        return y, remaining_steps, False
    return (
        y + bias.to(device=y.device, dtype=y.dtype) / bias_steps,
        remaining_steps - 1,
        True,
    )


def spikezip_bias_multi_step(
    y_seq: torch.Tensor,
    bias: torch.Tensor | None,
    remaining_steps: int,
    bias_steps: int,
) -> tuple[torch.Tensor, int, int]:
    r"""
    **API Language** - :ref:`中文 <spikezip_bias_multi_step-cn>` | :ref:`English <spikezip_bias_multi_step-en>`

    ----

    .. _spikezip_bias_multi_step-cn:

    * **中文**

    执行 SpikeZIP bias 的多步释放。函数只在序列前 ``min(T, remaining_steps)`` 步
    加入 ``bias / bias_steps``，并返回更新后的序列、剩余释放时间和本次实际释放
    步数。若没有 bias 或无需释放，则返回原序列且不原地修改输入。

    函数不读取或写入 ``MemoryModule`` memory，不管理 ``step_mode``、
    ``training/eval`` 或子模块状态。

    :param y_seq: 当前输出序列，第 0 维为时间
    :type y_seq: torch.Tensor
    :param bias: 可选 bias tensor
    :type bias: torch.Tensor or None
    :param remaining_steps: 剩余 bias 释放步数
    :type remaining_steps: int
    :param bias_steps: bias 总释放步数
    :type bias_steps: int
    :return: ``(y_seq_next, remaining_steps_next, released_steps)``
    :rtype: Tuple[torch.Tensor, int, int]

    ----

    .. _spikezip_bias_multi_step-en:

    * **English**

    Run multi-step SpikeZIP bias release. The function adds
    ``bias / bias_steps`` only to the first ``min(T, remaining_steps)`` sequence
    steps, and returns the updated sequence, remaining release time, and the
    number of steps released in this call. If there is no bias or no pending
    release, it returns the original sequence without in-place mutation.

    The function does not read or write ``MemoryModule`` memory and does not
    manage ``step_mode``, ``training/eval``, or child-module state.

    :param y_seq: Current output sequence with time at dimension 0
    :type y_seq: torch.Tensor
    :param bias: Optional bias tensor
    :type bias: torch.Tensor or None
    :param remaining_steps: Remaining bias release steps
    :type remaining_steps: int
    :param bias_steps: Total bias release steps
    :type bias_steps: int
    :return: ``(y_seq_next, remaining_steps_next, released_steps)``
    :rtype: Tuple[torch.Tensor, int, int]
    """
    released_steps = min(y_seq.shape[0], remaining_steps)
    if bias is None or released_steps <= 0:
        return y_seq, remaining_steps, 0
    bias = bias.to(device=y_seq.device, dtype=y_seq.dtype)
    y_next = y_seq.clone()
    y_next[:released_steps] = y_next[:released_steps] + bias / bias_steps
    return y_next, remaining_steps - released_steps, released_steps
