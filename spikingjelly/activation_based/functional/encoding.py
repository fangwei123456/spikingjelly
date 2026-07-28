from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


__all__ = [
    "latency_encode",
    "weighted_phase_encode",
]


def latency_encode(
    x: torch.Tensor,
    T: int,
    enc_function: str = "linear",
    alpha: Optional[float] = None,
) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <latency_encode-cn>` | :ref:`English <latency_encode-en>`

    ----

    .. _latency_encode-cn:

    * **中文**

    执行 LatencyEncoder 的无状态编码公式，将 ``0 <= x <= 1`` 的输入转换为时间维在
    第 0 维的 one-hot spike sequence。该函数不读取或写入 encoder module 的
    ``spike`` 或 ``t`` state。

    :param x: 输入强度张量，调用者负责保证取值范围为 ``[0, 1]``
    :type x: torch.Tensor
    :param T: 编码周期 / 最大类别数
    :type T: int
    :param enc_function: 编码函数，支持 ``"linear"`` 或 ``"log"``
    :type enc_function: str
    :param alpha: 对数编码的缩放系数。为 ``None`` 时由 ``T`` 计算默认值
    :type alpha: Optional[float]
    :return: 脉冲序列，shape 为 ``[T, *x.shape]``
    :rtype: torch.Tensor
    :raises NotImplementedError: ``enc_function`` 不是 ``"linear"`` 或 ``"log"`` 时抛出

    ----

    .. _latency_encode-en:

    * **English**

    Run the stateless LatencyEncoder formula, converting input intensities
    ``0 <= x <= 1`` to a one-hot spike sequence with the time dimension at axis
    0. This function does not read or write encoder module ``spike`` or ``t``
    state.

    :param x: Input intensity tensor; callers are responsible for keeping values
        in ``[0, 1]``
    :type x: torch.Tensor
    :param T: Encoding period / number of one-hot classes
    :type T: int
    :param enc_function: Encoding function, either ``"linear"`` or ``"log"``
    :type enc_function: str
    :param alpha: Logarithmic encoding scale. If ``None``, derive it from ``T``
    :type alpha: Optional[float]
    :return: Spike sequence shaped ``[T, *x.shape]``
    :rtype: torch.Tensor
    :raises NotImplementedError: If ``enc_function`` is neither ``"linear"`` nor
        ``"log"``
    """
    if enc_function == "log":
        if alpha is None:
            alpha = math.exp(T - 1.0) - 1.0
        t_f = (T - 1.0 - torch.log(alpha * x + 1.0)).round().long()
    elif enc_function == "linear":
        t_f = ((T - 1.0) * (1.0 - x)).round().long()
    else:
        raise NotImplementedError

    spike = F.one_hot(t_f, num_classes=T).to(x)
    dims = list(range(spike.ndim - 1))
    dims.insert(0, spike.ndim - 1)
    return spike.permute(dims)


def weighted_phase_encode(x: torch.Tensor, T: int) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <weighted_phase_encode-cn>` | :ref:`English <weighted_phase_encode-en>`

    ----

    .. _weighted_phase_encode-cn:

    * **中文**

    执行 ``WeightedPhaseEncoder`` 的无状态二进制加权相位编码公式。函数返回时间维在
    第 0 维的 spike sequence，不读取或写入 encoder module 的 ``spike`` 或 ``t``
    state。

    :param x: 输入数据，取值范围应为 ``[0, 1 - 2^{-T}]``
    :type x: torch.Tensor
    :param T: 编码相位数 / 周期
    :type T: int
    :return: 脉冲序列，shape 为 ``[T, *x.shape]``
    :rtype: torch.Tensor
    :raises AssertionError: ``x`` 不在 ``[0, 1 - 2^{-T}]`` 范围内时抛出

    ----

    .. _weighted_phase_encode-en:

    * **English**

    Run the stateless binary weighted-phase encoding formula for
    ``WeightedPhaseEncoder``. The function returns a spike sequence with the time
    dimension at axis 0 and does not read or write encoder module ``spike`` or
    ``t`` state.

    :param x: Input data in the range ``[0, 1 - 2^{-T}]``
    :type x: torch.Tensor
    :param T: Number of encoding phases / period
    :type T: int
    :return: Spike sequence shaped ``[T, *x.shape]``
    :rtype: torch.Tensor
    :raises AssertionError: If ``x`` is outside ``[0, 1 - 2^{-T}]``
    """
    assert (x >= 0).all() and (x <= 1 - 2 ** (-T)).all()
    inputs = x.clone()
    spike = torch.empty((T,) + x.shape, device=x.device)
    weight = 0.5
    for i in range(T):
        spike[i] = inputs >= weight
        inputs -= weight * spike[i]
        weight *= 0.5
    return spike
