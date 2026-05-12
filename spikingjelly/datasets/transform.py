from typing import Union

import numpy as np
import torch


__all__ = ["random_temporal_delete", "RandomTemporalDelete"]


def random_temporal_delete(
    x_seq: Union[torch.Tensor, np.ndarray], T_remain: int, batch_first: bool
):
    r"""
    **API Language:**
    :ref:`中文 <random_temporal_delete-cn>` | :ref:`English <random_temporal_delete-en>`

    ----

    .. _random_temporal_delete-cn:

    * **中文**

    在 `Deep Residual Learning in Spiking Neural Networks <https://arxiv.org/abs/2102.04159>`_ 中使用的随机时间删除数据增强。

    :param x_seq: 一个序列，其 `shape = [T, N, *]`，其中 `T` 是序列长度，`N` 是批次大小
    :type x_seq: Union[torch.Tensor, np.ndarray]

    :param T_remain: 剩余的长度
    :type T_remain: int

    :param batch_first: 如果 `True`，`x_seq` 将被视为 `shape = [N, T, *]`
    :type batch_first: bool

    :return: 长度为 `T_remain` 的序列，通过随机移除 `T - T_remain` 个切片获得
    :rtype: Union[torch.Tensor, np.ndarray]

    :raises ValueError: 当 ``T_remain`` 为负数，或大于当前时间维长度时由 ``numpy.random.choice`` 抛出。

    ----

    .. _random_temporal_delete-en:

    * **English**

    The random temporal delete data augmentation used in `Deep Residual Learning in Spiking Neural Networks <https://arxiv.org/abs/2102.04159>`_.

    :param x_seq: a sequence with `shape = [T, N, *]`, where `T` is the sequence length and `N` is the batch size
    :type x_seq: Union[torch.Tensor, np.ndarray]

    :param T_remain: the remained length
    :type T_remain: int

    :param batch_first: if `True`, `x_seq` will be regarded as `shape = [N, T, *]`
    :type batch_first: bool

    :return: the sequence with length `T_remain`, which is obtained by randomly removing `T - T_remain` slices
    :rtype: Union[torch.Tensor, np.ndarray]

    :raises ValueError: raised by ``numpy.random.choice`` when ``T_remain`` is
        negative or larger than the current time dimension length.

    ----

    * **代码示例 | Example**

    .. code-block:: python

        import torch
        from spikingjelly.datasets import random_temporal_delete

        T = 8
        T_remain = 5
        N = 4
        x_seq = torch.arange(0, N * T).view([N, T])
        print("x_seq=\n", x_seq)
        print(
            "random_temporal_delete(x_seq)=\n",
            random_temporal_delete(x_seq, T_remain, batch_first=True),
        )

    Outputs:

    .. code-block:: shell

        x_seq=
         tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
                [ 8,  9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31]])
        random_temporal_delete(x_seq)=
         tensor([[ 0,  1,  4,  6,  7],
                [ 8,  9, 12, 14, 15],
                [16, 17, 20, 22, 23],
                [24, 25, 28, 30, 31]])
    """
    if batch_first:
        sec_list = np.random.choice(x_seq.shape[1], T_remain, replace=False)
    else:
        sec_list = np.random.choice(x_seq.shape[0], T_remain, replace=False)
    sec_list.sort()
    if batch_first:
        return x_seq[:, sec_list]
    else:
        return x_seq[sec_list]


class RandomTemporalDelete(torch.nn.Module):
    r"""
    **API Language:**
    :ref:`中文 <RandomTemporalDelete-cn>` | :ref:`English <RandomTemporalDelete-en>`

    ----

    .. _RandomTemporalDelete-cn:

    * **中文**

    :func:`random_temporal_delete` 的 ``torch.nn.Module`` 封装。前向传播时会使用构造时给定的
    ``T_remain`` 和 ``batch_first`` 调用 :func:`random_temporal_delete`。

    ----

    .. _RandomTemporalDelete-en:

    * **English**

    A ``torch.nn.Module`` wrapper around :func:`random_temporal_delete`. During
    ``forward``, it calls :func:`random_temporal_delete` with the ``T_remain``
    and ``batch_first`` values provided at construction time.
    """

    def __init__(self, T_remain: int, batch_first: bool):
        r"""
        **API Language:**
        :ref:`中文 <RandomTemporalDelete.__init__-cn>` | :ref:`English <RandomTemporalDelete.__init__-en>`

        ----

        .. _RandomTemporalDelete.__init__-cn:

        * **中文**

        在 `Deep Residual Learning in Spiking Neural Networks <https://arxiv.org/abs/2102.04159>`_ 中使用的随机时间删除数据增强。
        详见 :func:`random_temporal_delete`。

        :param T_remain: 剩余的长度
        :type T_remain: int

        :param batch_first: 如果 `True`，`x_seq` 将被视为 `shape = [N, T, *]`
        :type batch_first: bool

        ----

        .. _RandomTemporalDelete.__init__-en:

        * **English**

        The random temporal delete data augmentation used in `Deep Residual Learning in Spiking Neural Networks <https://arxiv.org/abs/2102.04159>`_.
        Refer to :func:`random_temporal_delete` for more details.

        :param T_remain: the remained length
        :type T_remain: int

        :param batch_first: if `True`, `x_seq` will be regarded as `shape = [N, T, *]`
        :type batch_first: bool
        """
        super().__init__()
        self.T_remain = T_remain
        self.batch_first = batch_first

    def forward(
        self, x_seq: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        r"""
        **API Language:**
        :ref:`中文 <RandomTemporalDelete.forward-cn>` | :ref:`English <RandomTemporalDelete.forward-en>`

        ----

        .. _RandomTemporalDelete.forward-cn:

        * **中文**

        使用当前模块保存的 ``T_remain`` 和 ``batch_first`` 配置，对输入序列执行
        :func:`random_temporal_delete`。

        :param x_seq: 输入序列。其时间维布局由 ``batch_first`` 决定。
        :type x_seq: Union[torch.Tensor, np.ndarray]

        :return: 随机删除时间切片后的序列。
        :rtype: Union[torch.Tensor, np.ndarray]

        :raises ValueError: 当 ``self.T_remain`` 非法时，由
            :func:`random_temporal_delete` 内部的 ``numpy.random.choice`` 抛出

        ----

        .. _RandomTemporalDelete.forward-en:

        * **English**

        Apply :func:`random_temporal_delete` to the input sequence with the
        ``T_remain`` and ``batch_first`` configuration stored in this module.

        :param x_seq: input sequence. The time-dimension layout is determined
            by ``batch_first``.
        :type x_seq: Union[torch.Tensor, np.ndarray]

        :return: sequence after random temporal deletion.
        :rtype: Union[torch.Tensor, np.ndarray]

        :raises ValueError: raised by ``numpy.random.choice`` inside
            :func:`random_temporal_delete` when ``self.T_remain`` is invalid
        """
        return random_temporal_delete(x_seq, self.T_remain, self.batch_first)
