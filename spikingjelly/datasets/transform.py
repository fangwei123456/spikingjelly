from typing import Union

import numpy as np
import torch


__all__ = ["random_temporal_delete", "RandomTemporalDelete"]


def random_temporal_delete(
    x_seq: Union[torch.Tensor, np.ndarray], T_remain: int, batch_first
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

    def forward(self, x_seq: Union[torch.Tensor, np.ndarray]):
        return random_temporal_delete(x_seq, self.T_remain, self.batch_first)
