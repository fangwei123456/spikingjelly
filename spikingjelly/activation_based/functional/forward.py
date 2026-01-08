from typing import Callable, Union

import torch
from torch import Tensor
import torch.nn as nn


__all__ = [
    "multi_step_forward",
    "t_last_multi_step_forward",
    "chunk_multi_step_forward",
    "seq_to_ann_forward",
    "t_last_seq_to_ann_forward",
]


def multi_step_forward(
    x_seq: Tensor, single_step_module: Union[nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable]
):
    """
    **API Language:**
    :ref:`中文 <multi_step_forward-cn>` | :ref:`English <multi_step_forward-en>`

    ----

    .. _multi_step_forward-cn:

    * **中文**

    :param x_seq: ``shape=[T, batch_size, ...]`` 的输入tensor
    :type x_seq: torch.Tensor

    :param single_step_module: 一个或多个单步模块
    :type single_step_module: Union[nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable]

    :return: ``shape=[T, batch_size, ...]`` 的输出tensor
    :rtype: torch.Tensor

    在有状态的单步模块 ``single_step_module`` 上使用多步前向传播。函数内部将执行一个for循环，
    执行 ``T`` 次单步前向传播。

    ----

    .. _multi_step_forward-en:

    * **English**

    :param x_seq: the input tensor with ``shape=[T, batch_size, ...]``
    :type x_seq: torch.Tensor

    :param single_step_module: one or many single-step modules
    :type single_step_module: Union[nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable]

    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: torch.Tensor

    Applies multi-step forward on stateful ``single_step_module``. The function
    runs a for loop to execute single-step forward for ``T`` times.
    """
    y_seq = []
    if isinstance(single_step_module, (list, tuple, nn.Sequential)):
        for t in range(x_seq.shape[0]):
            x_seq_t = x_seq[t]
            for m in single_step_module:
                x_seq_t = m(x_seq_t)
            y_seq.append(x_seq_t)
    else:
        for t in range(x_seq.shape[0]):
            y_seq.append(single_step_module(x_seq[t]))

    return torch.stack(y_seq)


def t_last_multi_step_forward(
    x_seq: Tensor, single_step_module: Union[nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable]
):
    """
    **API Language:**
    :ref:`中文 <t_last_multi_step_forward-cn>` | :ref:`English <t_last_multi_step_forward-en>`

    ----

    .. _t_last_multi_step_forward-cn:

    * **中文**

    在单步模块 ``single_step_module`` 上使用多步前向传播。

    :param x_seq: ``shape=[batch_size, ..., T]`` 的输入tensor
    :type x_seq: Tensor

    :param single_step_module: 一个或多个单步模块
    :type single_step_module: Union[nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable]

    :return: ``shape=[batch_size, ..., T]`` 的输出tensor
    :rtype: torch.Tensor

    ----

    .. _t_last_multi_step_forward-en:

    * **English**

    Applies multi-step forward on ``single_step_module``.

    :param x_seq: the input tensor with ``shape=[batch_size, ..., T]``
    :type x_seq: torch.Tensor

    :param single_step_module: one or many single-step modules
    :type single_step_module: Union[nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable]

    :return: the output tensor with ``shape=[batch_size, ..., T]``
    :rtype: torch.torch.Tensor
    """
    y_seq = []
    if isinstance(single_step_module, (list, tuple, nn.Sequential)):
        for t in range(x_seq.shape[-1]):
            x_seq_t = x_seq[..., t]
            for m in single_step_module:
                x_seq_t = m(x_seq_t)
            y_seq.append(x_seq_t)
    else:
        for t in range(x_seq.shape[-1]):
            y_seq.append(single_step_module(x_seq[..., t]))

    return torch.stack(y_seq, dim=-1)


def chunk_multi_step_forward(split_size: int, x_seq: Tensor, multi_step_module: nn.Module):
    """
    **API Language:**
    :ref:`中文 <chunk_multi_step_forward-cn>` | :ref:`English <chunk_multi_step_forward-en>`

    ----

    .. _chunk_multi_step_forward-cn:

    * **中文**

    将 ``shape = [T, *]`` 的输入 ``x_seq`` 拆分成多个 ``shape = [split_size, *]`` 的小tensor(若 ``T % split_size != 0``，最后一个tensor的 ``shape[0]`` 会小于 ``split_size``)，然后逐个输入到 ``multi_step_module`` 中，再将输出重新拼接为 ``shape = [split_size, *]``。

    ``chunk_multi_step_forward`` 可以在使用很大的 ``T`` 进行不带梯度的推理(例如ANN2SNN)时使用，能够减少内存消耗量。

    :param split_size: 分割的尺寸
    :type split_size: int

    :param x_seq: 输入
    :type x_seq: torch.Tensor

    :param multi_step_module: 一个使用多步传播模式的网络
    :type multi_step_module: torch.nn.Module

    :return: 输出
    :rtype: torch.Tensor

    ----

    .. _chunk_multi_step_forward-en:

    * **English**

    Splits the input ``x_seq`` with ``shape = [T, *]`` to many tensor chunks with ``shape = [split_size, *]`` (if ``T % split_size != 0``,
    ``shape[0]`` of the last tensor chunk will be smaller than ``split_size``), and sends chunks to ``multi_step_module``,
    then concatenates the outputs to  ``shape = [split_size, *]``.

    ``chunk_multi_step_forward`` can be used for inference with a large ``T`` (e.g., ANN2SNN) to reduce the memory consumption.

    :param split_size: the split size
    :type split_size: int

    :param x_seq: the input tensor
    :type x_seq: Tensor

    :param multi_step_module: a network in multi-step mode
    :type multi_step_module: nn.Module

    :return: the output tensor
    :rtype: Tensor

    ----

    * **代码示例 | Example**

    .. code-block:: python

        import torch
        import torch.nn as nn
        from spikingjelly.activation_based import neuron, layer, functional

        net = nn.Sequential(
            layer.Linear(8, 4),
            neuron.IFNode(step_mode='m'),
            layer.Linear(4, 2),
            neuron.IFNode(step_mode='m'),
        )

        x_seq = torch.rand([1024, 8])
        with torch.no_grad():
            y_seq = functional.chunk_multi_step_forward(16, x_seq, net)
            print(y_seq.shape)
            # torch.Size([1024, 2])
    """
    y_seq = []
    for x in torch.split(x_seq, split_size):
        y_seq.append(multi_step_module(x))
    return torch.cat(y_seq, 0)


def seq_to_ann_forward(
    x_seq: Tensor, stateless_module: Union[nn.Module, list, tuple, nn.Sequential, Callable]
):
    """
    **API Language:**
    :ref:`中文 <seq_to_ann_forward-cn>` | :ref:`English <seq_to_ann_forward-en>`

    ----

    .. _seq_to_ann_forward-cn:

    * **中文**

    使用无状态层进行多步前向传播。输入 ``x_seq`` 的时间和批量维度将被展平，得到 ``[T*batch_size, ...]``
    形状的张量；随后，输入到无状态层中；最后，将输出张量恢复到序列形式 ``[T, batch_size, ...]`` 。

    :param x_seq: ``shape=[T, batch_size, ...]`` 的输入tensor
    :type x_seq: torch.Tensor

    :param stateless_module: 单个或多个无状态网络层
    :type stateless_module: Union[torch.nn.Module, list, tuple, torch.nn.Sequential, Callable]

    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: torch.Tensor

    ----

    .. _seq_to_ann_forward-en:

    * **English**

    Applied forward on stateless modules. Flatten the time and batch dimensions
    of ``x_seq`` so that ``shape=[T*batch_size, ...]``, feed the reshaped tensor
    to the stateless module(s), and reshape the output back to the sequence form
    ``shape=[T, batch_size, ...]``.

    :param x_seq: the input tensor with ``shape=[T, batch_size, ...]``
    :type x_seq: torch.Tensor

    :param stateless_module: one or many stateless modules
    :type stateless_module: Union[torch.nn.Module, list, tuple, torch.nn.Sequential, Callable]

    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: torch.Tensor
    """
    y_shape = [x_seq.shape[0], x_seq.shape[1]]
    y = x_seq.flatten(0, 1)
    if isinstance(stateless_module, (list, tuple, nn.Sequential)):
        for m in stateless_module:
            y = m(y)
    else:
        y = stateless_module(y)
    y_shape.extend(y.shape[1:])
    return y.view(y_shape)


def t_last_seq_to_ann_forward(x_seq: Tensor, stateless_module: Union[nn.Module, list, tuple, nn.Sequential, Callable]):
    """
    **API Language:**
    :ref:`中文 <t_last_seq_to_ann_forward-cn>` | :ref:`English <t_last_seq_to_ann_forward-en>`

    ----

    .. _t_last_seq_to_ann_forward-cn:

    * **中文**

    使用无状态层进行多步前向传播。

    .. note::
        SpikingJelly中默认序列数据形状为 ``shape=[T, batch_size, ...]``。
        但此函数是用于另一种格式，即 ``shape=[batch_size, ..., T]``。
        当使用 ``torch >= 2.0.0`` 时也有并行加速的效果。

    .. note::
        不能用于BN层，因为BN层的running mean/var是输入依赖的。
        对于BN层，只需要输入被当作是 ``shape = [N, C, ..]`` 即可并行计算，需要用户手动实现。

    :param x_seq: ``shape=[batch_size, ..., T]`` 的输入tensor
    :type x_seq: torch.Tensor

    :param stateless_module: 单个或多个无状态网络层
    :type stateless_module: Union[torch.nn.Module, list, tuple, torch.nn.Sequential, Callable]

    :return: the output tensor with ``shape=[batch_size, ..., T]``
    :rtype: torch.Tensor

    ----

    .. _t_last_seq_to_ann_forward-en:

    * **English**

    Applied forward on stateless modules.

    .. admonition:: Note
        :class: note

        The default shape of sequence data in SpikingJelly is ``shape=[T, batch_size, ...]``. However, this function is used for the other data format where  ``shape=[batch_size, ..., T]``. When using ``torch >= 2.0.0``, this function is computing in parallel.

    .. admonition:: Note
        :class: note

        This function can not be applied to wrap BN because its running mean/var depends on inputs. The BN can be computed in parallel as long as the input is regarded as ``shape = [N, C, ..]``, which can be implemented by user manually.

    :param x_seq: the input tensor with ``shape=[batch_size, ..., T]``
    :type x_seq: torch.Tensor

    :param stateless_module: one or many stateless modules
    :type stateless_module: Union[torch.nn.Module, list, tuple, torch.nn.Sequential, Callable]

    :return: the output tensor with ``shape=[batch_size, ..., T]``
    :rtype: torch.Tensor
    """
    if hasattr(torch, 'vmap'):
        vmap_f = torch.vmap(stateless_module, in_dims=-1, out_dims=-1)
        return vmap_f(x_seq)
    else:
        return t_last_multi_step_forward(x_seq, stateless_module)
