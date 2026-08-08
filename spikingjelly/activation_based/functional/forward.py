from typing import Callable, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "multi_step_forward",
    "t_last_multi_step_forward",
    "chunk_multi_step_forward",
    "seq_to_ann_forward",
    "t_last_seq_to_ann_forward",
]


def _apply_modules(x, modules):
    args = x if isinstance(x, tuple) else (x,)
    if isinstance(modules, (list, tuple)):
        for module in modules:
            x = module(*args)
            args = (x,)
        return x
    return modules(*args)


def _multi_step_forward(x_seq, single_step_module, time_dim):
    y_seq = [
        _apply_modules(x_seq.select(time_dim, t), single_step_module)
        for t in range(x_seq.shape[time_dim])
    ]
    return torch.stack(y_seq, dim=time_dim)


def multi_step_forward(
    x_seq: Tensor,
    single_step_module: Union[
        nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable
    ],
) -> Tensor:
    """
    **API Language** - :ref:`中文 <multi_step_forward-cn>` | :ref:`English <multi_step_forward-en>`

    ----

    .. _multi_step_forward-cn:

    * **中文**

    在单步模块 ``single_step_module`` 上使用多步前向传播。函数内部将执行一个for循环，
    执行 ``T`` 次单步前向传播。若 ``single_step_module`` 为多个模块，则每个时间步都会按顺序依次执行这些模块。

    :param x_seq: ``shape=[T, batch_size, ...]`` 的输入tensor
    :type x_seq: torch.Tensor

    :param single_step_module: 一个或多个单步模块
    :type single_step_module: Union[nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable]

    :return: ``shape=[T, batch_size, ...]`` 的输出tensor
    :rtype: torch.Tensor

    :raises Exception: 任何底层模块在某个时间步前向传播时抛出的异常都会原样向上传播

    ----

    .. _multi_step_forward-en:

    * **English**

    Applies multi-step forward on ``single_step_module``. The function runs a
    for loop to execute single-step forward for ``T`` times. If
    ``single_step_module`` contains multiple modules, they are applied
    sequentially at each time-step.

    :param x_seq: the input tensor with ``shape=[T, batch_size, ...]``
    :type x_seq: torch.Tensor

    :param single_step_module: one or many single-step modules
    :type single_step_module: Union[nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable]

    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: torch.Tensor

    :raises Exception: Any exception raised by an underlying module at any time step is propagated unchanged
    """
    return _multi_step_forward(x_seq, single_step_module, 0)


def t_last_multi_step_forward(
    x_seq: Tensor,
    single_step_module: Union[
        nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable
    ],
) -> Tensor:
    """
    **API Language** - :ref:`中文 <t_last_multi_step_forward-cn>` | :ref:`English <t_last_multi_step_forward-en>`

    ----

    .. _t_last_multi_step_forward-cn:

    * **中文**

    在单步模块 ``single_step_module`` 上使用多步前向传播。

    此函数适用于时间维位于最后一维的序列张量，即 ``shape=[batch_size, ..., T]``。
    它会沿最后一维逐个时间步取出切片，并在每个时间步顺序执行单步模块。

    :param x_seq: ``shape=[batch_size, ..., T]`` 的输入tensor
    :type x_seq: Tensor

    :param single_step_module: 一个或多个单步模块
    :type single_step_module: Union[nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable]

    :return: ``shape=[batch_size, ..., T]`` 的输出tensor
    :rtype: torch.Tensor

    :raises Exception: 任何底层模块在某个时间步前向传播时抛出的异常都会原样向上传播

    ----

    .. _t_last_multi_step_forward-en:

    * **English**

    Apply multi-step forward on ``single_step_module``.

    This helper is intended for sequence tensors whose time axis is the last
    dimension, i.e. ``shape=[batch_size, ..., T]``. It slices along the last
    dimension and applies the single-step module(s) at each time step.

    :param x_seq: the input tensor with ``shape=[batch_size, ..., T]``
    :type x_seq: torch.Tensor

    :param single_step_module: one or many single-step modules
    :type single_step_module: Union[nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable]

    :return: the output tensor with ``shape=[batch_size, ..., T]``
    :rtype: torch.Tensor

    :raises Exception: Any exception raised by an underlying module at any time step is propagated unchanged
    """
    return _multi_step_forward(x_seq, single_step_module, -1)


def chunk_multi_step_forward(
    split_size: int, x_seq: Tensor, multi_step_module: nn.Module
) -> Tensor:
    """
    **API Language** - :ref:`中文 <chunk_multi_step_forward-cn>` | :ref:`English <chunk_multi_step_forward-en>`

    ----

    .. _chunk_multi_step_forward-cn:

    * **中文**

    将 ``shape = [T, *]`` 的输入 ``x_seq`` 拆分成多个 ``shape = [split_size, *]`` 的小tensor(若 ``T % split_size != 0``，最后一个tensor的 ``shape[0]`` 会小于 ``split_size``)，然后逐个输入到 ``multi_step_module`` 中，再沿着 ``dim=0`` 将输出重新拼接，因此输出的首维长度仍为 ``T``。

    ``chunk_multi_step_forward`` 可以在使用很大的 ``T`` 进行不带梯度的推理(例如ANN2SNN)时使用，能够减少内存消耗量。

    :param split_size: 分割的尺寸
    :type split_size: int

    :param x_seq: 输入
    :type x_seq: torch.Tensor

    :param multi_step_module: 一个使用多步传播模式的网络
    :type multi_step_module: torch.nn.Module

    :return: 输出
    :rtype: torch.Tensor

    :raises Exception: 任何 ``multi_step_module`` 在某个分块上的前向传播异常都会原样向上传播

    ----

    .. _chunk_multi_step_forward-en:

    * **English**

    Splits the input ``x_seq`` with ``shape = [T, *]`` to many tensor chunks with ``shape = [split_size, *]`` (if ``T % split_size != 0``,
    ``shape[0]`` of the last tensor chunk will be smaller than ``split_size``), and sends chunks to ``multi_step_module``,
    then concatenates the outputs back along ``dim=0``, so the output keeps the original leading length ``T``.

    ``chunk_multi_step_forward`` can be used for inference with a large ``T`` (e.g., ANN2SNN) to reduce the memory consumption.

    :param split_size: the split size
    :type split_size: int

    :param x_seq: the input tensor
    :type x_seq: Tensor

    :param multi_step_module: a network in multi-step mode
    :type multi_step_module: nn.Module

    :return: the output tensor
    :rtype: Tensor

    :raises Exception: Any exception raised by ``multi_step_module`` on a chunk is propagated unchanged

    ----

    * **代码示例 | Example**

    .. code-block:: python

        import torch
        import torch.nn as nn
        from spikingjelly.activation_based import neuron, layer, functional

        net = nn.Sequential(
            layer.Linear(8, 4),
            neuron.IFNode(step_mode="m"),
            layer.Linear(4, 2),
            neuron.IFNode(step_mode="m"),
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
    x_seq: Union[Tensor, tuple[Tensor, ...]],
    stateless_module: Union[nn.Module, list, tuple, nn.Sequential, Callable],
) -> Union[Tensor, tuple[Tensor, ...]]:
    """
    **API Language** - :ref:`中文 <seq_to_ann_forward-cn>` | :ref:`English <seq_to_ann_forward-en>`

    ----

    .. _seq_to_ann_forward-cn:

    * **中文**

    使用无状态层进行多步前向传播。输入 ``x_seq`` 的时间和批量维度将被展平，得到 ``[T*batch_size, ...]``
    形状的张量；随后，输入到无状态层中；最后，将输出张量恢复到序列形式 ``[T, batch_size, ...]`` 。

    ``x_seq`` 也可以是 tensor tuple，例如同时输入池化值与池化索引。此时每个
    tensor 的形状均为 ``shape=[T, batch_size, ...]``，且 ``T`` 和 ``batch_size``
    必须相同；每个 tensor 的时间和批量维度会被分别展平，展平后的 tensor 作为
    位置参数一起输入到第一个无状态层中。若给出多个无状态层，则后续的层依次接收
    前一层的输出作为单个参数。因此第一个无状态层必须能够接收与tuple长度相同数量的
    位置参数；``torch.nn.Sequential`` 只接收单个输入，故不能作为第一层接收长度
    大于1的tuple。

    :param x_seq: ``shape=[T, batch_size, ...]`` 的输入tensor，或多个此类tensor组成的tuple
    :type x_seq: Union[torch.Tensor, tuple[torch.Tensor, ...]]

    :param stateless_module: 单个或多个无状态网络层
    :type stateless_module: Union[torch.nn.Module, list, tuple, torch.nn.Sequential, Callable]

    :return: ``shape=[T, batch_size, ...]`` 的输出tensor；若底层模块返回
        tensor tuple，则分别恢复每个 tensor 的时间维和批量维
    :rtype: Union[torch.Tensor, tuple[torch.Tensor, ...]]

    :raises ValueError: 当tuple中tensor的 ``[T, batch_size]`` 前两维不一致时抛出

    :raises Exception: 任何底层无状态模块在前向传播时抛出的异常都会原样向上传播

    ----

    .. _seq_to_ann_forward-en:

    * **English**

    Applied forward on stateless modules. Flatten the time and batch dimensions
    of ``x_seq`` so that ``shape=[T*batch_size, ...]``, feed the reshaped tensor
    to the stateless module(s), and reshape the output back to the sequence form
    ``shape=[T, batch_size, ...]``.

    ``x_seq`` can also be a tuple of tensors, e.g., pooled values together with
    pooling indices. In this case, every tensor must have
    ``shape=[T, batch_size, ...]`` with the same ``T`` and ``batch_size``; the
    time and batch dimensions of each tensor are flattened separately, and the
    flattened tensors are fed to the first stateless module as positional
    arguments. If several stateless modules are given, each subsequent module
    receives the previous module's output as a single argument. The first
    stateless module must therefore accept as many positional arguments as the
    tuple holds; ``torch.nn.Sequential`` accepts a single input only, so it can
    not be the first module for a tuple of more than one tensor.

    :param x_seq: the input tensor with ``shape=[T, batch_size, ...]``, or a tuple of such tensors
    :type x_seq: Union[torch.Tensor, tuple[torch.Tensor, ...]]

    :param stateless_module: one or many stateless modules
    :type stateless_module: Union[torch.nn.Module, list, tuple, torch.nn.Sequential, Callable]

    :return: the output tensor with ``shape=[T, batch_size, ...]``; if the
        underlying module returns a tuple of tensors, each tensor has its time
        and batch dimensions restored
    :rtype: Union[torch.Tensor, tuple[torch.Tensor, ...]]

    :raises ValueError: if the tensors in ``x_seq`` do not share the same ``[T, batch_size]`` leading dimensions

    :raises Exception: Any exception raised by an underlying stateless module is propagated unchanged
    """
    if isinstance(x_seq, tuple):
        leading_shape = x_seq[0].shape[:2]
        if any(item.shape[:2] != leading_shape for item in x_seq[1:]):
            raise ValueError(
                "expected all tensors in x_seq to share the same "
                "[T, batch_size] leading dimensions, but got tensors with "
                f"shapes {[tuple(item.shape) for item in x_seq]}!"
            )
        time_steps, batch_size = leading_shape
        x = tuple(item.flatten(0, 1) for item in x_seq)
    else:
        time_steps, batch_size = x_seq.shape[:2]
        x = x_seq.flatten(0, 1)
    y = _apply_modules(x, stateless_module)
    if isinstance(y, tuple):
        return tuple(item.unflatten(0, (time_steps, batch_size)) for item in y)
    return y.unflatten(0, (time_steps, batch_size))


def t_last_seq_to_ann_forward(
    x_seq: Tensor,
    stateless_module: Union[nn.Module, list, tuple, nn.Sequential, Callable],
) -> Union[Tensor, tuple[Tensor, ...]]:
    """
    **API Language** - :ref:`中文 <t_last_seq_to_ann_forward-cn>` | :ref:`English <t_last_seq_to_ann_forward-en>`

    ----

    .. _t_last_seq_to_ann_forward-cn:

    * **中文**

    使用无状态层进行多步前向传播。

    .. note::
        SpikingJelly中默认序列数据形状为 ``shape=[T, batch_size, ...]``。
        但此函数是用于另一种格式，即 ``shape=[batch_size, ..., T]``。
        此函数使用 ``torch.vmap`` 沿最后一维执行单步前向传播。``list`` 和
        ``tuple`` 中的模块会被逐项调用；``nn.Sequential`` 作为容器调用。

    .. note::
        不能用于BN层，因为BN层的running mean/var是输入依赖的。
        对于BN层，只需要输入被当作是 ``shape = [N, C, ..]`` 即可并行计算，需要用户手动实现。

    :param x_seq: ``shape=[batch_size, ..., T]`` 的输入tensor
    :type x_seq: torch.Tensor

    :param stateless_module: 单个或多个无状态网络层
    :type stateless_module: Union[torch.nn.Module, list, tuple, torch.nn.Sequential, Callable]

    :return: ``shape=[batch_size, ..., T]`` 的输出tensor；若底层模块返回
        tensor tuple，则分别恢复每个 tensor 的时间维
    :rtype: Union[torch.Tensor, tuple[torch.Tensor, ...]]

    :raises Exception: 任何底层无状态模块在前向传播时抛出的异常都会原样向上传播

    ----

    .. _t_last_seq_to_ann_forward-en:

    * **English**

    Applied forward on stateless modules.

    .. admonition:: Note
        :class: note

        The default shape of sequence data in SpikingJelly is
        ``shape=[T, batch_size, ...]``. However, this function is used for the
        other data format where ``shape=[batch_size, ..., T]``. This function
        uses ``torch.vmap`` to apply the single-step forward pass over the last
        dimension. Modules in a ``list`` or ``tuple`` are called in order, while
        an ``nn.Sequential`` is called as a container.

    .. admonition:: Note
        :class: note

        This function can not be applied to wrap BN because its running mean/var
        depends on inputs. The BN can be computed in parallel as long as the
        input is regarded as ``shape = [N, C, ..]``, which can be implemented
        by user manually.

    :param x_seq: the input tensor with ``shape=[batch_size, ..., T]``
    :type x_seq: torch.Tensor

    :param stateless_module: one or many stateless modules
    :type stateless_module: Union[torch.nn.Module, list, tuple, torch.nn.Sequential, Callable]

    :return: the output tensor with ``shape=[batch_size, ..., T]``; if the
        underlying module returns a tuple of tensors, each tensor has its time
        dimension restored
    :rtype: Union[torch.Tensor, tuple[torch.Tensor, ...]]

    :raises Exception: Any exception raised by an underlying stateless module is propagated unchanged
    """
    return torch.vmap(
        lambda x: _apply_modules(x, stateless_module), in_dims=-1, out_dims=-1
    )(x_seq)
