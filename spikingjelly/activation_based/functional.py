import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable

from . import neuron, base

from torch import Tensor

def reset_net(net: nn.Module):
    """
    * :ref:`API in English <reset_net-en>`

    .. _reset_net-cn:

    :param net: 任何属于 ``nn.Module`` 子类的网络

    :return: None

    将网络的状态重置。做法是遍历网络中的所有 ``Module``，若 ``m `` 为 ``base.MemoryModule`` 函数或者是拥有 ``reset()`` 方法，则调用 ``m.reset()``。

    * :ref:`中文API <reset_net-cn>`

    .. _reset_net-en:

    :param net: Any network inherits from ``nn.Module``

    :return: None

    Reset the whole network.  Walk through every ``Module`` as ``m``, and call ``m.reset()`` if this ``m`` is ``base.MemoryModule`` or ``m`` has ``reset()``.
    """
    for m in net.modules():
        if hasattr(m, 'reset'):
            if not isinstance(m, base.MemoryModule):
                logging.warning(f'Trying to call `reset()` of {m}, which is not spikingjelly.activation_based.base'
                                f'.MemoryModule')
            m.reset()

def set_step_mode(net: nn.Module, step_mode: str):
    """
    * :ref:`API in English <set_step_mode-en>`

    .. _set_step_mode-cn:

    :param net: 一个神经网络
    :type net: nn.Module
    :param step_mode: 's' (单步模式) 或 'm' (多步模式)
    :type step_mode: str
    :return: None

    将 ``net`` 中所有模块的步进模式设置为 ``step_mode`` 。

    .. note::

        :class:`spikingjelly.activation_based.layer.StepModeContainer`, :class:`spikingjelly.activation_based.layer.ElementWiseRecurrentContainer`,
        :class:`spikingjelly.activation_based.layer.LinearRecurrentContainer` 的子模块（不包含包装器本身）的 ``step_mode`` 不会被改变。


    * :ref:`中文 API <set_step_mode-cn>`

    .. _set_step_mode-en:

    :param net: a network
    :type net: nn.Module
    :param step_mode: 's' (single-step) or 'm' (multi-step)
    :type step_mode: str
    :return: None

    Set ``step_mode`` for all modules in ``net``.

    .. admonition:: Note
        :class: note

        The submodule (not including the container itself) of :class:`spikingjelly.activation_based.layer.StepModeContainer`, :class:`spikingjelly.activation_based.layer.ElementWiseRecurrentContainer`,
        :class:`spikingjelly.activation_based.layer.LinearRecurrentContainer` will not be changed.
    """
    from .layer import StepModeContainer, ElementWiseRecurrentContainer, LinearRecurrentContainer

    keep_step_mode_instance = (
        StepModeContainer, ElementWiseRecurrentContainer, LinearRecurrentContainer
    )
    # step_mode of sub-modules in keep_step_mode_instance will not be changed

    keep_step_mode_containers = []
    for m in net.modules():
        if isinstance(m, keep_step_mode_instance):
            keep_step_mode_containers.append(m)

    for m in net.modules():
        if hasattr(m, 'step_mode'):
            is_contained = False
            for container in keep_step_mode_containers:
                if not isinstance(m, keep_step_mode_instance) and m in container.modules():
                    is_contained = True
                    break
            if is_contained:
                # this function should not change step_mode of submodules in keep_step_mode_containers
                pass
            else:
                if not isinstance(m, (base.StepModule)):
                    logging.warning(f'Trying to set the step mode for {m}, which is not spikingjelly.activation_based'
                                    f'.base.StepModule')
                m.step_mode = step_mode


def set_backend(net: nn.Module, backend: str, instance: object or tuple = (nn.Module, )):
    """
    * :ref:`API in English <set_backend-en>`

    .. _set_backend-cn:

    :param net: 一个神经网络
    :type net: nn.Module
    :param backend: 使用哪个后端
    :type backend: str
    :param instance: 类型为 ``instance`` 的模块后端会被改变
    :type instance: nn.Module or tuple[nn.Module]
    :return: None

    将 ``net`` 中 所有类型为 ``instance`` 的模块后端更改为 ``backend``

    * :ref:`中文 API <set_backend-cn>`

    .. _set_backend-en:

    :param net: a network
    :type net: nn.Module
    :param backend: the backend to be set
    :type backend: str
    :param instance: the backend of which instance will be changed
    :type instance: nn.Module or tuple[nn.Module]
    :return: None

    Sets backends of all modules whose instance is ``instance`` in ``net`` to ``backend``
    """
    for m in net.modules():
        if isinstance(m, instance):
            if hasattr(m, 'backend'):
                if not isinstance(m, base.MemoryModule):
                    logging.warning(
                        f'Trying to set the backend for {m}, which is not spikingjelly.activation_based.base.MemoryModule')
                if backend in m.supported_backends:
                    m.backend = backend
                else:
                    logging.warning(f'{m} does not supports for backend={backend}. It will still use backend={m.backend}.')


def detach_net(net: nn.Module):
    """
    * :ref:`API in English <detach_net-en>`

    .. _detach_net-cn:

    :param net: 任何属于 ``nn.Module`` 子类的网络

    :return: None

    将网络与之前的时间步的计算图断开。做法是遍历网络中的所有 ``Module``，若 ``m `` 为 ``base.MemoryModule`` 函数或者是拥有 ``detach()`` 方法，则调用 ``m.detach()``。

    * :ref:`中文API <detach_net-cn>`

    .. _detach_net-en:

    :param net: Any network inherits from ``nn.Module``

    :return: None

    Detach the computation graph of the whole network from previous time-steps.  Walk through every ``Module`` as ``m``, and call ``m.detach()`` if this ``m`` is ``base.MemoryModule`` or ``m`` has ``detach()``.
    """
    for m in net.modules():
        if hasattr(m, 'detach'):
            if not isinstance(m, base.MemoryModule):
                logging.warning(f'Trying to call `detach()` of {m}, which is not spikingjelly.activation_based.base'
                                f'.MemoryModule')
            m.detach()

def spike_similar_loss(spikes: Tensor, labels: Tensor, kernel_type='linear', loss_type='mse', *args):
    """
    * :ref:`API in English <spike_similar_loss-en>`

    .. _spike_similar_loss-cn:

    :param spikes: shape=[N, M, T]，N个数据生成的脉冲
    :param labels: shape=[N, C]，N个数据的标签，\ ``labels[i][k] == 1``\ 表示数据i属于第k类，反之亦然，允许多标签
    :param str kernel_type: 使用内积来衡量两个脉冲之间的相似性，\ ``kernel_type``\ 是计算内积时，所使用的核函数种类
    :param str loss_type: 返回哪种损失，可以为'mse', 'l1', 'bce'
    :param args: 用于计算内积的额外参数
    :return: shape=[1]的tensor，相似损失

    将N个数据输入到输出层有M个神经元的SNN，运行T步，得到shape=[N, M, T]的脉冲。这N个数据的标签为shape=[N, C]的\ ``labels``。

    用shape=[N, N]的矩阵\ ``sim``\ 表示\ **实际相似度矩阵**，\ ``sim[i][j] == 1``\ 表示数据i与数据j相似，反之亦然。若\\
    \ ``labels[i]``\ 与\ ``labels[j]``\ 共享至少同一个标签，则认为他们相似，否则不相似。

    用shape=[N, N]的矩阵\ ``sim_p``\ 表示\ **输出相似度矩阵**，\ ``sim_p[i][j]``\ 的取值为0到1，值越大表示数据i与数据j的脉冲越相似。

    使用内积来衡量两个脉冲之间的相似性，\ ``kernel_type``\ 是计算内积时，所使用的核函数种类：

    - 'linear'，线性内积，:math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = \\boldsymbol{x_{i}}^{T}\\boldsymbol{y_{j}}`。

    - 'sigmoid'，Sigmoid内积，:math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = \\mathrm{sigmoid}(\\alpha \\boldsymbol{x_{i}}^{T}\\boldsymbol{y_{j}})`，其中 :math:`\\alpha = args[0]`。

    - 'gaussian'，高斯内积，:math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = \\mathrm{exp}(- \\frac{||\\boldsymbol{x_{i}} - \\boldsymbol{y_{j}}||^{2}}{2\\sigma^{2}})`，其中 :math:`\\sigma = args[0]`。

    当使用Sigmoid或高斯内积时，内积的取值范围均在[0, 1]之间；而使用线性内积时，为了保证内积取值仍然在[0, 1]之间，会进行归一化：\\
    按照 :math:`\\text{sim_p}[i][j]=\\frac{\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}})}{||\\boldsymbol{x_{i}}|| · ||\\boldsymbol{y_{j}}||}`。

    对于相似的数据，根据输入的\ ``loss_type``，返回度量\ ``sim``\ 与\ ``sim_p``\ 差异的损失：

    - 'mse' -- 返回sim与sim_p的均方误差（也就是l2误差）。

    - 'l1' -- 返回sim与sim_p的l1误差。

    - 'bce' -- 返回sim与sim_p的二值交叉熵误差。

    .. note::
        脉冲向量稀疏、离散，最好先使用高斯核进行平滑，然后再计算相似度。

    * :ref:`中文API <spike_similar_loss-cn>`

    .. _spike_similar_loss-en:

    :param spikes: shape=[N, M, T], output spikes corresponding to a batch of N inputs
    :param labels: shape=[N, C], labels of inputs, ``labels[i][k] == 1`` means the i-th input belongs to the k-th category and vice versa. Multi-label input is allowed.
    :param str kernel_type: Type of kernel function used when calculating inner products. The inner product is the similarity measure of two spikes.
    :param str loss_type: Type of loss returned. Can be: 'mse', 'l1', 'bce'
    :param args: Extra parameters for inner product
    :return: shape=[1], similarity loss

    A SNN consisting M neurons will receive a batch of N input data in each timestep (from 0 to T-1) and output a spike tensor of shape=[N, M, T]. The label is a tensor of shape=[N, C].

    The **groundtruth similarity matrix** ``sim`` has a shape of [N, N]. ``sim[i][j] == 1`` indicates that input i is similar to input j and vice versa. If and only if ``labels[i]`` and ``labels[j]`` have at least one common label, they are viewed as similar.

    The **output similarity matrix** ``sim_p`` has a shape of [N, N]. The value of ``sim_p[i][j]`` ranges from 0 to 1, represents the similarity between output spike from both input i and input j.

    The similarity is measured by inner product of two spikes. ``kernel_type`` is the type of kernel function when calculating inner product:

    - 'linear', Linear kernel, :math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = \\boldsymbol{x_{i}}^{T}\\boldsymbol{y_{j}}`.

    - 'sigmoid', Sigmoid kernel, :math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = \\mathrm{sigmoid}(\\alpha \\boldsymbol{x_{i}}^{T}\\boldsymbol{y_{j}})`, where :math:`\\alpha = args[0]`.

    - 'gaussian', Gaussian kernel，:math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = \\mathrm{exp}(- \\frac{||\\boldsymbol{x_{i}} - \\boldsymbol{y_{j}}||^{2}}{2\\sigma^{2}})`, where :math:`\\sigma = args[0]`.

    When Sigmoid or Gaussian kernel is applied, the inner product naturally lies in :math:`[0, 1]`. To make the value consistent when using linear kernel, the result will be normalized as: :math:`\\text{sim_p}[i][j]=\\frac{\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}})}{||\\boldsymbol{x_{i}}|| · ||\\boldsymbol{y_{j}}||}`.

    For similar data, return the specified discrepancy loss between ``sim`` and ``sim_p`` according to ``loss_type``.

    - 'mse' -- Return the Mean-Square Error (squared L2 norm) between sim and sim_p.

    - 'l1' -- Return the L1 error between sim and sim_p.

    - 'bce' -- Return the Binary Cross Entropy between sim and sim_p.

    .. admonition:: Note
        :class: note

        Since spike vectors are usually discrete and sparse, it would be better to apply Gaussian filter first to smooth the vectors before calculating similarities.
    """

    spikes = spikes.flatten(start_dim=1)

    sim_p = kernel_dot_product(spikes, spikes, kernel_type, *args)

    if kernel_type == 'linear':
        spikes_len = spikes.norm(p=2, dim=1, keepdim=True)
        sim_p = sim_p / ((spikes_len.mm(spikes_len.t())) + 1e-8)

    labels = labels.float()
    sim = labels.mm(labels.t()).clamp_max(1)  # labels.mm(labels.t())[i][j]位置的元素表现输入数据i和数据数据j有多少个相同的标签
    # 将大于1的元素设置为1，因为共享至少同一个标签，就认为他们相似

    if loss_type == 'mse':
        return F.mse_loss(sim_p, sim)
    elif loss_type == 'l1':
        return F.l1_loss(sim_p, sim)
    elif loss_type == 'bce':
        return F.binary_cross_entropy(sim_p, sim)
    else:
        raise NotImplementedError


def kernel_dot_product(x: Tensor, y: Tensor, kernel='linear', *args):
    """
    * :ref:`API in English <kernel_dot_product-en>`

    .. _kernel_dot_product-cn:
    
    :param x: shape=[N, M]的tensor，看作是N个M维向量
    :param y: shape=[N, M]的tensor，看作是N个M维向量
    :param str kernel: 计算内积时所使用的核函数
    :param args: 用于计算内积的额外的参数
    :return: ret, shape=[N, N]的tensor，``ret[i][j]``\ 表示\ ``x[i]``\ 和\ ``y[j]``\ 的内积

    计算批量数据\ ``x``\ 和\ ``y``\ 在核空间的内积。记2个M维tensor分别为 :math:`\\boldsymbol{x_{i}}` 和 :math:`\\boldsymbol{y_{j}}`，``kernel``\ 定义了不同形式的内积：

    - 'linear'，线性内积，:math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = \\boldsymbol{x_{i}}^{T}\\boldsymbol{y_{j}}`。

    - 'polynomial'，多项式内积，:math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = (\\boldsymbol{x_{i}}^{T}\\boldsymbol{y_{j}})^{d}`，其中 :math:`d = args[0]`。

    - 'sigmoid'，Sigmoid内积，:math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = \\mathrm{sigmoid}(\\alpha \\boldsymbol{x_{i}}^{T}\\boldsymbol{y_{j}})`，其中 :math:`\\alpha = args[0]`。

    - 'gaussian'，高斯内积，:math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = \\mathrm{exp}(- \\frac{||\\boldsymbol{x_{i}} - \\boldsymbol{y_{j}}||^{2}}{2\\sigma^{2}})`，其中 :math:`\\sigma = args[0]`。

    * :ref:`中文API <kernel_dot_product-cn>`

    .. _kernel_dot_product-en:

    :param x: Tensor of shape=[N, M]
    :param y: Tensor of shape=[N, M]
    :param str kernel: Type of kernel function used when calculating inner products.
    :param args: Extra parameters for inner product
    :return: ret, Tensor of shape=[N, N], ``ret[i][j]`` is inner product of ``x[i]`` and ``y[j]``.

    Calculate inner product of ``x`` and ``y`` in kernel space. These 2 M-dim tensors are denoted by :math:`\\boldsymbol{x_{i}}` and :math:`\\boldsymbol{y_{j}}`. ``kernel`` determine the kind of inner product: 

    - 'linear' -- Linear kernel, :math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = \\boldsymbol{x_{i}}^{T}\\boldsymbol{y_{j}}`.

    - 'polynomial' -- Polynomial kernel, :math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = (\\boldsymbol{x_{i}}^{T}\\boldsymbol{y_{j}})^{d}`, where :math:`d = args[0]`.

    - 'sigmoid' -- Sigmoid kernel, :math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = \\mathrm{sigmoid}(\\alpha \\boldsymbol{x_{i}}^{T}\\boldsymbol{y_{j}})`, where :math:`\\alpha = args[0]`.

    - 'gaussian' -- Gaussian kernel, :math:`\\kappa(\\boldsymbol{x_{i}}, \\boldsymbol{y_{j}}) = \\mathrm{exp}(- \\frac{||\\boldsymbol{x_{i}} - \\boldsymbol{y_{j}}||^{2}}{2\\sigma^{2}})`, where :math:`\\sigma = args[0]`.
    """
    if kernel == 'linear':
        return x.mm(y.t())
    elif kernel == 'polynomial':
        d = args[0]
        return x.mm(y.t()).pow(d)
    elif kernel == 'sigmoid':
        alpha = args[0]
        return torch.sigmoid(alpha * x.mm(y.t()))
    elif kernel == 'gaussian':
        sigma = args[0]
        N = x.shape[0]
        x2 = x.square().sum(dim=1)  # shape=[N]
        y2 = y.square().sum(dim=1)  # shape=[N]
        xy = x.mm(y.t())  # shape=[N, N]
        d_xy = x2.unsqueeze(1).repeat(1, N) + y2.unsqueeze(0).repeat(N, 1) - 2 * xy
        # d_xy[i][j]的元素是x[i]的平方和，加上y[j]的平方和，减去2倍的sum_{k} x[i][k]y[j][k]，因此
        # d_xy[i][j]就是x[i]和y[j]相减，平方，求和
        return torch.exp(- d_xy / (2 * sigma * sigma))
    else:
        raise NotImplementedError


def set_threshold_margin(output_layer: neuron.BaseNode, label_one_hot: Tensor,
                         eval_threshold=1.0, threshold0=0.9, threshold1=1.1):
    """
    * :ref:`API in English <set_threshold_margin-en>`

    .. _set_threshold_margin-cn:

    :param output_layer: 用于分类的网络的输出层，输出层输出shape=[batch_size, C]
    :param label_one_hot: one hot格式的样本标签，shape=[batch_size, C]
    :param float eval_threshold: 输出层神经元在测试（推理）时使用的电压阈值
    :param float threshold0: 输出层神经元在训练时，负样本的电压阈值
    :param float threshold1: 输出层神经元在训练时，正样本的电压阈值
    :return: None

    对于用来分类的网络，为输出层神经元的电压阈值设置一定的裕量，以获得更好的分类性能。

    类别总数为C，网络的输出层共有C个神经元。网络在训练时，当输入真实类别为i的数据，输出层中第i个神经元的电压阈值会被设置成\\
    ``threshold1``，而其他神经元的电压阈值会被设置成\ ``threshold0``。而在测试（推理）时，输出层中神经元的电压阈值被统一设置成\ ``eval_threshold``。

    * :ref:`中文API <set_threshold_margin-cn>`

    .. _set_threshold_margin-en:

    :param output_layer: The output layer of classification network, where the shape of output should be [batch_size, C]
    :param label_one_hot: Labels in one-hot format, shape=[batch_size, C]
    :param float eval_threshold: Voltage threshold of neurons in output layer when evaluating (inference)
    :param float threshold0: Voltage threshold of the corresponding neurons of **negative** samples in output layer when training
    :param float threshold1: Voltage threshold of the corresponding neurons of **positive** samples in output layer when training
    :return: None

    Set voltage threshold margin for neurons in the output layer to reach better performance in classification task.

    When there are C different classes, the output layer contains C neurons. During training, when the input with groundtruth label i are sent into the network, the voltage threshold of the i-th neurons in the output layer will be set to ``threshold1`` and the remaining will be set to ``threshold0``.
    
    During inference, the voltage thresholds of **ALL** neurons in the output layer will be set to ``eval_threshold``.
    """
    if output_layer.training:
        output_layer.v_threshold = torch.ones_like(label_one_hot) * threshold0
        output_layer.v_threshold[label_one_hot == 1] = threshold1
    else:
        output_layer.v_threshold = eval_threshold


def redundant_one_hot(labels: Tensor, num_classes: int, n: int):
    """
    * :ref:`API in English <redundant_one_hot-en>`

    .. _redundant_one_hot-cn:

    :param labels: shape=[batch_size]的tensor，表示\ ``batch_size``\ 个标签
    :param int num_classes: 类别总数
    :param int n: 表示每个类别所用的编码数量
    :return: shape=[batch_size, num_classes * n]的tensor

    对数据进行冗余的one-hot编码，每一类用 ``n`` 个1和 ``(num_classes - 1) * n`` 个0来编码。

    示例：

    .. code-block:: python

        >>> num_classes = 3
        >>> n = 2
        >>> labels = torch.randint(0, num_classes, [4])
        >>> labels
        tensor([0, 1, 1, 0])
        >>> codes = functional.redundant_one_hot(labels, num_classes, n)
        >>> codes
        tensor([[1., 1., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0.],
                [0., 0., 1., 1., 0., 0.],
                [1., 1., 0., 0., 0., 0.]])

    * :ref:`中文API <redundant_one_hot-cn>`

    .. _redundant_one_hot-en:

    :param labels: Tensor of shape=[batch_size], ``batch_size`` labels 
    :param int num_classes: The total number of classes.
    :param int n: The encoding length for each class.
    :return: Tensor of shape=[batch_size, num_classes * n]

    Redundant one-hot encoding for data. Each class is encoded to ``n`` 1's and  ``(num_classes - 1) * n`` 0's

    e.g.:

    .. code-block:: python

        >>> num_classes = 3
        >>> n = 2
        >>> labels = torch.randint(0, num_classes, [4])
        >>> labels
        tensor([0, 1, 1, 0])
        >>> codes = functional.redundant_one_hot(labels, num_classes, n)
        >>> codes
        tensor([[1., 1., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0.],
                [0., 0., 1., 1., 0., 0.],
                [1., 1., 0., 0., 0., 0.]])
    """
    redundant_classes = num_classes * n
    codes = torch.zeros(size=[labels.shape[0], redundant_classes], device=labels.device)
    for i in range(n):
        codes += F.one_hot(labels * n + i, redundant_classes)
    return codes


def first_spike_index(spikes: Tensor):
    """
    * :ref:`API in English <first_spike_index-en>`

    .. _first_spike_index-cn:

    :param spikes: shape=[*, T]，表示任意个神经元在t=0, 1, ..., T-1，共T个时刻的输出脉冲
    :return: index, shape=[*, T]，为 ``True`` 的位置表示该神经元首次释放脉冲的时刻

    输入若干个神经元的输出脉冲，返回一个与输入相同shape的 ``bool`` 类型的index。index为 ``True`` 的位置，表示该神经元首次释放脉冲的时刻。

    示例：

    .. code-block:: python

        >>> spikes = (torch.rand(size=[2, 3, 8]) >= 0.8).float()
        >>> spikes
        tensor([[[0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 1., 0.],
         [0., 1., 0., 0., 0., 1., 0., 1.]],

        [[0., 0., 1., 1., 0., 0., 0., 1.],
         [1., 1., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0.]]])
        >>> first_spike_index(spikes)
        tensor([[[False, False, False, False, False, False, False, False],
         [ True, False, False, False, False, False, False, False],
         [False,  True, False, False, False, False, False, False]],

        [[False, False,  True, False, False, False, False, False],
         [ True, False, False, False, False, False, False, False],
         [False, False, False,  True, False, False, False, False]]])

    * :ref:`中文API <first_spike_index-cn>`

    .. _first_spike_index-en:

    :param spikes: shape=[*, T], indicates the output spikes of some neurons when t=0, 1, ..., T-1.
    :return: index, shape=[*, T], the index of ``True`` represents the moment of first spike.

    Return an ``index`` tensor of the same shape of input tensor, which is the output spike of some neurons. The index of ``True`` represents the moment of first spike.

    e.g.:

    .. code-block:: python

        >>> spikes = (torch.rand(size=[2, 3, 8]) >= 0.8).float()
        >>> spikes
        tensor([[[0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 1., 0.],
         [0., 1., 0., 0., 0., 1., 0., 1.]],

        [[0., 0., 1., 1., 0., 0., 0., 1.],
         [1., 1., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0.]]])
        >>> first_spike_index(spikes)
        tensor([[[False, False, False, False, False, False, False, False],
         [ True, False, False, False, False, False, False, False],
         [False,  True, False, False, False, False, False, False]],

        [[False, False,  True, False, False, False, False, False],
         [ True, False, False, False, False, False, False, False],
         [False, False, False,  True, False, False, False, False]]])

    """
    with torch.no_grad():
        # 在时间维度上，2次cumsum后，元素为1的位置，即为首次发放脉冲的位置
        return spikes.cumsum(dim=-1).cumsum(dim=-1) == 1


def multi_step_forward(x_seq: Tensor, single_step_module: nn.Module or list[nn.Module] or tuple[nn.Module] or nn.Sequential or Callable):
    """
    * :ref:`API in English <multi_step_forward-en>`

    .. _multi_step_forward-cn:

    :param x_seq: ``shape=[T, batch_size, ...]`` 的输入tensor
    :type x_seq: Tensor
    :param single_step_module: 一个或多个单步模块
    :type single_step_module: torch.nn.Module or list[nn.Module] or tuple[nn.Module] or torch.nn.Sequential or Callable
    :return: ``shape=[T, batch_size, ...]`` 的输出tensor
    :rtype: torch.Tensor

    在单步模块 ``single_step_module`` 上使用多步前向传播。

    * :ref:`中文 API <multi_step_forward-cn>`

    .. _multi_step_forward-en:

    :param x_seq: the input tensor with ``shape=[T, batch_size, ...]``
    :type x_seq: torch.Tensor
    :param single_step_module: one or many single-step modules
    :type single_step_module: torch.nn.Module or list[nn.Module] or tuple[nn.Module] or torch.nn.Sequential or Callable
    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: torch.torch.Tensor

    Applies multi-step forward on ``single_step_module``.

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

def chunk_multi_step_forward(split_size: int, x_seq: Tensor, multi_step_module: nn.Module):
    """
    * :ref:`API in English <chunk_multi_step_forward-en>`

    .. _chunk_multi_step_forward-cn:

    :param split_size: 分割的尺寸
    :type split_size: int
    :param x_seq: 输入
    :type x_seq: Tensor
    :param multi_step_module: 一个使用多步传播模式的网络
    :type multi_step_module: nn.Module
    :return: 输出
    :rtype: Tensor

    将 ``shape = [T, *]`` 的输入 ``x_seq`` 拆分成多个 ``shape = [split_size, *]`` 的小tensor(若 ``T % split_size != 0``，最后\
    一个tensor的 ``shape[0]`` 会小于 ``split_size``)，然后逐个输入到 ``multi_step_module`` 中，再将输出重新拼接为 ``shape = [split_size, *]``。\

    ``chunk_multi_step_forward`` 可以在使用很大的 ``T`` 进行不带梯度的推理(例如ANN2SNN)时使用，能够减少内存消耗量。

    示例代码：

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

    * :ref:`中文 API <chunk_multi_step_forward-cn>`

    .. _chunk_multi_step_forward-en:

    :param split_size: the split size
    :type split_size: int
    :param x_seq: the input tensor
    :type x_seq: Tensor
    :param multi_step_module:
    :type multi_step_module: nn.Module
    :return: the output tensor
    :rtype: Tensor

    Splits the input ``x_seq`` with ``shape = [T, *]`` to many tensor chunks with ``shape = [split_size, *]`` (if ``T % split_size != 0``,\
    ``shape[0]`` of the last tensor chunk will be smaller than ``split_size``), and sends chunks to ``multi_step_module``,\
    then concatenates the outputs to  ``shape = [split_size, *]``.

    ``chunk_multi_step_forward`` can be used for inference with a large ``T`` (e.g., ANN2SNN) to reduce the memory consumption.

    Codes example:

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

def seq_to_ann_forward(x_seq: Tensor, stateless_module: nn.Module or list or tuple or nn.Sequential or Callable):
    """
    * :ref:`API in English <seq_to_ann_forward-en>`

    .. _seq_to_ann_forward-cn:

    :param x_seq: ``shape=[T, batch_size, ...]`` 的输入tensor
    :type x_seq: Tensor
    :param stateless_module: 单个或多个无状态网络层
    :type stateless_module: torch.nn.Module or list or tuple or torch.nn.Sequential or Callable
    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: Tensor

    * :ref:`中文 API <seq_to_ann_forward-cn>`

    .. _seq_to_ann_forward-en:

    :param x_seq: the input tensor with ``shape=[T, batch_size, ...]``
    :type x_seq: Tensor
    :param stateless_module: one or many stateless modules
    :type stateless_module: torch.nn.Module or list or tuple or torch.nn.Sequential or Callable
    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: Tensor

    Applied forward on stateless modules

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


def fused_conv2d_weight_of_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d):
    """
    * :ref:`API in English <fused_conv2d_weight_of_convbn2d-en>`

    .. _fused_conv2d_weight_of_convbn2d-cn:

    :param conv2d: 一个2D卷积层
    :type conv2d: torch.nn.Conv2d
    :param bn2d: 一个2D的BN层
    :type bn2d: torch.nn.BatchNorm2d
    :return: the weight of this fused module
    :rtype: Tensor

    ``{Conv2d-BatchNorm2d}`` 模块可以合并为一个单个的 ``{Conv2d}``，其中``BatchNorm2d`` 的参数会被吸收进 ``Conv2d``。
    本函数返回合并后的卷积的权重。

    .. note::

        这里按照 ``conv2d.bias`` 为 ``None`` 进行处理。原因参见 `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ 。

    * :ref:`中文 API <fused_conv2d_weight_of_convbn2d-cn>`

    .. _fused_conv2d_weight_of_convbn2d-en:

    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d
    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d
    :return: the weight of this fused module
    :rtype: Tensor

    A ``{Conv2d-BatchNorm2d}`` can be fused to a ``{Conv2d}`` module with ``BatchNorm2d`` 's parameters being absorbed into ``Conv2d``.
    This function returns the weight of this fused module.

    .. admonition:: Note
        :class: note

        We assert ``conv2d.bias`` is ``None``. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    """
    assert conv2d.bias is None
    return (conv2d.weight.transpose(0, 3) * bn2d.weight / (
            bn2d.running_var + bn2d.eps).sqrt()).transpose(0, 3)


def fused_conv2d_bias_of_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d):
    """
    * :ref:`API in English <fused_conv2d_bias_of_convbn2d-en>`

    .. _fused_conv2d_bias_of_convbn2d-cn:

    :param conv2d: 一个2D卷积层
    :type conv2d: torch.nn.Conv2d
    :param bn2d: 一个2D的BN层
    :type bn2d: torch.nn.BatchNorm2d
    :return: the weight of this fused module
    :rtype: Tensor

    ``{Conv2d-BatchNorm2d}`` 模块可以合并为一个单个的 ``{Conv2d}``，其中``BatchNorm2d`` 的参数会被吸收进 ``Conv2d``。
    本函数返回合并后的卷积的偏置项。

    .. note::

        这里按照 ``conv2d.bias`` 为 ``None`` 进行处理。原因参见 `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ 。

    * :ref:`中文 API <fused_conv2d_bias_of_convbn2d-cn>`

    .. _fused_conv2d_bias_of_convbn2d-en:

    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d
    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d
    :return: the weight of this fused module
    :rtype: Tensor

    A ``{Conv2d-BatchNorm2d}`` can be fused to a ``{Conv2d}`` module with ``BatchNorm2d`` 's parameters being absorbed into ``Conv2d``.
    This function returns the bias of this fused module.

    .. admonition:: Note
        :class: note

        We assert ``conv2d.bias`` is ``None``. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    """
    assert conv2d.bias is None
    return bn2d.bias - bn2d.running_mean * bn2d.weight / (bn2d.running_var + bn2d.eps).sqrt()


@torch.no_grad()
def scale_fused_conv2d_weight_of_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d, k=None, b=None):
    """
    * :ref:`API in English <scale_fused_conv2d_weight_of_convbn2d-en>`

    .. _scale_fused_conv2d_weight_of_convbn2d-cn:

    :param conv2d: 一个2D卷积层
    :type conv2d: torch.nn.Conv2d
    :param bn2d: 一个2D的BN层
    :type bn2d: torch.nn.BatchNorm2d
    :return: the weight of this fused module
    :rtype: Tensor

    ``{Conv2d-BatchNorm2d}`` 模块可以合并为一个单个的 ``{Conv2d}``，其中``BatchNorm2d`` 的参数会被吸收进 ``Conv2d``。
    本函数对 ``{Conv2d-BatchNorm2d}`` 模块整体的等效权重进行 ``weight = k * weight + b`` 的线性变换。

    .. note::

        这里按照 ``conv2d.bias`` 为 ``None`` 进行处理。原因参见 `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ 。

    * :ref:`中文 API <scale_fused_conv2d_weight_of_convbn2d-cn>`

    .. _scale_fused_conv2d_weight_of_convbn2d-en:

    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d
    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d
    :return: the weight of this fused module
    :rtype: Tensor

    A ``{Conv2d-BatchNorm2d}`` can be fused to a ``{Conv2d}`` module with ``BatchNorm2d`` 's parameters being absorbed into ``Conv2d``.
    This function applies a linear transform ``weight = k * weight + b`` on the equivalent weight of the whole ``{Conv2d-BatchNorm2d}``.

    .. admonition:: Note
        :class: note

        We assert ``conv2d.bias`` is ``None``. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    """
    assert conv2d.bias is None
    if k is not None:
        conv2d.weight.data *= k
    if b is not None:
        conv2d.weight.data += b


@torch.no_grad()
def scale_fused_conv2d_bias_of_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d, k=None, b=None):
    """
    * :ref:`API in English <scale_fused_conv2d_bias_of_convbn2d-en>`

    .. _scale_fused_conv2d_bias_of_convbn2d-cn:

    :param conv2d: 一个2D卷积层
    :type conv2d: torch.nn.Conv2d
    :param bn2d: 一个2D的BN层
    :type bn2d: torch.nn.BatchNorm2d
    :return: the weight of this fused module
    :rtype: Tensor

    ``{Conv2d-BatchNorm2d}`` 模块可以合并为一个单个的 ``{Conv2d}``，其中``BatchNorm2d`` 的参数会被吸收进 ``Conv2d``。
    本函数对 ``{Conv2d-BatchNorm2d}`` 模块整体的等效偏置项进行 ``bias = k * bias + b`` 的线性变换。

    .. note::

        这里按照 ``conv2d.bias`` 为 ``None`` 进行处理。原因参见 `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ 。

    * :ref:`中文 API <scale_fused_conv2d_bias_of_convbn2d-cn>`

    .. _scale_fused_conv2d_bias_of_convbn2d-en:

    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d
    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d
    :return: the weight of this fused module
    :rtype: Tensor

    A ``{Conv2d-BatchNorm2d}`` can be fused to a ``{Conv2d}`` module with ``BatchNorm2d`` 's parameters being absorbed into ``Conv2d``.
    This function applies a linear transform ``bias = k * bias + b`` on the equivalent bias of the whole ``{Conv2d-BatchNorm2d}``.

    .. admonition:: Note
        :class: note

        We assert ``conv2d.bias`` is ``None``. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    """
    assert conv2d.bias is None
    if k is not None:
        bn2d.bias.data *= k
        bn2d.running_mean *= k
    if b is not None:
        bn2d.bias.data += b


@torch.no_grad()
def fuse_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d):
    """
    * :ref:`API in English <fuse_convbn2d-en>`

    .. _fuse_convbn2d-cn:

    :param conv2d: 一个2D卷积层
    :type conv2d: torch.nn.Conv2d
    :param bn2d: 一个2D的BN层
    :type bn2d: torch.nn.BatchNorm2d
    :return: the weight of this fused module
    :rtype: Tensor

    ``{Conv2d-BatchNorm2d}`` 模块可以合并为一个单个的 ``{Conv2d}``，其中``BatchNorm2d`` 的参数会被吸收进 ``Conv2d``。
    本函数对返回这个等效的合并后的 ``{Conv2d}``。

    .. note::

        这里按照 ``conv2d.bias`` 为 ``None`` 进行处理。原因参见 `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ 。

    * :ref:`中文 API <fuse_convbn2d-cn>`

    .. _fuse_convbn2d-en:

    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d
    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d
    :return: the weight of this fused module
    :rtype: Tensor

    A ``{Conv2d-BatchNorm2d}`` can be fused to a ``{Conv2d}`` module with ``BatchNorm2d`` 's parameters being absorbed into ``Conv2d``.
    This function returns the fused ``{Conv2d}`` merged by ``{Conv2d-BatchNorm2d}``.

    .. admonition:: Note
        :class: note

        We assert ``conv2d.bias`` is ``None``. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    """
    fused_conv = nn.Conv2d(in_channels=conv2d.in_channels, out_channels=conv2d.out_channels,
                           kernel_size=conv2d.kernel_size,
                           stride=conv2d.stride, padding=conv2d.padding, dilation=conv2d.dilation,
                           groups=conv2d.groups, bias=True,
                           padding_mode=conv2d.padding_mode)
    fused_conv.weight.data = fused_conv2d_weight_of_convbn2d(conv2d, bn2d)
    fused_conv.bias.data = fused_conv2d_bias_of_convbn2d(conv2d, bn2d)
    return fused_conv

@torch.jit.script
def temporal_efficient_training_cross_entropy(x_seq: Tensor, target: torch.Tensor):
    """
    * :ref:`API in English <temporal_efficient_training_cross_entropy-en>`

    .. _temporal_efficient_training_cross_entropy-cn:

    :param x_seq: ``shape=[T, N, C, *]`` 的预测值，其中 ``C`` 是类别总数
    :type x_seq: torch.Tensor
    :param target: ``shape=[N]`` 的真实值，其中 ``target[i]`` 是真实类别
    :type target: torch.Tensor
    :return: the temporal efficient training cross entropy
    :rtype: torch.Tensor

    Temporal efficient training (TET) 交叉熵损失, 是每个时间步的交叉熵损失的平均。

    示例代码：

    .. code-block:: python

        def tet_ce_for_loop_version(x_seq: torch.Tensor, target: torch.LongTensor):
            loss = 0.
            for t in range(x_seq.shape[0]):
                loss += F.cross_entropy(x_seq[t], target)
            return loss / x_seq.shape[0]


        T = 8
        N = 4
        C = 10
        x_seq = torch.rand([T, N, C])
        target = torch.randint(low=0, high=C - 1, size=[N])
        print(f'max error = {(tet_ce_for_loop_version(x_seq, target) - temporal_efficient_training_cross_entropy(x_seq, target)).abs().max()}')
        # max error < 1e-6


    .. note::

        TET交叉熵是 `Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting <https://openreview.net/forum?id=_XNtisL32jv>`_ 一文提出的。

    * :ref:`中文 API <temporal_efficient_training_cross_entropy-cn>`

    .. _temporal_efficient_training_cross_entropy-en:

    :param x_seq: the predicted value with ``shape=[T, N, C, *]``, where ``C`` is the number of classes
    :type x_seq: torch.Tensor
    :param target: the ground truth tensor with ``shape=[N]``, where ``target[i]`` is the label
    :type target: torch.Tensor
    :return: the temporal efficient training cross entropy
    :rtype: torch.Tensor

    The temporal efficient training (TET) cross entropy, which is the mean of cross entropy of each time-step.

    Codes example:

    .. code-block:: python

        def tet_ce_for_loop_version(x_seq: torch.Tensor, target: torch.LongTensor):
            loss = 0.
            for t in range(x_seq.shape[0]):
                loss += F.cross_entropy(x_seq[t], target)
            return loss / x_seq.shape[0]


        T = 8
        N = 4
        C = 10
        x_seq = torch.rand([T, N, C])
        target = torch.randint(low=0, high=C - 1, size=[N])
        print(f'max error = {(tet_ce_for_loop_version(x_seq, target) - temporal_efficient_training_cross_entropy(x_seq, target)).abs().max()}')
        # max error < 1e-6


    .. admonition:: Note
        :class: note

        The TET cross entropy is proposed by `Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting <https://openreview.net/forum?id=_XNtisL32jv>`_.
    """
    x_seq = x_seq.transpose(0, 1).transpose(1, 2)  # [N, C, T, *]
    N, C, T = x_seq.shape[0], x_seq.shape[1], x_seq.shape[2]
    if x_seq.dim() == 3:
        # x_seq.shape = [N, C, T]
        # target.shape = [N]
        target = target.unsqueeze(1).repeat(1, T)  # [N, T]
    else:
        # x_seq.shape = [N, C, T, d1, d2, ..., dk]
        # target.shape = [N, d1, d2, ..., dk]
        rep_shape = [1, T]
        rep_shape.extend([1] * (x_seq.dim() - 3))
        target = target.unsqueeze(1).repeat(rep_shape)

    loss = F.cross_entropy(x_seq, target)
    return loss


def kaiming_normal_conv_linear_weight(net: nn.Module):
    """
    * :ref:`API in English <kaiming_normal_conv_linear_weight-en>`

    .. _kaiming_normal_conv_linear_weight-cn:

    :param net: 任何属于 ``nn.Module`` 子类的网络

    :return: None

    使用kaiming normal初始化 ``net` `中的所有 :class:`torch.nn._ConvNd` 和 :class:`torch.nn.Linear` 的权重（不包括偏置项）。参见 :class:`torch.nn.init.kaiming_normal_`。

    * :ref:`中文API <kaiming_normal_conv_linear_weight-cn>`

    .. _kaiming_normal_conv_linear_weight-en:

    :param net: Any network inherits from ``nn.Module``

    :return: None

    initialize all weights (not including bias) of :class:`torch.nn._ConvNd` and :class:`torch.nn.Linear` in ``net`` by the kaiming normal. See :class:`torch.nn.init.kaiming_normal_`
    for more details.
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))

@torch.jit.script
def delay(x_seq: torch.Tensor, delay_steps: int):
    """
    * :ref:`API in English <delay.__init__-en>`

    .. _delay.__init__-cn:

    :param x_seq: 输入的序列，``shape = [T, *]``
    :type x_seq: torch.Tensor
    :param delay_steps: 延迟的时间步数
    :type delay_steps: int
    :return: 延迟后的序列
    :rtype: torch.Tensor

    延迟函数，可以用来延迟输入，使得 ``y[t] = x[t - delay_steps]``。缺失的数据用0填充。

    代码示例：

        .. code-block:: python

            x = torch.rand([5, 2])
            x[3:].zero_()
            x.requires_grad = True
            y = delay(x, 1)
            print('x=')
            print(x)
            print('y=')
            print(y)
            y.sum().backward()
            print('x.grad=')
            print(x.grad)

    输出为：

        .. code-block:: bash

            x=
            tensor([[0.1084, 0.5698],
                    [0.4563, 0.3623],
                    [0.0556, 0.4704],
                    [0.0000, 0.0000],
                    [0.0000, 0.0000]], requires_grad=True)
            y=
            tensor([[0.0000, 0.0000],
                    [0.1084, 0.5698],
                    [0.4563, 0.3623],
                    [0.0556, 0.4704],
                    [0.0000, 0.0000]], grad_fn=<CatBackward0>)
            x.grad=
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [0., 0.]])

    * :ref:`中文API <delay.__init__-cn>`

    .. _delay.__init__-en:

    :param x_seq: the input sequence with ``shape = [T, *]``
    :type x_seq: torch.Tensor
    :param delay_steps: the number of delayed time-steps
    :type delay_steps: int
    :return: the delayed sequence
    :rtype: torch.Tensor


    A delay function that can delay inputs and makes ``y[t] = x[t - delay_steps]``. The nonexistent data will be regarded as 0.

    Codes example:

        .. code-block:: python

            x = torch.rand([5, 2])
            x[3:].zero_()
            x.requires_grad = True
            y = delay(x, 1)
            print('x=')
            print(x)
            print('y=')
            print(y)
            y.sum().backward()
            print('x.grad=')
            print(x.grad)

    The outputs are:

        .. code-block:: bash

            x=
            tensor([[0.1084, 0.5698],
                    [0.4563, 0.3623],
                    [0.0556, 0.4704],
                    [0.0000, 0.0000],
                    [0.0000, 0.0000]], requires_grad=True)
            y=
            tensor([[0.0000, 0.0000],
                    [0.1084, 0.5698],
                    [0.4563, 0.3623],
                    [0.0556, 0.4704],
                    [0.0000, 0.0000]], grad_fn=<CatBackward0>)
            x.grad=
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [0., 0.]])

    """
    # x_seq.shape = [T, *]
    y = torch.zeros_like(x_seq[0: delay_steps].data)
    return torch.cat((y, x_seq[0: x_seq.shape[0] - delay_steps]), 0)

def fptt_online_training_init_w_ra(optimizer: torch.optim.Optimizer) -> list:
    w_ra = []
    for item in optimizer.param_groups:
        for w in item['params']:
            w_ra.append(w.data)

    return w_ra

def fptt_online_training(model: nn.Module, optimizer: torch.optim.Optimizer, x_seq: torch.Tensor, target_seq: torch.Tensor, f_loss_t: Callable, alpha: float, w_ra: list) -> None:
    """
    :param model: the neural network
    :type model: nn.Module
    :param optimizer: the optimizer for the network
    :type optimizer: torch.optim.Optimizer
    :param x_seq: the input sequence
    :type x_seq: torch.Tensor
    :param target_seq: the output sequence
    :type target_seq: torch.Tensor
    :param f_loss_t: the loss function, which should has the formulation of ``def f_loss_t(x_t, y_t) -> torch.Tensor``
    :type f_loss_t: Callable
    :param alpha: the hyper-parameter
    :type alpha: float
    :param w_ra: the running average of params, which can be initialized by :class:`spikingjelly.activation_based.functional.fptt_online_training_init_w_ra`
    :type w_ra: list


    The FPTT online learning method proposed by `Training Recurrent Neural Networks via Forward Propagation Through Time <https://proceedings.mlr.press/v139/kag21a.html>`_ and used for SNN in `Accurate online training of dynamical spiking neural networks through Forward Propagation Through Time <https://arxiv.org/abs/2112.11231>`_ .

    Example:

    .. code-block:: python

        from spikingjelly.activation_based import neuron

        net = nn.Sequential(
            nn.Linear(8, 4),
            neuron.IFNode(),
            nn.Linear(4, 2),
            neuron.IFNode()
        )

        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

        T = 4
        N = 2
        w_ra = fptt_online_training_init_w_ra(optimizer)
        for epoch in range(2):

            x_seq = torch.rand([T, N, 8])
            target_seq = torch.rand([T, N, 2])

            fptt_online_training(model=net, optimizer=optimizer, x_seq=x_seq, target_seq=target_seq, f_loss_t=F.mse_loss, alpha=0.1, w_ra=w_ra)
            functional.reset_net(net)

    """
    T = x_seq.shape[0]

    grad__l_t_last__to__w_t = []

    for item in optimizer.param_groups:
        for w in item['params']:
            grad__l_t_last__to__w_t.append(0.)



    for t in range(T):
        optimizer.zero_grad()

        y_t = model(x_seq[t])
        loss_t = f_loss_t(y_t, target_seq[t])
        loss_reg = 0.
        i = 0
        for item in optimizer.param_groups:
            for w in item['params']:
                loss_reg = loss_reg + F.mse_loss(w, w_ra[i] + grad__l_t_last__to__w_t[i] / (2. * alpha))
                i += 1

        loss_reg = loss_reg * (alpha / 2.)

        loss = loss_t + loss_reg
        loss.backward()

        # update params
        optimizer.step()
        detach_net(model)

        # store hidden states
        states = []
        i = 0
        for m in model.modules():
            if isinstance(m, base.MemoryModule):
                states.append(copy.deepcopy(m._memories))
                i += 1

        # update w_ra
        optimizer.zero_grad()
        if t < T - 1:
            y_t = model(x_seq[t])
            loss_t = f_loss_t(y_t, target_seq[t])
            loss_t.backward()
            with torch.no_grad():
                i = 0
                for item in optimizer.param_groups:
                    for w in item['params']:
                        grad__l_t_last__to__w_t[i] = w.grad
                        w_ra[i] = (w_ra[i] + w) / 2. - w.grad / (2. * alpha)
                        i += 1
        optimizer.zero_grad()

        # recover hidden states
        i = 0
        for m in model.modules():
            if isinstance(m, base.MemoryModule):
                m._memories = states[i]
                i += 1




