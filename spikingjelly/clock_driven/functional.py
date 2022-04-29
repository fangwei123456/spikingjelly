import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import neuron

from torch import Tensor

def reset_net(net: nn.Module):
    '''
    * :ref:`API in English <reset_net-en>`

    .. _reset_net-cn:

    :param net: 任何属于 ``nn.Module`` 子类的网络

    :return: None

    将网络的状态重置。做法是遍历网络中的所有 ``Module``，若含有 ``reset()`` 函数，则调用。

    * :ref:`中文API <reset_net-cn>`

    .. _reset_net-en:

    :param net: Any network inherits from ``nn.Module``

    :return: None

    Reset the whole network.  Walk through every ``Module`` and call their ``reset()`` function if exists.
    '''
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()

def spike_cluster(v: Tensor, v_threshold, T_in: int):
    '''
    * :ref:`API in English <spike_cluster-en>`

    .. _spike_cluster-cn:

    :param v: shape=[T, N]，N个神经元在 t=[0, 1, ..., T-1] 时刻的电压值
    :param v_threshold: 神经元的阈值电压，float或者是shape=[N]的tensor
    :type v_threshold: float or tensor
    :param T_in: 脉冲聚类的距离阈值。一个脉冲聚类满足，内部任意2个相邻脉冲的距离不大于\ ``T_in``，而其内部任一脉冲与外部的脉冲距离大于\ ``T_in``。
    :return: 一个元组，包含
    
        - **N_o** -- shape=[N]，N个神经元的输出脉冲的脉冲聚类的数量

        - **k_positive** -- shape=[N]，bool类型的tensor，索引。需要注意的是，k_positive可能是一个全False的tensor

        - **k_negative** -- shape=[N]，bool类型的tensor，索引。需要注意的是，k_negative可能是一个全False的tensor 
    :rtype: (Tensor, Tensor, Tensor)

    `STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep Spiking Neural Networks <https://www.ijcai.org/Proceedings/2019/0189.pdf>`_\ 一文提出的脉冲聚类方法。如果想使用该文中定义的损失，可以参考如下代码：

    .. code-block:: python

        v_k_negative = out_v * k_negative.float().sum(dim=0)
        v_k_positive = out_v * k_positive.float().sum(dim=0)
        loss0 = ((N_o > N_d).float() * (v_k_negative - 1.0)).sum()
        loss1 = ((N_o < N_d).float() * (1.0 - v_k_positive)).sum()
        loss = loss0 + loss1

    * :ref:`中文API <spike_cluster-cn>`

    .. _spike_cluster-en:

    :param v: shape=[T, N], membrane potentials of N neurons when t=[0, 1, ..., T-1]
    :param v_threshold: Threshold voltage(s) of the neurons, float or tensor of the shape=[N]
    :type v_threshold: float or tensor
    :param T_in: Distance threshold of the spike clusters. A spike cluster satisfies that the distance of any two adjacent spikes within cluster is NOT greater than ``T_in`` and the distance between any internal and any external spike of cluster is greater than ``T_in``. 
    :return: A tuple containing
    
        - **N_o** -- shape=[N], numbers of spike clusters of N neurons' output spikes

        - **k_positive** -- shape=[N], tensor of type BoolTensor, indexes. Note that k_positive can be a tensor filled with False

        - **k_negative** -- shape=[N], tensor of type BoolTensor, indexes. Note that k_negative can be a tensor filled with False
    :rtype: (Tensor, Tensor, Tensor)

    A spike clustering method proposed in `STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep Spiking Neural Networks. <https://www.ijcai.org/Proceedings/2019/0189.pdf>`_ You can refer to the following code if this form of loss function is needed:

    .. code-block:: python

        v_k_negative = out_v * k_negative.float().sum(dim=0)
        v_k_positive = out_v * k_positive.float().sum(dim=0)
        loss0 = ((N_o > N_d).float() * (v_k_negative - 1.0)).sum()
        loss1 = ((N_o < N_d).float() * (1.0 - v_k_positive)).sum()
        loss = loss0 + loss1
    '''
    with torch.no_grad():

        spike = (v >= v_threshold).float()
        T = v.shape[0]

        N_o = torch.zeros_like(v[1])
        spikes_num = torch.ones_like(v[1]) * T * 2
        min_spikes_num = torch.ones_like(v[1]) * T * 2
        min_spikes_num_t = torch.ones_like(v[1]) * T * 2
        last_spike_t = - torch.ones_like(v[1]) * T_in * 2
        # 初始时，认为上一次的脉冲发放时刻是- T_in * 2，这样即便在0时刻发放脉冲，其与上一个脉冲发放时刻的间隔也大于T_in

        for t in range(T):
            delta_t = (t - last_spike_t) * spike[t]
            # delta_t[i] == 0的神经元i，当前时刻无脉冲发放
            # delta_t[i] > 0的神经元i，在t时刻释放脉冲，距离上次释放脉冲的时间差为delta_t[i]

            mask0 = (delta_t > T_in)  # 在t时刻释放脉冲，且距离上次释放脉冲的时间高于T_in的神经元
            mask1 = torch.logical_and(delta_t <= T_in, spike[t].bool())  # t时刻释放脉冲，但距离上次释放脉冲的时间不超过T_in的神经元



            temp_mask = torch.logical_and(mask0, min_spikes_num > spikes_num)
            min_spikes_num_t[temp_mask] = last_spike_t[temp_mask]
            min_spikes_num[temp_mask] = spikes_num[temp_mask]

            spikes_num[mask0] = 1
            N_o[mask0] += 1
            spikes_num[mask1] += 1
            last_spike_t[spike[t].bool()] = t




        mask = (spikes_num < min_spikes_num)
        min_spikes_num[mask] = spikes_num[mask]
        min_spikes_num_t[mask] = last_spike_t[mask]

        # 开始求解k_positive
        v_ = v.clone()
        v_min = v_.min().item()
        v_[spike.bool()] = v_min
        last_spike_t = - torch.ones_like(v[1]) * T_in * 2
        # 初始时，认为上一次的脉冲发放时刻是- T_in * 2，这样即便在0时刻发放脉冲，其与上一个脉冲发放时刻的间隔也大于T_in

        # 遍历t，若t距离上次脉冲发放时刻的时间不超过T_in则将v_设置成v_min
        for t in range(T):
            delta_t = (t - last_spike_t)

            mask = torch.logical_and(delta_t <= T_in, (1 - spike[t]).bool())
            # 表示与上次脉冲发放时刻距离不超过T_in且当前时刻没有释放脉冲（这些位置如果之后释放了脉冲，也会被归类到上次脉冲
            # 所在的脉冲聚类里）
            v_[t][mask] = v_min

            last_spike_t[spike[t].bool()] = t

        # 反着遍历t，若t距离下次脉冲发放时刻的时间不超过T_in则将v_设置成v_min
        next_spike_t = torch.ones_like(v[1]) * T_in * 2 + T
        for t in range(T - 1, -1, -1):
            delta_t = (next_spike_t - t)

            mask = torch.logical_and(delta_t <= T_in, (1 - spike[t]).bool())
            # 表示与下次脉冲发放时刻距离不超过T_in且当前时刻没有释放脉冲（这些位置如果之后释放了脉冲，也会被归类到下次脉冲
            # 所在的脉冲聚类里）
            v_[t][mask] = v_min

            next_spike_t[spike[t].bool()] = t


        k_positive = v_.argmax(dim=0)
        k_negative = min_spikes_num_t.long()
        arrange = torch.arange(0, T, device=v.device).unsqueeze(1).repeat(1, v.shape[1])
        k_positive = (arrange == k_positive)
        k_negative = (arrange == k_negative)

        # 需要注意的是，如果脉冲聚类太密集，导致找不到符合要求的k_positive，例如脉冲为[1 0 1 1]，T_in=1，此时得到的v_在0到T均为v_min，k_positive
        # 是1，但实际上1的位置不符合k_positive的定义，因为这个位置发放脉冲后，会与已有的脉冲聚类合并，不能生成新的脉冲聚类
        # 这种情况下，v_中的所有元素均为v_min
        # 用k_positive_mask来记录，k_positive_mask==False的神经元满足这种情况，用k_positive与k_positive_mask做and操作，可以去掉这些
        # 错误的位置
        # 但是v_.max(dim=0)[0] == v_min，也就是k_positive_mask==False的神经元，在0到T时刻的v_均为v_min，只有两种情况：
        #   1.v在0到T全部过阈值，一直在发放脉冲，因此才会出现v_在0到T均为v_min，这种情况下k_positive_mask==False
        #   2.v本身在0到T均为v_min，且从来没有发放脉冲，这是一种非常极端的情况，
        #     这种情况下k_positive_mask应该为True但却被设置成False，应该修正
        k_positive_mask = (v_.max(dim=0)[0] != v_min)

        # 修正情况2
        k_positive_mask[v.max(dim=0)[0] == v_min] = True
        # 在没有这行修正代码的情况下，如果v是全0的tensor，会错误的出现k_positive为空tensor

        k_positive = torch.logical_and(k_positive, k_positive_mask)

        return N_o, k_positive, k_negative

def spike_similar_loss(spikes:Tensor, labels:Tensor, kernel_type='linear', loss_type='mse', *args):
    '''
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
    '''

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

def kernel_dot_product(x:Tensor, y:Tensor, kernel='linear', *args):

    '''
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
    '''
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

def set_threshold_margin(output_layer:neuron.BaseNode, label_one_hot:Tensor,
                         eval_threshold=1.0, threshold0=0.9, threshold1=1.1):
    '''
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
    '''
    if output_layer.training:
        output_layer.v_threshold = torch.ones_like(label_one_hot) * threshold0
        output_layer.v_threshold[label_one_hot == 1] = threshold1
    else:
        output_layer.v_threshold = eval_threshold

def redundant_one_hot(labels:Tensor, num_classes:int, n:int):
    '''
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
    '''
    redundant_classes = num_classes * n
    codes = torch.zeros(size=[labels.shape[0], redundant_classes], device=labels.device)
    for i in range(n):
        codes += F.one_hot(labels * n + i, redundant_classes)
    return codes

def first_spike_index(spikes: Tensor):
    '''
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

    '''
    with torch.no_grad():
        # 在时间维度上，2次cumsum后，元素为1的位置，即为首次发放脉冲的位置
        return spikes.cumsum(dim=-1).cumsum(dim=-1) == 1

def multi_step_forward(x_seq: Tensor, single_step_module: nn.Module or list or tuple or nn.Sequential):
    """
    :param x_seq: shape=[T, batch_size, ...]
    :type x_seq: Tensor
    :param single_step_module: a single-step module, or a list/tuple that contains single-step modules
    :type single_step_module: torch.nn.Module or list or tuple or torch.nn.Sequential
    :return: y_seq, shape=[T, batch_size, ...]
    :rtype: Tensor

    See :class:`spikingjelly.clock_driven.layer.MultiStepContainer` for more details.
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

    for t in range(y_seq.__len__()):
        y_seq[t] = y_seq[t].unsqueeze(0)
    return torch.cat(y_seq, 0)

def seq_to_ann_forward(x_seq: Tensor, stateless_module: nn.Module or list or tuple or nn.Sequential):
    """
    :param x_seq: shape=[T, batch_size, ...]
    :type x_seq: Tensor
    :param stateless_module: a stateless module, e.g., 'torch.nn.Conv2d' or a list contains stateless modules, e.g., '[torch.nn.Conv2d, torch.nn.BatchNorm2d]
    :type stateless_module: torch.nn.Module or list or tuple or torch.nn.Sequential
    :return: y_seq, shape=[T, batch_size, ...]
    :rtype: Tensor

    See :class:`spikingjelly.clock_driven.layer.SeqToANNContainer` for more details.
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
    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d
    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d
    :return: the weight of this fused module
    :rtype: Tensor

    A {Conv2d-BatchNorm2d} can be fused to a {Conv2d} module with BatchNorm2d's parameters being absorbed into Conv2d.
    This function returns the weight of this fused module.

    .. admonition:: Note
        :class: note

        We assert `conv2d.bias` is `None`. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    """
    assert conv2d.bias is None
    return (conv2d.weight.transpose(0, 3) * bn2d.weight / (
                    bn2d.running_var + bn2d.eps).sqrt()).transpose(0, 3)


def fused_conv2d_bias_of_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d):
    """
    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d
    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d
    :return: the bias of this fused module
    :rtype: Tensor

    A {Conv2d-BatchNorm2d} can be fused to a {Conv2d} module with BatchNorm2d's parameters being absorbed into Conv2d.
    This function returns the bias of this fused module.

    .. admonition:: Note
        :class: note

        We assert `conv2d.bias` is `None`. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    """
    assert conv2d.bias is None
    return bn2d.bias - bn2d.running_mean * bn2d.weight / (bn2d.running_var + bn2d.eps).sqrt()


@torch.no_grad()
def scale_fused_conv2d_weight_of_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d, k=None, b=None):
    """
    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d
    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d

    A {Conv2d-BatchNorm2d} can be fused to a {Conv2d} module with BatchNorm2d's parameters being absorbed into Conv2d.
    This function sets the weight of this fused module to `weight * k + b`.

    .. admonition:: Note
        :class: note

        We assert `conv2d.bias` is `None`. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    """
    assert conv2d.bias is None
    if k is not None:
        conv2d.weight.data *= k
    if b is not None:
        conv2d.weight.data += b
        

@torch.no_grad()
def scale_fused_conv2d_bias_of_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d, k=None, b=None):
    """
    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d
    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d

    A {Conv2d-BatchNorm2d} can be fused to a {Conv2d} module with BatchNorm2d's parameters being absorbed into Conv2d.
    This function sets the bias of this fused module to `bias * k + b`.

    .. admonition:: Note
        :class: note

        We assert `conv2d.bias` is `None`. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    """
    assert conv2d.bias is None
    if k is not None:
        bn2d.bias.data *= k
        bn2d.running_mean *= k
    if b is not None:
        bn2d.bias.data += b

@torch.no_grad()
def fuse_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d, k=None, b=None):
    """
    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d
    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d
    :return: the fused Conv2d layer
    :rtype: torch.nn.Conv2d

    A {Conv2d-BatchNorm2d} can be fused to a {Conv2d} module with BatchNorm2d's parameters being absorbed into Conv2d.
    This function returns the fused module.

    .. admonition:: Note
        :class: note

        We assert `conv2d.bias` is `None`. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.
    """
    fused_conv = nn.Conv2d(in_channels=conv2d.in_channels, out_channels=conv2d.out_channels,
                     kernel_size=conv2d.kernel_size,
                     stride=conv2d.stride, padding=conv2d.padding, dilation=conv2d.dilation,
                     groups=conv2d.groups, bias=True,
                     padding_mode=conv2d.padding_mode)
    fused_conv.weight.data = fused_conv2d_weight_of_convbn2d(conv2d, bn2d)
    fused_conv.bias.data = fused_conv2d_bias_of_convbn2d(conv2d, bn2d)
    return fused_conv

def temporal_efficient_training_cross_entropy(x_seq: Tensor, target: torch.LongTensor):
    """
    :param x_seq: ``shape=[T, N, C, *]``, where ``C`` is the number of classes
    :type x_seq: Tensor
    :param target: ``shape=[N]``, where ``0 <= target[i] <= C-1``
    :type target: torch.LongTensor
    :return: the temporal efficient training cross entropy
    :rtype: Tensor

    The temporal efficient training (TET) cross entropy, which is the mean of cross entropy of each time-step.

    Codes example:

    .. code-block:: python

        def tet_ce_for_loop_version(x_seq: Tensor, target: torch.LongTensor):
            loss = 0.
            for t in range(x_seq.shape[0]):
                loss += F.cross_entropy(x_seq[t], target)
            return loss / x_seq.shape[0]

        T = 8
        N = 4
        C = 10
        x_seq = torch.rand([T, N, C])
        target = torch.randint(low=0, high=C-1, size=[N])
        print(tet_ce_for_loop_version(x_seq, target))
        print(temporal_efficient_training_cross_entropy(x_seq, target))


    .. admonition:: Tip
        :class: tip

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
    '''
    * :ref:`API in English <kaiming_normal_conv_linear_weight-en>`

    .. _reset_net-cn:

    :param net: 任何属于 ``nn.Module`` 子类的网络

    :return: None

    使用kaiming normal初始化 `net` 中的所有 :class:`torch.nn._ConvNd` 和 `:class:`torch.nn.Linear` 的权重（不包括偏置项）。参见 :class:`torch.nn.init.kaiming_normal_`。

    * :ref:`中文API <kaiming_normal_conv_linear_weight-cn>`

    .. _reset_net-en:

    :param net: Any network inherits from ``nn.Module``

    :return: None

    initialize all weights (not including bias) of :class:`torch.nn._ConvNd` and :class:`torch.nn.Linear` in `net` by the kaiming normal. See :class:`torch.nn.init.kaiming_normal_`
    for more details.
    '''
    for m in net.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))