import torch
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "kernel_dot_product",
    "spike_similar_loss",
    "temporal_efficient_training_cross_entropy",
]


def kernel_dot_product(x: Tensor, y: Tensor, kernel="linear", *args):
    r"""
    **API Language:**
    :ref:`中文 <kernel_dot_product-cn>` | :ref:`English <kernel_dot_product-en>`

    ----

    .. _kernel_dot_product-cn:

    * **中文**

    计算批量数据 ``x`` 和 ``y`` 在核空间的内积。记2个M维tensor分别为 :math:`\boldsymbol{x_{i}}` 和 :math:`\boldsymbol{y_{j}}` ， ``kernel`` 定义了不同形式的内积：

    - ``linear`` ，线性内积， :math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = \boldsymbol{x_{i}}^{T}\boldsymbol{y_{j}}`。

    - ``polynomial`` ，多项式内积， :math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = (\boldsymbol{x_{i}}^{T}\boldsymbol{y_{j}})^{d}`，其中 :math:`d = args[0]`。

    - ``sigmoid`` ，Sigmoid内积， :math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = \mathrm{sigmoid}(\alpha \boldsymbol{x_{i}}^{T}\boldsymbol{y_{j}})`，其中 :math:`\alpha = args[0]`。

    - ``gaussian`` ，高斯内积， :math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = \mathrm{exp}(- \frac{||\boldsymbol{x_{i}} - \boldsymbol{y_{j}}||^{2}}{2\sigma^{2}})` ，其中 :math:`\sigma = args[0]`。

    :param x: shape=[N, M]的tensor，看作是N个M维向量
    :type x: torch.Tensor

    :param y: shape=[N, M]的tensor，看作是N个M维向量
    :type y: torch.Tensor

    :param kernel: 计算内积时所使用的核函数
    :type kernel: str

    :param args: 用于计算内积的额外的参数

    :return: ret, shape=[N, N]的tensor， ``ret[i][j]`` 表示 ``x[i]`` 和 ``y[j]`` 的内积
    :rtype: torch.Tensor

    ----

    .. _kernel_dot_product-en:

    * **English**

    Calculate inner product of ``x`` and ``y`` in kernel space. These 2 M-dim tensors are denoted by :math:`\boldsymbol{x_{i}}` and :math:`\boldsymbol{y_{j}}`. ``kernel`` determine the kind of inner product:

    - ``linear`` -- Linear kernel, :math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = \boldsymbol{x_{i}}^{T}\boldsymbol{y_{j}}`.

    - ``polynomial`` -- Polynomial kernel, :math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = (\boldsymbol{x_{i}}^{T}\boldsymbol{y_{j}})^{d}`, where :math:`d = args[0]`.

    - ``sigmoid`` -- Sigmoid kernel, :math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = \mathrm{sigmoid}(\alpha \boldsymbol{x_{i}}^{T}\boldsymbol{y_{j}})`, where :math:`\alpha = args[0]`.

    - ``gaussian`` -- Gaussian kernel, :math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = \mathrm{exp}(- \frac{||\boldsymbol{x_{i}} - \boldsymbol{y_{j}}||^{2}}{2\sigma^{2}})`, where :math:`\sigma = args[0]`.

    :param x: Tensor of shape=[N, M]
    :type x: torch.Tensor

    :param y: Tensor of shape=[N, M]
    :type y: torch.Tensor

    :param kernel: Type of kernel function used when calculating inner products.
    :type kernel: str

    :param args: Extra parameters for inner product

    :return: ret, Tensor of shape=[N, N], ``ret[i][j]`` is inner product of ``x[i]`` and ``y[j]``.
    :rtype: torch.Tensor
    """
    if kernel == "linear":
        return x.mm(y.t())
    elif kernel == "polynomial":
        d = args[0]
        return x.mm(y.t()).pow(d)
    elif kernel == "sigmoid":
        alpha = args[0]
        return torch.sigmoid(alpha * x.mm(y.t()))
    elif kernel == "gaussian":
        sigma = args[0]
        N = x.shape[0]
        x2 = x.square().sum(dim=1)  # shape=[N]
        y2 = y.square().sum(dim=1)  # shape=[N]
        xy = x.mm(y.t())  # shape=[N, N]
        d_xy = x2.unsqueeze(1).repeat(1, N) + y2.unsqueeze(0).repeat(N, 1) - 2 * xy
        # d_xy[i][j]的元素是x[i]的平方和，加上y[j]的平方和，减去2倍的sum_{k} x[i][k]y[j][k]，因此
        # d_xy[i][j]就是x[i]和y[j]相减，平方，求和
        return torch.exp(-d_xy / (2 * sigma * sigma))
    else:
        raise NotImplementedError


def spike_similar_loss(
    spikes: Tensor, labels: Tensor, kernel_type="linear", loss_type="mse", *args
):
    r"""
    **API Language:**
    :ref:`中文 <spike_similar_loss-cn>` | :ref:`English <spike_similar_loss-en>`

    ----

    .. _spike_similar_loss-cn:

    * **中文**

    将N个数据输入到输出层有M个神经元的SNN，运行T步，得到shape=[N, M, T]的脉冲。这N个数据的标签为shape=[N, C]的 ``labels``。

    用shape=[N, N]的矩阵 ``sim`` 表示 **实际相似度矩阵**， ``sim[i][j] == 1`` 表示
    数据i与数据j相似，反之亦然。若 ``labels[i]`` 与 ``labels[j]`` 共享至少同一个标签，则
    认为他们相似，否则不相似。

    用shape=[N, N]的矩阵 ``sim_p`` 表示 **输出相似度矩阵** ， ``sim_p[i][j]`` 的取值
    为0到1，值越大表示数据i与数据j的脉冲越相似。

    使用内积来衡量两个脉冲之间的相似性， ``kernel_type`` 是计算内积时，所使用的核函数种类：

    - ``linear`` ，线性内积， :math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = \boldsymbol{x_{i}}^{T}\boldsymbol{y_{j}}`。

    - ``sigmoid`` ，Sigmoid内积， :math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = \mathrm{sigmoid}(\alpha \boldsymbol{x_{i}}^{T}\boldsymbol{y_{j}})`，其中 :math:`\alpha = args[0]`。

    - ``gaussian`` ，高斯内积，:math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = \mathrm{exp}(- \frac{||\boldsymbol{x_{i}} - \boldsymbol{y_{j}}||^{2}}{2\sigma^{2}})`，其中 :math:`\sigma = args[0]`。

    当使用Sigmoid或高斯内积时，内积的取值范围均在[0, 1]之间；而使用线性内积时，为了保证内积取值仍然在[0, 1]之间，会进行归一化，按照 :math:`\text{sim}_p[i][j]=\frac{\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}})}{||\boldsymbol{x_{i}}|| · ||\boldsymbol{y_{j}}||}`。

    对于相似的数据，根据输入的 ``loss_type`` ，返回度量 ``sim`` 与 ``sim_p`` 差异的损失：

    - ``mse`` -- 返回sim与sim_p的均方误差（也就是l2误差）。

    - ``l1`` -- 返回sim与sim_p的l1误差。

    - ``bce`` -- 返回sim与sim_p的二值交叉熵误差。

    .. note::
        脉冲向量稀疏、离散，最好先使用高斯核进行平滑，然后再计算相似度。

    :param spikes: shape=[N, M, T]，N个数据生成的脉冲
    :type spikes: torch.Tensor

    :param labels: shape=[N, C]，N个数据的标签， ``labels[i][k] == 1`` 表示数据i属于
        第k类，反之亦然，允许多标签
    :type labels: torch.Tensor

    :param kernel_type: 使用内积来衡量两个脉冲之间的相似性， ``kernel_type`` 是计算内积时，
        所使用的核函数种类
    :type kernel_type: str

    :param loss_type: 返回哪种损失，可以为'mse', 'l1', 'bce'
    :type loss_type: str

    :param args: 用于计算内积的额外参数

    :return: shape=[1]的tensor，相似损失
    :rtype: torch.Tensor

    ----

    .. _spike_similar_loss-en:

    * **English**

    A SNN consisting M neurons will receive a batch of N input data in each timestep (from 0 to T-1) and output a spike tensor of shape=[N, M, T]. The label is a tensor of shape=[N, C].

    The **groundtruth similarity matrix** ``sim`` has a shape of [N, N]. ``sim[i][j] == 1`` indicates that input i is similar to input j and vice versa. If and only if ``labels[i]`` and ``labels[j]`` have at least one common label, they are viewed as similar.

    The **output similarity matrix** ``sim_p`` has a shape of [N, N]. The value of ``sim_p[i][j]`` ranges from 0 to 1, represents the similarity between output spike from both input i and input j.

    The similarity is measured by inner product of two spikes. ``kernel_type`` is the type of kernel function when calculating inner product:

    - ``linear``, Linear kernel, :math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = \boldsymbol{x_{i}}^{T}\boldsymbol{y_{j}}`.

    - ``sigmoid``, Sigmoid kernel, :math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = \mathrm{sigmoid}(\alpha \boldsymbol{x_{i}}^{T}\boldsymbol{y_{j}})`, where :math:`\alpha = args[0]`.

    - ``gaussian``, Gaussian kernel，:math:`\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}}) = \mathrm{exp}(- \frac{||\boldsymbol{x_{i}} - \boldsymbol{y_{j}}||^{2}}{2\sigma^{2}})`, where :math:`\sigma = args[0]`.

    When Sigmoid or Gaussian kernel is applied, the inner product naturally lies in :math:`[0, 1]`. To make the value consistent when using linear kernel, the result will be normalized as: :math:`\text{sim}_p[i][j]=\frac{\kappa(\boldsymbol{x_{i}}, \boldsymbol{y_{j}})}{||\boldsymbol{x_{i}}|| · ||\boldsymbol{y_{j}}||}`.

    For similar data, return the specified discrepancy loss between ``sim`` and ``sim_p`` according to ``loss_type``.

    - ``mse`` -- Return the Mean-Square Error (squared L2 norm) between sim and sim_p.

    - ``l1`` -- Return the L1 error between sim and sim_p.

    - ``bce`` -- Return the Binary Cross Entropy between sim and sim_p.

    .. admonition:: Note
        :class: note

        Since spike vectors are usually discrete and sparse, it would be better to apply Gaussian filter first to smooth the vectors before calculating similarities.

    :param spikes: shape=[N, M, T], output spikes corresponding to a batch of N inputs
    :type spikes: torch.Tensor

    :param labels: shape=[N, C], labels of inputs, ``labels[i][k] == 1`` means
        the i-th input belongs to the k-th category and vice versa.
        Multi-label input is allowed.
    :type labels: torch.Tensor

    :param kernel_type: Type of kernel function used when calculating inner
        products. The inner product is the similarity measure of two spikes.
    :type kernel_type: str

    :param loss_type: Type of loss returned. Can be: 'mse', 'l1', 'bce'
    :type loss_type: str

    :param args: Extra parameters for inner product

    :return: shape=[1], similarity loss
    :rtype: torch.Tensor
    """
    spikes = spikes.flatten(start_dim=1)

    sim_p = kernel_dot_product(spikes, spikes, kernel_type, *args)

    if kernel_type == "linear":
        spikes_len = spikes.norm(p=2, dim=1, keepdim=True)
        sim_p = sim_p / ((spikes_len.mm(spikes_len.t())) + 1e-8)

    labels = labels.float()
    sim = labels.mm(labels.t()).clamp_max(
        1
    )  # labels.mm(labels.t())[i][j]位置的元素表现输入数据i和数据数据j有多少个相同的标签
    # 将大于1的元素设置为1，因为共享至少同一个标签，就认为他们相似

    if loss_type == "mse":
        return F.mse_loss(sim_p, sim)
    elif loss_type == "l1":
        return F.l1_loss(sim_p, sim)
    elif loss_type == "bce":
        return F.binary_cross_entropy(sim_p, sim)
    else:
        raise NotImplementedError


@torch.jit.script
def temporal_efficient_training_cross_entropy(x_seq: Tensor, target: Tensor):
    """
    **API Language:**
    :ref:`中文 <temporal_efficient_training_cross_entropy-cn>` | :ref:`English <temporal_efficient_training_cross_entropy-en>`

    ----

    .. _temporal_efficient_training_cross_entropy-cn:

    * **中文**

    Temporal efficient training (TET) 交叉熵损失, 是每个时间步的交叉熵损失的平均。

    .. note::

        TET交叉熵是 `Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting <https://openreview.net/forum?id=_XNtisL32jv>`_ 一文提出的。

    :param x_seq: ``shape=[T, N, C, *]`` 的预测值，其中 ``C`` 是类别总数
    :type x_seq: torch.Tensor

    :param target: ``shape=[N]`` 的真实值，其中 ``target[i]`` 是真实类别
    :type target: torch.Tensor

    :return: 损失值
    :rtype: torch.Tensor

    ----

    .. _temporal_efficient_training_cross_entropy-en:

    * **English**

    The temporal efficient training (TET) cross entropy, which is the mean of cross entropy of each time-step.

    .. admonition:: Note
        :class: note

        The TET cross entropy is proposed by `Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting <https://openreview.net/forum?id=_XNtisL32jv>`_.

    :param x_seq: the predicted value with ``shape=[T, N, C, *]``, where ``C`` is the number of classes
    :type x_seq: torch.Tensor

    :param target: the ground truth tensor with ``shape=[N]``, where ``target[i]`` is the label
    :type target: torch.Tensor

    :return: the temporal efficient training cross entropy
    :rtype: torch.Tensor

    ----

    * **示例代码 | Example**

    .. code-block:: python

        def tet_ce_for_loop_version(x_seq: torch.Tensor, target: torch.LongTensor):
            loss = 0.0
            for t in range(x_seq.shape[0]):
                loss += F.cross_entropy(x_seq[t], target)
            return loss / x_seq.shape[0]


        T = 8
        N = 4
        C = 10
        x_seq = torch.rand([T, N, C])
        target = torch.randint(low=0, high=C - 1, size=[N])
        print(
            f"max error = {(tet_ce_for_loop_version(x_seq, target) - temporal_efficient_training_cross_entropy(x_seq, target)).abs().max()}"
        )
        # max error < 1e-6
    """
    # x_seq.shape = [T, N, C, *]
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
