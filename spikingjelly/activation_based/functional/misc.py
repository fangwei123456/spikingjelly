import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from torch import fx
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .. import base, layer, neuron


__all__ = [
    "set_threshold_margin",
    "redundant_one_hot",
    "first_spike_index",
    "kaiming_normal_conv_linear_weight",
    "delay",
    "fuse_conv_bn_eval_modules",
    "pack_conv_bn_train_modules",
]


def _matches_module_pattern(pattern, node: fx.Node, modules) -> bool:
    if len(node.args) == 0:
        return False
    nodes = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != "call_module":
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def _replace_node_module(
    node: fx.Node, modules, new_module: torch.nn.Module
) -> None:
    def parent_name(target: str):
        *parent, name = target.rsplit(".", 1)
        return parent[0] if parent else "", name

    assert isinstance(node.target, str)
    parent, name = parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent], name, new_module)


class _EvalFusionTracer(fx.Tracer):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(
            m,
            (
                _TrainConvBnWrapper,
                layer.Conv1d,
                layer.Conv2d,
                layer.Conv3d,
                layer.BatchNorm1d,
                layer.BatchNorm2d,
                layer.BatchNorm3d,
                base.StepModule,
                base.MemoryModule,
                neuron.BaseNode,
            ),
        ):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class _TrainPackTracer(_EvalFusionTracer):
    pass


class _TrainConvBnWrapper(nn.Module):
    def __init__(self, conv: nn.Module, bn: nn.Module):
        super().__init__()
        self.conv = conv
        self.bn = bn

    def _packed_forward(self, x: Tensor) -> Tensor:
        t, n = x.shape[:2]
        x = x.flatten(0, 1)
        if isinstance(self.conv, layer.Conv1d):
            x = nn.Conv1d.forward(self.conv, x)
        elif isinstance(self.conv, layer.Conv2d):
            x = nn.Conv2d.forward(self.conv, x)
        elif isinstance(self.conv, layer.Conv3d):
            x = nn.Conv3d.forward(self.conv, x)
        else:
            raise TypeError(f"Unsupported packed conv type: {type(self.conv)!r}")

        if isinstance(self.bn, (layer.BatchNorm1d, layer.BatchNorm2d, layer.BatchNorm3d)):
            x = self.bn.super_forward(x)
        else:
            x = self.bn(x)
        return x.view(t, n, *x.shape[1:])

    def forward(self, x: Tensor) -> Tensor:
        if (
            isinstance(self.conv, (layer.Conv1d, layer.Conv2d, layer.Conv3d))
            and isinstance(self.bn, (layer.BatchNorm1d, layer.BatchNorm2d, layer.BatchNorm3d))
            and getattr(self.conv, "step_mode", None) == "m"
            and getattr(self.bn, "step_mode", None) == "m"
        ):
            return self._packed_forward(x)
        return self.bn(self.conv(x))


def fuse_conv_bn_eval_modules(net: nn.Module) -> fx.GraphModule:
    """
    **API Language:**
    :ref:`中文 <fuse_conv_bn_eval_modules-cn>` | :ref:`English <fuse_conv_bn_eval_modules-en>`

    ----

    .. _fuse_conv_bn_eval_modules-cn:

    * **中文**

    将评估模式下模型中的相邻 ``Conv*`` 与 ``BatchNorm*`` 模块融合为单个卷积模块。
    该函数同时支持原生 ``torch.nn`` 模块以及 SpikingJelly 的
    :class:`spikingjelly.activation_based.layer.Conv1d`,
    :class:`spikingjelly.activation_based.layer.Conv2d`,
    :class:`spikingjelly.activation_based.layer.Conv3d`,
    :class:`spikingjelly.activation_based.layer.BatchNorm1d`,
    :class:`spikingjelly.activation_based.layer.BatchNorm2d`,
    :class:`spikingjelly.activation_based.layer.BatchNorm3d`。

    输入模型必须处于 ``eval()`` 模式。返回值是融合后的 ``fx.GraphModule``。

    .. _fuse_conv_bn_eval_modules-en:

    * **English**

    Fuse adjacent ``Conv*`` and ``BatchNorm*`` modules in an evaluation-mode model
    into a single convolution module. Both native ``torch.nn`` layers and
    SpikingJelly activation-based ``layer.Conv*`` / ``layer.BatchNorm*`` wrappers
    are supported.

    The input model must be in ``eval()`` mode. The returned value is a fused
    ``fx.GraphModule``.
    """

    if net.training:
        raise ValueError("fuse_conv_bn_eval_modules only supports eval() models.")

    tracer = _EvalFusionTracer()
    graph = tracer.trace(net)
    fx_model = fx.GraphModule(tracer.root, graph)
    modules = dict(fx_model.named_modules())
    patterns = [
        (nn.Conv1d, nn.BatchNorm1d),
        (nn.Conv2d, nn.BatchNorm2d),
        (nn.Conv3d, nn.BatchNorm3d),
        (layer.Conv1d, layer.BatchNorm1d),
        (layer.Conv2d, layer.BatchNorm2d),
        (layer.Conv3d, layer.BatchNorm3d),
    ]

    for pattern in patterns:
        for node in list(fx_model.graph.nodes):
            if not _matches_module_pattern(pattern, node, modules):
                continue
            if len(node.args[0].users) > 1:
                continue
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            fused_conv = fuse_conv_bn_eval(conv, bn)
            _replace_node_module(node.args[0], modules, fused_conv)
            node.replace_all_uses_with(node.args[0])
            fx_model.graph.erase_node(node)

    fx_model.graph.lint()
    fx_model.delete_all_unused_submodules()
    fx_model.recompile()
    return fx_model


def pack_conv_bn_train_modules(net: nn.Module) -> fx.GraphModule:
    """
    **API Language:**
    :ref:`中文 <pack_conv_bn_train_modules-cn>` | :ref:`English <pack_conv_bn_train_modules-en>`

    ----

    .. _pack_conv_bn_train_modules-cn:

    * **中文**

    将训练模式下模型中的相邻 ``Conv*`` 与 ``BatchNorm*`` 模块打包为单个 wrapper，
    以减少多步 ``Conv -> BatchNorm`` 路径中的 ``view/flatten`` 往返。

    该函数不会像 ``fuse_conv_bn_eval_modules`` 那样融合权重；它只是将相邻层包装成一个
    compile-friendly 的训练模块。当前同时支持原生 ``torch.nn`` 的 ``Conv*`` / ``BatchNorm*``
    模块，以及 SpikingJelly activation-based ``layer.Conv*`` / ``layer.BatchNorm*`` 模块。

    输入模型必须处于 ``train()`` 模式。返回值是变换后的 ``fx.GraphModule``。

    .. _pack_conv_bn_train_modules-en:

    * **English**

    Pack adjacent ``Conv*`` and ``BatchNorm*`` modules in a training-mode model into
    a single wrapper to reduce redundant ``view/flatten`` hops along multi-step
    ``Conv -> BatchNorm`` paths.

    Unlike ``fuse_conv_bn_eval_modules``, this transform does not fuse weights. It
    only rewrites the module graph into a more compile-friendly training structure.
    Both native ``torch.nn`` layers and SpikingJelly activation-based
    ``layer.Conv*`` / ``layer.BatchNorm*`` wrappers are supported.

    The input model must be in ``train()`` mode. The returned value is the packed
    ``fx.GraphModule``.
    """
    if not net.training:
        raise ValueError("pack_conv_bn_train_modules only supports train() models.")

    tracer = _TrainPackTracer()
    graph = tracer.trace(net)
    fx_model = fx.GraphModule(tracer.root, graph)
    modules = dict(fx_model.named_modules())
    patterns = [
        (nn.Conv1d, nn.BatchNorm1d),
        (nn.Conv2d, nn.BatchNorm2d),
        (nn.Conv3d, nn.BatchNorm3d),
        (layer.Conv1d, layer.BatchNorm1d),
        (layer.Conv2d, layer.BatchNorm2d),
        (layer.Conv3d, layer.BatchNorm3d),
    ]

    for pattern in patterns:
        for node in list(fx_model.graph.nodes):
            if not _matches_module_pattern(pattern, node, modules):
                continue
            if len(node.args[0].users) > 1:
                continue
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            if (
                isinstance(conv, (layer.Conv1d, layer.Conv2d, layer.Conv3d))
                and isinstance(bn, (layer.BatchNorm1d, layer.BatchNorm2d, layer.BatchNorm3d))
                and getattr(conv, "step_mode", None) != getattr(bn, "step_mode", None)
            ):
                continue
            packed = _TrainConvBnWrapper(conv, bn)
            _replace_node_module(node.args[0], modules, packed)
            node.replace_all_uses_with(node.args[0])
            fx_model.graph.erase_node(node)

    fx_model.graph.lint()
    fx_model.delete_all_unused_submodules()
    fx_model.recompile()
    return fx_model


def set_threshold_margin(
    output_layer: neuron.BaseNode,
    label_one_hot: Tensor,
    eval_threshold=1.0,
    threshold0=0.9,
    threshold1=1.1,
):
    """
    **API Language:**
    :ref:`中文 <set_threshold_margin-cn>` | :ref:`English <set_threshold_margin-en>`

    ----

    .. _set_threshold_margin-cn:

    * **中文**

    对于用来分类的网络，为输出层神经元的电压阈值设置一定的裕量，以获得更好的分类性能。

    类别总数为C，网络的输出层共有C个神经元。网络在训练时，当输入真实类别为i的数据，
    输出层中第i个神经元的电压阈值会被设置成 ``threshold1`` ，而其他神经元的电压阈值会被设置成
    ``threshold0`` 。而在测试（推理）时，输出层中神经元的电压阈值被统一设置成 ``eval_threshold`` 。

    :param output_layer: 用于分类的网络的输出层，输出层输出shape=[batch_size, C]
    :type output_layer: neuron.BaseNode

    :param label_one_hot: one hot格式的样本标签，shape=[batch_size, C]
    :type label_one_hot: torch.Tensor

    :param eval_threshold: 输出层神经元在测试（推理）时使用的电压阈值
    :type threshold0: float

    :param threshold0: 输出层神经元在训练时，负样本的电压阈值
    :type threshold1: float

    :param threshold1: 输出层神经元在训练时，正样本的电压阈值
    :type threshold1: float

    :return: None

    ----

    .. _set_threshold_margin-en:

    * **English**

    Set voltage threshold margin for neurons in the output layer to reach better performance in classification task.

    When there are C different classes, the output layer contains C neurons.
    During training, when the input with groundtruth label i are sent into the network,
    the voltage threshold of the i-th neurons in the output layer will be set to
    ``threshold1`` and the remaining will be set to ``threshold0``.

    During inference, the voltage thresholds of **ALL** neurons in the output layer
    will be set to ``eval_threshold``.

    :param output_layer: The output layer of classification network, where the shape of output should be [batch_size, C]
    :type output_layer: neuron.BaseNode

    :param label_one_hot: Labels in one-hot format, shape=[batch_size, C]
    :type label_one_hot: torch.Tensor

    :param eval_threshold: Voltage threshold of neurons in output layer when evaluating (inference)
    :type threshold0: float

    :param threshold0: Voltage threshold of the corresponding neurons of **negative** samples in output layer when training
    :type threshold1: float

    :param threshold1: Voltage threshold of the corresponding neurons of **positive** samples in output layer when training
    :type threshold1: float

    :return: None
    """
    if output_layer.training:
        output_layer.v_threshold = torch.ones_like(label_one_hot) * threshold0
        output_layer.v_threshold[label_one_hot == 1] = threshold1
    else:
        output_layer.v_threshold = eval_threshold


def redundant_one_hot(labels: Tensor, num_classes: int, n: int):
    """
    **API Language:**
    :ref:`中文 <redundant_one_hot-cn>` | :ref:`English <redundant_one_hot-en>`

    ----

    .. _redundant_one_hot-cn:

    * **中文**

    对数据进行冗余的one-hot编码，每一类用 ``n`` 个1和 ``(num_classes - 1) * n`` 个0来编码。

    :param labels: shape=[batch_size]的tensor，表示 ``batch_size`` 个标签
    :type labels: torch.Tensor

    :param num_classes: 类别总数
    :type num_classes: int

    :param n: 表示每个类别所用的编码数量
    :type n: int

    :return: 形状为 ``[batch_size, num_classes * n]`` 的tensor
    :rtype: torch.Tensor

    ----

    .. _redundant_one_hot-en:

    * **English**

    Redundant one-hot encoding for data. Each class is encoded to ``n`` 1's and  ``(num_classes - 1) * n`` 0's

    :param labels: Tensor of shape=[batch_size], ``batch_size`` labels
    :type labels: torch.Tensor

    :param num_classes: The total number of classes.
    :type num_classes: int

    :param n: The encoding length for each class.
    :type n: int

    :return: Tensor of shape ``[batch_size, num_classes * n]``
    :rtype: torch.Tensor

    ----

    * **代码示例 | Example**

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
    **API Language:**
    :ref:`中文 <first_spike_index-cn>` | :ref:`English <first_spike_index-en>`

    ----

    .. _first_spike_index-cn:

    * **中文**

    输入若干个神经元的输出脉冲，返回一个与输入相同shape的 ``bool`` 类型的index。
    index为 ``True`` 的位置，表示该神经元首次释放脉冲的时刻。

    :param spikes: ``shape=[*, T]`` ，表示任意个神经元在 :math:`t=0, 1, ..., T-1` ， 共T个时刻的输出脉冲
    :type spikes: torch.Tensor

    :return: ``index`` ， ``shape=[*, T]`` ，为 ``True`` 的位置表示该神经元首次释放脉冲的时刻
    :rtype: torch.Tensor

    ----

    .. _first_spike_index-en:

    * **English**

    Return an ``index`` tensor of the same shape of input tensor, which is the output spike of some neurons. The location of ``True`` represents the moment of first spike.

    :param spikes: ``shape=[*, T]``, indicates the output spikes of some neurons when :math:`t=0, 1, ..., T-1`.
    :type spikes: torch.Tensor

    :return: index, ``shape=[*, T]``, the index of ``True`` represents the moment of first spike.
    :rtype: torch.Tensor

    ----

    * **代码示例 | Example**

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


def kaiming_normal_conv_linear_weight(net: nn.Module):
    """
    **API Language:**
    :ref:`中文 <kaiming_normal_conv_linear_weight-cn>` | :ref:`English <kaiming_normal_conv_linear_weight-en>`

    ----

    .. _kaiming_normal_conv_linear_weight-cn:

    * **中文**

    使用kaiming normal初始化 ``net` `中的所有 :class:`torch.nn._ConvNd` 和 :class:`torch.nn.Linear` 的权重（不包括偏置项）。
    参见 :class:`torch.nn.init.kaiming_normal_`。

    :param net: 任何属于 ``nn.Module`` 子类的网络
    :type net: torch.nn.Module

    :return: None

    ----

    .. _kaiming_normal_conv_linear_weight-en:

    * **English**

    Initialize all weights (not including bias) of :class:`torch.nn._ConvNd` and :class:`torch.nn.Linear` in ``net`` by the kaiming normal.
    See :class:`torch.nn.init.kaiming_normal_`
    for more details.

    :param net: Any network inherits from ``nn.Module``
    :type net: torch.nn.Module

    :return: None
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))


def delay(x_seq: torch.Tensor, delay_steps: int):
    """
    **API Language:**
    :ref:`中文 <delay-cn>` | :ref:`English <delay-en>`

    ----

    .. _delay-cn:

    * **中文**

    延迟函数，可以用来延迟输入，使得 ``y[t] = x[t - delay_steps]``。缺失的数据用0填充。

    :param x_seq: 输入的序列，``shape = [T, *]``
    :type x_seq: torch.Tensor

    :param delay_steps: 延迟的时间步数
    :type delay_steps: int

    :return: 延迟后的序列
    :rtype: torch.Tensor

    ----

    .. _delay-en:

    * **English**

    A delay function that can delay inputs and makes ``y[t] = x[t - delay_steps]``. The nonexistent data will be regarded as 0.

    :param x_seq: the input sequence with ``shape = [T, *]``
    :type x_seq: torch.Tensor

    :param delay_steps: the number of delayed time-steps
    :type delay_steps: int

    :return: the delayed sequence
    :rtype: torch.Tensor

    ----

    * **代码示例 | Example**

    .. code:: python

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

    Output:

    .. code:: bash

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
    y = torch.zeros_like(x_seq[0:delay_steps].data)
    return torch.cat((y, x_seq[0 : x_seq.shape[0] - delay_steps]), 0)
