ANN转换SNN
=======================================
本页作者：`DingJianhao <https://github.com/DingJianhao>`_、`fangwei123456 <https://github.com/fangwei123456>`_、`Lv Liuzhenghao <https://github.com/Lyu6PosHao>`_、`黄一凡 (AllenYolk) <https://github.com/AllenYolk>`_

English version: :doc:`../en/ann2snn`

.. admonition:: ANN2SNN 教程版本导航

    ANN2SNN public API 经历了三代教程：

    #. :doc:`更早期 clock-driven 时代 ANN2SNN API <../../legacy_tutorials/cn/5_ann2snn>`。
    #. :doc:`legacy pre-Recipe Converter API <../../legacy_tutorials/cn/ann2snn_converter_legacy>`，使用 ``Converter(mode=..., dataloader=...)`` 和 ``convert_to_spiking_neurons(model)``。
    #. 当前 Recipe API，即本页内容：``RateCodingRecipe`` 或 ``TransformerSpikeEquivalentRecipe`` 定义算法，``Converter.convert(model)`` 执行转换。

本节教程主要关注 ``spikingjelly.activation_based.ann2snn``。它展示如何使用当前 Recipe API 将训练好的前馈 ANN 转换为 SNN，并在 SpikingJelly 中仿真转换后的模型。

相关API见此处 `API参考 <https://spikingjelly.readthedocs.io/zh_CN/latest/spikingjelly.activation_based.ann2snn.html>`_ 。

当前实现基于 ``torch.fx``。``torch.fx`` 会将 ``nn.Module`` 实例 trace 为计算图表示，然后由 ANN2SNN recipe 对该计算图进行变换。

ANN转换SNN的理论基础
--------------------

与 ANN 相比，SNN 使用离散脉冲通信，这有利于高效通信，但直接训练 SNN 往往需要更多资源。一个实用路线是先训练 ANN，再将其转换为行为相近的 SNN。这就引出了如何连接 ANN 激活值和 SNN 发放率的问题。对于 rate-coded SNN，输出类别通常由脉冲计数读出。因此关键问题是：脉冲神经元的发放率能否近似 ANN 神经元的激活值？

ANN 中的 ReLU 激活与采用减法重置的 IF 神经元发放率有很强的相关性，其中膜电位会通过减去 :math:`V_{threshold}` 重置。``RateCodingRecipe`` 利用这一关系进行转换。这里的神经元更新方式就是 `神经元教程 <https://spikingjelly.readthedocs.io/zh_CN/latest/tutorials/cn/neuron.html>`_ 中介绍的 Soft reset 方式。

实验：IF神经元脉冲发放频率和输入的关系
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们给与恒定输入到IF神经元，观察其输出脉冲和脉冲发放频率。首先导入相关的模块，新建IF神经元层，确定输入并画出每个IF神经元的输入 :math:`x_{i}`：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron
    from spikingjelly import visualizing
    from matplotlib import pyplot as plt
    import numpy as np

    plt.rcParams['figure.dpi'] = 200
    if_node = neuron.IFNode(v_reset=None)
    T = 128
    x = torch.arange(-0.2, 1.2, 0.04)
    plt.scatter(torch.arange(x.shape[0]), x)
    plt.title('Input $x_{i}$ to IF neurons')
    plt.xlabel('Neuron index $i$')
    plt.ylabel('Input $x_{i}$')
    plt.grid(linestyle='-.')
    plt.show()

.. image:: ../../_static/tutorials/5_ann2snn/0.*
    :width: 100%

接下来，将输入送入到IF神经元层，并运行 ``T=128`` 步，观察各个神经元发放的脉冲、脉冲发放频率：

.. code-block:: python

    s_list = []
    for t in range(T):
        s_list.append(if_node(x).unsqueeze(0))

    out_spikes = np.asarray(torch.cat(s_list))
    visualizing.plot_1d_spikes(out_spikes, 'IF neurons\' spikes and firing rates', 't', 'Neuron index $i$')
    plt.show()

.. image:: ../../_static/tutorials/5_ann2snn/1.*
    :width: 100%

脉冲发放频率在一定范围内与输入 :math:`x_{i}` 的大小成正比。

接下来，让我们画出IF神经元脉冲发放频率和输入 :math:`x_{i}` 的曲线，并与 :math:`\mathrm{ReLU}(x_{i})` 对比：

.. code-block:: python

    plt.subplot(1, 2, 1)
    firing_rate = np.mean(out_spikes, axis=0)
    plt.plot(x, firing_rate)
    plt.title('Input $x_{i}$ and firing rate')
    plt.xlabel('Input $x_{i}$')
    plt.ylabel('Firing rate')
    plt.grid(linestyle='-.')

    plt.subplot(1, 2, 2)
    plt.plot(x, x.relu())
    plt.title('Input $x_{i}$ and ReLU($x_{i}$)')
    plt.xlabel('Input $x_{i}$')
    plt.ylabel('ReLU($x_{i}$)')
    plt.grid(linestyle='-.')
    plt.show()

.. image:: ../../_static/tutorials/5_ann2snn/2.*
    :width: 100%

两者的曲线几乎一致。但脉冲频率不可能高于1，因此IF神经元无法拟合ANN中ReLU输入大于1的情况。

理论证明
^^^^^^^^

文献 [#f1]_ 为ANN转SNN提供了理论基础，证明SNN中的IF神经元是ReLU激活函数在时间上的无偏估计器。

针对神经网络第一层，考虑 SNN 神经元发放率 :math:`r` 和对应 ANN 激活值之间的关系。假定输入恒定为 :math:`z \in [0,1]`。
对于采用减法重置的IF神经元，其膜电位V随时间变化为：

.. math::
    V_t=V_{t-1}+z-V_{threshold}\theta_t

其中：
 :math:`V_{threshold}` 为发放阈值，通常设为1.0。 :math:`\theta_t` 为输出脉冲。 :math:`T` 时间步内的平均发放率可以通过对膜电位求和得到：

.. math::
    \sum_{t=1}^{T} V_t= \sum_{t=1}^{T} V_{t-1}+z T-V_{threshold} \sum_{t=1}^{T}\theta_t

将含有 :math:`V_t` 的项全部移项到左边，两边同时除以 :math:`T` ：

.. math::
    \frac{V_T-V_0}{T} = z - V_{threshold}  \frac{\sum_{t=1}^{T}\theta_t}{T} = z- V_{threshold}  \frac{N}{T}

其中 :math:`N` 为 :math:`T` 时间窗口内的脉冲数， :math:`\frac{N}{T}` 就是发放率  :math:`r`。利用  :math:`z= V_{threshold} a`
即：

.. math::
    r = a- \frac{ V_T-V_0 }{T V_{threshold}}

故在仿真时间步  :math:`T` 无限长情况下:

.. math::
    r = a (a>0)

类似地，针对神经网络更高层，文献 [#f1]_ 进一步说明层间发放率满足：

.. math::
    r^l = W^l r^{l-1}+b^l- \frac{V^l_T}{T V_{threshold}}

完整推导见文献 [#f1]_ 。ann2snn中的方法基于该文献

转换到脉冲神经网络
^^^^^^^^^^^^^^^^^^^^^^^^

转换主要解决两个问题：

1. ANN为了快速训练和收敛提出了批归一化（Batch Normalization）。批归一化旨在将ANN输出归一化到0均值，这与SNN的特性相违背。因此，可以将BN的参数吸收到前面的参数层中（Linear、Conv2d）

2. 根据转换理论，ANN的每层输入输出需要被限制在[0,1]范围内，这就需要对参数进行缩放（模型归一化）

◆ BatchNorm参数吸收

假定BatchNorm的参数为 :math:`\gamma` (``BatchNorm.weight``)， :math:`\beta` (``BatchNorm.bias``)， :math:`\mu` (``BatchNorm.running_mean``) ，
:math:`\sigma` (``BatchNorm.running_var``，:math:`\sigma = \sqrt{\mathrm{running\_var}}`)。具体参数定义详见
`torch.nn.BatchNorm1d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d>`_ 。
参数模块（例如 Linear）具有权重 :math:`W` 和偏置 :math:`b` 。BatchNorm 参数吸收就是将 BatchNorm 的参数通过运算转移到参数模块的 :math:`W` 和 :math:`b` 中，使得数据输入新模块的输出和有 BatchNorm 时相同。
对此，新模型的 :math:`\bar{W}` 和 :math:`\bar{b}` 公式表示为：

.. math::
    \bar{W} = \frac{\gamma}{\sigma}  W

.. math::
    \bar{b} = \frac{\gamma}{\sigma} (b - \mu) + \beta

◆ 模型归一化

对于某个参数模块，假定得到了其输入张量和输出张量，其输入张量的最大值为 :math:`\lambda_{pre}` ,输出张量的最大值为 :math:`\lambda`
那么，归一化后的权重 :math:`\hat{W}` 为：

.. math::
    \hat{W} = W * \frac{\lambda_{pre}}{\lambda}

归一化后的偏置 :math:`\hat{b}` 为：

.. math::
    \hat{b} = \frac{b}{\lambda}

ANN每层输出中常常存在较大的离群值，导致整体神经元发放率降低。
鲁棒归一化将缩放因子从张量最大值改为张量的p分位点，推荐分位点为99.9% [#f1]_。

BatchNorm 融合和模型归一化是在脉冲替换前进行的代数变换。随后 rate-coding
recipe 会将 ReLU 激活替换为 IF 神经元。
对于 ANN 中的平均池化，转换后的模型保留空间下采样。由于 IF 神经元会在时间上近似
ReLU 激活，在空间下采样后立即再增加一个 IF 神经元通常对结果影响很小。
当前 rate-coding recipe 没有通用的最大池化转换规则。文献中使用基于动量累计脉冲的门控函数控制脉冲通道 [#f1]_ 。因此，本教程的示例模型仍推荐使用
``AvgPool2d``。
仿真时，依照该转换理论，转换后的 SNN 应输入恒定的模拟输入。使用 Poisson 编码器可能引入额外的准确率损失。

实现与可选配置
^^^^^^^^^^^^^^^^^^^^^^^^

当前 ann2snn API 将 **运行什么转换算法** 和 **如何执行转换流程** 分开：

* recipe 持有算法参数和算法相关的图变换。
* ``Converter`` 是执行器。它接收 recipe，用 ``torch.fx`` trace 模型，并通过
  ``convert(model)`` 依次调用 recipe 的转换步骤。

如果要做 ReLU-to-IFNode 的 rate-coding 转换，使用 ``RateCodingRecipe``。
该 recipe 需要校准 dataloader，因为它会在替换 ReLU 前统计激活范围。常见模式包括：

* ``mode="max"``：MaxNorm，使用最大激活值 [#f2]_。
* ``mode="99.9%"``：RobustNorm，使用 99.9% 激活分位点 [#f1]_。
* ``mode`` 为 ``(0, 1]`` 内的浮点数：按该比例缩放最大激活值。

``RateCodingRecipe`` 也持有 ``fuse_flag`` 等选项。当 ``fuse_flag=True``
（默认值）时，Conv-BatchNorm 会在校准前被融合。

最小 rate-coding 调用如下：

.. code-block:: python

    recipe = ann2snn.RateCodingRecipe(dataloader=train_loader, mode="max")
    snn = ann2snn.Converter(recipe=recipe).convert(ann)

转换后 ReLU 模块被删除，SNN 需要的新模块（包括 ``VoltageScaler``、
``IFNode`` 等）会作为 ``spiking_*`` 子模块创建在原父模块下。由于转换后模型
的类型为 ``fx.GraphModule``，所以可以使用 ``snn.graph.print_tabular()``
查看生成的计算图。更多 API 参见 `GraphModule <https://pytorch.org/docs/stable/fx.html?highlight=graphmodule#torch.fx.GraphModule>`_。

.. note::

    当前版本要求使用 ``Converter(recipe=...)`` 和 ``Converter.convert(model)``。
    旧 public algorithm methods 已移除，包括 ``convert_to_spiking_neurons()``、
    ``replace_by_td_operators()``、``fuse()``、``set_voltagehook()``、
    ``replace_by_neurons()`` 和 ``replace_by_ifnode()``。


识别MNIST
---------

构建并加载 ANN
^^^^^^^^^^^^^^^^^^^^^^^^

现在我们使用 ``ann2snn`` ，搭建一个简单卷积网络，对MNIST数据集进行分类。

完整的可运行示例是 ``spikingjelly.activation_based.ann2snn.examples.cnn_mnist``。
网络结构定义在 ``ann2snn.sample_models.mnist_cnn`` 中：

.. code-block:: python

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),

                nn.Conv2d(32, 32, 3, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),

                nn.Conv2d(32, 32, 3, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),

                nn.Flatten(),
                nn.Linear(32, 10),
            )

        def forward(self, x):
            x = self.network(x)
            return x

注意：如果遇到需要将tensor展开的情况，就在网络中定义一个 ``nn.Flatten`` 模块，在forward函数中需要使用定义的Flatten而不是view函数。

定义基本运行选项：

.. code-block:: python

    torch.random.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_dir = "./data/mnist"
    batch_size = 100
    T = 50

``T`` 是 SNN 推理时使用的仿真步数。转换前先创建 MNIST dataloader：

.. code-block:: python

    train_data_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    calibration_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_data_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data_dataset, batch_size=50, shuffle=True, drop_last=False
    )

示例脚本会下载预训练 checkpoint。先加载并验证 ANN：

.. code-block:: python

    ann2snn.download_url(
        "https://ndownloader.figshare.com/files/34960191",
        "./SJ-mnist-cnn_model-sample.pth",
    )
    model = mnist_cnn.CNN().to(device)
    model.load_state_dict(torch.load("SJ-mnist-cnn_model-sample.pth", map_location=device))
    acc = val(model, device, test_data_loader)
    print('ANN Validating Accuracy: %.4f' % (acc))

本次运行中，ANN 准确率为：

.. code-block:: shell

    ANN Validating Accuracy: 0.9870

使用Converter进行转换
^^^^^^^^^^^^^^^^^^^^^^^^

现在 ANN 已经完成训练和验证。接下来选择上文介绍的 rate-coding recipe，
传入确定性校准 dataloader，再由 ``Converter`` 执行转换：

.. code-block:: python

    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode="max")
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)

``snn_model`` 就是转换后的 SNN 模型。查看它的网络结构可以发现
``BatchNorm2d`` 模块不见了，这是因为默认的 rate-coding recipe 会在校准前
将 BatchNorm 参数吸收进前面的 Conv 层：

.. code-block:: python

    CNN(
      (network): Module(
        (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
        (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (8): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (12): Flatten(start_dim=1, end_dim=-1)
        (13): Linear(in_features=32, out_features=10, bias=True)
        (spiking_0): Module(
          (scaler0): VoltageScaler(0.193247)
          (if_node): IFNode(
            v_threshold=1.0, v_reset=None, detach_reset=False, step_mode=s, backend=torch
            (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
          )
          (scaler1): VoltageScaler(5.174733)
        )
        (spiking_1): Module(
          (scaler0): VoltageScaler(0.325697)
          (if_node): IFNode(
            v_threshold=1.0, v_reset=None, detach_reset=False, step_mode=s, backend=torch
            (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
          )
          (scaler1): VoltageScaler(3.070336)
        )
        (spiking_2): Module(
          (scaler0): VoltageScaler(0.121967)
          (if_node): IFNode(
            v_threshold=1.0, v_reset=None, detach_reset=False, step_mode=s, backend=torch
            (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
          )
          (scaler1): VoltageScaler(8.198915)
        )
      )
    )

snn_model的类型为 ``GraphModule`` ，参见 `GraphModule <https://pytorch.org/docs/stable/fx.html?highlight=graphmodule#torch.fx.GraphModule>`_ 。

调用 ``GraphModule.graph.print_tabular()`` 方法，用表格的形式查看模型的计算图的中间表示：

.. code-block:: shell

    # snn_model.graph.print_tabular()
    opcode       name                       target                     args                          kwargs
    -----------  -------------------------  -------------------------  ----------------------------  ------
    placeholder  x                          x                          ()                            {}
    call_module  network_0                  network.0                  (x,)                          {}
    call_module  network_spiking_0_scaler0  network.spiking_0.scaler0  (network_0,)                  {}
    call_module  network_spiking_0_if_node  network.spiking_0.if_node  (network_spiking_0_scaler0,)  {}
    call_module  network_spiking_0_scaler1  network.spiking_0.scaler1  (network_spiking_0_if_node,)  {}
    call_module  network_3                  network.3                  (network_spiking_0_scaler1,)  {}
    call_module  network_4                  network.4                  (network_3,)                  {}
    call_module  network_spiking_1_scaler0  network.spiking_1.scaler0  (network_4,)                  {}
    call_module  network_spiking_1_if_node  network.spiking_1.if_node  (network_spiking_1_scaler0,)  {}
    call_module  network_spiking_1_scaler1  network.spiking_1.scaler1  (network_spiking_1_if_node,)  {}
    call_module  network_7                  network.7                  (network_spiking_1_scaler1,)  {}
    call_module  network_8                  network.8                  (network_7,)                  {}
    call_module  network_spiking_2_scaler0  network.spiking_2.scaler0  (network_8,)                  {}
    call_module  network_spiking_2_if_node  network.spiking_2.if_node  (network_spiking_2_scaler0,)  {}
    call_module  network_spiking_2_scaler1  network.spiking_2.scaler1  (network_spiking_2_if_node,)  {}
    call_module  network_11                 network.11                 (network_spiking_2_scaler1,)  {}
    call_module  network_12                 network.12                 (network_11,)                 {}
    call_module  network_13                 network.13                 (network_12,)                 {}
    output       output                     output                     (network_13,)                 {}

其它 recipe 与自定义 recipe
^^^^^^^^^^^^^^^^^^^^^^^^^^^

上面的 MNIST 示例使用 rate coding，因为它要把 ReLU 激活转换为 IF 神经元。
对于 Transformer 模型，可以改用 ``TransformerSpikeEquivalentRecipe``：

.. code-block:: python

    recipe = ann2snn.TransformerSpikeEquivalentRecipe()
    td_model = ann2snn.Converter(recipe=recipe).convert(transformer_ann)

该 recipe 不需要 dataloader，不插入 ``VoltageHook``，也不运行 rate-coding
校准。它会把当前支持的 ANN 模块和 attention 调用替换为 TD /
spike-equivalent 算子，但不承诺完整的 fully spike-driven LLM 转换。

如果要添加新的转换算法，可以继承 ``ConversionRecipe``，只覆盖需要改变的
步骤方法。未覆盖的方法会使用基类的默认 no-op 实现。Recipe 不是执行器，
不应提供 ``convert()``、``run()`` 或 ``__call__()``：

.. code-block:: python

    class MyRecipe(ann2snn.ConversionRecipe):
        def replace(self, converter, fx_model):
            # Implement the algorithm-specific graph rewrite here.
            return fx_model

固定步骤顺序为 ``validate`` -> ``before_trace`` -> ``after_trace`` -> ``insert_observers`` -> ``calibrate`` -> ``replace`` -> ``finalize``。

自定义转换规则
^^^^^^^^^^^^^^

默认情况下，``RateCodingRecipe`` 使用 ``ReLURule`` 将 ``nn.ReLU`` 模块替换为
``VoltageScaler(1 / s) -> IFNode -> VoltageScaler(s)``。其中校准尺度 ``s``
由 ``ThresholdOptimizer`` 基于 ``VoltageHook`` 计算得到。

高级用户可以向 ``RateCodingRecipe`` 显式传入以下扩展点：

* ``rules`` 负责匹配 FX 计算图节点、插入校准 hook、查找完成校准的
  activation-hook 节点对，并把它们替换为 SNN 子图。
* ``NeuronFactory`` 负责创建替换时使用的神经元。默认工厂创建
  ``IFNode(v_threshold=1.0, v_reset=None)``。
* ``ThresholdOptimizer`` 负责从已校准的 ``VoltageHook`` 计算有限正标量阈值。

下面的最小规则只演示协议形状，不表示新的转换算法。它匹配 ``nn.Identity``，
并将完成校准的 identity 节点替换为另一个 ``nn.Identity`` 模块：

.. code-block:: python

    import torch
    import torch.nn as nn

    from spikingjelly.activation_based import ann2snn
    from spikingjelly.activation_based.ann2snn.modules import VoltageHook


    class IdentityRule:
        def match(self, node, modules):
            return node.op == "call_module" and type(modules[node.target]) is nn.Identity

        def insert_hooks(self, fx_model, node, hook_factory, hook_counts_per_prefix):
            target = f'{node.target}_voltage_hook'
            fx_model.add_submodule(target, hook_factory.create())
            with fx_model.graph.inserting_after(node):
                return fx_model.graph.call_module(target, args=(node,))

        def find_replacements(self, fx_model, modules):
            for hook_node in fx_model.graph.nodes:
                if hook_node.op == "call_module" and isinstance(
                    modules.get(hook_node.target), VoltageHook
                ):
                    yield hook_node.args[0], hook_node

        def replace_with_neurons(
            self, fx_model, activation_node, hook_node, neuron_factory, threshold_optimizer
        ):
            hook = fx_model.get_submodule(hook_node.target)
            threshold = threshold_optimizer.compute_threshold(hook)
            # IdentityRule 不使用脉冲阈值，但真实规则应将该标量接入替换子图。
            target = f'{activation_node.target}_spiking_identity'
            fx_model.add_submodule(target, nn.Identity())
            with fx_model.graph.inserting_after(hook_node):
                new_node = fx_model.graph.call_module(target, args=activation_node.args)
            hook_node.replace_all_uses_with(new_node)
            activation_node.replace_all_uses_with(new_node)
            fx_model.graph.erase_node(hook_node)
            fx_model.graph.erase_node(activation_node)


    recipe = ann2snn.RateCodingRecipe(
        dataloader=[torch.randn(2, 4)],
        rules=[IdentityRule()],
    )
    converter = ann2snn.Converter(recipe=recipe)

当前内置 ``ReLURule`` 路径只对标量阈值有明确语义。Per-channel 或张量阈值还需要先定义
shape、broadcast 和 ``VoltageScaler`` 语义，不属于当前转换路径。

不同转换模式的对比
^^^^^^^^^^^^^^^^^^^^^^^^

按照这个例子，我们分别定义模式为 ``max`` ，``99.9%``，``1.0/2``，``1.0/3``，``1.0/4``， ``1.0/5`` 情况下的SNN转换并分别推理T步得到准确率。

.. code-block:: python

    print('---------------------------------------------')
    print('Converting using MaxNorm')
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode="max")
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)
    print('Simulating...')
    mode_max_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_max_accs[-1]))

    print('---------------------------------------------')
    print('Converting using RobustNorm')
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode="99.9%")
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)
    print('Simulating...')
    mode_robust_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_robust_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/2 max(activation) as scales...')
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode=1.0 / 2)
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)
    print('Simulating...')
    mode_two_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_two_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/3 max(activation) as scales')
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode=1.0 / 3)
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)
    print('Simulating...')
    mode_three_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_three_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/4 max(activation) as scales')
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode=1.0 / 4)
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)
    print('Simulating...')
    mode_four_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_four_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/5 max(activation) as scales')
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode=1.0 / 5)
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)
    print('Simulating...')
    mode_five_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_five_accs[-1]))

以下输出是在 NVIDIA GeForce RTX 4090 上以 ``T=50`` 测得：

.. code-block:: shell

    ANN Validating Accuracy: 0.9870
    ---------------------------------------------
    Converting using MaxNorm
    Calibration: 3.76s
    Simulating...
    Simulation: 7.61s
    SNN accuracy (simulation 50 time-steps): 0.9771
    ---------------------------------------------
    Converting using RobustNorm
    Calibration: 12.08s
    Simulating...
    Simulation: 7.31s
    SNN accuracy (simulation 50 time-steps): 0.9848
    ---------------------------------------------
    Converting using 1/2 max(activation) as scales
    Calibration: 3.73s
    Simulating...
    Simulation: 7.28s
    SNN accuracy (simulation 50 time-steps): 0.9846
    ---------------------------------------------
    Converting using 1/3 max(activation) as scales
    Calibration: 3.72s
    Simulating...
    Simulation: 7.29s
    SNN accuracy (simulation 50 time-steps): 0.9825
    ---------------------------------------------
    Converting using 1/4 max(activation) as scales
    Calibration: 3.75s
    Simulating...
    Simulation: 7.27s
    SNN accuracy (simulation 50 time-steps): 0.9734
    ---------------------------------------------
    Converting using 1/5 max(activation) as scales
    Calibration: 3.70s
    Simulating...
    Simulation: 7.26s
    SNN accuracy (simulation 50 time-steps): 0.9472

在这个示例中，RobustNorm 和适中的缩放比例通常比 MaxNorm 在 SNN 早期时间步收敛得更快。
根据模型输出的随时间变化的准确率，我们可以绘制不同设置下的准确率图像。

.. code-block:: python

    fig = plt.figure()
    plt.plot(np.arange(0, T), mode_max_accs, label="mode: max")
    plt.plot(np.arange(0, T), mode_robust_accs, label="mode: 99.9%")
    plt.plot(np.arange(0, T), mode_two_accs, label="mode: 1.0/2")
    plt.plot(np.arange(0, T), mode_three_accs, label="mode: 1.0/3")
    plt.plot(np.arange(0, T), mode_four_accs, label="mode: 1.0/4")
    plt.plot(np.arange(0, T), mode_five_accs, label="mode: 1.0/5")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Acc")
    plt.show()

.. image:: ../../_static/tutorials/5_ann2snn/accuracy_mode_recipe_api.png

不同设置会在早期时间步收敛速度和最终准确率之间形成取舍。用户可以根据延迟和准确率需求选择归一化模式。

.. [#f1] Rueckauer B, Lungu I-A, Hu Y, Pfeiffer M and Liu S-C (2017) Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification. Front. Neurosci. 11:682.
.. [#f2] Diehl, Peter U. , et al. Fast classifying, high-accuracy spiking deep networks through weight and threshold balancing. Neural Networks (IJCNN), 2015 International Joint Conference on IEEE, 2015.

其他参考文献：

* Rueckauer, B., Lungu, I. A., Hu, Y., & Pfeiffer, M. (2016). Theory and tools for the conversion of analog to spiking convolutional neural networks. arXiv preprint arXiv:1612.04052.
* Sengupta, A., Ye, Y., Wang, R., Liu, C., & Roy, K. (2019). Going deeper in spiking neural networks: Vgg and residual architectures. Frontiers in neuroscience, 13, 95.
