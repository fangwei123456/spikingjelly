与 NIR 相互转换
#########################

本教程作者： `黄一凡 (AllenYolk) <https://github.com/AllenYolk>`_

English version: :doc:`../en/nir_exchange`

`Neuromorphic intermediate representation (NIR) <https://neuroir.org/docs/index.html>`_ 是一组计算原语，以图（节点+边）的形式描述了 SNN 的模块和连接，在不同的神经形态框架和技术栈之间通用。目前，NIR `被多个模拟器和硬件平台支持 <https://neuroir.org/docs/support.html>`_ 。SpikingJelly ``0.0.0.1.0`` 引入了 ``nir_exchange`` 包，使（满足一定条件的） SpikingJelly 模型能够与 NIR 图互相转换。借助 NIR 这一中间表示，用户可以更加轻松地进行硬件部署和框架迁移。

.. figure:: ../../_static/tutorials/nir_exchange/nir-schema.png
    :width: 100%

    图片来源： `What is the Neuromorphic Intermediate Representation (NIR)? <https://neuroir.org/docs/what.html>`_

SpikingJelly 的 ``nir_exchange`` 包提供了两个关键的用户接口：

* :func:`export_to_nir <spikingjelly.activation_based.nir_exchange.to_nir.export_to_nir>` ：将 SpikingJelly 模型导出为 NIR 图；
* :func:`import_from_nir <spikingjelly.activation_based.nir_exchange.from_nir.import_from_nir>` ：将 NIR 图导入为 SpikingJelly 模型。

本教程将对这两个函数展开介绍。

从 SpikingJelly 到 NIR
==========================

由于开发者精力有限且 NIR 本身只能表示少数几种模块，故目前 :func:`export_to_nir <spikingjelly.activation_based.nir_exchange.to_nir.export_to_nir>` 只支持以下 SpikingJelly / PyTorch 模块的转换：

* ``torch.nn.Linear``, :class:`layer.Linear <spikingjelly.activation_based.layer.Linear>`
* ``torch.nn.Conv2d``, :class:`layer.Conv2d <spikingjelly.activation_based.layer.Conv2d>`
* ``torch.nn.AvgPool2d``, :class:`layer.AvgPool2d <spikingjelly.activation_based.layer.AvgPool2d>`
* ``torch.nn.Flatten``, :class:`layer.Flatten <spikingjelly.activation_based.layer.Flatten>`
* :class:`IFNode <spikingjelly.activation_based.neuron.IFNode>`
* :class:`LIFNode <spikingjelly.activation_based.neuron.LIFNode>` and :class:`ParametricLIFNode <spikingjelly.activation_based.neuron.ParametricLIFNode>`

以下面的 SNN 模型为例：

.. code:: python

    import torch.nn as nn
    from spikingjelly.activation_based import layer, neuron

    net = nn.Sequential(
        layer.Conv2d(3, 16, 3, 1, 1, step_mode="s"),
        neuron.IFNode(),
        nn.AvgPool2d((2, 2)),
        layer.Flatten(step_mode="s"),
        nn.Linear(4096, 10),
        neuron.ParametricLIFNode(10., decay_input=False, v_reset=None),
    )

为了展示兼容性，这一示例故意混用了原生 PyTorch 的无状态层 ``nn.AvgPool2d, nn.Linear`` 和 SpikingJelly 包装后的无状态层 ``layer.Conv2d, layer.Flatten``。此外，本例中还使用了 ``neuron.IFNode`` 和 ``neuron.ParametricLIFNode`` 两种神经元模型。

调用 :func:`export_to_nir <spikingjelly.activation_based.nir_exchange.to_nir.export_to_nir>` ，即可将上述模型转换成 NIR 图并保存为 HDF5 文件：

.. code:: python

    import torch
    from spikingjelly.activation_based import nir_exchange

    graph = nir_exchange.export_to_nir(
        net,
        example_input=torch.rand(8, 3, 32, 32),
        save_path="./example.h5",
        dt=1e-4
    )
    print(graph)

:func:`export_to_nir <spikingjelly.activation_based.nir_exchange.to_nir.export_to_nir>` 参数的含义为：

* ``net`` ：SpikingJelly 模型；
* ``example_input`` ：模型输入的样例，用于确定子模块输入和输出的形状；
* ``save_path`` ：HDF5 文件路径，用于保存 NIR 图（若为 ``None`` ，则不保存）；
* ``dt`` ：NIR 模拟时间步长。建议设置成 ``1e-4`` 以对齐其它支持 NIR 的框架。

运行后，当前目录下出现文件 ``example.h5`` ，内含 NIR 图。终端打印出的结果大致为：

.. code:: text

    NIRGraph(
        nodes={
            'input_1': Input(input_type={'input': array([ 3, 32, 32])}, metadata={}),

            '_0': Conv2d(input_shape=(32, 32), weight=array(...), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=array(...), metadata={}),

            '_1': IF(r=array(...), v_threshold=array(...), v_reset=array(...), input_type={'input': array([16, 32, 32])}, output_type={'output': array([16, 32, 32])}, metadata={}),

            '_2': AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, metadata={}),

            '_3': Flatten(input_type={'input': array([16, 16, 16])}, start_dim=0, end_dim=-1, output_type={'output': array([4096])}, metadata={}),

            '_4': Affine(weight=array(...), bias=array(...), input_type={'input': array([4096])}, output_type={'output': array([10])}, metadata={}),

            '_5': LIF(tau=array(...), r=array(...), v_leak=array(...), v_threshold=array(...), v_reset=array(...), input_type={'input': array([10])}, output_type={'output': array([10])}, metadata={}),

            'output': Output(output_type={'output': array([10])}, metadata={})
        },

        edges=[
            ('input_1', '_0'), ('_0', '_1'), ('_1', '_2'), ('_2', '_3'),
            ('_3', '_4'), ('_4', '_5'), ('_5', 'output')
        ],

        input_type={'input_1': array([ 3, 32, 32])},
        output_type={'output': array([10])},
        metadata={}
    )

这里，我们只展示了 ``NIRGraph`` 的结构，省略了具体的参数数值。可见，NIR 图由节点 ``nodes`` 和边 ``edges`` 组成。节点对应 SNN 模块，边指示了节点的输入输出关系。

.. note::

    原模型中的 ``ParametricLIFNode`` 被转换成了 ``nir.LIF`` 节点。这是合理的，因为一旦膜电位时间常量 ``tau`` 固定下来，PLIF 神经元就将变成 LIF 神经元。

.. note::

    不同于 PyTorch 和 SpikingJelly 模型， ``NIRGraph`` 中的节点大多蕴含 **输入输出形状** 信息。例如，上方例子中的 ``'_3': Flatten(...)`` 节点指明了输入形状为 ``[16, 16, 16]`` ，输出形状为 ``[4096]`` ； ``'_5': LIF(...)`` 的输入输出形状则都为 ``[10]`` 。显然，NIR 图中的形状信息是不包含时间维度 ``T`` 和批量维度 ``B`` 的；换言之，NIR只 **描述单样本、单个时间步上的模型结构** 。

    PyTorch / SpikingJelly 模型的子模块不含输入输出形状信息，但 NIR 图却需要这些信息。为了获取输入输出形状信息，:func:`export_to_nir <spikingjelly.activation_based.nir_exchange.to_nir.export_to_nir>` 要求用户给出 ``example_input`` 样例输入。 ``example_input`` 可以具有时间或批量维度，具体取决于 PyTorch / SpikingJelly 模型的需求。 :func:`export_to_nir <spikingjelly.activation_based.nir_exchange.to_nir.export_to_nir>` 函数内部将调用 PyTorch 的 `ShapeProp <https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/shape_prop.py>`_ 功能来获取输入输出形状信息。

从 NIR 到 SpikingJelly
==========================

函数 :func:`import_from_nir <spikingjelly.activation_based.nir_exchange.from_nir.import_from_nir>` 可以将已有的 NIR 图转换成 SpikingJelly 模型。以上一节生成的 NIR 图为例：

.. code:: python

    gm = nir_exchange.import_from_nir(graph="./example.h5", dt=1e-4)
    print(gm)
    x = torch.rand(9, 3, 32, 32) # [B, C, H, W]
    y = gm(x) # forward pass
    print("y.shape =", y[0].shape) # y is a tuple; the 2nd element is each layer's state

此处，:func:`import_from_nir <spikingjelly.activation_based.nir_exchange.from_nir.import_from_nir>` 参数的含义是：

* ``graph`` ：若给出字符串，则视为保存有 NIR 图的 HDF5 文件路径；函数内部将从该文件中读出 ``NIRGraph`` 。若给出 ``NIRGraph`` 对象，则直接使用该对象。
* ``dt`` ：NIR 图的模拟时间步长。与 :func:`export_to_nir <spikingjelly.activation_based.nir_exchange.to_nir.export_to_nir>` 的 ``dt`` 参数一致。

该函数返回一个 ``torch.fx.GraphModule`` 对象，可以像 ``torch.nn.Module`` 那样直接调用以运行前向传播。前向传播返回一个二元组，第一个元素是模型的输出，第二个元素则是模型内子模块的状态字典（绝大多数情况下无需使用）。上方代码块的终端输出大致为：

.. code:: text

    GraphModule(
      (_0): Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), step_mode=s)
      (_1): IFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=s, backend=torch
        (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
      )
      (_2): AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, step_mode=s)
      (_3): Flatten(start_dim=1, end_dim=-1, step_mode=s)
      (_4): Linear(in_features=4096, out_features=10, bias=True)
      (_5): LIFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=s, backend=torch, tau=10.0
        (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
      )
    )

    def forward(self, input, state : typing_Dict[str,typing_Any] = {'_0': None, '_1': None, '_2': None, '_3': None, '_4': None, '_5': None, 'input_1': None, 'output': None}):
        ones = torch.ones(1);  ones = None
        input_1 = input
        _0 = self._0(input_1);  input_1 = None
        _1 = self._1(_0);  _0 = None
        _2 = self._2(_1);  _1 = None
        _3 = self._3(_2);  _2 = None
        _4 = self._4(_3);  _3 = None
        _5 = self._5(_4);  _4 = None
        return (_5, state)

    # To see more debug info, please use `graph_module.print_readable()`

    y.shape = torch.Size([9, 10])

可见，NIR 图被正确地转换成了 SpikingJelly 模型。模型的无状态层均来自 ``spikingjelly.activation_based.layer`` ，可以配置步进模式（见 ``step_mode`` 属性）。

目前， :func:`import_from_nir <spikingjelly.activation_based.nir_exchange.from_nir.import_from_nir>` 仅支持以下 NIR 节点类型：

* ``nir.Linear``, ``nir.Affine``
* ``nir.Conv2d``
* ``nir.AvgPool2d``
* ``nir.Flatten``
* ``nir.IF``
* ``nir.LIF``

.. note::

    :func:`import_from_nir <spikingjelly.activation_based.nir_exchange.from_nir.import_from_nir>` 还提供了 ``dtype`` ， ``device`` 和 ``step_mode`` 参数，用于控制所返回的 SpikingJelly 模型的数据类型、设备、步进模式。例如，可以通过以下方式得到多步模式的 SpikingJelly 模型：

    .. code:: python

        gm = nir_exchange.import_from_nir(
            "./example.h5", dt=1e-4, step_mode="m"
        )
        print(gm)
        x = torch.rand(7, 9, 3, 32, 32) # [T, B, C, H, W]
        y = gm(x)
        print("y.shape =", y[0].shape)