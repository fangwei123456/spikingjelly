从老版本迁移
=======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

English version: :doc:`../en/migrate_from_legacy`

本页先介绍 V2 的接口迁移，再保留 ``<=0.0.0.0.12`` 的历史子包迁移说明。
V2 有 breaking changes，旧配置需按下表修改。

V2：自动执行与接口迁移
-------------------------------------------

.. list-table:: 旧用法与当前用法
    :header-rows: 1
    :widths: 40 60

    * - 旧用法
      - 当前用法
    * - 神经元 ``backend=``、修改 ``.backend``
      - 删除配置，模块和输入移动到同一设备
    * - ``functional.set_backend``、``supported_backends``
      - 删除调用；排查时使用 ``functional.neuron_implementation``
    * - backend 专用 functional 函数
      - 使用公开的 ``if_step``、``lif_step`` 或 ``*_multi_step``；核对参数和返回值
    * - ``cuda_kernel/``、``triton_kernel/`` 私有导入
      - 使用公开 neuron、functional 或 precision API，不导入 ``spikingjelly._ops``
    * - Experimental IF/LIF/PLIF 类
      - 使用 ``IFNode``、``LIFNode``、``ParametricLIFNode``
    * - Auto CUDA 及旧代码生成／推理图工具
      - 自定义动力学使用 ``FlexSN``，不再维护用户生成的旧 kernel
    * - ``FlexSNKernel``、``FlexSN.kernel``
      - 使用 ``FlexSN.functional_forward``，按新签名传入状态及静态输入
    * - ``SpikeLinear``、``SpikeConv*``、``spike_linear``、``spike_conv*``
      - 普通 Linear/Conv 加 memopt；专门算法使用保留的融合或 packed/sparse 投影
    * - CuPy 依赖和 ``cupy11``／``cupy12`` extras
      - 删除安装项；按需安装 Triton或本地构建原生 CUDA 扩展

旧代码（不可在当前版本运行）：

.. code-block:: text

    neuron.LIFNode(step_mode="m", backend="cupy")
    functional.set_backend(net, "triton")

当前完整示例：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron, functional

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    node = neuron.LIFNode(step_mode="m").to(device)
    x = torch.rand(4, 2, 8, device=device, requires_grad=True)
    node(x).sum().backward()
    functional.reset_net(node)

IF/LIF/PLIF 序列 functional 接口显式接收初态，返回脉冲、最终状态和可选轨迹。例如：

.. code-block:: python

    spikes, v_final, v_seq = functional.lif_multi_step(
        x, torch.zeros_like(x[0]), tau=2.0, store_v_seq=True
    )

单步 ``lif_step`` 返回 ``(spike, v_next)``；多步 ``lif_multi_step`` 返回三个值。
迁移时按 :doc:`./neuron` 与公开 API 核对参数和返回值，避免只改函数名后缀。

正常使用无需选择实现。安装与诊断见 :doc:`./triton_backend`，精度配置见
:doc:`./precision`，自定义动力学见 :doc:`./flexsn`，省显存与投影见 :doc:`./memopt`。
不提供自动迁移脚本，也不保证旧完整模块 pickle/checkpoint 可直接恢复。
优先使用可信来源的 ``state_dict``，在当前模型定义下检查键和形状。

历史迁移：<=0.0.0.0.12
-------------------------------------------

下文保留早期子包和步进模式迁移说明；旧版本一侧的示例不能直接用于当前版本。
推荐同时阅读 :doc:`./basic_concept`。

子包重命名
-------------------------------------------
新版的SpikingJelly对子包进行了重命名，与老版本的对应关系为：

===============  ==================
老版本            新版本             
===============  ==================
clock_driven     activation_based
event_driven     timing_based    
===============  ==================

单步多步模块和传播模式
-------------------------------------------
``<=0.0.0.0.12`` 的老版本SpikingJelly，在默认情况下所有模块都是单步的，除非其名称含有前缀 ``MultiStep``。\
而新版的SpikingJelly，则不再使用前缀对单步和多步模块进行区分，取而代之的是同一个模块，拥有单步和多步两种步进模式，\
使用 ``step_mode`` 进行控制。具体信息可以参见 :doc:`./basic_concept`。

因而在新版本中不再有单独的多步模块，取而代之的则是融合了单步和多步的统一模块。例如，在老版本的SpikingJelly中，若想使用单步LIF神经元，\
是按照如下方式：

.. code-block:: python

    from spikingjelly.clock_driven import neuron

    lif = neuron.LIFNode()

在新版本中，所有模块默认是单步的，所以与老版本的代码几乎相同，除了将 ``clock_driven`` 换成了 ``activation_based``：

.. code-block:: python

    from spikingjelly.activation_based import neuron

    lif = neuron.LIFNode()

在老版本的SpikingJelly中，若想使用多步LIF神经元，是按照如下方式：

.. code-block:: python

    from spikingjelly.clock_driven import neuron

    lif = neuron.MultiStepLIFNode()

在新版本中，单步和多步模块进行了统一，因此只需要指定为多步模块即可：

.. code-block:: python

    from spikingjelly.activation_based import neuron

    lif = neuron.LIFNode(step_mode='m')


在老版本中，若想分别搭建一个逐步传播和逐层传播的网络，按照如下方式：

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.clock_driven import neuron, layer, functional

    with torch.no_grad():

        T = 4
        N = 2
        C = 4
        H = 8
        W = 8
        x_seq = torch.rand([T, N, C, H, W])

        # step-by-step
        net_sbs = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            neuron.IFNode()
        )
        y_seq = functional.multi_step_forward(x_seq, net_sbs)
        # y_seq.shape = [T, N, C, H, W]
        functional.reset_net(net_sbs)



        # layer-by-layer
        net_lbl = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(C),
            ),
            neuron.MultiStepIFNode()
        )
        y_seq = net_lbl(x_seq)
        # y_seq.shape = [T, N, C, H, W]
        functional.reset_net(net_lbl)


而在新版本中，由于单步和多步模块已经融合，可以通过 :class:`spikingjelly.activation_based.functional.set_step_mode` 对整个网络的步进模式进行转换。\
在所有模块使用单步模式时，整个网络就可以使用逐步传播；所有模块都使用多步模式时，整个网络就可以使用逐层传播：

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, layer, functional

    with torch.no_grad():

        T = 4
        N = 2
        C = 4
        H = 8
        W = 8
        x_seq = torch.rand([T, N, C, H, W])

        # the network uses step-by-step because step_mode='s' is the default value for all modules
        net = nn.Sequential(
            layer.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(C),
            neuron.IFNode()
        )
        y_seq = functional.multi_step_forward(x_seq, net)
        # y_seq.shape = [T, N, C, H, W]
        functional.reset_net(net)

        # set the network to use layer-by-layer
        functional.set_step_mode(net, step_mode='m')
        y_seq = net(x_seq)
        # y_seq.shape = [T, N, C, H, W]
        functional.reset_net(net)
