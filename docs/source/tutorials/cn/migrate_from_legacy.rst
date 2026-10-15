升级到 V2
=======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

English version: :doc:`../en/migrate_from_legacy`

V2 包含不兼容的接口变化，请按下表更新旧配置。

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

子包重命名
-------------------------------------------
新版的SpikingJelly对子包进行了重命名，与老版本的对应关系为：

===============  ==================
老版本            新版本             
===============  ==================
clock_driven     activation_based
event_driven     timing_based    
===============  ==================

早期版本的迁移示例见 `0.0.0.0.14 版本文档 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.14/>`_。
