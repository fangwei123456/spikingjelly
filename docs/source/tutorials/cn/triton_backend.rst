神经元自动执行
==========================

神经元模块根据输入张量的设备选择执行路径。CPU 使用 Torch 参考实现。CUDA 调用
注册的 PyTorch 算子，并为每个 CUDA 设备选择和缓存兼容实现。默认候选顺序为原生
CUDA、Triton、CuPy，最后为 Torch。

神经元 API 不再提供 backend 构造参数或可修改的 backend 属性。此规则适用于 IF、
LIF、PLIF、QIF、EIF、Izhikevich、I-LIF、ActivationAwareIF 和 STBIF。状态、重置、
替代梯度和步进模式仍由神经元模块管理。

诊断 CUDA 执行时，可查询实际选择的实现：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import functional, neuron

    device = torch.device("cuda:0")
    lif = neuron.LIFNode(step_mode="m").to(device)
    output = lif(torch.rand(4, 128, device=device))
    print(functional.neuron_implementation("lif", device))

查询会报告当前 provider，以及更高优先级候选不可用的原因。导入 SpikingJelly 不会
初始化 CUDA；第一次执行 CUDA 算子时才选择实现。SpikingJelly logger 只记录一次选择
结果。

高级诊断可以在 Python 启动前设置 ``SJ_<NEURON>_CUDA_IMPLEMENTATION``，例如
``SJ_LIF_CUDA_IMPLEMENTATION=triton``。默认值为 ``auto``。强制指定 provider 时，若
它不支持当前执行配置，算子会报错。未知 kernel 或执行错误也会直接报告，不会被回退
隐藏。修改这些变量后需要重启进程。

算子源码集中放在仓库根目录 ``ops/``，安装后位于 SpikingJelly 包内的
``spikingjelly._ops``。Triton 和 CuPy 保留各自的 JIT 编译及缓存。本地具备匹配的
PyTorch/CUDA 工具链时可以构建可选原生 CUDA 实现；常规 PyPI wheel 仍为纯 Python 包。
