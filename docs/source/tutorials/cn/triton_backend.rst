神经元自动执行
==========================

English version: :doc:`../en/triton_backend`

从 CPU 到 CUDA
----------------------------

普通神经元不需要 backend 参数或可修改的 backend 属性。只需创建神经元、将模块
和输入移动到目标设备。以下代码在 CPU 或 NVIDIA CUDA 上采用相同的训练流程：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import functional, neuron, surrogate

    torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for node_type in (neuron.IFNode, neuron.LIFNode, neuron.ParametricLIFNode):
        node = node_type(
            step_mode="m", surrogate_function=surrogate.ATan(), store_v_seq=True
        ).to(device)
        x = torch.rand(4, 2, 8, device=device, requires_grad=True)  # [T, N, C]
        parameters = list(node.parameters())
        optimizer = torch.optim.SGD(parameters, lr=0.01) if parameters else None
        before = [p.detach().clone() for p in parameters]
        spikes = node(x)
        loss = spikes.sum() + node.v.sum()
        loss.backward()
        assert spikes.shape == x.shape and torch.isfinite(x.grad).all()
        if optimizer is not None:
            assert all(p.grad is not None for p in parameters)
            optimizer.step()
            assert any(not torch.equal(old, p) for old, p in zip(before, parameters))
        functional.reset_net(node)  # Reset after backward/update.

独立 batch 在反向和参数更新后重置；连续序列的状态保留与 detach 见 :doc:`./neuron`。
安装与可选本地 CUDA 构建见 :doc:`/index`。

自动执行与能力边界
----------------------------

CPU 使用 Torch 参考实现。CUDA 在设备和执行路径首次使用时检查兼容实现，之后
复用选择，不在线测速。导入 SpikingJelly 不初始化 CUDA。eager 当前优先兼容的原生
CUDA，再检查 Triton 和 Torch；Inductor 展开优先 Triton，再检查原生 CUDA 和 Torch。
已验证 GPU 的优先级来自离线测试，不能保证某种实现对所有 workload 都最快。

统一执行接口覆盖 IF、LIF、PLIF、QIF、EIF、Izhikevich、I-LIF、ActivationAwareIF 和
STBIF。这不表示每类都支持训练、任意 dtype 或任意替代梯度；具体限制见各类 API。
其他神经元仍按其公开契约执行，不能据此推断它们都有独立 CUDA 内核。

普通融合路径支持的输入 dtype 为 FP32/FP16/BF16，并要求 FP32 状态和受支持的内置
替代梯度。低精度状态、自定义替代梯度等情况可以使用 Torch 参考公式。显式神经元
精度是另一种配置，见 :doc:`./precision`。FlexSN 的自定义 core 见 :doc:`./flexsn`。

编译与 CUDA Graph
----------------------------

eager 和 Inductor 展开有独立的自动选择。下面使用默认 FP32 状态，在编译前完成
一次设备初始化及 eager 前后向，并重置预热产生的状态：

.. code-block:: python

    import torch
    from torch import nn
    from spikingjelly.activation_based import functional, neuron, surrogate

    device = torch.device("cuda:0")
    model = (
        nn.Sequential(
            nn.Linear(16, 16),
            neuron.LIFNode(
                step_mode="m", surrogate_function=surrogate.ATan(), store_v_seq=True
            ),
            nn.Linear(16, 4),
        )
        .to(device)
        .train()
    )
    x = torch.rand(4, 2, 16, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # Initialize device/state and warm up an eager forward/backward.
    model(x).square().mean().backward()
    optimizer.zero_grad(set_to_none=True)
    functional.reset_net(model)
    compiled = torch.compile(model, fullgraph=True)
    output = compiled(x)
    output.square().mean().backward()
    optimizer.step()
    functional.reset_net(model)
    assert output.shape == (4, 2, 4)

示例面向 PyTorch Inductor。其他编译后端需要单独验证。显式精度组合应先按
:doc:`./precision` 初始化或预热，再进行图捕获。

首次调用包含加载、编译或 JIT 成本，不应当作稳定运行耗时。CUDA Graph 沿用被捕获
函数预热时的选择：捕获 eager 函数不会自动切换到 Triton，捕获 compiled 函数则
沿用编译路径。捕获前还需按 PyTorch CUDA Graph 规则完成内存和前后向预热。

编译可能改变融合与舍入，不能承诺与 eager 逐位一致。比较性能时保持模型、输入、
状态精度、reset 和同步方法一致；标准流程见仓库 ``benchmark/README.md``。

查询和日志
----------------------------

普通训练不需要查询实现。排查 CUDA 执行时，可查询 eager 或 Inductor 展开路径：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import functional

    device = torch.device("cuda:0")
    print(functional.neuron_implementation("lif", device))
    print(functional.neuron_implementation("lif", device, execution="compile"))

查询会初始化并缓存对应选择，不执行神经元计算，也不修改模块状态。返回字典含
``implementation`` 和 ``unavailable``，后者记录更早候选不可用的原因。
它不是单次调用的 profiler：低精度状态、自定义 surrogate 和显式精度配置仍遵循各自
路径，不能仅凭查询结果判断某次调用的实现。

包日志默认禁用，首次选择后不重复记录。应用可以在入口启用 INFO；启用前的日志
不会补发。``logger.remove()`` 影响全局 Loguru sinks，应仅由应用决定是否执行：

.. code-block:: python

    import sys
    from spikingjelly.logger import logger

    # Configure sinks at the application entry point, before feature imports.
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.enable("spikingjelly")

    import torch
    from spikingjelly.activation_based import neuron

    node = neuron.LIFNode(step_mode="m").cuda()
    node(torch.rand(4, 2, 8, device="cuda"))
    node.reset()

详细日志配置见 :doc:`/APIs/spikingjelly.logger`。

已验证环境的已知限制
----------------------------

RTX 5090、Torch 2.11.0+cu128、Triton 3.6.0 的 LIF 多步反向在
``store_v_seq=False`` 时触发 ``TritonGPUCoalesce``／``PassManager::run failed``。
改变尺寸或将 ATan 换成 Sigmoid 没有消除错误；完整轨迹路径以及单步反向通过。
原生 CUDA eager 的默认最终状态路径通过，Inductor 自动使用 Triton 时仍受此问题影响。

上述训练和编译示例显式设置 ``store_v_seq=True``，作为临时规避方式；它会增加与
时间步数成比例的电位轨迹显存。不能把它理解为 backend 参数，也不能声称默认最终
状态配置已通过该环境验收。本轮只更新教程，生产内核修复需要单独处理。

故障排查
----------------------------

* 缺少扩展或依赖、已知设备不兼容时，自动模式继续检查候选；没有可用候选会报错。
* 未知 kernel、JIT、OOM 或梯度错误直接报告，不用静默回退隐藏错误。
* 原生扩展加载不兼容时核对构建与运行的 Torch/CUDA 版本及目标 GPU，必要时重建。
* 高级诊断可以在启动 Python 前设置 ``SJ_LIF_CUDA_IMPLEMENTATION=triton``，
  或对应的 ``SJ_<NEURON>_CUDA_IMPLEMENTATION``。默认 ``auto``；可强制
  ``cuda``、``triton`` 或 ``torch``。不支持当前配置时严格报错。
* 环境变量、依赖或扩展安装改变后重启进程。这些变量不是模型构造参数或训练必要步骤。
* 模块状态与输入的形状、设备或精度不匹配时，先检查是否跨独立 batch 保留了旧状态。

SpikingJelly 当前包不依赖 CuPy，旧安装和接口迁移见 :doc:`./migrate_from_legacy`。
