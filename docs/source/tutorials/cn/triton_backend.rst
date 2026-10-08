神经元自动执行
==========================

English version: :doc:`../en/triton_backend`

从 CPU 到 CUDA
----------------------------

创建神经元后，将模块和输入移动到目标设备即可。神经元构造函数和属性不再提供
backend 选项。下面的 IF、LIF 和 PLIF 训练代码可在 CPU 或 NVIDIA CUDA 上运行：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import functional, neuron, surrogate

    torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for node_type in (neuron.IFNode, neuron.LIFNode, neuron.ParametricLIFNode):
        node = node_type(
            step_mode="m", surrogate_function=surrogate.ATan()
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

CPU 使用 Torch 参考实现。CUDA 在每个设备首次使用某条执行路径时检查兼容性，
随后复用已选实现。这个过程不进行在线测速，导入 SpikingJelly 也不会初始化 CUDA。
eager 按原生 CUDA、Triton、Torch 的顺序检查；Inductor 编译展开按 Triton、原生
CUDA、Torch 的顺序检查。各 GPU 架构使用相同的顺序，兼容性会影响可用实现，
具体 workload 的速度仍需实测。

自动分发覆盖 IF、LIF、PLIF、QIF、EIF、Izhikevich、I-LIF、ActivationAwareIF 和
STBIF。各类支持的训练模式、dtype 和替代梯度不同，详见对应 API。其他神经元
沿用各自的执行方式，其中一些没有独立 CUDA 内核。

普通融合路径接受 FP32/FP16/BF16 输入，要求状态为 FP32，替代梯度为受支持的内置
类型。低精度状态和自定义替代梯度可使用 Torch 参考公式。未显式配置精度时，
模块的膜电位跟随输入 dtype，已有 FP32 状态也会转换。因此普通 FP16/BF16
模块调用使用参考递推。functional 接口可以直接传入低精度输入和 FP32 初态；
模块存储精度的覆盖方式见 :doc:`./precision`。自定义 FlexSN core 见 :doc:`./flexsn`。

编译与 CUDA Graph
----------------------------

eager 和 Inductor 编译展开分别选择实现。下面使用默认 FP32 状态，先运行一次
eager 前后向来初始化设备并预热，再重置状态并编译模型：

.. code-block:: python

    import torch
    from torch import nn
    from spikingjelly.activation_based import functional, neuron, surrogate

    device = torch.device("cuda:0")
    model = (
        nn.Sequential(
            nn.Linear(16, 16),
            neuron.LIFNode(
                step_mode="m", surrogate_function=surrogate.ATan()
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

首次调用的耗时包含加载、编译或 JIT；测速应在预热后进行。CUDA Graph 保留被捕获
函数的选择，因此 eager 捕获仍使用 eager 实现，compiled 捕获使用编译路径的实现。
捕获前还需按 PyTorch CUDA Graph 要求准备内存并预热前后向。

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

查询会初始化并缓存对应路径的选择，保留模块状态且不执行神经元计算。返回字典中，
``implementation`` 是绑定实现，``unavailable`` 记录此前候选不可用的原因。
低精度状态、自定义 surrogate 和显式精度配置可能采用其他路径，查询结果不能
代替对某次调用的 profiling。

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

最终状态与完整轨迹
----------------------------

默认的 ``store_v_seq=False`` 只保留最终电位。监控时间轨迹时可设为 ``True``，
额外的轨迹显存随时间步数增长。两种配置均支持输入与初态梯度。最终状态路径已在
RTX 5090、Torch 2.11.0+cu128、Triton 3.6.0 上通过 FP32/FP16/BF16 输入的 eager
和 fullgraph 前后向验证，编译时可以直接使用默认配置。

故障排查
----------------------------

* 缺少扩展或依赖、已知设备不兼容时，自动模式继续检查候选；没有可用候选会报错。
* kernel、JIT、OOM 或梯度错误会直接报错，自动选择不会掩盖这些执行失败。
* 原生扩展加载不兼容时核对构建与运行的 Torch/CUDA 版本及目标 GPU，必要时重建。
* 高级诊断可以在启动 Python 前设置 ``SJ_LIF_CUDA_IMPLEMENTATION=triton``，
  或对应的 ``SJ_<NEURON>_CUDA_IMPLEMENTATION``。默认 ``auto``；可强制
  ``cuda``、``triton`` 或 ``torch``。不支持当前配置时严格报错。
* 修改环境变量、依赖或扩展安装后，需要重启进程。普通训练无需设置这些诊断变量。
* 模块状态与输入的形状、设备或精度不匹配时，先检查是否跨独立 batch 保留了旧状态。

SpikingJelly 当前包不依赖 CuPy，旧安装和接口迁移见 :doc:`./migrate_from_legacy`。
