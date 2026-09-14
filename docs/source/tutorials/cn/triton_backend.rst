Triton 后端
===========================

本教程作者： `黄一凡 (AllenYolk) <https://github.com/AllenYolk>`_

English version: :doc:`../en/triton_backend`

SpikingJelly ``0.0.0.1.0`` 版本引入了 `Triton <https://github.com/triton-lang/triton>`_ 后端，作为本框架继 PyTorch 和 CuPy 之后的第三种后端。相比使用 CUDA 撰写的 CuPy 后端，Triton 后端具有更好的可读性、扩展性和可维护性，更容易达到较高的 GPU 利用率，且有扩展到 `其他硬件平台 <https://gitcode.com/Ascend/triton-ascend>`_ 的潜力。

本教程聚焦于预定义神经元的 Triton 后端用法。关于基于自定义动力学函数自动生成内核，请参考 :doc:`./flexsn`。

本教程需要如下的准备及前置知识：

#. `安装好 Triton <https://triton-lang.org/main/getting-started/installation.html>`_ 。推荐使用 ``triton >= 3.3.1`` 。
#. 熟悉 SpikingJelly 的 :doc:`./neuron` 模块。

前向传播与反向传播
-------------------------

神经元 Triton 后端的启用方式和 CuPy 后端类似。以 ``LIFNode`` 为例：

.. code:: python

    import torch
    from spikingjelly.activation_based import neuron

    n = neuron.LIFNode(step_mode="m", backend="triton").to("cuda:0")
    x = torch.randn([16, 1, 3, 32, 32], device="cuda:0") # [T, B, C, H, W]

    s = n(x)
    print(s.device, s.shape, s.mean())
    # cuda:0 torch.Size([16, 1, 3, 32, 32]) tensor(0.0313, device='cuda:0')

这里，我们构造了一个以多步模式 ``step_mode="m"`` 运行的 LIF 神经元，并启用 Triton 后端。将神经元和输入张量都移动到 ``"cuda:0"`` 设备上后，即可使用 Triton 后端完成前向传播计算。 Triton 后端当然也支持反向传播，且会得到与其它后端（几乎）完全相同的结果：

.. code:: python

    import torch
    import torch.nn.functional as F
    from spikingjelly.activation_based import neuron

    n_triton = neuron.LIFNode(
        step_mode="m", backend="triton", store_v_seq=True
    ).to("cuda:0")
    n_torch = neuron.LIFNode(
        step_mode="m", backend="torch", store_v_seq=True
    ).to("cuda:0")

    x = torch.randn([16, 1, 3, 32, 32], device="cuda:0") # [T, B, C, H, W]
    x_triton = x.clone().requires_grad_(True)
    x_torch = x.clone().requires_grad_(True)

    s_triton = n_triton(x_triton)
    s_torch = n_torch(x_torch)
    v_triton = n_triton.v_seq
    v_torch = n_torch.v_seq

    grad = torch.randn_like(s_triton)
    s_triton.backward(grad)
    s_torch.backward(grad)

    assert torch.allclose(s_triton, s_torch)
    print(s_triton.mean()) # tensor(0.0309, device='cuda:0', grad_fn=<MeanBackward0>)
    assert torch.allclose(v_triton, v_torch)
    print(v_triton.mean()) # tensor(-0.0702, device='cuda:0', grad_fn=<MeanBackward0>)
    assert torch.allclose(x_triton.grad, x_torch.grad, rtol=1e-6, atol=1e-6)
    print(
        F.cosine_similarity(x_triton.grad.flatten(), x_torch.grad.flatten(), dim=0)
    ) # tensor(1., device='cuda:0')

速度测算
------------------

Triton 后端支持 ``torch.float16``。以下 benchmark 使用 ``triton.testing`` 对比不同后端的速度：

.. code:: python

    import torch
    import triton
    from spikingjelly.activation_based import neuron, functional

    DEVICE = "cuda:0"

    def forward_backward(net, x_seq):
        y_seq  = net(x_seq)
        y_seq.sum().backward()
        x_seq.grad = None
        functional.reset_net(net)


    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["T"],
            x_vals=[4*i for i in range(1, 9)],
            line_arg="backend",
            line_vals=["torch", "cupy", "triton"],
            line_names=["torch", "cupy", "triton"],
            styles=[
                ('green', ':'), ('blue', '--'), ('red', '-.'),
            ],
            ylabel='Execution Time (ms)',
            plot_name='Performance-float16',
            args={"N": 64, "C": 64*32*32, 'dtype': torch.float16},
        )
    )
    def benchmark(T, N, C, dtype, backend):
        net = neuron.LIFNode(
            backend=backend,
            step_mode='m',
        ).to(device=DEVICE, dtype=dtype)
        x_seq = torch.rand(
            [T, N, C], device=DEVICE, dtype=dtype, requires_grad=True
        )
        results = triton.testing.do_bench(
            lambda: forward_backward(net, x_seq),
            quantiles=[0.5, 0.2, 0.8],
            grad_to_none=[x_seq]
        )
        return results

    benchmark.run(save_path="./logs", print_data=True, show_plots=True)

在单个 GeForce RTX 4090 上运行，结果如下：

.. code:: text

    Performance-float16:
        T      torch      cupy    triton
    0   4.0   0.992784  0.667648  0.771072
    1   8.0   3.459072  1.338368  0.857088
    2  12.0   7.058432  1.988608  1.289216
    3  16.0  11.737088  2.630736  1.704896
    4  20.0  17.557505  3.263488  2.115584
    5  24.0  24.517120  3.902464  2.533376
    6  28.0  32.649216  4.535296  2.940928
    7  32.0  41.872896  5.189120  3.365888

.. image:: ../../_static/tutorials/triton_backend/Performance-float16.png
    :width: 100%

可见，数据规模和序列长度 ``T`` 都较大时，Triton 后端相比 CuPy 和 PyTorch 后端具有明显的速度优势。

与 ``torch.compile`` 组合
-----------------------------

.. warning::

    当前 Triton 神经元会将非 contiguous 输入转换为 contiguous。对卷积 SNN，
    Inductor 的默认布局优化可能选择 channels-last 卷积，从而在卷积与神经元之间
    插入重排或拷贝。建议以
    ``torch.compile(..., options={"layout_optimization": False})`` 为起点，并在
    目标 GPU、模型和 batch size 上复测。CuPy 神经元也有相同的连续布局限制；
    纯全连接 SNN 不存在 NCHW/channels-last 卷积布局冲突。

Triton 神经元可以被 ``torch.compile`` 捕获，但完整图捕获不保证端到端加速。
逐 kernel profile 定位到一次确定的回退机制：默认 Inductor 为卷积选择 NHWC，
而 Triton LIF 的固定 stride 要求迫使网络恢复 NCHW，并同时选择了较慢的卷积 kernel。
在 RTX 4090 的最小红例（SEW-ResNet18、B=32、T=4、136×136）中，5 个 step 的
GPU 时间如下：

.. list-table:: 布局策略的因果对照
    :header-rows: 1

    * - 路径
      - 总时间 (ms)
      - 卷积 (ms)
      - 布局转换 (ms / 次数)
      - LIF (ms)
    * - Triton eager
      - 38.079
      - 18.874
      - 0 / 0
      - 4.300
    * - 默认 compile+Triton
      - 47.923
      - 33.411
      - 2.794 / 195
      - 4.650
    * - 关闭布局优化
      - **36.866**
      - 20.995
      - 0 / 0
      - 4.010

关闭布局优化将该例的 speedup 从 0.800× 修正为 1.040×；LIF 本身只解释了
0.350 ms 回退。吞吐 workload 可使用：

.. code-block:: python

    compiled_model = torch.compile(
        model,
        backend="inductor",
        options={
            "layout_optimization": False,
            "max_autotune": True,
            "triton.cudagraphs": False,
            "triton.cudagraph_trees": False,
        },
    )

``layout_optimization=False`` 是关键修正。``max_autotune`` 会增加首次编译时间，
在 SEW-ResNet18 上只额外改善约 1.2%。

以下结果来自独占、按需租用的 RTX 5090，使用 PyTorch 2.11.0+cu128、Triton
3.6.0、T=4、LIF、FP32 和 224×224 输入。每个 case 在新进程和独立 Inductor
cache 中运行，重复三轮；全部 compile case 均为 1 张图、0 graph break、
0 recompile。表中是三轮中位数，推理 batch 为 64，训练 batch 为 16。

.. list-table:: 修正后的四模式端到端时延
    :header-rows: 1

    * - 模型 / 阶段
      - Torch eager (ms)
      - Torch compile (ms)
      - Triton eager (ms)
      - Triton compile (ms)
      - compile / Triton eager
    * - VGG 推理
      - 389.204
      - 163.176
      - 176.896
      - **150.719**
      - 1.174×
    * - VGG 训练
      - 251.198
      - 160.743
      - 139.855
      - **130.926**
      - 1.068×
    * - SEW 推理
      - 54.266
      - 28.823
      - 27.868
      - **27.350**
      - 1.019×
    * - SEW 训练
      - 71.232
      - 23.870
      - 47.136
      - **20.481**
      - 2.298×
    * - Spikformer 推理
      - 60.318
      - 34.739
      - 35.078
      - **32.860**
      - 1.068×
    * - Spikformer 训练
      - 73.473
      - 29.458
      - 51.102
      - **26.795**
      - 1.905×

修正后的 compile+Triton 在六项和每一轮中均快于 eager+Triton，完整数据可
:download:`下载为 CSV
<../../_static/tutorials/triton_backend/compile-backends-rtx5090-tuned.csv>`。
完整排序 ``compile+Triton > eager+Triton > compile+Torch > eager+Torch``
只在 VGG 训练和 SEW 推理成立，因为其余模式中 compile+Torch 超过了 eager+Triton。

SEW-ResNet18 推理的 batch 1、2、4、8、16、32、64、128 配对 speedup 分别为
3.147×、2.601×、2.535×、2.156×、1.112×、1.019×、1.019×、1.012×；完整数据可
:download:`下载为 CSV
<../../_static/tutorials/triton_backend/compile-sew18-rtx5090-tuned.csv>`。
大 batch 的 1%–2% 收益具有硬件敏感性，应在目标环境复测。

推理模型还可以先调用 :func:`fuse_conv_bn_eval_modules
<spikingjelly.activation_based.functional.conv_bn_fusion.fuse_conv_bn_eval_modules>`。
它将 VGG Triton eager 从 176.896 ms 降至 157.898 ms；公平地融合四种模式后，
Torch compile 为 143.950 ms，快于 Triton compile 的 149.803 ms，因此该优化
应独立评估。

.. admonition:: 警告
    :class: warning

    在使用预定义的 Triton 神经元内核时，需注意：

    * ``IFNode``、``LIFNode`` 和 ``PLIFNode`` 提供支持推理与训练的预定义 Triton 内核。实验性的 ``ActivationAwareIFNode`` 另提供支持标量或逐通道阈值与膜电位偏移的多步、仅推理 Triton 内核；训练或求梯度时会明确报错，不会回退到 PyTorch。
    * Triton 后端应在 GPU 上运行。
    * Triton 后端仅支持多步运行模式 ``step_mode="m"`` 。
