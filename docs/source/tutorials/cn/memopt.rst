训练显存优化
=========================================

本教程作者： `黄一凡 (AllenYolk) <https://github.com/AllenYolk>`_

English version: :doc:`../en/memopt`

本团队在ICLR 2026发表的新工作 `Towards Lossless Memory-efficient Training of Spiking Neural Networks via Gradient Checkpointing and Spike Compression <https://openreview.net/forum?id=nrBJ0Uvj7c>`_ 提出了基于梯度检查点和脉冲压缩的深度SNN训练显存自动优化工具（源代码位于 `Github <https://github.com/AllenYolk/snn-gradient-checkpointing>`_ ）。利用该工具，用户只需添加少量代码，便可以在不损失精度且不过多影响速度的前提下，大幅降低深度SNN训练时的显存占用。

该工具已经集成到 ``spikingjelly.activation_based.memopt`` 子包中，可应用于几乎所有以多步模式运行的 spikingjelly SNN 模型。本教程将介绍其使用方式。

方法原理
+++++++++++++++++++++++

显存占用分析
---------------------

从图1可以看出，SNN的训练显存峰值远大于结构相似的ANN。而且， **中间特征** （下图浅蓝色部分）占据了SNN峰值显存的绝大部分（96%以上）；这些中间特征在前向传播期间被缓存下来，以供反向传播计算梯度时使用。因此，减少中间特征显存占用是降低SNN训练显存的关键。

.. figure:: ../../_static/tutorials/memopt/memory-bar.png
    :width: 100%

    图1. 在ImageNet训练期间，不同ANN和SNN在达到峰值显存时的显存breakdown [#huang2026gc]_ 。

若将深度SNN视作若干个 **“权重-归一化-神经元”模块** （后亦简称为 **“层”** ）的堆叠，那么中间特征又可以细分成两个部分：

1. **输入** ：通常是二值脉冲向量。但也有例外，如网络的输入通常是浮点值，以及SEW ResNet [#fang2021sew]_ 中可能含非二值整数值。
2. **内部状态** ：权重和归一化层的中间计算结果，以及神经元的内部状态等。

梯度检查点 + 脉冲压缩
------------------------

为了降低 **内部状态** 的显存占用，可以对每一层施加 **梯度检查点 (gradient checkpointing, GC)** [#chen2016gc]_ 。具体而言，在执行第 :math:`l` 层的前向传播时，只缓存其输入 :math:`\mathbf{S}^{l-1}` 以及其他必要的权重；所有内部状态在完成计算后立即丢弃，不再缓存。在执行第 :math:`l` 层的反向传播时，首先使用 :math:`\mathbf{S}^{l-1}` 和权重重新计算该层前向传播以获得内部状态（即重构该层计算图），然后再计算梯度。如此一来，同一时刻最多只有一层的内部状态会存在于显存中，峰值显存得以大幅降低。我们称施加了上述变换的、只有输入被缓存的层为 **梯度检查点片段 (GC segment)** ；将常规层转换为梯度检查点片段后，需要多进行一次额外前向传播，故训练耗时增加。

即使施加了逐层梯度检查点，每层的 **输入** 仍需缓存。前文提到，深度SNN中绝大多数层的输入都是二值脉冲张量。然而，在 spikingjelly 等框架内部，二值张量使用浮点（ ``float32``, ``float16``, ...）表示；这保证了计算的兼容性，却带来了存储上的巨大冗余。为此，可以在缓存每层输入之前先进行 **无损脉冲压缩** ，将二值浮点张量 :math:`\mathbf{S}^{l-1}` 压缩到更紧凑的形式 :math:`\tilde{\mathbf{S}}^{l-1}` 以节省显存；重新计算前向传播时，解压 :math:`\tilde{\mathbf{S}}^{l-1}` 即可无损恢复出原始输入 :math:`\mathbf{S}^{l-1}` 。实验表明，基于比特表示的压缩器（用1比特表示一个0/1值）兼具速度和压缩率，因此被选为默认的脉冲压缩器。

图2(b)展示了梯度检查点+脉冲压缩施加后的前向/反向传播计算流程。更多细节，参见原文算法1 [#huang2026gc]_ 。

.. figure:: ../../_static/tutorials/memopt/method.png
    :width: 100%

    图2. 方法流程图。带有虚线黑框的灰色方形表示检查点片段 [#huang2026gc]_ 。

检查点结构自适应调整
-------------------------

施加逐层梯度检查点+脉冲压缩后，一个训练iteration内的如化情况如图3橙色折线所示。优化后，虽然相比传统BPTT（蓝色折线）峰值显存已大幅降低，但全局峰值显存却远大于在其他层上运行时的临时显存占用。对此，我们设计了一系列检查点片段分割策略，以引入更多需缓存的输入为代价，降低关键检查点片段的大小；此外，也可地择性将一些检查点片段还原为常规层，以略微增加临时显存开销为代价，加快训练速度，同时不增加峰值显存。具体流程为：

1. **空间分割**：找出峰值显存开销所在的检查点片段，将其沿空间分割成两个更小的检查点片段。重复此步骤，直到无法进一步降低峰值显存。见图2(c)。
2. **时间分割**：找出峰值显存开销所在的检查点片段，将其沿时间轴分割成 :math:`k` 个更小的检查点片段。重复此步骤，直到无法进一步降低峰值显存。见图2(d)。
3. **贪心还原**：测量每个检查点片段的前向传播用时，并降序排列。按序尝试将每个检查点片段还原为常规层。一步变换后，若峰值显存不增加，则保留；否则撤销这一步变换。

更多细节，参见原文算法2 [#huang2026gc]_ 。

.. figure:: ../../_static/tutorials/memopt/curve.png
    :width: 100%

    图3. Spiking VGG在CIFAR10-DVS上训练的一个iteration内显存消耗变化情况 [#huang2026gc]_ 。

.. note::

    先考虑空间分割，再考虑时间分割；换言之，**时间分割仅仅作为空间分割的补充**。这是因为：时间分割与时间维度并行方法不兼容；而且，这限制了沿着时间步的内核融合（原本可将 :math:`T` 步融合到一个内核，分割后则需运行 :math:`k` 个 :math:`T/k` 步的内核），降低了速度。

使用说明
++++++++++++++++++++++++

选择使用方式
------------

``memopt`` 有两种入口：

* 已经知道网络的哪一段适合重算时，直接使用 ``checkpoint`` 或
  ``checkpoint_module``。
* 希望自动寻找检查点结构时，使用 ``optimize_memory``。

建议先尝试手工检查点。它更直接，也不需要额外的搜索。论文中的自动调整策略已封装为
``optimize_memory``，作为可选的高层预设。

手工设置检查点
--------------

如果检查点范围不是一个完整模块，直接把函数或可调用对象传给
:func:`checkpoint <spikingjelly.activation_based.memopt.checkpoint>`：

.. code-block:: python

    from spikingjelly.activation_based import memopt

    y = memopt.checkpoint(block, x)

如果要重算的范围正好是一个模块，使用 :func:`checkpoint_module
<spikingjelly.activation_based.memopt.checkpoint_module>`：

.. code-block:: python

    model.blocks[2] = memopt.checkpoint_module(model.blocks[2])

``checkpoint_module`` 不改变参数对象、参数名或 ``state_dict`` 键，因此可以在
包装前后使用同一份权重。它还会显式传递神经元状态。BatchNorm 的 running
statistics 等 buffer 在一次训练迭代中只更新一次，不会因 backward 重算而重复
更新。

压缩检查点输入
--------------

检查点仍需保存输入。如果输入是脉冲，可以同时压缩以位置参数传入的第一个
tensor：

.. code-block:: python

    model.spike_block = memopt.checkpoint_module(
        model.spike_block,
        compressor=memopt.BitSpikeCompressor(),
    )

内置压缩器的用途如下：

* ``BitSpikeCompressor`` 将 8 个二值脉冲打包到 1 byte。
* ``BooleanSpikeCompressor`` 将二值脉冲保存为 ``bool``。
* ``Uint8SpikeCompressor`` 保存能由 ``uint8`` 表示的整数脉冲。
* ``SparseSpikeCompressor`` 只保存非零位置，适合非常稀疏的二值脉冲。

Bit、Boolean 和 Sparse 压缩要求输入严格为 0 或 1。手工选择压缩器时，memopt
不会检查输入值。普通浮点激活若误用这些压缩器，解压后的数值会改变。

也可以传入自己的压缩器。对象只需实现 ``compress(tensor)`` 和
``decompress(payload)``。shape、dtype 和 device 等本次调用所需的信息应放在
payload 中，不要保存在压缩器实例上。这样同一个压缩器才能安全地用于并发调用。

沿时间维分块
------------

``checkpoint_module`` 可以把序列分成多个时间块，依次重算：

.. code-block:: python

    model.neuron = memopt.checkpoint_module(
        model.neuron,
        chunks=2,
        chunked_args=(0,),
        time_dim=0,
    )

时间分块不只是一个显存开关，它会改变模块的执行顺序。只有按时间顺序分块计算仍
保持原有语义时才能使用。普通多步神经元会在块之间传递状态，适合这种方式；训练态
BatchNorm、跨时间注意力以及依赖完整序列统计量的运算通常不适合。

所有被切分的输入必须具有相同且非零的时间长度，``chunks`` 不能大于该长度。
tensor 输出沿 ``time_dim`` 拼接；非 tensor 输出必须在各块中保持相同。

使用自动预设
------------

:func:`optimize_memory <spikingjelly.activation_based.memopt.optimize_memory>`
会原地修改模型，并返回同一个对象。下面假设模型中已经定义了
``ResidualBlock``：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import memopt, neuron

    def split_residual(module):
        if isinstance(module, ResidualBlock):
            return module.conv, module.neuron
        return ()

    sample = torch.zeros(4, 8, 128, device="cuda")
    model.cuda()
    memopt.optimize_memory(
        model,
        targets=ResidualBlock,
        example_forward=lambda current: current(sample),
        level=3,
        checkpoint_budget="balanced",
        split_fn=split_residual,
        can_chunk=lambda module: isinstance(module, neuron.BaseNode),
    )

``example_forward`` 应使用与真实训练相同的 shape、dtype、device 和训练模式，并
返回至少一个可求导的浮点 tensor。自动搜索只依据这次运行做决定，因此样本必须能
代表实际训练负载。

``level`` 控制搜索深度，各级包含前一级的结果：

``0``
    不做任何修改，也不需要 ``example_forward``。
``1``
    观察各个 target 的第一个 tensor 输入，优先为输入较大的模块设置检查点。
``2``
    尝试用 ``split_fn`` 把一个大检查点拆成多个小检查点。只有峰值显存下降时才保留。
``3``
    对 ``can_chunk`` 认可的检查点尝试时间分块。
``4``
    测量各检查点的前向开销，在不增加当前峰值显存的前提下移除代价较高的检查点。

``checkpoint_budget`` 决定 level 1 覆盖多少候选模块：``"speed"``、
``"balanced"`` 和 ``"memory"`` 分别选择 50%、75% 和 100%。候选模块按输入
大小排序，大小相同时保持模型中的原顺序。

如果启用了 ``compress``，预设只会在所有相关 rank 都观察到严格二值输入时自动
使用 bit 压缩。``split_fn`` 应返回至少两个互不重叠的已注册后代模块；不适用时
返回空 tuple。``can_chunk`` 只应对确实可以沿时间维切分的模块返回 ``True``。

``level=2..4`` 会反复运行前向和反向，因此只适合在训练开始前搜索一次，并要求模型和
样本位于 CUDA。每次尝试后，框架会恢复随机数状态、buffer、神经元状态和已有
梯度。发生 OOM 或显存没有下降时，本次修改会被撤销。

分布式训练
----------

应在 DDP 或 FSDP 包装模型之前调用 ``optimize_memory``。使用 PP 时，
``process_group`` 必须包含当前 pipeline stage 的全部 DP 和 TP rank。各 rank 必须
按相同顺序调用该函数。memopt 会汇总组内观测，让所有 rank 生成相同的模型结构。

内置的分布式视觉训练会自动创建这个进程组，并提供
``memopt_level``、``memopt_checkpoint_budget`` 和
``memopt_compress_inputs``。MCore 训练提供 level 和 budget 配置，但只在预先
确定的 Transformer 边界设置检查点，不会强行进行空间或时间切分。

评测、预测、生成和模型导出不会保留训练期的检查点包装。由于
``checkpoint_module`` 保持 ``state_dict`` 兼容，推理时不需要转换权重。

神经元后端与 ``torch.compile``
--------------------------------

memopt 不会替换神经元后端。只要神经元的函数式 forward 路径支持对应实现，Torch、
CuPy 和 Triton 都可以放在检查点内。自定义后端如果不支持这条路径，也不会因为包装
了 memopt 而自动兼容。正式训练前，应使用实际模型、dtype、后端和分布式拓扑完成
一次前向与反向测试。

``memopt.checkpoint`` 使用 PyTorch non-reentrant checkpoint。无压缩、
Boolean 压缩和 bit 压缩路径支持 ``torch.compile(..., fullgraph=True)``。Sparse
压缩后的大小随输入变化，编译时可能需要动态 shape。

从旧 API 迁移
---------------

.. list-table::
    :header-rows: 1

    * - 旧用法
      - 新用法
    * - ``input_compressed_gc``
      - ``checkpoint(..., compressor=...)``
    * - ``GCContainer`` / ``TCGCContainer``
      - ``checkpoint_module``
    * - ``memory_optimization``
      - ``optimize_memory``
    * - 模块上的 ``__spatial_split__``
      - 调用 ``optimize_memory`` 时传入 ``split_fn``

旧版的可变压缩器基类、summary/profile 对象和兼容别名不再提供。


.. [#huang2026gc] Huang, Y., Fang, W., Hao, Z., Ma, Z., & Tian Y. (2026). Towards Lossless Memory-efficient Training of Spiking Neural Networks via Gradient Checkpointing and Spike Compression. The Fourteenth International Conference on Learning Representations.
.. [#fang2021sew] Fang, W., Yu, Z., Chen, Y., Huang, T., Masquelier, T., & Tian, Y. (2021). Deep residual learning in spiking neural networks. Advances in neural information processing systems, 34, 21056-21069.
.. [#chen2016gc] Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174.
