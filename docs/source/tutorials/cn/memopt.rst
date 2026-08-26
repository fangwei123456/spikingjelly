训练显存优化
============

English version: :doc:`../en/memopt`

``spikingjelly.activation_based.memopt`` 分为两层。底层 API 允许用户直接决定
checkpoint 边界； :func:`optimize_memory
<spikingjelly.activation_based.memopt.optimize_memory>` 则是基于论文 `Towards
Lossless Memory-efficient Training of Spiking Neural Networks via Gradient
Checkpointing and Spike Compression
<https://openreview.net/forum?id=nrBJ0Uvj7c>`_ 的可选高层预设。使用 memopt
不要求网络采用论文预设。

自定义 Checkpoint
-----------------

对任意 callable 使用 :func:`checkpoint
<spikingjelly.activation_based.memopt.checkpoint>` ：

.. code-block:: python

    from spikingjelly.activation_based import memopt, neuron

    y = memopt.checkpoint(block, x)

当模块边界已经能表达重算区域时，使用 :func:`checkpoint_module
<spikingjelly.activation_based.memopt.checkpoint_module>` ：

.. code-block:: python

    model.blocks[2] = memopt.checkpoint_module(model.blocks[2])

wrapper 不改变参数名、参数对象或 ``state_dict`` 键。状态化神经元通过显式
functional state 重算。BatchNorm running statistics 等模块 buffer 只提交一次，
不会在 backward 重算时再次更新。

时间分块必须显式指定，而且只有沿 ``time_dim`` 切分输入不改变模块语义时才能
使用：

.. code-block:: python

    model.neuron = memopt.checkpoint_module(
        model.neuron,
        chunks=2,
        chunked_args=(0,),
        time_dim=0,
    )

被切分输入必须具有相同且非零的时间长度，分块数不能超过该长度。tensor 输出沿
``time_dim`` 拼接；非 tensor 输出叶子在所有分块中必须相同。训练态 BatchNorm、
跨时间注意力，以及依赖完整时间 batch 的运算不得做时间分块。

输入压缩
--------

压缩器是任何提供 ``compress(tensor)`` 和 ``decompress(payload)`` 的无状态对象。
shape、dtype 等单次调用信息全部由 payload 持有，因此同一实例可以安全服务并发
调用。

.. code-block:: python

    model.spike = memopt.checkpoint_module(
        model.spike,
        compressor=memopt.BitSpikeCompressor(),
    )

``BitSpikeCompressor`` 和 ``BooleanSpikeCompressor`` 要求输入严格为 0 或 1。
``Uint8SpikeCompressor`` 用于整数值脉冲。只有非零索引比稠密表示更小时，
``SparseSpikeCompressor`` 才有意义。``NullSpikeCompressor`` 保持输入值和 dtype
不变。

论文预设
--------

高层预设原地修改模型，并返回同一个模型对象：

.. code-block:: python

    sample = torch.zeros(4, 8, 128, device="cuda")
    memopt.optimize_memory(
        model,
        targets=(ResidualBlock,),
        example_forward=lambda current: current(sample),
        level=3,
        checkpoint_budget="balanced",
        split_fn=lambda block: (block.conv, block.neuron),
        can_chunk=lambda module: isinstance(module, neuron.BaseNode),
    )

各级策略累进生效：

``0``
    严格 no-op，不要求提供 ``example_forward``。
``1``
    执行一次代表性前向，按照首个 tensor 输入的最大观测大小选择 target 模块。
``2``
    尝试 ``split_fn`` 返回的后代模块；只有训练峰值显存严格下降才保留拆分。
``3``
    对最终 checkpoint 叶子调用一次 ``can_chunk``，并尝试增大时间分块数。
``4``
    用 5 次 warmup 和 10 次测量统计 forward 开销，在不超过当前显存峰值的前提下
    贪心移除开销较大的 checkpoint。

``checkpoint_budget`` 的 ``"speed"``、``"balanced"`` 和 ``"memory"`` 分别
选择 50%、75% 和 100% 的候选模块；大小相同则保持模型顺序。只有所有相关 rank
观测到的首个 tensor 输入都严格二值时，预设才自动启用 bit 压缩。

level 2-4 要求模型和代表性输入均位于 CUDA。每次 profiling 后都会恢复 RNG、
training flag、buffer、神经元 memory 和既有参数梯度。``split_fn`` 必须返回至少
两个互不重叠的已注册后代模块。失败或没有降低峰值的候选会被还原。

分布式训练与后端
----------------

``process_group`` 应包含当前 PP stage 的全部 DP 和 TP rank。激活大小和显存峰值
按组内最大值聚合，二值资格按最小值聚合，stage leader 广播结构选择。所有 rank
必须以相同顺序调用 ``optimize_memory``。

内置 distributed Vision recipe 会构造这个 DP×TP stage group，并提供
``memopt_level``、``memopt_checkpoint_budget`` 和
``memopt_compress_inputs``。MCore 训练提供相同字段。评测、预测、生成和 artifact
导出始终构建无 wrapper 模型；透明 ``state_dict`` 使推理无需恢复训练期 wrapper。

公共 ``memopt.checkpoint`` 使用 PyTorch non-reentrant 实现，不负责选择神经元
后端。Torch、CuPy 和 Triton 的兼容范围取决于对应神经元正常的
functional-forward 支持。应对实际
训练使用的模型、dtype、后端、编译器和分布式拓扑做验证；不能执行 functional
神经元路径的自定义后端不会因 memopt 自动兼容。

核心无压缩路径和稠密 Boolean/bit 压缩支持
``torch.compile(..., fullgraph=True)``。Sparse payload 大小随数据变化，可能需要
编译器的动态 shape 支持。

从旧 API 迁移
---------------

旧 ``memory_optimization``、``input_compressed_gc``、``GCContainer``、
``TCGCContainer``、可变压缩器基类、summary/profile 对象，以及模块上的
``__spatial_split__`` 协议均已删除。论文预设使用 ``optimize_memory``；更简单的
网络专项策略直接组合 ``checkpoint`` 与 ``checkpoint_module``。不保留兼容别名。
