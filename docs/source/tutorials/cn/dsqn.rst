使用深度脉冲Q网络玩Atari游戏
====================================

本教程作者： `Ding Chen <https://github.com/lucifer2859>`_

.. warning::

    欢迎参与本教程的英文翻译。

你将用替代梯度训练深度脉冲 Q 网络（DSQN），根据 Atari 游戏画面选择动作。替代梯度用平滑函数的导数近似脉冲发放函数的导数，让网络能够反向传播。

从头搭建一个简单的深度脉冲Q网络
-------------------------------

在PyTorch中搭建神经网络时，我们可以简单地使用 ``nn.Sequential`` 将多个网络层堆叠得到一个前馈网络，输入数据将依序流经各个网络层得到输出。

Atari游戏的观察经过预处理成为尺寸为 :math:`84\times 84` 的灰度帧，然后利用帧堆叠减轻部分可观察性，并由动作空间决定离散动作的数目。一个常用的卷积 Q 网络如下：

.. code-block:: python

    import torch.nn as nn

    input_shape = (4, 84, 84)
    n_actions = 4
    net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),

            nn.Linear(512, n_actions)
        )

我们也可以用完全类似结构的SNN来进行强化学习任务。就这个网络而言，只需要先去掉所有的激活函数，再将神经元添加到原来激活函数的位置，这里我们选择的是LIF神经元。神经元之间的连接层需要用 ``spikingjelly.activation_based.layer`` 包装：

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import functional, layer, neuron, surrogate

    input_shape = (4, 84, 84)
    n_actions = 4
    net = nn.Sequential(
            layer.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),

            layer.Conv2d(32, 64, kernel_size=4, stride=2),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),

            layer.Conv2d(64, 64, kernel_size=3, stride=1),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),

            layer.Flatten(),
            layer.Linear(64 * 7 * 7, 512),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),

            layer.Linear(512, n_actions),
            neuron.NonSpikingLIFNode(decode='max-mem')
        )

输入中的 4 个通道对应堆叠的灰度帧，``n_actions`` 应与环境的动作数一致。将同一批画面重复输入 ``T`` 步，网络会为每个动作输出一个 Q 值，即该动作的预期累计回报：

.. code-block:: python

    functional.set_step_mode(net, step_mode="m")
    T = 8
    frames = torch.rand(2, *input_shape)
    q = net(frames.unsqueeze(0).expand(T, -1, -1, -1, -1))
    print(q.shape)  # torch.Size([2, 4])
    functional.reset_net(net)


``neuron.NonSpikingLIFNode`` 的 ``decode`` 参数决定如何从膜电压序列中取出 Q 值。脉冲层使用 ``surrogate.ATan`` 计算替代梯度。完整脚本还支持 ``fr-mlp``：对最后一层 LIF 的脉冲在时间维取平均，再接 ``nn.Linear`` 输出 Q 值。

SpikingJelly中提供了4种膜电压编码方法，用作非脉冲神经元中膜电压序列的统计量，其中\ ``last-mem``\代表最终膜电压，\ ``max-mem``\代表最大膜电压，\ ``max-abs-mem``\代表绝对值最大的膜电压（保留符号），而\ ``mean-mem``\代表平均膜电压。通过这种方式，SNN可以输出任意大小的浮点值，适用于强化学习中的Q值。

训练DSQN
-----------

先设置学习率和仿真步数 ``T``。训练脚本使用 Adam 优化器，第一个卷积层与随后的脉冲神经元层构成可学习的编码器。

训练代码的编写需要遵循以下三个要点：

1. 脉冲神经元的输出是二值的。因此网络需要运行一段时间，即使用\ ``T``\ 个时刻后非脉冲神经元的膜电压统计量作为决策依据。

2. DSQN的损失函数与DQN算法相同。

3. 每次网络仿真结束后，调用 ``functional.reset_net(net)`` 重置网络状态，其中 ``net`` 是本次仿真的网络。

DSQN 的完整代码位于 ``spikingjelly/activation_based/examples/dsqn``。脚本使用该目录中的本地 ``ptan`` 和 ``utils`` 包，因此应先进入目录再运行：

.. code-block:: shell

    cd spikingjelly/activation_based/examples/dsqn
    python train.py --cuda --game breakout --T 8 --dec_type max-mem --seed 123

运行训练或 ``--help`` 前，按该目录的 README 配置 ``gym==0.18.0``、``gym[atari]`` 和 ``atari-py==0.2.5``。脚本依赖 Gym 的四项返回值 ``step`` 接口和 Atari 环境，安装 Gymnasium 后不能直接替代。

增大仿真步数 ``T`` 会增加反向传播时保存的中间状态和显存占用。过长的序列还可能导致随时间反向传播（BPTT）时梯度衰减或爆炸，因此应根据验证结果选择 ``T``。

训练结果
--------

详细的训练结果与分析可以参见 `相关论文 <https://arxiv.org/abs/2201.09754>`_。
