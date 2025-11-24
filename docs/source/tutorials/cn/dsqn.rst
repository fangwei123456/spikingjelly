使用深度脉冲Q网络玩Atari游戏
====================================
本教程作者：\ `Ding Chen <https://github.com/lucifer2859>`__

本节教程将介绍如何使用替代梯度方法训练一个深度脉冲Q网络。

从头搭建一个简单的深度脉冲Q网络
-------------------------

在PyTorch中搭建神经网络时，我们可以简单地使用\ ``nn.Sequential``\ 将多个网络层堆叠得到一个前馈网络，输入数据将依序流经各个网络层得到输出。

Atari游戏的观察经过预处理成为尺寸为\ :math:`84\times 84`\ 的灰度帧，然后利用帧堆叠减轻部分可观察性，并由动作空间决定离散动作的数目。一个简单的单层ANN网络如下：

.. code-block:: python

    nn.Sequential(
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

我们也可以用完全类似结构的SNN来进行强化学习任务。就这个网络而言，只需要先去掉所有的激活函数，再将神经元添加到原来激活函数的位置，这里我们选择的是LIF神经元。神经元之间的连接层需要用\ ``spikingjelly.activation_based.layer``\ 包装：

.. code-block:: python

    nn.Sequential(
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
            neuron.NonSpikingLIFNode(decode=dec_type)
        )

其中非脉冲神经元的膜电压编码方法需要通过参数\ ``dec_type``\ 设置，替代函数这里选择\ ``surrogate.ATan``\。

SpikingJelly中提供了4种膜电压编码方法，用作非脉冲神经元中膜电压序列的统计量，其中\ ``last-mem``\代表最终膜电压，\ ``max-mem``\代表最大膜电压，\ ``max-abs-mem``\代表最大绝对值的膜电压，而\ ``mean-mem``\代表平均膜电压。通过这种方式，SNN可以输出任意大小的浮点值，适用于强化学习中的Q值。

训练DSQN
-----------

首先指定好训练参数如学习率等以及若干其他配置

优化器默认使用Adam，并且使用第一个卷积层与随后的脉冲神经元层作为可学习的编码器

训练代码的编写需要遵循以下三个要点：

1. 脉冲神经元的输出是二值的。因此网络需要运行一段时间，即使用\ ``T``\ 个时刻后非脉冲神经元的膜电压统计量作为决策依据。

2. DSQN的损失函数与DQN算法相同。

3. 每次网络仿真结束后，需要\ **重置**\ 网络状态

DSQN的完整代码位于\ ``activation_based.examples.DSQN``\。

.. code-block:: shell

    python train.py --cuda --game breakout --T 8 --dec_type max-mem --seed 123

    usage: train.py [--cuda] [--game GAME] [--T T] [--dec_type DEC] [--early_stop]
                    [--eval_q] [--sticky_actions] [--frame_num FN] [--seed SEED]

    LIF DSQN Training

    optional arguments:
    --cuda              enable cuda
    --game GAME         ATARI game (gym)
    --T T               simulation time
    --dec_type DEC      type of SNN decoder, e.g. max-mem, mean-mem, max-abs-mem, last-mem, fr-mlp
    --early_stop        use stop reward to stop early
    --eval_q            record the Q-value (eval)
    --sticky_actions    use sticky actions
    --frame_num FN      number of frames
    --seed SEED         random seed to use

需要注意的是，训练这样的SNN，所需显存数量与仿真时长 ``T`` 线性相关，更长的 ``T`` 相当于使用更小的仿真步长，训练更为“精细”，但训练效果不一定更好。\ ``T``
太大时，SNN在时间上展开后会变成一个非常深的网络，这将导致BPTT计算梯度时容易衰减或爆炸。

训练结果
--------

详细的训练结果与分析可以参见 `相关论文 <https://arxiv.org/abs/2201.09754>`_。
