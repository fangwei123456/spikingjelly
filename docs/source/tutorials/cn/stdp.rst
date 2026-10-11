局部突触可塑性与 STDP
==========================

English version: :doc:`../en/stdp`

定位与支持范围
----------------------

``learning`` 提供局部突触权重更新工具。STDP 的活动相关性更新不是分类损失的梯度，
把更新量写入 ``weight.grad`` 是借用优化器的更新通道，不保证任务正确率提升。
编码、竞争、稳态、初始化和读出仍是训练方法的一部分。

支持范围、时间常数、张量布局、奖励对齐和迁移见
:doc:`../../APIs/spikingjelly.activation_based.learning`。
普通 STDP 使用 ``1-1/tau`` 衰减；MSTDP/MSTDPET 使用 ``exp(-1/tau)``。
它们保留各自离散定义，不应仅为公式统一而替换。

规则更新与状态
----------------------

设当前 pre/post 脉冲为 ``p[t]``、``q[t]``，普通 STDP 更新为：

.. math::

   a_t &= (1-1/\tau_{pre})a_{t-1}+p_t \\
   b_t &= (1-1/\tau_{post})b_{t-1}+q_t \\
   \Delta W_t &= F_{post}(W)(q_t\otimes a_t)-F_{pre}(W)(b_t\otimes p_t)

batch 和卷积空间位置求和；多步模式还对时间求和。改变 batch 或空间大小会改变
更新尺度，需要调用者显式调整学习率或 ``scale``。当前脉冲先进入 trace；
同一步发放仅在调制函数合适时抵消。固定权重下的多步累计不等价于逐步改权重。

下面直接使用更新量，不使用梯度或优化器：

.. code-block:: python

    import torch
    from torch import nn
    from spikingjelly.activation_based import learning, neuron

    fc = nn.Linear(4, 3, bias=False)
    sn = neuron.IFNode()
    learner = learning.STDPLearner("s", fc, sn, 20., 20.)
    try:
        with torch.no_grad():
            sn(fc((torch.rand(2, 4) > 0.7).float()))
            delta_w = learner.step(on_grad=False, scale=1 / 2)
            fc.weight.add_(delta_w, alpha=0.01)
        learner.reset()
        sn.reset()
    finally:
        learner.close()


``step`` 已消费记录，但保留 trace。独立样本或 episode 之间调用 ``reset``；
连续活动学习中保留 trace。更换 batch 大小前同时重置神经元与学习器。
评估前显式 ``disable``，因为 ``net.eval()`` 不会停用活动记录；结束后 ``close``。
不发放的 post 神经元在零 post trace 下可能完全不更新，不能靠 loss 显示训练状态。

机制演示
----------------------

以下脚本演示 trace、活动和权重变化，默认无 GUI，输出到未跟踪目录。
它们不是任务训练 benchmark；``--show`` 可打开窗口，``--output-dir`` 可改变目录。

.. code-block:: bash

    uv run python -m spikingjelly.activation_based.examples.stdp_trace
    uv run python -m spikingjelly.activation_based.examples.mstdp
    uv run python -m spikingjelly.activation_based.examples.mstdpet


奖励学习器每步只接受一对活动；``reward[t]`` 作用于 ``eligibility[t-1]``。
初始更新为零，最终活动之后的奖励需后续前向推进；无记录的 step 不推进时间。
MSTDP 支持标量或 ``[N]`` reward，MSTDPET 支持标量。
创建 MSTDP 时不再传 ``batch_size``；如原来积累多个前向，应改为每步前向后调用 step。

与梯度训练组合
----------------------

按层分工和同参数叠加是不同实验。按层分工时，反传后清除只用 STDP 的参数梯度，
再注入 STDP 更新；叠加时保留任务梯度。不要 detach loss 后尝试反传。
``sn`` 必须是实际产生 post 脉冲的模块，不能按下标把 BatchNorm 等中间层误传进去。
只有无 momentum、无 weight decay 的 SGD 对应简单的 ``W += lr * delta_w``；
其他优化器会改变更新动力学。下面只演示同参数叠加的调用流程，不证明效果：

.. code-block:: python

    import torch
    from torch import nn
    from spikingjelly.activation_based import learning, neuron

    fc = nn.Linear(4, 3, bias=False)
    sn = neuron.IFNode()
    learner = learning.STDPLearner("s", fc, sn, 20., 20.)
    optimizer = torch.optim.SGD(fc.parameters(), lr=0.01)
    try:
        optimizer.zero_grad()
        output = sn(fc((torch.rand(2, 4) > 0.5).float()))
        loss = (output - 1).square().mean()
        loss.backward()
        learner.step(on_grad=True, scale=0.01)
        optimizer.step()
        learner.reset()
        sn.reset()
    finally:
        learner.close()


完整 MNIST 实例
----------------------

``stdp_mnist`` 使用 784→400 的无 bias 前馈层和自适应 LIF。
所有可学习前馈权重由 STDP 更新，无反传、预训练或可训练读出；循环连接固定。
竞争通过延迟一个时间步的抑制电导形成，没有 argmax 赢家选择。
每个神经元独立越阈发放并只重置自身电位，同一步允许多个神经元发放。
发放后的不应期只作用于该神经元，不直接重置其他神经元。

现有 ``LinearRecurrentContainer`` 接收拼接的 ``[I_ff, 0]``，分别输出兴奋电流
和抑制电导增量。循环矩阵 R 的对角为 0，其他元素为正幅值 g；它汇总
其他神经元的上一时刻脉冲，并通过抑制电导通道产生负反馈。
子神经元保留电导 G 和不应期计数 r，使用以下离散动力学：

.. math::

    G[t] &= e^{-1/\tau_{inh}}G[t-1] + R s[t-1] \\
    H[t] &= \frac{0.99V[t-1]+I_{ff}[t]-G[t]}{1+G[t]} \\
    s_j[t] &= \mathbf{1}\{H_j[t]\ge 1+\theta_j^{-}[t],\ r_j[t-1]=0\}.

不应期中的神经元保持电位 0，不发放。其他神经元继续积分；发放神经元
自身重置到 0。抑制反转电位为 -1，隐式电导更新避免强抑制造成 Euler
过冲。时间单位是模拟时间步，膜时间常数为 100。
训练时先计算比较阈值用的偏移 ``theta_minus=exp(-1/1e5)*theta``，
发放后再按 batch 平均发放量增加，得到下一时刻保留的偏移。
初始偏移独立均匀分布于 ``[0, theta_initial)``，打破完全相同阈值的对称性。
这是简化的教学动力学，不是 Diehl–Cook 原论文复现。

每批结束用 ``functional.reset_net`` 清空循环输出、膜电位、电导和不应期，
再清空 STDP trace；阈值偏移保留。类别指派与评估期间冻结权重及阈值偏移，
停止监视器记录，仍按相同神经元动力学计算活动。固定种子控制初始化和编码。

训练集用 split seed 42 划分为 50,000 张无监督学习、5,000 张类别指派、
5,000 张验证；官方 10,000 张测试图像在配方冻结后评估。
标签不进入权重学习。类别指派用各神经元对各类的平均发放量，分类用每类
已指派神经元发放量的平均值；无有效发放的样本计错。
冻结前馈权重对照使用相同初始化、输入脉冲、循环机制和阈值更新规则，
单独建立固定类别指派。

.. code-block:: bash

    uv run python -m spikingjelly.activation_based.examples.stdp_mnist \
        --device cuda:0 --download --seed 0 \
        --output-dir logs/plasticity/conductance/seed-0
    uv run python -m spikingjelly.activation_based.examples.stdp_mnist \
        --device cuda:0 --checkpoint logs/plasticity/conductance/seed-0/selected.pt \
        --output-dir logs/plasticity/conductance/seed-0

默认参数为 700 步、batch 32、一轮训练；像素脉冲概率 ``pixel*0.064``，
基础阈值 1、抑制增量 1、电导时间常数 5、不应期 5、初始偏移上限 1。
STDP 的 pre/post trace 时间常数均为 20，LTD 调制 ``1e-4*weight``，
LTP 调制 ``0.02*(1-weight)``，阈值增加系数 0.5。
更新量除以实际 batch 大小，不按时间步平均。
初始权重为正均匀随机值，行和归一化目标为 78.4；每次更新依次限制到
``[0,1]``、按行归一化、再限制到 ``[0,1]``。

``--inhibition``、``--inhibition-tau``、``--refractory-steps`` 控制循环动力学，
``--v-threshold``、``--theta-plus``、``--theta-initial`` 控制阈值。
电导时间常数须有限且正；不应期是非负整数，初始偏移上限须有限且非负。
``--inhibition 0`` 关闭反馈；不应期 0 允许相邻步骤连续发放。
需要比较多轮时用 ``--epochs`` 和 ``--select-best``，仅根据验证集选 checkpoint，
并列选更早轮次。其他种子用冻结的相同轮数。
``--train-samples`` 等小样本选项用于检查，不能证明完整测试正确率。
``--checkpoint`` 使用保存的动力学、编码与类别映射，不重新训练或指派类别。
不支持旧硬竞争、负电流或非单位时间步实验的 checkpoint；需使用原实验源码
或重新训练。新实验输出到独立的 ``logs/plasticity/conductance`` 目录。

验证状态
----------------------

配方仅按验证集选择：三个完整训练候选为
``(theta_plus, ltp)=(0.2,0.01)、(0.5,0.01)、(0.5,0.02)``。
每个候选完成前两轮，第三轮在预设实验预算处停止，未完成的状态不参与选择。
各候选最好验证正确率分别为 85.82%、87.00%、87.20%；选择最后一组的一轮
checkpoint。对应感受野平均余弦相似度约 0.744，数字 8 指派数为 134/400；
弱稳态短训练曾达到约 0.959 和 352/400，表明分工有所改善，但这些诊断
不能替代任务验收。完整测试结果如下。模型和类别映射在测试前冻结，全部 10,000 张样本参与，
未按测试结果选择配方或忽略种子。

.. list-table:: 电导竞争的完整 MNIST 测试
   :header-rows: 1

   * - Seed
     - STDP
     - 冻结前馈对照
     - 差值（百分点）
   * - 0
     - 87.51%
     - 29.89%
     - +57.62
   * - 1
     - 86.66%
     - 29.56%
     - +57.10
   * - 2
     - 85.88%
     - 28.70%
     - +57.18

三种子均值 **86.68%**，样本标准差 **0.82 个百分点**，
达到平均正确率高于 85% 的门槛，且每个种子也高于 85%。
所有测试样本均有活动。环境为 Python 3.11.14、PyTorch 2.7.1+cu118、
torchvision 0.22.1+cu118，正式测试使用 g2 的 A100-SXM4-80GB。
此结果仅支持本节浅层配方，不代表纯 STDP 可以训练好任意深层网络。

复现其他种子时使用相同默认配方和一轮训练，将上述命令的 ``--seed`` 改为
1、2，并使用各自输出目录。checkpoint 保存权重、阈值偏移、固定类别映射和配置；
环境与指标记录在 JSON 中，独立评估不要求重新训练。所有配方筛选、失败实验及验收
账本均保留在实验输出目录；实验未引入自动重试静默样本或提高输入强度。


参考
----------------------

* `Diehl and Cook (2015) <https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full>`_
* `Florian (2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_
