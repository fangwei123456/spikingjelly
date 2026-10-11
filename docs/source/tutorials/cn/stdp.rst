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

``stdp_mnist`` 使用 784→3200 的无 bias 前馈层、同一类电流 LIF 和
``ElementWiseRecurrentContainer``。全部前馈权重从随机初始化，由现有
``STDPLearner`` 直接更新并施加显式权重约束；不使用梯度、预训练或可训练读出。
固定负循环连接形成软竞争，允许同时多个神经元放电，不筛选单个赢家，也不把邻居膜电位归零。

模型与局部适应
^^^^^^^^^^^^^^^^^^^^^^

令输入脉冲为 x，前馈电流为 z=Wx。第 j 个神经元接收：

.. math::

    U_j[t] &= g_j\left(z_j[t]-\frac{1}{N}\sum_k z_k[t]\right)
      -a\left(\sum_k s_k[t-1]-s_j[t-1]\right),\\
    I_j[t] &= I_j[t-1]+\frac{U_j[t]-I_j[t-1]}{5},\\
    H_j[t] &= V_j[t-1]+\frac{I_j[t]+B_j[t-1]-A_j[t-1]-V_j[t-1]}{100}.

这是普通 SJ ``SimpleLIFNode(tau=100, decay_input=True)`` 电流积分。
基础阈值为 0.01，自身放电后复位到 0，自身不应期 5 步；不应期内只固定自身膜电位。
负电流可使膜电位低于零，不设置电压下限或电导式分母。
输入电流减去同一样本的群体平均以抑制共有驱动；循环抑制仅由其他神经元上一步发放驱动。

所有神经元具有同样的参数，通过各自活动历史形成适应状态差异：

.. math::

    A_j[t] &= \exp(-1/200)A_j[t-1]+0.01s_j[t],\\
    B_j[t] &= B_j[t-1]+\eta_B\left(\frac{120}{NT}
        -\frac{1}{M}\sum_{n=1}^{M}s_{nj}[t]\right).

A 是快速自身放电适应电流，降低重复响应；B 是慢速、可正可负的局部兴奋性电流，
以无标签目标率误差调整参与度。M 为实际 batch 大小，N 为神经元数，T 为样本步数。
B 初始均匀分布于 [-0.01,0]，训练后冻结。这是工程上的局部稳态控制，
不宣称精确生物通道模型、全局稳定性或在权重持续变化时精确实现目标率。
正 B 可产生背景放电，因此不以“更稀疏”或“均率平衡”代替任务正确率。

输入增益在每个训练 batch 结束更新。C 为固定 split seed 42 的 50,000 张学习图像
的像素样本协方差，不读取标签。令 ``f_j=W_j-mean(W,dim=0)``：

.. math::

    \sigma_j^2=f_j C f_j^\top,\qquad
    g_j=\sqrt{\sigma_{j,\mathrm{initial}}^2/\sigma_j^2}.

该约束保留随机初始化时每个神经元的投影输入方差尺度，减小权重学习导致的工作点漂移。
退化或非有限方差会明确报错。C、初始参考方差和增益随 checkpoint 保存；评估不更新它们。
这是训练集无标签统计约束，不是任务梯度或测试集统计适应。

编码、规则与读出
^^^^^^^^^^^^^^^^^^^^^^

输入使用普通 ``SimpleIFNode(v_threshold=1, v_reset=None)``，每个样本、像素的初始相位
独立均匀分布于 [0,1)。每步累加 ``pixel*0.064`` 并作自身软复位，700 步的
每像素计数与期望值误差不超过约一个脉冲。此规则发放编码降低计数噪声，也改变时间相关性；
不能将其解释为与 Poisson 整条学习轨迹等价。

STDP pre/post trace 均为 20 步，使用当前脉冲加入后的 trace。
``f_pre=nu*weight``、``f_post=nu*(1-weight)``，默认 nu=0.00229619779275505。
每步 ``step(on_grad=False, scale=1/实际batch大小)``，直接 ``weight += delta``；
时间维不平均。随后限制到 [0,1]、行和归一化到 78.4、再限制到 [0,1]。
写入 ``.grad`` 的其他用法仍只是优化器交付通道；本实例不使用该通道。

每批清空膜电位、A、不应期、滤波电流、循环输出和 STDP trace；保留 B、权重和输入增益。
类别指派和评估冻结全部持久学习状态，快动力学仍继续。标签只用于训练后的统计类别指派和指标。
每个神经元指派给平均发放次数最高的类别；不发放者未指派。预测按各类已指派神经元的
原始发放次数取平均；全部分数为零的样本未分类并计为错误。

固定划分为学习 50,000、类别指派 5,000、验证 5,000，官方测试 10,000。
默认完整训练一轮，batch 32、700 步。默认负循环强度 a=0.09978118161925603，
``--homeostasis-gain`` 为 0.007774997243242785，``--bias-initial`` 为 0.01，
``--v-threshold`` 为原生电压 0.01。所有时间常数单位为模拟步，电流滤波 tau 至少为 1。

.. code-block:: bash

    uv run --no-sync python -m spikingjelly.activation_based.examples.stdp_mnist \
        --device cuda:0 --seed 0 --data-dir logs/plasticity/data --download \
        --output-dir logs/plasticity/current/seed-0
    uv run --no-sync python -m spikingjelly.activation_based.examples.stdp_mnist \
        --device cuda:0 --checkpoint logs/plasticity/current/seed-0/selected.pt \
        --data-dir logs/plasticity/data --output-dir logs/plasticity/current/seed-0

其他种子使用相同默认配方与一轮训练，改为 ``--seed 1``、``--seed 2`` 并使用独立输出目录。
``--checkpoint`` 使用已保存配置和映射，不重新指派或训练；重放结果不要求重新训练。
小样本选项用于运行检查，不能证明完整测试效果。

迁移与实测结果
^^^^^^^^^^^^^^^^^^^^^^

本实例替换此前电导形式。``--theta-plus`` 和 ``--theta-initial`` 改为原生电流单位的
``--homeostasis-gain`` 与 ``--bias-initial``；不自动读取旧电导或硬竞争 checkpoint。
旧实验使用归档源码；当前电流实验的旧字段仅通过单次显式配置转换生成新格式，所有张量和映射保持不变。
没有长期兼容分支。新输出目录为 ``logs/plasticity/current``，与旧电导实验分离；重复试验请用独立目录。

配方与一轮训练由独立验证选择：同规模随机编码完整训练验证 88.76%，
冻结该模型仅改 IF 评估编码为 89.16%，IF 从头短训练为 86.72%/冻结对照 82.94%，
IF 从头完整训练为 89.58%/对照 83.04%。这些是不同阶段，不能互相冒充同一次训练。
随后固定相同配方、轮数，seed 1、2 从头训练；全部三个模型和映射固定后才进行官方测试。
不根据测试分数改配方，也不忽略低分种子。

.. list-table:: 普通 IF 编码、电流 LIF 的完整 MNIST 测试
   :header-rows: 1

   * - Seed
     - STDP
     - 冻结前馈对照
     - 差值（百分点）
   * - 0
     - 90.63%
     - 84.01%
     - +6.62
   * - 1
     - 89.98%
     - 83.20%
     - +6.78
   * - 2
     - 90.06%
     - 84.06%
     - +6.00

三种子均值 **90.2233%**，总体标准差 **0.2894 个百分点**，平均配对提升 **6.4667 个百分点**。
每种子均评估全部 10,000 张。seed 0 的一个无活动样本计为错误。
环境为 Python 3.11.14、PyTorch 2.7.1+cu118、torchvision 0.22.1+cu118；测试使用 V100S-PCIE-32GB。
这些测得的浅层配方结果不是 Diehl–Cook 原论文复现，也不支持任意深层网络的纯 STDP 性能声明。
原始 JSON、配置、checkpoint、源码哈希和命令均保留；代码迁移必须通过同 checkpoint 的数值及完整评估复核。


参考
----------------------

* `Diehl and Cook (2015) <https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full>`_
* `Florian (2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_
