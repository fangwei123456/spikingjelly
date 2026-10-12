STDP学习
=======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

English version: :doc:`../en/stdp`

本页介绍局部突触可塑性，并演示其与任务梯度的组合；最后给出经过验证的 MNIST 配方。
支持范围、时间单位、奖励对齐和接口迁移见
:doc:`../../APIs/spikingjelly.activation_based.learning`。

生物可解释性的学习规则一直备受SNN研究者的关注。在SpikingJelly中提供了STDP(Spike Timing Dependent Plasticity) \
学习器，可以用于卷积或全连接层的权重学习。


STDP(Spike Timing Dependent Plasticity)
-----------------------------------------------------

STDP(Spike Timing Dependent Plasticity)的经典实验见 [#STDP]_，是在生物实验中发现的一种突触可塑性机制。实验发现，突触权重 \
受到突触连接的前神经元(pre)和后神经元(post)的脉冲发放的影响，具体而言是：

如果pre神经元先发放脉冲，post神经元后发放脉冲，则突触的权重会增大；
如果pre神经元后发放脉冲，post神经元先发放脉冲，则突触的权重会减小。

生理实验数据拟合的曲线如下图 [#STDP_figure]_ 所示：

.. image:: ../../_static/tutorials/stdp/stdp.*
    :width: 100%


STDP可以使用如下公式进行拟合：

.. math::

    \begin{align}
	\begin{split}
	\Delta w_{ij} =
	\begin{cases}
		A\exp(\frac{-|t_{i}-t_{j}|}{\tau_{+}}) , t_{i} \leq t_{j}, A > 0\\
		B\exp(\frac{-|t_{i}-t_{j}|}{\tau_{-}}) , t_{i} > t_{j}, B < 0
	\end{cases}
    \end{split}
    \end{align}

其中 :math:`A, B` 是突触权重变化的最大值，:math:`\tau_{+}, \tau_{-}` 是时间常数。

上述标准的STDP公式在实践中使用较为繁琐，因其需要记录前后神经元所有的脉冲发放时刻。实践中通常使用迹 [#Trace]_ 的方式来实现STDP。

对于pre神经元 :math:`i` 和post神经元 :math:`j`，分别使用迹 :math:`tr_{pre}[i]` 和 :math:`tr_{post}[j]` 来记录其脉冲发放。迹的更新类似于LIF神经元：

.. math::

    tr_{pre}[i][t] = tr_{pre}[i][t-1] -\frac{tr_{pre}[i][t-1]}{\tau_{pre}} + s[i][t]

    tr_{post}[j][t] = tr_{post}[j][t-1] -\frac{tr_{post}[j][t-1]}{\tau_{post}} + s[j][t]

其中 :math:`\tau_{pre}, \tau_{post}` 是pre和post迹的时间常数，:math:`s[i][t], s[j][t]` 是在 :math:`t` 时刻pre神经元 :math:`i` 和post神经元 :math:`j` \
发放的脉冲，取值仅为0或1。

突触权重的更新按照：

.. math::

    \Delta W[i][j][t] = F_{post}(w[i][j][t]) \cdot tr_{pre}[i][t] \cdot s[j][t] - F_{pre}(w[i][j][t]) \cdot tr_{post}[j][t] \cdot s[i][t]

其中 :math:`F_{pre}, F_{post}` 是控制突触改变量的函数。

实现中的权重布局为 ``W[post, pre]``；上式的生物记号使用 pre=i、post=j。
STDP 采用 ``1-1/tau`` 衰减，要求有限 ``tau>=1``；当前脉冲先加入 trace 再计算更新。
更新按 batch、卷积空间位置和多步时间求和，调用者需要显式调整 ``scale``。
同时发放是否抵消取决于权重调制函数；固定权重的序列累计不等于逐步改权重。

STDP学习器
-----------------------------------------------------
:class:`spikingjelly.activation_based.learning.STDPLearner` 提供了STDP学习器的实现，支持卷积和全连接层，请读者先阅读其API文档以获取使用方法。

我们使用 ``STDPLearner`` 搭建一个简单的 ``4x3`` 网络，并且将权重设置为 ``0.4``：

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, layer, learning
    from matplotlib import pyplot as plt

    def f_weight(x):
        return torch.clamp(x, -1, 1.)

    torch.manual_seed(0)

    w_min, w_max = -1., 1.
    tau_pre, tau_post = 2., 2.
    N_in, N_out = 4, 3
    T = 128
    batch_size = 2
    lr = 0.01
    net = nn.Sequential(
        layer.Linear(N_in, N_out, bias=False),
        neuron.IFNode()
    )
    nn.init.constant_(net[0].weight.data, 0.4)

``STDPLearner`` 可以将负的权重的更新量 ``- delta_w * scale`` 叠加到参数的梯度上，这只是借用优化器更新通道，不是任务损失梯度，也不保证任务正确率提高。

我们可以将其和优化器、学习率调节器等深度学习中的模块一起使用。这里我们使用最简单的权重更新策略：

.. math::

    W = W - lr \cdot \nabla W

其中 :math:`\nabla W` 是使用STDP得到的权重更新量取负后的 ``- delta_w * scale``。因而借助优化器可以实现 ``weight.data = weight.data - lr * weight.grad = weight.data + lr * delta_w * scale``。


这可以使用最朴素的 :class:`torch.optim.SGD` 实现，只需要设置 ``momentum=0.``：

.. code-block:: python

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.)

接下来生成输入脉冲，设置 ``STDPLearner``：

.. code-block:: python

    in_spike = (torch.rand([T, batch_size, N_in]) > 0.7).float()
    learner = learning.STDPLearner(step_mode='s', synapse=net[0], sn=net[1],
                                   tau_pre=tau_pre, tau_post=tau_post,
                                   f_pre=f_weight, f_post=f_weight)

接下来送入数据计算。需要注意的是，为了便于画图，我们对数据与权重进行切片，这样使得 ``shape = [T, *, *]`` 变为 ``shape = [T]``：

.. code-block:: python

    out_spike = []
    trace_pre = []
    trace_post = []
    weight = []
    with torch.no_grad():
        for t in range(T):
            optimizer.zero_grad()
            out_spike.append(net(in_spike[t]))
            learner.step(on_grad=True)
            optimizer.step()
            net[0].weight.data.clamp_(w_min, w_max)
            weight.append(net[0].weight.data.clone())
            trace_pre.append(learner.trace_pre)
            trace_post.append(learner.trace_post)

    out_spike = torch.stack(out_spike)   # [T, batch_size, N_out]
    trace_pre = torch.stack(trace_pre)   # [T, batch_size, N_in]
    trace_post = torch.stack(trace_post) # [T, batch_size, N_out]
    weight = torch.stack(weight)         # [T, N_out, N_in]

    in_spike = in_spike[:, 0, 0]
    out_spike = out_spike[:, 0, 0]
    trace_pre = trace_pre[:, 0, 0]
    trace_post = trace_post[:, 0, 0]
    weight = weight[:, 0, 0]

    learner.close()

完整的代码位于 ``spikingjelly/activation_based/examples/plasticity/stdp_trace.py``。

将 ``in_spike, out_spike, trace_pre, trace_post, weight`` 画出，得到下图：

.. image:: ../../_static/tutorials/stdp/stdp_trace.*
    :width: 100%

这与 [#Trace]_ 中的Fig.3是一致的（注意下图中使用 `j` 表示pre神经元，`i` 表示后神经元，与我们相反）：

.. image:: ../../_static/tutorials/stdp/trace_paper_fig3.*
    :width: 100%


这里的随机数据只演示按层分工的调用流程，不是有效任务训练配方。
按层分工需在 backward 后清除 STDP 层的任务梯度；同参数叠加则保留该梯度。
``sn`` 必须指向实际 post 活动模块；无 momentum、无 weight decay 的 SGD 才对应上述直接更新。
``step()`` 消费活动记录并保留 trace；``reset()`` 清学习状态与记录。
评估需显式 ``disable()``，``net.eval()`` 不会停止记录；结束时用 ``close()`` 释放 hook。

与梯度下降混合使用
-----------------------------------------------------
在SNN中一种广泛使用STDP的做法是，使用STDP和梯度下降分别训练网路中的不同层。下面介绍如何使用 ``STDPLearner`` 实现这一做法。

我们的目标是搭建一个深度卷积SNN，使用STDP训练卷积层，使用梯度下降法训练全连接层。首先定义超参数：

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import SGD, Adam
    from spikingjelly.activation_based import learning, layer, neuron, functional

    T = 8
    N = 2
    C = 3
    H = 32
    W = 32
    lr = 0.1
    tau_pre = 2.
    tau_post = 100.
    step_mode = 'm'

我们使用 ``shape = [T, N, C, H, W] = [8, 2, 3, 32, 32]`` 的输入。


接下来定义STDP的权重函数以及网络，这里我们搭建的是一个简单的卷积SNN，且使用多步模式：

.. code-block:: python

    def f_weight(x):
        return torch.clamp(x, -1, 1.)


    net = nn.Sequential(
        layer.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Flatten(),
        layer.Linear(16 * 8 * 8, 64, bias=False),
        neuron.IFNode(),
        layer.Linear(64, 10, bias=False),
        neuron.IFNode(),
    )

    functional.set_step_mode(net, step_mode)

我们希望使用STDP训练 ``layer.Conv2d``，其他层使用梯度下降训练。首先定义使用STDP训练的层类型：

.. code-block:: python

    instances_stdp = (layer.Conv2d, )

对于每个类型为 ``instances_stdp`` 的层，我们都使用一个STDP学习器：

.. code-block:: python

    stdp_learners = []

    for i in range(net.__len__()):
        if isinstance(net[i], instances_stdp):
            stdp_learners.append(
                learning.STDPLearner(step_mode=step_mode, synapse=net[i], sn=net[i+1], tau_pre=tau_pre, tau_post=tau_post,
                                    f_pre=f_weight, f_post=f_weight)
            )

接下来进行参数分组，将类型为 ``instances_stdp`` 的层参数和其他类型的层的参数，分别放置到不同的优化器中。\
这里我们使用 ``Adam`` 作为梯度下降训练的参数的优化器，使用 ``SGD`` 作为STDP训练的参数的优化器：


.. code-block:: python

    params_stdp = []
    for m in net.modules():
        if isinstance(m, instances_stdp):
            for p in m.parameters():
                params_stdp.append(p)

    params_stdp_set = set(params_stdp)
    params_gradient_descent = []
    for p in net.parameters():
        if p not in params_stdp_set:
            params_gradient_descent.append(p)

    optimizer_gd = Adam(params_gradient_descent, lr=lr)
    optimizer_stdp = SGD(params_stdp, lr=lr, momentum=0.)

在实际任务中，输入和输出应该是从数据集中抽样得到的，我们这里仅仅是做示例，因此手动生成：

.. code-block:: python

    x_seq = (torch.rand([T, N, C, H, W]) > 0.5).float()
    target = torch.randint(low=0, high=10, size=[N])

接下来就是参数优化的主要步骤了，在实际任务中下面的代码通常会放到训练的主循环中。我们的代码与纯梯度下降会略有不同。

首先清零所有梯度，进行前向传播，计算出损失，反向传播：

.. code-block:: python

    optimizer_gd.zero_grad()
    optimizer_stdp.zero_grad()
    y = net(x_seq).mean(0)
    loss = F.cross_entropy(y, target)
    loss.backward()

需要注意的是，尽管 ``optimizer_gd`` 只会对 ``params_gradient_descent`` 中的参数进行梯度下降，但调用 ``loss.backward()`` \
后整个网络中所有的参数都会计算出梯度，包括那些我们只想使用STDP进行优化的参数。

因此，我们需要将使用梯度下降得到的 ``params_stdp`` 的梯度进行清零：

.. code-block:: python

    optimizer_stdp.zero_grad()

接下来就是使用STDP学习器计算出参数更新量，然后使用2个优化器，对整个网络的参数进行更新：


.. code-block:: python

    for i in range(stdp_learners.__len__()):
        stdp_learners[i].step(on_grad=True)

    optimizer_gd.step()
    optimizer_stdp.step()


以 ``STDPLearner`` 为代表的所有学习器都是 ``MemoryModule`` 的子类，其内部记忆状态包括了突触前后神经元的迹 ``trace_pre, trace_post`` ；\
另外，学习器内部用于记录神经元活动的监视器存储了突触前后神经元的发放历史；这些发放历史也可以视作学习器的内部记忆状态。因此，必须及时调用学习器的 ``reset()`` 方法，\
来清空其学习状态。``step()`` 已消费活动记录；未消费的记录才会继续积累。通常的做法是：在每个batch结束后，将学习器和网络一起重制：

.. code-block:: python

    functional.reset_net(net)
    for i in range(stdp_learners.__len__()):
        stdp_learners[i].reset()


完整的示例代码如下：

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import SGD, Adam
    from spikingjelly.activation_based import learning, layer, neuron, functional

    T = 8
    N = 2
    C = 3
    H = 32
    W = 32
    lr = 0.1
    tau_pre = 2.
    tau_post = 100.
    step_mode = 'm'

    def f_weight(x):
        return torch.clamp(x, -1, 1.)


    net = nn.Sequential(
        layer.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Flatten(),
        layer.Linear(16 * 8 * 8, 64, bias=False),
        neuron.IFNode(),
        layer.Linear(64, 10, bias=False),
        neuron.IFNode(),
    )

    functional.set_step_mode(net, step_mode)

    instances_stdp = (layer.Conv2d, )

    stdp_learners = []

    for i in range(net.__len__()):
        if isinstance(net[i], instances_stdp):
            stdp_learners.append(
                learning.STDPLearner(step_mode=step_mode, synapse=net[i], sn=net[i+1], tau_pre=tau_pre, tau_post=tau_post,
                                    f_pre=f_weight, f_post=f_weight)
            )


    params_stdp = []
    for m in net.modules():
        if isinstance(m, instances_stdp):
            for p in m.parameters():
                params_stdp.append(p)

    params_stdp_set = set(params_stdp)
    params_gradient_descent = []
    for p in net.parameters():
        if p not in params_stdp_set:
            params_gradient_descent.append(p)

    optimizer_gd = Adam(params_gradient_descent, lr=lr)
    optimizer_stdp = SGD(params_stdp, lr=lr, momentum=0.)



    x_seq = (torch.rand([T, N, C, H, W]) > 0.5).float()
    target = torch.randint(low=0, high=10, size=[N])

    optimizer_gd.zero_grad()
    optimizer_stdp.zero_grad()

    y = net(x_seq).mean(0)
    loss = F.cross_entropy(y, target)
    loss.backward()



    optimizer_stdp.zero_grad()

    for i in range(stdp_learners.__len__()):
        stdp_learners[i].step(on_grad=True)

    optimizer_gd.step()
    optimizer_stdp.step()

    functional.reset_net(net)
    for i in range(stdp_learners.__len__()):
        stdp_learners[i].reset()

    for learner in stdp_learners:
        learner.close()

可塑性机制演示
----------------------

三个脚本分别演示 STDP trace、reward 调制和 eligibility trace，不是任务 benchmark。
默认无 GUI，保存到 ``logs/plasticity``；可用 ``--show`` 和 ``--output-dir``。

.. code-block:: bash

    uv run --no-sync python -m spikingjelly.activation_based.examples.plasticity.stdp_trace
    uv run --no-sync python -m spikingjelly.activation_based.examples.plasticity.mstdp
    uv run --no-sync python -m spikingjelly.activation_based.examples.plasticity.mstdpet

MSTDP/MSTDPET 使用 ``exp(-1/tau)``，不是普通 STDP 的离散衰减。
每次前向后立即调用 reward step；``reward[t]`` 调制上一时刻 eligibility，首次更新为零。
无记录不推进时间，最终活动后的 reward 需后续前向。MSTDP 接受标量或 ``[M]`` reward，
MSTDPET 接受标量；创建 MSTDP 不再传 ``batch_size``。

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

    A_{nj}[t] &= \exp(-1/200)A_{nj}[t-1]+0.01s_{nj}[t],\\
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

    uv run --no-sync python -m spikingjelly.activation_based.examples.plasticity.stdp_mnist \
        --device cuda:0 --seed 0 --data-dir logs/plasticity/data --download \
        --output-dir logs/plasticity/current/seed-0
    uv run --no-sync python -m spikingjelly.activation_based.examples.plasticity.stdp_mnist \
        --device cuda:0 --checkpoint logs/plasticity/current/seed-0/selected.pt \
        --data-dir logs/plasticity/data --output-dir logs/plasticity/current/seed-0

其他种子使用相同默认配方与一轮训练，改为 ``--seed 1``、``--seed 2`` 并使用独立输出目录。
``--checkpoint`` 使用已保存配置和映射，不重新指派或训练；重放结果不要求重新训练。
小样本选项用于运行检查，不能证明完整测试效果。

实测结果
^^^^^^^^^^^^^^^^^^^^^^

配方与一轮训练由独立验证选择：同规模随机编码完整训练验证 88.76%，
冻结该模型仅改 IF 评估编码为 89.16%，IF 从头短训练为 86.72%/冻结对照 82.94%，
IF 从头完整训练为 89.58%/对照 83.04%。这些是不同阶段，不能互相冒充同一次训练。
随后固定相同配方、轮数，seed 1、2 从头训练；全部三个模型和映射固定后才进行官方测试。
不根据测试分数改配方，也不忽略低分种子。

冻结前馈对照保留相同初始化和逐位相同的输入脉冲，只关闭 STDP 权重更新；
训练期间仍适应 B，保留相同快动力学，并独立进行训练后的类别指派。
因此它不是完全不学习的网络。验收门槛是三种子平均至少 90%，而非每种子至少 90%。

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

.. [#STDP] Bi, Guo-qiang, and Mu-ming Poo. "Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type." Journal of neuroscience 18.24 (1998): 10464-10472.

.. [#STDP_figure] Froemke, Robert C., et al. "Contribution of individual spikes in burst-induced long-term synaptic modification." Journal of neurophysiology (2006).

.. [#Trace] Morrison, Abigail, Markus Diesmann, and Wulfram Gerstner. "Phenomenological models of synaptic plasticity based on spike timing." Biological cybernetics 98.6 (2008): 459-478.

* `Diehl and Cook (2015) <https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full>`_
* `Florian (2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_
