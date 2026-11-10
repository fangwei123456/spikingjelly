使用单层全连接SNN识别MNIST
====================================

本教程作者：\ `Yanqi-Chen <https://github.com/Yanqi-Chen>`__

English version: :doc:`../en/lif_fc_mnist`

你将搭建一个全连接 SNN，用泊松编码器生成输入脉冲，并通过替代梯度训练它识别 MNIST 数字。

从头搭建一个简单的SNN网络
-------------------------

在PyTorch中搭建神经网络时，我们可以简单地使用\ ``nn.Sequential``\ 将多个网络层堆叠得到一个前馈网络，输入数据将依序流经各个网络层得到输出。

`MNIST数据集 <http://yann.lecun.com/exdb/mnist/>`__\ 包含若干尺寸为\ :math:`28\times 28`\ 的8位灰度图像，总共有0~9共10个类别。以MNIST的分类为例，一个简单的单层ANN网络如下：

.. code-block:: python

    import torch
    import torch.nn as nn
    import time
    import torch.nn.functional as F
    from torch.cuda import amp
    from spikingjelly.activation_based import layer, neuron, surrogate, encoding, functional

    nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10, bias=False),
        nn.Softmax(dim=1)
        )

把 ANN 的 Softmax 换成 LIF 神经元，就得到这个示例使用的 SNN。连接层使用 :mod:`spikingjelly.activation_based.layer` 中的实现：

.. code-block:: python

    tau = 2.0
    net = nn.Sequential(
        layer.Flatten(),
        layer.Linear(28 * 28, 10, bias=False),
        neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
        )

其中膜电位衰减常数\ :math:`\tau`\ 需要通过参数\ ``tau``\ 设置，替代函数这里选择\ ``surrogate.ATan``\。

训练SNN网络
-----------

首先指定好训练参数如学习率等以及若干其他配置

优化器默认使用Adam，以及使用泊松编码器，在每次输入图片时进行脉冲编码

.. code-block:: python

    lr = 1e-3
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 使用泊松编码器
    encoder = encoding.PoissonEncoder()

编码器用法见 :mod:`spikingjelly.activation_based.encoding`。

训练循环包含三个步骤：

1. 脉冲神经元的输出是二值的，而直接将单次运行的结果用于分类极易受到编码带来的噪声干扰。因此一般认为脉冲网络的输出是输出层一段时间内的\ **发放频率**\ （或称发放率），发放率的高低表示该类别的响应大小。因此网络需要运行一段时间，即使用\ ``T``\ 个时刻后的\ **平均发放率**\ 作为分类依据。

2. 我们希望的理想结果是除了正确的神经元\ **以最高频率发放**\ ，其他神经元\ **保持静默**\ 。这里以 one-hot 发放率为目标，使用 MSE 损失。

3. 每次网络仿真结束后，需要\ **重置**\ 网络状态，即调用 ``functional.reset_net(net)``

结合以上三点，得到训练循环的核心代码如下。以下片段使用 :mod:`spikingjelly.activation_based.examples.lif_fc_mnist` 中的参数、DataLoader 和 AMP scaler 初始化：

.. code-block:: python

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            # 混合精度训练
            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.
                    # 运行T个时间步
                    for t in range(args.T):
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    out_fr = out_fr / args.T
                    # out_fr是shape=[batch_size, 10]的tensor
                    # 记录整个仿真时长内，输出层的10个神经元的脉冲发放率
                    loss = F.mse_loss(out_fr, label_onehot)
                    # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
                    # 这样的损失函数会使得：当标签i给定时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            train_acc += (out_fr.argmax(1) == label).float().sum().item()
            
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            functional.reset_net(net)

完整示例位于 :mod:`spikingjelly.activation_based.examples.lif_fc_mnist`，并使用 TensorBoard 保存训练日志。先查看命令行参数：

.. code-block:: shell

    python -m spikingjelly.activation_based.examples.lif_fc_mnist --help

训练该 SNN 所需的显存与仿真步数 ``T`` 线性相关，更长的 ``T`` 不一定带来更好的训练效果。\ ``T``
太大时，SNN在时间上展开后会变成一个非常深的网络，这将导致BPTT计算梯度时容易衰减或爆炸。

另外由于我们使用了泊松编码器，因此需要较大的 ``T``\ 保证编码带来的噪声不太大。

训练结果
--------

取\ ``tau=2.0,T=100,batch_size=64,lr=1e-3``\ ，对应的运行命令为

.. code-block:: shell

    python -m spikingjelly.activation_based.examples.lif_fc_mnist -tau 2.0 -T 100 -device cuda:0 -b 64 -epochs 100 -data-dir <PATH to MNIST> -amp -opt adam -lr 1e-3 -j 8

参考结果（commit ``5ca5c06e``）：

该次运行启用了混合精度。训练 100 个 Epoch 后，测试集最高正确率为 92.9%。脚本会保存检查点和 TensorBoard 日志，并在训练结束后写出 ``v_t_array.npy`` 与 ``s_t_array.npy``。正确率曲线如下：

.. image:: ../../_static/tutorials/lif_fc_mnist/acc.*
    :width: 100%

选取测试集中第一张图片：

.. image:: ../../_static/tutorials/lif_fc_mnist/input.png

用训好的模型进行分类，得到分类结果

.. code-block:: shell

   Firing rate: [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]

保存的膜电位和脉冲序列可以绘制为以下两种视图：

.. image:: ../../_static/tutorials/lif_fc_mnist/1d_spikes.*
    :width: 100%

.. image:: ../../_static/tutorials/lif_fc_mnist/2d_heatmap.*
    :width: 100%

这个样本被分类为数字 7，只有类别 7 对应的输出神经元发放了脉冲。完整训练代码见 `activation_based/examples/lif_fc_mnist.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/lif_fc_mnist.py>`_。
