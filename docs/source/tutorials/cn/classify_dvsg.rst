分类 DVS Gesture
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

English version: :doc:`../en/classify_dvsg`

你将把 :doc:`./neuromorphic_datasets` 中生成的 DVS128 Gesture 帧送入卷积 SNN，完成 11 类手势分类。

网络结构
-------------------------------------------
我们将使用 [#PLIF]_ 一文中定义的网络，其结构如下：

.. image:: ../../_static/tutorials/classify_dvsg/network.png
    :width: 100%

[#PLIF]_ 一文中的所有网络都在 :class:`spikingjelly.activation_based.model.parametric_lif_net` 中进行了定义，其中用于DVS Gesture的网络结构为：

.. code-block:: python

    # spikingjelly.activation_based.model.parametric_lif_net

    from copy import deepcopy

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import layer

    class DVSGestureNet(nn.Module):
        def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
            super().__init__()

            conv = []
            for i in range(5):
                if not conv:
                    in_channels = 2
                else:
                    in_channels = channels

                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))
                conv.append(layer.MaxPool2d(2, 2))

            self.conv_fc = nn.Sequential(
                *conv,

                layer.Flatten(),
                layer.Dropout(0.5),
                layer.Linear(channels * 4 * 4, 512),
                spiking_neuron(**deepcopy(kwargs)),

                layer.Dropout(0.5),
                layer.Linear(512, 110),
                spiking_neuron(**deepcopy(kwargs)),

                layer.VotingLayer(10)
            )

        def forward(self, x: torch.Tensor):
            return self.conv_fc(x)

训练
-------------------------------------------
训练循环沿用 :doc:`./conv_fashion_mnist`。你需要把数据集、输入形状和类别数改为 DVS128 Gesture 的设置。

定义网络，使用多步模式。神经元会根据输入设备自动选择实现：

.. code-block:: python

    # spikingjelly.activation_based.examples.classify_dvsg

    import torch
    import sys
    import torch.nn.functional as F
    from torch.cuda import amp
    from spikingjelly.activation_based import functional, surrogate, neuron
    from spikingjelly.activation_based.model import parametric_lif_net
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    import time
    import os
    import argparse
    import datetime

    def main():
        # ...
        net = parametric_lif_net.DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

        functional.set_step_mode(net, 'm')
        # ...


新建数据集：

.. code-block:: python

    # spikingjelly.activation_based.examples.classify_dvsg

    def main():
        # ...
        train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
        test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
        # ...

``DataLoader`` 返回的帧形状为 ``[N, T, C, H, W]``，其中第 0 维是批次。多步模式要求时间维在前，因此需要转置为 ``[T, N, C, H, W]``：

.. code-block:: python

    # spikingjelly.activation_based.examples.classify_dvsg

    def main():
        # ...
        for epoch in range(start_epoch, args.epochs):
            for frame, label in train_data_loader:
                optimizer.zero_grad()
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                # ...

            with torch.no_grad():
                for frame, label in test_data_loader:
                    frame = frame.to(args.device)
                    frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                    # ...

        # ...

DVS128 Gesture 有 11 类，因此 one-hot 目标的类别数设为 11：

.. code-block:: python

    # spikingjelly.activation_based.examples.classify_dvsg

    def main():
        # ...
        label_onehot = F.one_hot(label, 11).float()
        # ...

``DVSGestureNet`` 输出形状为 ``[T, N, 11]``。末层 ``VotingLayer`` 对每组 10 个神经元取平均，所以每步输出不是二值脉冲；对时间维取平均得到分类分数。

.. code-block:: python

    # spikingjelly.activation_based.model.parametric_lif_net

    class DVSGestureNet(nn.Module):
        # ...
        def forward(self, x: torch.Tensor):
            return self.conv_fc(x)

使用时间平均后的分类分数计算损失和正确率：

.. code-block:: python

    # spikingjelly.activation_based.examples.classify_dvsg

    def main():
        # ...
        out_fr = net(frame).mean(0)
        loss = F.mse_loss(out_fr, label_onehot)
        # ...

运行我们的网络：

.. code-block:: shell

    python -m spikingjelly.activation_based.examples.classify_dvsg -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir /datasets/DVSGesture/ -amp -opt adam -lr 0.001 -j 8

参考结果（commit ``e38dee59``，CuPy 后端）：最高测试正确率为 ``0.9375``。

该次运行的正确率曲线如下：


.. image:: ../../_static/tutorials/classify_dvsg/dvsg_logs.*
    :width: 100%


.. [#PLIF] Fang, Wei, et al. "Incorporating learnable membrane time constant to enhance learning of spiking neural networks." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 2661-2671. `CVF <https://openaccess.thecvf.com/content/ICCV2021/html/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.html>`_
