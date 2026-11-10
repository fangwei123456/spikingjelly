使用卷积SNN识别Fashion-MNIST
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

English version: :doc:`../en/conv_fashion_mnist`

你将搭建一个卷积 SNN，对 `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ 中的服饰图片进行分类。每张图片都是 ``1 * 28 * 28`` 的灰度图，与 MNIST 的数据格式相同。

网络结构
-------------------------------------------
我们使用最常见的卷积神经网络结构。具体而言，网络结构为：

``{Conv2d-BatchNorm2d-IFNode-MaxPool2d}×2 - Flatten - {Linear-IFNode}×2``

网络结构的定义如下：

完整代码见 :mod:`spikingjelly.activation_based.examples.conv_fashion_mnist`。

.. code-block:: python

    # spikingjelly.activation_based.examples.conv_fashion_mnist
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from spikingjelly.activation_based import neuron, functional, surrogate, layer
    from torch.utils.tensorboard import SummaryWriter
    import os
    import time
    import argparse
    from torch.cuda import amp
    import sys
    import datetime
    from spikingjelly import visualizing

    class CSNN(nn.Module):
        def __init__(self, T: int, channels: int):
            super().__init__()
            self.T = T

            self.conv_fc = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 14 * 14

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 7 * 7

            layer.Flatten(),
            layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(channels * 4 * 4, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            )

为了更快的训练速度，我们将网络设置成多步模式，并根据设备自动选择神经元实现。在上面的构造函数末尾添加如下代码：

.. code-block:: python

    # spikingjelly.activation_based.examples.conv_fashion_mnist

    class CSNN(nn.Module):
        def __init__(self, T: int, channels: int):
            # ...
            functional.set_step_mode(self, step_mode='m')

这个网络会在每个时间步直接接收同一张图片。前三层 ``{Conv2d-BatchNorm2d-IFNode}`` 将像素转换为脉冲，因此不需要单独的输入编码器。

网络的输入直接是 ``shape=[N, C, H, W]`` 的图片，我们将其添加时间维度，并扩展为 ``shape=[T, N, C, H, W]`` 的序列，然后送入到网络层。网络的输出定义为最后一层脉冲神经元的\
脉冲发放频率。因而，在同一个 ``CSNN`` 类中添加如下的前向传播方法：

.. code-block:: python

    # spikingjelly.activation_based.examples.conv_fashion_mnist
    class CSNN(nn.Module):
        def forward(self, x: torch.Tensor):
            # x.shape = [N, C, H, W]
            x_seq = x.unsqueeze(0).expand(self.T, -1, -1, -1, -1)  # [N, C, H, W] -> [T, N, C, H, W]
            x_seq = self.conv_fc(x_seq)
            fr = x_seq.mean(0)
            return fr


网络训练
-------------------------------------------
沿用 :doc:`./lif_fc_mnist` 的 MSE 损失、分类规则和每批次状态重置。训练和评估时直接执行 ``out_fr = net(img)``，因为 ``CSNN.forward()`` 已完成 T 步计算并取平均。数据集改为 Fashion-MNIST：

.. code-block:: python

    # spikingjelly.activation_based.examples.conv_fashion_mnist

    train_set = torchvision.datasets.FashionMNIST(
            root=args.data_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True)

    test_set = torchvision.datasets.FashionMNIST(
            root=args.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True)

可以使用如下命令查看训练参数：

.. code-block:: shell

    python -m spikingjelly.activation_based.examples.conv_fashion_mnist -h

我们使用如下命令进行训练，其中为了加快训练速度，启用了混合精度训练：

.. code-block:: shell

    python -m spikingjelly.activation_based.examples.conv_fashion_mnist -T 4 -device cuda:0 -b 256 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -opt sgd -lr 0.1 -j 8

参考结果（commit ``b8231711``，CuPy 后端）：最高测试正确率为 ``0.9308``，正确率曲线如下：

.. image:: ../../_static/tutorials/conv_fashion_mnist/fmnist_logs.*
    :width: 100%

可视化编码器
-------------------------------------------
首个 ``{Conv2d-BatchNorm2d-IFNode}`` 已经把图片编码为脉冲。为了查看这些脉冲，从 ``CSNN`` 中取出前三层，并在同一个类中添加以下方法：

.. code-block:: python

    # spikingjelly.activation_based.examples.conv_fashion_mnist
    class CSNN(nn.Module):
        # ...
        def spiking_encoder(self):
            return self.conv_fc[0:3]
    def main():
        # ...
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            max_test_acc = checkpoint['max_test_acc']
            if args.save_es is not None and args.save_es != '':
                net.eval()
                encoder = net.spiking_encoder()
                with torch.no_grad():
                    for img, label in test_data_loader:
                        img = img.to(args.device)
                        label = label.to(args.device)
                        # img.shape = [N, C, H, W]
                        img_seq = img.unsqueeze(0).expand(net.T, -1, -1, -1, -1)  # [N, C, H, W] -> [T, N, C, H, W]
                        spike_seq = encoder(img_seq)
                        functional.reset_net(encoder)
                        to_pil_img = torchvision.transforms.ToPILImage()
                        vs_dir = os.path.join(args.save_es, 'visualization')
                        os.mkdir(vs_dir)

                        img = img.cpu()
                        spike_seq = spike_seq.cpu()

                        img = F.interpolate(img, scale_factor=4, mode='bilinear')
                        # 28 * 28 is too small to read. So, we interpolate it to a larger size

                        for i in range(label.shape[0]):
                            vs_dir_i = os.path.join(vs_dir, f'{i}')
                            os.mkdir(vs_dir_i)
                            to_pil_img(img[i]).save(os.path.join(vs_dir_i, f'input.png'))
                            for t in range(net.T):
                                print(f'saving {i}-th sample with t={t}...')
                                # spike_seq.shape = [T, N, C, H, W]

                                visualizing.plot_2d_feature_map(spike_seq[t][i], 8, spike_seq.shape[2] // 8, 2, f'$S[{t}]$')
                                plt.savefig(os.path.join(vs_dir_i, f's_{t}.png'), pad_inches=0.02)
                                plt.savefig(os.path.join(vs_dir_i, f's_{t}.pdf'), pad_inches=0.02)
                                plt.savefig(os.path.join(vs_dir_i, f's_{t}.svg'), pad_inches=0.02)
                                plt.clf()

                        exit()
        # ...


加载训练好的模型，将 ``batch_size`` 设为 4，并把 4 张图片及其脉冲保存到 ``./logs``：

.. code-block:: shell

    python -m spikingjelly.activation_based.examples.conv_fashion_mnist -T 4 -device cuda:0 -b 4 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -opt sgd -lr 0.1 -j 8 -resume ./logs/T4_b256_sgd_lr0.1_c128_amp/checkpoint_latest.pth -save-es ./logs

.. note::

    检查点使用 ``batch_size=256`` 进行训练（检查点路径中的 ``b256`` 表示），而上述推理命令使用 ``batch_size=4`` 仅保存4个样本用于可视化。网络处于 ``eval()`` 模式，BatchNorm 使用训练阶段学到的运行统计量。


脚本把图片保存到 ``./logs/visualization``。以下是其中 2 张输入图片及其编码脉冲：

.. image:: ../../_static/tutorials/conv_fashion_mnist/visualization/0/input.*
    :width: 100%

.. image:: ../../_static/tutorials/conv_fashion_mnist/visualization/0/s_0.*
    :width: 100%

.. image:: ../../_static/tutorials/conv_fashion_mnist/visualization/0/s_1.*
    :width: 100%

.. image:: ../../_static/tutorials/conv_fashion_mnist/visualization/0/s_2.*
    :width: 100%

.. image:: ../../_static/tutorials/conv_fashion_mnist/visualization/0/s_3.*
    :width: 100%

.. image:: ../../_static/tutorials/conv_fashion_mnist/visualization/3/input.*
    :width: 100%

.. image:: ../../_static/tutorials/conv_fashion_mnist/visualization/3/s_0.*
    :width: 100%

.. image:: ../../_static/tutorials/conv_fashion_mnist/visualization/3/s_1.*
    :width: 100%

.. image:: ../../_static/tutorials/conv_fashion_mnist/visualization/3/s_2.*
    :width: 100%

.. image:: ../../_static/tutorials/conv_fashion_mnist/visualization/3/s_3.*
    :width: 100%
