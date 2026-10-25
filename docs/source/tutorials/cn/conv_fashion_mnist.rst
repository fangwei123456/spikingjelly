使用卷积SNN识别Fashion-MNIST
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

English version: :doc:`../en/conv_fashion_mnist`

在本节教程中，我们将搭建一个卷积脉冲神经网络，对 `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ 数据集进行
分类。Fashion-MNIST数据集，与MNIST数据集的格式相同，均为 ``1 * 28 * 28`` 的灰度图片。

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

将图片直接输入到SNN，而不是编码后再输入，是近年来深度SNN的常见做法，我们在此教程中也使用这样的方法。在这种情况下，实际的 ``图片-脉冲`` 编码是由网络中的前三层，也就是 \
``{Conv2d-BatchNorm2d-IFNode}`` 完成。

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
沿用 :doc:`./lif_fc_mnist` 的 MSE 损失、分类规则和每批次状态重置。训练和评估时，将泊松编码及外部 T 步累加、平均替换为 ``out_fr = net(img)``；``CSNN.forward()`` 已执行 T 步并取平均。数据集改为 Fashion-MNIST：

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

    uv run --no-sync python -m spikingjelly.activation_based.examples.conv_fashion_mnist -h

我们使用如下命令进行训练，其中为了加快训练速度，启用了混合精度训练：

.. code-block:: shell

    uv run --no-sync python -m spikingjelly.activation_based.examples.conv_fashion_mnist -T 4 -device cuda:0 -b 256 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -opt sgd -lr 0.1 -j 8

以下结果来自历史版本，未在当前 V2 上复测。

输出为：

.. code-block:: shell

    Namespace(T=4, device='cuda:0', b=256, epochs=64, j=8, data_dir='/datasets/FashionMNIST/', out_dir='./logs', resume=None, amp=True, opt='sgd', momentum=0.9, lr=0.1, channels=128)
    Mkdir ./logs/T4_b256_sgd_lr0.1_c128_amp.
    Namespace(T=4, device='cuda:0', b=256, epochs=64, j=8, data_dir='/datasets/FashionMNIST/', out_dir='./logs', resume=None, amp=True, opt='sgd', momentum=0.9, lr=0.1, channels=128)
    ./logs/T4_b256_sgd_lr0.1_c128_amp
    epoch =0, train_loss = 0.0325, train_acc = 0.7875, test_loss = 0.0248, test_acc = 0.8543, max_test_acc = 0.8543
    train speed = 7109.7899 images/s, test speed = 7936.2602 images/s
    escape time = 2022-05-24 21:42:15

    Namespace(T=4, device='cuda:0', b=256, epochs=64, j=8, data_dir='/datasets/FashionMNIST/', out_dir='./logs', resume=None, amp=True, opt='sgd', momentum=0.9, lr=0.1, channels=128)
    ./logs/T4_b256_sgd_lr0.1_c128_amp
    epoch =1, train_loss = 0.0217, train_acc = 0.8734, test_loss = 0.0201, test_acc = 0.8758, max_test_acc = 0.8758
    train speed = 7712.5343 images/s, test speed = 7902.5029 images/s
    escape time = 2022-05-24 21:43:13

    ...

    Namespace(T=4, device='cuda:0', b=256, epochs=64, j=8, data_dir='/datasets/FashionMNIST/', out_dir='./logs', resume=None, amp=True, opt='sgd', momentum=0.9, lr=0.1, channels=128)
    ./logs/T4_b256_sgd_lr0.1_c128_amp
    epoch =63, train_loss = 0.0024, train_acc = 0.9941, test_loss = 0.0113, test_acc = 0.9283, max_test_acc = 0.9308
    train speed = 7627.8147 images/s, test speed = 7868.9090 images/s
    escape time = 2022-05-24 21:42:16

最终获得了 ``max_test_acc = 0.9308`` 的性能。

下图展示了训练过程中正确率的变化：

.. image:: ../../_static/tutorials/conv_fashion_mnist/fmnist_logs.*
    :width: 100%

可视化编码器
-------------------------------------------
如前所述，我们将图片直接送入网络，实际的编码过程是由网络中的首个 ``{Conv2d-BatchNorm2d-IFNode}`` 实现的。现在让我们提取出网络中的编码器，输入图片，\
并将输出脉冲可视化。在同一个 ``CSNN`` 类中添加如下的方法：

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
                                plt.savefig(os.path.join(vs_dir_i, f's_{t}.png'))
                                plt.savefig(os.path.join(vs_dir_i, f's_{t}.pdf'))
                                plt.savefig(os.path.join(vs_dir_i, f's_{t}.svg'))
                                plt.clf()

                        exit()
        # ...


我们加载已经训练好的模型，设置 ``batch_size=4`` （表示我们只保存4张图片和对应的编码后的脉冲），将图片保存到 ``./logs`` 下，按照如下命令运行：

.. code-block:: shell

    python -m spikingjelly.activation_based.examples.conv_fashion_mnist -T 4 -device cuda:0 -b 4 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -opt sgd -lr 0.1 -j 8 -resume ./logs/T4_b256_sgd_lr0.1_c128_amp/checkpoint_latest.pth -save-es ./logs

.. note::

    检查点使用 ``batch_size=256`` 进行训练（检查点路径中的 ``b256`` 表示），而上述推理命令使用 ``batch_size=4`` 仅保存4个样本用于可视化。网络处于 ``eval()`` 模式，BatchNorm 使用训练阶段学到的运行统计量。


运行后图片会保存到 ``./logs/visualization`` 文件夹中。下面展示2个输入图片，和对应的编码后的脉冲：

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
