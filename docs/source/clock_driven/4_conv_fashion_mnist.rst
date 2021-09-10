时间驱动：使用卷积SNN识别Fashion-MNIST
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

在本节教程中，我们将搭建一个卷积脉冲神经网络，对 `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`__ 数据集进行
分类。Fashion-MNIST数据集，与MNIST数据集的格式相同，均为 ``1 * 28 * 28`` 的灰度图片。

网络结构
-----------------

ANN中常见的卷积神经网络，大多数是卷积+全连接层的形式，我们在SNN中也使用类似的结构。导入相关的模块，继承 ``torch.nn.Module``，定义我
们的网络：

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from spikingjelly.clock_driven import neuron, functional, surrogate, layer
    from torch.utils.tensorboard import SummaryWriter
    import os
    import time
    import argparse
    import numpy as np
    from torch.cuda import amp
    _seed_ = 2020
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)

    class PythonNet(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T

接下来，我们在 ``PythonNet`` 的成员变量中添加卷积层和全连接层。我们添加2个卷积-BN-池化层：

.. code-block:: python

    self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 7 * 7
        )

``1 * 28 * 28`` 的输入经过这样的卷积层作用后，得到 ``128 * 7 * 7`` 的输出脉冲。

这样的卷积层，其实可以起到编码器的作用：在上一届教程，MNIST识别的代码中，我们使用泊松编码器，将图片编码成脉冲。实际上我们完全可以直接将
图片送入SNN，在这种情况下，SNN中的首层脉冲神经元层及其之前的层，可以看作是一个参数可学习的自编码器。具体而言，我们刚才定义的卷积层中的这些层：

.. code-block:: python

    nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    neuron.IFNode(surrogate_function=surrogate.ATan())

这3层网络，接收图片作为输入，输出脉冲，可以看作是编码器。

接下来，我们定义2层全连接网络，输出分类的结果。Fashion-MNIST共有10类，因
此输出层是10个神经元。

.. code-block:: python

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128 * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.Linear(128 * 4 * 4, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

接下来，定义前向传播：

.. code-block:: python

    def forward(self, x):
        x = self.static_conv(x)

        out_spikes_counter = self.fc(self.conv(x))
        for t in range(1, self.T):
            out_spikes_counter += self.fc(self.conv(x))

        return out_spikes_counter / self.T

避免重复计算
-------------------

我们可以直接训练这个网络，就像之前的MNIST分类那样。但我们如果重新审视网络的结构，可以发现，有一些计算是重复的：对于网络的前2层，即下面代码中的高亮部分：

.. code-block:: python
    :emphasize-lines: 2, 3

    self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 7 * 7
        )

这2层接收的输入图片，并不随 ``t`` 变化，但在 ``for`` 循环中，每次 ``img`` 都会重新经过这2层的计算，得到相同的输出。我们可以提取出这2层，
不参与时间上的循环。完整的代码如下：

.. code-block:: python

    class PythonNet(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T

            self.static_conv = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
            )

            self.conv = nn.Sequential(
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                nn.MaxPool2d(2, 2),  # 14 * 14

                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                nn.MaxPool2d(2, 2)  # 7 * 7

            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 7 * 7, 128 * 4 * 4, bias=False),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                nn.Linear(128 * 4 * 4, 10, bias=False),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
            )


        def forward(self, x):
            x = self.static_conv(x)

            out_spikes_counter = self.fc(self.conv(x))
            for t in range(1, self.T):
                out_spikes_counter += self.fc(self.conv(x))

            return out_spikes_counter / self.T


对于输入是不随时间变化的SNN，虽然SNN整体是有状态的，但网络的前几层可能没有状态，我们可以单独提取出这些层，将它们放到在时间上的循环之外，
避免额外计算。

训练网络
-----------------
完整的代码位于 :class:`spikingjelly.clock_driven.examples.conv_fashion_mnist`，训练命令如下：

.. code-block:: shell

    Classify Fashion-MNIST

    optional arguments:
      -h, --help            show this help message and exit
      -T T                  simulating time-steps
      -device DEVICE        device
      -b B                  batch size
      -epochs N             number of total epochs to run
      -j N                  number of data loading workers (default: 4)
      -data_dir DATA_DIR    root dir of Fashion-MNIST dataset
      -out_dir OUT_DIR      root dir for saving logs and checkpoint
      -resume RESUME        resume from the checkpoint path
      -amp                  automatic mixed precision training
      -cupy                 use cupy neuron and multi-step forward mode
      -opt OPT              use which optimizer. SDG or Adam
      -lr LR                learning rate
      -momentum MOMENTUM    momentum for SGD
      -lr_scheduler LR_SCHEDULER
                            use which schedule. StepLR or CosALR
      -step_size STEP_SIZE  step_size for StepLR
      -gamma GAMMA          gamma for StepLR
      -T_max T_MAX          T_max for CosineAnnealingLR

其中 ``-cupy`` 是使用cupy后端和多步神经元，关于它的更多信息参见 :doc:`../clock_driven/10_propagation_pattern` 和 :doc:`../clock_driven/11_cext_neuron_with_lbl`。

检查点会被保存在 ``tensorboard`` 日志文件的同级目录下。实验机器使用 `Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz` 的CPU和 `GeForce RTX 2080 Ti` 的GPU。

.. code-block:: shell

    (pytorch-env) root@e8b6e4800dae4011eb0918702bd7ddedd51c-fangw1598-0:/# python -m spikingjelly.clock_driven.examples.conv_fashion_mnist -opt SGD -data_dir /userhome/datasets/FashionMNIST/ -amp

    Namespace(T=4, T_max=64, amp=True, b=128, cupy=False, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)
    PythonNet(
      (static_conv): Sequential(
        (0): Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv): Sequential(
        (0): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=False
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=False
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (fc): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Linear(in_features=6272, out_features=2048, bias=False)
        (2): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=False
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (3): Linear(in_features=2048, out_features=10, bias=False)
        (4): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=False
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
      )
    )
    Mkdir ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp.
    Namespace(T=4, T_max=64, amp=True, b=128, cupy=False, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)
    ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp
    epoch=0, train_loss=0.028124165828697957, train_acc=0.8188267895299145, test_loss=0.023525000348687174, test_acc=0.8633, max_test_acc=0.8633, total_time=16.86261749267578
    Namespace(T=4, T_max=64, amp=True, b=128, cupy=False, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)
    ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp
    epoch=1, train_loss=0.018544567498163536, train_acc=0.883613782051282, test_loss=0.02161250041425228, test_acc=0.8745, max_test_acc=0.8745, total_time=16.618073225021362
    Namespace(T=4, T_max=64, amp=True, b=128, cupy=False, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)

    ...

    ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp
    epoch=62, train_loss=0.0010829827882937538, train_acc=0.997512686965812, test_loss=0.011441250185668468, test_acc=0.9316, max_test_acc=0.933, total_time=15.976636171340942
    Namespace(T=4, T_max=64, amp=True, b=128, cupy=False, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)
    ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp
    epoch=63, train_loss=0.0010746361010835525, train_acc=0.9977463942307693, test_loss=0.01154562517106533, test_acc=0.9296, max_test_acc=0.933, total_time=15.83976149559021

运行64轮训练后，训练集和测试集上的正确率如下：

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/train.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/test.*
    :width: 100%

在训练64个epoch后，最高测试集正确率可以达到93.3%，对于SNN而言是非常不错的性能，仅仅略低于 `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`__
的BenchMark中使用Normalization, random horizontal flip, random vertical flip, random translation, random rotation的ResNet18的94.9%正确率。

可视化编码器
------------------------------------

正如我们在前文中所述，直接将数据送入SNN，则首个脉冲神经元层及其之前的层，可以看作是一个可学习的编码器。具体而言，是我们的网络中如
下所示的高亮部分：

.. code-block:: python
    :emphasize-lines: 5, 6, 10

    class Net(nn.Module):
        def __init__(self, T):
            ...
            self.static_conv = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
            )

            self.conv = nn.Sequential(
                neuron.IFNode(surrogate_function=surrogate.ATan()),
            ...
            )

现在让我们来查看一下，训练好的编码器，编码效果如何。让我们新建一个python文件，导入相关的模块，并重新定义一个 ``batch_size=1`` 的数据加载器，因为我们想要一
张图片一张图片的查看：

.. code-block:: python

    from matplotlib import pyplot as plt
    import numpy as np
    from spikingjelly.clock_driven.examples.conv_fashion_mnist import PythonNet
    from spikingjelly import visualizing
    import torch
    import torch.nn as nn
    import torchvision

    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=1,
        shuffle=True,
        drop_last=False)

从保存网络的位置，即 ``log_dir`` 目录下，加载训练好的网络，并提取出编码器。在CPU上运行即可：

.. code-block:: python

    net = torch.load('./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp/checkpoint_max.pth', 'cpu')['net']
    encoder = nn.Sequential(
        net.static_conv,
        net.conv[0]
    )
    encoder.eval()

接下来，从数据集中抽取一张图片，送入编码器，并查看输出脉冲的累加值 :math:`\sum_{t} S_{t}`。为了显示清晰，我们还对输出的 ``feature_map``
的像素值做了归一化，将数值范围线性变换到 ``[0, 1]``。

.. code-block:: python

    with torch.no_grad():
        # 每遍历一次全部数据集，就在测试集上测试一次
        for img, label in test_data_loader:
            fig = plt.figure(dpi=200)
            plt.imshow(img.squeeze().numpy(), cmap='gray')
            # 注意输入到网络的图片尺寸是 ``[1, 1, 28, 28]``，第0个维度是 ``batch``，第1个维度是 ``channel``
            # 因此在调用 ``imshow`` 时，先使用 ``squeeze()`` 将尺寸变成 ``[28, 28]``
            plt.title('Input image', fontsize=20)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            out_spikes = 0
            for t in range(net.T):
                out_spikes += encoder(img).squeeze()
                # encoder(img)的尺寸是 ``[1, 128, 28, 28]``，同样使用 ``squeeze()`` 变换尺寸为 ``[128, 28, 28]``
                if t == 0 or t == net.T - 1:
                    out_spikes_c = out_spikes.clone()
                    for i in range(out_spikes_c.shape[0]):
                        if out_spikes_c[i].max().item() > out_spikes_c[i].min().item():
                            # 对每个feature map做归一化，使显示更清晰
                            out_spikes_c[i] = (out_spikes_c[i] - out_spikes_c[i].min()) / (out_spikes_c[i].max() - out_spikes_c[i].min())
                    visualizing.plot_2d_spiking_feature_map(out_spikes_c, 8, 16, 1, None)
                    plt.title('$\\sum_{t} S_{t}$ at $t = ' + str(t) + '$', fontsize=20)
                    plt.show()

下面展示2个输入图片，以及在最开始 ``t=0`` 和最后 ``t=7`` 时刻的编码器输出的累计脉冲 :math:`\sum_{t} S_{t}`：

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/x0.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/y00.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/y07.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/x1.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/y10.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/y17.*
    :width: 100%

观察可以发现，编码器的累计输出脉冲 :math:`\sum_{t} S_{t}` 非常接近原图像的轮廓，表明这种自学习的脉冲编码器，有很强的编码能力。