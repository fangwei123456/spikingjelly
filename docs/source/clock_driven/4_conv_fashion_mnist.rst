时间驱动：使用卷积SNN识别Fashion-MNIST
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

在本节教程中，我们将搭建一个卷积脉冲神经网络，对 `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ 数据集进行
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
    import readline
    class Net(nn.Module):
        def __init__(self, tau, v_threshold=1.0, v_reset=0.0):

接下来，我们在 ``Net`` 的成员变量中添加卷积层和全连接层。``SpikingJelly`` 的开发者们在实验中发现，对于不含时间信息、静态的图片数据，
卷积层中的神经元用 ``IFNode`` 效果更好一些。我们添加2个卷积-BN-池化层：

.. code-block:: python

    self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 7 * 7
        )

``1 * 28 * 28`` 的输入经过这样的卷积层作用后，得到 ``128 * 7 * 7`` 的输出脉冲。

这样的卷积层，其实可以起到编码器的作用：在上一届教程，MNIST识别的代码中，我们使用泊松编码器，将图片编码成脉冲。实际上我们完全可以直接将
图片送入SNN，在这种情况下，SNN中的首层脉冲神经元层及其之前的层，可以看作是一个参数可学习的自编码器。例如我们刚才定义的卷积层中的这些层：

.. code-block:: python

    nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan())

这3层网络，接收图片作为输入，输出脉冲，可以看作是编码器。

接下来，我们定义3层全连接网络，输出分类的结果。全连接层一般起到分类器的作用，使用 ``LIFNode`` 性能会更好。Fashion-MNIST共有10类，因
此输出层是10个神经元；为了减少过拟合，我们还使用了 ``layer.Dropout``，关于它的更多信息可以参阅API文档。

.. code-block:: python

    self.fc = nn.Sequential(
        nn.Flatten(),
        layer.Dropout(0.7),
        nn.Linear(128 * 7 * 7, 128 * 3 * 3, bias=False),
        neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        layer.Dropout(0.7),
        nn.Linear(128 * 3 * 3, 128, bias=False),
        neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        nn.Linear(128, 10, bias=False),
        neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
    )

接下来，定义前向传播。前向传播非常简单，先经过卷积，在经过全连接即可：

.. code-block:: python

    def forward(self, x):
        return self.fc(self.conv(x))

避免重复计算
-------------------

我们可以直接训练这个网络，就像之前的MNIST分类那样：

.. code-block:: python

        for img, label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()

            optimizer.zero_grad()

            # 运行T个时长，out_spikes_counter是shape=[batch_size, 10]的tensor
            # 记录整个仿真时长内，输出层的10个神经元的脉冲发放次数
            for t in range(T):
                if t == 0:
                    out_spikes_counter = net(encoder(img).float())
                else:
                    out_spikes_counter += net(encoder(img).float())

            # out_spikes_counter / T 得到输出层10个神经元在仿真时长内的脉冲发放频率
            out_spikes_counter_frequency = out_spikes_counter / T

            # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
            # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            loss.backward()
            optimizer.step()
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            functional.reset_net(net)

但我们如果重新审视网络的结构，可以发现，有一些计算是重复的：对于网络的前2层，即下面代码中的高亮部分：

.. code-block:: python
    :emphasize-lines: 2, 3

    self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 7 * 7
        )

这2层接收的输入图片，并不随 ``t`` 变化，但在 ``for`` 循环中，每次 ``img`` 都会重新经过这2层的计算，得到相同的输出。我们提取出这些层，
同时将时间上的循环封装进网络本身，方便计算。新的网络结构完整定义为：

.. code-block:: python

    class Net(nn.Module):
        def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
            super().__init__()
            self.T = T

            self.static_conv = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
            )

            self.conv = nn.Sequential(
                neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
                nn.MaxPool2d(2, 2),  # 14 * 14

                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
                nn.MaxPool2d(2, 2)  # 7 * 7

            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                layer.Dropout(0.7),
                nn.Linear(128 * 7 * 7, 128 * 3 * 3, bias=False),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
                layer.Dropout(0.7),
                nn.Linear(128 * 3 * 3, 128, bias=False),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
                nn.Linear(128, 10, bias=False),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
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
完整的代码位于 `clock_driven/examples/conv_fashion_mnist.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/conv_fashion_mnist.py>`_。
也可以通过命令行直接运行。会将训练过程中测试集正确率最高的网络保存在 ``tensorboard`` 日志文件的同级目录下。

.. code-block:: python

    >>> from spikingjelly.clock_driven.examples import conv_fashion_mnist
    >>> conv_fashion_mnist.main()
    输入运行的设备，例如“cpu”或“cuda:0”
     input device, e.g., "cpu" or "cuda:0": cuda:9
    输入保存Fashion MNIST数据集的位置，例如“./”
     input root directory for saving Fashion MNIST dataset, e.g., "./": ./fmnist
    输入batch_size，例如“64”
     input batch_size, e.g., "64": 64
    输入学习率，例如“1e-3”
     input learning rate, e.g., "1e-3": 1e-3
    输入仿真时长，例如“8”
     input simulating steps, e.g., "8": 8
    输入LIF神经元的时间常数tau，例如“2.0”
     input membrane time constant, tau, for LIF neurons, e.g., "2.0": 2.0
    输入训练轮数，即遍历训练集的次数，例如“100”
     input training epochs, e.g., "100": 100
    输入保存tensorboard日志文件的位置，例如“./”
     input root directory for saving tensorboard logs, e.g., "./": ./logs_conv_fashion_mnist

运行100轮训练后，训练batch和测试集上的正确率如下：

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/train.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/test.*
    :width: 100%

在训练100个epoch后，最高测试集正确率可以达到94.3%，对于SNN而言是非常不错的性能，仅仅略低于 `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_
的BenchMark中使用Normalization, random horizontal flip, random vertical flip, random translation, random rotation的ResNet18的94.9%正确率。

可视化编码器
------------------------------------

正如我们在前文中所述，直接将数据送入SNN，则首个脉冲神经元层及其之前的层，可以看作是一个可学习的编码器。具体而言，是我们的网络中如
下所示的高亮部分：

.. code-block:: python
    :emphasize-lines: 5, 6, 10

    class Net(nn.Module):
        def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
            ...
            self.static_conv = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
            )

            self.conv = nn.Sequential(
                neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            ...

现在让我们来查看一下，训练好的编码器，编码效果如何。让我们新建一个python文件，导入相关的模块，并重新定义一个 ``batch_size=1`` 的数据加载器，因为我们想要一
张图片一张图片的查看：

.. code-block:: python

    from matplotlib import pyplot as plt
    import numpy as np
    from spikingjelly.clock_driven.examples.conv_fashion_mnist import Net
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

    net = torch.load('./logs_conv_fashion_mnist/net_max_acc.pt', 'cpu')
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

观察可以发现，编码器的累计输出脉冲 :math:`\sum_{t} S_{t}` 非常接近原图像的轮廓，表面这种自学习的脉冲编码器，有很强的编码能力。