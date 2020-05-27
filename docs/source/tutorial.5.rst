软反向传播 SpikingFlow.softbp
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

本节教程主要关注 ``SpikingFlow.softbp``，介绍软反向传播的概念、可微分SNN神经元的使用方式。

需要注意的是，``SpikingFlow.softbp`` 是一个相对独立的包，与其他的 ``SpikingFlow.*`` 中的神经元、突触等组件不能混用。

软反向传播的灵感，来源于以下两篇文章:

Mentzer F, Agustsson E, Tschannen M, et al. Conditional probability models for deep image \
compression[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4394-4402.

Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for \
training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.


SNN之于RNN
----------
可以将SNN中的神经元看作是一种RNN，它的输入是电压增量（或者是电流，但为了方便，在 ``SpikingFlow.softbp`` 中用电压增量），\
隐藏状态是膜电压，输出是脉冲。这样的SNN神经元是具有马尔可夫性的：当前时刻的输出只与当前时刻的输入、神经元自身的状态有关。

可以用以下描述方程来描述任意的SNN：

.. math::
    H(t) & = f(V(t-1), X(t)) \\
    S(t) & = g(H(t) - V_{threshold}) = \Theta(H(t) - V_{threshold}) \\
    V(t) & = H(t) \cdot (1 - S(t)) + V_{reset} \cdot S(t)

其中 :math:`V(t)` 是神经元的膜电压；:math:`X(t)` 是外源输入，例如电压增量；:math:`H(t)` 是神经元的隐藏状态，可以理解为\
神经元还没有发放脉冲前的瞬时电压；:math:`f(V(t-1), X(t))` 是神经元的状态更新方程，不同的神经元，区别就在于更新方程不同。

例如对于LIF神经元，状态更新方程，及其离散化的方程如下：

.. math::
    \tau_{m} \frac{\mathrm{d}V(t)}{\mathrm{d}t} = -(V(t) - V_{reset}) + X(t)

    \tau_{m} (V(t) - V(t-1)) = -(V(t-1) - V_{reset}) + X(t)

由于状态更新方程不能描述脉冲发放的过程，因此我们用 :math:`H(t)` 来代替 :math:`V(t)`，用 :math:`V(t)` 表示完成脉冲发放（或者\
不发放）过程后的神经元膜电压。

:math:`S(t)` 是神经元发放的脉冲，:math:`g(x)=\Theta(x)` 是阶跃函数，或者按RNN的习惯称为门控函数，输出仅为0或1，可以表示脉冲\
的发放过程，定义为

.. math::
    \Theta(x) =
    \begin{cases}
    1, & x \geq 0 \\
    0, & x < 0
    \end{cases}

发放脉冲，则电压重置为 :math:`V_{reset}`；没有发放脉冲，则电压不变，这就是描述方程的最后一个方程，电压状态转移方程。

硬前向与软反向
-------------
RNN使用可微分的门控函数，例如tanh函数。而SNN的门控函数 :math:`g(x)=\Theta(x)` 显然是不可微分的，这就导致了SNN虽然一定程度上\
与RNN非常相似，但不能用梯度下降、反向传播来训练。但我们可以用一个形状与 :math:`g(x)=\Theta(x)` 非常相似，但可微分的门控函数\
:math:`\sigma(x)` 去替换它。

我们这一方法的核心思想是：在前向传播时，使用 :math:`g(x)=\Theta(x)`，神经元的输出是离散的0和1，我们的网络仍然是SNN；而反向\
传播时，使用近似门控函数 :math:`g'(x)=\sigma'(x)` 来求梯度。最常见的近似 :math:`g(x)=\Theta(x)` 的门控函数\
即为sigmoid函数 :math:`\sigma(\alpha x)=\frac{1}{1 + exp(-\alpha x)}`，:math:`\alpha` 可以控制函数的平滑程\
度，越大的 :math:`\alpha` 会越逼近 :math:`\Theta(x)` 但梯度越不光滑，网络也会越难以训练。近似门控函数引入后，电压状态转移\
函数 :math:`V(t) = H(t) \cdot (1 - S(t)) + V_{reset} \cdot S(t)` 也会随之改变。下图显示了不同的 :math:`\alpha` 以及电压\
状态转移方程：

.. image:: ./_static/tutorials/5-1.png

默认的近似门控函数为 ``SpikingFlow.softbp.soft_pulse_function.Sigmoid()``。近似门控函数是 ``softbp`` 包中基类神经元构造函数\
的参数之一：

.. code-block:: python

    class BaseNode(nn.Module):
        def __init__(self, v_threshold=1.0, v_reset=0.0, pulse_soft=soft_pulse_function.Sigmoid()):
            '''
            :param v_threshold: 神经元的阈值电压
            :param v_reset: 神经元的重置电压
            :param pulse_soft: 反向传播时用来计算脉冲函数梯度的替代函数，即软脉冲函数
            '''
             super().__init__()
            self.v_threshold = v_threshold
            self.v_reset = v_reset
            self.v = v_reset
            self.pulse_soft = pulse_soft

在 ``SpikingFlow.softbp.soft_pulse_function`` 中还提供了其他的可选近似门控函数。

如果想要自定义新的近似门控函数，可以参考 ``soft_pulse_function.Sigmoid()`` 的代码实现。

硬前向传播与软反向传播，在PyTorch中很容易实现，参考 ``SpikingFlow.softbp.neuron.BaseNode`` 中的 ``spiking``：

.. code-block:: python

        def spiking(self):
            if self.training:
                spike_hard = (self.v >= self.v_threshold).float()
                spike_soft = self.pulse_soft(self.v - self.v_threshold)
                v_hard = self.v_reset * spike_hard + self.v * (1 - spike_hard)
                v_soft = self.v_reset * spike_soft + self.v * (1 - spike_soft)
                self.v = v_soft + (v_hard - v_soft).detach_()
                return spike_soft + (spike_hard - spike_soft).detach_()
            else:
                spike_hard = (self.v >= self.v_threshold).float()
                self.v = self.v_reset * spike_hard + self.v * (1 - spike_hard)
                return spike_hard

前向传播时，该函数返回 ``spike_soft + spike_hard - spike_soft`` 即 ``spike_hard``，但计算图却是按照函数返回 ``spike_soft``\
建立的，因为 ``(spike_hard - spike_soft).detach_()`` 使得 ``spike_hard - spike_soft`` 被从计算图中剔除，因此反向传播时按\
照前向传播为 ``spike_soft`` 来计算梯度。

作为激活函数的SNN神经元
----------------------
解决了SNN的微分问题后，我们的SNN神经元可以像激活函数那样，嵌入到使用PyTorch搭建的任意网络中去了。在 ``SpikingFlow.softbp.neuron`` 中\
已经实现了IF神经元和LIF神经元，可以很方便地搭建各种网络，例如一个简单的全连接网络：\

.. code-block:: python

    net = nn.Sequential(
            nn.Linear(100, 10, bias=False),
            neuron.LIFNode(tau=100.0, v_threshold=1.0, v_reset=5.0)
            )

MNIST分类
--------
现在我们使用 ``SpikingFlow.softbp.neuron`` 中的LIF神经元，搭建一个双层全连接网络，对MNIST数据集进行分类：

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import sys
    sys.path.append('.')
    import SpikingFlow.softbp.neuron as neuron
    import SpikingFlow.encoding as encoding
    from torch.utils.tensorboard import SummaryWriter
    import readline

    class Net(nn.Module):
        def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0):
            super().__init__()
            # 网络结构，简单的双层全连接网络，每一层之后都是LIF神经元
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 14 * 14, bias=False),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
                nn.Linear(14 * 14, 10, bias=False),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
            )

        def forward(self, x):
            return self.fc(x)

        def reset_(self):
            for item in self.modules():
                if hasattr(item, 'reset'):
                    item.reset()
    def main():
        device = input('输入运行的设备，例如“CPU”或“cuda:0”  ')
        dataset_dir = input('输入保存MNIST数据集的位置，例如“./”  ')
        batch_size = int(input('输入batch_size，例如“64”  '))
        learning_rate = float(input('输入学习率，例如“1e-3”  '))
        T = int(input('输入仿真时长，例如“50”  '))
        tau = float(input('输入LIF神经元的时间常数tau，例如“100.0”  '))
        train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”  '))
        log_dir = input('输入保存tensorboard日志文件的位置，例如“./”  ')

        writer = SummaryWriter(log_dir)

        # 初始化数据加载器
        train_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False)

        # 初始化网络
        net = Net(tau=tau).to(device)
        # 使用Adam优化器
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        # 使用泊松编码器
        encoder = encoding.PoissonEncoder()
        train_times = 0
        for _ in range(train_epoch):
            net.train()
            for img, label in train_data_loader:
                img = img.to(device)
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

                # 损失函数为输出层神经元的脉冲发放频率，与真实类别的交叉熵
                # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0
                loss = F.cross_entropy(out_spikes_counter_frequency, label.to(device))
                loss.backward()
                optimizer.step()
                # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
                net.reset_()

                # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
                correct_rate = (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()
                writer.add_scalar('train_correct_rate', correct_rate, train_times)
                if train_times % 1024 == 0:
                    print(device, dataset_dir, batch_size, learning_rate, T, tau, train_epoch, log_dir)
                    print('train_times', train_times, 'train_correct_rate', correct_rate)
                train_times += 1

            net.eval()
            with torch.no_grad():
                # 每遍历一次全部数据集，就在测试集上测试一次
                test_sum = 0
                correct_sum = 0
                for img, label in test_data_loader:
                    img = img.to(device)
                    for t in range(T):
                        if t == 0:
                            out_spikes_counter = net(encoder(img).float())
                        else:
                            out_spikes_counter += net(encoder(img).float())

                    correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
                    test_sum += label.numel()
                    net.reset_()

                writer.add_scalar('test_correct_rate', correct_sum / test_sum, train_times)

    if __name__ == '__main__':
        main()

这份代码位于 ``SpikingFlow.softbp.examples.mnist.py``。进入 ``SpikingFlow`` 的根目录（也就是GitHub仓库的根目录），直接运行\
即可，例如：

.. code-block:: bash

    (pytorch-env) wfang@pami:~/SpikingFlow$ python SpikingFlow/softbp/examples/mnist.py
    输入运行的设备，例如“CPU”或“cuda:0”  cuda:0
    输入保存MNIST数据集的位置，例如“./”  ./tempdir
    输入batch_size，例如“64”  256
    输入学习率，例如“1e-3”  1e-2
    输入仿真时长，例如“50”  50
    输入LIF神经元的时间常数tau，例如“100.0”  100.0
    输入训练轮数，即遍历训练集的次数，例如“100”  1000
    输入保存tensorboard日志文件的位置，例如“./”  ./tempdir

需要注意的是，训练这样的SNN，所需显存数量与仿真时长 ``T`` 线性相关，更长的 ``T`` 相当于使用更小的仿真步长，\
训练更为“精细”，训练效果也一般更好。这个模型只占用了 ``276MB`` 显存，但在之后的CIFAR10示例中，由于CNN的引入，使得显存消耗量\
剧增。

我们的这个模型，在Tesla K80上训练一个半小时，tensorboard记录的数据如下所示：

.. image:: ./_static/tutorials/5-2.png

这个模型最终能够达到98%的测试集正确率，如下图所示，注意下图中的“epoch”表示训练次数，而代码中的“epoch”表示遍历一次训练集：

.. image:: ./_static/tutorials/5-3.png

如果使用训练集增强的方法，例如给训练集图片加上一些随机噪声、仿射变换等，则训练好的网络泛化能力会进一步提升，最高能达到99%以上\
的测试集正确率。

CIFAR10分类
----------
我们的这种方法，具有的一大优势就是可以无缝嵌入到任意的PyTorch搭建的网络中。因此CNN的引入是非常简单而自然的。我们用CNN来进行\
CIFAR10分类任务，训练的代码与进行MNIST分类几乎相同，只需要更改一下网络结构和数据集。

.. code-block:: python

    class Net(nn.Module):
        def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0):
            super().__init__()
            # 网络结构，卷积-卷积-最大池化堆叠，最后接一个全连接层
            self.conv = nn.Sequential(
                nn.Conv2d(3, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(256),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),  # 16 * 16

                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(256),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),  # 8 * 8

                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(256),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),  # 4 * 4

            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 10, bias=False),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
                                    )

        def forward(self, x):
            return self.fc(self.conv(x))

        def reset_(self):
            for item in self.modules():
                if hasattr(item, 'reset'):
                    item.reset()
    def main():
        device = input('输入运行的设备，例如“CPU”或“cuda:0”  ')
        dataset_dir = input('输入保存CIFAR10数据集的位置，例如“./”  ')
        batch_size = int(input('输入batch_size，例如“64”  '))
        learning_rate = float(input('输入学习率，例如“1e-3”  '))
        T = int(input('输入仿真时长，例如“50”  '))
        tau = float(input('输入LIF神经元的时间常数tau，例如“100.0”  '))
        train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”  '))
        log_dir = input('输入保存tensorboard日志文件的位置，例如“./”  ')

        writer = SummaryWriter(log_dir)

        # 初始化数据加载器
        train_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                root=dataset_dir,
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                root=dataset_dir,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False)
         # 后面的代码与MNIST分类相同，不再展示


这份代码位于 ``SpikingFlow.softbp.examples.cifar10.py``，运行方法与之前的MNIST的代码相同。需要注意的是，由于CNN的引入，CNN层\
后也跟有LIF神经元，CNN层的输出是一个高维矩阵，因此其后的LIF神经元数量众多，导致这个模型极端消耗显存。在大约 ``batch_size=32``\
，仿真时长 ``T=50`` 的情况下，这个模型几乎要消耗 ``12G`` 的显存。训练这样庞大模型，Tesla K80的算力显得捉襟见肘。我们在TITAN RTX\
上训练大约60小时，网络才能收敛，测试集正确率大约为80%。使用训练集增强的方法，同样可以提高泛化能力。

.. image:: ./_static/tutorials/5-4.png

模型流水线
----------
如前所述，在包含SNN神经元的网络中引入CNN后，显存的消耗量剧增。有时一个网络太大，以至于单个GPU无法放下。在这种情况下，我们可以\
将一个网络分割到多个GPU存放，充分利用多GPU闲置显存的优势。但使用这一方法，数据需要在多个GPU之间来回复制，在一定程度上会降低\
训练速度。

``SpikingFlow.softbp.ModelPipeline`` 是一个基于流水线多GPU串行并行的基类，使用者只需要继承 ``ModelPipeline``，然后调\
用 ``append(nn_module, gpu_id)``，就可以将 ``nn_module`` 添加到流水线中，并且 ``nn_module`` 会被运行在 ``gpu_id`` 上。\
在调用模型进行计算时， ``forward(x, split_sizes)`` 中的 ``split_sizes`` 指的是输入数据 ``x`` 会在维度0上被拆分成\
每 ``split_size`` 一组，得到 ``[x[0], x[1], ...]``，这些数据会被串行的送入 ``module_list`` 中保存的各个模块进行计算。

例如将模型分成4部分，因而 ``module_list`` 中有4个子模型；将输入分割为3部分，则每次调用 ``forward(x, split_sizes)`` ，函数内部的\
计算过程如下：

.. code-block:: python

        step=0     x0, x1, x2  |m0|    |m1|    |m2|    |m3|

        step=1     x0, x1      |m0| x2 |m1|    |m2|    |m3|

        step=2     x0          |m0| x1 |m1| x2 |m2|    |m3|

        step=3                 |m0| x0 |m1| x1 |m2| x2 |m3|

        step=4                 |m0|    |m1| x0 |m2| x1 |m3| x2

        step=5                 |m0|    |m1|    |m2| x0 |m3| x1, x2

        step=6                 |m0|    |m1|    |m2|    |m3| x0, x1, x2

不使用流水线，则任何时刻只有一个GPU在运行，而其他GPU则在等待这个GPU的数据；而使用流水线，例如上面计算过程中的 ``step=3`` 到\
``step=4``，尽管在代码的写法为顺序执行：

.. code-block:: python

    x0 = m1(x0)
    x1 = m2(x1)
    x2 = m3(x2)

但由于PyTorch优秀的特性，上面的3行代码实际上是并行执行的，因为这3个在CUDA上的计算使用各自的数据，互不影响。

我们将之前的CIFAR10代码更改为多GPU流水线形式，修改后的代码位于 ``SpikingFlow.softbp.examples.cifar10.py``。它的内容\
与 ``SpikingFlow.softbp.examples.cifar10.py`` 基本类似，我们只看主要的改动部分。

模型的定义，直接继承了 ``ModelPipeline``。将模型拆成了5个部分，由于越靠前的层，输入的尺寸越大，越消耗显存，因此前面的少部分层\
会直接被单独分割出，而后面的很多层则放到了一起。需要注意的是，每次训练后仍然要重置LIF神经元的电压，因此要额外写一个重置函
\数 ``reset_()``：

.. code-block:: python

    class Net(softbp.ModelPipeline):
        def __init__(self, gpu_list, tau=100.0, v_threshold=1.0, v_reset=0.0):
            super().__init__()
            # 网络结构，卷积-卷积-最大池化堆叠，最后接一个全连接层

            self.append(
                nn.Sequential(
                    nn.Conv2d(3, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256)
                ),
                gpu_list[0]
            )

            self.append(
                nn.Sequential(
                    neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
                ),
                gpu_list[1]
            )

            self.append(
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.MaxPool2d(2, 2),
                    nn.BatchNorm2d(256)
                ),
                gpu_list[2]
            )

            self.append(
                nn.Sequential(
                    neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)  # 16 * 16
                ),
                gpu_list[3]
            )

            self.append(
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.MaxPool2d(2, 2),
                    nn.BatchNorm2d(256),
                    neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),  # 8 * 8
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.MaxPool2d(2, 2),
                    nn.BatchNorm2d(256),
                    neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),  # 4 * 4
                    nn.Flatten(),
                    nn.Linear(256 * 4 * 4, 10, bias=False),
                    neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
                ),
                gpu_list[4]
            )

        def reset_(self):
            for item in self.modules():
                if hasattr(item, 'reset'):
                    item.reset()

运行这份代码，由于分割的第0部分和第3部分占用的显存较小，因此将它们全部放在 ``0`` 号GPU上，而其他部分则各独占一个GPU：

.. code-block:: bash

    (pytorch-env) wfang@pami:~/SpikingFlow$ python ./SpikingFlow/softbp/examples/cifar10mp.py
    输入使用的5个gpu，例如“0,1,2,0,3”  0,1,2,0,3
    输入保存CIFAR10数据集的位置，例如“./”  ./tempdir
    输入batch_size，例如“64”  64
    输入split_sizes，例如“16”  4
    输入学习率，例如“1e-3”  1e-3
    输入仿真时长，例如“50”  50
    输入LIF神经元的时间常数tau，例如“100.0”  100.0
    输入训练轮数，即遍历训练集的次数，例如“100”  100
    输入保存tensorboard日志文件的位置，例如“./”  ./tempdir

稳定运行后，查看各个GPU显存的占用：

.. code-block:: bash

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0      4465      C   python                                      5950MiB |
    |    1      4465      C   python                                      9849MiB |
    |    2      4465      C   python                                      9138MiB |
    |    3      4465      C   python                                      8936MiB |
    +-----------------------------------------------------------------------------+

对于模型的不同分割方法会造成不同的显存占用情况。建议首先做一个简单的分割，然后用很小的 ``batch_size`` 和 ``split_sizes`` 去\
运行，再检查各个GPU显存的负载是否均衡，根据负载情况来重新调整分割。

分割后的模型， ``batch_size=64, split_size=4``，根据tensorboard的记录显示，在Tesla K80上30分钟训练了116次；使用其他相同\
的参数，令 ``batch_size=64, split_size=2``，30分钟训练了62次；令 ``batch_size=64, split_size=32``，30分钟训练了272次；\
令 ``batch_size=64, split_size=16``，30分钟训练了230次；令 ``batch_size=32, split_size=8``，30分钟训练335\
次；令 ``batch_size=32, split_size=16``，30分钟训练460次；令 ``batch_size=32, split_size=32``，30分钟训练466次；不使用\
模型流水线、完全在同一个GPU上运行的 ``SpikingFlow.softbp.examples.cifar10.py``， ``batch_size=16``，30分钟训练759次。对\
比如下表所示:


+---------------+------------+------------+---------------------+
| *.py          | batch_size | split_size | images/minute       |
+===============+============+============+=====================+
|               | 64         | 32         |        580          |
|               |------------+------------+---------------------+
|               | 64         | 16         |        490          |
|               |------------+------------+---------------------+
|               | 64         | 4          |        247          |
|               |------------+------------+---------------------+
| cifar10mp.py  | 64         | 2          |        132          |
|               |------------+------------+---------------------+
|               | 32         | 8          |        357          |
|               |------------+------------+---------------------+
|               | 32         | 16         |        490          |
|               |------------+------------+---------------------+
|               | 32         | 32         |        497          |
+---------------+------------+------------+---------------------+
| cifar10.py    | 16         | \\         |        404          |
+---------------+------------+------------+---------------------+

可以发现，参数的选择对于训练速度至关重要。合适的参数，例如 ``batch_size=64, split_size=32``，训练速度已经明显超过单卡运行了。

持续恒定输入的更快方法
---------------------
上文的代码中，我们使用泊松编码器 ``encoding.PoissonEncoder()``，它以概率来生成脉冲，因此在不同时刻，这一编码器的输出是不同的。\
如果我们使用恒定输入，则每次前向传播时，不需要再重新启动一次流水线，而是可以启动一次流水线并一直运行，达到更快的速度、更小的\
显存占用。对于持续 ``T`` 的恒定输入 ``x``，可以直接调用 ``ModelPipeline.constant_forward(self, x, T, reduce)`` 进行计算。

我们将之前的代码进行更改，去掉编码器部分，将图像数据直接送入SNN。在这种情况下，我们可以认为SNN的第一个卷积层起到了“编码器”的作\
用：它接收图像作为输入，并输出脉冲。这种能够参与训练的编码器，通常能够起到比泊松编码器更好的编码效果，最终网络的分类性能也会有\
一定的提升。更改后的代码位于 ``SpikingFlow.softbp.examples.cifar10cmp.py``，代码的更改非常简单，主要体现在：

.. code-block:: python

    class Net(softbp.ModelPipeline):
        ...
        # 使用父类的constant_forward来覆盖父类的forward
        def forward(self, x, T):
            return self.constant_forward(x, T, True)
        ...

    def main():
        ...
        # 直接将图像送入网络，不需要编码器
        out_spikes_counter_frequency = net(img, T) / T
        ...


设置 ``batch_size=32``，模型在显卡上的分布与之前相同，30分钟训练715次；\
去掉编码器但不使用 ``ModelPipeline.constant_forward(self, x, T, reduce)``， ``batch_size=64, split_size=32``，30分钟\
训练276次。可以发现，去掉编码器后网络的训练速度会变慢；使用这一方法能够起到一倍以上的加速。