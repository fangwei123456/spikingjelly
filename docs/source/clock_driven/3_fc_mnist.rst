时间驱动：使用单层全连接SNN识别MNIST
====================================

本教程作者：\ `Yanqi-Chen <https://github.com/Yanqi-Chen>`__

本节教程将介绍如何使用编码器与替代梯度方法训练一个最简单的MNIST分类网络。

从头搭建一个简单的SNN网络
-------------------------

在PyTorch中搭建神经网络时，我们可以简单地使用\ ``nn.Sequential``\ 将多个网络层堆叠得到一个前馈网络，输入数据将依序流经各个网络层得到输出。

`MNIST数据集 <http://yann.lecun.com/exdb/mnist/>`__\ 包含若干尺寸为\ :math:`28\times 28`\ 的8位灰度图像，总共有0~9共10个类别。以MNIST的分类为例，一个简单的单层ANN网络如下：

.. code-block:: python
   :emphasize-lines: 4

   net = nn.Sequential(
       nn.Flatten(),
       nn.Linear(28 * 28, 10, bias=False),
       nn.Softmax()
       )

我们也可以用完全类似结构的SNN来进行分类任务。就这个网络而言，只需要先去掉所有的激活函数，再将神经元添加到原来激活函数的位置，这里我们选择的是LIF神经元：

.. code-block:: python
   :emphasize-lines: 4

   net = nn.Sequential(
       nn.Flatten(),
       nn.Linear(28 * 28, 10, bias=False),
       neuron.LIFNode(tau=tau)
       )

其中膜电位衰减常数\ :math:`\tau`\ 需要通过参数\ ``tau``\ 设置。

训练SNN网络
-----------

首先指定好训练参数以及若干其他配置

.. code:: python

   device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
   dataset_dir = input('输入保存MNIST数据集的位置，例如“./”\n input root directory for saving MNIST dataset, e.g., "./": ')
   batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
   learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
   T = int(input('输入仿真时长，例如“100”\n input simulating steps, e.g., "100": '))
   tau = float(input('输入LIF神经元的时间常数tau，例如“100.0”\n input membrane time constant, tau, for LIF neurons, e.g., "100.0": '))
   train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”\n input training epochs, e.g., "100": '))
   log_dir = input('输入保存tensorboard日志文件的位置，例如“./”\n input root directory for saving tensorboard logs, e.g., "./": ')

优化器使用Adam，以及使用泊松编码器，在每次输入图片时进行脉冲编码

.. code-block:: python

   # 使用Adam优化器
   optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
   # 使用泊松编码器
   encoder = encoding.PoissonEncoder()

训练代码的编写需要遵循以下三个要点：

1. 脉冲神经元的输出是二值的，而直接将单次运行的结果用于分类极易受到干扰。因此一般认为脉冲网络的输出是输出层一段时间内的\ **发放频率**\ （或称发放率），发放率的高低表示该类别的响应大小。因此网络需要运行一段时间，即使用\ ``T``\ 个时刻后的\ **平均发放率**\ 作为分类依据。

2. 我们希望的理想结果是除了正确的神经元\ **以最高频率发放**\ ，其他神经元\ **保持静默**\ 。常常采用交叉熵损失或者MSE损失，这里我们使用实际效果更好的MSE损失。

3. 每次网络仿真结束后，需要\ **重置**\ 网络状态

结合以上三点，得到训练循环的代码如下：

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

完整的代码位于\ ``clock_driven.examples.lif_fc_mnist.py``\ ，在代码中我们还使用了Tensorboard来保存训练日志。可以直接在Python命令行运行它：

.. code-block:: python

   >>> import spikingjelly.clock_driven.examples.lif_fc_mnist as lif_fc_mnist
   >>> lif_fc_mnist.main()

需要注意的是，训练这样的SNN，所需显存数量与仿真时长 ``T`` 线性相关，更长的 ``T`` 相当于使用更小的仿真步长，训练更为“精细”，但训练效果不一定更好。\ ``T``
太大时，SNN在时间上展开后会变成一个非常深的网络，这将导致梯度的传递容易衰减或爆炸。

另外由于我们使用了泊松编码器，因此需要较大的 ``T``\ 。

训练结果
--------

取\ ``tau=2.0,T=100,batch_size=128,lr=1e-3``\ ，训练100个Epoch后，将会输出四个npy文件。测试集上的最高正确率为92.5%，通过matplotlib可视化得到的正确率曲线如下

.. image:: ../_static/tutorials/clock_driven/3_fc_mnist/acc.*
    :width: 100%

选取测试集中第一张图片：

.. image:: ../_static/tutorials/clock_driven/3_fc_mnist/input.png

用训好的模型进行分类，得到分类结果

.. code-block:: python

   Firing rate: [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]

通过\ ``visualizing``\ 模块中的函数可视化得到输出层的电压以及脉冲如下图所示

.. image:: ../_static/tutorials/clock_driven/3_fc_mnist/1d_spikes.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/3_fc_mnist/2d_heatmap.*
    :width: 100%

可以看到除了正确类别对应的神经元外，其它神经元均未发放任何脉冲。完整的训练代码可见 `clock_driven/examples/lif_fc_mnist.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/lif_fc_mnist.py>`_ 。