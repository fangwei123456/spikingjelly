时间驱动：使用单层全连接SNN识别MNIST
====================================

本教程作者：\ `Yanqi-Chen <https://github.com/Yanqi-Chen>`__

本节教程将介绍如何使用编码器与替代梯度方法训练一个最简单的MNIST分类网络。

从头搭建一个简单的SNN网络
-------------------------

在PyTorch中搭建神经网络时，我们可以简单地使用\ ``nn.Sequential``\ 将多个网络层堆叠得到一个前馈网络，输入数据将依序流经各个网络层得到输出。

`MNIST数据集 <http://yann.lecun.com/exdb/mnist/>`__\ 包含若干尺寸为\ :math:`28\times 28`\ 的8位灰度图像，总共有0~9共10个类别。以MNIST的分类为例，一个简单的单层ANN网络如下：

.. code-block:: python

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10, bias=False),
        nn.Softmax()
        )

我们也可以用完全类似结构的SNN来进行分类任务。就这个网络而言，只需要先去掉所有的激活函数，再将神经元添加到原来激活函数的位置，这里我们选择的是LIF神经元：

.. code-block:: python

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10, bias=False),
        neuron.LIFNode(tau=tau)
        )

其中膜电位衰减常数\ :math:`\tau`\ 需要通过参数\ ``tau``\ 设置。

训练SNN网络
-----------

首先指定好训练参数如学习率等以及若干其他配置

优化器使用Adam，以及使用泊松编码器，在每次输入图片时进行脉冲编码

.. code-block:: python

    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 使用泊松编码器
    encoder = encoding.PoissonEncoder()

训练代码的编写需要遵循以下三个要点：

1. 脉冲神经元的输出是二值的，而直接将单次运行的结果用于分类极易受到干扰。因此一般认为脉冲网络的输出是输出层一段时间内的\ **发放频率**\ （或称发放率），发放率的高低表示该类别的响应大小。因此网络需要运行一段时间，即使用\ ``T``\ 个时刻后的\ **平均发放率**\ 作为分类依据。

2. 我们希望的理想结果是除了正确的神经元\ **以最高频率发放**\ ，其他神经元\ **保持静默**\ 。常常采用交叉熵损失或者MSE损失，这里我们使用实际效果更好的MSE损失。

3. 每次网络仿真结束后，需要\ **重置**\ 网络状态

结合以上三点，得到训练循环的代码如下：

.. code-block:: python

    net.train()
    print("Epoch {}:".format(epoch))
    print("Training...")
    for img, label in tqdm(train_data_loader):
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

        # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
        train_accuracy = (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()
        
        writer.add_scalar('train_accuracy', train_accuracy, train_times)
        train_accs.append(train_accuracy)

        train_times += 1

完整的代码位于\ ``clock_driven.examples.lif_fc_mnist.py``\ ，在代码中我们还使用了Tensorboard来保存训练日志。可以直接在命令行运行它：

.. code-block:: shell

    $ python <PATH>/lif_fc_mnist.py --help
    usage: lif_fc_mnist.py [-h] [--device DEVICE] [--dataset-dir DATASET_DIR] [--log-dir LOG_DIR] [--model-output-dir MODEL_OUTPUT_DIR] [-b BATCH_SIZE] [-T T] [--lr LR] [--tau TAU] [-N EPOCH]

    spikingjelly LIF MNIST Training

    optional arguments:
    -h, --help            show this help message and exit
    --device DEVICE       运行的设备，例如“cpu”或“cuda:0” Device, e.g., "cpu" or "cuda:0"
    --dataset-dir DATASET_DIR
                            保存MNIST数据集的位置，例如“./” Root directory for saving MNIST dataset, e.g., "./"
    --log-dir LOG_DIR     保存tensorboard日志文件的位置，例如“./” Root directory for saving tensorboard logs, e.g., "./"
    --model-output-dir MODEL_OUTPUT_DIR
                            模型保存路径，例如“./” Model directory for saving, e.g., "./"
    -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Batch 大小，例如“64” Batch size, e.g., "64"
    -T T, --timesteps T   仿真时长，例如“100” Simulating timesteps, e.g., "100"
    --lr LR, --learning-rate LR
                            学习率，例如“1e-3” Learning rate, e.g., "1e-3":
    --tau TAU             LIF神经元的时间常数tau，例如“100.0” Membrane time constant, tau, for LIF neurons, e.g., "100.0"
    -N EPOCH, --epoch EPOCH
                            训练epoch，例如“100” Training epoch, e.g., "100"

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

.. code-block:: shell

   Firing rate: [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]

通过\ ``visualizing``\ 模块中的函数可视化得到输出层的电压以及脉冲如下图所示

.. image:: ../_static/tutorials/clock_driven/3_fc_mnist/1d_spikes.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/3_fc_mnist/2d_heatmap.*
    :width: 100%

可以看到除了正确类别对应的神经元外，其它神经元均未发放任何脉冲。完整的训练代码可见 `clock_driven/examples/lif_fc_mnist.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/lif_fc_mnist.py>`_ 。
