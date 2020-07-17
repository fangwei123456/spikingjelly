Clock_driven SpikingFlow.clock_driven
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_
        `lucifer2859 <https://github.com/lucifer2859>`_

This tutorial focuses on ``SpikingFlow.clock_driven``, introduce the clock-driven simulation method, the concept of gradient substitution method, and the use of differentiable SNN neurons.

Surrogate gradient method is a new method emerging in recent years. For more information about this method, please refer to the following overview:

Neftci E, Mostafa H, Zenke F, et al. Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-based optimization to spiking neural networks[J]. IEEE Signal Processing Magazine, 2019, 36(6): 51-63.

The download address for this article can be found at `arXiv <https://arxiv.org/abs/1901.09948>`_ .

SNN compared with RNN
----------
The neuron in SNN can be regarded as a kind of RNN, and its input is the voltage increment (or the product of current and membrane resistance, but for convenience, ``clock_driven.neuron`` uses voltage increment). The hidden state is the membrane voltage, and the output is a pulse. Such SNN neurons are Markovian: the output at the current time is only related to the input at the current time and the state of the neuron itself.

You can use three discrete equations —— charge, discharge, reset —— to describe any discrete pulse neuron:


.. math::
    H(t) & = f(V(t-1), X(t)) \\
    S(t) & = g(H(t) - V_{threshold}) = \Theta(H(t) - V_{threshold}) \\
    V(t) & = H(t) \cdot (1 - S(t)) + V_{reset} \cdot S(t)

where :math:`V(t)` is the membrane voltage of the neuron; :math:`X(t)` is an external source input, such as voltage increment; :math:`H(t)` is the hidden state of the neuron, which can be understood as the instant before the neuron has not fired a pulse; :math:`f(V(t-1), X(t))` is the state update equation of the neuron. Different neurons differ in the update equation.

For example, for a LIF neuron, describing the differential equation of its dynamics below a threshold, and the corresponding difference equation are:

.. math::
    \tau_{m} \frac{\mathrm{d}V(t)}{\mathrm{d}t} = -(V(t) - V_{reset}) + X(t)

    \tau_{m} (V(t) - V(t-1)) = -(V(t-1) - V_{reset}) + X(t)

The corresponding charging equation is

.. math::
    f(V(t - 1), X(t)) = V(t - 1) + \frac{1}{\tau_{m}}(-(V(t - 1) - V_{reset}) + X(t))


In the discharge equation, :math:`S(t)` is a pulse fired from a neuron, :math:`g(x)=\Theta(x)` is a step function. RNN is used to call it a gating function. In SNN, it is called a pulse function. The output of the pulse function is only 0 or 1, which can represent the firing process of pulse, defined as

.. math::
    \Theta(x) =
    \begin{cases}
    1, & x \geq 0 \\
    0, & x < 0
    \end{cases}

Reset means the voltage reset process: when a pulse is fired, the voltage is reset to :math:`V_{reset}`; If no pulse is fired, the voltage remains unchanged.

Surrogate Gradient Method
-------------
RNN uses differentiable gating functions, such as the tanh function. Obviously, the pulse function of SNN :math:`g(x)=\Theta(x)` is not differentiable, which leads to the fact that SNN is very similar to RNN to a certain extent, but it cannot be trained by gradient descent and back-propagation. We can use a gating function that is very similar to :math:`g(x)=\Theta(x)` , but differentiable :math:`\sigma(x)` to replace it.

The core idea of ​​this method is: when forwarding, using :math:`g(x)=\Theta(x)`, the output of the neuron is discrete 0 and 1, and our network is still SNN; When back-propagation, the gradient of the surrogate gradient function :math:`g'(x)=\sigma'(x)` is used to replace the gradient of the pulse function. The most common gradient substitution function is the sigmoid function :math:`\sigma(\alpha x)=\frac{1}{1 + exp(-\alpha x)}`，:math:`\alpha` can control the smoothness of the function, the fuction with larger :math:`\alpha` will be closer to :math:`\Theta(x)`, but when it gets closer to :math:`x=0`, the gradient will be more likely to explode, and when it gets farther to :math:`x=0`, the gradient will be more likely to disappear. This makes the network more difficult to train. The following figure shows the shape of the gradient substitution function and the shape of the corresponding reset equation for different :math:`\alpha`:

.. image:: ./_static/tutorials/clock_driven/1.png

默认的梯度替代函数为 ``clock_driven.surrogate.Sigmoid()``，在 ``clock_driven.surrogate`` 中还提供了其他的可选近似门控函数。
梯度替代函数是 ``clock_driven.neuron`` 中神经元构造函数的参数之一：

.. code-block:: python

    class BaseNode(nn.Module):
        def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), monitor_state=False):
            '''
            :param v_threshold: 神经元的阈值电压
            :param v_reset: 神经元的重置电压。如果不为None，当神经元释放脉冲后，电压会被重置为v_reset；如果设置为None，则电压会被减去阈值
            :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
            :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。
                            若为True，则self.monitor是一个字典，键包括'v'和's'，分别记录电压和输出脉冲。对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量
                            转换为numpy数组后的值。还需要注意，self.reset()函数会清空这些链表

如果想要自定义新的近似门控函数，可以参考 ``clock_driven.surrogate`` 中的代码实现。通常是定义 ``torch.autograd.Function``，然后\
将其封装成一个 ``torch.nn.Module`` 的子类。

将脉冲神经元嵌入到深度网络
------------------------
解决了脉冲神经元的微分问题后，我们的脉冲神经元可以像激活函数那样，嵌入到使用PyTorch搭建的任意网络中，使得网络成为一个SNN。在 ``clock_driven.neuron`` 中\
已经实现了一些经典神经元，可以很方便地搭建各种网络，例如一个简单的全连接网络：\

.. code-block:: python

    net = nn.Sequential(
            nn.Linear(100, 10, bias=False),
            neuron.LIFNode(tau=100.0, v_threshold=1.0, v_reset=5.0)
            )

使用双层全连接网络进行MNIST分类
-----------------------------
现在我们使用 ``clock_driven.neuron`` 中的LIF神经元，搭建一个双层全连接网络，对MNIST数据集进行分类。

首先定义我们的网络结构：

.. code-block:: python

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

定义我们的超参数：

.. code-block:: python

    device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    dataset_dir = input('输入保存MNIST数据集的位置，例如“./”\n input root directory for saving MNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“100”\n input simulating steps, e.g., "100": '))
    tau = float(input('输入LIF神经元的时间常数tau，例如“100.0”\n input membrane time constant, tau, for LIF neurons, e.g., "100.0": '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”\n input training epochs, e.g., "100": '))
    log_dir = input('输入保存tensorboard日志文件的位置，例如“./”\n input root directory for saving tensorboard logs, e.g., "./": ')

初始化数据加载器、网络、优化器，以及编码器（我们使用泊松编码器，将MNIST图像编码成脉冲序列）：

.. code-block:: python

    # 初始化网络
    net = Net(tau=tau).to(device)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # 使用泊松编码器
    encoder = encoding.PoissonEncoder()

网络的训练很简单。将网络运行 ``T`` 个时间步长，对输出层10个神经元的输出脉冲进行累加，得到输出层脉冲释放次数 ``out_spikes_counter``；\
使用脉冲释放次数除以仿真时长，得到输出层脉冲发放频率 ``out_spikes_counter_frequency = out_spikes_counter / T``。我们希望当输入\
图像的实际类别是 ``i`` 时，输出层中第 ``i`` 个神经元有最大的激活程度，而其他神经元都保持沉默。因此损失函数自然定义为输出层脉冲\
发放频率 ``out_spikes_counter_frequency`` 与实际类别进行one hot编码后得到的 ``label_one_hot`` 的交叉熵，或MSE。我们使用MSE，\
因为实验发现MSE的效果更好一些。尤其需要注意的是，SNN是有状态，或者说有记忆的网络，因此在输入新数据前，一定要将网络的状态重置，\
这可以通过调用 ``clock_driven.functional.reset_net(net)`` 来实现。训练的代码如下：

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

测试的代码与训练代码相比更为简单：

.. code-block:: python

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
            functional.reset_net(net)

        writer.add_scalar('test_accuracy', correct_sum / test_sum, epoch)

完整的代码位于 ``clock_driven.examples.lif_fc_mnist.py``，在代码中我们还使用了Tensorboard来保存训练日志。可以直接在Python命令行运行它：

    .. code-block:: python

        >>> import SpikingFlow.clock_driven.examples.lif_fc_mnist as lif_fc_mnist
        >>> lif_fc_mnist.main()
        输入运行的设备，例如“cpu”或“cuda:0”
         input device, e.g., "cpu" or "cuda:0": cuda:15
        输入保存MNIST数据集的位置，例如“./”
         input root directory for saving MNIST dataset, e.g., "./": ./mnist
        输入batch_size，例如“64”
         input batch_size, e.g., "64": 128
        输入学习率，例如“1e-3”
         input learning rate, e.g., "1e-3": 1e-3
        输入仿真时长，例如“100”
         input simulating steps, e.g., "100": 50
        输入LIF神经元的时间常数tau，例如“100.0”
         input membrane time constant, tau, for LIF neurons, e.g., "100.0": 100.0
        输入训练轮数，即遍历训练集的次数，例如“100”
         input training epochs, e.g., "100": 100
        输入保存tensorboard日志文件的位置，例如“./”
         input root directory for saving tensorboard logs, e.g., "./": ./logs_lif_fc_mnist
        cuda:15 ./mnist 128 0.001 50 100.0 100 ./logs_lif_fc_mnist
        train_times 0 train_accuracy 0.109375
        cuda:15 ./mnist 128 0.001 50 100.0 100 ./logs_lif_fc_mnist
        train_times 1024 train_accuracy 0.5078125
        cuda:15 ./mnist 128 0.001 50 100.0 100 ./logs_lif_fc_mnist
        train_times 2048 train_accuracy 0.7890625
        ...
        cuda:15 ./mnist 128 0.001 50 100.0 100 ./logs_lif_fc_mnist
        train_times 46080 train_accuracy 0.9296875

需要注意的是，训练这样的SNN，所需显存数量与仿真时长 ``T`` 线性相关，更长的 ``T`` 相当于使用更小的仿真步长，训练更为“精细”，\
但训练效果不一定更好，因此 ``T`` 太大，SNN在时间上展开后就会变成一个非常深的网络，梯度的传递容易衰减或爆炸。由于我们使用了泊松\
编码器，因此需要较大的 ``T``。

我们的这个模型，在Tesla K80上训练100个epoch，大约需要75分钟。训练时每个batch的正确率、测试集正确率的变化情况如下：

.. image:: ./_static/examples/clock_driven/lif_fc_mnist/accuracy_curve.png


最终达到大约92%的测试集正确率，这并不是一个很高的正确率，因为我们使用了非常简单的网络结构，以及泊松编码器。我们完全可以去掉泊松\
编码器，将图像直接送入SNN，在这种情况下，首层LIF神经元可以被视为编码器。