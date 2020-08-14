时间驱动：使用卷积SNN识别Fashion-MNIST
=======================================
本教程作者：


.. code-block:: python

    >>> from spikingflow.clock_driven.examples import conv_fashion_mnist
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


