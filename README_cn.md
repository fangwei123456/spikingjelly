# SpikingFlow ![GitHub last commit](https://img.shields.io/github/last-commit/fangwei123456/spikingflow) [![Documentation Status](https://readthedocs.org/projects/spikingflow/badge/?version=latest)](https://spikingflow.readthedocs.io/zh_CN/latest) [![PyPI](https://img.shields.io/pypi/v/spikingflow)](https://pypi.org/project/spikingflow) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spikingflow)](https://pypi.org/project/spikingflow) ![License](https://img.shields.io/github/license/fangwei123456/spikingflow)

[English](https://github.com/fangwei123456/spikingflow/blob/master/README.md) | 中文

![demo](demo.png)

[SpikingFlow](https://github.com/fangwei123456/spikingflow) 是一个基于 [PyTorch](https://pytorch.org/) ，使用脉冲神经网络(Spiking Neuron Network, SNN)进行深度学习的框架。

SpikingFlow的文档使用中英双语编写： https://spikingflow.readthedocs.io

## 安装

注意，SpikingFlow是基于PyTorch的，需要确保环境中已经安装了PyTorch，才能安装SpikingFlow。

从 [PyPI](https://pypi.org/project/spikingflow/) 安装：

```bash
pip install spikingflow
```

或者对于开发者，下载源代码，进行代码补充、修改和测试：

```bash
git clone https://github.com/fangwei123456/spikingflow.git
```

## 以前所未有的简单方式搭建SNN

SpikingFlow非常易于使用。使用SpikingFlow搭建SNN，就像使用PyTorch搭建ANN一样简单：

```python
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
```

这个简单的网络，使用泊松编码器，在MNIST的测试集上可以达到92%的正确率。 更多信息，参见[时间驱动的教程](https://spikingflow.readthedocs.io/zh_CN/latest/tutorial.clock_driven.html)。可以通过Python命令行直接运行这份代码，训练MNIST分类：

```python
>>> import spikingflow.clock_driven.examples.lif_fc_mnist as lif_fc_mnist
>>> lif_fc_mnist.main()
```

阅读[spikingflow.clock_driven.examples](https://spikingflow.readthedocs.io/zh_CN/latest/spikingflow.clock_driven.examples.html)以探索更多先进的神经网络！

## 设备支持

-   [x] Nvidia GPU
-   [x] CPU

像使用PyTorch一样简单。

```python
>>> net = nn.Sequential(nn.Flatten(), neuron.LIFNode(tau=tau))
>>> net = net.to(device) # Can be CPU or CUDA devices
```

## 项目信息

北京大学信息科学技术学院数字媒体所媒体学习组 [Multimedia Learning Group](https://pkuml.org/) 是SpikingFlow的主要开发者。

开发人员名单可见于 https://github.com/fangwei123456/spikingflow/graphs/contributors。

欢迎各位开发者参与此框架的开发！
