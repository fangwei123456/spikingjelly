# SpikingFlow

[![Documentation Status](https://readthedocs.org/projects/spikingflow/badge/?version=latest)](https://spikingflow.readthedocs.io/zh_CN/latest)
[![HitCount](http://hits.dwyl.com/fangwei123456/SpikingFlow.svg)](http://hits.dwyl.com/fangwei123456/SpikingFlow)

[README in English](https://github.com/fangwei123456/SpikingFlow/blob/master/README.md)

[SpikingFlow](https://github.com/fangwei123456/SpikingFlow) 是一个基于 [PyTorch](https://pytorch.org/) ，使用脉冲神经网络(Spiking Neuron Network, SNN)进行深度学习的框架。

SpikingFlow的文档使用中英双语编写： https://spikingflow.readthedocs.io

## 安装

注意，SpikingFlow是基于PyTorch的，需要确保环境中已经安装了PyTorch，才能安装SpikingFlow。

从 [PyPI](https://pypi.org/project/SpikingFlow/) 安装：

```bash
pip install SpikingFlow
```

或者对于开发者，下载源代码，进行代码补充、修改和测试：

```bash
git clone https://github.com/fangwei123456/SpikingFlow.git
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
>>> import SpikingFlow.clock_driven.examples.lif_fc_mnist as lif_fc_mnist
>>> lif_fc_mnist.main()
```

## 项目信息

北京大学信息科学技术学院数字媒体所媒体学习组 [Multimedia Learning Group](https://pkuml.org/) 是SpikingFlow的主要开发者。

开发人员名单可见于 https://github.com/fangwei123456/SpikingFlow/graphs/contributors。

欢迎各位开发者参与此框架的开发！
