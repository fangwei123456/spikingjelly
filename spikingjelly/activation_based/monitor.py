import torch
import numpy as np
from torch import nn
from typing import Callable, Union, Optional
from spikingjelly.activation_based import neuron
import threading
from torch.utils.tensorboard import SummaryWriter
import os
import time
import re
import datetime

def unpack_len1_tuple(x: Union[tuple, torch.Tensor]):
    if isinstance(x, tuple) and x.__len__() == 1:
        return x[0]
    else:
        return x


class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()


class OutputMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, instance: Optional[Union[type, tuple[type, ...]]] = None, function_on_output: Callable = lambda x: x):
        """
        * :ref:`API in English <OutputMonitor-en>`

        .. _OutputMonitor-cn:

        :param net: 一个神经网络
        :type net: nn.Module
        :param instance: 被监视的模块的数据类型。若为 ``None`` 则表示类型为 ``type(net)``
        :type instance: Optional[Union[type, tuple[type, ...]]]
        :param function_on_output: 作用于被监控的模块输出的自定义的函数
        :type function_on_output: Callable

        对 ``net`` 中所有类型为 ``instance`` 的模块的输出使用 ``function_on_output`` 作用后，记录到类型为 `list`` 的 ``self.records`` 中。
        可以通过 ``self.enable()`` 和 ``self.disable()`` 来启用或停用这个监视器。
        可以通过 ``self.clear_recorded_data()`` 来清除已经记录的数据。
        
        阅读监视器的教程以获得更多信息。

        示例代码：

        .. code-block:: python

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = layer.Linear(8, 4)
                    self.sn1 = neuron.IFNode()
                    self.fc2 = layer.Linear(4, 2)
                    self.sn2 = neuron.IFNode()
                    functional.set_step_mode(self, 'm')

                def forward(self, x_seq: torch.Tensor):
                    x_seq = self.fc1(x_seq)
                    x_seq = self.sn1(x_seq)
                    x_seq = self.fc2(x_seq)
                    x_seq = self.sn2(x_seq)
                    return x_seq

            net = Net()
            for param in net.parameters():
                param.data.abs_()

            mtor = monitor.OutputMonitor(net, instance=neuron.IFNode)

            with torch.no_grad():
                y = net(torch.rand([1, 8]))
                print(f'mtor.records={mtor.records}')
                # mtor.records=[tensor([[0., 0., 0., 1.]]), tensor([[0., 0.]])]
                print(f'mtor[0]={mtor[0]}')
                # mtor[0]=tensor([[0., 0., 0., 1.]])
                print(f'mtor.monitored_layers={mtor.monitored_layers}')
                # mtor.monitored_layers=['sn1', 'sn2']
                print(f"mtor['sn1']={mtor['sn1']}")
                # mtor['sn1']=[tensor([[0., 0., 0., 1.]])]


        * :ref:`中文 API <OutputMonitor-cn>`

        .. _OutputMonitor-en:

        :param net: a network
        :type net: nn.Module
        :param instance: the instance of modules to be monitored. If ``None``, it will be regarded as ``type(net)``
        :type instance: Optional[Union[type, tuple[type, ...]]]
        :param function_on_output: the function that applies on the monitored modules' outputs
        :type function_on_output: Callable

        Applies ``function_on_output`` on outputs of all modules whose instances are ``instance`` in ``net``, and records
        the data into ``self.records``, which is a ``list``.
        Call ``self.enable()`` or ``self.disable()`` to enable or disable the monitor.
        Call ``self.clear_recorded_data()`` to clear the recorded data.

        Refer to the tutorial about the monitor for more details.

        Codes example:

        .. code-block:: python

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = layer.Linear(8, 4)
                    self.sn1 = neuron.IFNode()
                    self.fc2 = layer.Linear(4, 2)
                    self.sn2 = neuron.IFNode()
                    functional.set_step_mode(self, 'm')

                def forward(self, x_seq: torch.Tensor):
                    x_seq = self.fc1(x_seq)
                    x_seq = self.sn1(x_seq)
                    x_seq = self.fc2(x_seq)
                    x_seq = self.sn2(x_seq)
                    return x_seq

            net = Net()
            for param in net.parameters():
                param.data.abs_()

            mtor = monitor.OutputMonitor(net, instance=neuron.IFNode)

            with torch.no_grad():
                y = net(torch.rand([1, 8]))
                print(f'mtor.records={mtor.records}')
                # mtor.records=[tensor([[0., 0., 0., 1.]]), tensor([[0., 0.]])]
                print(f'mtor[0]={mtor[0]}')
                # mtor[0]=tensor([[0., 0., 0., 1.]])
                print(f'mtor.monitored_layers={mtor.monitored_layers}')
                # mtor.monitored_layers=['sn1', 'sn2']
                print(f"mtor['sn1']={mtor['sn1']}")
                # mtor['sn1']=[tensor([[0., 0., 0., 1.]])]
        """
        super().__init__()
        self.function_on_output = function_on_output
        if instance is None:
            instance = type(net)
        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook(name)))

    def create_hook(self, name):
        def hook(m, x, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.function_on_output(unpack_len1_tuple(y)))
        return hook



class InputMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, instance: Optional[Union[type, tuple[type, ...]]] = None, function_on_input: Callable = lambda x: x):
        """
        * :ref:`API in English <InputMonitor-en>`

        .. _InputMonitor-cn:

        :param net: 一个神经网络
        :type net: nn.Module
        :param instance: 被监视的模块的数据类型。若为 ``None`` 则表示类型为 ``type(net)``
        :type instance: Optional[Union[type, tuple[type, ...]]]
        :param function_on_input: 作用于被监控的模块输入的自定义的函数
        :type function_on_input: Callable

        对 ``net`` 中所有类型为 ``instance`` 的模块的输入使用 ``function_on_input`` 作用后，记录到类型为 `list`` 的 ``self.records`` 中。
        可以通过 ``self.enable()`` 和 ``self.disable()`` 来启用或停用这个监视器。
        可以通过 ``self.clear_recorded_data()`` 来清除已经记录的数据。
        
        阅读监视器的教程以获得更多信息。

        示例代码：

        .. code-block:: python

            import torch
            import torch.nn as nn
            from spikingjelly.activation_based import monitor, neuron, functional, layer

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = layer.Linear(8, 4)
                    self.sn1 = neuron.IFNode()
                    self.fc2 = layer.Linear(4, 2)
                    self.sn2 = neuron.IFNode()
                    functional.set_step_mode(self, 'm')

                def forward(self, x_seq: torch.Tensor):
                    x_seq = self.fc1(x_seq)
                    x_seq = self.sn1(x_seq)
                    x_seq = self.fc2(x_seq)
                    x_seq = self.sn2(x_seq)
                    return x_seq

            net = Net()
            for param in net.parameters():
                param.data.abs_()

            mtor = monitor.InputMonitor(net, instance=neuron.IFNode)

            with torch.no_grad():
                y = net(torch.rand([1, 8]))
                print(f'mtor.records={mtor.records}')
                # mtor.records=[tensor([[1.0165, 1.1934, 0.9347, 0.9539]]), tensor([[0.9115, 0.9508]])]
                print(f'mtor[0]={mtor[0]}')
                # mtor[0]=tensor([[1.0165, 1.1934, 0.9347, 0.9539]])
                print(f'mtor.monitored_layers={mtor.monitored_layers}')
                # mtor.monitored_layers=['sn1', 'sn2']
                print(f"mtor['sn1']={mtor['sn1']}")
                # mtor['sn1']=[tensor([[1.0165, 1.1934, 0.9347, 0.9539]])]



        * :ref:`中文 API <InputMonitor-cn>`

        .. _InputMonitor-en:

        :param net: a network
        :type net: nn.Module
        :param instance: the instance of modules to be monitored. If ``None``, it will be regarded as ``type(net)``
        :type instance: Any or tuple
        :param function_on_input: the function that applies on the monitored modules' inputs
        :type function_on_input: Callable

        Applies ``function_on_input`` on inputs of all modules whose instances are ``instance`` in ``net``, and records
        the data into ``self.records``, which is a ``list``.
        Call ``self.enable()`` or ``self.disable()`` to enable or disable the monitor.
        Call ``self.clear_recorded_data()`` to clear the recorded data.

        Refer to the tutorial about the monitor for more details.

        Codes example:

        .. code-block:: python

            import torch
            import torch.nn as nn
            from spikingjelly.activation_based import monitor, neuron, functional, layer

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = layer.Linear(8, 4)
                    self.sn1 = neuron.IFNode()
                    self.fc2 = layer.Linear(4, 2)
                    self.sn2 = neuron.IFNode()
                    functional.set_step_mode(self, 'm')

                def forward(self, x_seq: torch.Tensor):
                    x_seq = self.fc1(x_seq)
                    x_seq = self.sn1(x_seq)
                    x_seq = self.fc2(x_seq)
                    x_seq = self.sn2(x_seq)
                    return x_seq

            net = Net()
            for param in net.parameters():
                param.data.abs_()

            mtor = monitor.InputMonitor(net, instance=neuron.IFNode)

            with torch.no_grad():
                y = net(torch.rand([1, 8]))
                print(f'mtor.records={mtor.records}')
                # mtor.records=[tensor([[1.0165, 1.1934, 0.9347, 0.9539]]), tensor([[0.9115, 0.9508]])]
                print(f'mtor[0]={mtor[0]}')
                # mtor[0]=tensor([[1.0165, 1.1934, 0.9347, 0.9539]])
                print(f'mtor.monitored_layers={mtor.monitored_layers}')
                # mtor.monitored_layers=['sn1', 'sn2']
                print(f"mtor['sn1']={mtor['sn1']}")
                # mtor['sn1']=[tensor([[1.0165, 1.1934, 0.9347, 0.9539]])]
        """
        super().__init__()
        self.function_on_input = function_on_input
        if instance is None:
            instance = type(net)
        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook(name)))

    def create_hook(self, name):
        def hook(m, x, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.function_on_input(unpack_len1_tuple(x)))

        return hook


class AttributeMonitor(BaseMonitor):
    def __init__(self, attribute_name: str, pre_forward: bool, net: nn.Module, instance: Optional[Union[type, tuple[type, ...]]] = None,
                 function_on_attribute: Callable = lambda x: x):
        """
        * :ref:`API in English <AttributeMonitor-en>`

        .. _AttributeMonitor-cn:

        :param attribute_name: 要监控的成员变量的名字
        :type attribute_name: str
        :param pre_forward: 若为 ``True``，则记录模块在完成前向传播前的成员变量，否则记录完成前向传播后的变量
        :type pre_forward: bool
        :param net: 一个神经网络
        :type net: nn.Module
        :param instance: 被监视的模块的数据类型。若为 ``None`` 则表示类型为 ``type(net)``
        :type instance: Optional[Union[type, tuple[type, ...]]]
        :param function_on_attribute: 作用于被监控的模块 ``m`` 的成员 ``m.attribute_name`` 的自定义的函数
        :type function_on_attribute: Callable

        对 ``net`` 中所有类型为 ``instance`` 的模块 ``m`` 的成员 ``m.attribute_name`` 使用 ``function_on_attribute`` 作用后，记录到类型为 `list`` 的  ``self.records``。
        可以通过 ``self.enable()`` 和 ``self.disable()`` 来启用或停用这个监视器。
        可以通过 ``self.clear_recorded_data()`` 来清除已经记录的数据。
        
        阅读监视器的教程以获得更多信息。

        示例代码：

        .. code-block:: python

            import torch
            import torch.nn as nn
            from spikingjelly.activation_based import monitor, neuron, functional, layer

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = layer.Linear(8, 4)
                    self.sn1 = neuron.IFNode()
                    self.fc2 = layer.Linear(4, 2)
                    self.sn2 = neuron.IFNode()
                    functional.set_step_mode(self, 'm')

                def forward(self, x_seq: torch.Tensor):
                    x_seq = self.fc1(x_seq)
                    x_seq = self.sn1(x_seq)
                    x_seq = self.fc2(x_seq)
                    x_seq = self.sn2(x_seq)
                    return x_seq

            net = Net()
            for param in net.parameters():
                param.data.abs_()

            mtor = monitor.AttributeMonitor('v', False, net, instance=neuron.IFNode)

            with torch.no_grad():
                y = net(torch.rand([1, 8]))
                print(f'mtor.records={mtor.records}')
                # mtor.records=[tensor([0.0000, 0.6854, 0.0000, 0.7968]), tensor([0.4472, 0.0000])]
                print(f'mtor[0]={mtor[0]}')
                # mtor[0]=tensor([0.0000, 0.6854, 0.0000, 0.7968])
                print(f'mtor.monitored_layers={mtor.monitored_layers}')
                # mtor.monitored_layers=['sn1', 'sn2']
                print(f"mtor['sn1']={mtor['sn1']}")
                # mtor['sn1']=[tensor([0.0000, 0.6854, 0.0000, 0.7968])]





        * :ref:`中文 API <AttributeMonitor-cn>`

        .. _AttributeMonitor-en:

        :param attribute_name: the monitored attribute's name
        :type attribute_name: str
        :param pre_forward: If ``True``, recording the attribute before forward, otherwise recording the attribute after forward
        :type pre_forward: bool
        :param net: a network
        :type net: nn.Module
        :param instance: the instance of modules to be monitored. If ``None``, it will be regarded as ``type(net)``
        :type instance: Optional[Union[type, tuple[type, ...]]]
        :param function_on_attribute: the function that applies on each monitored module's attribute
        :type function_on_attribute: Callable

        Applies ``function_on_attribute`` on ``m.attribute_name`` of each monitored module ``m`` whose instance is ``instance`` in ``net``, and records
        the data into ``self.records``, which is a ``list``.
        Call ``self.enable()`` or ``self.disable()`` to enable or disable the monitor.
        Call ``self.clear_recorded_data()`` to clear the recorded data.

        Refer to the tutorial about the monitor for more details.

        Codes example:

        .. code-block:: python

            import torch
            import torch.nn as nn
            from spikingjelly.activation_based import monitor, neuron, functional, layer

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = layer.Linear(8, 4)
                    self.sn1 = neuron.IFNode()
                    self.fc2 = layer.Linear(4, 2)
                    self.sn2 = neuron.IFNode()
                    functional.set_step_mode(self, 'm')

                def forward(self, x_seq: torch.Tensor):
                    x_seq = self.fc1(x_seq)
                    x_seq = self.sn1(x_seq)
                    x_seq = self.fc2(x_seq)
                    x_seq = self.sn2(x_seq)
                    return x_seq

            net = Net()
            for param in net.parameters():
                param.data.abs_()

            mtor = monitor.AttributeMonitor('v', False, net, instance=neuron.IFNode)

            with torch.no_grad():
                y = net(torch.rand([1, 8]))
                print(f'mtor.records={mtor.records}')
                # mtor.records=[tensor([0.0000, 0.6854, 0.0000, 0.7968]), tensor([0.4472, 0.0000])]
                print(f'mtor[0]={mtor[0]}')
                # mtor[0]=tensor([0.0000, 0.6854, 0.0000, 0.7968])
                print(f'mtor.monitored_layers={mtor.monitored_layers}')
                # mtor.monitored_layers=['sn1', 'sn2']
                print(f"mtor['sn1']={mtor['sn1']}")
                # mtor['sn1']=[tensor([0.0000, 0.6854, 0.0000, 0.7968])]

        """
        super().__init__()
        self.attribute_name = attribute_name
        self.function_on_attribute = function_on_attribute
        if instance is None:
            instance = type(net)

        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                if pre_forward:
                    self.hooks.append(
                        m.register_forward_pre_hook(self.create_hook(name))
                    )
                else:
                    self.hooks.append(
                        m.register_forward_hook(self.create_hook(name))
                    )

    def create_hook(self, name):
        def hook(m, x, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.function_on_attribute(m.__getattr__(self.attribute_name)))

        return hook

class GradInputMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, instance: Optional[Union[type, tuple[type, ...]]] = None, function_on_grad_input: Callable = lambda x: x):
        """
        * :ref:`API in English <GradInputMonitor-en>`

        .. _GradInputMonitor-cn:

        :param net: 一个神经网络
        :type net: nn.Module
        :param instance: 被监视的模块的数据类型。若为 ``None`` 则表示类型为 ``type(net)``
        :type instance: Optional[Union[type, tuple[type, ...]]]
        :param function_on_grad_input: 作用于被监控的模块输出的输入的梯度的函数
        :type function_on_grad_input: Callable

        对 ``net`` 中所有类型为 ``instance`` 的模块的输入的梯度使用 ``function_on_grad_input`` 作用后，记录到类型为 `list`` 的 ``self.records`` 中。
        可以通过 ``self.enable()`` 和 ``self.disable()`` 来启用或停用这个监视器。
        可以通过 ``self.clear_recorded_data()`` 来清除已经记录的数据。
        
        阅读监视器的教程以获得更多信息。

        .. Note::

            对于一个模块，输入为 :math:`X`，输出为 :math:`Y`，损失为 :math:`L`，则 ``GradInputMonitor`` 记录的是对输入的梯度 :math:`\\frac{\\partial L}{\\partial X}`。


        示例代码：

        .. code-block:: python

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = layer.Linear(8, 4)
                    self.sn1 = neuron.IFNode()
                    self.fc2 = layer.Linear(4, 2)
                    self.sn2 = neuron.IFNode()
                    functional.set_step_mode(self, 'm')

                def forward(self, x_seq: torch.Tensor):
                    x_seq = self.fc1(x_seq)
                    x_seq = self.sn1(x_seq)
                    x_seq = self.fc2(x_seq)
                    x_seq = self.sn2(x_seq)
                    return x_seq

            net = Net()
            for param in net.parameters():
                param.data.abs_()

            mtor = monitor.GradInputMonitor(net, instance=neuron.IFNode)

            with torch.no_grad():
                y = net(torch.rand([1, 8]))
                print(f'mtor.records={mtor.records}')
                # mtor.records=[tensor([0.0000, 0.6854, 0.0000, 0.7968]), tensor([0.4472, 0.0000])]
                print(f'mtor[0]={mtor[0]}')
                # mtor[0]=tensor([0.0000, 0.6854, 0.0000, 0.7968])
                print(f'mtor.monitored_layers={mtor.monitored_layers}')
                # mtor.monitored_layers=['sn1', 'sn2']
                print(f"mtor['sn1']={mtor['sn1']}")
                # mtor['sn1']=[tensor([0.0000, 0.6854, 0.0000, 0.7968])]



        * :ref:`中文 API <GradInputMonitor-cn>`

        .. _GradInputMonitor-en:

        :param net: a network
        :type net: nn.Module
        :param instance: the instance of modules to be monitored. If ``None``, it will be regarded as ``type(net)``
        :type instance: Optional[Union[type, tuple[type, ...]]]
        :param function_on_grad_input: the function that applies on the grad of monitored modules' inputs
        :type function_on_grad_input: Callable

        Applies ``function_on_grad_input`` on grad of inputs of all modules whose instances are ``instance`` in ``net``, and records
        the data into ``self.records``, which is a ``list``.
        Call ``self.enable()`` or ``self.disable()`` to enable or disable the monitor.
        Call ``self.clear_recorded_data()`` to clear the recorded data.

        Refer to the tutorial about the monitor for more details.

        .. admonition:: Note
            :class: note

            Denote the input and output of the monitored module as :math:`X` and :math:`Y`, and the loss is :math:`L`, then ``GradInputMonitor`` will record the gradient of input, which is :math:`\\frac{\\partial L}{\\partial X}`.

        Codes example:

        .. code-block:: python

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = layer.Linear(8, 4)
                    self.sn1 = neuron.IFNode()
                    self.fc2 = layer.Linear(4, 2)
                    self.sn2 = neuron.IFNode()
                    functional.set_step_mode(self, 'm')

                def forward(self, x_seq: torch.Tensor):
                    x_seq = self.fc1(x_seq)
                    x_seq = self.sn1(x_seq)
                    x_seq = self.fc2(x_seq)
                    x_seq = self.sn2(x_seq)
                    return x_seq

            net = Net()
            for param in net.parameters():
                param.data.abs_()

            mtor = monitor.GradInputMonitor(net, instance=neuron.IFNode)

            with torch.no_grad():
                y = net(torch.rand([1, 8]))
                print(f'mtor.records={mtor.records}')
                # mtor.records=[tensor([0.0000, 0.6854, 0.0000, 0.7968]), tensor([0.4472, 0.0000])]
                print(f'mtor[0]={mtor[0]}')
                # mtor[0]=tensor([0.0000, 0.6854, 0.0000, 0.7968])
                print(f'mtor.monitored_layers={mtor.monitored_layers}')
                # mtor.monitored_layers=['sn1', 'sn2']
                print(f"mtor['sn1']={mtor['sn1']}")
                # mtor['sn1']=[tensor([0.0000, 0.6854, 0.0000, 0.7968])]
        """
        super().__init__()
        self.function_on_grad_input = function_on_grad_input
        if instance is None:
            instance = type(net)

        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                if torch.__version__ >= torch.torch_version.TorchVersion('1.8.0'):
                    self.hooks.append(m.register_full_backward_hook(self.create_hook(name)))
                else:
                    self.hooks.append(m.register_backward_hook(self.create_hook(name)))

    def create_hook(self, name):
        def hook(m, grad_input, grad_output):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.function_on_grad_input(unpack_len1_tuple(grad_input)))

        return hook


class GradOutputMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, instance: Optional[Union[type, tuple[type, ...]]] = None, function_on_grad_output: Callable = lambda x: x):
        """
        * :ref:`API in English <GradOutputMonitor-en>`

        .. _GradOutputMonitor-cn:

        :param net: 一个神经网络
        :type net: nn.Module
        :param instance: 被监视的模块的数据类型。若为 ``None`` 则表示类型为 ``type(net)``
        :type instance: Optional[Union[type, tuple[type, ...]]]
        :param function_on_grad_output: 作用于被监控的模块输出的输出的的梯度的函数
        :type function_on_grad_output: Callable

        对 ``net`` 中所有类型为 ``instance`` 的模块的输出的梯度使用 ``function_on_grad_output`` 作用后，记录到类型为 `list`` 的 ``self.records`` 中。
        可以通过 ``self.enable()`` 和 ``self.disable()`` 来启用或停用这个监视器。
        可以通过 ``self.clear_recorded_data()`` 来清除已经记录的数据。
        
        阅读监视器的教程以获得更多信息。

        .. Note::

            对于一个模块，输入为 :math:`X`，输出为 :math:`Y`，损失为 :math:`L`，则 ``GradOutputMonitor`` 记录的是对输出的梯度 :math:`\\frac{\\partial L}{\\partial Y}`。

        示例代码：

        .. code-block:: python

            import torch
            import torch.nn as nn
            from spikingjelly.activation_based import monitor, neuron, functional, layer

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = layer.Linear(8, 4)
                    self.sn1 = neuron.IFNode()
                    self.fc2 = layer.Linear(4, 2)
                    self.sn2 = neuron.IFNode()
                    functional.set_step_mode(self, 'm')

                def forward(self, x_seq: torch.Tensor):
                    x_seq = self.fc1(x_seq)
                    x_seq = self.sn1(x_seq)
                    x_seq = self.fc2(x_seq)
                    x_seq = self.sn2(x_seq)
                    return x_seq

            net = Net()
            for param in net.parameters():
                param.data.abs_()

            mtor = monitor.GradOutputMonitor(net, instance=neuron.IFNode)

            net(torch.rand([1, 8])).sum().backward()
            print(f'mtor.records={mtor.records}')
            # mtor.records=[tensor([[1., 1.]]), tensor([[0.1372, 0.1081, 0.0880, 0.1089]])]
            print(f'mtor[0]={mtor[0]}')
            # mtor[0]=tensor([[1., 1.]])
            print(f'mtor.monitored_layers={mtor.monitored_layers}')
            # mtor.monitored_layers=['sn1', 'sn2']
            print(f"mtor['sn1']={mtor['sn1']}")
            # mtor['sn1']=[tensor([[0.1372, 0.1081, 0.0880, 0.1089]])]



        * :ref:`中文 API <GradOutputMonitor-cn>`

        .. _GradOutputMonitor-en:

        :param net: a network
        :type net: nn.Module
        :param instance: the instance of modules to be monitored. If ``None``, it will be regarded as ``type(net)``
        :type instance: Optional[Union[type, tuple[type, ...]]]
        :param function_on_grad_output: the function that applies on the grad of monitored modules' inputs
        :type function_on_grad_output: Callable

        Applies ``function_on_grad_output`` on grad of outputs of all modules whose instances are ``instance`` in ``net``, and records
        the data into ``self.records``, which is a ``list``.
        Call ``self.enable()`` or ``self.disable()`` to enable or disable the monitor.
        Call ``self.clear_recorded_data()`` to clear the recorded data.

        Refer to the tutorial about the monitor for more details.

        .. admonition:: Note
            :class: note

            Denote the input and output of the monitored module as :math:`X` and :math:`Y`, and the loss is :math:`L`, then ``GradOutputMonitor`` will record the gradient of output, which is :math:`\\frac{\\partial L}{\\partial Y}`.


        Codes example:

        .. code-block:: python

            import torch
            import torch.nn as nn
            from spikingjelly.activation_based import monitor, neuron, functional, layer

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = layer.Linear(8, 4)
                    self.sn1 = neuron.IFNode()
                    self.fc2 = layer.Linear(4, 2)
                    self.sn2 = neuron.IFNode()
                    functional.set_step_mode(self, 'm')

                def forward(self, x_seq: torch.Tensor):
                    x_seq = self.fc1(x_seq)
                    x_seq = self.sn1(x_seq)
                    x_seq = self.fc2(x_seq)
                    x_seq = self.sn2(x_seq)
                    return x_seq

            net = Net()
            for param in net.parameters():
                param.data.abs_()

            mtor = monitor.GradOutputMonitor(net, instance=neuron.IFNode)

            net(torch.rand([1, 8])).sum().backward()
            print(f'mtor.records={mtor.records}')
            # mtor.records=[tensor([[1., 1.]]), tensor([[0.1372, 0.1081, 0.0880, 0.1089]])]
            print(f'mtor[0]={mtor[0]}')
            # mtor[0]=tensor([[1., 1.]])
            print(f'mtor.monitored_layers={mtor.monitored_layers}')
            # mtor.monitored_layers=['sn1', 'sn2']
            print(f"mtor['sn1']={mtor['sn1']}")
            # mtor['sn1']=[tensor([[0.1372, 0.1081, 0.0880, 0.1089]])]

        """

        super().__init__()
        self.function_on_grad_output = function_on_grad_output
        if instance is None:
            instance = type(net)
        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                if torch.__version__ >= torch.torch_version.TorchVersion('1.8.0'):
                    self.hooks.append(m.register_full_backward_hook(self.create_hook(name)))
                else:
                    self.hooks.append(m.register_backward_hook(self.create_hook(name)))

    def create_hook(self, name):
        def hook(m, grad_input, grad_output):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.function_on_grad_output(unpack_len1_tuple(grad_output)))

        return hook

class GPUMonitor(threading.Thread):
    def __init__(self, log_dir: Optional[str] = None, gpu_ids: tuple = (0,), interval: float = 600., start_now=True):
        """
        * :ref:`API in English <GPUMonitor.__init__-en>`

        .. _GPUMonitor.__init__-cn:

        :param log_dir: 使用 ``tensorboard`` 保存GPU数据的文件夹. 若为None，则日志不会保存，而是直接 ``print``
        :type log_dir: Optional[str]
        :param gpu_ids: 监视的GPU，例如 ``(0, 1, 2, 3)``。默认为 ``(0, )``
        :type gpu_ids: tuple
        :param interval: 记录数据的间隔，单位是秒
        :type interval: float
        :param start_now: 若为 ``True`` 则初始化后会立刻开始记录数据，否则需要手动调用 ``start()`` 后才开始记录数据
        :type start_now:

        GPU监视器，可以开启一个新的线程来记录 ``gpu_ids`` 的使用率和显存使用情况，每 ``interval`` 秒记录一次数据。

        .. Warning::

            在主线程的工作完成后一定要调用GPU监视器的 ``stop()`` 函数，否则主线程不会退出。

        Codes example:

        .. code-block:: python

            import time

            gm = GPUMonitor(interval=1)
            time.sleep(2)  # make the main thread sleep
            gm.stop()

            # The outputs are:

            # 2022-04-28 10:52:25
            # utilization.gpu [%], memory.used [MiB]
            # 0 %, 376 MiB

        * :ref:`中文API <GPUMonitor.__init__-cn>`

        .. _GPUMonitor.__init__-en:

        :param log_dir: the directory for saving logs with tensorboard. If it is None, this module will print logs
        :type log_dir: Optional[str]
        :param gpu_ids: the id of GPUs to be monitored, e.g., ``(0, 1, 2, 3)``. The default value is ``(0, )``
        :type gpu_ids: tuple
        :param interval: the recording interval (in seconds)
        :type interval: float
        :param start_now: if true, the monitor will start to record now. Otherwise, it will start after the user call ``start()`` manually
        :type start_now:

        The GPU monitor, which starts a new thread to record the utilization and memory used of ``gpu_ids`` every ``interval`` seconds.

        .. admonition:: Warning
            :class: warning

            Do not forget to call this module's ``stop()`` after the main thread finishes its job, otherwise the main thread will never stop!

        Codes example:

        .. code-block:: python

            import time

            gm = GPUMonitor(interval=1)
            time.sleep(2)  # make the main thread sleep
            gm.stop()

            # The outputs are:

            # 2022-04-28 10:52:25
            # utilization.gpu [%], memory.used [MiB]
            # 0 %, 376 MiB
        """
        super().__init__()
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.stopped = False
        self.cmds = 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
        self.cmds += ' -i '
        id_str = []
        for gpu_id in self.gpu_ids:
            id_str.append(str(gpu_id))
        self.cmds += ','.join(id_str)
        self.step = 0

        if log_dir is None:
            self.writer = None
        else:
            self.writer = SummaryWriter(os.path.join(log_dir, 'gpu_monitor'))

        if start_now:
            self.start()

    def stop(self):
        self.stopped = True

    def run(self):
        while not self.stopped:
            with os.popen(self.cmds) as fp:
                outputs = fp.read()
                if self.writer is not None:
                    outputs = outputs.split('\n')[1:-1]
                    # skip the first row 'utilization.gpu [%], memory.used [MiB]' and the last row ('\n')
                    for i in range(outputs.__len__()):
                        utilization_memory = re.findall(r'\d+', outputs[i])
                        utilization = int(utilization_memory[0])
                        memory_used = int(utilization_memory[1])
                        self.writer.add_scalar(f'utilization_{self.gpu_ids[i]}', utilization, self.step)
                        self.writer.add_scalar(f'memory_used_{self.gpu_ids[i]}', memory_used, self.step)
                else:
                    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    print(outputs)
                    '''
                    2022-04-20 18:14:26
                    utilization.gpu [%], memory.used [MiB]
                    4 %, 1816 MiB
                    0 %, 1840 MiB
                    0 %, 1840 MiB
                    0 %, 1720 MiB
                    '''
            time.sleep(self.interval)
            self.step += 1
