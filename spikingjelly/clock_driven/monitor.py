import torch
import numpy as np
from torch import nn
from spikingjelly.clock_driven import neuron

try:
    from spikingjelly.cext import neuron as cext_neuron
except ImportError:
    cext_neuron = None

class Monitor:
    def __init__(self, net: nn.Module, device: str = None, backend: str = 'numpy'):
        '''
        * :ref:`API in English <Monitor.__init__-en>`

        .. _Monitor.__init__-cn:

        :param net: 要监视的网络
        :type net: nn.Module
        :param device: 监视数据的存储和处理的设备，仅当backend为 ``'torch'`` 时有效。可以为 ``'cpu', 'cuda', 'cuda:0'`` 字符串或者 ``torch.device`` 类型，默认为 ``None``
        :type device: str, optional
        :param backend: 监视数据的处理后端。可以为 ``'torch', 'numpy'`` ，默认为 ``'numpy'``
        :type backend: str, optional

        * :ref:`中文API <Monitor.__init__-cn>`

        .. _Monitor.__init__-en:

        :param net: Network to be monitored
        :type net: nn.Module
        :param device: Device carrying and processing monitored data. Only take effect when backend is set to ``'torch'``. Can be string ``'cpu', 'cuda', 'cuda:0'`` or ``torch.device``, defaults to ``None``
        :type device: str, optional
        :param backend: Backend processing monitored data, can be ``'torch', 'numpy'``, defaults to ``'numpy'``
        :type backend: str, optional
        '''
    
        super().__init__()
        self.module_dict = dict()
        for name, module in net.named_modules():
            if (cext_neuron is not None and isinstance(module, cext_neuron.BaseNode)) or isinstance(module, neuron.BaseNode):
                self.module_dict[name] = module
                #setattr(module, 'monitor', self)

        # 'torch' or 'numpy'
        self.net = net
        self.backend = backend

        if isinstance(device, str) and self.backend == 'torch':
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise ValueError('Expected a cuda or cpu device, but got: {}'.format(device))

    def enable(self):
        '''
        * :ref:`API in English <Monitor.enable-en>`

        .. _Monitor.enable-cn:

        启用Monitor的监视功能，开始记录数据

        * :ref:`中文API <Monitor.enable-cn>`

        .. _Monitor.enable-en:

        Enable Monitor. Start recording data.
        '''
        self.handle = dict.fromkeys(self.module_dict, None)
        self.neuron_cnt = dict.fromkeys(self.module_dict, None)

        for name, module in self.module_dict.items():
            setattr(module, 'neuron_cnt', self.neuron_cnt[name])

            # 初始化前向时钩子的句柄
            self.handle[name] = module.register_forward_hook(self.forward_hook)
                
        self.reset()

    def disable(self):
        '''
        * :ref:`API in English <Monitor.disable-en>`

        .. _Monitor.disable-cn:

        禁用Monitor的监视功能，不再记录数据

        * :ref:`中文API <Monitor.disable-cn>`

        .. _Monitor.disable-en:

        Disable Monitor. Stop recording data.
        '''
        for name, module in self.module_dict.items():
            delattr(module, 'neuron_cnt')
            delattr(module, 'fire_mask')
            delattr(module, 'firing_time')
            delattr(module, 'cnt')

            # 删除钩子
            self.handle[name].remove()
    
    # 暂时只监视脉冲发放
    @torch.no_grad()
    def forward_hook(self, module, input, output):
        if module.__class__.__name__.startswith('MultiStep'):
            output_shape = output.shape
            data = output.view([-1,] + list(output_shape[2:])).clone() # 对于多步模块的输出[T, batchsize, ...]的前两维进行合并
        else:
            data = output.clone()

        # Numpy
        if self.backend == 'numpy':
            data = data.cpu().numpy()
            if module.neuron_cnt is None:
                module.neuron_cnt = data[0].size # 神经元数量
            module.firing_time += np.sum(data) # data中脉冲总数量
            module.cnt += data.size # data本身的尺寸（T*batchsize*神经元数量）
            fire_mask = (np.sum(data, axis=0) > 0) # 各神经元位置是否发放过脉冲的mask（Bool类型）

        # PyTorch
        else:
            data = data.to(self.device)
            if module.neuron_cnt is None:
                module.neuron_cnt = data[0].numel()
            module.firing_time += torch.sum(data)
            module.cnt += data.numel()
            fire_mask = (torch.sum(data, dim=0) > 0)
        
        # PyTorch与Numpy的Bool Tensor的logical_or操作均可以直接用|表示。并且可以直接与Python的Bool类型进行运算，但是第一个操作数必须是Bool Tensor，不能是Python的Bool类型
        module.fire_mask = fire_mask | module.fire_mask 


    def reset(self):
        '''
        * :ref:`API in English <Monitor.reset-en>`

        .. _Monitor.reset-cn:

        清空之前的记录数据

        * :ref:`中文API <Monitor.reset-cn>`

        .. _Monitor.reset-en:

        Delete previously recorded data
        '''
        for name, module in self.module_dict.items():
            setattr(module, 'fire_mask', False)
            setattr(module, 'firing_time', 0)
            setattr(module, 'cnt', 0)

    def get_avg_firing_rate(self, all: bool = True, module_name: str = None) -> torch.Tensor or float:
        '''
        * :ref:`API in English <Monitor.get_avg_firing_rate-en>`

        .. _Monitor.get_avg_firing_rate-cn:

        :param all: 是否为所有层的总平均发放率，默认为 ``True``
        :type all: bool, optional
        :param module_name: 层的名称，仅当all为 ``False`` 时有效
        :type module_name: str, optional
        :return: 所关心层的平均发放率
        :rtype: torch.Tensor or float

        * :ref:`中文API <Monitor.get_avg_firing_rate-cn>`

        .. _Monitor.get_avg_firing_rate-en:

        :param all: Whether needing firing rate averaged on all layers, defaults to ``True``
        :type all: bool, optional
        :param module_name: Name of concerned layer. Only take effect when all is ``False``
        :type module_name: str, optional
        :return: Averaged firing rate on concerned layers
        :rtype: torch.Tensor or float
        '''
        if all:
            ttl_firing_time = 0
            ttl_cnt = 0
            for name, module in self.module_dict.items():
                ttl_firing_time += module.firing_time
                ttl_cnt += module.cnt
            return ttl_firing_time / ttl_cnt 
        else:
            if module_name not in self.module_dict.keys():
                raise ValueError(f'Invalid module_name \'{module_name}\'')
            module = self.module_dict[module_name]
            return module.firing_time / module.cnt


    def get_nonfire_ratio(self, all: bool = True, module_name: str = None) -> torch.Tensor or float:
        '''
        * :ref:`API in English <Monitor.get_nonfire_ratio-en>`

        .. _Monitor.get_nonfire_ratio-cn:

        :param all: 是否为所有层的静默神经元比例，默认为 ``True``
        :type all: bool, optional
        :param module_name: 层的名称，仅当all为 ``False`` 时有效
        :type module_name: str, optional
        :return: 所关心层的静默神经元比例
        :rtype: torch.Tensor or float

        * :ref:`中文API <Monitor.get_nonfire_ratio-cn>`

        .. _Monitor.get_nonfire_ratio-en:

        :param all: Whether needing ratio of silent neurons of all layers, defaults to ``True``
        :type all: bool, optional
        :param module_name: Name of concerned layer. Only take effect when all is ``False``
        :type module_name: str, optional
        :return: Ratio of silent neurons on concerned layers
        :rtype: torch.Tensor or float
        '''
        if all:
            ttl_neuron_cnt = 0
            ttl_zero_cnt = 0
            for name, module in self.module_dict.items():
                if self.backend == 'numpy':
                    ttl_zero_cnt += np.logical_not(module.fire_mask).sum()
                elif self.backend == 'torch':
                    ttl_zero_cnt += torch.logical_not(module.fire_mask).sum()
                ttl_neuron_cnt += module.neuron_cnt
            return ttl_zero_cnt / ttl_neuron_cnt
        else:
            if module_name not in self.module_dict.keys():
                raise ValueError(f'Invalid module_name \'{module_name}\'')

            module = self.module_dict[module_name]
            if self.backend == 'numpy':
                return np.logical_not(module.fire_mask).sum() / module.neuron_cnt
            elif self.backend == 'torch':
                return torch.logical_not(module.fire_mask).sum() / module.neuron_cnt