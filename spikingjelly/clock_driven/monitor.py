import torch
import numpy as np
from torch import nn
from spikingjelly.clock_driven import neuron
from spikingjelly.cext import neuron as cext_neuron

class Monitor:
    def __init__(self, net: nn.Module, device: str = None, backend: str = 'numpy'):
        '''
        * :ref:`API in English <Monitor.__init__-en>`

        .. _Monitor.__init__-cn:

        :param net: 要监视的网络
        :type net: nn.Module
        :param device: 监视数据的存储和处理的设备，仅当backend为 ``'torch'`` 时有效。可以为 ``'cpu', 'cuda', 'cuda:0'`` 等，默认为 ``None``
        :type device: str, optional
        :param backend: 监视数据的处理后端。可以为 ``'torch', 'numpy'`` ，默认为 ``'numpy'``
        :type backend: str, optional

        * :ref:`中文API <Monitor.__init__-cn>`

        .. _Monitor.__init__-en:

        :param net: Network to be monitored
        :type net: nn.Module
        :param device: Device carrying and processing monitored data. Only take effect when backend is set to ``'torch'``. Can be ``'cpu', 'cuda', 'cuda:0'``, et al., defaults to ``None``
        :type device: str, optional
        :param backend: Backend processing monitored data, can be ``'torch', 'numpy'``, defaults to ``'numpy'``
        :type backend: str, optional
        '''
    
        super().__init__()
        self.module_dict = dict()
        for name, module in net.named_modules():
            if isinstance(module, (cext_neuron.BaseNode, neuron.BaseNode)):
                self.module_dict[name] = module
                #setattr(module, 'monitor', self)

        # 'torch' or 'numpy'
        self.net = net
        self.backend = backend

        if self.backend == 'torch':
            self.device = torch.device(device)

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
        self.v = dict.fromkeys(self.module_dict, None)
        self.s = dict.fromkeys(self.module_dict, None)

        for name, module in self.net.named_modules():
            if isinstance(module, (cext_neuron.BaseNode, neuron.BaseNode)):
                self.v[name] = []
                self.s[name] = []
                setattr(module, 'v_list', self.v[name])
                setattr(module, 's_list', self.s[name])
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
        for name, module in self.net.named_modules():
            if isinstance(module, (cext_neuron.BaseNode, neuron.BaseNode)):
                delattr(module, 'v_list')
                delattr(module, 's_list')
                # 删除钩子
                self.handle[name].remove()
    
    # 暂时只监视脉冲发放
    def forward_hook(self, module, input, output):
        with torch.no_grad():
            if module.__class__.__name__.startswith('MultiStep'):
                output_shape = output.shape
                data = output.view([-1,] + list(output_shape[2:])).clone() # 对于多步模块的输出[T, batch size, ...]的前两维进行合并
            else:
                data = output.clone()

            if self.backend == 'numpy':
                data = data.cpu().numpy()
            else:
                data = data.to(self.device)

            module.s_list.append(data)


    def reset(self):
        '''
        * :ref:`API in English <Monitor.reset-en>`

        .. _Monitor.reset-cn:

        清空之前的记录数据

        * :ref:`中文API <Monitor.reset-cn>`

        .. _Monitor.reset-en:

        Delete previously recorded data
        '''

        for v_list in self.v.values():
            v_list.clear()
        
        for s_list in self.s.values():
            s_list.clear()
        # 数据与字典置空
        self.neuron_cnt = dict.fromkeys(self.module_dict, None)
        self.avg_firing_rate_by_layer = dict.fromkeys(self.module_dict, None)
        self.nonfire_ratio_by_layer = dict.fromkeys(self.module_dict, None)

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
            ttl_firing_cnt = 0
            ttl_neuron_cnt = 0
            for name in self.module_dict.keys():
                # 尽量复用已经计算出的结果
                if self.avg_firing_rate_by_layer[name] is None:
                    # 这里记录的平均发放率实际上对总样本数、仿真时间、神经元尺寸三个维度都做了平均
                    if self.backend == 'numpy':
                        self.avg_firing_rate_by_layer[name] = np.concatenate(self.s[name]).mean()
                        if self.neuron_cnt[name] is None:
                            self.neuron_cnt[name] = self.s[name][0][0].size
                    elif self.backend == 'torch':
                        self.avg_firing_rate_by_layer[name] = torch.cat(self.s[name]).mean()
                        if self.neuron_cnt[name] is None:
                            self.neuron_cnt[name] = self.s[name][0][0].numel()
                ttl_firing_cnt += self.avg_firing_rate_by_layer[name] * self.neuron_cnt[name]
                ttl_neuron_cnt += self.neuron_cnt[name]
            return ttl_firing_cnt / ttl_neuron_cnt 
        else:
            if module_name not in self.module_dict.keys():
                raise ValueError(f'invalid module_name {module_name}')

            # 尽量复用已经计算出的结果
            if self.avg_firing_rate_by_layer[module_name] is None:
                if self.backend == 'numpy':
                    self.avg_firing_rate_by_layer[module_name] = np.concatenate(self.s[module_name]).mean()
                elif self.backend == 'torch':
                    self.avg_firing_rate_by_layer[module_name] = torch.cat(self.s[module_name]).mean() 
            return self.avg_firing_rate_by_layer[module_name]


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
            for name in self.module_dict.keys():
                # 尽量复用已经计算出的结果
                if self.nonfire_ratio_by_layer[name] is None:
                    # 这里记录的平均发放率实际上对总样本数、仿真时间、神经元尺寸三个维度都做了平均
                    if self.backend == 'numpy':
                        ttl_firing_times = np.concatenate(self.s[name]).sum(axis=0)
                        if self.neuron_cnt[name] is None:
                            self.neuron_cnt[name] = self.s[name][0][0].size
                        self.nonfire_ratio_by_layer[name] = (ttl_firing_times == 0).astype(float).sum() / ttl_firing_times.size
                    elif self.backend == 'torch':
                        ttl_firing_times = torch.cat(self.s[name]).sum(dim=0)
                        if self.neuron_cnt[name] is None:
                            self.neuron_cnt[name] = self.s[name][0][0].numel()
                        self.nonfire_ratio_by_layer[name] = (ttl_firing_times == 0).float().sum() / ttl_firing_times.numel()

                ttl_neuron_cnt += self.neuron_cnt[name]
                ttl_zero_cnt += self.nonfire_ratio_by_layer[name] * self.neuron_cnt[name]
            return ttl_zero_cnt / ttl_neuron_cnt
        else:
            if module_name not in self.module_dict.keys():
                raise ValueError(f'invalid module_name \'{module_name}\'')

            # 尽量复用已经计算出的结果
            if self.nonfire_ratio_by_layer[module_name] is None:
                if self.backend == 'numpy':
                    ttl_firing_times = np.concatenate(self.s[module_name]).sum(axis=0)
                    self.nonfire_ratio_by_layer[module_name] = (ttl_firing_times == 0).astype(float).sum() / ttl_firing_times.size
                elif self.backend == 'torch':
                    ttl_firing_times = torch.cat(self.s[module_name]).sum(dim=0)
                    self.nonfire_ratio_by_layer[module_name] = (ttl_firing_times == 0).float().sum() / ttl_firing_times.numel()
            return self.nonfire_ratio_by_layer[module_name]