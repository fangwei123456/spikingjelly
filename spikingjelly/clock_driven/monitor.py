import torch
import numpy as np
from torch import nn
from spikingjelly.clock_driven import neuron

from typing import Callable, Tuple, Dict, Optional

class Monitor:
    def __init__(self, net: nn.Module, device: Optional[str] = None, backend: Optional[str] = 'numpy'):
    
        super().__init__()
        self.module_dict = dict()
        for name, module in net.named_modules():
            if isinstance(module, neuron.BaseNode):
                self.module_dict[name] = module
                #setattr(module, 'monitor', self)

        # 'torch' or 'numpy'
        self.net = net
        self.backend = backend

        if self.backend == 'torch':
            self.device = torch.device(device)

    def enable(self):
        # 初始化前向时钩子的句柄
        self.handle = dict.fromkeys(self.module_dict, None)
        self.v = dict.fromkeys(self.module_dict, None)
        self.s = dict.fromkeys(self.module_dict, None)

        for name, module in self.net.named_modules():
            if isinstance(module, neuron.BaseNode):
                self.v[name] = []
                self.s[name] = []
                setattr(module, 'v_list', self.v[name])
                setattr(module, 's_list', self.s[name])
                self.handle[name] = module.register_forward_hook(self.forward_hook)
        
        self.reset()

    def disable(self):
        for name, module in self.net.named_modules():
            if isinstance(module, neuron.BaseNode):
                delattr(module, 'v_list')
                delattr(module, 's_list')
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

        for v_list in self.v.values():
            v_list.clear()
        
        for s_list in self.s.values():
            s_list.clear()
        # 数据与字典置空
        self.neuron_cnt = dict.fromkeys(self.module_dict, None)
        self.avg_firing_rate_by_layer = dict.fromkeys(self.module_dict, None)
        self.nonfire_ratio_by_layer = dict.fromkeys(self.module_dict, None)

    def get_avg_firing_rate(self, all: Optional[bool] = True, module_name: Optional[str] = None):
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


    def get_nonfire_ratio(self, all: Optional[bool] = True, module_name: Optional[str] = None):
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
                    ttl_firing_times = np.concatenate(self.s[name]).sum(axis=0)
                    self.nonfire_ratio_by_layer[module_name] = (ttl_firing_times == 0).astype(float).sum() / ttl_firing_times.size
                elif self.backend == 'torch':
                    ttl_firing_times = torch.cat(self.s[module_name]).sum(dim=0)
                    self.nonfire_ratio_by_layer[module_name] = (ttl_firing_times == 0).float().sum() / ttl_firing_times.numel()
            return self.nonfire_ratio_by_layer[module_name]