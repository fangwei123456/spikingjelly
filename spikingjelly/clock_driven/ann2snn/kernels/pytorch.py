import numpy as np
import torch
import torch.nn as nn
import copy
from collections import defaultdict

def layer_reduction(model: nn.Module) -> nn.Module:
    relu_linker = {}  # 字典类型，用于通过relu层在network中的序号确定relu前参数化模块的序号
    param_module_relu_linker = {}  # 字典类型，用于通过relu前在network中的参数化模块的序号确定relu层序号
    activation_range = defaultdict(float)  # 字典类型，保存在network中的序号对应层的激活最大值（或某分位点值）

    module_len = 0
    module_list = nn.ModuleList([])
    last_parammodule_idx = 0
    for n, m in model.named_modules():
        Name = m.__class__.__name__
        # 加载激活层
        if isinstance(m,nn.Softmax):
            Name = 'ReLU'
            print(UserWarning("Replacing Softmax by ReLU."))
        if isinstance(m,nn.ReLU) or Name == "ReLU":
            module_list.append(m)
            relu_linker[module_len] = last_parammodule_idx
            param_module_relu_linker[last_parammodule_idx] = module_len
            module_len += 1
            activation_range[module_len] = -1e5
        # 加载BatchNorm层
        if isinstance(m,(nn.BatchNorm2d,nn.BatchNorm1d)):
            if isinstance(module_list[last_parammodule_idx], (nn.Conv2d,nn.Linear)):
                absorb(module_list[last_parammodule_idx], m)
            else:
                module_list.append(copy.deepcopy(m))
        # 加载有参数的层
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            module_list.append(m)
            last_parammodule_idx = module_len
            module_len += 1
        # 加载无参数层
        if isinstance(m,nn.MaxPool2d):
            module_list.append(m)
            module_len += 1
        if isinstance(m,nn.AvgPool2d):
            module_list.append(nn.AvgPool2d(kernel_size=m.kernel_size, stride=m.stride, padding=m.padding))
            module_len += 1
        # if isinstance(m,nn.Flatten):
        if m.__class__.__name__ == "Flatten":
            module_list.append(m)
            module_len += 1
    network = torch.nn.Sequential(*module_list)
    setattr(network,'param_module_relu_linker',param_module_relu_linker)
    setattr(network, 'activation_range', activation_range)
    return network

def rate_normalization(model: nn.Module, data: torch.Tensor, **kargs) -> nn.Module:
    if not hasattr(model,"activation_range") or not hasattr(model,"param_module_relu_linker"):
        raise(AttributeError("run layer_reduction first!"))
    try:
        robust_norm = kargs['robust']
    except KeyError:
        robust_norm = False
    x = data
    i = 0
    with torch.no_grad():
        for n, m in model.named_modules():
            Name = m.__class__.__name__
            if Name in ['Conv2d', 'ReLU', 'MaxPool2d', 'AvgPool2d', 'Flatten', 'Linear']:
                x = m.forward(x)
                a = x.cpu().numpy().reshape(-1)
                if robust_norm:
                    model.activation_range[i] = np.percentile(a[np.nonzero(a)], 99.9)
                else:
                    model.activation_range[i] = np.max(a)
                i += 1
    i = 0
    last_lambda = 1.0
    for n, m in model.named_modules():
        Name = m.__class__.__name__
        if Name in ['Conv2d', 'ReLU', 'MaxPool2d', 'AvgPool2d', 'Flatten', 'Linear']:
            if Name in ['Conv2d', 'Linear']:
                relu_output_layer = model.param_module_relu_linker[i]
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.data = m.weight.data * last_lambda / model.activation_range[relu_output_layer]
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data = m.bias.data / model.activation_range[relu_output_layer]
                last_lambda = model.activation_range[relu_output_layer]
            i += 1
    return model

def save_model(model: nn.Module, f):
    if isinstance(f,str):
        torch.save(model,f)
    return

def absorb(param_module, bn_module):
    if_2d = len(param_module.weight.size()) == 4  # 判断是否为BatchNorm2d
    bn_std = torch.sqrt(bn_module.running_var.data + bn_module.eps)
    if not if_2d:
        if param_module.bias is not None:
            param_module.weight.data = param_module.weight.data * bn_module.weight.data.view(-1, 1) / bn_std.view(
                -1,
                1)
            param_module.bias.data = (param_module.bias.data - bn_module.running_mean.data.view(
                -1)) * bn_module.weight.data.view(-1) / bn_std.view(
                -1) + bn_module.bias.data.view(-1)
        else:
            param_module.weight.data = param_module.weight.data * bn_module.weight.data.view(-1, 1) / bn_std.view(
                -1,
                1)
            param_module.bias.data = (torch.zeros_like(
                bn_module.running_mean.data.view(-1)) - bn_module.running_mean.data.view(
                -1)) * bn_module.weight.data.view(-1) / bn_std.view(-1) + bn_module.bias.data.view(-1)
    else:
        if param_module.bias is not None:
            param_module.weight.data = param_module.weight.data * bn_module.weight.data.view(-1, 1, 1,
                                                                                             1) / bn_std.view(-1, 1,
                                                                                                              1, 1)
            param_module.bias.data = (param_module.bias.data - bn_module.running_mean.data.view(
                -1)) * bn_module.weight.data.view(-1) / bn_std.view(
                -1) + bn_module.bias.data.view(-1)
        else:
            param_module.weight.data = param_module.weight.data * bn_module.weight.data.view(-1, 1, 1,
                                                                                             1) / bn_std.view(-1, 1,
                                                                                                              1, 1)
            param_module.bias.data = (torch.zeros_like(
                bn_module.running_mean.data.view(-1)) - bn_module.running_mean.data.view(
                -1)) * bn_module.weight.data.view(-1) / bn_std.view(-1) + bn_module.bias.data.view(-1)
    return param_module