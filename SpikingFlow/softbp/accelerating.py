import torch
import torch.nn as nn
import torch.nn.functional as F


class multiply_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, spike: torch.Tensor):
        # y = x * spike
        # x乘spike，等价于将x中spike == 0的位置全部填充为0
        assert x.shape == spike.shape, print('x.shape != spike.shape')  # 禁用广播机制
        mask = torch.logical_not(spike.bool())
        if x.requires_grad and spike.requires_grad:
            ctx.save_for_backward(mask, x)
        elif x.requires_grad and not spike.requires_grad:
            ctx.save_for_backward(mask)
        elif not x.requires_grad and spike.requires_grad:
            ctx.save_for_backward(x)
        return x.masked_fill(mask, 0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        grad_spike = None
        # grad_x = grad_output * spike
        # grad_spike = grad_output * x
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_x = grad_output.masked_fill(ctx.saved_tensors[0], 0)
            grad_spike = grad_output * ctx.saved_tensors[1]
        elif ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            grad_x = grad_output.masked_fill(ctx.saved_tensors[0], 0)
        elif not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_spike = grad_output * ctx.saved_tensors[0]

        return grad_x, grad_spike


class add_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, spike: torch.Tensor):
        # y = x + spike
        # x乘spike，等价于将x中spike == 1的位置增加1
        assert x.shape == spike.shape, print('x.shape != spike.shape')  # 禁用广播机制
        mask = spike.bool()
        y = x.clone()
        y[mask] += 1
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        grad_spike = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        if ctx.needs_input_grad[1]:
            grad_spike = grad_output

        return grad_x, grad_spike


class subtract_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, spike: torch.Tensor):
        # y = x - spike
        # x乘spike，等价于将x中spike == 1的位置减去1
        assert x.shape == spike.shape, print('x.shape != spike.shape')  # 禁用广播机制
        mask = spike.bool()
        y = x.clone()
        y[mask] -= 1
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        grad_spike = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        if ctx.needs_input_grad[1]:
            grad_spike = - grad_output

        return grad_x, grad_spike

def add(x: torch.Tensor, spike: torch.Tensor):
    '''
    :param x: 任意tensor
    :param spike: 脉冲tensor。要求spike中的元素只能为0或1，且spike.shape必须与x.shape相同
    :return: x + spike

    针对与脉冲这一特殊的数据类型，进行前反向传播加速并保持数值稳定的加法运算。
    '''
    return add_spike.apply(x, spike)

def sub(x: torch.Tensor, spike: torch.Tensor):
    '''
    :param x: 任意tensor
    :param spike: 脉冲tensor。要求spike中的元素只能为0或1，且spike.shape必须与x.shape相同
    :return: x - spike

    针对与脉冲这一特殊的数据类型，进行前反向传播加速并保持数值稳定的减法运算。
    '''
    return subtract_spike.apply(x, spike)

def mul(x: torch.Tensor, spike: torch.Tensor):
    '''
    :param x: 任意tensor
    :param spike: 脉冲tensor。要求spike中的元素只能为0或1，且spike.shape必须与x.shape相同
    :return: x * spike

    针对与脉冲这一特殊的数据类型，进行前反向传播加速并保持数值稳定的乘法运算。
    '''
    return multiply_spike.apply(x, spike)




class soft_vlotage_transform_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        # v = v - spike * v_threshold
        mask = spike.bool()  # 表示释放脉冲的位置
        if spike.requires_grad:
            ctx.v_threshold = v_threshold
        ret = v.clone()
        ret[mask] -= v_threshold
        return ret  # 释放脉冲的位置，电压设置为v_reset，out-of-place操作

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_v = None
        grad_spike = None
        if ctx.needs_input_grad[0]:
            grad_v = grad_output  # 因为输出对v的梯度是全1
        if ctx.needs_input_grad[1]:
            grad_spike = - ctx.v_threshold * grad_output
        return grad_v, grad_spike, None

def soft_vlotage_transform(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
    '''
    :param v: 重置前电压
    :param spike: 释放的脉冲
    :param v_threshold: 阈值电压
    :return: 重置后的电压

    根据释放的脉冲，以soft方式重置电压，即释放脉冲后，电压会减去阈值：:math:`v = v - s \\cdot v_{threshold}`。

    该函数针对脉冲数据进行了前反向传播的加速，并能节省内存，且保持数值稳定。
    '''
    return soft_vlotage_transform_function.apply(v, spike, v_threshold)

class hard_voltage_transform_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        # v = v * (1 - spikes) + v_reset * spikes
        mask = spike.bool()  # 表示释放脉冲的位置
        if v.requires_grad and spike.requires_grad:
            ctx.save_for_backward(mask, v_reset - v)
        elif v.requires_grad and not spike.requires_grad:
            ctx.save_for_backward(mask)
        elif not v.requires_grad and spike.requires_grad:
            ctx.save_for_backward(v_reset - v)

        return v.masked_fill(mask, v_reset)  # 释放脉冲的位置，电压设置为v_reset，out-of-place操作

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_v = None
        grad_spike = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_v = grad_output.masked_fill(ctx.saved_tensors[0], 0)
            grad_spike = grad_output * ctx.saved_tensors[1]
        elif ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            grad_v = grad_output.masked_fill(ctx.saved_tensors[0], 0)
        elif not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_spike = grad_output * ctx.saved_tensors[0]

        return grad_v, grad_spike, None

def hard_voltage_transform(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
    '''
    :param v: 重置前电压
    :param spike: 释放的脉冲
    :param v_reset: 重置电压
    :return: 重置后的电压

    根据释放的脉冲，以hard方式重置电压，即释放脉冲后，电压会直接置为重置电压：:math:`v = v \\cdot (1-s) + v_{reset} \\cdot s`。

    该函数针对脉冲数据进行了前反向传播的加速，并能节省内存，且保持数值稳定。
    '''
    return hard_voltage_transform_function.apply(v, spike, v_reset)