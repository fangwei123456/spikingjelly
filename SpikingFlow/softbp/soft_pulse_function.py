import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearLeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a=1, b=0.01, c=0.5):
        ctx.save_for_backward(x)
        ctx.a = a
        ctx.b = b
        ctx.c = c
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0]
            grad_x = (ctx.b * (x > ctx.c).float() + ctx.b * (x < -ctx.c).float() \
                     + ctx.a * (x >= -ctx.c).float() * (x <= ctx.c).float()) * grad_output

        return grad_x, None, None, None


def bilinear_leaky_relu(x, a=1, b=0.01, c=0.5):
    '''
    :param x: 输入数据
    :param a: -c <= x <= c 时反向传播的梯度
    :param b: x > c 或 x < -c 时反向传播的梯度
    :param c: 决定梯度区间的参数
    :return: 前向传播时候，返回 (x >= 0).float()

    双线性的脉冲发放函数。前向为

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0
        \\end{cases}

    反向为

    .. math::
        g'(x) =
        \\begin{cases}
        a, & -c \\leq x \\leq c \\\\
        b, & x < -c ~or~ x > c
        \\end{cases}

    '''
    return BilinearLeakyReLU.apply(a, b, c)

class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            s_x = torch.sigmoid(ctx.alpha * ctx.saved_tensors[0])
            grad_x = ctx.alpha * (1 - s_x) * s_x * grad_output

        return grad_x, None


def sigmoid(x, alpha=1.0):
    '''
    :param x: 输入数据
    :param alpha: 控制反向传播时梯度的平滑程度的参数
    :return: 前向传播时候，返回 (x >= 0).float()

    反向传播时使用sigmoid的梯度的脉冲发放函数。前向为

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0
        \\end{cases}

    反向为

    .. math::
        g'(x) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x)
    '''
    return Sigmoid.apply(x, alpha)

class SignSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta=1.0):
        ctx.save_for_backward(x)
        ctx.beta = beta
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0]
            grad_x = ctx.beta * (2 - ctx.beta * x * torch.tanh(ctx.beta * x / 2)) / (1 + torch.cosh(ctx.beta * x)) \
                     * grad_output

        return grad_x, None



def sign_swish(x, beta=5.0):
    '''
    :param x: 输入数据
    :param beta: 控制反向传播的参数
    :return: 前向传播时候，返回 (x >= 0).float()

    反向传播时使用swish的梯度的脉冲发放函数。前向为

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0
        \\end{cases}

    反向为

    .. math::
        g'(x) = \\frac{\\beta (2 - \\beta x \\mathrm{tanh} \\frac{\\beta x}{2})}{1 + \\mathrm{cosh}(\\beta x)}
    '''
    return SignSwish.apply(x, beta)
