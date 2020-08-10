import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def heaviside(x: torch.Tensor):
    '''
    * :ref:`API in English <heaviside.__init__-en>`
    .. _heaviside.__init__-cn:

    :param x: 输入tensor
    :return: 输出tensor

    heaviside阶跃函数，定义为

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0 \\\\
        \\end{cases}

    阅读 `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_ 以获得更多信息。

    * :ref:`中文API <heaviside.__init__-cn>`
    .. _heaviside.__init__-en:

    :param x: the input tensor
    :return: the output tensor

    The heaviside function, which is defined by

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0 \\\\
        \\end{cases}

    For more information, see `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_.

    '''
    return (x >= 0).to(x.dtype)

class piecewise_quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            mask_zero = (x.abs() > 1 / alpha)
            grad_x = -alpha * alpha * x.abs() + alpha
            grad_x.masked_fill_(mask_zero, 0)
            ctx.save_for_backward(grad_x)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.saved_tensors[0]
        return grad_x, None

class PiecewiseQuadratic(nn.Module):
    def __init__(self, alpha=1.0, spiking=True):
        '''
        * :ref:`API in English <PiecewiseQuadratic.__init__-en>`
        .. _PiecewiseQuadratic.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用分段二次函数的梯度（三角形函数）的脉冲发放函数。反向传播为

        .. math::
            g'(x) = 
            \\begin{cases}
            0, & |x| > \\frac{1}{\\alpha} \\\\
            -\\alpha^2|x|+\\alpha, & |x| \\leq \\frac{1}{\\alpha} 
            \\end{cases}

        对应的原函数为

        .. math::
            g(x) = 
            \\begin{cases}
            0, & x < -\\frac{1}{\\alpha} \\\\
            -\\frac{1}{2}\\alpha^2|x|x + \\alpha x + \\frac{1}{2}, & |x| \\leq \\frac{1}{\\alpha}  \\\\
            1, & x > \\frac{1}{\\alpha} \\\\
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseQuadratic.png

        * :ref:`中文API <PiecewiseQuadratic.__init__-cn>`
        .. _PiecewiseQuadratic.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The piecewise quadratic surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = 
            \\begin{cases}
            0, & |x| > \\frac{1}{\\alpha} \\\\
            -\\alpha^2|x|+\\alpha, & |x| \\leq \\frac{1}{\\alpha} 
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) = 
            \\begin{cases}
            0, & x < -\\frac{1}{\\alpha} \\\\
            -\\frac{1}{2}\\alpha^2|x|x + \\alpha x + \\frac{1}{2}, & |x| \\leq \\frac{1}{\\alpha}  \\\\
            1, & x > \\frac{1}{\\alpha} \\\\
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseQuadratic.png

        '''
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
        if spiking:
            self.f = piecewise_quadratic.apply
        else:
            self.f = self.primitive_function
    def forward(self, x):
        return self.f(x, self.alpha)
    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        mask0 = (x > 1.0 / alpha).float()
        mask1 = (x.abs() <= 1.0 / alpha).float()
        
        return mask0 + mask1 * (-(alpha ** 2) / 2 * x.square() * x.sign() + alpha * x + 0.5)

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.PiecewiseQuadratic(alpha=1.5, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=1.5$')

    # surrogate_function = surrogate.PiecewiseQuadratic(alpha=1.5, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=1.5$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('Piecewise quadratic surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()

class piecewise_leaky_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w=1, c=0.01):
        if x.requires_grad:
            mask_width = (x.abs() < w)
            mask_c = mask_width.logical_not()
            grad_x = x.masked_fill(mask_width, 1 / w).masked_fill(mask_c, c)
            ctx.save_for_backward(grad_x)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.saved_tensors[0]
        return grad_x, None, None, None

class PiecewiseLeakyReLU(nn.Module):
    def __init__(self, w=1, c=0.01, spiking=True):
        '''
        * :ref:`API in English <PiecewiseLeakyReLU.__init__-en>`
        .. _PiecewiseLeakyReLU.__init__-cn:

        :param w: ``-w <= x <= w`` 时反向传播的梯度为 ``1 / 2w``
        :param c: ``x > w`` 或 ``x < -w`` 时反向传播的梯度为 ``c``
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        分段线性的近似脉冲发放函数。梯度为

        .. math::
            g'(x) =
            \\begin{cases}
            \\frac{1}{w}, & -w \\leq x \\leq w \\\\
            c, & x < -w ~or~ x > w
            \\end{cases}

        对应的原函数为

        .. math::
            g(x) =
            \\begin{cases}
            cx + cw, & x < -w \\\\
            \\frac{1}{2w}x + \\frac{1}{2}, & -w \\leq x \\leq w \\\\
            cx - cw + 1, & x > w \\\\
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseLeakyReLU.png

        * :ref:`中文API <PiecewiseLeakyReLU.__init__-cn>`
        .. _PiecewiseLeakyReLU.__init__-en:

        :param w: when ``-w <= x <= w`` the gradient is ``1 / 2w``
        :param c: when ``x > w`` or ``x < -w`` the gradient is ``c``
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The piecewise surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) =
            \\begin{cases}
            \\frac{1}{w}, & -w \\leq x \\leq w \\\\
            c, & x < -w ~or~ x > w
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \\begin{cases}
            cx + cw, & x < -w \\\\
            \\frac{1}{2w}x + \\frac{1}{2}, & -w \\leq x \\leq w \\\\
            cx - cw + 1, & x > w
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseLeakyReLU.png
        '''
        super().__init__()
        self.w = w
        self.c = c
        self.spiking = spiking
        if spiking:
            self.f = piecewise_leaky_relu.apply
        else:
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.w, self.c)


    @staticmethod
    def primitive_function(x: torch.Tensor, w, c):
        mask0 = (x < -w).float()
        mask1 = (x > w).float()
        mask2 = torch.ones_like(x) - mask0 - mask1
        if c == 0:
            return mask2 * (x / (2 * w) + 1 / 2) + mask1
        else:
            cw = c * w
            return mask0 * (c * x + cw) + mask1 * (c * x + (- cw + 1)) \
                   + mask2 * (x / (2 * w) + 1 / 2)

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.PiecewiseLeakyReLU(w=1, c=0.1, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $w=1, c=0.1$')

    # surrogate_function = surrogate.PiecewiseLeakyReLU(w=1, c=0.1, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $w=1, c=0.1$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('PiecewiseLeakyReLU surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()

class piecewise_exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.alpha / 2 * (-ctx.alpha * ctx.saved_tensors[0].abs()).exp()
        return grad_x, None

class PiecewiseExp(nn.Module):
    def __init__(self, alpha=1.0, spiking=True):
        '''
        * :ref:`API in English <PiecewiseExp.__init__-en>`
        .. _PiecewiseExp.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用分段指数函数的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{2}e^{-\\alpha |x|}

        对应的原函数为

        .. math::
            g(x) = 
            \\begin{cases}
            \\frac{1}{2}e^{\\alpha x}, & x < 0 \\\\
            1 - \\frac{1}{2}e^{-\\alpha x}, & x \\geq 0 
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseExp.png

        * :ref:`中文API <PiecewiseExp.__init__-cn>`
        .. _PiecewiseExp.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The piecewise exponential surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2}e^{-\\alpha |x|}

        The primitive function is defined by

        .. math::
            g(x) = 
            \\begin{cases}
            \\frac{1}{2}e^{\\alpha x}, & x < 0 \\\\
            1 - \\frac{1}{2}e^{-\\alpha x}, & x \\geq 0 
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseExp.png

        '''
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
        if spiking:
            self.f = piecewise_exp.apply
        else:
            self.f = self.primitive_function
    def forward(self, x):
        return self.f(x, self.alpha)
    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        exp_x = 0.5 * (x.abs() * -alpha).exp()
        mask_positive = (x > 0).float()
        
        return mask_positive - exp_x * (mask_positive * 2 - 1)

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.PiecewiseExp(alpha=2, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=2$')

    # surrogate_function = surrogate.PiecewiseExp(alpha=2, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=2$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('Piecewise exponential surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            s_x = torch.sigmoid(ctx.alpha * ctx.saved_tensors[0])
            grad_x = grad_output * s_x * (1 - s_x) * ctx.alpha
        return grad_x, None

class Sigmoid(nn.Module):
    def __init__(self, alpha=1.0, spiking=True):
        '''
        * :ref:`API in English <Sigmoid.__init__-en>`
        .. _Sigmoid.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用sigmoid的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x)

        对应的原函数为

        .. math::
            g(x) = \\mathrm{sigmoid}(\\alpha x) = \\frac{1}{1+e^{-\\alpha x}}

        .. image:: ./_static/API/clock_driven/surrogate/Sigmoid.png

        * :ref:`中文API <Sigmoid.__init__-cn>`
        .. _Sigmoid.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The sigmoid surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x)

        The primitive function is defined by

        .. math::
            g(x) = \\mathrm{sigmoid}(\\alpha x) = \\frac{1}{1+e^{-\\alpha x}}

        .. image:: ./_static/API/clock_driven/surrogate/Sigmoid.png

        '''
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
        if spiking:
            self.f = sigmoid.apply
        else:
            self.f = self.primitive_function
    def forward(self, x):
        return self.f(x, self.alpha)
    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid()

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.Sigmoid(alpha=5, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=5$')

    # surrogate_function = surrogate.Sigmoid(alpha=5, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=5$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('Sigmoid surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()

class fast_sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, inv_alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.inv_alpha = inv_alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.inv_alpha / 2 / (ctx.inv_alpha + ctx.saved_tensors[0].abs()).square() * grad_output
        return grad_x, None

class FastSigmoid(nn.Module):
    def __init__(self, alpha=2.0, spiking=True):
        '''
        * :ref:`API in English <FastSigmoid.__init__-en>`
        .. _FastSigmoid.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用fast sigmoid的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + |\\alpha x|)^{2}} = \\frac{1}{2\\alpha(\\frac{1}{\\alpha} + |x|)^{2}}

        对应的原函数为

        .. math::
            g(x) = \\frac{1}{2} (\\frac{\\alpha x}{1 + |\\alpha x|} + 1)
            = \\frac{1}{2} (\\frac{x}{\\frac{1}{\\alpha} + |x|} + 1)

        .. image:: ./_static/API/clock_driven/surrogate/FastSigmoid.png

        * :ref:`中文API <FastSigmoid.__init__-cn>`
        .. _FastSigmoid.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The fast sigmoid surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + |\\alpha x|)^{2}}

        The primitive function is defined by

        .. math::
            g(x) = \\frac{1}{2} (\\frac{\\alpha x}{1 + |\\alpha x|} + 1)

        .. image:: ./_static/API/clock_driven/surrogate/FastSigmoid.png

        '''
        super().__init__()
        assert alpha > 0, 'alpha must be lager than 0'
        self.inv_alpha = 1 / alpha
        self.spiking = spiking
        if spiking:
            self.f = fast_sigmoid.apply
        else:
            self.f = self.primitive_function
    def forward(self, x):
        return self.f(x, self.inv_alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, inv_alpha):
        return (x / (inv_alpha + x.abs()) + 1) / 2

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.FastSigmoid(alpha=3, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=3$')

    # surrogate_function = surrogate.FastSigmoid(alpha=3, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=3$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('FastSigmoid surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()

class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, half_alpha, half_pi_alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.half_pi_alpha = half_pi_alpha
            ctx.half_alpha = half_alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.half_alpha / (1 + (ctx.half_pi_alpha * ctx.saved_tensors[0]).square())
        return grad_x, None, None

class ATan(nn.Module):
    def __init__(self, alpha=2.0, spiking=True):
        '''
        * :ref:`API in English <ATan.__init__-en>`
        .. _ATan.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用反正切函数arc tangent的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

        对应的原函数为

        .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}

        .. image:: ./_static/API/clock_driven/surrogate/ATan.png

        * :ref:`中文API <ATan.__init__-cn>`
        .. _ATan.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The arc tangent surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

        The primitive function is defined by

        .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}

        .. image:: ./_static/API/clock_driven/surrogate/ATan.png

        '''
        super().__init__()
        self.half_pi_alpha = math.pi / 2 * alpha
        self.spiking = spiking
        if spiking:
            self.coefficient = alpha / 2
            self.f = atan.apply
        else:
            self.coefficient = 1 / math.pi
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.coefficient, self.half_pi_alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, coefficient, half_pi_alpha):
        return coefficient * (half_pi_alpha * x).atan() + 0.5

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.ATan(alpha=3, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=3$')

    # surrogate_function = surrogate.ATan(alpha=3, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=3$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('ATan surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()

class nonzero_sign_log_abs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, inv_alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.inv_alpha = inv_alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output / (ctx.saved_tensors[0].abs() + ctx.inv_alpha)
        return grad_x, None

class NonzeroSignLogAbs(nn.Module):
    def __init__(self, alpha=1.0, spiking=True):
        '''
        * :ref:`API in English <LogAbs.__init__-en>`
        .. _LogAbs.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        .. warning::
            原函数的输出范围并不是(0, 1)。它的优势是反向传播的计算量特别小。

        反向传播时使用NonzeroSignLogAbs的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{1 + |\\alpha x|} = \\frac{1}{\\frac{1}{\\alpha} + |x|}

        对应的原函数为

        .. math::
            g(x) = \\mathrm{NonzeroSign}(x) \\log (|\\alpha x| + 1)

        其中

            .. math::
                \\mathrm{NonzeroSign}(x) =
                \\begin{cases}
                1, & x \\geq 0 \\\\
                -1, & x < 0 \\\\
                \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/NonzeroSignLogAbs.png

        * :ref:`中文API <LogAbs.__init__-cn>`
        .. _LogAbs.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        .. admonition:: Warning
            :class: warning

            The output range the primitive function is not (0, 1). The advantage of this function is that computation
            cost is small when backward.

        The NonzeroSignLogAbs surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{1 + |\\alpha x|} = \\frac{1}{\\frac{1}{\\alpha} + |x|}

        The primitive function is defined by

        .. math::
            g(x) = \\mathrm{NonzeroSign}(x) \\log (|\\alpha x| + 1)

        where

        .. math::
            \\mathrm{NonzeroSign}(x) =
            \\begin{cases}
            1, & x \\geq 0 \\\\
            -1, & x < 0 \\\\
            \\end{cases}

        .. image:: ./_static/API/clock_driven/surrogate/NonzeroSignLogAbs.png

        '''
        super().__init__()
        self.spiking = spiking
        if spiking:
            self.coefficient = 1 / alpha
            self.f = nonzero_sign_log_abs.apply
        else:
            self.coefficient = alpha
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.coefficient)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        # the gradient of ``(heaviside(x) * 2 - 1) * (alpha * x.abs() + 1).log()`` by autograd is wrong at ``x==0``
        mask_p = heaviside(x) * 2 - 1
        return mask_p * (alpha * mask_p * x + 1).log()

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.NonzeroSignLogAbs(alpha=1, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=1$')

    # surrogate_function = surrogate.NonzeroSignLogAbs(alpha=1, spiking=False)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=1$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('NonzeroSignLogAbs surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()

class erf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.alpha / math.sqrt(math.pi) * (-((ctx.saved_tensors[0] * ctx.alpha).square())).exp()
        return grad_x, None 

class Erf(nn.Module):
    def __init__(self, alpha=2.0, spiking=True):
        '''
        * :ref:`API in English <Erf.__init__-en>`
        .. _Erf.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用高斯误差函数(erf)的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{\\sqrt{\pi}}e^{-\\alpha^2x^2}

        对应的原函数为

        .. math::
            :nowrap:

            \\begin{split}
            g(x) &= \\frac{1}{2}(1-\\text{erf}(-\\alpha x)) \\\\
            &= \\frac{1}{2} \\text{erfc}(-\\alpha x) \\\\
            &= \\frac{1}{\\sqrt{\\pi}}\int_{-\\infty}^{\\alpha x}e^{-t^2}dt
            \\end{split}

        .. image:: ./_static/API/clock_driven/surrogate/Erf.png

        * :ref:`中文API <Erf.__init__-cn>`
        .. _Erf.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The Gaussian error (erf) surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{\\sqrt{\pi}}e^{-\\alpha^2x^2}

        The primitive function is defined by

        .. math::
            :nowrap:

            \\begin{split}
            g(x) &= \\frac{1}{2}(1-\\text{erf}(-\\alpha x)) \\\\
            &= \\frac{1}{2} \\text{erfc}(-\\alpha x) \\\\
            &= \\frac{1}{\\sqrt{\\pi}}\int_{-\\infty}^{\\alpha x}e^{-t^2}dt
            \\end{split}

        .. image:: ./_static/API/clock_driven/surrogate/Erf.png

        '''
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
        if spiking:
            self.f = erf.apply
        else:
            self.f = self.primitive_function
    def forward(self, x):
        return self.f(x, self.alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return torch.erfc(-alpha * x) / 2

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.Erf(alpha=2, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=2$')

    # surrogate_function = surrogate.Erf(alpha=2, spiking=False)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=2$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('Gaussian error surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()