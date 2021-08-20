import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# TODO 审查代码，优化，尽量用inplace操作
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
    return (x >= 0).to(x)

def check_manual_grad(primitive_function, spiking_function, eps=1e-5):
    '''
    :param primitive_function: 梯度替代函数的原函数
    :type primitive_function: callable
    :param spiking_function: 梯度替代函数
    :type spiking_function: callable
    :param eps: 最大误差
    :type eps: float

    梯度替代函数的反向传播一般是手写的，可以用此函数去检查手写梯度是否正确。

    此函数检查梯度替代函数spiking_function的反向传播，与原函数primitive_function的反向传播结果是否一致。“一致”被定义为，两者的误差不超过eps。

    示例代码：

    .. code-block:: python

        surrogate.check_manual_grad(surrogate.ATan.primitive_function, surrogate.atan.apply)
    '''
    alpha = torch.tensor(1.0, dtype=torch.float)
    x = torch.arange(-16, 16, 32 / 8192)
    x.requires_grad_(True)
    primitive_function(x, alpha).sum().backward()
    x_grad_auto = x.grad.clone()
    x.grad.zero_()
    spiking_function(x, alpha).sum().backward()
    x_grad_manual = x.grad.clone()
    assert (x_grad_manual - x_grad_auto).abs().max().item() <= eps, 'x.grad is wrong!'
    print('grad check pass')

class SurrogateFunctionBase(nn.Module):
    def __init__(self, alpha, spiking=True):
        super().__init__()
        self.spiking = spiking
        self.alpha = alpha

    def set_spiking_mode(self, spiking: bool):
        self.spiking = spiking

    def extra_repr(self):
        return f'alpha={self.alpha}, spiking={self.spiking}'

    @staticmethod
    def spiking_function(x, alpha):
        raise NotImplementedError

    @staticmethod
    def primitive_function(x, alpha):
        raise NotImplementedError

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return self.spiking_function(x, self.alpha)
        else:
            return self.primitive_function(x, self.alpha)


class MultiArgsSurrogateFunctionBase(nn.Module):
    def __init__(self, spiking: bool, *args, **kwargs):
        super().__init__()
        self.spiking = spiking

    def set_spiking_mode(self, spiking: bool):
        self.spiking = spiking


class piecewise_quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x_abs = ctx.saved_tensors[0].abs()
            mask = (x_abs > (1 / ctx.alpha))
            grad_x = (grad_output * (- (ctx.alpha ** 2) * x_abs + ctx.alpha)).masked_fill_(mask, 0)
        return grad_x, None


class PiecewiseQuadratic(SurrogateFunctionBase):
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

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseQuadratic.*
            :width: 100%

        该函数在文章 [#esser2016convolutional]_ [#STBP]_ [#LSNN]_ [#neftci2019surrogate]_ [#panda2020toward]_ 中使用。

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

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseQuadratic.*
            :width: 100%

        The function is used in [#esser2016convolutional]_ [#STBP]_ [#LSNN]_ [#neftci2019surrogate]_ [#panda2020toward]_.

        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_quadratic.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        mask0 = (x > (1.0 / alpha)).to(x)
        mask1 = (x.abs() <= (1.0 / alpha)).to(x)

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
            grad_x = ctx.alpha / 2 * (- ctx.alpha * ctx.saved_tensors[0].abs()).exp_() * grad_output

        return grad_x, None


class PiecewiseExp(SurrogateFunctionBase):
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

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseExp.*
            :width: 100%

        该函数在文章 [#SLAYER]_ [#neftci2019surrogate]_ 中使用。

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

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseExp.*
            :width: 100%

        The function is used in [#SLAYER]_ [#neftci2019surrogate]_ .
        '''
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_exp.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        mask_nonnegative = heaviside(x)
        mask_sign = mask_nonnegative * 2 - 1
        exp_x = (mask_sign * x * -alpha).exp_() / 2

        return mask_nonnegative - exp_x * mask_sign

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
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha

        return grad_x, None


class Sigmoid(SurrogateFunctionBase):
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

        .. image:: ./_static/API/clock_driven/surrogate/Sigmoid.*
            :width: 100%

        该函数在文章 [#STBP]_ [#roy2019scaling]_ [#SNNLSTM]_ [#SNU]_ 中使用。

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

        .. image:: ./_static/API/clock_driven/surrogate/Sigmoid.*
            :width: 100%

        The function is used in  [#STBP]_ [#roy2019scaling]_ [#SNNLSTM]_ [#SNU]_ .
        '''
        super().__init__(alpha, spiking)


    @staticmethod
    def spiking_function(x, alpha):
        return sigmoid.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid()

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        alpha = str(self.alpha) + 'f'
        if dtype == 'fp32':
            code = f'''
            const float sigmoid_ax = 1.0f / (1.0f + expf(- {alpha} * {x}));
            const float {y} = (1.0f - sigmoid_ax) * sigmoid_ax * {alpha};
            '''
        elif dtype == 'fp16':
            code = f'''
            const half alpha = __float2half({alpha});
            const half sigmoid_ax = __hdiv(__float2half(1.0f), __hadd(hexp(__hneg(__hmul(alpha, {x}))), __float2half(1.0f)));
            const half {y} = __hmul(__hmul(__hsub(__float2half(1.0f), sigmoid_ax), sigmoid_ax), alpha);
            '''
        else:
            raise NotImplementedError
        return code

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


class soft_sign(torch.autograd.Function):
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
            grad_x = grad_output / (2 * ctx.alpha * (1 / ctx.alpha + ctx.saved_tensors[0].abs()).pow_(2))
        return grad_x, None


class SoftSign(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        '''
        * :ref:`API in English <SoftSign.__init__-en>`
        .. _SoftSign.__init__-cn:

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数

        反向传播时使用soft sign的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + |\\alpha x|)^{2}} = \\frac{1}{2\\alpha(\\frac{1}{\\alpha} + |x|)^{2}}

        对应的原函数为

        .. math::
            g(x) = \\frac{1}{2} (\\frac{\\alpha x}{1 + |\\alpha x|} + 1)
            = \\frac{1}{2} (\\frac{x}{\\frac{1}{\\alpha} + |x|} + 1)

        .. image:: ./_static/API/clock_driven/surrogate/SoftSign.*
            :width: 100%

        该函数在文章 [#SuperSpike]_ [#neftci2019surrogate]_ 中使用。

        * :ref:`中文API <SoftSign.__init__-cn>`
        .. _SoftSign.__init__-en:

        :param alpha: parameter to control smoothness of gradient
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation

        The soft sign surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + |\\alpha x|)^{2}}

        The primitive function is defined by

        .. math::
            g(x) = \\frac{1}{2} (\\frac{\\alpha x}{1 + |\\alpha x|} + 1)

        .. image:: ./_static/API/clock_driven/surrogate/SoftSign.*
            :width: 100%

        The function is used in [#SuperSpike]_ [#neftci2019surrogate]_ .
        '''
        super().__init__(alpha, spiking)
        assert alpha > 0, 'alpha must be lager than 0'

    @staticmethod
    def spiking_function(x, alpha):
        return soft_sign.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (F.softsign(x * alpha) + 1) / 2

    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200)
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    # surrogate_function = surrogate.SoftSign(alpha=3, spiking=False)
    # y = surrogate_function(x)
    # plt.plot(x.data, y.data, label='Primitive, $\\alpha=3$')

    # surrogate_function = surrogate.SoftSign(alpha=3, spiking=True)
    # x.requires_grad_(True)
    # y = surrogate_function(x)
    # z = y.sum()
    # z.backward()
    # plt.plot(x.data, x.grad, label='Gradient, $\\alpha=3$')
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title('SoftSign surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(linestyle='--')
    # plt.show()


class atan(torch.autograd.Function):
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
            grad_x = ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * ctx.saved_tensors[0]).pow_(2)) * grad_output

        return grad_x, None

class ATan(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        '''
        * :ref:`API in English <ATan.__init__-en>`
        .. _ATan.__init__-cn:

        反向传播时使用反正切函数arc tangent的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

        对应的原函数为

        .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}

        .. image:: ./_static/API/clock_driven/surrogate/ATan.*
            :width: 100%

        * :ref:`中文API <ATan.__init__-cn>`
        .. _ATan.__init__-en:

        The arc tangent surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

        The primitive function is defined by

        .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}

        .. image:: ./_static/API/clock_driven/surrogate/ATan.*
            :width: 100%
        '''
        super().__init__(alpha, spiking)


    @staticmethod
    def spiking_function(x, alpha):
        return atan.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (math.pi / 2 * alpha * x).atan_() / math.pi + 0.5

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        alpha = str(self.alpha) + 'f'
        if dtype == 'fp32':
            code = f'''
            const float M_PI_2__alpha__x = ((float) 1.57079632679489661923) * {alpha} * {x};
            const float {y} = {alpha} / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
            '''
        elif dtype == 'fp16':
            code = f'''
            const half alpha =  __float2half({alpha});
            #if __CUDACC_VER_MAJOR__ >= 11
            const half M_PI_2__alpha__x = __hmul(__hmul(__double2half(1.57079632679489661923), {alpha}), {x});
            #else
            const half M_PI_2__alpha__x = __hmul(__hmul(__float2half((float) 1.57079632679489661923), {alpha}), {x});
            #endif
            const half {y} = __hdiv(__hdiv({alpha}, __float2half(2.0f)), __hfma(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half(1.0f)));
            '''
        else:
            raise NotImplementedError
        return code

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
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output / (1 / ctx.alpha + ctx.saved_tensors[0].abs())


        return grad_x, None


class NonzeroSignLogAbs(SurrogateFunctionBase):
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

        .. image:: ./_static/API/clock_driven/surrogate/NonzeroSignLogAbs.*
            :width: 100%

        该函数在文章  中使用。

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

        .. image:: ./_static/API/clock_driven/surrogate/NonzeroSignLogAbs.*
            :width: 100%

        The function is used in  .
        '''
        super().__init__(alpha, spiking)


    @staticmethod
    def spiking_function(x, alpha):
        return nonzero_sign_log_abs.apply(x, alpha)

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
            grad_x = grad_output * (- (ctx.saved_tensors[0] * ctx.alpha).pow_(2)).exp_() * (ctx.alpha / math.sqrt(math.pi))

        return grad_x, None


class Erf(SurrogateFunctionBase):
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

        .. image:: ./_static/API/clock_driven/surrogate/Erf.*
            :width: 100%
        
        该函数在文章 [#esser2015backpropagation]_ [#STBP]_ [#SRNN]_ 中使用。

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

        .. image:: ./_static/API/clock_driven/surrogate/Erf.*
            :width: 100%

        The function is used in [#esser2015backpropagation]_ [#STBP]_ [#SRNN]_.
        '''
        super().__init__(alpha, spiking)


    @staticmethod
    def spiking_function(x, alpha):
        return erf.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return torch.erfc_(-alpha * x) / 2

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


class piecewise_leaky_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w=1, c=0.01):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.w = w
            ctx.c = c
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            mask_width = (ctx.saved_tensors[0].abs() < ctx.w)
            mask_c = mask_width.logical_not()
            grad_x = grad_output * ctx.saved_tensors[0].masked_fill(mask_width, 1 / ctx.w).masked_fill(mask_c, ctx.c)
        return grad_x, None, None


class PiecewiseLeakyReLU(MultiArgsSurrogateFunctionBase):
    def __init__(self, w=1., c=0.01, spiking=True):
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

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseLeakyReLU.*
            :width: 100%

        该函数在文章 [#yin2017algorithm]_ [#STBP]_ [#huh2018gradient]_ [#wu2019direct]_ [#STCA]_ [#roy2019scaling]_ [#LISNN]_ [#DECOLLE]_ 中使用。

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

        .. image:: ./_static/API/clock_driven/surrogate/PiecewiseLeakyReLU.*
            :width: 100%

        The function is used in [#yin2017algorithm]_ [#STBP]_ [#huh2018gradient]_ [#wu2019direct]_ [#STCA]_ [#roy2019scaling]_ [#LISNN]_ [#DECOLLE]_.
        '''
        super().__init__(spiking)
        assert w > 0.
        self.w = w
        self.c = c
        self.spiking = spiking
        if spiking:
            self.f = self.spiking_function
        else:
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.w, self.c)

    @staticmethod
    def spiking_function(x: torch.Tensor, w, c):
        return piecewise_leaky_relu.apply(x, w, c)

    @staticmethod
    def primitive_function(x: torch.Tensor, w, c):
        mask0 = (x < -w).to(x)
        mask1 = (x > w).to(x)
        mask2 = torch.ones_like(x.data) - mask0 - mask1
        if c == 0:
            return mask2 * (x / (2 * w) + 1 / 2) + mask1
        else:
            cw = c * w
            return mask0 * (c * x + cw) + mask1 * (c * x + (- cw + 1)) \
                   + mask2 * (x / (2 * w) + 1 / 2)

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        w = str(self.w) + 'f'
        w_inv = str(1. / self.w) + 'f'
        c = str(self.c) + 'f'
        code = ''
        if dtype == 'fp32':


            code += f'const float x_abs = fabsf({x});'
            code += f'float {y};'

            code += f'if (x_abs > {w})'
            code += '{'

            code += f'{y} = {c};'

            code += '}'

            code += f'else'
            code += '{'

            code += f'{y} = {w_inv};'

            code += '}'

        elif dtype == 'fp16':

            code += f'const half x_abs = __habs({x});'
            code += f'half {y};'
            code += f'if (__hge(x_abs, __float2half({w})))'
            code += '{'

            code += f'{y} = __float2half({c});'

            code += '}'

            code += f'else'
            code += '{'

            code += f'{y} = __float2half({w_inv});'

            code += '}'
        else:
            raise NotImplementedError
        return code

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