import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cuda_kernel.auto_cuda import cfunction

tab4_str = "\t\t\t\t"  # used for aligning code
curly_bracket_l = "{"
curly_bracket_r = "}"


def heaviside(x: torch.Tensor):
    r"""
    **API Language:**
    :ref:`中文 <heaviside-cn>` | :ref:`English <heaviside-en>`

    ----

    .. _heaviside-cn:

    * **中文**

    heaviside阶跃函数，定义为

    .. math::
        g(x) =
        \begin{cases}
        1, & x \geq 0 \\
        0, & x < 0 \\
        \end{cases}

    阅读 `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_ 以获得更多信息。

    :param x: 输入tensor
    :type x: torch.Tensor
    :return: 输出tensor
    :rtype: torch.Tensor

    ----

    .. _heaviside-en:

    * **English**

    The heaviside function, which is defined by

    .. math::
        g(x) =
        \begin{cases}
        1, & x \geq 0 \\
        0, & x < 0 \\
        \end{cases}

    For more information, see `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_.

    :param x: the input tensor
    :type x: torch.Tensor
    :return: the output tensor
    :rtype: torch.Tensor
    """
    return (x >= 0).to(x)


def check_manual_grad(primitive_function, spiking_function, *args, **kwargs):
    r"""
    **API Language:**
    :ref:`中文 <check_manual_grad-cn>` | :ref:`English <check_manual_grad-en>`

    ----

    .. _check_manual_grad-cn:

    * **中文**

    梯度替代函数的反向传播一般是手写的，可以用此函数去检查手写梯度是否正确。

    此函数检查梯度替代函数spiking_function的反向传播，与原函数primitive_function的反向传播结果是否一致。
    "一致"被定义为，两者的误差不超过eps。

    :param primitive_function: 梯度替代函数的原函数
    :type primitive_function: callable

    :param spiking_function: 梯度替代函数
    :type spiking_function: callable

    ----

    .. _check_manual_grad-en:

    * **English**

    The manual gradient of surrogate gradient functions is usually written by hand, and this function can be used to check if the manual gradient is correct.

    This function checks if the backward pass of surrogate function spiking_function is consistent with the backward pass of the primitive function primitive_function.
    "Consistency" is defined as the error between the two not exceeding eps.

    :param primitive_function: the primitive function of surrogate gradient
    :type primitive_function: callable

    :param spiking_function: the surrogate function
    :type spiking_function: callable

    ----

    * **代码示例 | Example**

    .. code-block:: python

        def s2nn_apply(x, alpha, beta):
            return surrogate.s2nn.apply(x, alpha, beta)


        surrogate.check_manual_grad(
            surrogate.S2NN.primitive_function, s2nn_apply, alpha=4.0, beta=1.0
        )
    :return: 无返回值，直接打印对比结果
    :rtype: None
    """
    x = torch.arange(-2, 2, 32 / 8192)
    # x = torch.as_tensor([-1., 0., 1.])
    x.requires_grad_(True)
    primitive_function(x, *args, **kwargs).sum().backward()
    x_grad_auto = x.grad.clone()
    x.grad.zero_()
    spiking_function(x, *args, **kwargs).sum().backward()
    x_grad_manual = x.grad.clone()
    print("auto   grad", x_grad_auto)
    print("manual grad", x_grad_manual)
    abs_error = (x_grad_manual - x_grad_auto).abs()
    idx = abs_error.argmax()
    print("max error", abs_error[idx], "occurs at")
    print(f"x[{idx}] = {x[idx]}")
    print("auto   grad", x_grad_auto[idx])
    print("manual grad", x_grad_manual[idx])


def check_cuda_grad(neu, surrogate_function, device, *args, **kwargs):
    r"""
    **API Language:**
    :ref:`中文 <check_cuda_grad-cn>` | :ref:`English <check_cuda_grad-en>`

    ----

    .. _check_cuda_grad-cn:

    * **中文**

    检查CUDA（CuPy）后端的梯度是否正确。将CuPy后端的梯度与PyTorch后端的梯度进行比较。

    :param neu: 神经元类
    :type neu: type
    :param surrogate_function: 替代函数类（未实例化）
    :type surrogate_function: type
    :param device: 设备
    :type device: str or torch.device
    :param args: 传递给替代函数的位置参数
    :type args: tuple
    :param kwargs: 传递给替代函数的关键字参数
    :type kwargs: dict
    :return: 无返回值，直接打印对比结果
    :rtype: None

    .. admonition:: Example
        :class: tip

        .. code-block:: python

            check_cuda_grad(
                neuron.IFNode, surrogate.S2NN, device="cuda:1", alpha=4.0, beta=1.0
            )

    ----

    .. _check_cuda_grad-en:

    * **English**

    Check whether the CUDA (CuPy) backend gradient is correct, by comparing the gradient of CuPy backend
    and the PyTorch backend.

    :param neu: neuron class
    :type neu: type
    :param surrogate_function: surrogate function class (not instantiated)
    :type surrogate_function: type
    :param device: device
    :type device: str or torch.device
    :param args: positional arguments to pass to the surrogate function
    :type args: tuple
    :param kwargs: keyword arguments to pass to the surrogate function
    :type kwargs: dict
    :return: no return value, directly prints comparison results
    :rtype: None

    .. admonition:: Example
        :class: tip

        .. code-block:: python

            check_cuda_grad(
                neuron.IFNode, surrogate.S2NN, device="cuda:1", alpha=4.0, beta=1.0
            )
    """
    # check_cuda_grad(neuron.IFNode, surrogate.S2NN, device='cuda:1', alpha=4., beta=1.)
    for dtype in [torch.float, torch.half]:
        print(dtype)
        net = neu(surrogate_function=surrogate_function(*args, **kwargs), step_mode="m")
        net.to(device)
        x = torch.arange(-2, 2, 32 / 8192, device=device, dtype=dtype)
        x.requires_grad_(True)
        net.backend = "torch"
        net(x.unsqueeze(0)).sum().backward()
        x_grad_py = x.grad.clone()
        x.grad.zero_()
        net.reset()
        net.backend = "cupy"
        net(x.unsqueeze(0)).sum().backward()

        x_grad_cp = x.grad.clone()
        # print('python grad', x_grad_py)
        # print('cupy   grad', x_grad_cp)
        abs_error = (x_grad_cp - x_grad_py).abs()
        idx = abs_error.argmax()
        print("max error", abs_error[idx], "occurs at")
        print(f"x[{idx}] = {x[idx]}")
        print("python grad", x_grad_py[idx])
        print("cupy   grad", x_grad_cp[idx])


def plot_surrogate_function(surrogate_function):
    r"""
    **API Language:**
    :ref:`中文 <plot_surrogate_function-cn>` | :ref:`English <plot_surrogate_function-en>`

    ----

    .. _plot_surrogate_function-cn:

    * **中文**

    绘制替代函数的原函数和梯度图像。同时绘制Heaviside阶跃函数作为参考。

    :param surrogate_function: 替代函数模块的实例
    :type surrogate_function: SurrogateFunctionBase
    :return: 无返回值，直接保存并显示图像
    :rtype: None

    ----

    .. _plot_surrogate_function-en:

    * **English**

    Plot the primitive function and gradient of the surrogate function. Also plots the Heaviside step function
    as a reference.

    :param surrogate_function: an instance of surrogate function module
    :type surrogate_function: SurrogateFunctionBase
    :return: no return value, directly saves and shows the figure
    :rtype: None
    """
    import matplotlib.pyplot as plt
    import scienceplots  # noqa

    W, H = plt.rcParams["figure.figsize"]

    plt.style.use(["science", "muted", "grid"])
    plt.figure(dpi=200, figsize=(W, H))
    x = torch.arange(-2.5, 2.5, 0.001)
    plt.plot(x.data, heaviside(x), label="Heaviside", linestyle="-.")

    surrogate_function.set_spiking_mode(False)
    y = surrogate_function(x)
    plt.plot(x.data, y.data, label="Primitive")

    surrogate_function.set_spiking_mode(True)
    x.requires_grad_(True)
    y = surrogate_function(x)
    z = y.sum()
    z.backward()
    plt.plot(x.data, x.grad, label="Gradient")

    plt.xlim(-2, 2)
    plt.legend()
    plt.title(f"{surrogate_function.__class__.__name__} surrogate function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(linestyle="--")
    plt.savefig(f"./{surrogate_function.__class__.__name__}.pdf", bbox_inches="tight")
    plt.savefig(f"./{surrogate_function.__class__.__name__}.svg", bbox_inches="tight")
    plt.show()


class SurrogateFunctionBase(nn.Module):
    r"""
    **API Language:**
    :ref:`中文 <SurrogateFunctionBase-cn>` | :ref:`English <SurrogateFunctionBase-en>`

    ----

    .. _SurrogateFunctionBase-cn:

    * **中文**

    所有替代函数模块的基类。提供脉冲发放模式（spiking mode）和常规模式之间的切换，
    以及CUDA代码生成的基础框架。

    :param spiking: 是否输出脉冲。默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。
        若为 ``False`` 则不使用替代梯度，前向传播时使用原函数。
    :type spiking: bool
    :return: ``SurrogateFunctionBase`` 实例
    :rtype: SurrogateFunctionBase

    ----

    .. _SurrogateFunctionBase-en:

    * **English**

    Base class for all surrogate function modules. Provides switching between spiking mode and
    regular mode, as well as the basic framework for CUDA code generation.

    :param spiking: whether to output spikes. Default is ``True`` which means using ``heaviside`` in forward
        propagation and using surrogate gradient in backward propagation. If ``False``, no surrogate gradient
        is used, and the primitive function is used in forward propagation.
    :type spiking: bool
    :return: a ``SurrogateFunctionBase`` instance
    :rtype: SurrogateFunctionBase
    """

    def __init__(self, spiking=True, **kwargs):
        super().__init__()
        self.spiking = spiking
        self._sg_param_names = tuple(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_spiking_mode(self, spiking: bool):
        self.spiking = spiking

    @property
    def _sg_params(self) -> dict:
        return {k: getattr(self, k) for k in self._sg_param_names}

    def extra_repr(self):
        parts = [f"{k}={v}" for k, v in self._sg_params.items()]
        parts.append(f"spiking={self.spiking}")
        return ", ".join(parts)

    @staticmethod
    def spiking_function(x, **kwargs):
        raise NotImplementedError

    @staticmethod
    def primitive_function(x, **kwargs):
        raise NotImplementedError

    def cuda_code(self, x: str, y: str, dtype="fp32"):
        raise NotImplementedError

    def cuda_code_start_comments(self):
        return f"// start: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code"

    def cuda_code_end_comments(self):
        return f"// end: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code"

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return self.spiking_function(x, **self._sg_params)
        else:
            return self.primitive_function(x, **self._sg_params)

    def cuda_codes(self, y: str, x: str, dtype: str):
        # new version
        raise NotImplementedError


def piecewise_quadratic_backward(
    grad_output: torch.Tensor, x: torch.Tensor, alpha: float
):
    r"""
    **API Language:**
    :ref:`中文 <piecewise_quadratic_backward-cn>` | :ref:`English <piecewise_quadratic_backward-en>`

    ----

    .. _piecewise_quadratic_backward-cn:

    * **中文**

    PiecewiseQuadratic 替代函数的梯度计算。 :math:`g'(x) = \begin{cases} 0, & |x| > \frac{1}{\alpha} \\\\ -\alpha^2|x|+\alpha, & |x| \leq \frac{1}{\alpha} \\end{cases}`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 控制函数形状的参数
    :type alpha: float
    :return: 梯度张量, None
    :rtype: tuple

    ----

    .. _piecewise_quadratic_backward-en:

    * **English**

    Gradient computation for the PiecewiseQuadratic surrogate function.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: parameter to control the shape of the function
    :type alpha: float
    :return: gradient tensor, None
    :rtype: tuple
    """
    x_abs = x.abs()
    mask = x_abs > (1 / alpha)
    grad_x = (grad_output * (-(alpha**2) * x_abs + alpha)).masked_fill_(mask, 0)
    return grad_x, None


class piecewise_quadratic(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <piecewise_quadratic-cn>` | :ref:`English <piecewise_quadratic-en>`

    ----

    .. _piecewise_quadratic-cn:

    * **中文**

    PiecewiseQuadratic 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 piecewise_quadratic_backward 自定义梯度。

    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _piecewise_quadratic-en:

    * **English**

    The torch.autograd.Function wrapper for the PiecewiseQuadratic surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by piecewise_quadratic_backward.

    :return: spike tensor with the same shape as input x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, alpha):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_quadratic_backward(
            grad_output, ctx.saved_tensors[0], ctx.alpha
        )


class PiecewiseQuadratic(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <PiecewiseQuadratic.__init__-cn>` | :ref:`English <PiecewiseQuadratic.__init__-en>`

        ----

        .. _PiecewiseQuadratic.__init__-cn:

        * **中文**


        反向传播时使用分段二次函数的梯度（三角形函数）的脉冲发放函数。反向传播为

        .. math::
            g'(x) =
            \begin{cases}
            0, & |x| > \frac{1}{\alpha} \\
            -\alpha^2|x|+\alpha, & |x| \leq \frac{1}{\alpha}
            \end{cases}

        对应的原函数为

        .. math::
            g(x) =
            \begin{cases}
            0, & x < -\frac{1}{\alpha} \\
            -\frac{1}{2}\alpha^2|x|x + \alpha x + \frac{1}{2}, & |x| \leq \frac{1}{\alpha}  \\
            1, & x > \frac{1}{\alpha} \\
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseQuadratic.*
            :width: 100%

        该函数在文章 [#esser2016convolutional]_ [#STBP]_ [#LSNN]_ [#neftci2019surrogate]_ [#panda2020toward]_ 中使用。

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :type alpha: float

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。
            若为 ``False`` 则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _PiecewiseQuadratic.__init__-en:

        * **English**

        The piecewise quadratic surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) =
            \begin{cases}
            0, & |x| > \frac{1}{\alpha} \\
            -\alpha^2|x|+\alpha, & |x| \leq \frac{1}{\alpha}
            \end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \begin{cases}
            0, & x < -\frac{1}{\alpha} \\
            -\frac{1}{2}\alpha^2|x|x + \alpha x + \frac{1}{2}, & |x| \leq \frac{1}{\alpha}  \\
            1, & x > \frac{1}{\alpha} \\
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseQuadratic.*
            :width: 100%

        The function is used in [#esser2016convolutional]_ [#STBP]_ [#LSNN]_ [#neftci2019surrogate]_ [#panda2020toward]_.

        :param alpha: parameter to control smoothness of gradient
        :type alpha: float

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: ``heaviside(x)`` 或原函数的结果，形状与 ``x`` 相同
        :rtype: torch.Tensor
        """
        super().__init__(spiking=spiking, alpha=alpha)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_quadratic.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        mask0 = (x > (1.0 / alpha)).to(x)
        mask1 = (x.abs() <= (1.0 / alpha)).to(x)

        return mask0 + mask1 * (
            -(alpha**2) / 2 * x.square() * x.sign() + alpha * x + 0.5
        )

    @staticmethod
    def backward(grad_output, x, alpha):
        return piecewise_quadratic_backward(grad_output, x, alpha)[0]


def piecewise_exp_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    r"""
    **API Language:**
    :ref:`中文 <piecewise_exp_backward-cn>` | :ref:`English <piecewise_exp_backward-en>`

    ----

    .. _piecewise_exp_backward-cn:

    * **中文**

    PiecewiseExp 替代函数的梯度计算。 :math:`g'(x) = \frac{\alpha}{2}e^{-\alpha |x|}`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 控制函数形状的参数
    :type alpha: float
    :return: 梯度张量, None
    :rtype: tuple

    ----

    .. _piecewise_exp_backward-en:

    * **English**

    Gradient computation for the PiecewiseExp surrogate function.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: parameter to control the shape of the function
    :type alpha: float
    :return: gradient tensor, None
    :rtype: tuple
    """
    return alpha / 2 * (-alpha * x.abs()).exp_() * grad_output, None


class piecewise_exp(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <piecewise_exp-cn>` | :ref:`English <piecewise_exp-en>`

    ----

    .. _piecewise_exp-cn:

    * **中文**

    PiecewiseExp 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 piecewise_exp_backward 自定义梯度。

    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _piecewise_exp-en:

    * **English**

    The torch.autograd.Function wrapper for the PiecewiseExp surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by piecewise_exp_backward.

    :return: spike tensor with the same shape as input x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, alpha):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_exp_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class PiecewiseExp(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <PiecewiseExp.__init__-cn>` | :ref:`English <PiecewiseExp.__init__-en>`

        ----

        .. _PiecewiseExp.__init__-cn:

        * **中文**

        反向传播时使用分段指数函数的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \frac{\alpha}{2}e^{-\alpha |x|}

        对应的原函数为

        .. math::
            g(x) =
            \begin{cases}
            \frac{1}{2}e^{\alpha x}, & x < 0 \\
            1 - \frac{1}{2}e^{-\alpha x}, & x \geq 0
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseExp.*
            :width: 100%

        该函数在文章 [#SLAYER]_ [#neftci2019surrogate]_ 中使用。

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :type alpha: float

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _PiecewiseExp.__init__-en:

        * **English**

        The piecewise exponential surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \frac{\alpha}{2}e^{-\alpha |x|}

        The primitive function is defined by

        .. math::
            g(x) =
            \begin{cases}
            \frac{1}{2}e^{\alpha x}, & x < 0 \\
            1 - \frac{1}{2}e^{-\alpha x}, & x \geq 0
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseExp.*
            :width: 100%

        The function is used in [#SLAYER]_ [#neftci2019surrogate]_ .

        :param alpha: parameter to control smoothness of gradient
        :type alpha: float

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: ``heaviside(x)`` 或原函数的结果，形状与 ``x`` 相同
        :rtype: torch.Tensor
        """
        super().__init__(spiking=spiking, alpha=alpha)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_exp.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        mask_nonnegative = heaviside(x)
        mask_sign = mask_nonnegative * 2.0 - 1.0
        exp_x = (mask_sign * x * -alpha).exp_() / 2.0
        return mask_nonnegative - exp_x * mask_sign

    @staticmethod
    def backward(grad_output, x, alpha):
        return piecewise_exp_backward(grad_output, x, alpha)[0]


def sigmoid_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    r"""
    **API Language:**
    :ref:`中文 <sigmoid_backward-cn>` | :ref:`English <sigmoid_backward-en>`

    ----

    .. _sigmoid_backward-cn:

    * **中文**

    Sigmoid 替代函数的梯度计算。 :math:`g'(x) = \alpha \cdot (1 - \mathrm{sigmoid}(\alpha x)) \cdot \mathrm{sigmoid}(\alpha x)`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 控制函数形状的参数
    :type alpha: float
    :return: 梯度张量, None
    :rtype: tuple

    ----

    .. _sigmoid_backward-en:

    * **English**

    Gradient computation for the Sigmoid surrogate function.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: parameter to control the shape of the function
    :type alpha: float
    :return: gradient tensor, None
    :rtype: tuple
    """
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1.0 - sgax) * sgax * alpha, None


class sigmoid(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <sigmoid-cn>` | :ref:`English <sigmoid-en>`

    ----

    .. _sigmoid-cn:

    * **中文**

    Sigmoid 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 sigmoid_backward 自定义梯度。

    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _sigmoid-en:

    * **English**

    The torch.autograd.Function wrapper for the Sigmoid surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by sigmoid_backward.

    :return: spike tensor with the same shape as input x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, alpha):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha

    @staticmethod
    def backward(ctx, grad_output):
        return sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Sigmoid(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <Sigmoid.__init__-cn>` | :ref:`English <Sigmoid.__init__-en>`

        ----

        .. _Sigmoid.__init__-cn:

        * **中文**

        反向传播时使用sigmoid的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \alpha * (1 - \mathrm{sigmoid} (\alpha x)) \mathrm{sigmoid} (\alpha x)

        对应的原函数为

        .. math::
            g(x) = \mathrm{sigmoid}(\alpha x) = \frac{1}{1+e^{-\alpha x}}

        .. image:: ../_static/API/activation_based/surrogate/Sigmoid.*
            :width: 100%

        该函数在文章 [#STBP]_ [#roy2019scaling]_ [#SNNLSTM]_ [#SNU]_ 中使用。

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :type alpha: float

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _Sigmoid.__init__-en:

        * **English**

        The sigmoid surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \alpha * (1 - \mathrm{sigmoid} (\alpha x)) \mathrm{sigmoid} (\alpha x)

        The primitive function is defined by

        .. math::
            g(x) = \mathrm{sigmoid}(\alpha x) = \frac{1}{1+e^{-\alpha x}}

        .. image:: ../_static/API/activation_based/surrogate/Sigmoid.*
            :width: 100%

        The function is used in  [#STBP]_ [#roy2019scaling]_ [#SNNLSTM]_ [#SNU]_ .

        :param alpha: parameter to control smoothness of gradient
        :type alpha: float

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: ``heaviside(x)`` 或原函数的结果，形状与 ``x`` 相同
        :rtype: torch.Tensor
        """
        super().__init__(spiking=spiking, alpha=alpha)

    @staticmethod
    def spiking_function(x: torch.Tensor, alpha: float):
        return sigmoid.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        return (x * alpha).sigmoid()

    @staticmethod
    def backward(grad_output, x, alpha):
        return sigmoid_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype="fp32") -> str:
        sg_name = "sg_" + self._get_name()
        alpha = str(self.alpha) + "f"
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """

        if dtype == "fp32":
            code += f"""
            {tab4_str}const float {sg_name}_sigmoid_ax = 1.0f / (1.0f + expf(- {alpha} * {x}));
            {tab4_str}const float {y} = (1.0f - {sg_name}_sigmoid_ax) * {sg_name}_sigmoid_ax * {alpha};
            """
        elif dtype == "fp16":
            code += f"""
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2({sg_name}_alpha, {x}))), __float2half2_rn(1.0f)));
            {tab4_str}const half2 {y} = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_sigmoid_ax), {sg_name}_sigmoid_ax), {sg_name}_alpha);
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.sigmoid_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)


def soft_sign_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    r"""
    **API Language:**
    :ref:`中文 <soft_sign_backward-cn>` | :ref:`English <soft_sign_backward-en>`

    ----

    .. _soft_sign_backward-cn:

    * **中文**

    Soft Sign 替代函数的梯度计算。 :math:`\frac{\partial \text{SoftSign}}{\partial x} = \frac{1}{2\alpha (1 + |x/\alpha|)^2}`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 控制函数形状的参数
    :type alpha: float
    :return: 梯度张量, None
    :rtype: tuple

    ----

    .. _soft_sign_backward-en:

    * **English**

    Gradient computation for the Soft Sign surrogate function.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: parameter to control the shape of the function
    :type alpha: float
    :return: gradient tensor, None
    :rtype: tuple
    """
    return grad_output / (2 * alpha * (1 / alpha + x.abs()).pow_(2)), None


class soft_sign(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <soft_sign-cn>` | :ref:`English <soft_sign-en>`

    ----

    .. _soft_sign-cn:

    * **中文**

    SoftSign 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 soft_sign_backward 自定义梯度。

    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _soft_sign-en:

    * **English**

    The torch.autograd.Function wrapper for the SoftSign surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by soft_sign_backward.

    :return: spike tensor with the same shape as input x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, alpha):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha

    @staticmethod
    def backward(ctx, grad_output):
        return soft_sign_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class SoftSign(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <SoftSign.__init__-cn>` | :ref:`English <SoftSign.__init__-en>`

        ----

        .. _SoftSign.__init__-cn:

        * **中文**

        反向传播时使用soft sign的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \frac{\alpha}{2(1 + |\alpha x|)^{2}} = \frac{1}{2\alpha(\frac{1}{\alpha} + |x|)^{2}}

        对应的原函数为

        .. math::
            g(x) = \frac{1}{2} (\frac{\alpha x}{1 + |\alpha x|} + 1)
            = \frac{1}{2} (\frac{x}{\frac{1}{\alpha} + |x|} + 1)

        .. image:: ../_static/API/activation_based/surrogate/SoftSign.*
            :width: 100%

        该函数在文章 [#SuperSpike]_ [#neftci2019surrogate]_ 中使用。

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :type alpha: float

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _SoftSign.__init__-en:

        * **English**

        The soft sign surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \frac{\alpha}{2(1 + |\alpha x|)^{2}}

        The primitive function is defined by

        .. math::
            g(x) = \frac{1}{2} (\frac{\alpha x}{1 + |\alpha x|} + 1)

        .. image:: ../_static/API/activation_based/surrogate/SoftSign.*
            :width: 100%

        The function is used in [#SuperSpike]_ [#neftci2019surrogate]_.

        :param alpha: parameter to control smoothness of gradient
        :type alpha: float

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: ``heaviside(x)`` 或原函数的结果，形状与 ``x`` 相同
        :rtype: torch.Tensor
        """
        super().__init__(spiking=spiking, alpha=alpha)
        assert alpha > 0, "alpha must be lager than 0"

    @staticmethod
    def spiking_function(x, alpha):
        return soft_sign.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        return (F.softsign(x * alpha) + 1.0) / 2.0

    @staticmethod
    def backward(grad_output, x, alpha):
        return soft_sign_backward(grad_output, x, alpha)[0]


def super_spike_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    r"""
    **API Language:**
    :ref:`中文 <super_spike_backward-cn>` | :ref:`English <super_spike_backward-en>`

    ----

    .. _super_spike_backward-cn:

    * **中文**

    SuperSpike 替代函数的梯度计算。 :math:`g'(x) = \frac{\alpha}{(1 + |x|)^{2}}`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 控制函数形状的参数
    :type alpha: float
    :return: 梯度张量, None
    :rtype: tuple

    ----

    .. _super_spike_backward-en:

    * **English**

    Gradient computation for the SuperSpike surrogate function.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: parameter to control the shape of the function
    :type alpha: float
    :return: gradient tensor, None
    :rtype: tuple
    """
    return alpha * grad_output / torch.pow(torch.abs(x) + 1.0, 2), None


class super_spike(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <super_spike-cn>` | :ref:`English <super_spike-en>`

    ----

    .. _super_spike-cn:

    * **中文**

    SuperSpike 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 super_spike_backward 自定义梯度。

    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _super_spike-en:

    * **English**

    The torch.autograd.Function wrapper for the SuperSpike surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by super_spike_backward.

    :return: spike tensor with the same shape as input x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, alpha):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha

    @staticmethod
    def backward(ctx, grad_output):
        return super_spike_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class SuperSpike(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <SuperSpike.__init__-cn>` | :ref:`English <SuperSpike.__init__-en>`

        ----

        .. _SuperSpike.__init__-cn:

        * **中文**

        `SuperSpike: Supervised learning in multi-layer spiking neural networks <https://arxiv.org/abs/1705.11146>`_ 提出的反向传播时使用SuperSpike的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \frac{\alpha}{(1 + (|x|))^{2}}

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :type alpha: float

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _SuperSpike.__init__-en:

        * **English**

        The SuperSpike surrogate spiking function proposed by `SuperSpike: Supervised learning in multi-layer spiking neural networks <https://arxiv.org/abs/1705.11146>`_. The gradient is defined by

        .. math::
            g'(x) = \frac{\alpha}{(1 + (|x|))^{2}}

        :param alpha: parameter to control smoothness of gradient
        :type alpha: float

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: ``heaviside(x)`` 或原函数的结果，形状与 ``x`` 相同
        :rtype: torch.Tensor
        """
        super().__init__(spiking=spiking, alpha=alpha)

    @staticmethod
    def spiking_function(x, alpha):
        return super_spike.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        raise NotImplementedError

    @staticmethod
    def backward(grad_output, x, alpha):
        return super_spike_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype="fp32"):
        raise NotImplementedError

    def cuda_codes(self, y: str, x: str, dtype: str):
        raise NotImplementedError


def atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    r"""
    **API Language:**
    :ref:`中文 <atan_backward-cn>` | :ref:`English <atan_backward-en>`

    ----

    .. _atan_backward-cn:

    * **中文**

    ATan 替代函数的梯度计算。 :math:`g'(x) = \frac{\alpha}{2(1 + (\frac{\pi}{2}\alpha x)^{2})}`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 控制函数形状的参数
    :type alpha: float
    :return: 梯度张量, None
    :rtype: tuple

    ----

    .. _atan_backward-en:

    * **English**

    Gradient computation for the ATan surrogate function.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: parameter to control the shape of the function
    :type alpha: float
    :return: gradient tensor, None
    :rtype: tuple
    """
    a = alpha / 2
    ax = math.pi * a * x
    return a / (1 + ax * ax) * grad_output, None


class atan(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <atan-cn>` | :ref:`English <atan-en>`

    ----

    .. _atan-cn:

    * **中文**

    ATan 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 atan_backward 自定义梯度。

    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _atan-en:

    * **English**

    The torch.autograd.Function wrapper for the ATan surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by atan_backward.

    :return: spike tensor with the same shape as input x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, alpha):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha

    @staticmethod
    def backward(ctx, grad_output):
        return atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class ATan(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <ATan.__init__-cn>` | :ref:`English <ATan.__init__-en>`

        ----

        .. _ATan.__init__-cn:

        * **中文**

        反向传播时使用反正切函数arc tangent的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \frac{\alpha}{2(1 + (\frac{\pi}{2}\alpha x)^{2})}

        对应的原函数为

        .. math::
            g(x) = \frac{1}{\pi} \arctan(\frac{\pi}{2}\alpha x) + \frac{1}{2}

        .. image:: ../_static/API/activation_based/surrogate/ATan.*
            :width: 100%

        该函数在文章 [#Huh2018]_ [#huh2018gradient]_ 中使用。

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :type alpha: float

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _ATan.__init__-en:

        * **English**

        The arc tangent surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \frac{\alpha}{2(1 + (\frac{\pi}{2}\alpha x)^{2})}

        The primitive function is defined by

        .. math::
            g(x) = \frac{1}{\pi} \arctan(\frac{\pi}{2}\alpha x) + \frac{1}{2}

        .. image:: ../_static/API/activation_based/surrogate/ATan.*
            :width: 100%

        The function is used in [#Huh2018]_ [#huh2018gradient]_.

        :param alpha: parameter to control smoothness of gradient
        :type alpha: float

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: ``heaviside(x)`` 或原函数的结果，形状与 ``x`` 相同
        :rtype: torch.Tensor
        """
        super().__init__(spiking=spiking, alpha=alpha)

    @staticmethod
    def spiking_function(x, alpha):
        return atan.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        return (math.pi / 2 * alpha * x).atan_() / math.pi + 0.5

    @staticmethod
    def backward(grad_output, x, alpha):
        return atan_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype="fp32"):
        sg_name = "sg_" + self._get_name()
        alpha = str(self.alpha) + "f"
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """
        if dtype == "fp32":
            code += f"""
            {tab4_str}const float {sg_name}_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * {alpha} * {x};
            {tab4_str}const float {y} = {alpha} / 2.0f / (1.0f + {sg_name}_M_PI_2__alpha__x * {sg_name}_M_PI_2__alpha__x);
            """
        elif dtype == "fp16":
            code += f"""
            {tab4_str}const half2 {sg_name}_alpha =  __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), {sg_name}_alpha), {x});
            {tab4_str}const half2 {y} = __h2div(__h2div({sg_name}_alpha, __float2half2_rn(2.0f)), __hfma2({sg_name}_M_PI_2__alpha__x, {sg_name}_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.atan_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)


def nonzero_sign_log_abs_backward(
    grad_output: torch.Tensor, x: torch.Tensor, alpha: float
):
    r"""
    **API Language:**
    :ref:`中文 <nonzero_sign_log_abs_backward-cn>` | :ref:`English <nonzero_sign_log_abs_backward-en>`

    ----

    .. _nonzero_sign_log_abs_backward-cn:

    * **中文**

    NonzeroSignLogAbs 替代函数的梯度计算。 :math:`g'(x) = \frac{\alpha}{1 + |\alpha x|} = \frac{1}{\frac{1}{\alpha} + |x|}`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 控制函数形状的参数
    :type alpha: float
    :return: 梯度张量, None
    :rtype: tuple

    ----

    .. _nonzero_sign_log_abs_backward-en:

    * **English**

    Gradient computation for the NonzeroSignLogAbs surrogate function.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: parameter to control the shape of the function
    :type alpha: float
    :return: gradient tensor, None
    :rtype: tuple
    """
    return grad_output / (1 / alpha + x.abs()), None


class nonzero_sign_log_abs(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <nonzero_sign_log_abs-cn>` | :ref:`English <nonzero_sign_log_abs-en>`

    ----

    .. _nonzero_sign_log_abs-cn:

    * **中文**

    NonzeroSignLogAbs 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 nonzero_sign_log_abs_backward 自定义梯度。

    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _nonzero_sign_log_abs-en:

    * **English**

    The torch.autograd.Function wrapper for the NonzeroSignLogAbs surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by nonzero_sign_log_abs_backward.

    :return: spike tensor with the same shape as input x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, alpha):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha

    @staticmethod
    def backward(ctx, grad_output):
        return nonzero_sign_log_abs_backward(
            grad_output, ctx.saved_tensors[0], ctx.alpha
        )


class NonzeroSignLogAbs(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <NonzeroSignLogAbs.__init__-cn>` | :ref:`English <NonzeroSignLogAbs.__init__-en>`

        ----

        .. _NonzeroSignLogAbs.__init__-cn:

        * **中文**

        .. warning::
            原函数的输出范围并不是(0, 1)。它的优势是反向传播的计算量特别小。

        反向传播时使用NonzeroSignLogAbs的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \frac{\alpha}{1 + |\alpha x|} = \frac{1}{\frac{1}{\alpha} + |x|}

        对应的原函数为

        .. math::
            g(x) = \mathrm{NonzeroSign}(x) \log (|\alpha x| + 1)

        其中

            .. math::
                \mathrm{NonzeroSign}(x) =
                \begin{cases}
                1, & x \geq 0 \\
                -1, & x < 0 \\
                \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/NonzeroSignLogAbs.*
            :width: 100%

        该函数在文章 [#yin2017algorithm]_ [#STBP]_ [#SuperSpike]_ 中使用。

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :type alpha: float

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _NonzeroSignLogAbs.__init__-en:

        * **English**

        .. admonition:: Warning
            :class: warning

            The output range the primitive function is not (0, 1). The advantage of this function is that computation
            cost is small when backward.

        The NonzeroSignLogAbs surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \frac{\alpha}{1 + |\alpha x|} = \frac{1}{\frac{1}{\alpha} + |x|}

        The primitive function is defined by

        .. math::
            g(x) = \mathrm{NonzeroSign}(x) \log (|\alpha x| + 1)

        where

            .. math::
                \mathrm{NonzeroSign}(x) =
                \begin{cases}
                1, & x \geq 0 \\
                -1, & x < 0 \\
                \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/NonzeroSignLogAbs.*
            :width: 100%

        The function is used in [#yin2017algorithm]_ [#STBP]_ [#SuperSpike]_.

        :param alpha: parameter to control smoothness of gradient
        :type alpha: float

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: 无返回值
        :rtype: None
        """
        super().__init__(spiking=spiking, alpha=alpha)

    @staticmethod
    def spiking_function(x, alpha):
        return nonzero_sign_log_abs.apply(x, alpha)

    @staticmethod
    def backward(grad_output, x, alpha):
        return nonzero_sign_log_abs_backward(grad_output, x, alpha)[0]

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        # the gradient of ``(heaviside(x) * 2 - 1) * (alpha * x.abs() + 1).log()`` by autograd is wrong at ``x==0``
        mask_p = heaviside(x) * 2.0 - 1.0
        return mask_p * (alpha * mask_p * x + 1).log()


def erf_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    r"""
    **API Language:**
    :ref:`中文 <erf_backward-cn>` | :ref:`English <erf_backward-en>`

    ----

    .. _erf_backward-cn:

    * **中文**

    Erf 替代函数的梯度计算。 :math:`g'(x) = \frac{\alpha}{\\sqrt{\pi}}e^{-\alpha^{2}x^{2}}`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 控制函数形状的参数
    :type alpha: float
    :return: 梯度张量, None
    :rtype: tuple

    ----

    .. _erf_backward-en:

    * **English**

    Gradient computation for the Erf surrogate function.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: parameter to control the shape of the function
    :type alpha: float
    :return: gradient tensor, None
    :rtype: tuple
    """
    return grad_output * (-(x * alpha).pow_(2)).exp_() * (
        alpha / math.sqrt(math.pi)
    ), None


class erf(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <erf-cn>` | :ref:`English <erf-en>`

    ----

    .. _erf-cn:

    * **中文**

    Erf 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 erf_backward 自定义梯度。

    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _erf-en:

    * **English**

    The torch.autograd.Function wrapper for the Erf surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by erf_backward.

    :return: spike tensor with the same shape as input x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, alpha):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha

    @staticmethod
    def backward(ctx, grad_output):
        return erf_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Erf(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <Erf.__init__-cn>` | :ref:`English <Erf.__init__-en>`

        ----

        .. _Erf.__init__-cn:

        * **中文**

        反向传播时使用高斯误差函数(erf)的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) = \frac{\alpha}{\sqrt{\pi}}e^{-\alpha^{2}x^{2}}

        对应的原函数为

        .. math::
            :nowrap:

            \begin{split}
            g(x) &= \frac{1}{2}(1-\text{erf}(-\alpha x)) \\
            &= \frac{1}{2} \text{erfc}(-\alpha x) \\
            &= \frac{1}{\sqrt{\pi}}\int_{-\infty}^{\alpha x}e^{-t^{2}}dt
            \end{split}

        .. image:: ../_static/API/activation_based/surrogate/Erf.*
            :width: 100%

        该函数在文章 [#esser2015backpropagation]_ [#STBP]_ [#SRNN]_ 中使用。

        :param alpha: 控制反向传播时梯度的平滑程度的参数
        :type alpha: float

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _Erf.__init__-en:

        * **English**

        The Gaussian error (erf) surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \frac{\alpha}{\sqrt{\pi}}e^{-\alpha^{2}x^{2}}

        The primitive function is defined by

        .. math::
            :nowrap:

            \begin{split}
            g(x) &= \frac{1}{2}(1-\text{erf}(-\alpha x)) \\
            &= \frac{1}{2} \text{erfc}(-\alpha x) \\
            &= \frac{1}{\sqrt{\pi}}\int_{-\infty}^{\alpha x}e^{-t^{2}}dt
            \end{split}

        .. image:: ../_static/API/activation_based/surrogate/Erf.*
            :width: 100%

        The function is used in [#esser2015backpropagation]_ [#STBP]_ [#SRNN]_.

        :param alpha: parameter to control smoothness of gradient
        :type alpha: float

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: 无返回值
        :rtype: None
        """
        super().__init__(spiking=spiking, alpha=alpha)

    @staticmethod
    def spiking_function(x, alpha):
        return erf.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        return torch.erfc_(-alpha * x) / 2.0

    @staticmethod
    def backward(grad_output, x, alpha):
        return erf_backward(grad_output, x, alpha)[0]


def piecewise_leaky_relu_backward(
    grad_output: torch.Tensor, x: torch.Tensor, w: float, c: float
):
    r"""
    **API Language:**
    :ref:`中文 <piecewise_leaky_relu_backward-cn>` | :ref:`English <piecewise_leaky_relu_backward-en>`

    ----

    .. _piecewise_leaky_relu_backward-cn:

    * **中文**

    PiecewiseLeakyReLU 替代函数的梯度计算。 :math:`g'(x) = \begin{cases} \frac{1}{2w}, & -w \leq x \leq w \\\\ c, & x < -w \text{ or } x > w \\end{cases}`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param w: 控制窗口宽度的参数
    :type w: float
    :param c: 窗口外的梯度值
    :type c: float
    :return: 梯度张量, None, None
    :rtype: tuple

    ----

    .. _piecewise_leaky_relu_backward-en:

    * **English**

    Gradient computation for the PiecewiseLeakyReLU surrogate function.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param w: parameter to control the width of the window
    :type w: float
    :param c: gradient value outside the window
    :type c: float
    :return: gradient tensor, None, None
    :rtype: tuple
    """
    mask_width = x.abs() < w
    mask_c = mask_width.logical_not()
    return (
        grad_output * x.masked_fill(mask_width, 1 / (2 * w)).masked_fill(mask_c, c),
        None,
        None,
    )


class piecewise_leaky_relu(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <piecewise_leaky_relu-cn>` | :ref:`English <piecewise_leaky_relu-en>`

    ----

    .. _piecewise_leaky_relu-cn:

    * **中文**

    PiecewiseLeakyReLU 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 piecewise_leaky_relu_backward 自定义梯度。

    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _piecewise_leaky_relu-en:

    * **English**

    The torch.autograd.Function wrapper for the PiecewiseLeakyReLU surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by piecewise_leaky_relu_backward.

    :return: spike tensor with the same shape as input x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x: torch.Tensor, w=1, c=0.01):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, w, c = inputs
        ctx.save_for_backward(x)
        ctx.w = w
        ctx.c = c

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_leaky_relu_backward(
            grad_output, ctx.saved_tensors[0], ctx.w, ctx.c
        )


class PiecewiseLeakyReLU(SurrogateFunctionBase):
    def __init__(self, w=1.0, c=0.01, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <PiecewiseLeakyReLU.__init__-cn>` | :ref:`English <PiecewiseLeakyReLU.__init__-en>`

        ----

        .. _PiecewiseLeakyReLU.__init__-cn:

        * **中文**

        分段线性的近似脉冲发放函数。梯度为

        .. math::
            g'(x) =
            \begin{cases}
            \frac{1}{2w}, & -w \leq x \leq w \\
            c, & x < -w ~or~ x > w
            \end{cases}

        对应的原函数为

        .. math::
            g(x) =
            \begin{cases}
            cx + cw, & x < -w \\
            \frac{1}{2w}x + \frac{1}{2}, & -w \leq x \leq w \\
            cx - cw + 1, & x > w \\
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseLeakyReLU.*
            :width: 100%

        该函数在文章 [#yin2017algorithm]_ [#STBP]_ [#huh2018gradient]_ [#wu2019direct]_ [#STCA]_ [#roy2019scaling]_ [#LISNN]_ [#DECOLLE]_ 中使用。

        :param w: ``-w <= x <= w`` 时反向传播的梯度为 ``1 / 2w``
        :type w: float

        :param c: ``x > w`` 或 ``x < -w`` 时反向传播的梯度为 ``c``
        :type c: float

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _PiecewiseLeakyReLU.__init__-en:

        * **English**

        The piecewise surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) =
            \begin{cases}
            \frac{1}{2w}, & -w \leq x \leq w \\
            c, & x < -w ~or~ x > w
            \end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \begin{cases}
            cx + cw, & x < -w \\
            \frac{1}{2w}x + \frac{1}{2}, & -w \leq x \leq w \\
            cx - cw + 1, & x > w
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/PiecewiseLeakyReLU.*
            :width: 100%

        The function is used in [#yin2017algorithm]_ [#STBP]_ [#huh2018gradient]_ [#wu2019direct]_ [#STCA]_ [#roy2019scaling]_ [#LISNN]_ [#DECOLLE]_.

        :param w: when ``-w <= x <= w`` the gradient is ``1 / 2w``
        :type w: float

        :param c: when ``x > w`` or ``x < -w`` the gradient is ``c``
        :type c: float

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: 无返回值
        :rtype: None
        """
        assert w > 0.0
        super().__init__(spiking=spiking, w=w, c=c)

    @staticmethod
    def spiking_function(x: torch.Tensor, w, c):
        return piecewise_leaky_relu.apply(x, w, c)

    @staticmethod
    def backward(grad_output, x, w, c):
        return piecewise_leaky_relu_backward(grad_output, x, w, c)[0]

    @staticmethod
    def primitive_function(x: torch.Tensor, w: float, c: float):
        mask0 = (x < -w).to(x)
        mask1 = (x > w).to(x)
        mask2 = torch.ones_like(x.data) - mask0 - mask1
        if c == 0:
            return mask2 * (x / (2 * w) + 1 / 2) + mask1
        else:
            cw = c * w
            return (
                mask0 * (c * x + cw)
                + mask1 * (c * x + (-cw + 1))
                + mask2 * (x / (2 * w) + 1 / 2)
            )

    def cuda_code(self, x: str, y: str, dtype="fp32"):
        sg_name = "sg_" + self._get_name()
        w = str(self.w) + "f"
        w_inv = str(1.0 / self.w) + "f"
        c = str(self.c) + "f"
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """

        if dtype == "fp32":
            code += f"""
            {tab4_str}const float {sg_name}_x_abs = fabsf({x});
            float {y};
            if ({sg_name}_x_abs > {w})
            {curly_bracket_l}
                {y} = {c};
            {curly_bracket_r}
            else
            {curly_bracket_l}
                {y} = {w_inv};
            {curly_bracket_r}
            """
        elif dtype == "fp16":
            code += f"""
            {tab4_str}const half2 {sg_name}_x_abs = __habs2({x});
            {tab4_str}const half2 {sg_name}_x_abs_ge_w = __hge2({sg_name}_x_abs, __float2half2_rn({w}));
            {tab4_str}half2 {y} = __hadd2(__hmul2(__float2half2_rn({c}),  {sg_name}_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_x_abs_ge_w), __float2half2_rn({w_inv})));
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.piecewise_leaky_relu_backward(
            y=y, x=x, w=self.w, c=self.c, dtype=dtype
        )


class squarewave_fourier_series(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <squarewave_fourier_series-cn>` | :ref:`English <squarewave_fourier_series-en>`

    ----

    .. _squarewave_fourier_series-cn:

    * **中文**

    SquarewaveFourierSeries 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用傅里叶级数近似的方法计算梯度。

    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _squarewave_fourier_series-en:

    * **English**

    The torch.autograd.Function wrapper for the SquarewaveFourierSeries surrogate function. Forward uses ``heaviside``,
    backward computes the gradient via Fourier series approximation.

    :return: spike tensor with the same shape as input x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x: torch.Tensor, n: int, T_period: float):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, n, T_period = inputs
        ctx.save_for_backward(x)
        ctx.n = n
        ctx.T_period = T_period

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = 0.0
        x = ctx.saved_tensors[0]
        w = math.pi * 2.0 / ctx.T_period
        for i in range(1, ctx.n):
            grad_x += torch.cos_((2 * i - 1.0) * w * x)

        grad_x *= 4.0 / ctx.T_period
        grad_x *= grad_output

        return grad_x, None, None


class SquarewaveFourierSeries(SurrogateFunctionBase):
    def __init__(self, n: int = 2, T_period: float = 8, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <SquarewaveFourierSeries.__init__-cn>` | :ref:`English <SquarewaveFourierSeries.__init__-en>`

        ----

        .. _SquarewaveFourierSeries.__init__-cn:

        * **中文**

        使用傅里叶级数近似方波的脉冲发放函数。反向传播使用傅里叶级数展开计算梯度。

        对应的原函数为

        .. math::
            g(x) = \frac{1}{2} + \frac{2}{\pi} \sum_{i=1}^{n-1} \frac{\sin((2i-1)\omega x)}{2i-1}

        其中 :math:`\omega = \frac{2\pi}{T}`。

        .. image:: ../_static/API/activation_based/surrogate/SquarewaveFourierSeries.*
            :width: 100%

        :param n: 傅里叶级数的项数
        :type n: int
        :param T_period: 方波的周期
        :type T_period: float
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _SquarewaveFourierSeries.__init__-en:

        * **English**

        The square-wave Fourier series surrogate spiking function. The gradient is computed via Fourier series expansion.

        The primitive function is defined by

        .. math::
            g(x) = \frac{1}{2} + \frac{2}{\pi} \sum_{i=1}^{n-1} \frac{\sin((2i-1)\omega x)}{2i-1}

        where :math:`\omega = \frac{2\pi}{T}`.

        .. image:: ../_static/API/activation_based/surrogate/SquarewaveFourierSeries.*
            :width: 100%

        :param n: number of terms in Fourier series
        :type n: int
        :param T_period: period of the square wave
        :type T_period: float
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: 无返回值
        :rtype: None
        """
        assert isinstance(n, int) and T_period > 0.0
        super().__init__(spiking=spiking, n=n, T_period=T_period)

    @staticmethod
    def spiking_function(x: torch.Tensor, n, T_period):
        return squarewave_fourier_series.apply(x, n, T_period)

    @staticmethod
    def primitive_function(x: torch.Tensor, n: int, T_period: float):
        w = math.pi * 2.0 / T_period
        ret = torch.zeros_like(x.data)
        for i in range(1, n):
            c = 2 * i - 1.0
            ret += torch.sin(c * w * x) / c

        return 0.5 + 2.0 / math.pi * ret


class s2nn(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <s2nn-cn>` | :ref:`English <s2nn-en>`

    ----

    .. _s2nn-cn:

    * **中文**

    S2NN 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 S2NN 自定义梯度（负半轴为 sigmoid 梯度，正半轴为 β/(x+1)）。

    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _s2nn-en:

    * **English**

    The torch.autograd.Function wrapper for the S2NN surrogate function. Forward uses ``heaviside``,
    backward uses the S2NN custom gradient (sigmoid gradient for negative x, β/(x+1) for non-negative x).

    :return: spike tensor with the same shape as input x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x: torch.Tensor, alpha: float, beta: float):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha, beta = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.beta = beta

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sgax = torch.sigmoid(ctx.alpha * x)
        grad_x = torch.where(
            x < 0.0, ctx.alpha * sgax * (1.0 - sgax), ctx.beta / (x + 1.0)
        )
        return grad_x * grad_output, None, None


class S2NN(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, beta=1.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <S2NN.__init__-cn>` | :ref:`English <S2NN.__init__-en>`

        ----

        .. _S2NN.__init__-cn:

        * **中文**

        `S2NN: Time Step Reduction of Spiking Surrogate Gradients for Training Energy Efficient Single-Step Neural Networks <https://arxiv.org/abs/2201.10879>`_ 提出的S2NN替代函数。反向传播为

        .. math::
            g'(x) = \begin{cases}
                \alpha * (1 - \mathrm{sigmoid} (\alpha x)) \mathrm{sigmoid} (\alpha x), x < 0 \\
                \frac{\beta}{(x + 1)}, x \ge 0
            \end{cases}

        对应的原函数为

        .. math::
            g(x) = \begin{cases}
                \mathrm{sigmoid} (\alpha x), x < 0 \\
                \beta \mathrm{ln}(x + 1) + 1, x \ge 0
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/S2NN.*
            :width: 100%

        :param alpha: 控制 ``x < 0`` 时梯度的参数
        :type alpha: float

        :param beta: 控制 ``x >= 0`` 时梯度的参数
        :type beta: float

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _S2NN.__init__-en:

        * **English**

        The S2NN surrogate spiking function, which is proposed by `S2NN: Time Step Reduction of Spiking Surrogate Gradients for Training Energy Efficient Single-Step Neural Networks <https://arxiv.org/abs/2201.10879>`_. The gradient is defined by

        .. math::
            g'(x) = \begin{cases}
                \alpha * (1 - \mathrm{sigmoid} (\alpha x)) \mathrm{sigmoid} (\alpha x), x < 0 \\
                \frac{\beta}{x + 1}, x \ge 0
            \end{cases}

        The primitive function is defined by

        .. math::
            g(x) = \begin{cases}
                \mathrm{sigmoid} (\alpha x), x < 0 \\
                \beta \mathrm{ln}(x + 1) + 1, x \ge 0
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/S2NN.*
            :width: 100%

        :param alpha: the param that controls the gradient when ``x < 0``
        :type alpha: float

        :param beta: the param that controls the gradient when ``x >= 0``
        :type beta: float

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: 无返回值
        :rtype: None
        """
        super().__init__(spiking=spiking, alpha=alpha, beta=beta)

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function

        return f(x, self.alpha, self.beta)

    @staticmethod
    def spiking_function(x: torch.Tensor, alpha, beta):
        return s2nn.apply(x, alpha, beta)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float, beta: float):
        return torch.where(
            x < 0.0,
            torch.sigmoid(x * alpha),
            beta * torch.log((x + 1.0).abs_() + 1e-5) + 0.5,
        )
        # abs and 1e-5 are used to avoid nan

    def cuda_code(self, x: str, y: str, dtype="fp32"):
        sg_name = "sg_" + self._get_name()
        alpha = str(self.alpha) + "f"
        beta = str(self.beta) + "f"
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """

        if dtype == "fp32":
            code += f"""
            {tab4_str}const float {sg_name}_sigmoid_ax = 1.0f / (1.0f + expf(- {alpha} * {x}));
            {tab4_str}const float {sg_name}_mask_l = (float)({x} < 0.0f);
            {tab4_str}const float {y} = (1.0f - {sg_name}_sigmoid_ax) * {sg_name}_sigmoid_ax * {alpha} * {sg_name}_mask_l + {beta} / ({x} + 1.0f) * (1.0f - {sg_name}_mask_l);
            """
        elif dtype == "fp16":
            code += f"""
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2({sg_name}_alpha, {x}))), __float2half2_rn(1.0f)));
            {tab4_str}const half2 {sg_name}_mask_l = __hlt2({x}, __float2half2_rn(0.0f));
            {tab4_str}const half2 {y} = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_sigmoid_ax), {sg_name}_sigmoid_ax), {sg_name}_alpha), {sg_name}_mask_l), __hmul2(__h2div(__float2half2_rn({beta}), __hadd2({x}, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), {sg_name}_mask_l)));
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.s2nn_backward(
            y=y, x=x, alpha=self.alpha, beta=self.beta, dtype=dtype
        )


class q_pseudo_spike(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <q_pseudo_spike-cn>` | :ref:`English <q_pseudo_spike-en>`

    ----

    .. _q_pseudo_spike-cn:

    * **中文**

    QPseudoSpike 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 :math:`q`-PseudoSpike 自定义梯度。

    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _q_pseudo_spike-en:

    * **English**

    The torch.autograd.Function wrapper for the QPseudoSpike surrogate function. Forward uses ``heaviside``,
    backward uses the :math:`q`-PseudoSpike custom gradient.

    :return: spike tensor with the same shape as input x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, alpha):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        x = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            grad_x = (
                (1 + 2 / (ctx.alpha - 1) * x.abs()).pow_(-ctx.alpha)
            ) * grad_output
        return grad_x, None


class QPseudoSpike(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <QPseudoSpike.__init__-cn>` | :ref:`English <QPseudoSpike.__init__-en>`

        ----

        .. _QPseudoSpike.__init__-cn:

        * **中文**

        `Surrogate Gradients Design <https://arxiv.org/abs/2202.00282>`_ 提出的 :math:`q`-PseudoSpike替代函数。反向传播为

        .. math::
            g'(x) = (1+\frac{2|x|}{\alpha-1})^{-\alpha}

        其中 :math:`\alpha>1` 对应原文中的 :math:`q`。

        对应的原函数为

        .. math::
            g(x) =
            \begin{cases}
            \frac{1}{2}(1-\frac{2x}{\alpha-1})^{1-\alpha}, & x < 0 \\
            1 - \frac{1}{2}(1+\frac{2x}{\alpha-1})^{1-\alpha}, & x \geq 0.
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/QPseudoSpike.*
            :width: 100%

        :param alpha: 控制反向传播时梯度函数尾部厚度的参数
        :type alpha: float

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _QPseudoSpike.__init__-en:

        * **English**

        The :math:`q`-PseudoSpike surrogate spiking function, which is first proposed in `Surrogate Gradients Design <https://arxiv.org/abs/2202.00282>`_. The gradient is defined by

        .. math::
            g'(x) = (1+\frac{2|x|}{\alpha-1})^{-\alpha}

        where :math:`\alpha>1` corresponds to :math:`q` in paper.

        The primitive function is defined by

        .. math::
            g(x) =
            \begin{cases}
            \frac{1}{2}(1-\frac{2x}{\alpha-1})^{1-\alpha}, & x < 0 \\
            1 - \frac{1}{2}(1+\frac{2x}{\alpha-1})^{1-\alpha}, & x \geq 0.
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/QPseudoSpike.*
            :width: 100%

        :param alpha: parameter to control tail fatness of gradient
        :type alpha: float

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: 无返回值
        :rtype: None
        """
        super().__init__(spiking=spiking, alpha=alpha)

    @staticmethod
    def spiking_function(x, alpha):
        return q_pseudo_spike.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        mask_nonnegative = heaviside(x)
        mask_sign = mask_nonnegative * 2.0 - 1.0

        return mask_nonnegative - mask_sign * (
            0.5 * ((1.0 + 2.0 / (alpha - 1.0) * x * mask_sign).pow_(1.0 - alpha))
        )

    def cuda_code(self, x: str, y: str, dtype="fp32"):
        sg_name = "sg_" + self._get_name()
        alpha = str(self.alpha) + "f"
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """

        if dtype == "fp32":
            code += f"""
            {tab4_str}const float {sg_name}_base = 1.0f + 2.0f / ({alpha} - 1.0f) * fabsf({x});
            {tab4_str}const float {y} = powf({sg_name}_base, -{alpha});
            """
        elif dtype == "fp16":
            code += f"""
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2({x})), __hsub2({sg_name}_alpha, __float2half2_rn(1.0f))));
            {tab4_str}const half2 {y} = h2exp2(__hmul2(h2log2({sg_name}_base), __hneg2({sg_name}_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.q_pseudo_spike_backward(
            y=y, x=x, alpha=self.alpha, dtype=dtype
        )


def leaky_k_relu_backward(
    grad_output: torch.Tensor, x: torch.Tensor, leak: float, k: float
):
    r"""
    **API Language:**
    :ref:`中文 <leaky_k_relu_backward-cn>` | :ref:`English <leaky_k_relu_backward-en>`

    ----

    .. _leaky_k_relu_backward-cn:

    * **中文**

    LeakyKReLU 替代函数的梯度计算。 :math:`g'(x) = \begin{cases} k, & x \geq 0 \\\\ leak, & x < 0 \\end{cases}`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param leak: ``x < 0`` 时的梯度值
    :type leak: float
    :param k: ``x \geq 0`` 时的梯度值
    :type k: float
    :return: 梯度张量, None, None
    :rtype: tuple

    ----

    .. _leaky_k_relu_backward-en:

    * **English**

    Gradient computation for the LeakyKReLU surrogate function.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param leak: gradient value when ``x < 0``
    :type leak: float
    :param k: gradient value when ``x \geq 0``
    :type k: float
    :return: gradient tensor, None, None
    :rtype: tuple
    """
    mask1 = (x >= 0.0).to(x)
    grad_x = mask1 * k + (1.0 - mask1) * leak
    return grad_output * grad_x, None, None


class leaky_k_relu(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <leaky_k_relu-cn>` | :ref:`English <leaky_k_relu-en>`

    ----

    .. _leaky_k_relu-cn:

    * **中文**

    LeakyKReLU 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 leaky_k_relu_backward 自定义梯度。

    :param x: 输入张量
    :type x: torch.Tensor
    :param leak: ``x < 0`` 时的梯度值
    :type leak: float
    :param k: ``x \geq 0`` 时的梯度值
    :type k: float
    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _leaky_k_relu-en:

    * **English**

    The torch.autograd.Function wrapper for the LeakyKReLU surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by leaky_k_relu_backward.

    :param x: input tensor
    :type x: torch.Tensor
    :param leak: gradient value when ``x < 0``
    :type leak: float
    :param k: gradient value when ``x \geq 0``
    :type k: float
    :return: spike tensor, same shape as x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, leak, k):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, leak, k = inputs
        ctx.save_for_backward(x)
        ctx.leak = leak
        ctx.k = k

    @staticmethod
    def backward(ctx, grad_output):
        return leaky_k_relu_backward(grad_output, ctx.saved_tensors[0], ctx.leak, ctx.k)


class LeakyKReLU(SurrogateFunctionBase):
    def __init__(self, leak: float = 0.0, k: float = 1.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <LeakyKReLU.__init__-cn>` | :ref:`English <LeakyKReLU.__init__-en>`

        ----

        .. _LeakyKReLU.__init__-cn:

        * **中文**

        反向传播时使用LeakyKReLU的梯度的脉冲发放函数。反向传播为

        .. math::
            g'(x) =
            \begin{cases}
            k, & x \geq 0 \\
            leak, & x < 0 \\
            \end{cases}

        对应的原函数为

        .. math::
            g(x) =
            \begin{cases}
            k \cdot x, & x \geq 0 \\
            leak \cdot x, & x < 0 \\
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/LeakyKReLU.*
            :width: 100%

        该函数在文章 [#yin2017algorithm]_ [#STBP]_ [#superSpike]_ 中使用。

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        :param leak: ``x < 0`` 时的梯度值
        :type leak: float

        :param k: ``x >= 0 `` 时的梯度值
        :type k: float

        :return: 无返回值
        :rtype: None

        ----

        .. _LeakyKReLU.__init__-en:

        * **English**

        The LeakyKReLU surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) =
            \begin{cases}
            k, & x \geq 0 \\
            leak, & x < 0 \\
            \end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \begin{cases}
            k \cdot x, & x \geq 0 \\
            leak \cdot x, & x < 0 \\
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/LeakyKReLU.*
            :width: 100%

        The function is used in [#yin2017algorithm]_ [#STBP]_ [#superSpike]_.

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool

        :param leak: gradient when ``x < 0``
        :type leak: float

        :param k: gradient when ``x >= 0 ``
        :type k: float

        :return: No return value.
        :rtype: None
        """
        super().__init__(spiking=spiking, leak=leak, k=k)

    @staticmethod
    def spiking_function(x, leak, k):
        return leaky_k_relu.apply(x, leak, k)

    @staticmethod
    def primitive_function(x: torch.Tensor, leak: float, k: float):
        mask1 = (x >= 0.0).to(x)
        return (leak * (1.0 - mask1) + k * mask1) * x

    @staticmethod
    def backward(grad_output, x, leak, k):
        return leaky_k_relu_backward(grad_output, x, leak, k)[0]

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function

        return f(x, self.leak, self.k)

    def cuda_code(self, x: str, y: str, dtype="fp32"):
        sg_name = "sg_" + self._get_name()
        leak = str(self.leak) + "f"
        k = str(self.k) + "f"
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """

        if dtype == "fp32":
            code += f"""
            {tab4_str}const float {sg_name}_mask1 = (float) ({x} >= 0.0f);
            {tab4_str}const float {y} = {leak} * (1.0f - {sg_name}_mask1) + {k} * {sg_name}_mask1;
            """
        elif dtype == "fp16":
            code += f"""
            {tab4_str}const half2 {sg_name}_mask1 = __hgeu2({x}, __float2half2_rn(0.0f));
            {tab4_str}const half2 {y} = __hfma2(__float2half2_rn({k}), {sg_name}_mask1, __hmul2(__float2half2_rn({leak}), __hsub2(__float2half2_rn(1.0f), {sg_name}_mask1)));
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.leaky_k_relu_backward(
            y=y, x=x, leak=self.leak, k=self.k, dtype=dtype
        )


def fake_numerical_gradient_backward(
    grad_output: torch.Tensor, x: torch.Tensor, alpha: float
):
    r"""
    **API Language:**
    :ref:`中文 <fake_numerical_gradient_backward-cn>` | :ref:`English <fake_numerical_gradient_backward-en>`

    ----

    .. _fake_numerical_gradient_backward-cn:

    * **中文**

    FakeNumericalGradient 替代函数的梯度计算。模拟数值梯度的行为。

    :math:`g'(x) = \begin{cases} \frac{2 - sign(x)}{x}, & |\frac{2 - sign(x)}{x}| \leq \alpha \\\\ \alpha \cdot sign(\frac{2 - sign(x)}{x}), & |\frac{2 - sign(x)}{x}| > \alpha \\end{cases}`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 梯度裁剪阈值
    :type alpha: float
    :return: 梯度张量, None
    :rtype: tuple

    ----

    .. _fake_numerical_gradient_backward-en:

    * **English**

    Gradient computation for the FakeNumericalGradient surrogate function, which emulates numerical gradient.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: gradient clipping threshold
    :type alpha: float
    :return: gradient tensor, None
    :rtype: tuple
    """
    grad_x = torch.clamp_max(((x >= 0.0) * 2.0 - 1.0) / x, alpha)
    return grad_output * grad_x, None


class fake_numerical_gradient(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <fake_numerical_gradient-cn>` | :ref:`English <fake_numerical_gradient-en>`

    ----

    .. _fake_numerical_gradient-cn:

    * **中文**

    FakeNumericalGradient 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 fake_numerical_gradient_backward 自定义梯度。

    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 梯度裁剪阈值
    :type alpha: float
    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _fake_numerical_gradient-en:

    * **English**

    The torch.autograd.Function wrapper for the FakeNumericalGradient surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by fake_numerical_gradient_backward.

    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: gradient clipping threshold
    :type alpha: float
    :return: spike tensor, same shape as x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, alpha):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha

    @staticmethod
    def backward(ctx, grad_output):
        return fake_numerical_gradient_backward(
            grad_output, ctx.saved_tensors[0], ctx.alpha
        )


class FakeNumericalGradient(SurrogateFunctionBase):
    def __init__(self, alpha=0.3):
        r"""
        **API Language:**
        :ref:`中文 <FakeNumericalGradient.__init__-cn>` | :ref:`English <FakeNumericalGradient.__init__-en>`

        ----

        .. _FakeNumericalGradient.__init__-cn:

        * **中文**

        模拟数值梯度的脉冲发放函数，反向传播为

        .. math::
            g'(x) = \mathrm{clip}(\frac{\mathrm{sign}(x)}{x}, \alpha)

        :param alpha: 梯度裁剪阈值
        :type alpha: float

        :return: 无返回值
        :rtype: None

        ----

        .. _FakeNumericalGradient.__init__-en:

        * **English**

        The fake numerical gradient surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \mathrm{clip}(\frac{\mathrm{sign}(x)}{x}, \alpha)

        :param alpha: gradient clip threshold
        :type alpha: float

        :return: No return value.
        :rtype: None
        """
        super().__init__(spiking=True, alpha=alpha)

    @staticmethod
    def spiking_function(x, alpha):
        return fake_numerical_gradient.apply(x, alpha)

    @staticmethod
    def backward(grad_output, x, alpha):
        return fake_numerical_gradient_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype="fp32"):
        sg_name = "sg_" + self._get_name()
        alpha = str(self.alpha) + "f"
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """

        if dtype == "fp32":
            code += f"""
            {tab4_str}const float {sg_name}_sign = (float) ({x} >= 0.0f) * 2.0f - 1.0f;
            {tab4_str}const float {y} = min({sg_name}_sign / {x}, {alpha});
            """
        elif dtype == "fp16":
            code += f"""
            {tab4_str}const half2 {sg_name}_sign = __hfma2(__hgeu2({x}, __float2half2_rn(0.0f)), __float2half2_rn(2.0f), __float2half2_rn(-1.0f));
            #if (__CUDA_ARCH__ < 800)
            {tab4_str}const half2 {sg_name}_grad_x = __h2div({sg_name}_sign, {x});
            {tab4_str}const half2 {sg_name}_grad_max = __float2half2_rn({alpha});
            {tab4_str}const half2 {y} = make_half2({sg_name}_grad_x.x <= {sg_name}_grad_max.x ? {sg_name}_grad_x.x : {sg_name}_grad_max.x, {sg_name}_grad_x.y <= {sg_name}_grad_max.y ? {sg_name}_grad_x.y : {sg_name}_grad_max.y);
            #else
            {tab4_str}const half2 {y} = __hmin2(__h2div({sg_name}_sign, {x}), __float2half2_rn({alpha}));
            #endif
            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.fake_numerical_gradient_backward(
            y=y, x=x, alpha=self.alpha, dtype=dtype
        )


def log_tailed_relu_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    r"""
    **API Language:**
    :ref:`中文 <log_tailed_relu_backward-cn>` | :ref:`English <log_tailed_relu_backward-en>`

    ----

    .. _log_tailed_relu_backward-cn:

    * **中文**

    LogTailedReLU 替代函数的梯度计算。

    :math:`g'(x) = \begin{cases} \alpha, & x \leq 0 \\ 1, & 0 < x \leq 1 \\ \frac{1}{x}, & x > 1 \end{cases}`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 控制函数形状的参数
    :type alpha: float
    :return: 梯度张量, None
    :rtype: tuple

    ----

    .. _log_tailed_relu_backward-en:

    * **English**

    Gradient computation for the LogTailedReLU surrogate function.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: parameter to control function shape
    :type alpha: float
    :return: gradient tensor, None
    :rtype: tuple
    """
    mask_gt1 = x > 1.0
    mask_le0 = x <= 0.0
    grad_x = torch.ones_like(grad_output)
    grad_x[mask_gt1] = 1.0 / x[mask_gt1]
    grad_x[mask_le0] = alpha
    return grad_output * grad_x, None


class log_tailed_relu(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <log_tailed_relu-cn>` | :ref:`English <log_tailed_relu-en>`

    ----

    .. _log_tailed_relu-cn:

    * **中文**

    LogTailedReLU 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 log_tailed_relu_backward 自定义梯度。

    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 控制函数形状的参数
    :type alpha: float
    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _log_tailed_relu-en:

    * **English**

    The torch.autograd.Function wrapper for the LogTailedReLU surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by log_tailed_relu_backward.

    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: parameter to control function shape
    :type alpha: float
    :return: spike tensor, same shape as x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, alpha):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha

    @staticmethod
    def backward(ctx, grad_output):
        return log_tailed_relu_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class LogTailedReLU(SurrogateFunctionBase):
    def __init__(self, alpha=0.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <LogTailedReLU.__init__-cn>` | :ref:`English <LogTailedReLU.__init__-en>`

        ----

        .. _LogTailedReLU.__init__-cn:

        * **中文**

        `Deep Learning with Low Precision by Half-wave Gaussian Quantization <https://arxiv.org/abs/1702.00953>`_ 提出的 Log-tailed ReLU替代函数。反向传播为

        .. math::
            g'(x) =
            \begin{cases}
            \alpha, & x \leq 0 \\
            1, & 0 < x \leq 0 \\
            \frac{1}{x}, x > 1 \\
            \end{cases}

        对应的原函数为

        .. math::
            g(x) =
            \begin{cases}
            \alpha x, & x \leq 0 \\
            x, & 0 < x \leq 0 \\
            log(x), x > 1 \\
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/LogTailedReLU.*
            :width: 100%

        该函数在文章 [#STBP]_ [#huh2018gradient]_ [#neftci2019surrogate]_ 中使用。

        :param alpha: 控制反向传播时梯度的参数
        :type alpha: float

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        ----

        .. _LogTailedReLU.__init__-en:

        * **English**

        The Log-tailed ReLU surrogate spiking function, which is first proposed in `Deep Learning with Low Precision by Half-wave Gaussian Quantization <https://arxiv.org/abs/1702.00953>`_. The gradient is defined by

        .. math::
            g'(x) =
            \begin{cases}
            \alpha, & x \leq 0 \\
            1, & 0 < x \leq 0 \\
            \frac{1}{x}, x > 1 \\
            \end{cases}

        The primitive function is defined by

        .. math::
            g(x) =
            \begin{cases}
            \alpha x, & x \leq 0 \\
            x, & 0 < x \leq 0 \\
            log(x), x > 1 \\
            \end{cases}

        .. image:: ../_static/API/activation_based/surrogate/LogTailedReLU.*
            :width: 100%

        The function is used in [#STBP]_ [#huh2018gradient]_ [#neftci2019surrogate]_.

        :param alpha: parameter to control gradient
        :type alpha: float

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool
        :return: 无返回值
        :rtype: None
        """
        super().__init__(spiking=spiking, alpha=alpha)

    @staticmethod
    def spiking_function(x, alpha):
        return log_tailed_relu.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        mask_ge1 = (x > 1.0).to(x)
        y = (1.0 - mask_ge1) * F.leaky_relu(x, alpha) + mask_ge1 * (torch.log(x) + 1.0)
        return y

    @staticmethod
    def backward(grad_output, x, alpha):
        return log_tailed_relu_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype="fp32"):
        sg_name = "sg_" + self._get_name()
        alpha = str(self.alpha) + "f"
        code = f"""
            {tab4_str}{self.cuda_code_start_comments()}
        """

        if dtype == "fp32":
            code += f"""
            {tab4_str}float {y} = 0.0f;
            {tab4_str}if({x} <= 0.0f)
            {tab4_str}{curly_bracket_l}{y} = {alpha};{curly_bracket_r}
            {tab4_str}else if({x} <= 1.0f)
            {tab4_str}{curly_bracket_l}{y} = 1.0f;{curly_bracket_r}
            {tab4_str}else
            {tab4_str}{curly_bracket_l}{y} = 1.0f / {x};{curly_bracket_r}
            """
        elif dtype == "fp16":
            code += f"""
            {tab4_str}const half {sg_name}_alpha = __float2half_rn({alpha});

            {tab4_str}half {sg_name}_{y}_low;
            {tab4_str}const half {sg_name}_{x}_low = __low2half({x});
            {tab4_str}if(__hle({sg_name}_{x}_low, __float2half_rn(0.0f)))
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_low = {sg_name}_alpha;{curly_bracket_r}
            {tab4_str}else if(__hle({sg_name}_{x}_low, __float2half_rn(1.0f)))
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_low = __float2half_rn(1.0f);{curly_bracket_r}
            {tab4_str}else
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_low = __hdiv(__float2half_rn(1.0f), {sg_name}_{x}_low);{curly_bracket_r}

            {tab4_str}half {sg_name}_{y}_high;
            {tab4_str}const half {sg_name}_{x}_high = __high2half({x});
            {tab4_str}if(__hle({sg_name}_{x}_high, __float2half_rn(0.0f)))
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_high = {sg_name}_alpha;{curly_bracket_r}
            {tab4_str}else if(__hle({sg_name}_{x}_high, __float2half_rn(1.0f)))
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_high = __float2half_rn(1.0f);{curly_bracket_r}
            {tab4_str}else
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_high = __hdiv(__float2half_rn(1.0f), {sg_name}_{x}_high);{curly_bracket_r}


            {tab4_str}const half2 {y} = __halves2half2({sg_name}_{y}_low, {sg_name}_{y}_high);

            """
        else:
            raise NotImplementedError
        code += f"""
            {tab4_str}{self.cuda_code_end_comments()}
        """
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.log_tailed_relu_backward(
            y=y, x=x, alpha=self.alpha, dtype=dtype
        )


class deterministic_pass(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <deterministic_pass-cn>` | :ref:`English <deterministic_pass-en>`

    ----

    .. _deterministic_pass-cn:

    * **中文**

    DeterministicPass 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播直接传递梯度（直通估计器）。

    :param x: 输入张量
    :type x: torch.Tensor
    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _deterministic_pass-en:

    * **English**

    The torch.autograd.Function wrapper for the DeterministicPass surrogate function. Forward uses ``heaviside``,
    backward passes the gradient through directly (straight-through estimator).

    :param x: input tensor
    :type x: torch.Tensor
    :return: spike tensor, same shape as x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class DeterministicPass(SurrogateFunctionBase):
    def __init__(self, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <DeterministicPass.__init__-cn>` | :ref:`English <DeterministicPass.__init__-en>`

        ----

        .. _DeterministicPass.__init__-cn:

        * **中文**

        直通估计器（Straight-Through Estimator, STE）替代函数。前向传播使用 ``heaviside`` 阶跃函数，反向传播直接传递梯度。

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。
            若为 ``False`` 则直接返回输入。
        :type spiking: bool

        :return: 无返回值
        :rtype: None

        ----

        .. _DeterministicPass.__init__-en:

        * **English**

        The straight-through estimator (STE) surrogate function. Forward uses ``heaviside`` step function,
        backward directly passes the gradient through.

        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, returns the input directly.
        :type spiking: bool

        :return: No return value.
        :rtype: None
        """
        super().__init__(spiking=spiking)

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return deterministic_pass.apply(x)
        else:
            return x

    @staticmethod
    def backward(grad_output, x, alpha):
        return grad_output


class poisson_pass(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <poisson_pass-cn>` | :ref:`English <poisson_pass-en>`

    ----

    .. _poisson_pass-cn:

    * **中文**

    PoissonPass 替代函数的 torch.autograd.Function 封装。前向传播使用伯努利采样生成脉冲（泊松发放），反向传播直接传递梯度。

    :param x: 输入张量
    :type x: torch.Tensor
    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _poisson_pass-en:

    * **English**

    The torch.autograd.Function wrapper for the PoissonPass surrogate function. Forward uses Bernoulli sampling
    to generate spikes (Poisson firing), backward passes the gradient through directly.

    :param x: input tensor
    :type x: torch.Tensor
    :return: spike tensor, same shape as x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x):
        return torch.bernoulli(x).float()

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class PoissonPass(SurrogateFunctionBase):
    def __init__(self, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <PoissonPass.__init__-cn>` | :ref:`English <PoissonPass.__init__-en>`

        ----

        .. _PoissonPass.__init__-cn:

        * **中文**

        泊松发放替代函数。前向传播使用伯努利采样生成随机脉冲（泊松过程），反向传播直接传递梯度。

        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用伯努利采样生成脉冲而在反向传播使用替代梯度。
            若为 ``False`` 则直接返回输入。
        :type spiking: bool

        :return: 无返回值
        :rtype: None

        ----

        .. _PoissonPass.__init__-en:

        * **English**

        The Poisson firing surrogate function. Forward uses Bernoulli sampling to generate stochastic spikes
        (Poisson process), backward directly passes the gradient through.

        :param spiking: whether output spikes. The default is ``True`` which means that using Bernoulli sampling
            in forward propagation and using surrogate gradient in backward propagation. If ``False``, returns
            the input directly.
        :type spiking: bool

        :return: No return value.
        :rtype: None
        """
        super().__init__(spiking=spiking)

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return poisson_pass.apply(x)
        else:
            return x

    @staticmethod
    def backward(grad_output, x, alpha):
        return grad_output


def rect_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    r"""
    **API Language:**
    :ref:`中文 <rect_backward-cn>` | :ref:`English <rect_backward-en>`

    ----

    .. _rect_backward-cn:

    * **中文**

    Rect 替代函数的梯度计算。

    :math:`g'(x) = \alpha \cdot \mathbf{1}_{|x| < 0.5/\alpha}`

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 控制矩形窗口宽度的参数
    :type alpha: float
    :return: 梯度张量, None
    :rtype: tuple

    ----

    .. _rect_backward-en:

    * **English**

    Gradient computation for the Rect surrogate function.

    :param grad_output: gradient of the upstream
    :type grad_output: torch.Tensor
    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: parameter to control rectangular window width
    :type alpha: float
    :return: gradient tensor, None
    :rtype: tuple
    """
    return alpha * (x.abs() < 0.5 / alpha).to(x) * grad_output, None


class rect(torch.autograd.Function):
    r"""
    **API Language:**
    :ref:`中文 <rect-cn>` | :ref:`English <rect-en>`

    ----

    .. _rect-cn:

    * **中文**

    Rect 替代函数的 torch.autograd.Function 封装。前向传播使用 ``heaviside``，反向传播使用 rect_backward 自定义梯度。

    :param x: 输入张量
    :type x: torch.Tensor
    :param alpha: 控制矩形窗口宽度的参数
    :type alpha: float
    :return: 脉冲张量，与输入 x 形状相同
    :rtype: torch.Tensor

    ----

    .. _rect-en:

    * **English**

    The torch.autograd.Function wrapper for the Rect surrogate function. Forward uses ``heaviside``,
    backward uses the custom gradient defined by rect_backward.

    :param x: input tensor
    :type x: torch.Tensor
    :param alpha: parameter to control rectangular window width
    :type alpha: float
    :return: spike tensor, same shape as x
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(x, alpha):
        return heaviside(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, alpha = inputs
        ctx.save_for_backward(x)
        ctx.alpha = alpha

    @staticmethod
    def backward(ctx, grad_output):
        return rect_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Rect(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        r"""
        **API Language:**
        :ref:`中文 <Rect.__init__-cn>` | :ref:`English <Rect.__init__-en>`

        ----

        .. _Rect.__init__-cn:

        * **中文**

        矩形窗替代函数。反向传播为

        .. math::
            g'(x) = \begin{cases}
            \alpha, & |x| < \frac{0.5}{\alpha} \\\\
            0, & |x| \geq \frac{0.5}{\alpha}
            \\end{cases}

        对应的原函数为

        .. math::
            g(x) = \mathrm{clip}(\alpha x + 0.5, 0, 1)

        .. image:: ../_static/API/activation_based/surrogate/Rect.*
            :width: 100%

        :param alpha: 控制矩形窗口宽度的参数
        :type alpha: float
        :param spiking: 是否输出脉冲，默认为 ``True``，在前向传播时使用 ``heaviside`` 而在反向传播使用替代梯度。若为 ``False``
            则不使用替代梯度，前向传播时，使用反向传播时的梯度替代函数对应的原函数
        :type spiking: bool

        :return: 无返回值
        :rtype: None

        ----

        .. _Rect.__init__-en:

        * **English**

        The rectangular window surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \begin{cases}
            \alpha, & |x| < \frac{0.5}{\alpha} \\\\
            0, & |x| \geq \frac{0.5}{\alpha}
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) = \mathrm{clip}(\alpha x + 0.5, 0, 1)

        .. image:: ../_static/API/activation_based/surrogate/Rect.*
            :width: 100%

        :param alpha: parameter to control the width of the rectangular window
        :type alpha: float
        :param spiking: whether output spikes. The default is ``True`` which means that using ``heaviside`` in forward
            propagation and using surrogate gradient in backward propagation. If ``False``, in forward propagation,
            using the primitive function of the surrogate gradient function used in backward propagation
        :type spiking: bool

        :return: No return value.
        :rtype: None
        """
        super().__init__(spiking=spiking, alpha=alpha)

    @staticmethod
    def spiking_function(x, alpha):
        return rect.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        return torch.clamp(alpha * x + 0.5, min=0.0, max=1.0)

    @staticmethod
    def backward(grad_output, x, alpha):
        return rect_backward(grad_output, x, alpha)[0]


_has_cuda_ = [
    ATan,
    Sigmoid,
    PiecewiseLeakyReLU,
    S2NN,
    QPseudoSpike,
    LeakyKReLU,
    FakeNumericalGradient,
]
