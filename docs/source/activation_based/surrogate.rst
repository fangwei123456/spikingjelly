梯度替代
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

在 :doc:`../activation_based/neuron` 中我们已经提到过，描述神经元放电过程的 :math:`S[t] = \Theta(H[t] - V_{threshold})`，使用了一个Heaviside阶跃函数：

.. math::
    \Theta(x) =
    \begin{cases}
    1, & x \geq 0 \\
    0, & x < 0
    \end{cases}

按照定义，其导数为冲激函数：

.. math::
    \delta(x) = 
    \begin{cases}
    +\infty, & x = 0 \\
    0, & x \neq 0
    \end{cases}

直接使用冲激函数进行梯度下降，显然会使得网络的训练及其不稳定。为了解决这一问题，各种梯度替代法(the surrogate gradient method)被相继提出，参见\
此综述 `Surrogate Gradient Learning in Spiking Neural Networks <https://arxiv.org/abs/1901.09948>`_。

替代函数在神经元中被用于生成脉冲，查看 :class:`BaseNode.neuronal_fire <spikingjelly.activation_based.neuron.BaseNode.neuronal_fire>` 的源代码可以发现：

.. code-block:: python

    # spikingjelly.activation_based.neuron
    class BaseNode(base.MemoryModule):
        def __init__(..., surrogate_function: Callable = surrogate.Sigmoid(), ...)
        # ...
        self.surrogate_function = surrogate_function
        # ...
        

        def neuronal_fire(self):
            return self.surrogate_function(self.v - self.v_threshold)


梯度替代法的原理是，在前向传播时使用 :math:`y = \Theta(x)`，而在反向传播时则使用 :math:`\frac{\mathrm{d}y}{\mathrm{d}x} = \sigma'(x)`，而非\
:math:`\frac{\mathrm{d}y}{\mathrm{d}x} = \Theta'(x)`，其中 :math:`\sigma(x)` 即为替代函数。:math:`\sigma(x)` 通常是一个形状与 :math:`\Theta(x)` \
类似，但光滑连续的函数。

在 :class:`spikingjelly.activation_based.surrogate` 中提供了一些常用的替代函数，其中Sigmoid函数 :math:`\sigma(x, \alpha) = \frac{1}{1 + \exp(-\alpha x)}` \
为 :class:`spikingjelly.activation_based.surrogate.Sigmoid`，下图展示了原始的Heaviside阶跃函数 ``Heaviside``、 ``alpha=5`` 时的Sigmoid原函数 ``Primitive`` \
以及其梯度 ``Gradient``：

.. image:: ../_static/API/activation_based/surrogate/Sigmoid.*
    :width: 100%


替代函数的使用比较简单，使用替代函数就像是使用函数一样：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import surrogate

    sg = surrogate.Sigmoid(alpha=4.)

    x = torch.rand([8]) - 0.5
    x.requires_grad = True
    y = sg(x)
    y.sum().backward()
    print(f'x={x}')
    print(f'y={y}')
    print(f'x.grad={x.grad}')

输出为：

.. code-block:: shell

    x=tensor([-0.1303,  0.4976,  0.3364,  0.4296,  0.2779,  0.4580,  0.4447,  0.2466],
       requires_grad=True)
    y=tensor([0., 1., 1., 1., 1., 1., 1., 1.], grad_fn=<sigmoidBackward>)
    x.grad=tensor([0.9351, 0.4231, 0.6557, 0.5158, 0.7451, 0.4759, 0.4943, 0.7913])

每个替代函数，除了有形如 :class:`spikingjelly.activation_based.surrogate.Sigmoid` 的模块风格API，也提供了形如 :class:`spikingjelly.activation_based.surrogate.sigmoid` 函数风格的API。\
模块风格的API使用驼峰命名法，而函数风格的API使用下划线命名法，关系类似于 ``torch.nn`` 和 ``torch.nn.functional``，下面是几个示例：

===============  ===============
模块              函数
===============  ===============
``Sigmoid``      ``sigmoid``
``SoftSign``     ``soft_sign``
``LeakyKReLU``   ``leaky_k_relu``
===============  ===============

下面是函数风格API的用法示例：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import surrogate

    alpha = 4.
    x = torch.rand([8]) - 0.5
    x.requires_grad = True
    y = surrogate.sigmoid.apply(x, alpha)
    y.sum().backward()
    print(f'x={x}')
    print(f'y={y}')
    print(f'x.grad={x.grad}')


替代函数通常会有1个或多个控制形状的超参数，例如 :class:`spikingjelly.activation_based.surrogate.Sigmoid` 中的 ``alpha``。\
SpikingJelly中替代函数的形状参数，默认情况下是使得替代函数梯度最大值为1，这在一定程度上可以避免梯度累乘导致的梯度爆炸问题。