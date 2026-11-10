梯度替代
=======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

English version: :doc:`../en/surrogate`

神经元放电方程（见 :doc:`./neuron`）:math:`S[t] = \Theta(H[t] - V_{threshold})`，使用了 Heaviside 阶跃函数：

.. math::
    \Theta(x) =
    \begin{cases}
    1, & x \geq 0 \\
    0, & x < 0
    \end{cases}

阶跃函数的导数在 x≠0 处为 0，在 x=0 处不存在，梯度下降得不到有用的梯度。你可以用替代梯度法提供反向传播所需的梯度，参见综述 `Surrogate Gradient Learning in Spiking Neural Networks <https://arxiv.org/abs/1901.09948>`_。

:meth:`SimpleBaseNode.neuronal_fire <spikingjelly.activation_based.neuron.SimpleBaseNode.neuronal_fire>`
使用替代函数计算脉冲。你可以直接调用它观察阈值判断：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron

    node = neuron.SimpleIFNode()
    node.v = torch.tensor([0.5, 1.0, 1.5])
    print(node.neuronal_fire())
    # tensor([0., 1., 1.])


梯度替代法的原理是，在前向传播时使用 :math:`y = \Theta(x)`，而在反向传播时则使用 :math:`\frac{\mathrm{d}y}{\mathrm{d}x} = \sigma'(x)`，其中 :math:`\sigma(x)` 即为替代函数，一般是形状与 :math:`\Theta(x)` 相近、但光滑连续的函数。

在 :mod:`spikingjelly.activation_based.surrogate` 中提供了一些常用的替代函数，其中Sigmoid函数 :math:`\sigma(x, \alpha) = \frac{1}{1 + \exp(-\alpha x)}` \
为 :class:`spikingjelly.activation_based.surrogate.Sigmoid`，下图展示了原始的Heaviside阶跃函数 ``Heaviside``、 ``alpha=5`` 时的Sigmoid原函数 ``Primitive`` \
以及其梯度 ``Gradient``：

.. image:: ../../_static/API/activation_based/surrogate/Sigmoid.*
    :width: 100%


调用 Sigmoid 替代函数，观察前向输出和输入梯度：

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

示例输出（随机数会随运行变化）：

.. code-block:: shell

    x=tensor([-0.1303,  0.4976,  0.3364,  0.4296,  0.2779,  0.4580,  0.4447,  0.2466],
       requires_grad=True)
    y=tensor([0., 1., 1., 1., 1., 1., 1., 1.], grad_fn=<sigmoidBackward>)
    x.grad=tensor([0.9351, 0.4231, 0.6557, 0.5158, 0.7451, 0.4759, 0.4943, 0.7913])

常用替代函数有形如 :class:`spikingjelly.activation_based.surrogate.Sigmoid` 的模块风格API，也提供了形如 :class:`spikingjelly.activation_based.surrogate.sigmoid` 函数风格的API。\
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


替代函数可以有一个或多个控制形状的超参数，例如 :class:`spikingjelly.activation_based.surrogate.Sigmoid` 中的 ``alpha``。\
例如 Sigmoid（alpha=4）和 ATan（alpha=2）的默认值使 x=0 处梯度为 1；其他函数见 API 文档。

替代梯度与自动执行
----------------------------

普通融合神经元路径支持 Sigmoid、ATan、PiecewiseQuadratic、PiecewiseExp、
SoftSign、SuperSpike 和 Erf；替代梯度对象仍通过 ``surrogate_function`` 传入，
执行后端会自动选择。运行以下示例，检查前向输出和反向梯度：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron, surrogate

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    node = neuron.LIFNode(
        step_mode="m", surrogate_function=surrogate.ATan(alpha=2.0)
    ).to(device)
    x = torch.rand(4, 2, 8, device=device, requires_grad=True)
    spikes = node(x)
    (spikes.sum() + node.v.sum()).backward()
    assert torch.isfinite(x.grad).all()
    node.reset()

普通 IF/LIF/PLIF 也接受符合其 API 要求的其他替代梯度或 Python 可调用对象，可通过
Torch 参考公式执行。支持连续原函数的类（例如 Sigmoid）可用 ``spiking=False`` 返回原函数值，融合阶跃发放路径只接受
固定的内置配置；自定义或带可训练参数的替代梯度需使用对应的参考路径。

显式神经元精度要求受支持的内置替代梯度，见 :doc:`./precision`。
融合 ``if_linear``/``lif_linear`` 支持一阶梯度，七种内置替代梯度可编译训练；
自定义替代梯度仅支持 eager 训练，见 :doc:`./memopt`。使用显式精度或融合投影时，
需按该接口检查替代梯度支持范围。
