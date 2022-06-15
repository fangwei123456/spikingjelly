Surrogate Gradient Method
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

As mentioned in :doc:`../activation_based_en/neuron`, the Heaviside function :math:`S[t] = \Theta(H[t] - V_{threshold})` is used to describe the neuronal firing.\
The Heaviside function is:
.. math::
    \Theta(x) =
    \begin{cases}
    1, & x \geq 0 \\
    0, & x < 0
    \end{cases}

按照定义，其导数为冲激函数：

Its derivative is the unit impulse function, which is defined by: 

.. math::
    \delta(x) = 
    \begin{cases}
    +\infty, & x = 0 \\
    0, & x \neq 0
    \end{cases}

If we use the unit impulse function to calculate the gradient and apply the gradient descent, the training will be very unstable. To solve this problem, the surrogate gradient method \
is proposed. Refer to `Surrogate Gradient Learning in Spiking Neural Networks <https://arxiv.org/abs/1901.09948>`_ for more details.

The surrogate function is used to generate spikes, which can be found in the codes of :class:`BaseNode.neuronal_fire <spikingjelly.activation_based.neuron.BaseNode.neuronal_fire>`:

.. code-block:: python

    # spikingjelly.activation_based.neuron
    class BaseNode(base.MemoryModule):
        def __init__(..., surrogate_function: Callable = surrogate.Sigmoid(), ...)
        # ...
        self.surrogate_function = surrogate_function
        # ...
        

        def neuronal_fire(self):
            return self.surrogate_function(self.v - self.v_threshold)

The surrogate gradient method uses :math:`y = \Theta(x)` in forward and :math:`\frac{\mathrm{d}y}{\mathrm{d}x} = \sigma'(x)`, rather than :math:`\frac{\mathrm{d}y}{\mathrm{d}x} = \Theta'(x)` \
in backward, where :math:`\sigma(x)` is the surrogate function. In most cases, :math:`\sigma(x)` is a continuous and smooth function whose shape is similar to :math:`\Theta(x)`.\ 
:class:`spikingjelly.activation_based.surrogate` provides many frequently-used surrogate functions. For example, the Sigmoid function :class:`spikingjelly.activation_based.surrogate.Sigmoid` is :math:`\sigma(x, \alpha) = \frac{1}{1 + \exp(-\alpha x)}`.\ 
The following figure shows the primitive Heaviside function, the sigmoid function when ``alpha=5`` and its gradient:

.. image:: ../_static/API/activation_based/surrogate/Sigmoid.*
    :width: 100%

We can use the surrogate function easily, just as we use other functions:

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

The outputs are:

.. code-block:: shell

    x=tensor([-0.1303,  0.4976,  0.3364,  0.4296,  0.2779,  0.4580,  0.4447,  0.2466],
       requires_grad=True)
    y=tensor([0., 1., 1., 1., 1., 1., 1., 1.], grad_fn=<sigmoidBackward>)
    x.grad=tensor([0.9351, 0.4231, 0.6557, 0.5158, 0.7451, 0.4759, 0.4943, 0.7913])

All surrogate functions have a module style API, e.g., :class:`spikingjelly.activation_based.surrogate.Sigmoid`, and a functional style API, e.g., :class:`spikingjelly.activation_based.surrogate.sigmoid`.\ 
The module style API uses Camel-Case to name modules, while the functional API uses Snake-Case to name functions. Their relation are similar to ``torch.nn`` and ``torch.nn.functional``.\ 
Here are some examples:

===============  ===============
module             function
===============  ===============
``Sigmoid``      ``sigmoid``
``SoftSign``     ``soft_sign``
``LeakyKReLU``   ``leaky_k_relu``
===============  ===============

Here is an example of using the functional API:

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

Most surrogate functions have one or many hyper-parameters to control the shape, e.g., ``alpha`` of :class:`spikingjelly.activation_based.surrogate.Sigmoid`. \
In SpikingJelly, the default shape hyper-parameters are set to make the maximum of the surrogate function's gradient to be 1, which can relieve the gradient vanishing or exploding problem caused by the cumulative product of gradients.
