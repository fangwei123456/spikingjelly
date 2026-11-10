Surrogate Gradient Method
=======================================

Author: `fangwei123456 <https://github.com/fangwei123456>`_

中文版： :doc:`../cn/surrogate`

The Heaviside function :math:`S[t] = \Theta(H[t] - V_{threshold})` is used to describe the neuronal firing.\
See :doc:`./neuron` for the firing equation. The Heaviside function is:

.. math::

    \Theta(x) =
    \begin{cases}
    1, & x \geq 0 \\
    0, & x < 0
    \end{cases}

The step function has zero derivative for x≠0 and no derivative at x=0, so gradient descent receives no useful gradient. Use a surrogate gradient to supply the gradient needed for backpropagation. Refer to `Surrogate Gradient Learning in Spiking Neural Networks <https://arxiv.org/abs/1901.09948>`_ for more details.

:meth:`SimpleBaseNode.neuronal_fire <spikingjelly.activation_based.neuron.SimpleBaseNode.neuronal_fire>`
uses a surrogate function to compute spikes. Call it directly to inspect the threshold decision:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron

    node = neuron.SimpleIFNode()
    node.v = torch.tensor([0.5, 1.0, 1.5])
    print(node.neuronal_fire())
    # tensor([0., 1., 1.])

The surrogate gradient method uses :math:`y = \Theta(x)` in forward and :math:`\frac{\mathrm{d}y}{\mathrm{d}x} = \sigma'(x)` in backward, where the surrogate function :math:`\sigma(x)` is usually a smooth, continuous function shaped like :math:`\Theta(x)`.\
:mod:`spikingjelly.activation_based.surrogate` provides many frequently-used surrogate functions. For example, the Sigmoid function :class:`spikingjelly.activation_based.surrogate.Sigmoid` is :math:`\sigma(x, \alpha) = \frac{1}{1 + \exp(-\alpha x)}`.\
The following figure shows the primitive Heaviside function, the sigmoid function when ``alpha=5`` and its gradient:

.. image:: ../../_static/API/activation_based/surrogate/Sigmoid.*
    :width: 100%

Call the Sigmoid surrogate and inspect its forward output and input gradient:

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

Example output (random values will vary):

.. code-block:: shell

    x=tensor([-0.1303,  0.4976,  0.3364,  0.4296,  0.2779,  0.4580,  0.4447,  0.2466],
       requires_grad=True)
    y=tensor([0., 1., 1., 1., 1., 1., 1., 1.], grad_fn=<sigmoidBackward>)
    x.grad=tensor([0.9351, 0.4231, 0.6557, 0.5158, 0.7451, 0.4759, 0.4943, 0.7913])

Common surrogate functions have a module style API, e.g., :class:`spikingjelly.activation_based.surrogate.Sigmoid`, and a functional style API, e.g., :class:`spikingjelly.activation_based.surrogate.sigmoid`.\
The module style API uses Camel-Case to name modules, while the functional API uses Snake-Case to name functions. Their relationship is similar to ``torch.nn`` and ``torch.nn.functional``.\
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
For example, the defaults of Sigmoid (alpha=4) and ATan (alpha=2) give a gradient of 1 at x=0. See the API documentation for other functions.

Surrogates and automatic execution
----------------------------------

Ordinary fused neuron paths support Sigmoid, ATan, PiecewiseQuadratic,
PiecewiseExp, SoftSign, SuperSpike and Erf. Pass the object as ``surrogate_function``
and let execution select the backend. Run this example to check forward outputs and gradients:

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

Ordinary IF/LIF/PLIF neurons also accept other surrogates or Python callables
that satisfy their APIs and can run through Torch reference equations.
For classes that implement a continuous primitive, such as Sigmoid,
``spiking=False`` returns that primitive. Fused step-function firing
accepts fixed built-in configurations; custom surrogates and trainable surrogate
parameters use the corresponding reference path.

Explicit neuron precision requires supported built-in surrogates; see
:doc:`./precision`. Fused ``if_linear``/``lif_linear`` support first gradients;
the seven built-ins support compiled training, while custom surrogates support
eager training only. See :doc:`./memopt`. Check the surrogate support of the
specific interface when using explicit precision or fused projections.
