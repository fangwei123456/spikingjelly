STDP Learning
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

Researchers of SNNs are always interested in biological learning rules. In SpkingJelly, STDP(Spike Timing Dependent Plasticity) \
is also provided and can be applied to convolutional or linear layers.

STDP(Spike Timing Dependent Plasticity)
-----------------------------------------------------

STDP(Spike Timing Dependent Plasticity) is proposed by [#STDP]_, which is a synaptic plasticity rule found in biological \
neural system. The experiments in the biological neural systems find that the weight of synapse is influenced by the firing time of spikes \
of the pre and post neuron. More specific, STDP can be formulated as:

If the pre neuron fires early and the post neuron fires later, then the weight will increase;
If the pre neuron fires later while the post neuron fires early, then the weight will decrease.

The curve [#STDP_figure]_ that fits the experiments data is as follows:

.. image:: ../_static/tutorials/activation_based/stdp/stdp.*
    :width: 100%

We can use the following equation to describe STDP:

.. math::

    \begin{align}
    \begin{split}
    \Delta w_{ij} =
    \begin{cases}
        A\exp(\frac{-|t_{i}-t_{j}|}{\tau_{+}}) , t_{i} \leq t_{j}, A > 0\\
        B\exp(\frac{-|t_{i}-t_{j}|}{\tau_{-}}) , t_{i} > t_{j}, B < 0
    \end{cases}
    \end{split}
    \end{align}

where :math:`A, B` are the maximum of weight variation, and :math:`\tau_{+}, \tau_{-}` are time constants.

However, the above equation is seldom used in practicals because it needs to record all firing times of pre and post neurons.\
The trace method [#Trace]_  is a more popular method to implement STDP.

For the pre neuron :math:`i` and the post neuron :math:`j`, we use the traces :math:`tr_{pre}[i]` and :math:`tr_{post}[j]` to track their firing. The update of \
traces are similar to the LIF neuron:

.. math::

    tr_{pre}[i][t] = tr_{pre}[i][t] -\frac{tr_{pre}[i][t-1]}{\tau_{pre}} + s[i][t]

    tr_{post}[j][t] = tr_{pre}[i][t] -\frac{tr_{post}[j][t-1]}{\tau_{post}} + s[j][t]

where :math:`\tau_{pre}, \tau_{post}` are time constants of the pre and post neuron. :math:`s[i][t], s[j][t]` are the \
spikes at time-step :math:`t` of the pre neuron :math:`i` and the post neuron :math:`j`, which can only be 0 or 1.

The update of weight is:

.. math::

    \Delta W[i][j][t] = F_{post}(w[i][j][t]) \cdot tr_{j}[t] \cdot s[j][t] - F_{pre}(w[i][j][t]) \cdot tr_{i}[t] \cdot s[i][t]

where :math:`F_{pre}, F_{post}` are functions that control how weight changes.

STDP Learner
-----------------------------------------------------
:class:`spikingjelly.activation_based.learning.STDPLearner` can apply STDP learning on convolutional or linear layers. \
Please read the api doc first to learn how to use it.

Now let us use ``STDPLearner`` to build the simplest ``1x1`` SNN with only one pre and one post neuron. \
And we set the weight as ``0.4``:

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, layer, learning
    from matplotlib import pyplot as plt
    torch.manual_seed(0)

    def f_weight(x):
        return torch.clamp(x, -1, 1.)

    tau_pre = 2.
    tau_post = 100.
    T = 128
    N = 1
    lr = 0.01
    net = nn.Sequential(
        layer.Linear(1, 1, bias=False),
        neuron.IFNode()
    )
    nn.init.constant_(net[0].weight.data, 0.4)

``STDPLearner`` can add the negative weight variation ``- delta_w * scale`` on the gradient of weight, which makes it compatible with deep learning methods. We can use \
the optimizer, learning rate scheduler with ``STDPLearner`` together. 

In this example, we use the simplest parameter update method:

.. math::

    W = W - lr \cdot \nabla W

where :math:`\nabla W` is ``- delta_w * scale``. Thus, the optimizer will apply \
``weight.data = weight.data - lr * weight.grad = weight.data + lr * delta_w * scale``.

We can implement the above parameter update method by the plain :class:`torch.optim.SGD` with ``momentum=0.``:

.. code-block:: python

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.)

Then we create the input spikes and set ``STDPLearner``:

.. code-block:: python

    in_spike = (torch.rand([T, N, 1]) > 0.7).float()
    stdp_learner = learning.STDPLearner(step_mode='s', synapse=net[0], sn=net[1], tau_pre=tau_pre, tau_post=tau_post,
                                        f_pre=f_weight, f_post=f_weight)

Then we send data to the network. Note that to plot the figure, we will ``squeeze()`` the data, which reshape them from ``shape = [T, N, 1]`` \
to ``shape = [T]``:

.. code-block:: python

    out_spike = []
    trace_pre = []
    trace_post = []
    weight = []
    with torch.no_grad():
        for t in range(T):
            optimizer.zero_grad()
            out_spike.append(net(in_spike[t]).squeeze())
            stdp_learner.step(on_grad=True)  # add ``- delta_w * scale`` on grad
            optimizer.step()
            weight.append(net[0].weight.data.clone().squeeze())
            trace_pre.append(stdp_learner.trace_pre.squeeze())
            trace_post.append(stdp_learner.trace_post.squeeze())

    in_spike = in_spike.squeeze()
    out_spike = torch.stack(out_spike)
    trace_pre = torch.stack(trace_pre)
    trace_post = torch.stack(trace_post)
    weight = torch.stack(weight)

The complete codes are available at ``spikingjelly/activation_based/examples/stdp_trace.py``:

Let us plot ``in_spike, out_spike, trace_pre, trace_post, weight``:

.. image:: ../_static/tutorials/activation_based/stdp/trace.*
    :width: 100%

This figure is similar to Fig.3 in [#Trace]_  (note that they use `j` as the pre neuron and `i` as the post neuron, while we use the opposite symbol):

.. image:: ../_static/tutorials/activation_based/stdp/trace_paper_fig3.*
    :width: 100%


.. [#STDP] Bi, Guo-qiang, and Mu-ming Poo. "Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type." Journal of neuroscience 18.24 (1998): 10464-10472.

.. [#STDP_figure] Froemke, Robert C., et al. "Contribution of individual spikes in burst-induced long-term synaptic modification." Journal of neurophysiology (2006).

.. [#Trace] Morrison, Abigail, Markus Diesmann, and Wulfram Gerstner. "Phenomenological models of synaptic plasticity based on spike timing." Biological cybernetics 98.6 (2008): 459-478.