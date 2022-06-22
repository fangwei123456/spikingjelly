Recurrent Connection and Stateful Synapse
============================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

Self-connected Modules
-----------------------------
Recurrent connection is the connection from outputs to inputs, e.g., the SRNN(recurrent networks of spiking neurons) in [#Effective]_, which are shown in the following figure:

.. image:: ../_static/tutorials/activation_based/15_recurrent_connection_and_stateful_synapse/SRNN_example.*
    :width: 100%

We can add recurrent connection to modules by SpikingJelly easily. Considering the most simple case that we add a recurrent connection to the spiking neruons layer to make its \
outputs :math:`s[t]` at time-step :math:`t` add to the external input :math:`x[t+1]` as the input to the neuron at the next time-step. We can use :class:`spikingjelly.activation_based.layer.ElementWiseRecurrentContainer` to implement this idea.\ 
:class:`ElementWiseRecurrentContainer <spikingjelly.activation_based.layer.ElementWiseRecurrentContainer>` is a container that add a recurrent connection to any ``sub_module``.\ 
The connection can be specified as a user-defined element-wise operation :math:`z=f(x, y)`. Denote :math:`x[t]` as the external input for the whole module (container and ``sub_module``) at time-step :math:`t`, :math:`i[t]` and :math:`y[t]` are the input and output of ``sub_module`` \
(note that :math:`y[t]` is also the outputs of the whole module), then we can get

.. math::

    i[t] = f(x[t], y[t-1])

where :math:`f` is a user-defined element-wise function. We regard :math:`y[-1] = 0`.

Let us use ``ElementWiseRecurrentContainer`` to wrap one IF neuron. We set the element-wise function as addition:

.. math::

    i[t] = x[t] + y[t-1].

The external intpus are :math:`x[t]=[1.5, 0, ..., 0]`:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import layer, functional, neuron

    T = 8
    N = 1

    def element_wise_add(x, y):
        return x + y

    net = layer.ElementWiseRecurrentContainer(neuron.IFNode(), element_wise_add)
    print(net)
    x = torch.zeros([T, N])
    x[0] = 1.5
    for t in range(T):
        print(t, f'x[t]={x[t]}, s[t]={net(x[t])}')

    functional.reset_net(net)

The outputs are:

.. code-block:: bash

    ElementWiseRecurrentContainer(
    element-wise function=<function element_wise_add at 0x00000158FC15ACA0>, step_mode=s
    (sub_module): IFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=s, backend=torch
        (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
    )
    )
    0 x[t]=tensor([1.5000]), s[t]=tensor([1.])
    1 x[t]=tensor([0.]), s[t]=tensor([1.])
    2 x[t]=tensor([0.]), s[t]=tensor([1.])
    3 x[t]=tensor([0.]), s[t]=tensor([1.])
    4 x[t]=tensor([0.]), s[t]=tensor([1.])
    5 x[t]=tensor([0.]), s[t]=tensor([1.])
    6 x[t]=tensor([0.]), s[t]=tensor([1.])
    7 x[t]=tensor([0.]), s[t]=tensor([1.])

We can find that even when :math:`t \ge 1`, :math:`x[t]=0`, the neuron can still fire spikes because of the recurrent connection.

We can use :class:`spikingjelly.activation_based.layer.LinearRecurrentContainer` to implement the more complex recurrent connection.

Stateful Synapse
-----------------------
Some papers, e.g., [#Unsupervised]_ and [#Exploiting]_ , use the stateful synapses. By placing :class:`spikingjelly.activation_based.layer.SynapseFilter` after the synapse to filter the output current, \
we can get the stateful synapse:

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import layer, functional, neuron

    stateful_conv = nn.Sequential(
        layer.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
        layer.SynapseFilter(tau=100.)
    )

Experiments on Sequential FashionMNIST
------------------------------------------
Now let us do some simple experiments on Sequential FashionMNIST to verify whether the recurrent connection or the stateful synapse can promote the network's \
ability on the memory task. The Sequential FashionMNIST dataset is a modified FashionMNIST dataset. Images will be sent to the network row by row or column by column, rather than \
be sent entirely. To classify correctly, the network should have good memory ability. We will send images column by column, which is similar to how humans read the book from left to right:

.. image:: ../_static/tutorials/activation_based/recurrent_connection_and_stateful_synapse/samples/a.*
    :width: 50%

The following figure shows the column that is being sent:

.. image:: ../_static/tutorials/activation_based/recurrent_connection_and_stateful_synapse/samples/b.*
    :width: 50%

First, let us import some packages:

.. code:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.datasets
    from spikingjelly.activation_based import neuron, surrogate, layer, functional
    from torch.cuda import amp
    import os, argparse
    from torch.utils.tensorboard import SummaryWriter
    import time
    import datetime
    import sys

Define the plain feedforward network ``PlainNet``:

.. code:: python

    class PlainNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                layer.Linear(28, 32),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                layer.Linear(32, 10),
                neuron.IFNode(surrogate_function=surrogate.ATan())
            )

        def forward(self, x: torch.Tensor):
            return self.fc(x).mean(0)

By adding a :class:`spikingjelly.activation_based.layer.SynapseFilter` behind the first spiking neurons layer of ``PlainNet``, we can get the network ``StatefulSynapseNet``:

.. code:: python

    class StatefulSynapseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                layer.Linear(28, 32),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                layer.SynapseFilter(tau=2., learnable=True),
                layer.Linear(32, 10),
                neuron.IFNode(surrogate_function=surrogate.ATan())
            )

        def forward(self, x: torch.Tensor):
            return self.fc(x).mean(0)

By adding a recurrent connection implemented by :class:`spikingjelly.activation_based.layer.LinearRecurrentContainer` to ``PlainNet``, we can get ``FeedBackNet``

.. code:: python

    class FeedBackNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc = nn.Sequential(
                layer.Linear(28, 32),
                layer.LinearRecurrentContainer(
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
                    in_features=32, out_features=32, bias=True
                ),
                layer.Linear(32, 10),
                neuron.IFNode(surrogate_function=surrogate.ATan())
            )

        def forward(self, x: torch.Tensor):
            return self.fc(x).mean(0)


The following figure shows the network structure of three networks:

.. image:: ../_static/tutorials/activation_based/recurrent_connection_and_stateful_synapse/ppt/nets.png
    :width: 100%

The complete codes are saved in `spikingjelly.activation_based.examples.rsnn_sequential_fmnist <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/rsnn_sequential_fmnist.py>`_. We can run by the following commands:

.. code:: shell

    usage: rsnn_sequential_fmnist.py [-h] [-model MODEL] [-device DEVICE] [-b B] [-epochs N] [-j N] [-data-dir DATA_DIR] [-out-dir OUT_DIR] [-resume RESUME] [-amp] [-cupy] [-opt OPT] [-momentum MOMENTUM] [-lr LR]

    Classify Sequential Fashion-MNIST

    optional arguments:
    -h, --help          show this help message and exit
    -model MODEL        use which model, "plain", "ss" (StatefulSynapseNet) or "fb" (FeedBackNet)
    -device DEVICE      device
    -b B                batch size
    -epochs N           number of total epochs to run
    -j N                number of data loading workers (default: 4)
    -data-dir DATA_DIR  root dir of Fashion-MNIST dataset
    -out-dir OUT_DIR    root dir for saving logs and checkpoint
    -resume RESUME      resume from the checkpoint path
    -amp                automatic mixed precision training
    -cupy               use cupy backend
    -opt OPT            use which optimizer. SDG or Adam
    -momentum MOMENTUM  momentum for SGD
    -lr LR              learning rate


Train three networks:

.. code:: shell

    python -m spikingjelly.activation_based.examples.rsnn_sequential_fmnist -device cuda:0 -b 256 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8 -model plain

    python -m spikingjelly.activation_based.examples.rsnn_sequential_fmnist -device cuda:0 -b 256 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8 -model fb

    python -m spikingjelly.activation_based.examples.rsnn_sequential_fmnist -device cuda:0 -b 256 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8 -model ss

The following figures show the accuracy curves during training:

.. image:: ../_static/tutorials/activation_based/recurrent_connection_and_stateful_synapse/rsnn_train_acc.*
    :width: 100%


.. image:: ../_static/tutorials/activation_based/recurrent_connection_and_stateful_synapse/rsnn_test_acc.*
    :width: 100%


We can find that both ``StatefulSynapseNet`` and ``FeedBackNet`` have higher accuracy than ``PlainNet``, indicating that recurrent connection and stateful synapse can promote the network's memory ability.

.. [#Effective] Yin B, Corradi F, Bohté S M. Effective and efficient computation with multiple-timescale spiking recurrent neural networks[C]//International Conference on Neuromorphic Systems 2020. 2020: 1-8.

.. [#Unsupervised] Diehl P U, Cook M. Unsupervised learning of digit recognition using spike-timing-dependent plasticity[J]. Frontiers in computational neuroscience, 2015, 9: 99.

.. [#Exploiting] Fang H, Shrestha A, Zhao Z, et al. Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network[J].
