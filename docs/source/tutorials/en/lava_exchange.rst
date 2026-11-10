Convert to Lava for Loihi Deployment
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

中文版： :doc:`../cn/lava_exchange`

Thanks to `AllenYolk <https://github.com/AllenYolk>`_ and `banzhuangonglxh <https://github.com/banzhuangonglxh>`_ for their contributions to `lava_exchange` 


Introduction of Lava
-------------------------------------------

`Lava <https://github.com/lava-nc/lava>`_ is a neuromorphic computing framework, which is mainly developed by Intel and supports deploying on Intel Loihi. Lava provides a sub-package `Lava DL <https://github.com/lava-nc/lava-dl>`_ \
for deep learning, which can be used to build and train deep SNNs.

To deploy SNNs on Loihi, we need to use Lava. SpikingJelly provides conversion modules to convert the SNN trained by SpikingJelly to the Lava SNN format. And then we can \
run this SNN on Loihi. The workflow is:

``SpikingJelly -> Lava DL -> Lava -> Loihi``

The modules related to Lava are defined in :mod:`spikingjelly.activation_based.lava_exchange`.

.. note::

    Lava-DL is optional. SpikingJelly does not install it, and there is no
    ``spikingjelly[lava]`` extra. ``TNX_to_NXT``, ``to_lava_neuron``,
    ``BlockContainer`` and the other conversion callables are defined only
    when ``import lava.lib.dl.slayer`` succeeds; otherwise those names are
    missing.

    PyPI ``lava-dl==0.6.0`` requires Python >=3.10,<3.11 and torch>=2.3.1,<2.4,
    while SpikingJelly 2.0.0 requires Python >=3.11 and torch>=2.6. These
    releases have incompatible dependency ranges; prepare a compatible
    Lava-DL and SpikingJelly environment before running the conversion examples.


Basic Conversion
-------------------------------------------

Data Format Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The default data format in Lava DL is ``shape = [N, *, T]``, where ``N`` is the batch dimension and ``T`` is the time-step dimension. \
However, the module of SpikingJelly in multi-step mode (``step_mode = 'm'``) uses the data format as ``shape = [T, N, *]``. Thus, ``lava_exchange`` provides two \
conversion functions, :func:`TNX_to_NXT <spikingjelly.activation_based.lava_exchange.TNX_to_NXT>` and :func:`NXT_to_TNX <spikingjelly.activation_based.lava_exchange.NXT_to_TNX>` for \
conversion between two formats. Here is an example:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import lava_exchange

    T = 6
    N = 4
    C = 2

    x_seq = torch.rand([T, N, C])

    x_seq_la = lava_exchange.TNX_to_NXT(x_seq)
    print(f'x_seq_la.shape=[N, C, T]={x_seq_la.shape}')

    x_seq_sj = lava_exchange.NXT_to_TNX(x_seq_la)
    print(f'x_seq_sj.shape=[T, N, C]={x_seq_sj.shape}')

The outputs are:

.. code-block:: shell

    x_seq_la.shape=[N, C, T]=torch.Size([4, 2, 6])
    x_seq_sj.shape=[T, N, C]=torch.Size([6, 4, 2])


Neuron Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can convert IF and LIF neurons. Both require ``v_reset=0.0``, and LIF
also requires ``decay_input=False``. Other settings raise ``ValueError``.


We can use :func:`to_lava_neuron <spikingjelly.activation_based.lava_exchange.to_lava_neuron>` to convert. Here is an example:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import lava_exchange, neuron

    if_sj = neuron.IFNode(v_threshold=1., v_reset=0., step_mode='m')
    if_la = lava_exchange.to_lava_neuron(if_sj)

    T = 8
    N = 2
    C = 1

    x_seq_sj = torch.rand([T, N, C])
    x_seq_la = lava_exchange.TNX_to_NXT(x_seq_sj)

    print('output of sj(reshaped to NXT):\n', lava_exchange.TNX_to_NXT(if_sj(x_seq_sj)))
    print('output of lava:\n', if_la(x_seq_la))

Compare the two printed spike tensors to inspect the converted output. Lava DL quantizes neuron dynamics, so inputs near the threshold can produce different spikes. Both the IF and LIF examples use unseeded ``torch.rand``, so values vary between runs.

Here is an example of using the LIF neuron:


.. code-block:: python

    import torch
    from spikingjelly.activation_based import lava_exchange, neuron

    if_sj = neuron.LIFNode(tau=50., decay_input=False, v_threshold=1., v_reset=0., step_mode='m')
    if_la = lava_exchange.to_lava_neuron(if_sj)

    T = 8
    N = 2
    C = 1

    x_seq_sj = torch.rand([T, N, C])
    x_seq_la = lava_exchange.TNX_to_NXT(x_seq_sj)

    print('output of sj:\n', lava_exchange.TNX_to_NXT(if_sj(x_seq_sj)))
    print('output of lava:\n', if_la(x_seq_la))


Synapse Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can convert 2D convolution, linear, and 2D pooling layers with these constraints:

* Convolution and linear layers must use ``bias=False``.
* Converted pooling layers compute sums without dividing by the number of elements in the window.

Here is an example:

.. code-block:: python

    from spikingjelly.activation_based import lava_exchange, layer

    conv = layer.Conv2d(3, 4, kernel_size=3, stride=1, bias=False)
    fc = layer.Linear(4, 2, bias=False)
    ap = layer.AvgPool2d(2, 2)

    conv_la = lava_exchange.conv2d_to_lava_synapse_conv(conv)
    fc_la = lava_exchange.linear_to_lava_synapse_dense(fc)
    sp_la = lava_exchange.avgpool2d_to_lava_synapse_pool(ap)

    print(f'conv_la={conv_la}')
    print(f'fc_la={fc_la}')
    print(f'sp_la={sp_la}')

The outputs are:

.. code-block:: shell

    The Lava-DL pooling layer applies sum pooling; returning a sum-pooling layer instead of average pooling.
    conv_la=Conv(3, 4, kernel_size=(3, 3, 1), stride=(1, 1, 1), bias=False)
    fc_la=Dense(4, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    sp_la=Pool(1, 1, kernel_size=(2, 2, 1), stride=(2, 2, 1), bias=False)


Almost all synapses in Lava DL are based on :class:`torch.nn.Conv3d`. Thus, when we print them, we will find that ``kernel_size`` and ``stride`` are tuples with \
three elements.



BlockContainer
-------------------------------------------
The workflow for using Lava DL is:

1. using `Blocks <https://lava-nc.org/lava-lib-dl/slayer/block/modules.html>`_  in Lava DL to build and train the deep SNN
2. exporting the SNN to the hdf5 file
3. using Lava to read the hdf5 file and rebuild the SNN, then the SNN can run on Loihi or the CPU-simulated Loihi

For more details, please refer to `Lava: Deep Learning <https://lava-nc.org/dl.html#deep-learning>`_.

`Blocks <https://lava-nc.org/lava-lib-dl/slayer/block/modules.html>`_ can be regarded as the ensemble of a synapse layer and a neuron layer. For example, \
:class:`lava.lib.dl.slayer.block.cuba.Conv` is composed of a convolutional layer and a CUBA (current-based) LIF neuron layer.

For deployment, ``Blocks`` quantizes synaptic weights and neuron dynamics.
The computation is ``quantize(synapse) + quantize(neuron)``.

SpikingJelly provides :class:`BlockContainer <spikingjelly.activation_based.lava_exchange.BlockContainer>` to mimic ``Blocks`` in Lava. The features of ``BlockContainer`` \
are as follows:

* supports surrogate gradient training
* quantizes synapses and neuron dynamics to align with Lava DL computation
* converts to :class:`lava.lib.dl.slayer.block`

``BlockContainer`` uses :class:`lava_exchange.CubaLIFNode <spikingjelly.activation_based.lava_exchange.CubaLIFNode>` and automatically converts
:class:`IFNode <spikingjelly.activation_based.neuron.IFNode>` and :class:`LIFNode <spikingjelly.activation_based.neuron.LIFNode>` arguments to it.
This conversion uses the default surrogate and reset-gradient settings of ``CubaLIFNode``. To specify these settings, pass a configured ``CubaLIFNode`` directly. For example:

.. code-block:: python

    from spikingjelly.activation_based import lava_exchange, layer, neuron

    fc_block_sj = lava_exchange.BlockContainer(
        synapse=layer.Linear(8, 1, bias=False),
        neu=neuron.IFNode(),
        step_mode='m'
    )

    print('fc_block_sj=\n', fc_block_sj)

    fc_block_la = fc_block_sj.to_lava_block()
    print('fc_block_la=\n', fc_block_la)

The outputs are:

.. code-block:: shell

    fc_block_sj=
    BlockContainer(
    (synapse): Linear(in_features=8, out_features=1, bias=False)
    (neuron): CubaLIFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=m
        (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
    )
    )
    fc_block_la=
    Dense(
    (neuron): Neuron()
    (synapse): Dense(8, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    )


MNIST CSNN Example
-------------------------------------------
You can use these blocks to build a convolutional SNN for MNIST, then convert the trained network to Lava DL.


The SNN is defined as:

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import lava_exchange, neuron, surrogate

    class MNISTNet(nn.Module):
        def __init__(self, channels: int = 16):
            super().__init__()
            self.conv_fc = nn.Sequential(
                lava_exchange.BlockContainer(
                    nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False),
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
                ),

                lava_exchange.BlockContainer(
                    nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False),
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
                ),
                # 14 * 14

                lava_exchange.BlockContainer(
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
                ),

                lava_exchange.BlockContainer(
                    nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False),
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
                ),

                # 7 * 7

                lava_exchange.BlockContainer(
                    nn.Flatten(),
                    None
                ),
                lava_exchange.BlockContainer(
                    nn.Linear(channels * 7 * 7, 128, bias=False),
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
                ),

                lava_exchange.BlockContainer(
                    nn.Linear(128, 10, bias=False),
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
                ),
            )

        def forward(self, x):
            return self.conv_fc(x)

Add the following ``to_lava`` method to ``MNISTNet``, at the same indentation level as ``forward``, so you can call it after training:

.. code-block:: python

    def to_lava(self):
        ret = []

        for i in range(self.conv_fc.__len__()):
            m = self.conv_fc[i]
            if isinstance(m, lava_exchange.BlockContainer):
                ret.append(m.to_lava_block())

        return nn.Sequential(*ret)


During training, ``lava_exchange.BlockContainer`` quantizes synaptic weights and neuron dynamics, which can affect accuracy. The following training-loop excerpt requires ``net``, ``args``, the optimizer, and data loaders from the complete script:

.. code-block:: python

    import torch.nn.functional as F
    from spikingjelly.activation_based import encoding, functional

    functional.set_step_mode(net, "m")
    encoder = encoding.PoissonEncoder(step_mode='m')
    # ...
    for img, label in train_data_loader:
        optimizer.zero_grad()
        img = img.to(args.device)
        label = label.to(args.device)
        img = img.unsqueeze(0).expand(args.T, -1, -1, -1, -1)

        fr = net(encoder(img)).mean(0)
        loss = F.cross_entropy(fr, label)
        loss.backward()
        optimizer.step()
        functional.reset_net(net)
        # ...


After training, we can convert this SNN to Lava DL and check the accuracy:

.. code-block:: python

    net_ladl = net.to_lava().to(args.device)
    net_ladl.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for img, label in test_data_loader:
            img = img.to(args.device)
            label = label.to(args.device)
            img = img.unsqueeze(0).expand(args.T, -1, -1, -1, -1)
            img = encoder(img)
            img = lava_exchange.TNX_to_NXT(img)
            fr = net_ladl(img).mean(-1)
            loss = F.cross_entropy(fr, label)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (fr.argmax(1) == label).float().sum().item()

    test_loss /= test_samples
    test_acc /= test_samples

    print('test acc[lava dl] =', test_acc)

Finally, we can export the SNN in Lava DL format to an hdf5 file, which can then be read by Lava. Lava can rebuild the SNN and run the SNN on Loihi, or the CPU-simulated Loihi.\
Refer to `Network Exchange (NetX) Library <https://lava-nc.org/dl.html#network-exchange-netx-library>`_ for more details.

The export function is:

.. code-block:: python

    import os
    import h5py

    def export_hdf5(net, filename):
        with h5py.File(filename, "w") as h:
            layer = h.create_group("layer")
            for i, b in enumerate(net):
                handle = layer.create_group(f"{i}")
                b.export_hdf5(handle)

    export_hdf5(net_ladl, os.path.join(args.out_dir, 'net_la.net'))

The complete codes are stored in :mod:`spikingjelly.activation_based.examples.lava_mnist`. The arguments are defined as:

.. code-block:: shell

    python -m spikingjelly.activation_based.examples.lava_mnist -h
    usage: lava_mnist.py [-h] [-T T] [-b B] [-device DEVICE] [-data-dir DATA_DIR]
                        [-channels CHANNELS] [-epochs EPOCHS] [-lr LR] [-out-dir OUT_DIR]

    options:
    -h, --help          show this help message and exit
    -T T                simulating time-steps
    -b B                batch size
    -device DEVICE      device
    -data-dir DATA_DIR  root dir of the MNIST dataset
    -channels CHANNELS  channels of CSNN
    -epochs EPOCHS      training epochs
    -lr LR              learning rate
    -out-dir OUT_DIR    path for saving weights


Run the following command to train the network, check conversion accuracy, and export an HDF5 file.

Reference results (MNIST, ``T=32``, 16 training epochs):

.. code-block:: shell

    python -m spikingjelly.activation_based.examples.lava_mnist -T 32 -device cuda:0 -b 128 -epochs 16 -data-dir /datasets/MNIST/ -lr 0.1 -channels 16
    Namespace(T=32, b=128, device='cuda:0', data_dir='/datasets/MNIST/', channels=16, epochs=16, lr=0.1, out_dir='./')
    Namespace(T=32, b=128, device='cuda:0', data_dir='/datasets/MNIST/', channels=16, epochs=16, lr=0.1, out_dir='./')
    epoch = 0, train_loss = 1.7607, train_acc = 0.7245, test_loss = 1.5243, test_acc = 0.9443, max_test_acc = 0.9443

    # ...

    Namespace(T=32, b=128, device='cuda:0', data_dir='/datasets/MNIST/', channels=16, epochs=16, lr=0.1, out_dir='./')
    epoch = 15, train_loss = 1.4743, train_acc = 0.9881, test_loss = 1.4760, test_acc = 0.9855, max_test_acc = 0.9860
    finish training
    test acc[sj] = 0.9855
    test acc[lava dl] = 0.9863
    save net.state_dict() to ./net.pt
    save net_ladl.state_dict() to ./net_ladl.pt
    export net_ladl to ./net_la.net
