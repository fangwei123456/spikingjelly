Convert to Lava for Loihi Deployment
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

Thanks to `AllenYolk <https://github.com/AllenYolk>`_ and `banzhuangonglxh <https://github.com/banzhuangonglxh>`_ for their contributions to `lava_exchange` 


Introduction of Lava
-------------------------------------------

`Lava <https://github.com/lava-nc/lava>`_ is a neuromorphic computing framework, which is mainly developed by Intel and supports deploying on Intel Loihi. Lava provides a sub-package `Lava DL <https://github.com/lava-nc/lava-dl>`_ \
for deep learning, which can be used to build and train deep SNNs.

To deploy SNNs on Loihi, we need to use Lava. SpikingJelly provides conversion modules to convert the SNN trained by SpikingJelly to the Lava SNN format. And then we can \
run this SNN on Loihi. The workflow is:

``SpikingJelly -> Lava DL -> Lava -> Loihi``

The modules related to Lava are defined in :class:`spikingjelly.activation_based.lava_exchange`.


Basic Conversion
-------------------------------------------

Data Format Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The default data format in Lava DL is ``shape = [N, *, T]``, where ``N`` is the batch dimension and ``T`` is the time-step dimension. \
However, the module of SpikingJelly in multi-step mode (``step_mode = 'm'``) uses the data format as ``shape = [T, N, *]``. Thus, ``lava_exchange`` provides two \
conversion functions, :class:`TNX_to_NXT <spikingjelly.activation_based.lava_exchange.TNX_to_NXT>` and :class:`NXT_to_TNX <spikingjelly.activation_based.lava_exchange.NXT_to_TNX>` for \
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
Neurons in SpikingJelly can be converted to neurons in Lava DL. Due to the limited time and energy of developers, SpikingJelly only supports the IF neuron \
and the LIF neuron, which are two of the most popular neurons in spiking deep learning. Other neurons will be considered to add according to user requirements.


We can use :class:`to_lava_neuron <spikingjelly.activation_based.lava_exchange.to_lava_neuron>` to convert. Here is an example:

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

The outputs are:

.. code-block:: shell

    output of sj(reshaped to NXT):
    tensor([[[0., 0., 1., 0., 1., 0., 0., 0.]],

            [[0., 1., 0., 1., 0., 1., 0., 1.]]])
    output of lava:
    tensor([[[0., 0., 1., 0., 1., 0., 0., 0.]],

            [[0., 1., 0., 1., 0., 1., 0., 1.]]])

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

The outputs are:

.. code-block:: shell

    output of sj:
    tensor([[[0., 1., 0., 1., 0., 0., 1., 0.]],

            [[0., 0., 1., 0., 0., 1., 0., 1.]]])
    output of lava:
    tensor([[[0., 1., 0., 1., 0., 0., 1., 0.]],

            [[0., 0., 1., 0., 0., 1., 0., 1.]]])

Synapse Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The frequently-used convolutional layer, linear layer, and pooling layer can be converted. Note that

* bias is not supported
* Lava only supports sum pooling, which can be regarded as average pooling without average

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

    WARNING:root:The lava slayer pool layer applies sum pooling, rather than average pooling. `avgpool2d_to_lava_synapse_pool` will return a sum pooling layer.
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
:class:`lava.lib.dl.slayer.block.cuba.Conv` is composed of a convolutional layer and a CUDA LIF neuron layer.

Note that ``Blocks`` is designed for SNN deployment. Thus, synapses and neuronal dynamics are quantized in ``Blocks``. Thus, ``Blocks`` is not a simple \
``synapse + neuron ``, but ``quantize(synapse) + quantize(neuron)``.

SpikingJelly provides :class:`BlockContainer <spikingjelly.activation_based.lava_exchange.BlockContainer>` to mimic ``Blocks`` in Lava. The features of ``BlockContainer`` \
are as follows:

* supports for surrogate gradient training
* synapses and neuronal dynamics are quantized
* the outputs are identical to ``Blocks`` of Lava DL when giving the same inputs
* supports for converting to :class:`lava.lib.dl.slayer.block`

For the moment, ``BlockContainer`` only supports for :class:`lava_exchange.CubaLIFNode <spikingjelly.activation_based.lava_exchange.CubaLIFNode>`. But it also \
supports for converting :class:`IFNode <spikingjelly.activation_based.neuron.IFNode>` or :class:`LIFNode <spikingjelly.activation_based.neuron.LIFNode>` \
in init args to ``CubaLIFNode``. Here is an example:


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
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=m, backend=torch
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
Now let us train a spiking convolutional SNN for classifying MNIST, and then convert this network to Lava DL format.


The SNN is defined as:

.. code-block:: python

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

We add a conversion function to convert the SNN to Lava DL format, which can be used after training:

.. code-block:: python

    def to_lava(self):
        ret = []

        for i in range(self.conv_fc.__len__()):
            m = self.conv_fc[i]
            if isinstance(m, lava_exchange.BlockContainer):
                ret.append(m.to_lava_block())

        return nn.Sequential(*ret)


Then, we train this SNN. The training process has no much difference from other SNNs. Note that the quantization inside ``lava_exchange.BlockContainer`` will 
reduce accuracy. An example of the training codes is:

.. code-block:: python

    encoder = encoding.PoissonEncoder(step_mode='m')
    # ...
    for img, label in train_data_loader:
        optimizer.zero_grad()
        img = img.to(args.device)
        label = label.to(args.device)
        img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)

        fr = net(encoder(img)).mean(0)
        loss = F.cross_entropy(fr, label)
        loss.backward()
        optimizer.step()
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
            img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
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

    def export_hdf5(net, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(net):
            handle = layer.create_group(f'{i}')
            b.export_hdf5(handle)

    export_hdf5(net_ladl, os.path.join(args.out_dir, 'net_la.net'))

The complete codes are stored in :class:`spikingjelly.activation_based.examples.lava_mnist`. The arguments are defined as:

.. code-block:: shell

    (lava-env) wfang@mlg-ThinkStation-P920:~/tempdir/w1$ python -m spikingjelly.activation_based.examples.lava_mnist -h
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


When we run this script, it will firstly train a SNN, then convert the SNN to Lava DL format and run an inference, and finally export the SNN to the hdf5 file:

.. code-block:: shell

    (lava-env) wfang@mlg-ThinkStation-P920:~/tempdir/w1$ python -m spikingjelly.activation_based.examples.lava_mnist -T 32 -device cuda:0 -b 128 -epochs 16 -data-dir /datasets/MNIST/ -lr 0.1 -channels 16
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

