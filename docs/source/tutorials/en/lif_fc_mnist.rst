Single fully connected SNN to classify MNIST
========================================================

Author: `Yanqi-Chen <https://github.com/Yanqi-Chen>`__

中文版： :doc:`../cn/lif_fc_mnist`

Translator: `Lv Liuzhenghao <https://github.com/Lyu6PosHao>`_

You will build a fully connected SNN, encode input spikes with a Poisson encoder, and train it with surrogate gradients to classify MNIST digits.

Build a simple SNN
-------------------------

When building neural networks with PyTorch, we can simply use ``nn.Sequential`` to stack layers to get a feedforward network, where input data will flow through each layer in order to get the output.

The `MNIST dataset <http://yann.lecun.com/exdb/mnist/>`__ contains :math:`28\times 28` 8-bit grayscale images labeled with digits from 0 to 9. This single-layer ANN shows the starting structure:

.. code-block:: python

    import torch
    import torch.nn as nn
    import time
    import torch.nn.functional as F
    from torch.cuda import amp
    from spikingjelly.activation_based import layer, neuron, surrogate, encoding, functional

    nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10, bias=False),
        nn.Softmax(dim=1)
        )

Replace the ANN's Softmax with an LIF neuron to obtain the SNN used in this example. The connection layers come from :mod:`spikingjelly.activation_based.layer`:

.. code-block:: python

    tau = 2.0
    net = nn.Sequential(
        layer.Flatten(),
        layer.Linear(28 * 28, 10, bias=False),
        neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
        )

The membrane potential constant :math:`\tau` is set by ``tau`` , and ``surrogate.ATan`` is used as the surrogate gradient function.

Train the SNN
----------------

Training parameters like learning rate and other configurations need to be set:

Adam is used as the optimizer by default, and the poisson encoder is used to encode input images as spikes.

.. code-block:: python

    lr = 1e-3
    # Use Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # Use PoissonEncoder
    encoder = encoding.PoissonEncoder()

See :mod:`spikingjelly.activation_based.encoding` for encoder usage.

The training loop has three steps:

1. The output of the spiking neuron is binary, and the output of a single run is easily disturbed by the noise caused by coding. Therefore, it is generally to consider **the firing rate**  of the output layer in a period of time as the output of SNN. The value of the firing rate indicates the response intensity of the corresponding category. So we should run the SNN for a period of time ``T`` and take **the average firing rate** in ``T`` as classifying evidence.

2. The ideal outcome is that except for the proper neurons **firing at the highest rate**, the other neurons **keep silent**. Here we use MSE loss with one-hot firing-rate targets.

3. After each network simulation, the network state should be reset by ``functional.reset_net(net)``.

The core training codes are as follows. This excerpt uses the arguments, DataLoader,
and AMP scaler setup from :mod:`spikingjelly.activation_based.examples.lif_fc_mnist`:

.. code-block:: python

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            # Mixed-precision training
            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.
                    # Run T time steps
                    for t in range(args.T):
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    out_fr = out_fr / args.T
                    # out_fr is tensor whose shape is [batch_size, 10]
                    # The firing rate of 10 neurons in the output layer was recorded during the whole simulation period
                    loss = F.mse_loss(out_fr, label_onehot)
                    # The loss function is the MSE between the firing rate of the output layer and the true category.
                    # The loss function will cause the firing rate of the correct neuron in the output layer to approach 1 when the label i is given, and the firing rate of the other neurons to approach 0.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            # The correct rate is calculated as follows. The subscript i of the neuron with the highest firing rate in the output layer is considered as the result of classification.
            train_acc += (out_fr.argmax(1) == label).float().sum().item()
            
            # After optimizing the parameters, the state of the network should be reset because the neurons of the SNN have “memory”.
            functional.reset_net(net)

The complete example is :mod:`spikingjelly.activation_based.examples.lif_fc_mnist`, which writes TensorBoard logs. Start by viewing its command-line arguments:

.. code-block:: shell

    python -m spikingjelly.activation_based.examples.lif_fc_mnist --help

The memory required to train such an SNN is linearly related to the number of simulation steps ``T``.
A larger ``T`` does not necessarily improve training. When ``T`` is too large, the SNN unfolds in time and becomes a very deep network,
which can cause gradients to vanish or explode during BPTT.

In addition, since we use the poisson encoder, a large ``T`` is needed to ensure that the coding noise is not too large.

Training results
------------------------

We set ``tau=2.0,T=100,batch_size=64,lr=1e-3`` , the corresponding command is:

.. code-block:: shell

    python -m spikingjelly.activation_based.examples.lif_fc_mnist -tau 2.0 -T 100 -device cuda:0 -b 64 -epochs 100 -data-dir <PATH to MNIST> -amp -opt adam -lr 1e-3 -j 8

Reference results (commit ``5ca5c06e``):

This run used mixed precision. After 100 epochs, the highest test accuracy was 92.9%. The script saves checkpoints and TensorBoard logs, then writes ``v_t_array.npy`` and ``s_t_array.npy`` after training. The accuracy curve is:

.. image:: ../../_static/tutorials/lif_fc_mnist/acc.*
    :width: 100%

Select the first image in the test dataset:

.. image:: ../../_static/tutorials/lif_fc_mnist/input.png

The classification results are obtained by using the trained model:

.. code-block:: shell

   Firing rate: [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]

The saved membrane-voltage and spike sequences can be plotted in these two views:

.. image:: ../../_static/tutorials/lif_fc_mnist/1d_spikes.*
    :width: 100%

.. image:: ../../_static/tutorials/lif_fc_mnist/2d_heatmap.*
    :width: 100%

The sample is classified as digit 7, and only the output neuron for class 7 fires. The complete training code is in `activation_based/examples/lif_fc_mnist.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/lif_fc_mnist.py>`_.
