Clock driven: Use convolutional SNN to identify Fashion-MNIST
=============================================================================================

Author: `fangwei123456 <https://github.com/fangwei123456>`_

Translator: `YeYumin <https://github.com/YEYUMIN>`_

In this tutorial, we will build a convolutional spike neural network to classify the `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`__ dataset.
The Fashion-MNIST dataset has the same format as the MNIST dataset, and both are ``1 * 28 * 28`` grayscale images.

Network structure
----------------------------

Most of the common convolutional neural networks in ANN are in the form of convolution + fully-connected layers.
We also use a similar structure in SNN. Let us import modules, inherit ``torch.nn.Module`` to define our network:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from spikingjelly.clock_driven import neuron, functional, surrogate, layer
    from torch.utils.tensorboard import SummaryWriter
    import os
    import time
    import argparse
    import numpy as np
    from torch.cuda import amp
    _seed_ = 2020
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)

    class PythonNet(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T

Then we add convolutional layers and a fully-connected layers to ``PythonNet``. We add two Conv-BN-Pooling::

.. code-block:: python

    self.conv = nn.Sequential(
        nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        nn.MaxPool2d(2, 2),  # 14 * 14

        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        nn.MaxPool2d(2, 2)  # 7 * 7
        )

The input with ``shape=[N, 1, 28, 28]`` will be converted to spikes with ``shape=[N, 128, 7, 7]``.

Such convolutional layers can actually function as an encoder: in the previous tutorial (classify MNIST), we used a
Poisson encoder to encode pictures into spikes. However, we can directly send the picture
to the SNN. In this case, the first spike neurons layer (SN) and the layers before SN can be regarded as an
auto-encoder with learnable parameters. Specifically, teh auto-encoder is composed of the following layers:

.. code-block:: python

    nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    neuron.IFNode(surrogate_function=surrogate.ATan())

These layers receive images as input and output spikes, which can be regarded as an encoder.

Next, we add two fully-connected layers as the classifier. There are 10 neurons in output layer because the classes number
in Fashion-MNIST is 10.

.. code-block:: python

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128 * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.Linear(128 * 4 * 4, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

Now let us define the forward function.

.. code-block:: python

    def forward(self, x):
        x = self.static_conv(x)

        out_spikes_counter = self.fc(self.conv(x))
        for t in range(1, self.T):
            out_spikes_counter += self.fc(self.conv(x))

        return out_spikes_counter / self.T

Avoid Duplicated Computing
--------------------------------

We can train this network directly, just like the previous MNIST classification. But if we re-examine the structure of
the network, we can find that some calculations are duplicated. For the first two layers of the network (the highlighted
part of the following codes):

.. code-block:: python
    :emphasize-lines: 2, 3

    self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 7 * 7
        )

The input images are static and do not change with ``t``. But they will be involved in ``for`` loop. At each time-step,
they will flow through the first two layers with the same calculation. We can remove them from ``for`` loop in time-steps.
The complete codes are:

.. code-block:: python

    class PythonNet(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T

            self.static_conv = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
            )

            self.conv = nn.Sequential(
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                nn.MaxPool2d(2, 2),  # 14 * 14

                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                nn.MaxPool2d(2, 2)  # 7 * 7

            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 7 * 7, 128 * 4 * 4, bias=False),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
                nn.Linear(128 * 4 * 4, 10, bias=False),
                neuron.IFNode(surrogate_function=surrogate.ATan()),
            )


        def forward(self, x):
            x = self.static_conv(x)

            out_spikes_counter = self.fc(self.conv(x))
            for t in range(1, self.T):
                out_spikes_counter += self.fc(self.conv(x))

            return out_spikes_counter / self.T

We put these stateless layers to ``self.static_conv`` to avoid duplicated calculations.

Training network
----------------------------
The complete codes are available at :class:`spikingjelly.clock_driven.examples.conv_fashion_mnist`. The tarining arguments are:

.. code-block:: shell

    Classify Fashion-MNIST

    optional arguments:
      -h, --help            show this help message and exit
      -T T                  simulating time-steps
      -device DEVICE        device
      -b B                  batch size
      -epochs N             number of total epochs to run
      -j N                  number of data loading workers (default: 4)
      -data_dir DATA_DIR    root dir of Fashion-MNIST dataset
      -out_dir OUT_DIR      root dir for saving logs and checkpoint
      -resume RESUME        resume from the checkpoint path
      -amp                  automatic mixed precision training
      -cupy                 use cupy neuron and multi-step forward mode
      -opt OPT              use which optimizer. SDG or Adam
      -lr LR                learning rate
      -momentum MOMENTUM    momentum for SGD
      -lr_scheduler LR_SCHEDULER
                            use which schedule. StepLR or CosALR
      -step_size STEP_SIZE  step_size for StepLR
      -gamma GAMMA          gamma for StepLR
      -T_max T_MAX          T_max for CosineAnnealingLR

The checkpoint will be saved in the same level directory of the ``tensorboard`` log file. The server for training this
network uses `Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz` CPU and `GeForce RTX 2080 Ti` GPU.

.. code-block:: shell

    (pytorch-env) root@e8b6e4800dae4011eb0918702bd7ddedd51c-fangw1598-0:/# python -m spikingjelly.clock_driven.examples.conv_fashion_mnist -opt SGD -data_dir /userhome/datasets/FashionMNIST/ -amp

    Namespace(T=4, T_max=64, amp=True, b=128, cupy=False, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)
    PythonNet(
      (static_conv): Sequential(
        (0): Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv): Sequential(
        (0): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=False
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=False
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (fc): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Linear(in_features=6272, out_features=2048, bias=False)
        (2): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=False
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (3): Linear(in_features=2048, out_features=10, bias=False)
        (4): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=False
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
      )
    )
    Mkdir ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp.
    Namespace(T=4, T_max=64, amp=True, b=128, cupy=False, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)
    ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp
    epoch=0, train_loss=0.028124165828697957, train_acc=0.8188267895299145, test_loss=0.023525000348687174, test_acc=0.8633, max_test_acc=0.8633, total_time=16.86261749267578
    Namespace(T=4, T_max=64, amp=True, b=128, cupy=False, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)
    ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp
    epoch=1, train_loss=0.018544567498163536, train_acc=0.883613782051282, test_loss=0.02161250041425228, test_acc=0.8745, max_test_acc=0.8745, total_time=16.618073225021362
    Namespace(T=4, T_max=64, amp=True, b=128, cupy=False, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)

    ...

    ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp
    epoch=62, train_loss=0.0010829827882937538, train_acc=0.997512686965812, test_loss=0.011441250185668468, test_acc=0.9316, max_test_acc=0.933, total_time=15.976636171340942
    Namespace(T=4, T_max=64, amp=True, b=128, cupy=False, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)
    ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp
    epoch=63, train_loss=0.0010746361010835525, train_acc=0.9977463942307693, test_loss=0.01154562517106533, test_acc=0.9296, max_test_acc=0.933, total_time=15.83976149559021

After running 100 rounds of training, the correct rates on the training batch and test set are as follows:

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/train.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/test.*
    :width: 100%

After training for 64 epochs, the highest test set accuracy rate can reach 93.3%, which is a very good accuracy for
SNN. It is only slightly lower than ResNet18 (93.3%) with Normalization, random horizontal flip, random vertical flip,
random translation and random rotation in the BenchMark `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`__.

Visual Encoder
------------------------------------
As we said in the above text, the first spike neurons layer (SN) and the layers before SN can be regarded as an auto-encoder with learnable parameters. Specifically, it is the highlighted part of our network shown below:

.. code-block:: python
    :emphasize-lines: 5, 6, 10

    class Net(nn.Module):
        def __init__(self, T):
            ...
            self.static_conv = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
            )

            self.conv = nn.Sequential(
                neuron.IFNode(surrogate_function=surrogate.ATan()),
            ...
            )

Now let's take a look at the output spikes of the trained encoder. Let's create a new python file, import related
modules, and redefine a data loader with ``batch_size=1``, because we want to view pictures one by one:

.. code-block:: python

    from matplotlib import pyplot as plt
    import numpy as np
    from spikingjelly.clock_driven.examples.conv_fashion_mnist import PythonNet
    from spikingjelly import visualizing
    import torch
    import torch.nn as nn
    import torchvision

    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=1,
        shuffle=True,
        drop_last=False)

We load net from the checkpoint:

.. code-block:: python

    net = torch.load('./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp/checkpoint_max.pth', 'cpu')['net']
    encoder = nn.Sequential(
        net.static_conv,
        net.conv[0]
    )
    encoder.eval()

Let us extract a image from the data set, send it to the encoder, and check the accumulated value :math:`\sum_{t} S_{t}` of the output spikes. In order to show clearly, we also normalize the pixel values of the output ``feature_map`` with linearly transformation to ``[0, 1]``.

.. code-block:: python

    with torch.no_grad():
        # every time all the data sets are traversed, test once on the test set
        for img, label in test_data_loader:
            fig = plt.figure(dpi=200)
            plt.imshow(img.squeeze().numpy(), cmap='gray')
            # Note that the size of the image input to the network is ``[1, 1, 28, 28]``, the 0th dimension is ``batch``, and the first dimension is ``channel``
            # therefore, when calling ``imshow``, first use ``squeeze()`` to change the size to ``[28, 28]``
            plt.title('Input image', fontsize=20)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            out_spikes = 0
            for t in range(net.T):
                out_spikes += encoder(img).squeeze()
                # the size of encoder(img) is ``[1, 128, 28, 28]``，the same use ``squeeze()`` transform size to ``[128, 28, 28]``
                if t == 0 or t == net.T - 1:
                    out_spikes_c = out_spikes.clone()
                    for i in range(out_spikes_c.shape[0]):
                        if out_spikes_c[i].max().item() > out_spikes_c[i].min().item():
                            # Normalize each feature map to make the display clearer
                            out_spikes_c[i] = (out_spikes_c[i] - out_spikes_c[i].min()) / (out_spikes_c[i].max() - out_spikes_c[i].min())
                    visualizing.plot_2d_spiking_feature_map(out_spikes_c, 8, 16, 1, None)
                    plt.title('$\\sum_{t} S_{t}$ at $t = ' + str(t) + '$', fontsize=20)
                    plt.show()

The following figure shows two input iamges and the cumulative spikes :math:`\sum_{t} S_{t}` encoded by the encoder at ``t=0`` and ``t=7``:

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/x0.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/y00.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/y07.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/x1.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/y10.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/y17.*
    :width: 100%

It can be found that the cumulative spikes :math:`\sum_{t} S_{t}` are very similar to the origin images, indicating that the encoder has strong coding ability.
