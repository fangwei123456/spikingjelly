Time-driven: Use convolutional SNN to identify Fashion-MNIST
============================================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

Translator: `YeYumin <https://github.com/YEYUMIN>`_

In this tutorial, we will build a convolutional pulse neural network to classify the `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ dataset.
The Fashion-MNIST data set has the same format as the MNIST data set, and both are ``1 * 28 * 28`` grayscale images.

Network structure
----------------------------

Most of the common convolutional neural networks in ANN are in the form of convolution + fully connected layers.
We also use a similar structure in SNN. Import related modules, inherit ``torch.nn.Module``, and define our network:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from spikingjelly.clock_driven import neuron, functional, surrogate, layer
    from torch.utils.tensorboard import SummaryWriter
    import sys
    if sys.platform != 'win32':
        import readline
    class Net(nn.Module):
        def __init__(self, tau, v_threshold=1.0, v_reset=0.0):

Next, we add a convolutional layer and a fully connected layer to the member variables of ``Net``. The developers of
``SpikingJelly`` found in experiments that for neurons in the convolutional layer, it is better to use ``IFNode`` for
static image data without time information. We add 2 convolution-BN-pooling layers:

.. code-block:: python

    self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 7 * 7
        )

After the input of ``1 * 28 * 28`` undergoes such a convolutional layer, an output pulse of ``128 * 7 * 7`` is obtained.

Such a convolutional layer can actually function as an encoder: in the previous tutorial, in the code of MNIST
classification, we used a Poisson encoder to encode pictures into pulses. In fact, we can directly send the picture
to the SNN. In this case, the first spike neuron layer and the previous layer in the SNN can be regarded as an
auto-encoder with learnable parameters. For example, these layers in the convolutional layer we just defined:

.. code-block:: python

    nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan())

This 3-layer network, which receives pictures as input and outputs pulses, can be regarded as an encoder.

Next, we define a 3-layer fully connected network and output the classification results. The fully connected
layer generally functions as a classifier, and the performance of using ``LIFNode`` will be better. Fashion-MNIST
has 10 categories, so the output layer is 10 neurons, in order to reduce over-fitting, we also use ``layer.Dropout``.
For more information about it, please refer to the API documentation.

.. code-block:: python

    self.fc = nn.Sequential(
        nn.Flatten(),
        layer.Dropout(0.7),
        nn.Linear(128 * 7 * 7, 128 * 3 * 3, bias=False),
        neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        layer.Dropout(0.7),
        nn.Linear(128 * 3 * 3, 128, bias=False),
        neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        nn.Linear(128, 10, bias=False),
        neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
    )

Next, define forward propagation. Forward propagation is very simple, first go through convolution and then go through full connection:

.. code-block:: python

    def forward(self, x):
        return self.fc(self.conv(x))

Avoid repeat counting
--------------------------------

We can train this network directly, just like the previous MNIST classification:

.. code-block:: python

        for img, label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()

            optimizer.zero_grad()

            # run the time of T，out_spikes_counter is the tensor of shape=[batch_size, 10]
            # record the number of pulse firings of 10 neurons in the output layer during the entire simulation duration
            for t in range(T):
                if t == 0:
                    out_spikes_counter = net(encoder(img).float())
                else:
                    out_spikes_counter += net(encoder(img).float())

            # out_spikes_counter / T obtain the pulse firing frequency of 10 neurons in the output layer during the simulation time
            out_spikes_counter_frequency = out_spikes_counter / T

            # the loss function is the pulse firing frequency of the neurons in the output layer, and the MSE of the true category
            # such a loss function will make the pulse firing frequency of the i-th neuron in the output layer approach 1 when the category i is input, and the pulse firing frequency of other neurons will approach 0
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            loss.backward()
            optimizer.step()
            # after optimizing the parameters once, the state of the network needs to be reset, because the neurons of SNN have "memory"
            functional.reset_net(net)

But if we re-examine the structure of the network, we can find that some calculations are repeated, for the first 2
layers of the network, the highlighted part of the following code:

.. code-block:: python
    :emphasize-lines: 2, 3

    self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 7 * 7
        )

The input image received by these two layers does not change with ``t`` , but in the ``for`` loop, each time ``img`` will
recalculate these two layers to get the same output. We extract these layers and encapsulate the time loop into the
network itself to facilitate calculation. The new network structure is fully defined as:

.. code-block:: python

    class Net(nn.Module):
        def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
            super().__init__()
            self.T = T

            self.static_conv = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
            )

            self.conv = nn.Sequential(
                neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
                nn.MaxPool2d(2, 2),  # 14 * 14

                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
                nn.MaxPool2d(2, 2)  # 7 * 7

            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                layer.Dropout(0.7),
                nn.Linear(128 * 7 * 7, 128 * 3 * 3, bias=False),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
                layer.Dropout(0.7),
                nn.Linear(128 * 3 * 3, 128, bias=False),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
                nn.Linear(128, 10, bias=False),
                neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            )


        def forward(self, x):
            x = self.static_conv(x)

            out_spikes_counter = self.fc(self.conv(x))
            for t in range(1, self.T):
                out_spikes_counter += self.fc(self.conv(x))

            return out_spikes_counter / self.T


For SNN whose input does not change with time, although the SNN is stateful as a whole, the first few layers of the
network may not be stateful. We can extract these layers separately and put them out of the time loop to avoid
additional calculations .

Training network
----------------------------
The complete code is located in `clock_driven/examples/conv_fashion_mnist.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/conv_fashion_mnist.py>`_.
It can also be run directly from the command line.The network with the highest accuracy of the test set during the
training process will be saved in the same level directory of the ``tensorboard`` log file.

.. code-block:: python

    >>> from spikingjelly.clock_driven.examples import conv_fashion_mnist
    >>> conv_fashion_mnist.main()
    输入运行的设备，例如“cpu”或“cuda:0”
     input device, e.g., "cpu" or "cuda:0": cuda:9
    输入保存Fashion MNIST数据集的位置，例如“./”
     input root directory for saving Fashion MNIST dataset, e.g., "./": ./fmnist
    输入batch_size，例如“64”
     input batch_size, e.g., "64": 64
    输入学习率，例如“1e-3”
     input learning rate, e.g., "1e-3": 1e-3
    输入仿真时长，例如“8”
     input simulating steps, e.g., "8": 8
    输入LIF神经元的时间常数tau，例如“2.0”
     input membrane time constant, tau, for LIF neurons, e.g., "2.0": 2.0
    输入训练轮数，即遍历训练集的次数，例如“100”
     input training epochs, e.g., "100": 100
    输入保存tensorboard日志文件的位置，例如“./”
     input root directory for saving tensorboard logs, e.g., "./": ./logs_conv_fashion_mnist

After running 100 rounds of training, the correct rates on the training batch and test set are as follows:

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/train.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/4_conv_fashion_mnist/test.*
    :width: 100%

After training for 100 epochs, the highest test set accuracy rate can reach 94.3%, which is a very good performance for
SNN, only slightly lower than the use of Normalization, random horizontal flip, random vertical flip, random translation
in the BenchMark of `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_, ResNet18 of random rotation has a 94.9% correct rate.

Visual encoder
------------------------------------

As we said in the previous article, if the data is directly fed into the SNN, the first spike neuron layer and the layers
before it can be regarded as a learnable encoder. Specifically, it is the highlighted part of our network as shown below:

.. code-block:: python
    :emphasize-lines: 5, 6, 10

    class Net(nn.Module):
        def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
            ...
            self.static_conv = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
            )

            self.conv = nn.Sequential(
                neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            ...

Now let's take a look at the coding effect of the trained encoder. Let's create a new python file, import related
modules, and redefine a data loader with ``batch_size=1``, because we want to view one picture by one:

.. code-block:: python

    from matplotlib import pyplot as plt
    import numpy as np
    from spikingjelly.clock_driven.examples.conv_fashion_mnist import Net
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

Load the trained network from the location where the network is saved, that is, under the ``log_dir`` directory, and extract the encoder. Just run on the CPU:

.. code-block:: python

    net = torch.load('./logs_conv_fashion_mnist/net_max_acc.pt', 'cpu')
    encoder = nn.Sequential(
        net.static_conv,
        net.conv[0]
    )
    encoder.eval()

Next, extract a picture from the data set, send it to the encoder, and check the accumulated value :math:`\sum_{t} S_{t}` of the output
pulse. In order to display clearly, we also normalized the pixel value of the output ``feature_map``, and linearly transformed
the value range to ``[0, 1]``.

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

The following shows two input pictures and the cumulative pulse :math:`\sum_{t} S_{t}` output by the encoder at the begin time of ``t=0`` and the end time ``t=7``:

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

Observation shows that the cumulative output pulse :math:`\sum_{t} S_{t}` of the encoder is very close to the contour of the original image.
It seems that this kind of self-learning pulse encoder has strong coding ability.
