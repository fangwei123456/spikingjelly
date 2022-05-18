spikingjelly.clock_driven.ann2snn
=======================================
Author: `DingJianhao <https://github.com/DingJianhao>`_, `fangwei123456 <https://github.com/fangwei123456>`_

This tutorial focuses on ``spikingjelly.clock_driven.ann2snn``, introduce how to convert the trained feedforward ANN to SNN and simulate it on the SpikingJelly framework.

There are two sets of implementations in earlier implementations: ONNX-based and PyTorch-based. Due to the instability of ONNX, this version is an enhanced version of PyTorch, which natively supports complex topologies (such as ResNet). Let's have a look!

Theoretical basis of ANN2SNN
----------------------------

Compared with ANN, the generated pulses of SNN are discrete, which is conducive to efficient communication. Today, with the popularity of ANN, the direct training of SNN requires more resources. Naturally, we will think of using the now very mature ANN to convert to SNN, and hope that SNN can have similar performance. This involves the problem of how to build a bridge between ANN and SNN. Now the mainstream way of SNN is to use frequency encoding, so for the output layer, we will use the number of neuron output pulses to judge the category. Is there a relationship between the release rate and ANN?

Fortunately, there is a strong correlation between the nonlinear activation of ReLU neurons in ANN and the firing rate of IF neurons in SNN (reset by subtracting the threshold: math:`V_{threshold}`). this feature to convert. The neuron update method mentioned here is the Soft method mentioned in `Time-driven tutorial <https://spikingjelly.readthedocs.io/zh_CN/latest/clock_driven/0_neuron.html>`_.

Experiment: Relationship between IF neuron spiking frequency and input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We gave constant input to the IF neuron and observed its output spikes and spike firing frequency. First import the relevant modules, create a new IF neuron layer, determine the input and draw the input of each IF neuron :math:`x_{i}`:

.. code-block:: python

    import torch
    from spikingjelly.clock_driven import neuron
    from spikingjelly import visualizing
    from matplotlib import pyplot as plt
    import numpy as np

    plt.rcParams['figure.dpi'] = 200
    if_node = neuron.IFNode(v_reset=None)
    T = 128
    x = torch.arange(-0.2, 1.2, 0.04)
    plt.scatter(torch.arange(x.shape[0]), x)
    plt.title('Input $x_{i}$ to IF neurons')
    plt.xlabel('Neuron index $i$')
    plt.ylabel('Input $x_{i}$')
    plt.grid(linestyle='-.')
    plt.show()

.. image:: ../_static/tutorials/clock_driven/5_ann2snn/0.*
    :width: 100%

Next, send the input to the IF neuron layer, and run the ``T=128`` step to observe the pulses and pulse firing frequency of each neuron:

.. code-block:: python

    s_list = []
    for t in range(T):
        s_list.append(if_node(x).unsqueeze(0))

    out_spikes = np.asarray(torch.cat(s_list))
    visualizing.plot_1d_spikes(out_spikes, 'IF neurons\' spikes and firing rates', 't', 'Neuron index $i$')
    plt.show()

.. image:: ../_static/tutorials/clock_driven/5_ann2snn/1.*
    :width: 100%

It can be found that the frequency of the pulse firing is within a certain range, which is proportional to the size of the input :math:`x_{i}`.

Next, let's plot the firing frequency of the IF neuron against the input :math:`x_{i}` and compare it with :math:`\mathrm{ReLU}(x_{i})`:

.. code-block:: python

    plt.subplot(1, 2, 1)
    firing_rate = np.mean(out_spikes, axis=1)
    plt.plot(x, firing_rate)
    plt.title('Input $x_{i}$ and firing rate')
    plt.xlabel('Input $x_{i}$')
    plt.ylabel('Firing rate')
    plt.grid(linestyle='-.')

    plt.subplot(1, 2, 2)
    plt.plot(x, x.relu())
    plt.title('Input $x_{i}$ and ReLU($x_{i}$)')
    plt.xlabel('Input $x_{i}$')
    plt.ylabel('ReLU($x_{i}$)')
    plt.grid(linestyle='-.')
    plt.show()

.. image:: ../_static/tutorials/clock_driven/5_ann2snn/2.*
    :width: 100%

It can be found that the two curves are almost the same. It should be noted that the pulse frequency cannot be higher than 1, so the IF neuron cannot fit the input of the ReLU in the ANN is larger than 1.

Theoretical basis of ANN2SNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The literature [#f1]_ provides a theoretical basis for analyzing the conversion of ANN to SNN. The theory shows that the IF neuron in SNN is an unbiased estimator of ReLU activation function over time.

For the first layer of the neural network, the input layer, discuss the relationship between the firing rate of SNN neurons :math:`r` and the activation in the corresponding ANN. Assume that the input is constant as :math:`z \in [0,1]`.
For the IF neuron reset by subtraction, its membrane potential V changes with time as follows:

.. math::
	V_t=V_{t-1}+z-V_{threshold}\theta_t

Where:
:math:`V_{threshold}` is the firing threshold, usually set to 1.0. :math:`\theta_t` is the output spike. The average firing rate in the :math:`T` time steps can be obtained by summing the membrane potential:

.. math::
	\sum_{t=1}^{T} V_t= \sum_{t=1}^{T} V_{t-1}+z T-V_{threshold} \sum_{t=1}^{T}\theta_t

Move all the items containing :math:`V_t` to the left, and divide both sides by :math:`T`:

.. math::
	\frac{V_T-V_0}{T} = z - V_{threshold}  \frac{\sum_{t=1}^{T}\theta_t}{T} = z- V_{threshold}  \frac{N}{T}

Where :math:`N` is the number of pulses in the time step of :math:`T`, and :math:`\frac{N}{T}` is the issuing rate :math:`r`. Use :math:`z = V_{threshold} a`
which is:

.. math::
	r = a- \frac{ V_T-V_0 }{T V_{threshold}}

Therefore, when the simulation time step :math:`T` is infinite:

.. math::
	r = a (a>0)

Similarly, for the higher layers of the neural network, literature [#f1]_ further explains that the inter-layer firing rate satisfies:

.. math::
	r^l = W^l r^{l-1}+b^l- \frac{V^l_T}{T V_{threshold}}

For details, please refer to [#f1]_. The methods in ann2snn also mainly come from [#f1]_ .

Converting to spiking neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversion mainly solves two problems:

1. ANN proposes Batch Normalization for fast training and convergence. Batch normalization aims to normalize the ANN output to 0 mean, which is contrary to the properties of SNNs. Therefore, the parameters of BN can be absorbed into the previous parameter layers (Linear, Conv2d)

2. According to the transformation theory, the input and output of each layer of ANN need to be limited to the range of [0,1], which requires scaling the parameters (model normalization)

◆ BatchNorm parameter absorption

Assume that the parameters of BatchNorm are: math:`\gamma` (``BatchNorm.weight``), :math:`\beta` (``BatchNorm.bias``), :math:`\mu` (``BatchNorm. .running_mean``) ,
:math:`\sigma` (``BatchNorm.running_var``, :math:`\sigma = \sqrt{\mathrm{running\_var}}`). For specific parameter definitions, see
`torch.nn.BatchNorm1d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm1d>`_ .
Parameter modules (eg Linear) have parameters :math:`W` and :math:`b` . BatchNorm parameter absorption is to transfer the parameters of BatchNorm to :math:`W` and :math:`b` of the parameter module by operation, so that the output of the new module of data input is the same as when there is BatchNorm.
For this, the :math:`\bar{W}` and :math:`\bar{b}` formulas for the new model are expressed as:

.. math::
    \bar{W} = \frac{\gamma}{\sigma} W

.. math::
    \bar{b} = \frac{\gamma}{\sigma} (b - \mu) + \beta

◆ Model Normalization

For a parameter module, it is assumed that its input tensor and output tensor are obtained, the maximum value of its input tensor is: math:`\lambda_{pre}`, and the maximum value of its output tensor is: math:`\lambda `
Then, the normalized weight :math:`\hat{W}` is:

.. math::
     \hat{W} = W * \frac{\lambda_{pre}}{\lambda}

The normalized bias :math:`\hat{b}` is:

.. math::
     \hat{b} = \frac{b}{\lambda}

Although the distribution of the output of each layer of ANN obeys a certain distribution, there are often large outliers in the data, which will lead to a decrease in the overall neuron firing rate.
To address this, robust normalization adjusts the scaling factor from the maximum value of the tensor to the p-quantile of the tensor. The recommended quantile value in the literature is 99.9.

So far, what we have done with neural networks is numerically equivalent. The current model should perform the same as the original model.

In the conversion, we need to change the ReLU activation function in the original model into IF neurons.
For average pooling in ANN, we need to convert it to spatial downsampling. Since IF neurons can be equivalent to the ReLU activation function. Adding IF neurons or not after spatial downsampling has minimal effect on the results.
There is currently no very ideal solution for max pooling in ANNs. The best solution so far is to control the pulse channel [#f1]_ with a gating function based on momentum accumulated pulses. Here we still recommend using avgpool2d.
When simulating, according to the transformation theory, the SNN needs to input a constant analog input. Using a Poisson encoder will bring about a reduction in accuracy.

Implementation and optional configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ann2snn framework will receive another major update in April 2022. The two categories of parser and simulator have been cancelled. Using the converter class replaces the previous solution. The current scheme is more compact and has more room for transformation settings.

◆ Converter class
This class is used to convert ReLU's ANN to SNN. Three common patterns are implemented here.
The most common is the maximum current switching mode, which utilizes the upper and lower activation limits of the front and rear layers so that the case with the highest firing rate corresponds to the case where the activation achieves the maximum value. Using this mode requires setting the parameter mode to ``max``[#f2]_.
The 99.9% current switching mode utilizes the 99.9% activation quantile to limit the upper activation limit. Using this mode requires setting the parameter mode to ``99.9%``[#f1]_.
In the scaling conversion mode, the user needs to specify the scaling parameters into the mode, and the current can be limited by the activated maximum value after scaling. Using this mode requires setting the parameter mode to a float of 0-1.

Classify MNIST
--------------

Now we use ``ann2snn`` to build a simple convolutional network to classify the MNIST dataset.

First define our network structure (see ``ann2snn.sample_models.mnist_cnn``):

.. code-block:: python

    class ANN(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1),
                nn.BatchNorm2d(32, eps=1e-3),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),

                nn.Conv2d(32, 32, 3, 1),
                nn.BatchNorm2d(32, eps=1e-3),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),

                nn.Conv2d(32, 32, 3, 1),
                nn.BatchNorm2d(32, eps=1e-3),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),

                nn.Flatten(),
                nn.Linear(32, 10),
                nn.ReLU()
            )

        def forward(self,x):
            x = self.network(x)
            return x

Note: If you need to expand the tensor, define a ``nn.Flatten`` module in the network, and use the defined Flatten instead of the view function in the forward function.

Define our hyperparameters:

.. code-block:: python

    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda'
    dataset_dir = 'G:/Dataset/mnist'
    batch_size = 100
    T = 50

Here T is the inference time step used in inference for a while.

If you want to train, you also need to initialize the data loader, optimizer, loss function, for example:

.. code-block::python

    lr = 1e-3
    epochs = 10
    # define the loss function
    loss_function = nn.CrossEntropyLoss()
    # Use Adam optimizer
    optimizer = torch.optim.Adam(ann.parameters(), lr=lr, weight_decay=5e-4)

Train the ANN. In the example, our model is trained for 10 epochs. The test set accuracy changes during training are as follows:

.. code-block::python

    Epoch: 0 100%|██████████| 600/600 [00:05<00:00, 112.04it/s]
    Validating Accuracy: 0.972
    Epoch: 1 100%|██████████| 600/600 [00:05<00:00, 105.43it/s]
    Validating Accuracy: 0.986
    Epoch: 2 100%|██████████| 600/600 [00:05<00:00, 107.49it/s]
    Validating Accuracy: 0.987
    Epoch: 3 100%|██████████| 600/600 [00:05<00:00, 109.26it/s]
    Validating Accuracy: 0.990
    Epoch: 4 100%|██████████| 600/600 [00:05<00:00, 103.98it/s]
    Validating Accuracy: 0.984
    Epoch: 5 100%|██████████| 600/600 [00:05<00:00, 100.42it/s]
    Validating Accuracy: 0.989
    Epoch: 6 100%|██████████| 600/600 [00:06<00:00, 96.24it/s]
    Validating Accuracy: 0.991
    Epoch: 7 100%|██████████| 600/600 [00:05<00:00, 104.97it/s]
    Validating Accuracy: 0.992
    Epoch: 8 100%|██████████| 600/600 [00:05<00:00, 106.45it/s]
    Validating Accuracy: 0.991
    Epoch: 9 100%|██████████| 600/600 [00:05<00:00, 111.93it/s]
    Validating Accuracy: 0.991

After training the model, we quickly load the model to test the performance of the saved model:

.. code-block::python

    model.load_state_dict(torch.load('SJ-mnist-cnn_model-sample.pth'))
    acc = val(model, device, test_data_loader)
    print('ANN Validating Accuracy: %.4f' % (acc))

The output is as follows:

.. code-block::python

    100%|██████████| 200/200 [00:02<00:00, 89.44it/s]
    ANN Validating Accuracy: 0.9870

Converting with Converter is very simple, you only need to set the mode you want to use in the parameters. For example, to use MaxNorm, you need to define an ``ann2snn.Converter`` first, and forward the model to this object:

.. code-block::python

    model_converter = ann2snn.Converter(mode='max', dataloader=train_data_loader)
    snn_model = model_converter(model)

snn_model is the output SNN model.

Following this example, we define the modes as ``max``, ``99.9%``, ``1.0/2``, ``1.0/3``, ``1.0/4``, ``1.0/ 5`` case SNN transformation and separate inference T steps to get the accuracy.

.. code-block::python

    print('---------------------------------------------')
    print('Converting using MaxNorm')
    model_converter = ann2snn.Converter(mode='max', dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('Simulating...')
    mode_max_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_max_accs[-1]))

    print('---------------------------------------------')
    print('Converting using RobustNorm')
    model_converter = ann2snn.Converter(mode='99.9%', dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('Simulating...')
    mode_robust_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_robust_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/2 max(activation) as scales...')
    model_converter = ann2snn.Converter(mode=1.0 / 2, dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('Simulating...')
    mode_two_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_two_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/3 max(activation) as scales')
    model_converter = ann2snn.Converter(mode=1.0 / 3, dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('Simulating...')
    mode_three_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_three_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/4 max(activation) as scales')
    model_converter = ann2snn.Converter(mode=1.0 / 4, dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('Simulating...')
    mode_four_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_four_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/5 max(activation) as scales')
    model_converter = ann2snn.Converter(mode=1.0 / 5, dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('Simulating...')
    mode_five_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_five_accs[-1]))

Observe the control bar output:

.. code-block::python

    ---------------------------------------------
    Converting using MaxNorm
    100%|██████████| 600/600 [00:04<00:00, 128.25it/s] Simulating...
    100%|██████████| 200/200 [00:13<00:00, 14.44it/s] SNN accuracy (simulation 50 time-steps): 0.9777
    ---------------------------------------------
    Converting using RobustNorm
    100%|██████████| 600/600 [00:19<00:00, 31.06it/s] Simulating...
    100%|██████████| 200/200 [00:13<00:00, 14.75it/s] SNN accuracy (simulation 50 time-steps): 0.9841
    ---------------------------------------------
    Converting using 1/2 max(activation) as scales...
    100%|██████████| 600/600 [00:04<00:00, 126.64it/s] ]Simulating...
    100%|██████████| 200/200 [00:13<00:00, 14.90it/s] SNN accuracy (simulation 50 time-steps): 0.9844
    ---------------------------------------------
    Converting using 1/3 max(activation) as scales
    100%|██████████| 600/600 [00:04<00:00, 126.27it/s] Simulating...
    100%|██████████| 200/200 [00:13<00:00, 14.73it/s] SNN accuracy (simulation 50 time-steps): 0.9828
    ---------------------------------------------
    Converting using 1/4 max(activation) as scales
    100%|██████████| 600/600 [00:04<00:00, 128.94it/s] Simulating...
    100%|██████████| 200/200 [00:13<00:00, 14.47it/s] SNN accuracy (simulation 50 time-steps): 0.9747
    ---------------------------------------------
    Converting using 1/5 max(activation) as scales
    100%|██████████| 600/600 [00:04<00:00, 121.18it/s] Simulating...
    100%|██████████| 200/200 [00:13<00:00, 14.42it/s] SNN accuracy (simulation 50 time-steps): 0.9487
    ---------------------------------------------

The speed of model conversion can be seen to be very fast. Model inference speed of 200 steps takes only 11s to complete (GTX 2080ti).
Based on the time-varying accuracy of the model output, we can plot the accuracy for different settings.

.. code-block::python

    fig = plt.figure()
    plt.plot(np.arange(0, T), mode_max_accs, label='mode: max')
    plt.plot(np.arange(0, T), mode_robust_accs, label='mode: 99.9%')
    plt.plot(np.arange(0, T), mode_two_accs, label='mode: 1.0/2')
    plt.plot(np.arange(0, T), mode_three_accs, label='mode: 1.0/3')
    plt.plot(np.arange(0, T), mode_four_accs, label='mode: 1.0/4')
    plt.plot(np.arange(0, T), mode_five_accs, label='mode: 1.0/5')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Acc')
    plt.show()

.. image:: ../_static/tutorials/clock_driven/5_ann2snn/accuracy_mode.png

Different settings can get different results, some inference speed is fast, but the final accuracy is low, and some inference is slow, but the accuracy is high. Users can choose model settings according to their needs.

.. [#f1] Rueckauer B, Lungu I-A, Hu Y, Pfeiffer M and Liu S-C (2017) Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification. Front. Neurosci. 11:682.
.. [#f2] Diehl, Peter U. , et al. Fast classifying, high-accuracy spiking deep networks through weight and threshold balancing. Neural Networks (IJCNN), 2015 International Joint Conference on IEEE, 2015.
.. [#f3] Rueckauer, B., Lungu, I. A., Hu, Y., & Pfeiffer, M. (2016). Theory and tools for the conversion of analog to spiking convolutional neural networks. arXiv preprint arXiv:1612.04052.
.. [#f4] Sengupta, A., Ye, Y., Wang, R., Liu, C., & Roy, K. (2019). Going deeper in spiking neural networks: Vgg and residual architectures. Frontiers in neuroscience, 13, 95.
