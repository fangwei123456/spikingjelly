SpikingFlow.ann2snn
=======================================
Author: `DingJianhao <https://github.com/DingJianhao>`_

This tutorial focuses on ``SpikingFlow.ann2snn``，introduce how to convert the trained feedforward ANN to SNN and simulate it on the SpikingFlow framework.

Currently support conversion of Pytorch modules including ``nn.Conv2d`` , ``nn.Linear`` , ``nn.MaxPool2d`` , ``nn.AvgPool2d`` , ``nn.BatchNorm1d`` , ``nn.BatchNorm2d`` , ``nn.Flatten`` , ``nn.ReLU`` ,other module solutions are under development...

Theoretical basis of ANN2SNN
----------------------------

Compared with ANN, SNN generates discrete spikes, which is conducive to efficient communication. Today, ANN is popular, while direct training of SNN requires far more resources. Naturally, people will think of using very mature ANN to switch to SNN, and hope that SNN can have similar performance. This leads to the question of how to build a bridge between ANN and SNN. The current SNN mainstream method is to use frequency coding. So for the output layer, we will use the number of neuron output spikes to determine the category. Is the firing rate related to ANN?

Fortunately, there is a strong correlation between the non-linear activation of ReLU neurons in ANN and the firing rate of IF neurons in SNN (reset by subtracting the threshold :math:`V_{threshold}` ). We can use this feature for conversion. The following figure shows this correspondence: the left figure is a curve obtained by giving a constant input to an IF neuron and observing its firing over a period of time. The right one is the ReLU activation curve, which satisfies :math:`activation = max(input,0)`.

.. image:: ./_static/tutorials/ann2snn/relu_if.png

The literature [#f1]_ provides a theoretical basis for analyzing the conversion of ANN to SNN. The theory shows that the IF neuron in SNN is an unbiased estimator of ReLU activation function over time.

For the first layer of the neural network, the input layer, discuss the relationship between the firing rate of SNN neurons :math:`r` and the activation in the corresponding ANN. Assume that the input is constant as :math:`z \in [0,1]`.
For the IF neuron reset by subtraction, its membrane potential V changes with time as follows:

.. math::
	V(t)=V(t-1)+z-V_{threshold}\theta_t

Where:
:math:`V_{threshold}` is the firing threshold, usually set to 1.0. :math:`\theta_t` is the output spike. The average firing rate in the :math:`T` time steps can be obtained by summing the membrane potential:

.. math::
	\sum_{t=1}^{T} V(t)= \sum_{t=1}^{T} V(t-1)+zT-V_{threshold} \sum_{t=1}^{T}\theta_t

Move all the items containing :math:`V(t`) to the left, and divide both sides by :math:`T`:

.. math::
	\frac{V(T)-V(0)}{T} = z - V_{threshold}  \frac{\sum_{t=1}^{T}\theta_t}{T} = z- V_{threshold}  \frac{N}{T}

Where :math:`N` is the number of pulses in the time step of :math:`T`, and :math:`\frac{N}{T}` is the issuing rate :math:`r`. Use :math:`z = V_{threshold} a`
which is:

.. math::
	r = a- \frac{ V(T)-V(0) }{T V_{threshold}}

Therefore, when the simulation time step :math:`T` is infinite:

.. math::
	r = a (a>0)

Similarly, for the higher layers of the neural network, literature [#f1]_ further explains that the inter-layer firing rate satisfies:

.. math::
	r^l = W^l r^{l-1}+b^l- \frac{V^l(T)}{T V_{threshold}}

For details, please refer to [#f1]_. The methods in ann2snn also mainly come from [#f1]_ .

Conversion and simulation
-------------------------

Specifically, there are two main steps for converting feedforward ANN to SNN: model parsing and model simulation.

model parsing
^^^^^^^^^^^^^

Model parsing mainly solves two problems:

1. Researchers propose Batch Normalization for fast training and convergence. Batch normalization aims to normalize the output of ANN to 0 mean, which is contrary to the characteristics of SNN. Therefore, the parameters of BN need to be absorbed into the previous parameter layer (Linear, Conv2d)

2. According to the conversion theory, the input and output of each layer of ANN need to be limited to the range of [0,1], which requires scaling of the parameters (model normalization)

◆ Absorbing BatchNorm parameters

Assume that the parameters of BatchNorm are :math:`\gamma` (BatchNorm.weight), :math:`\beta` (BatchNorm.bias), :math:`\mu`(BatchNorm.running_mean), :math:`\sigma`(BatchNorm.running_std, square root of running_var).For specific parameter definitions, see ``torch.nn.batchnorm``.
Parameter modules (such as Linear) have parameters :math:`W` and :math:`b`. Absorbing BatchNorm parameters is transfering the parameters of BatchNorm to :math:`W` and :math:`b` of the parameter module through calculation，, so that the output of the data in new module is the same as when there is BatchNorm.
In this regard, the new model's :math:`\bar{W}` and :math:`\bar{b}` formulas are expressed as:

.. math::
	\bar{W} = \frac{\gamma}{\sigma}  W

.. math::
	\bar{b} = \frac{\gamma}{\sigma} (b - \mu) + \beta

◆ Model normalization

For a parameter module, assuming that the input tensor and output tensor are obtained, the maximum value of the input tensor is :math:`\lambda_{pre}`, and the maximum value of the output tensor is :math:`\lambda`
Then, the normalized weight :math:`\hat{W}` is:

.. math::
	\hat{W} = W * \frac{\lambda_{pre}}{\lambda}

The normalized bias :math:`\hat{b}` is:

.. math::
	\hat{b} = b / \lambda

Although the output distribution of each layer of ANN obeys a certain distribution, there are often large outliers in the data, which will reduce the overall neuron firing rate.
To solve this problem, robust normalization adjusts the scaling factor from the maximum value of the tensor to the p-percentile of the tensor. The recommended percentile value in the literature is 99.9

So far, the operations we have done on neural networks are completely equivalent. The performance of the current model should be the same as the original model.

Model simulation
^^^^^^^^^^^^^^^^

Before simulation, we need to change the ReLU activation function in the original model into an IF neuron.
For the average pooling in ANN, we need to transform it into spatial subsampling. Because IF neuron can be equivalent to ReLU activation function. Adding IF neurons after spatial downsampling has little effect on the results.
There is currently no ideal solution for maximum pooling in ANN. The best solution at present is to control the spike channel [#f1]_ with a gated function based on the momentum accumulation spike. This is also the default method in ann2snn. There are also literatures proposing to use spatial subsampling to replace Maxpool2d.

In simulation, according to the conversion theory, SNN needs to input a constant analog input. Using a Poisson encoder will bring about a decrease in accuracy. Both Poisson coding and constant input have been implemented, and one can perform different experiments if interested.

Optional configuration
^^^^^^^^^^^^^^^^^^^^^^

In view of the various optional configurations in the conversion, the ``Config`` class implemented in ``ann2snn.utils`` is used to load the default configuration and save the configuration. By loading the default configuration in Config and modifying it, one can set the parameters required when running.

Below are the introductions of the configuration corresponding to different parameters, the feasible input range, and why this configuration is needed.

(1) conf['parser']['robust_norm']

Available value：``bool``

Note：when ``True``, use robust normalization

(2) conf['simulation']['reset_to_zero']

Available value: ``None``, floating point

Note: When floating point, voltage of neurons that just fired spikes will be set to :math:``V_{reset}``; when ``None``, voltage of neurons that just fired spikes will subtract :math:``V_{threshold}``. For model that need normalization, setting to ``None`` is default, which has theoretical guaratee.

(3) conf['simulation']['encoder']['possion']

Available value：``bool``

Note: When ``True``, use Possion encoder; otherwise, use constant input over T steps.

(4) conf['simulation']['avg_pool']['has_neuron']

Available value：``bool``

Note: When ``True``, avgpool2d is converted to spatial subsampling with a layer of IF neurons; otherwise, it is only converted to spatial subsampling.

(5) conf['simulation']['max_pool']['if_spatial_avg']

Available value：``bool``

Note: When ``True``,maxpool2d is converted to avgpool2d. As referred in many literatures, this method will cause accuracy degrading.

(6) conf['simulation']['max_pool']['if_wta']

Available value：``bool``

Note: When ``True``, maxpool2d in SNN is identical with maxpool2d in ANN. Using maxpool2d in ANN means that when a spike is available in the Receptive Field, output a spike.

(7) conf['simulation']['max_pool']['momentum']

Available value: ``None``, floating point [0,1]

Note: By default, maxpool2d layer is converted into a gated function controled channel based on momentum cumulative spikes. When set to ``None``, the spike is accumulated directly. If set to floating point in the range of [0,1], spike momentum is accumulated.

The default configuration is:

.. code-block:: python

	default_config = 
	{
	'simulation':
		{
		'reset_to_zero': False,
		'encoder':
			{
			'possion': False
			},
		'avg_pool':
			{
			'has_neuron': True
			},
		'max_pool':
			{
			'if_spatial_avg': False,
			'if_wta': False,
			'momentum': None
			}
		},
	'parser':
		{
		'robust_norm': True
		}
	}



MNIST classification
--------------------

Now, use ``ann2snn`` to build a simple convolutional network to classify the MNIST dataset.

First define our network structure:

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

Note: In the defined network, the order of module definition must be consistent with the forward order, otherwise it will affect the automatic analysis of the network.It is best to use ``nn.Sequence(·)`` to completely define the network. After each Conv2d and Linear layer, a ReLU layer must be placed, which can be separated by a BatchNorm layer. No ReLU is added after the pooling layer. If you encounter a situation where you need to expand the tensor, define a ``nn.Flatten`` module in the network. In the forward function, you need to use the defined Flatten instead of the view function.

Define our hyperparameters:

.. code-block:: python

	device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    dataset_dir = input('输入保存MNIST数据集的位置，例如“./”\n input root directory for saving MNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“100”\n input simulating steps, e.g., "100": '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“10”\n input training epochs, e.g., "10": '))
    model_name = input('输入模型名字，例如“mnist”\n input model name, for log_dir generating , e.g., "mnist": ')

The program searches for the trained model archive (a file with the same name as `model_name`) according to the specified folder, and all subsequent temporary files will be stored in that folder.

Load the default conversion configuration and save

.. code-block:: python

	config = utils.Config.default_config
	print('ann2snn config:\n\t', config)
	utils.Config.store_config(os.path.join(log_dir,'default_config.json'),config)


Initialize data loader, network, optimizer, loss function

.. code-block:: python

	# Initialize the network
	ann = ANN().to(device)
	# Define loss function
	loss_function = nn.CrossEntropyLoss()
	# Use Adam optimizer
	optimizer = torch.optim.Adam(ann.parameters(), lr=learning_rate, weight_decay=5e-4)

Train ANN and test it regularly. You can also use the pre-written training program in utils during training.

.. code-block:: python

	for epoch in range(train_epoch):
		# Train the network using a pre-prepared code in ''utils''
		utils.train_ann(net=ann,
						device=device,
						data_loader=train_data_loader,
						optimizer=optimizer,
						loss_function=loss_function,
						epoch=epoch
						)
		# Validate the network using a pre-prepared code in ''utils''
		acc = utils.val_ann(net=ann,
							device=device,
							data_loader=test_data_loader,
							epoch=epoch
							)
		if best_acc <= acc:
			utils.save_model(ann, log_dir, model_name+'.pkl')

The complete code is located in ``ann2snn.examples.if_cnn_mnist.py``, in the code we also use Tensorboard to save training logs. You can run it directly on the Python command line:

.. code-block:: python

    >>> import SpikingFlow.ann2snn.examples.if_cnn_mnist as if_cnn_mnist
    >>> if_cnn_mnist.main()
    输入运行的设备，例如“cpu”或“cuda:0”
     input device, e.g., "cpu" or "cuda:0": cuda:15
    输入保存MNIST数据集的位置，例如“./”
     input root directory for saving MNIST dataset, e.g., "./": ./mnist
    输入batch_size，例如“64”
     input batch_size, e.g., "64": 128
    输入学习率，例如“1e-3”
     input learning rate, e.g., "1e-3": 1e-3
    输入仿真时长，例如“100”
     input simulating steps, e.g., "100": 100
    输入训练轮数，即遍历训练集的次数，例如“10”
     input training epochs, e.g., "10": 10
    输入模型名字，用于自动生成日志文档，例如“mnist”
     input model name, for log_dir generating , e.g., "mnist"

    If the input of the main function is not a folder with valid files, an automatic log file folder is automatically generated.
     Terminal outputs root directory for saving logs, e.g., "./": ./log-mnist1596804385.476601

    Epoch 0 [1/937] ANN Training Loss:2.252 Accuracy:0.078
    Epoch 0 [101/937] ANN Training Loss:1.424 Accuracy:0.669
    Epoch 0 [201/937] ANN Training Loss:1.117 Accuracy:0.773
    Epoch 0 [301/937] ANN Training Loss:0.953 Accuracy:0.795
    Epoch 0 [401/937] ANN Training Loss:0.865 Accuracy:0.788
    Epoch 0 [501/937] ANN Training Loss:0.807 Accuracy:0.792
    Epoch 0 [601/937] ANN Training Loss:0.764 Accuracy:0.795
    Epoch 0 [701/937] ANN Training Loss:0.726 Accuracy:0.834
    Epoch 0 [801/937] ANN Training Loss:0.681 Accuracy:0.880
    Epoch 0 [901/937] ANN Training Loss:0.641 Accuracy:0.888
    Epoch 0 [100/100] ANN Validating Loss:0.328 Accuracy:0.881
    Save model to: ./log-mnist1596804385.476601\mnist.pkl
    ...
    Epoch 9 [901/937] ANN Training Loss:0.036 Accuracy:0.990
    Epoch 9 [100/100] ANN Validating Loss:0.042 Accuracy:0.988
    Save model to: ./log-mnist1596804957.0179427\mnist.pkl

In the example, this model is trained for 10 epochs. The changes in the accuracy of the test set during training are as follows:

.. image:: ./_static/tutorials/ann2snn/accuracy_curve.png

In the end, the accuracy on test dataset is 98.8%.

Take a part of the data from the training set and use it for the normalization step of the model. Here we take 1/500 of the training data, which is 100 pictures. But it should be noted that the range of the data tensor taken from the dataset is [0, 255], and it needs to be divided by 255 to become a floating point tensor in the range of [0.0, 1.0] to match the feasible range of firing rate.

.. code-block:: python

	norm_set_len = int(train_data_dataset.data.shape[0] / 500)
    print('Using %d pictures as norm set'%(norm_set_len))
    norm_set = train_data_dataset.data[:norm_set_len, :, :].float() / 255
    norm_tensor = torch.FloatTensor(norm_set).view(-1,1,28,28)

Call the standard conversion function ``standard_conversion`` implemented in ``ann2snn.utils`` to realize ANN conversion and SNN simulation.

.. code-block:: python

	utils.standard_conversion(model_name=model_name,
                              norm_data=norm_tensor,
                              test_data_loader=test_data_loader,
                              device=device,
                              T=T,
                              log_dir=log_dir,
                              config=config
                              )

In the process, the normalized model structure is output:

.. code-block:: python

	ModelParser(
	  (network): Sequential(
		(0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
		(1): ReLU()
		(2): AvgPool2d(kernel_size=2, stride=2, padding=0)
		(3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
		(4): ReLU()
		(5): AvgPool2d(kernel_size=2, stride=2, padding=0)
		(6): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
		(7): ReLU()
		(8): AvgPool2d(kernel_size=2, stride=2, padding=0)
		(9): Flatten()
		(10): Linear(in_features=32, out_features=10, bias=True)
		(11): ReLU()
	  )
	)

At the same time, one can also observe the structure of SNN:

.. code-block:: python

	SNN(
	  (network): Sequential(
		(0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
		(1): IFNode(
		  v_threshold=1.0, v_reset=None
		  (surrogate_function): Sigmoid()
		)
		(2): AvgPool2d(kernel_size=2, stride=2, padding=0)
		(3): IFNode(
		  v_threshold=1.0, v_reset=None
		  (surrogate_function): Sigmoid()
		)
		(4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
		(5): IFNode(
		  v_threshold=1.0, v_reset=None
		  (surrogate_function): Sigmoid()
		)
		(6): AvgPool2d(kernel_size=2, stride=2, padding=0)
		(7): IFNode(
		  v_threshold=1.0, v_reset=None
		  (surrogate_function): Sigmoid()
		)
		(8): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
		(9): IFNode(
		  v_threshold=1.0, v_reset=None
		  (surrogate_function): Sigmoid()
		)
		(10): AvgPool2d(kernel_size=2, stride=2, padding=0)
		(11): IFNode(
		  v_threshold=1.0, v_reset=None
		  (surrogate_function): Sigmoid()
		)
		(12): Flatten()
		(13): Linear(in_features=32, out_features=10, bias=True)
		(14): IFNode(
		  v_threshold=1.0, v_reset=None
		  (surrogate_function): Sigmoid()
		)
	  )
	)

It can be seen that the activation of ReLU in the ANN model is replaced by the IFNode of SNN. Each layer of AvgPool2d is followed by a layer of IFNode.

Due to the long time of model simulation, the current accuracy and simulation progress are continuously output:

.. code-block:: python

	[SNN Simulating... 1.00%] Acc:0.990
	[SNN Simulating... 2.00%] Acc:0.990
	[SNN Simulating... 3.00%] Acc:0.990
	[SNN Simulating... 4.00%] Acc:0.988
	[SNN Simulating... 5.00%] Acc:0.990
	……
	[SNN Simulating... 95.00%] Acc:0.986
	[SNN Simulating... 96.00%] Acc:0.986
	[SNN Simulating... 97.00%] Acc:0.986
	[SNN Simulating... 98.00%] Acc:0.986
	[SNN Simulating... 99.00%] Acc:0.987
	SNN Simulating Accuracy:0.987
	Summary:	ANN Accuracy:98.7900%  	SNN Accuracy:98.6500% [Decreased 0.1400%]

Through the final output, we can know that the accuracy of ANN's MNIST classification is 98.79%. The accuracy of the converted SNN is 98.65%. The conversion resulted in a 0.14% performance degradation.

.. [#f1] Rueckauer B, Lungu I-A, Hu Y, Pfeiffer M and Liu S-C (2017) Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification. Front. Neurosci. 11:682.
.. [#f2] Diehl, Peter U. , et al. Fast classifying, high-accuracy spiking deep networks through weight and threshold balancing. Neural Networks (IJCNN), 2015 International Joint Conference on IEEE, 2015.
.. [#f3] Rueckauer, B., Lungu, I. A., Hu, Y., & Pfeiffer, M. (2016). Theory and tools for the conversion of analog to spiking convolutional neural networks. arXiv preprint arXiv:1612.04052.
.. [#f4] Sengupta, A., Ye, Y., Wang, R., Liu, C., & Roy, K. (2019). Going deeper in spiking neural networks: Vgg and residual architectures. Frontiers in neuroscience, 13, 95.