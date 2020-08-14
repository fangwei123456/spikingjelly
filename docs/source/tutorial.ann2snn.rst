前馈ANN转换SNN SpikingFlow.ann2snn
=======================================
本教程作者： `DingJianhao <https://github.com/DingJianhao>`_

本节教程主要关注 ``SpikingFlow.ann2snn``，介绍如何将训练好的前馈ANN转换SNN，并且在SpikingFlow框架上进行仿真。

目前暂时支持Pytorch中实现的包含 ``nn.Conv2d`` , ``nn.Linear`` , ``nn.MaxPool2d`` , ``nn.AvgPool2d`` , ``nn.BatchNorm1d`` , ``nn.BatchNorm2d`` , ``nn.Flatten`` , ``nn.ReLU`` 的前馈神经网络的转换，其他模块方案正在开发中...

ANN转换SNN的理论基础
--------------------

SNN相比于ANN，产生的脉冲是离散的，这有利于高效的通信。在ANN大行其道的今天，SNN的直接训练需要较多资源。自然我们会想到使用现在非常成熟的ANN转换到SNN，希望SNN也能有类似的表现。这就牵扯到如何搭建起ANN和SNN桥梁的问题。现在SNN主流的方式是采用频率编码，因此对于输出层，我们会用神经元输出脉冲数来判断类别。发放率和ANN有没有关系呢？

幸运的是，ANN中的ReLU神经元非线性激活和SNN中IF神经元(采用减去阈值 :math:`V_{threshold}` 方式重置)的发放率有着极强的相关性，我们可以借助这个特性来进行转换。下图就展示了这种对应关系：左图是给一个IF神经元恒定输入，观察其一段时间发放情况得到的曲线。右边是ReLU激活的曲线，满足 :math:`activation = max(input,0)` 。

.. image:: ./_static/tutorials/ann2snn/relu_if.png

文献 [#f1]_ 对ANN转SNN提供了解析的理论基础。理论说明，SNN中的IF神经元是ReLU激活函数在时间上的无偏估计器。

针对神经网络第一层即输入层，讨论SNN神经元的发放率 :math:`r` 和对应ANN中激活的关系。假定输入恒定为 :math:`z \in [0,1]`。
对于采用减法重置的IF神经元，其膜电位V随时间变化为：

.. math::
	V(t)=V(t-1)+z-V_{threshold}\theta_t

其中：
 :math:`V_{threshold}` 为发放阈值，通常设为1.0。 :math:`\theta_t` 为输出脉冲。 :math:`T` 时间步内的平均发放率可以通过对膜电位求和得到：

.. math::
	\sum_{t=1}^{T} V(t)= \sum_{t=1}^{T} V(t-1)+zT-V_{threshold} \sum_{t=1}^{T}\theta_t

将含有 :math:`V(t`) 的项全部移项到左边，两边同时除以 :math:`T` ：

.. math::
	\frac{V(T)-V(0)}{T} = z - V_{threshold}  \frac{\sum_{t=1}^{T}\theta_t}{T} = z- V_{threshold}  \frac{N}{T}

其中 :math:`N` 为 :math:`T` 时间步内脉冲数， :math:`\frac{N}{T}` 就是发放率  :math:`r`。利用  :math:`z= V_{threshold} a` 
即：

.. math::
	r = a- \frac{ V(T)-V(0) }{T V_{threshold}}

故在仿真时间步  :math:`T` 无限长情况下:

.. math::
	r = a (a>0)

类似地，针对神经网络更高层，文献 [#f1]_ 进一步说明层间发放率满足：

.. math::
	r^l = W^l r^{l-1}+b^l- \frac{V^l(T)}{T V_{threshold}}

详细的说明见文献 [#f1]_ 。ann2snn中的方法也主要来自文献 [#f1]_ 

转换和仿真
----------

具体地，进行前馈ANN转SNN主要有两个步骤：即模型分析（英文：parse，直译：句法分析）和仿真模拟。

模型分析
^^^^^^^^

模型分析主要解决两个问题：

1、ANN为了快速训练和收敛提出了批归一化（Batch Normalization）。批归一化旨在将ANN输出归一化到0均值，这与SNN的特性相违背。因此，需要将BN的参数吸收到前面的参数层中（Linear、Conv2d）

2、根据转换理论，ANN的每层输入输出需要被限制在[0,1]范围内，这就需要对参数进行缩放（模型归一化）

◆ BatchNorm参数吸收

假定BatchNorm的参数为 :math:`\gamma` (BatchNorm.weight)， :math:`\beta` (BatchNorm.bias)， :math:`\mu`(BatchNorm.running_mean) ， :math:`\sigma` (BatchNorm.running_var running_var开根号)。具体参数定义详见 ``torch.nn.batchnorm`` 。
参数模块（例如Linear）具有参数 :math:`W` 和 :math:`b` 。BatchNorm参数吸收就是将BatchNorm的参数通过运算转移到参数模块的 :math:`W`和 :math:`b` 中，使得数据输入新模块的输出和有BatchNorm时相同。
对此，新模型的 :math:`\bar{W}` 和 :math:`\bar{b}` 公式表示为：

.. math::
	\bar{W} = \frac{\gamma}{\sigma}  W

.. math::
	\bar{b} = \frac{\gamma}{\sigma} (b - \mu) + \beta

◆ 模型归一化

对于某个参数模块，假定得到了其输入张量和输出张量，其输入张量的最大值为 :math:`\lambda_{pre}` ,输出张量的最大值为 :math:`\lambda` 
那么，归一化后的权重 :math:`\hat{W}` 为：

.. math::
	\hat{W} = W * \frac{\lambda_{pre}}{\lambda}

归一化后的偏置 :math:`\hat{b}` 为：

.. math::
	\hat{b} = b / \lambda

ANN每层输出的分布虽然服从某个特定分布，但是数据中常常会存在较大的离群值，这会导致整体神经元发放率降低。
为了解决这一问题，鲁棒归一化将缩放因子从张量的最大值调整为张量的p分位点。文献中推荐的分位点值为99.9

到现在为止，我们对神经网络做的操作，在数值上是完全等价的。当前的模型表现应该与原模型相同。

模型仿真
^^^^^^^^

仿真前，我们需要将原模型中的ReLU激活函数变为IF神经元。
对于ANN中的平均池化，我们需要将其转化为空间下采样。由于IF神经元可以等效ReLU激活函数。空间下采样后增加IF神经元与否对结果的影响极小。
对于ANN中的最大池化，目前没有非常理想的方案。目前的最佳方案为使用基于动量累计脉冲的门控函数控制脉冲通道 [#f1]_ 。这也是ann2snn的默认方式。还有文献提出使用空间下采样替代Maxpool2d。

仿真时，依照转换理论，SNN需要输入恒定的模拟输入。使用Poisson编码器将会带来准确率的降低。Poisson编码和恒定输入方式均已实现，感兴趣可通过配置进行不同实验。

可选配置
^^^^^^^^

鉴于转换中有多种可选配置， ``ann2snn.utils`` 中实现 ``Config`` 类用来加载默认配置和保存配置。
通过加载Config中的默认配置并修改，可以设定自己模型运行时所需要的参数。

下面，将介绍不同参数对应的配置，可行的输入范围，以及为什么要这个配置

(1)conf['parser']['robust_norm']

可行值： ``bool`` 类型

说明：当设置为 ``True`` ，使用鲁棒归一化

(2)conf['simulation']['reset_to_zero']

可行值： ``None`` , 浮点数

说明：当设置为 ``None`` ，神经元重置的时候采用减去 :math:`V_{threshold}` 的方式；当为浮点数时，刚刚发放的神经元会被设置为 :math:`V_{reset}` 。对于需要归一化的转换模型，设置为 ``None`` 是推荐的方式，具有理论保证.

(3)conf['simulation']['encoder']['possion']

可行值： ``bool`` 类型

说明：当设置为 ``True`` ，输入采用泊松编码器；否则，采用浮点数持续的输入仿真时长T时间。

(4)conf['simulation']['avg_pool']['has_neuron']

可行值： ``bool`` 类型

说明：当设置为 ``True`` ，平均池化层被转化为空间下采样加上一层IF神经元；否则，平均池化层仅被转化为空间下采样。

(5)conf['simulation']['max_pool']['if_spatial_avg']

可行值： ``bool`` 类型

说明：当设置为``True``，最大池化层被转化为平均池化。这个方式根据文献可能会导致精度下降。

(6)conf['simulation']['max_pool']['if_wta']

可行值： ``bool`` 类型

说明：当设置为 ``True`` ，最大池化层和ANN中最大池化一样。使用ANN的最大池化意味着当感受野中一旦有脉冲即输出1。

(7)conf['simulation']['max_pool']['momentum']

可行值： ``None`` , [0,1]内浮点数

说明：最大池化层被转化为基于动量累计脉冲的门控函数控制脉冲通道。当设置为 ``None`` ，直接累计脉冲；若为[0,1]浮点数，进行脉冲动量累积。

默认配置为：

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



识别MNIST
---------

现在我们使用 ``ann2snn`` ，搭建一个简单卷积网络，对MNIST数据集进行分类。

首先定义我们的网络结构：

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

注意：定义的网络中，模块定义的顺序必须和前向的顺序保持一致，否则会影响网络的自动分析。最好使用 ``nn.Sequence(·)`` 完整定义好网络。每一个Conv2d和Linear层后，必须要放一个ReLU层，其间可以隔着一个BatchNorm层。池化层后不加ReLU。如果遇到需要将tensor展开的情况，就在网络中定义一个 ``nn.Flatten`` 模块，在forward函数中需要使用定义的Flatten而不是view函数。

定义我们的超参数：

.. code-block:: python

	device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    dataset_dir = input('输入保存MNIST数据集的位置，例如“./”\n input root directory for saving MNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“100”\n input simulating steps, e.g., "100": '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“10”\n input training epochs, e.g., "10": '))
    model_name = input('输入模型名字，例如“mnist”\n input model name, for log_dir generating , e.g., "mnist": ')

程序按照指定的文件夹搜寻训练好的模型存档（和 `model_name` 同名的文件），之后的所有临时文件都会储存到文件夹中。

加载默认的转换配置并保存

.. code-block:: python

	config = utils.Config.default_config
	print('ann2snn config:\n\t', config)
	utils.Config.store_config(os.path.join(log_dir,'default_config.json'),config)


初始化数据加载器、网络、优化器、损失函数

.. code-block:: python

	# 初始化网络
	ann = ANN().to(device)
	# 定义损失函数
	loss_function = nn.CrossEntropyLoss()
	# 使用Adam优化器
	optimizer = torch.optim.Adam(ann.parameters(), lr=learning_rate, weight_decay=5e-4)

训练ANN，并定期测试。训练时也可以使用utils中预先写好的训练程序

.. code-block:: python

	for epoch in range(train_epoch):
		# 使用utils中预先写好的训练程序训练网络
		# 训练程序的写法和经典ANN中的训练也是一样的
		# Train the network using a pre-prepared code in ''utils''
		utils.train_ann(net=ann,
						device=device,
						data_loader=train_data_loader,
						optimizer=optimizer,
						loss_function=loss_function,
						epoch=epoch
						)
		# 使用utils中预先写好的验证程序验证网络输出
		# Validate the network using a pre-prepared code in ''utils''
		acc = utils.val_ann(net=ann,
							device=device,
							data_loader=test_data_loader,
							epoch=epoch
							)
		if best_acc <= acc:
			utils.save_model(ann, log_dir, model_name+'.pkl')

完整的代码位于 ``ann2snn.examples.if_cnn_mnist.py`` ，在代码中我们还使用了Tensorboard来保存训练日志。可以直接在Python命令行运行它：

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

    如果main函数的输入不是具有有效文件的文件夹，自动生成一个日志文件文件夹
    If the input of the main function is not a folder with valid files, an automatic log file folder is automatically generated.
    第一行输出为保存日志文件的位置，例如“./log-mnist1596804385.476601”
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

示例中，这个模型训练10个epoch。训练时测试集准确率变化情况如下：

.. image:: ./_static/tutorials/ann2snn/accuracy_curve.png

最终达到98.8%的测试集准确率。

从训练集中，取出一部分数据，用于模型的归一化步骤。这里我们取的是训练数据的1/500，也就是100张图片。但是要注意，从dataset中取出的数据tensor范围为[0，255]，需要除以255变为[0.0,1.0]范围的浮点数来匹配脉冲频率的可行域。

.. code-block:: python

	norm_set_len = int(train_data_dataset.data.shape[0] / 500)
    print('Using %d pictures as norm set'%(norm_set_len))
    norm_set = train_data_dataset.data[:norm_set_len, :, :].float() / 255
    norm_tensor = torch.FloatTensor(norm_set).view(-1,1,28,28)

调用``ann2snn.utils``中实现的标准转换函数``standard_conversion``就可以实现ANN的转换加上SNN仿真。

.. code-block:: python

	utils.standard_conversion(model_name=model_name,
                              norm_data=norm_tensor,
                              test_data_loader=test_data_loader,
                              device=device,
                              T=T,
                              log_dir=log_dir,
                              config=config
                              )

过程中，归一化后的模型结构被输出:

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

同时，我们也观察一下SNN的结构：

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

可以看出，ANN模型中的ReLU激活被SNN的IFNode取代。每一层AvgPool2d后都跟了一层IFNode。

模型仿真由于时间较长，持续输出当前准确率和仿真进度:

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

通过最后的输出，可以知道，ANN的MNIST分类准确率为98.79%。转换后的SNN准确率为98.65%。转换带来了0.14%的性能下降。

.. [#f1] Rueckauer B, Lungu I-A, Hu Y, Pfeiffer M and Liu S-C (2017) Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification. Front. Neurosci. 11:682.
.. [#f2] Diehl, Peter U. , et al. Fast classifying, high-accuracy spiking deep networks through weight and threshold balancing. Neural Networks (IJCNN), 2015 International Joint Conference on IEEE, 2015.
.. [#f3] Rueckauer, B., Lungu, I. A., Hu, Y., & Pfeiffer, M. (2016). Theory and tools for the conversion of analog to spiking convolutional neural networks. arXiv preprint arXiv:1612.04052.
.. [#f4] Sengupta, A., Ye, Y., Wang, R., Liu, C., & Roy, K. (2019). Going deeper in spiking neural networks: Vgg and residual architectures. Frontiers in neuroscience, 13, 95.