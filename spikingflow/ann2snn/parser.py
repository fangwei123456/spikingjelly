import torch
import torch.nn as nn
import os
import numpy as np


class ModelParser(nn.Module):
    def __init__(self):
        '''
        * :ref:`API in English <ModelParser.__init__-en>`

        .. _ModelParser.__init__-cn:
        ModelParser是ANN分析器，主要功能有两方面：
            
        - 其一是，将模型中的批正则化参数吸收到之前的有参数的模块中
            
        - 其二是，将模型进行归一化以适应SNN的发放率
        
        ModelParser在调用parser函数后，将原始ANN的参数化结构转移到到了 `ModelParser.network` 对象中

        * :ref:`中文API <ModelParser.__init__-cn>`

        .. _ModelParser.__init__-en:

        ModelParser is an ANN Parser with two main functions:
        
        - One is to absorb the batch normalization parameters into the previous modules with parameters.
        
        - The second is to naturalize the model to accommodate the release rate of SNN.
        
        After calling the parse function, ModelParser transfers the structure of the original ANN to the object `ModelParser.network`.
        '''
        super().__init__()
        self.module_list = []
        self.module_len = 0
        self.network = torch.nn.Sequential()  # 分析的模型保存在network对象中

        self.relu_linker = {}                 # 字典类型，用于通过relu层在network中的序号确定relu前参数化模块的序号
        self.param_module_relu_linker = {}    # 字典类型，用于通过relu前在network中的参数化模块的序号确定relu层序号

        self.activation_range = {}            # 字典类型，保存在network中的序号对应层的激活最大值（或某分位点值）

    def forward(self,x):
        return self.network(x)

    def parse(self, model, log_dir):
        '''
        * :ref:`API in English <ModelParser.parse-en>`

        .. _ModelParser.parse-cn:

        :param model: 待分析的ANN模型
        :param log_dir: 用于保存临时文件的文件夹名
        :return: 分析后的ANN模型

        分析模型。目前支持的模块有： ``Linear`` ， ``Conv2d`` ， ``ReLU`` ， ``BatchNorm1d`` ， ``BatchNorm2d`` ， ``MaxPool2d`` ， ``AvgPool2d`` ，``Flatten`` 。
        
        暂时仅支持具有前馈结构的神经网络，无法支持resnet，rnn等高级结构。
        
        具体的步骤是：
        
        遍历输入模型的模块，对不同模块类型进行不同的操作：
            1、对于Linear和Conv2d等有weight和bias的模块，保存待分析参数，加入模块队列并记录编号，便于后续BatchNorm参数吸收时搜寻模块。
            2、对于Softmax，使用ReLU进行替代，Softmax关于某个输入变量是单调递增的，意味着ReLU并不会对输出的正确性造成太大影响。
            3、对于BatchNorm，将其参数吸收进对应的参数化模块，其中：BatchNorm1d默认其上一个模块为Linear，BatchNorm2d默认其上一个模块为Conv2d。
                假定BatchNorm的参数为 :math:`\gamma` (BatchNorm.weight)， :math:`\beta` (BatchNorm.bias)， :math:`\mu`(BatchNorm.running_mean) ， :math:`\sigma` (BatchNorm.running_var running_var开根号)。具体参数定义详见 ``torch.nn.batchnorm`` 。参数模块（例如Linear）具有参数 :math:`W` 和 :math:`b` 。BatchNorm参数吸收就是将BatchNorm的参数通过运算转移到参数模块的 :math:`W`和 :math:`b` 中，使得数据输入新模块的输出和有BatchNorm时相同。
                对此，新模型的 :math:`\bar{W}` 和 :math:`\bar{b}` 公式表示为：

            .. math::
                \bar{W} = \frac{\gamma}{\sigma}  W

            .. math::
                \bar{b} = \frac{\gamma}{\sigma} (b - \mu) + \beta
                
            4、对于AvgPool2d、MaxPool2d、Flatten，加入模块队列
        最后将模块队列使用 ``torch.nn.Sequential`` 组成一个Pytorch神经网络，可以使用 `ModelParser.network` 对象访问
        输出的模型表现应该与原模型相同。

        * :ref:`API in English <ModelParser.parse-cn>`

        .. _ModelParser.parse-en:

        :param model: ANN model to be parsed
        :param log_dir: Name of the folder used to save temporary files
        :return: Parsed ANN model

        Parse the model. The modules currently supported are: Linear, Conv2d, ReLU, BatchNorm1d, BatchNorm2d, MaxPool2d, AvgPool2d, Flatten.
        For the time being, only neural networks with feed-forward structures are supported, and advanced structures such as ResNets and RNNs are not supported.
        Steps:
        Traverse the modules of the input model and do different things with different module types:
        
        1. For `Linear` and `Conv2d` and other modules with weight and bias, save the parameters, add to the module list and record the seq, so as to fast locate the module when absorbing the BatchNorm parameter.
        
        2. For `Softmax`, use `ReLU` instead. `Softmax` is monotonically increasing w.r.t one input variable. That means using `ReLU` does not have much effect on the correctness of the output.
        
        3. For `BatchNorm`, parameters are absorbed into the corresponding module with parameters, wherein: BatchNorm1d should be after Linear; BatchNorm2d should be after Conv2d.
            
            Assume that the parameters of BatchNorm are :math:`\gamma` (BatchNorm.weight), :math:`\beta` (BatchNorm.bias), :math:`\mu`(BatchNorm.running_mean), :math:`\sigma`(BatchNorm.running_std, square root of running_var).For specific parameter definitions, see ``torch.nn.batchnorm``. Parameter modules (such as Linear) have parameters :math:`W` and :math:`b`. Absorbing BatchNorm parameters is transfering the parameters of BatchNorm to :math:`W` and :math:`b` of the parameter module through calculation，, so that the output of the data in new module is the same as when there is BatchNorm.

            In this regard, the new model's :math:`\bar{W}` and :math:`\bar{b}` formulas are expressed as:

            .. math::
                \bar{W} = \frac{\gamma}{\sigma}  W

            .. math::
                \bar{b} = \frac{\gamma}{\sigma} (b - \mu) + \beta
    
        4. For AvgPool2d, MaxPool2d, Flatten, add to the module list
        
        Finally, the module list is used to form a Pytorch neural network using ``torch.nn.Sequencial`` (namely `ModelParser.network`)
        
        Note that the performance of the output model should be the same as that of the original model.
        '''
        self.module_len = 0
        self.module_list = []
        last_parammodule_idx = 0
        for n,m in model.named_modules():
            Name = m.__class__.__name__
            if hasattr(m,'weight') or hasattr(m,'bias'):
                # 保存待分析的参数
                torch.save(m,os.path.join(log_dir,'%s_%s_parsed.pkl'%(Name,n)))
                m = torch.load(os.path.join(log_dir,'%s_%s_parsed.pkl'%(Name,n)))
            # 加载激活层
            if Name == 'Softmax':
                Name = 'ReLU'
                print("Replacing Softmax by ReLU.")
            if Name == 'ReLU':
                self.module_list.append(m)
                self.relu_linker[self.module_len] = last_parammodule_idx
                self.param_module_relu_linker[last_parammodule_idx] = self.module_len
                self.activation_range[self.module_len] = -1e5
                self.module_len += 1
            # 加载BatchNorm层
            if Name == 'BatchNorm2d':
                # BatchNorm2d 前的参数模块必须是Conv2d
                assert (self.module_list[last_parammodule_idx].__class__.__name__ == 'Conv2d')
                self.absorb_batchnorm(self.module_list[last_parammodule_idx], m)
            if Name == 'BatchNorm1d':
                # BatchNorm1d 前的参数模块必须是Linear
                assert (self.module_list[last_parammodule_idx].__class__.__name__ == 'Linear')
                self.absorb_batchnorm(self.module_list[last_parammodule_idx], m)
            # 加载有参数的层
            if Name == 'Linear':
                self.module_list.append(m)
                last_parammodule_idx = self.module_len
                self.module_len += 1
            if Name == 'Conv2d':
                self.module_list.append(m)
                last_parammodule_idx = self.module_len
                self.module_len += 1
            # 加载无参数层
            if Name == 'MaxPool2d':
                #Name = 'AvgPool2d'
                self.module_list.append(m)
                self.module_len += 1
                # print("Replacing max by average pooling.")
            if Name == 'AvgPool2d':
                self.module_list.append(nn.AvgPool2d(kernel_size=m.kernel_size,
                                                     stride=m.stride,
                                                     padding=m.padding
                                                     ))
                self.module_len += 1
            if Name == 'Flatten':
                self.module_list.append(m)
                self.module_len += 1
        self.network = torch.nn.Sequential(*self.module_list)

    def absorb_batchnorm(self, param_module, bn_module):
        '''
        * :ref:`API in English <ModelParser.absorb_batchnorm-en>`

        .. _ModelParser.absorb_batchnorm-cn:

        :param param_module: Pytorch中的参数模块(例如 Linear, Conv2d)
        :param bn_module: Pytorch中BatchNorm模块
        :return: ``None``

        将其参数吸收进对应的参数模块，其中：BatchNorm1d默认其上一个模块为Linear，BatchNorm2d默认其上一个模块为Conv2d

        * :ref:`API in English <ModelParser.absorb_batchnorm-cn>`

        .. _ModelParser.absorb_batchnorm-en:

        :param param_module: Pytorch Parametric modules (e.g. Linear, Conv2d)
        :param bn_module: Pytorch BatchNorm modules
        :return: ``None``

        BN parameters are absorbed into the corresponding module with parameters, wherein: BatchNorm1d should be after Linear; BatchNorm2d should be after Conv2d.

        '''
        if_2d = len(param_module.weight.size()) == 4 # 判断是否为BatchNorm2d
        bn_std = torch.sqrt(bn_module.running_var.data + bn_module.eps)
        if not if_2d:
            if param_module.bias is not None:
                param_module.weight.data = param_module.weight.data * bn_module.weight.data.view(-1, 1) / bn_std.view(-1,
                                                                                                                    1)
                param_module.bias.data = (param_module.bias.data - bn_module.running_mean.data.view(
                    -1)) * bn_module.weight.data.view(-1) / bn_std.view(
                    -1) + bn_module.bias.data.view(-1)
            else:
                param_module.weight.data = param_module.weight.data * bn_module.weight.data.view(-1, 1) / bn_std.view(-1,
                                                                                                                    1)
                param_module.bias.data = (torch.zeros_like(
                    bn_module.running_mean.data.view(-1)) - bn_module.running_mean.data.view(
                    -1)) * bn_module.weight.data.view(-1) / bn_std.view(-1) + bn_module.bias.data.view(-1)
        else:
            if param_module.bias is not None:
                param_module.weight.data = param_module.weight.data * bn_module.weight.data.view(-1, 1, 1,
                                                                                               1) / bn_std.view(-1, 1,
                                                                                                                1, 1)
                param_module.bias.data = (param_module.bias.data - bn_module.running_mean.data.view(
                    -1)) * bn_module.weight.data.view(-1) / bn_std.view(
                    -1) + bn_module.bias.data.view(-1)
            else:
                param_module.weight.data = param_module.weight.data * bn_module.weight.data.view(-1, 1, 1,
                                                                                               1) / bn_std.view(-1, 1,
                                                                                                                1, 1)
                param_module.bias.data = (torch.zeros_like(
                    bn_module.running_mean.data.view(-1)) - bn_module.running_mean.data.view(
                    -1)) * bn_module.weight.data.view(-1) / bn_std.view(-1) + bn_module.bias.data.view(-1)

    def normalize_model(self,norm_tensor,log_dir,robust=False):
        '''
        * :ref:`API in English <ModelParser.normalize_model-en>`

        .. _ModelParser.normalize_model-cn:

        :param norm_tensor: 用于归一化模型的数据
        :param log_dir: 用于保存临时文件的文件夹名
        :param robust: 当取值为 ``True`` 时使用文献[2]中提出的鲁棒归一化
        :return: ``None``

        模型归一化的对象为吸收了BatchNorm参数后的ANN中的参数模块（Linear, Conv2d），其目的在于限制每层激活的输出范围在[0,1]范围内，
        如此便可使得模型在转换为SNN时的脉冲发放率在[0,:math:`r_max`]范围内。
        模型归一化在文献[1]中被提出，所提出的归一化利用权重的最大最小值。但是[1]中的方法不涉及神经网络中存在bias的情况。
        为了适应更多的神经网络，此处参考文献[2]实现归一化模块：通过缩放因子缩放权重和偏置项。
        对于某个参数模块，假定得到了其输入张量和输出张量，其输入张量的最大值为 :math:`\lambda_{pre}` ,输出张量的最大值为 :math:`\lambda` 。那么，归一化后的权重 :math:`\hat{W}` 为：

        .. math::
            \hat{W} = W * \frac{\lambda_{pre}}{\lambda}

           归一化后的偏置 :math:`\hat{b}` 为：

        .. math::
            \hat{b} = b / \lambda
        ANN每层输出的分布虽然服从某个特定分布，但是数据中常常会存在较大的离群值，这会导致整体神经元发放率降低。
        为了解决这一问题，鲁棒归一化将缩放因子从张量的最大值调整为张量的p-分位点。[2]中推荐的分位点值为99.9%。
        
        更多内容见文献[2]

        * :ref:`API in English <ModelParser.normalize_model-cn>`

        .. _ModelParser.normalize_model-en:

        :param norm_tensor: tensors used to normalize the model
        :param log_dir: Name of the folder used to save temporary files
        :param robust: when ``True``, use robust normalization proposed in [2]
        :return: ``None``

        Model normalization is designed for modules that have parameters (Linear, Conv2d) and have absorbed the BatchNorm parameters.
        The purpose is to limit the output range of each layer to the range of [0,1]
        This allows the model to be in the range of the firing rate when converted to SNN in the range of [0,:math:`r_max`].
        Model normalization is proposed in [1], and the proposed normalization takes advantage of the maximum value of the weight.
        However, the method in [1] does not involve bias in the neural network.
        To accommodate more neural networks, model normalization is implemented based on [2]: scaling weights and bias through scaling factors.
        For a parameter module, assuming that the input tensor and output tensor are obtained, the maximum value of the input tensor is :math:`\lambda_{pre}`, and the maximum value of the output tensor is :math:`\lambda`. Then, the normalized weight :math:`\hat{W}` is:

        .. math::
            \hat{W} = W * \frac{\lambda_{pre}}{\lambda}

        The normalized bias :math:`\hat{b}` is:

        .. math::
            \hat{b} = b / \lambda
        
        Although the distribution of the output of the ANN per layer is subject to a particular distribution, there are often large outliers, which results in a decrease in the overall firing rate.
        To solve this problem, robust normalization adjusts the scaling factor from tensor's maximum value to tensor's p-percentile. The recommended p is 99.9%[2].

        [1] Diehl, Peter U. , et al. "Fast classifying, high-accuracy spiking deep networks through weight and threshold
        balancing." Neural Networks (IJCNN), 2015 International Joint Conference on IEEE, 2015.
        
        [2] Rueckauer B, Lungu I-A, Hu Y, Pfeiffer M and Liu S-C (2017) Conversion of Continuous-Valued Deep Networks to
        Efficient Event-Driven Networks for Image Classification. Front. Neurosci. 11:682.

        '''
        if robust:
            print('Using robust normalization...')
        else:
            print('Using weight normalization...')

        print('normalize with bias...')
        x = norm_tensor
        i = 0
        with torch.no_grad():
            for n, m in self.named_modules():
                Name = m.__class__.__name__
                if Name in ['Conv2d','ReLU','MaxPool2d','AvgPool2d','Flatten','Linear']:
                    x = m.forward(x)
                    torch.save(x.cpu(), os.path.join(log_dir, '%s_%s_activation.pkl' % (Name, n)))
                    a = x.cpu().numpy().reshape(-1)
                    if robust:
                        self.activation_range[i] = np.percentile(a[np.nonzero(a)], 99.9)
                    else:
                        self.activation_range[i] = np.max(a)
                    i += 1
        i = 0
        last_lambda = 1.0
        for n, m in self.named_modules():
            Name = m.__class__.__name__
            if Name in ['Conv2d', 'ReLU', 'MaxPool2d', 'AvgPool2d', 'Flatten', 'Linear']:
                if Name in ['Conv2d', 'Linear']:
                    relu_output_layer = self.param_module_relu_linker[i]
                    if hasattr(m, 'weight') and m.weight is not None:
                        m.weight.data = m.weight.data * last_lambda / self.activation_range[relu_output_layer]
                    if hasattr(m, 'bias') and m.bias is not None:
                        m.bias.data = m.bias.data / self.activation_range[relu_output_layer]
                    last_lambda = self.activation_range[relu_output_layer]
                i += 1