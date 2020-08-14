import torch
import torch.nn.functional as F
import os
import spikingflow.clock_driven.ann2snn.parser as parser
import spikingflow.clock_driven.ann2snn.simulation as sim
import numpy as np
import json


class Config:
    def __init__(self):
        '''
        * :ref:`API in English <Config.__init__-en>`

        .. _Config.__init__-cn:

        ANN2SNN中自动转换由模型分析和仿真模拟两个过程构成，其中存在比较多的可配置项。
        通过加载Config中的默认配置并修改，可以设定自己模型运行时所需要的参数。
        下面，将介绍不同参数对应的配置，可行的输入范围，以及为什么要这个配置
        
        I 模型分析
        
        (1)conf['parser']['robust_norm']
        
        可行值： ``bool`` 类型
        
        说明：当设置为 ``True`` ，使用鲁棒归一化
        robust_norm即鲁棒归一化在文献 `Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification` 
        中被提出。
        ANN每层输出的分布虽然服从某个特定分布，但是数据中常常会存在较大的离群值，这会导致整体神经元发放率降低。为了解决这一问题，鲁棒归一化将缩放因子从张量的最大值调整为张量的p分位点。文献中推荐的分位点值为99.9%

        II 仿真模拟
        
        (1)conf['simulation']['reset_to_zero']
        
        可行值： ``None`` , 浮点数
        
        说明：当设置为 ``None`` ，神经元重置的时候采用减去 `v_threshold` 的方式；当为浮点数时，刚刚发放的神经元会被设置为 `v_reset` 。
        对于需要归一化的转换模型，设置为``None``是推荐的方式，具有理论保证。

        (2)conf['simulation']['encoder']['possion']
        
        可行值：``bool`` 类型
        
        说明：当设置为 ``True`` ，输入采用泊松编码器；否则，采用浮点数持续的输入仿真时长T时间。默认为  ``False``  

        (3)conf['simulation']['avg_pool']['has_neuron']
        
        可行值： ``bool`` 类型
        
        说明：当设置为 ``True`` ，平均池化层被转化为空间下采样加上一层IF神经元；否则，平均池化层仅被转化为空间下采样。默认为 ``True`` 

        (4)conf['simulation']['max_pool']['if_spatial_avg']
        
        可行值： ``bool`` 类型
        
        说明：当设置为 ``True`` ，最大池化层被转化为平均池化。这个方式根据文献可能会导致精度下降。默认为 ``False`` 

        (5)conf['simulation']['max_pool']['if_wta']
        
        可行值：``bool`` 类型
        说明：当设置为 ``True`` ，最大池化层和ANN中最大池化一样。使用ANN的最大池化意味着当感受野中一旦有脉冲即输出1。默认为 ``False`` 

        (6)conf['simulation']['max_pool']['momentum']
        可行值： ``None`` , [0,1]内浮点数
        说明：最大池化层被转化为基于动量累计脉冲的门控函数控制脉冲通道。当设置为 ``None`` ，直接累计脉冲；若为[0,1]浮点数，进行脉冲动量累积。MaxPool2d默认采用此方式。

        请注意当前的版本为较初步的版本，其后版本中的新特性配置也会以字典形式出现在配置中。

        * :ref:`API in English <Config.__init__-cn>`

        .. _Config.__init__-en:

        The automatic transformation i.e. ANN2SNN consists of two processes, model parsing and simulation, in which there are many configurable items.
        By loading the default configuration in ``Config`` and modifying, one can set the parameters needed when model runs.
        Below, the configuration is described for the different parameters.

        I Model parsing
        
        (1)conf['parser']['robust_norm']
        
        Available value：``bool``
        
        Note：when ``True``, use robust normalization
            Robust normalization is proposed in `Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification`
            Although the distribution of the output of the ANN per layer is subject to a particular distribution,
            there are often large outliers, which results in a decrease in the overall firing rate.
            To solve this problem, robust normalization adjusts the scaling factor from tensor's maximum value to
            tensor's p-percentile. The recommended p is 99.9%

        II Model simulation
        
        (1)conf['simulation']['reset_to_zero']
        
        Available value: ``None``, floating point
        
        Note: When floating point, voltage of neurons that just fired spikes will be set to
            ``v_reset``; when ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        For model that need normalization, setting to ``None`` is default, which has theoretical guaratee.

        (2)conf['simulation']['encoder']['possion']
        
        Available value: ``bool``
        
        Note: When ``True``, use Possion encoder; otherwise, use constant input over T steps. Default value is ``False``

        (3)conf['simulation']['avg_pool']['has_neuron']
        
        Available value: ``bool``
        
        Note: When ``True``, avgpool2d is converted to spatial subsampling with a layer of IF neurons; otherwise, it is only converted to spatial subsampling. Default value is ``True``

        (4)conf['simulation']['max_pool']['if_spatial_avg']
        
        Available value: ``bool``
        
        Note: When ``True``,maxpool2d is converted to avgpool2d. As referred in many literatures, this method will cause accuracy degrading. Default value is ``False``

        (5)conf['simulation']['max_pool']['if_wta']
        
        Available value: ``bool``
        
        Note: When ``True``, maxpool2d in SNN is identical with maxpool2d in ANN. Using maxpool2d in ANN means that when a spike is available in the Receptive Field, output a spike. Default value is ``False``

        (6)conf['simulation']['max_pool']['momentum']
        
        Available value: ``None``, floating point
        
        Note: By default, maxpool2d layer is converted into a gated function controled channel based on momentum cumulative spikes. When set to ``None``, the spike is accumulated directly. If set to floating point in the range of [0,1], spike momentum is accumulated.

        Note that the current version is a more preliminary version, and the new feature configuration in subsequent versions will also appear as dictionary items in ``config``.
        '''

    default_config = {'simulation':
                          {'reset_to_zero': False,
                           'encoder':
                               {'possion': False
                                },
                           'avg_pool':
                               {'has_neuron': True
                                },
                           'max_pool':
                               {'if_spatial_avg': False,
                                'if_wta': False,
                                'momentum': None
                                }
                           },
                      'parser':
                          {'robust_norm': True
                           }
                      }

    @staticmethod
    def store_config(filename, config):
        '''
        * :ref:`API in English <Config.store_config-en>`

        .. _Config.store_config-cn:

        :param filename: 保存配置的文件名
        :param config: 配置信息
        :return: ``None``

        保存配置文件

        * :ref:`中文API <Config.store_config-cn>`

        .. _Config.store_config-en:

        :param filename: configure filename
        :param config: configuration information
        :return: ``None`` 

        Store configuration file
        '''
        with open(filename, 'w') as fw:
            json.dump(config, fw)

    @staticmethod
    def load_config(filename):
        '''
        * :ref:`API in English <Config.load_config-en>`

        .. _Config.load_config-cn:

        :param filename: 保存了配置的文件名
        :return: ``None``

        加载配置文件

        * :ref:`中文API <Config.load_config-cn>`

        .. _Config.load_config-en:

        :param filename: configure filename
        :return: ``None``

        Load configuration file
        '''
        with open(filename,'r') as f:
            config = json.load(f)
            return config


def train_ann(net, device, data_loader, optimizer, loss_function, epoch=None):
    '''
    * :ref:`API in English <train_ann-en>`

    .. _train_ann-cn:

    :param net: 训练的模型
    :param device: 运行的设备
    :param data_loader: 训练集
    :param optimizer: 神经网络优化器
    :param loss_function: 损失函数
    :param epoch: 当前训练期数
    :return: ``None``

    经典的神经网络训练程序预设，便于直接调用训练网络

    * :ref:`中文API <train_ann-cn>`

    .. _train_ann-en:

    :param net: network to train
    :param device: running device
    :param data_loader: training data loader
    :param optimizer: neural network optimizer
    :param loss_function: neural network loss function
    :param epoch: current training epoch
    :return: ``None``

    Preset classic neural network training program
    '''
    net.train()
    losses = []
    correct = 0.0
    total = 0.0
    for batch, (img, label) in enumerate(data_loader):
        img = img.to(device)
        optimizer.zero_grad()
        out = net(img)
        loss = loss_function(out, label.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        correct += (out.max(dim=1)[1] == label.to(device)).float().sum().item()
        total += out.shape[0]
        if batch % 100 == 0:
            acc = correct / total
            print('Epoch %d [%d/%d] ANN Training Loss:%.3f Accuracy:%.3f' % (epoch,
                                                                     batch+1,
                                                                     len(data_loader),
                                                                     np.array(losses).mean(),
                                                                     acc))
            correct = 0.0
            total = 0.0


def val_ann(net, device, data_loader, epoch=None):
    '''
    * :ref:`API in English <val_ann-en>`

    .. _val_ann-cn:

    :param net: 待验证的模型
    :param device: 运行的设备
    :param data_loader: 测试集
    :param epoch: 当前训练期数
    :return: 验证准确率

    经典的神经网络训练程序预设，便于直接调用训练网络

    * :ref:`中文API <val_ann-cn>`

    .. _val_ann-en:

    :param net: network to test
    :param device: running device
    :param data_loader: testing data loader
    :param epoch: current training epoch
    :return: testing accuracy

    Preset classic neural network training program
    '''
    net.eval()
    correct = 0.0
    total = 0.0
    losses = []
    for batch, (img, label) in enumerate(data_loader):
        img = img.to(device)
        out = net(img)
        loss = F.cross_entropy(out, label.to(device))
        correct += (out.max(dim=1)[1] == label.to(device)).float().sum().item()
        total += out.shape[0]
        losses.append(loss.item())
    acc = correct / total
    if epoch == None:
        print('ANN Validating Accuracy:%.3f'%(acc))
    else:
        print('Epoch %d [%d/%d] ANN Validating Loss:%.3f Accuracy:%.3f' % (epoch,
                                                                           batch+1,
                                                                           len(data_loader),
                                                                           np.array(losses).mean(),
                                                                           acc))
    return acc


def save_model(net, log_dir, file_name):
    '''
    * :ref:`API in English <save_model-en>`

    .. _save_model-cn:

    :param net: 要保存的模型
    :param log_dir: 日志文件夹
    :param file_name: 文件名
    :return: ``None``

    保存模型的参数，以两种形式保存，分别为Pytorch保存的完整模型（适用于网络模型中只用了Pytorch预设模块的）
    以及模型参数（适用于网络模型中有自己定义的非参数模块无法保存完整模型）

    * :ref:`中文API <save_model-cn>`

    .. _save_model-en:

    :param net: network model to save
    :param log_dir: log file folder
    :param file_name: file name
    :return: ``None``

    Save the model, which is saved in two forms, the full model saved by Pytorch (for the network model only possessing
    the Pytorch preset module) and model parameters only (for network models that have their own defined nonparametric
    modules. In that case, Pytorch cannot save the full model)
    '''
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    torch.save(net, os.path.join(log_dir,file_name))
    torch.save(net.state_dict(), os.path.join(log_dir,'param_'+file_name))
    print('Save model to:',os.path.join(log_dir,file_name))


def standard_conversion(model_name, norm_data, test_data_loader, device, T, log_dir, config,
                        load_state_dict=False, ann=None):
    '''
    * :ref:`API in English <standard_conversion-en>`

    .. _standard_conversion-cn:

    :param model_name: 模型名字，用于文件夹中寻找保存的模型
    :param norm_data: 用于模型归一化的数据，其格式以能够作为网络输入为准。这部分数据应当从训练集抽取
    :param test_data_loader: 测试数据加载器，用于仿真
    :param device: 运行的设备
    :param T: 仿真时长
    :param log_dir: 用于保存临时文件的日志文件夹
    :param config: 用于转换的配置
    :param load_state_dict: 如果希望使用state dict加载的模型，将此参数设置为 ``True`` 
    :param ann: 用于加载state dict的模型，使用的模块均为Pytorch内置模块
    :return: ``None``

    对加载的模型（或模型参数）进行模型转化并且对转化后SNN进行仿真，输出仿真结果
    用户自定义的转换和测试程序可以仿造此程序

    * :ref:`中文API <standard_conversion-cn>`

    .. _standard_conversion-en:

    :param model_name: model name is used to find saved model in log_dir
    :param norm_data: data used to normalize the model. Format of the data should be capable of being fed into the model. Norm data should be randomly choosen from training data
    :param test_data_loader: testing data loader, used for simulating
    :param device: running device
    :param T: simulating steps
    :param log_dir: Name of the folder used to save temporary files
    :param config: conversion config
    :param load_state_dict: set ``True`` if one want to load saved 'state dict'
    :param ann: ANN used to load state dict. Its modules should be Pytorch modules.
    :return: ``None``

    Convert the loaded model (or model loaded from parameters) and simulate the converted SNN.
    Output the simulation process and conversion loss
    User-defined conversion and validation function can learn from this function.
    '''
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print('Load best model for Model:%s...' % (model_name))
    if not load_state_dict:
        ann = torch.load(os.path.join(log_dir, model_name + '.pkl'))
    else:
        assert(ann is not None)
        ann.load_state_dict(torch.load(os.path.join(log_dir, 'param_' + model_name + '.pkl')))

    # 分析模型
    # Parse the model
    parsed_ann = parser.ModelParser()
    parsed_ann.parse(ann, log_dir)

    # 测试分析后的模型的ann性能，理论上应该和分析前模型相同
    # Test the parsed model, it should perform as good as before
    ann_acc = val_ann(parsed_ann, device, test_data_loader)

    # 保存分析的模型
    # Save the parsed model
    save_model(parsed_ann, log_dir, 'parsed_' + model_name + '.pkl')

    # 归一化模型
    # Normalize the parsed model
    parsed_ann.normalize_model(norm_data.to(device), log_dir, robust=config['parser']['robust_norm'])
    print('Print Parsed ANN model Structure:')
    print(parsed_ann)

    # 保存归一化模型
    # Save the normalized model
    save_model(parsed_ann, log_dir, 'normalized_' + model_name + '.pkl')

    # 定义一个模板snn模型，并加载归一化后的模型
    # Define an empty snn model and load the normalized model into the empty snn
    if config['simulation']['reset_to_zero']:
        snn = sim.SNN(v_reset=0.0)
    else:
        snn = sim.SNN()
    snn.load_parsed_model(parsed_model=parsed_ann,
                          avg_pool_has_neuron=config['simulation']['avg_pool']['has_neuron'],
                          max_pool_spatial_avg=config['simulation']['max_pool']['if_spatial_avg'],
                          max_pool_wta=config['simulation']['max_pool']['if_wta'],
                          max_pool_momentum=config['simulation']['max_pool']['momentum'])
    print('Print Simulated SNN model Structure:')
    print(snn)

    # 测试SNN在验证集上的性能
    # Test performance of SNN
    snn_acc = sim.simulate_snn(snn=snn,
                               device=device,
                               data_loader=test_data_loader,
                               T=T,
                               possion=config['simulation']['encoder']['possion'])

    # 输出Summary和转换带来的准确率损失
    # Output the conversion loss (loss on Accuracy)
    print('Summary:\tANN Accuracy:%.4f%%  \tSNN Accuracy:%.4f%% [%s %.4f%%]' % (ann_acc * 100, snn_acc * 100,
                                                                                'Increased' if snn_acc > ann_acc else 'Decreased',
                                                                                np.abs(ann_acc * 100 - snn_acc * 100)
                                                                                ))