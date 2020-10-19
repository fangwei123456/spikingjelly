import torch
import torch.nn as nn
import spikingjelly.clock_driven.neuron as neuron
import spikingjelly.clock_driven.ann2snn.modules as modules
import spikingjelly.clock_driven.encoding as encoding

class PyTorch_Converter(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=None):
        '''
        * :ref:`API in English <SNN.__init__-en>`

        .. _SNN.__init__-cn:

        :param v_threshold: v_threshold一般设置为1.0，对应ReLU的spiking版本:IFNode
        :param v_reset: v_reset设置为 ``None`` ，神经元reset的时候采用减去v_threshold的方式；否则，刚刚发放的脉冲会被设置为v_reset

        SNN类用来加载归一化好的模型，并且将其中的torch.nn.ReLU转化为spikingjelly.IFNode。加载后的SNN可以使用simulate_snn函数进行仿真。

        * :ref:`API in English <SNN.__init__-cn>`

        .. _SNN.__init__-en:

        :param v_threshold: v_threshold typically is set to 1.0, which corresponds IFNode
        :param v_reset: If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``

        class SNN is used to load a normalized model and replace torch.nn.ReLU to spikingjelly.IFnode.
        Loaded SNN can be simulated using function simulate_snn below.
        '''
        super().__init__()
        self.module_list = []
        self.module_len = 0
        self.network = torch.nn.Sequential()
        self.v_threshold = v_threshold
        self.v_reset = v_reset

    def load_parsed_model(self, parsed_model, avg_pool_has_neuron=True,
                          max_pool_spatial_avg=False, max_pool_wta=False, max_pool_momentum=None):
        '''
        * :ref:`API in English <SNN.load_parsed_model-en>`

        .. _SNN.load_parsed_model-cn:

        :param parsed_model: 待转化的ANN模型
        :param avg_pool_has_neuron: 当设置为 ``True`` ，平均池化层被转化为空间下采样加上一层IF神经元；否则，平均池化层仅被转化为空间下采样。默认为 ``True``
        :param max_pool_spatial_avg: 当设置为 ``True`` ，最大池化层被转化为平均池化。默认为 ``False``
        :param max_pool_wta: 当设置为 ``True`` ，最大池化层和ANN中最大池化一样。使用ANN的最大池化意味着当感受野中一旦有脉冲即输出1。默认为 ``False``
        :param max_pool_momentum: 默认地，最大池化层被转化为基于动量累计脉冲的门控函数控制脉冲通道。当设置为 ``None`` ，直接累计脉冲；若为[0,1]浮点数，进行脉冲动量累积
        :return: ``None``

        将准备好的SNN转化为ANN。其中的关键步骤为：

        1、将原来的ReLU激活转换为IF神经元

        2、基于函数参数确定AvgPool2d和MaxPool2d在SNN中的对应形式
        其中由于AvgPool2d的输入是0、1的脉冲序列。由于平均池化，其参数均为正数，所以在ANN中在其后加上ReLU激活或者不加并不影响ANN的结果。
        同样地，SNN中，AvgPool2d后是否增加IF神经元层是可选的。默认AvgPool2d后接IF神经元层。

        对于MaxPool2d，不同的转换方式可能需要对MaxPool2d有不同的处理。目前提供三种方案。
        第一种：转换为受门控函数控制的池化。这个方案可能会涉及脉冲的动量累积或者直接累积。目前此方案具备最佳解决方案的潜力，被设为默认方案。虽然网络依然可能会因为MaxPool2d而降低准确率。
        详情见 ``ann2snn.modules.MaxPool2d``

        第二种：转换为空间下采样。最终结果就是使用AvgPool2d替换MaxPool2d
        第三种：不转换。如此和ANN中MaxPool2d表现出的WTA性质一致。可能会因为脉冲变得更加密集而影响准确率。

        * :ref:`中文API <SNN.load_parsed_model-cn>`

        .. _SNN.load_parsed_model-en:

        :param parsed_model: ANN model to be converted to SNN
        :param avg_pool_has_neuron: when ``True``, avgpool2d is converted to spatial subsampling with a layer of IF neurons;
                        otherwise, it is only converted to spatial subsampling. Default value is ``True``
        :param max_pool_spatial_avg: when ``True``,maxpool2d is converted to avgpool2d. Default value is ``False``
        :param max_pool_wta: when ``True``, maxpool2d in SNN is identical with maxpool2d in ANN. Using maxpool2d in ANN means that when a spike is available in the Receptive Field, output a spike. Default value is ``False``
        :param max_pool_momentum: By default, maxpool2d layer is converted into a gated function controled channel based on momentum cumulative spikes. When set to ``None``, the spike is accumulated directly. If set to fp number in the range of [0,1], spike momentum is accumulated.
        :return: ``None``

        Convert prepared ANN into SNN.
        The key steps are:

        1. convert the original ReLU activation to IF neurons.

        2. based on function parameters, determine the corresponding form of AvgPool2d and MaxPool2d in SNN.

        The input of AvgPool2d is the spike train of 0 and 1. Since the equivalent parameters of AvgPool2d are positive, whether ReLU activation is added to ANN does not affect the output of ANN.

        Similarly, in SNN, adding the IF neuron layer after AvgPool2d is optional. By default, AvgPool2d is followed by a IF neuron layer.

        For MaxPool2d, different conversion methods may require different treatments for MaxPool2d.
        Three options are currently available.

        I.Converting to pooling controlled by a gated function. This scheme may involve the accumulation of pulses, either momentum or accumulation. Currently this scheme has the potential to be the best solution and is set as the default solution. Although the network may still be less accurate due to MaxPool2d.
            See ``ann2snn.modules.MaxPool2d`` for details.

        II.Converting to spatial subsampling. which is equivalent to take MaxPool2d as AvgPool2d.

        III.Donot convert. This is consistent with the WTA properties shown by MaxPool2d in ANN. Accuracy may be affected as spikes become denser.
        '''
        for n, m in parsed_model.named_modules():
            Name = m.__class__.__name__
            if Name == 'Linear':
                self.module_list.append(m)
            if Name == 'Conv2d':
                self.module_list.append(m)
            if Name == 'ReLU':
                self.module_list.append(neuron.IFNode(v_threshold=self.v_threshold,
                                                      v_reset=self.v_reset))
            if Name == 'MaxPool2d':
                if max_pool_wta:
                    self.module_list.append(nn.MaxPool2d(
                        kernel_size=m.kernel_size,
                        stride=m.stride,
                        padding=m.padding
                    ))
                elif max_pool_spatial_avg:
                    self.module_list.append(nn.AvgPool2d(
                        kernel_size=m.kernel_size,
                        stride=m.stride,
                        padding=m.padding
                    ))
                else:
                    self.module_list.append(modules.MaxPool2d(
                        kernel_size=m.kernel_size,
                        stride=m.stride,
                        padding=m.padding,
                        momentum=max_pool_momentum
                    ))
            if Name == 'AvgPool2d':
                self.module_list.append(m)
                if avg_pool_has_neuron:
                    self.module_list.append(neuron.IFNode(v_threshold=self.v_threshold,
                                                          v_reset=self.v_reset))
            if Name == 'Flatten':
                self.module_list.append(m)
            self.module_len += 1
        self.network = torch.nn.Sequential(*self.module_list)

    def forward(self, x):
        return self.network(x)