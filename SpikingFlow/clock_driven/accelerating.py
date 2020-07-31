import torch
import torch.nn as nn
import torch.nn.functional as F

class spike_multiply_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_a: torch.Tensor, spike_b: torch.Tensor):
        # y = spike_a * spike_b
        assert spike_a.shape == spike_b.shape, print('x.shape != spike.shape')  # 禁用广播机制
        if spike_a.dtype == torch.bool:
            mask_a = spike_a
        else:
            mask_a = spike_a.bool()
        if spike_b.dtype == torch.bool:
            mask_b = spike_b
        else:
            mask_b = spike_b.bool()

        if spike_a.requires_grad and spike_b.requires_grad:
            ctx.save_for_backward(mask_a, mask_b)
        elif spike_a.requires_grad and not spike_b.requires_grad:
            ctx.save_for_backward(mask_b)
        elif not spike_a.requires_grad and spike_b.requires_grad:
            ctx.save_for_backward(mask_a)
        ret = mask_a.logical_and(mask_b).float()
        ret.requires_grad_(spike_a.requires_grad or spike_b.requires_grad)
        return ret

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_spike_a = None
        grad_spike_b = None
        # grad_spike_a = grad_output * grad_spike_b
        # grad_spike_b = grad_output * grad_spike_a
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_spike_a = grad_output.masked_fill(torch.logical_not(ctx.saved_tensors[1]), 0)
            grad_spike_b = grad_output.masked_fill(torch.logical_not(ctx.saved_tensors[0]), 0)
        elif ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            grad_spike_a = grad_output.masked_fill(torch.logical_not(ctx.saved_tensors[0]), 0)
        elif not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_spike_b = grad_output.masked_fill(torch.logical_not(ctx.saved_tensors[0]), 0)

        return grad_spike_a, grad_spike_b

class multiply_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, spike: torch.Tensor):
        # y = x * spike
        # x乘spike，等价于将x中spike == 0的位置全部填充为0
        assert x.shape == spike.shape, print('x.shape != spike.shape')  # 禁用广播机制
        if spike.dtype == torch.bool:
            mask = torch.logical_not(spike)
        else:
            mask = torch.logical_not(spike.bool())
        if x.requires_grad and spike.requires_grad:
            ctx.save_for_backward(mask, x)
        elif x.requires_grad and not spike.requires_grad:
            ctx.save_for_backward(mask)
        elif not x.requires_grad and spike.requires_grad:
            ctx.save_for_backward(x)
        return x.masked_fill(mask, 0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        grad_spike = None
        # grad_x = grad_output * spike
        # grad_spike = grad_output * x
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_x = grad_output.masked_fill(ctx.saved_tensors[0], 0)
            grad_spike = grad_output * ctx.saved_tensors[1]
        elif ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            grad_x = grad_output.masked_fill(ctx.saved_tensors[0], 0)
        elif not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_spike = grad_output * ctx.saved_tensors[0]

        return grad_x, grad_spike


class add_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, spike: torch.Tensor):
        # y = x + spike
        # x乘spike，等价于将x中spike == 1的位置增加1
        assert x.shape == spike.shape, print('x.shape != spike.shape')  # 禁用广播机制
        if spike.dtype == torch.bool:
            mask = spike
        else:
            mask = spike.bool()
        y = x.clone()
        y[mask] += 1
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        grad_spike = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        if ctx.needs_input_grad[1]:
            grad_spike = grad_output

        return grad_x, grad_spike


class subtract_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, spike: torch.Tensor):
        # y = x - spike
        # x乘spike，等价于将x中spike == 1的位置减去1
        assert x.shape == spike.shape, print('x.shape != spike.shape')  # 禁用广播机制
        if spike.dtype == torch.bool:
            mask = spike
        else:
            mask = spike.bool()
        y = x.clone()
        y[mask] -= 1
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        grad_spike = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        if ctx.needs_input_grad[1]:
            grad_spike = - grad_output

        return grad_x, grad_spike

def add(x: torch.Tensor, spike: torch.Tensor):
    '''
    * :ref:`API in English <add-en>`

    .. _add-cn:

    :param x: 任意tensor
    :param spike: 脉冲tensor。要求 ``spike`` 中的元素只能为 ``0`` 和 ``1``，或只为 ``False`` 和 ``True``，且 ``spike.shape`` 必须与 ``x.shape`` 相同
    :return: ``x + spike``

    针对与脉冲这一特殊的数据类型，进行前反向传播加速并保持数值稳定的加法运算。

    * :ref:`中文API <add-cn>`

    .. _add-en:

    :param x: an arbitrary tensor
    :param spike: a spike tensor. The elements in ``spike`` must be ``0`` and ``1`` or ``False`` and ``True``, and ``spike.shape`` should be same
        with ``x.shape``
    :return: ``x + spike``

    Add operation for an arbitrary tensor and a spike tensor, which is specially optimized for memory, speed, and
    numerical stability.
    '''
    return add_spike.apply(x, spike)

def sub(x: torch.Tensor, spike: torch.Tensor):
    '''
    * :ref:`API in English <sub-en>`

    .. _sub-cn:

    :param x: 任意tensor
    :param spike: 脉冲tensor。要求 ``spike`` 中的元素只能为 ``0`` 和 ``1``，或只为 ``False`` 和 ``True``，且 ``spike.shape`` 必须与 ``x.shape`` 相同
    :return: ``x - spike``

    针对与脉冲这一特殊的数据类型，进行前反向传播加速并保持数值稳定的减法运算。

    * :ref:`中文API <sub-cn>`

    .. _sub-en:

    :param x: an arbitrary tensor
    :param spike: a spike tensor. The elements in ``spike`` must be ``0`` and ``1`` or ``False`` and ``True``, and ``spike.shape`` should be same
        with ``x.shape``
    :return: ``x - spike``

    Subtract operation for an arbitrary tensor and a spike tensor, which is specially optimized for memory, speed, and
    numerical stability.
    '''
    return subtract_spike.apply(x, spike)

def mul(x: torch.Tensor, spike: torch.Tensor, x_is_spike=False):
    '''
    * :ref:`API in English <mul-en>`

    .. _mul-cn:

    :param x: 任意tensor
    :param spike: 脉冲tensor。要求 ``spike`` 中的元素只能为 ``0`` 和 ``1``，或只为 ``False`` 和 ``True``，且 ``spike.shape`` 必须与 ``x.shape`` 相同
    :param x_is_spike: ``x`` 是否也是脉冲数据，即满足元素只能为 ``0`` 和 ``1``，或只为 ``False`` 和 ``True``。若 ``x`` 满足
        这一条件，则会调用更高级别的加速
    :return: ``x * spike``

    针对与脉冲这一特殊的数据类型，进行前反向传播加速并保持数值稳定的乘法运算。

    * :ref:`中文API <mul-cn>`

    .. _mul-en:

    :param x: an arbitrary tensor
    :param spike: a spike tensor. The elements in ``spike`` must be ``0`` and ``1`` or ``False`` and ``True``, and ``spike.shape`` should be same
        with ``x.shape``
    :param x_is_spike: whether ``x`` is the spike. When the elements in ``x`` are ``0`` and ``1`` or ``False`` and ``True``,
        this param can be ``True`` and this function will call an advanced accelerator
    :return: ``x * spike``

    Multiplication operation for an arbitrary tensor and a spike tensor, which is specially optimized for memory, speed, and
    numerical stability.
    '''
    if x_is_spike:
        return spike_multiply_spike.apply(x, spike)
    else:
        return multiply_spike.apply(x, spike)




class soft_vlotage_transform_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        # v = v - spike * v_threshold
        mask = spike.bool()  # 表示释放脉冲的位置
        if spike.requires_grad:
            ctx.v_threshold = v_threshold
        ret = v.clone()
        ret[mask] -= v_threshold
        return ret  # 释放脉冲的位置，电压设置为v_reset，out-of-place操作

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_v = None
        grad_spike = None
        if ctx.needs_input_grad[0]:
            grad_v = grad_output  # 因为输出对v的梯度是全1
        if ctx.needs_input_grad[1]:
            grad_spike = - ctx.v_threshold * grad_output
        return grad_v, grad_spike, None

def soft_voltage_transform(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
    '''
    * :ref:`API in English <soft_voltage_transform-en>`

    .. _soft_voltage_transform-cn:

    :param v: 重置前电压
    :param spike: 释放的脉冲
    :param v_threshold: 阈值电压
    :return: 重置后的电压

    根据释放的脉冲，以soft方式重置电压，即释放脉冲后，电压会减去阈值：:math:`v = v - s \\cdot v_{threshold}`。

    该函数针对脉冲数据进行了前反向传播的加速，并能节省内存，且保持数值稳定。

    * :ref:`中文API <soft_voltage_transform-cn>`

    .. _soft_voltage_transform-en:

    :param v: voltage before reset
    :param spike: fired spikes
    :param v_threshold: threshold voltage
    :return: voltage after reset

    Reset the voltage according to fired spikes in a soft way, which means that voltage of neurons that just fired spikes
    will subtract ``v_threshold``: :math:`v = v - s \\cdot v_{threshold}`.

    This function is specially optimized for memory, speed, and numerical stability.
    '''
    return soft_vlotage_transform_function.apply(v, spike, v_threshold)

class hard_voltage_transform_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        # v = v * (1 - spikes) + v_reset * spikes
        mask = spike.bool()  # 表示释放脉冲的位置
        if v.requires_grad and spike.requires_grad:
            ctx.save_for_backward(mask, v_reset - v)
        elif v.requires_grad and not spike.requires_grad:
            ctx.save_for_backward(mask)
        elif not v.requires_grad and spike.requires_grad:
            ctx.save_for_backward(v_reset - v)

        return v.masked_fill(mask, v_reset)  # 释放脉冲的位置，电压设置为v_reset，out-of-place操作

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_v = None
        grad_spike = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_v = grad_output.masked_fill(ctx.saved_tensors[0], 0)
            grad_spike = grad_output * ctx.saved_tensors[1]
        elif ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            grad_v = grad_output.masked_fill(ctx.saved_tensors[0], 0)
        elif not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_spike = grad_output * ctx.saved_tensors[0]

        return grad_v, grad_spike, None

def hard_voltage_transform(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
    '''
    * :ref:`API in English <hard_voltage_transform-en>`

    .. _hard_voltage_transform-cn:

    :param v: 重置前电压
    :param spike: 释放的脉冲
    :param v_threshold: 阈值电压
    :return: 重置后的电压

    根据释放的脉冲，以hard方式重置电压，即释放脉冲后，电压会被直接设置成 ``v_reset``。

    该函数针对脉冲数据进行了前反向传播的加速，并能节省内存，且保持数值稳定。

    * :ref:`中文API <hard_voltage_transform-cn>`

    .. _hard_voltage_transform-en:

    :param v: voltage before reset
    :param spike: fired spikes
    :param v_threshold: threshold voltage
    :return: voltage after reset

    Reset the voltage according to fired spikes in a hard way, which means that voltage of neurons that just fired spikes
    will be set to ``v_reset``.

    This function is specially optimized for memory, speed, and numerical stability.
    '''
    return hard_voltage_transform_function.apply(v, spike, v_reset)

class ModelPipeline(nn.Module):
    def __init__(self):
        '''
        一个基于流水线多GPU串行并行的基类，使用者只需要继承 ``ModelPipeline``，然后调\
        用 ``append(nn_module, gpu_id)``，就可以将 ``nn_module`` 添加到流水线中，并且 ``nn_module`` 会被运行在 ``gpu_id`` 上。\
        在调用模型进行计算时， ``forward(x, split_sizes)`` 中的 ``split_sizes`` 指的是输入数据 ``x`` 会在维度0上被拆分成\
        每 ``split_size`` 一组，得到 ``[x[0], x[1], ...]``，这些数据会被串行的送入 ``module_list`` 中保存的各个模块进行计算。

        例如将模型分成4部分，因而 ``module_list`` 中有4个子模型；将输入分割为3部分，则每次调用 ``forward(x, split_sizes)`` ，函数内部的\
        计算过程如下：

        .. code-block:: python

                step=0     x0, x1, x2  |m0|    |m1|    |m2|    |m3|

                step=1     x0, x1      |m0| x2 |m1|    |m2|    |m3|

                step=2     x0          |m0| x1 |m1| x2 |m2|    |m3|

                step=3                 |m0| x0 |m1| x1 |m2| x2 |m3|

                step=4                 |m0|    |m1| x0 |m2| x1 |m3| x2

                step=5                 |m0|    |m1|    |m2| x0 |m3| x1, x2

                step=6                 |m0|    |m1|    |m2|    |m3| x0, x1, x2

        不使用流水线，则任何时刻只有一个GPU在运行，而其他GPU则在等待这个GPU的数据；而使用流水线，例如上面计算过程中的 ``step=3`` 到\
        ``step=4``，尽管在代码的写法为顺序执行：

        .. code-block:: python

            x0 = m1(x0)
            x1 = m2(x1)
            x2 = m3(x2)

        但由于PyTorch优秀的特性，上面的3行代码实际上是并行执行的，因为这3个在CUDA上的计算使用各自的数据，互不影响。
        用于解决显存不足的模型流水线。将一个模型分散到各个GPU上，流水线式的进行训练。

        运行时建议先取一个很小的batch_size，然后观察各个GPU的显存占用，并调整每个module_list中包含的模型比例。
        '''
        super().__init__()
        self.module_list = nn.ModuleList()
        self.gpu_list = []


    def append(self, nn_module, gpu_id):
        '''
        :param nn_module: 新添加的module
        :param gpu_id:  该模型所在的GPU，不需要带“cuda:”的前缀。例如“2”
        :return: None

        将nn_module添加到流水线中，nn_module会运行在设备gpu_id上。添加的nn_module会按照它们的添加顺序运行。例如首先添加了\
        fc1，又添加了fc2，则实际运行是按照input_data->fc1->fc2->output_data的顺序运行。
        '''
        self.module_list.append(nn_module.to('cuda:' + gpu_id))
        self.gpu_list.append('cuda:' + gpu_id)

    def constant_forward(self, x, T, reduce=True):
        '''
        :param x: 输入数据
        :param T: 运行时长
        :param reduce: 为True则返回运行T个时长，得到T个输出的和；为False则返回这T个输出
        :return: T个输出的和或T个输出

        让本模型以恒定输入x运行T次，这常见于使用频率编码的SNN。这种方式比forward(x, split_sizes)的运行速度要快很多。
        '''

        pipeline = []  # pipeline[i]中保存要送入到m[i]的数据
        for i in range(self.gpu_list.__len__()):
            pipeline.append(None)

        pipeline[0] = x.to(self.gpu_list[0])

        # 跑满pipeline
        # 假设m中有5个模型，m[0] m[1] m[2] m[3] m[4]，则代码执行顺序为
        #
        # p[ 1 ] = m[ 0 ](p[ 0 ])
        #
        # p[ 2 ] = m[ 1 ](p[ 1 ])
        # p[ 1 ] = m[ 0 ](p[ 0 ])
        #
        # p[ 3 ] = m[ 2 ](p[ 2 ])
        # p[ 2 ] = m[ 1 ](p[ 1 ])
        # p[ 1 ] = m[ 0 ](p[ 0 ])
        #
        # p[ 4 ] = m[ 3 ](p[ 3 ])
        # p[ 3 ] = m[ 2 ](p[ 2 ])
        # p[ 2 ] = m[ 1 ](p[ 1 ])
        # p[ 1 ] = m[ 0 ](p[ 0 ])

        for i in range(0, self.gpu_list.__len__()):
            for j in range(i, 0, -1):
                if j - 1 == 0:
                    pipeline[j] = self.module_list[j - 1](pipeline[j - 1])
                else:
                    pipeline[j] = self.module_list[j - 1](pipeline[j - 1].to(self.gpu_list[j - 1]))

        t = 0  # 记录从流水线输出的总数量
        while True:
            for i in range(self.gpu_list.__len__(), 0, -1):
                if i == self.gpu_list.__len__():
                    # 获取输出
                    if t == 0:
                        if reduce:
                            ret = self.module_list[i - 1](pipeline[i - 1].to(self.gpu_list[i - 1]))
                        else:
                            ret = []
                            ret.append(self.module_list[i - 1](pipeline[i - 1].to(self.gpu_list[i - 1])))
                    else:
                        if reduce:
                            ret += self.module_list[i - 1](pipeline[i - 1].to(self.gpu_list[i - 1]))
                        else:
                            ret.append(self.module_list[i - 1](pipeline[i - 1].to(self.gpu_list[i - 1])))
                    t += 1
                    if t == T:
                        if reduce == False:
                            return torch.cat(ret, dim=0)
                        return ret

                else:
                    pipeline[i] = self.module_list[i - 1](pipeline[i - 1].to(self.gpu_list[i - 1]))

    def forward(self, x, split_sizes):
        '''
        :param x: 输入数据
        :param split_sizes: 输入数据x会在维度0上被拆分成每split_size一组，得到[x0, x1, ...]，这些数据会被串行的送入\
        module_list中的各个模块进行计算
        :return: 输出数据
        '''

        assert x.shape[0] % split_sizes == 0, print('x.shape[0]不能被split_sizes整除！')
        x = list(x.split(split_sizes, dim=0))
        x_pos = []  # x_pos[i]记录x[i]应该通过哪个module

        for i in range(x.__len__()):
            x_pos.append(i + 1 - x.__len__())

        while True:
            for i in range(x_pos.__len__() - 1, -1, -1):
                if 0 <= x_pos[i] <= self.gpu_list.__len__() - 1:
                    x[i] = self.module_list[x_pos[i]](x[i].to(self.gpu_list[x_pos[i]]))
                x_pos[i] += 1
            if x_pos[0] == self.gpu_list.__len__():
                break

        return torch.cat(x, dim=0)