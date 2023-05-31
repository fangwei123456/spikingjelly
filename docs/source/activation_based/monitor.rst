监视器
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

在 :class:`spikingjelly.activation_based.monitor` 中定义了几个通用的监视器类，用户可以使用这些监视器实现复杂的数据\
记录功能。下面以一个简单的网络为例进行介绍。


基本使用
-------------------------------------------
所有的监视器的用法类似，以 :class:`spikingjelly.activation_based.monitor.OutputMonitor` 为例进行介绍。

首先我们搭建起一个简单的多步网络。为了避免无脉冲释放，我们将权重全部设置为正值：

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import monitor, neuron, functional, layer

    net = nn.Sequential(
        layer.Linear(8, 4),
        neuron.IFNode(),
        layer.Linear(4, 2),
        neuron.IFNode()
    )

    for param in net.parameters():
        param.data.abs_()

    functional.set_step_mode(net, 'm')

:class:`spikingjelly.activation_based.monitor.OutputMonitor` 可以记录网络中任何类型为 ``instance`` 的模块的输出。\
脉冲神经元层的输出即为脉冲，因此我们可以使用 :class:`OutputMonitor <spikingjelly.activation_based.monitor.OutputMonitor>` \
来构建一个脉冲监视器，记录网络中所有 ``neuron.IFNode`` 的输出脉冲：

.. code-block:: python

    spike_seq_monitor = monitor.OutputMonitor(net, neuron.IFNode)
    T = 4
    N = 1
    x_seq = torch.rand([T, N, 8])

    with torch.no_grad():
        net(x_seq)

要记录的数据，会根据生成顺序，保存在 ``.records`` 的 ``list`` 中：

.. code-block:: python

    print(f'spike_seq_monitor.records=\n{spike_seq_monitor.records}')

输出为：

.. code-block:: shell

    spike_seq_monitor.records=
    [tensor([[[0., 0., 0., 0.]],

            [[1., 1., 1., 1.]],

            [[0., 0., 0., 0.]],

            [[1., 1., 1., 1.]]]), tensor([[[0., 0.]],

            [[1., 0.]],

            [[0., 1.]],

            [[1., 0.]]])]

也可以使用索引操作，直接访问被记录的第 ``i`` 个数据：

.. code-block:: python

    print(f'spike_seq_monitor[0]={spike_seq_monitor[0]}')

输出为：

.. code-block:: shell

    spike_seq_monitor[0]=tensor([[[0., 0., 0., 0.]],

            [[1., 1., 1., 1.]],

            [[0., 0., 0., 0.]],

            [[1., 1., 1., 1.]]])
    

``.monitored_layers`` 记录了被监视器监控的层的名字：

.. code-block:: python

    print(f'net={net}')
    print(f'spike_seq_monitor.monitored_layers={spike_seq_monitor.monitored_layers}')

输出为：

.. code-block:: shell

    net=Sequential(
    (0): Linear(in_features=8, out_features=4, bias=True)
    (1): IFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=m, backend=torch
        (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
    )
    (2): Linear(in_features=4, out_features=2, bias=True)
    (3): IFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=m, backend=torch
        (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
    )
    )
    spike_seq_monitor.monitored_layers=['1', '3']

可以直接通过层的名字作为索引，访问某一层被记录的数据。这返回的是一个 ``list`` ：

.. code-block:: python

    print(f"spike_seq_monitor['1']={spike_seq_monitor['1']}")

输出为：

.. code-block:: shell

    spike_seq_monitor['1']=[tensor([[[0., 0., 0., 0.]],

        [[1., 1., 1., 1.]],

        [[0., 0., 0., 0.]],

        [[1., 1., 1., 1.]]])]

可以通过调用 ``.clear_recorded_data()`` 来清空已经记录的数据：

.. code-block:: python

    spike_seq_monitor.clear_recorded_data()
    print(f'spike_seq_monitor.records={spike_seq_monitor.records}')
    print(f"spike_seq_monitor['1']={spike_seq_monitor['1']}")

输出为：

.. code-block:: shell

    spike_seq_monitor.records=[]
    spike_seq_monitor['1']=[]

所有的 ``monitor`` 在析构时都会自动删除已经注册的钩子，但python的内存回收机制并不保证在手动调用 ``del`` 时一定会进行析构。因此删除一个监视器，并不能保证钩子也立刻被删除：

.. code-block:: python

    del spike_seq_monitor
    # 钩子可能仍然在起作用

若想立刻删除钩子，应该通过以下方式：

.. code-block:: python

    spike_seq_monitor.remove_hooks()


:class:`OutputMonitor <spikingjelly.activation_based.monitor.OutputMonitor>` 还支持在记录数据时就对数据进行简单的处理，只需要\
指定构造函数中的 ``function_on_output`` 即可。 ``function_on_output`` 的默认值是 ``lambda x: x``，也就是默认不进行任何处理。\
我们想要记录每个时刻的脉冲发放频率，首先要定义脉冲发放频率如何计算：

.. code-block:: python

    def cal_firing_rate(s_seq: torch.Tensor):
        # s_seq.shape = [T, N, *]
        return s_seq.flatten(1).mean(1)

接下来就可以以此来构建发放率监视器：

.. code-block:: python

    fr_monitor = monitor.OutputMonitor(net, neuron.IFNode, cal_firing_rate)

通过 ``.disable()`` 可以让 ``monitor`` 暂停记录，而 ``.enable()`` 则可以让其重新开始记录：

.. code-block:: python

    with torch.no_grad():
        functional.reset_net(net)
        fr_monitor.disable()
        net(x_seq)
        functional.reset_net(net)
        print(f'after call fr_monitor.disable(), fr_monitor.records=\n{fr_monitor.records}')

        fr_monitor.enable()
        net(x_seq)
        print(f'after call fr_monitor.enable(), fr_monitor.records=\n{fr_monitor.records}')
        functional.reset_net(net)
        del fr_monitor

输出为：

.. code-block:: shell

    after call fr_monitor.disable(), fr_monitor.records=
    []
    after call fr_monitor.enable(), fr_monitor.records=
    [tensor([0.0000, 1.0000, 0.5000, 1.0000]), tensor([0., 1., 0., 1.])]

记录模块成员变量
-------------------------------------------
若想记录模块的成员变量，例如神经元的电压，可以通过 :class:`spikingjelly.activation_based.monitor.AttributeMonitor` \
实现。

神经元构造参数中的 ``store_v_seq: bool = False`` 表示在默认情况下，只记录当前时刻的电压，不记录所有时刻的电压序列。现在\
我们想记录所有时刻的电压，则将其更改为 ``True``：

.. code-block:: python

    for m in net.modules():
        if isinstance(m, neuron.IFNode):
            m.store_v_seq = True

接下来，新建记录电压序列的监视器并进行记录：

.. code-block:: python

    v_seq_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=net, instance=neuron.IFNode)
    with torch.no_grad():
        net(x_seq)
        print(f'v_seq_monitor.records=\n{v_seq_monitor.records}')
        functional.reset_net(net)
        del v_seq_monitor

输出为：

.. code-block:: shell

    v_seq_monitor.records=
    [tensor([[[0.8102, 0.8677, 0.8153, 0.9200]],

            [[0.0000, 0.0000, 0.0000, 0.0000]],

            [[0.0000, 0.8129, 0.0000, 0.9263]],

            [[0.0000, 0.0000, 0.0000, 0.0000]]]), tensor([[[0.2480, 0.4848]],

            [[0.0000, 0.0000]],

            [[0.8546, 0.6674]],

            [[0.0000, 0.0000]]])]

记录模块输入
-------------------------------------------
设置输入监视器的方法，和设置输出监视器的如出一辙：

.. code-block:: python

    input_monitor = monitor.InputMonitor(net, neuron.IFNode)
    with torch.no_grad():
        net(x_seq)
        print(f'input_monitor.records=\n{input_monitor.records}')
        functional.reset_net(net)
        del input_monitor

输出为：

.. code-block:: shell

    input_monitor.records=
    [tensor([[[1.1710, 0.7936, 0.9325, 0.8227]],

            [[1.4373, 0.7645, 1.2167, 1.3342]],

            [[1.6011, 0.9850, 1.2648, 1.2650]],

            [[0.9322, 0.6143, 0.7481, 0.9770]]]), tensor([[[0.8072, 0.7733]],

            [[1.1186, 1.2176]],

            [[1.0576, 1.0153]],

            [[0.4966, 0.6030]]])]

记录模块的输入梯度 :math:`\frac{\partial L}{\partial Y}`
--------------------------------------------------------------------------------------
如果我们想要记录每一层脉冲神经元的输入梯度 :math:`\frac{\partial L}{\partial S}`，则可以使用 \
:class:`spikingjelly.activation_based.monitor.GradOutputMonitor` 轻松实现：

.. code-block:: python

    spike_seq_grad_monitor = monitor.GradOutputMonitor(net, neuron.IFNode)
    net(x_seq).sum().backward()
    print(f'spike_seq_grad_monitor.records=\n{spike_seq_grad_monitor.records}')
    functional.reset_net(net)
    del spike_seq_grad_monitor

输出为：

.. code-block:: python

    spike_seq_grad_monitor.records=
    [tensor([[[1., 1.]],

            [[1., 1.]],

            [[1., 1.]],

            [[1., 1.]]]), tensor([[[ 0.0803,  0.0383,  0.1035,  0.1177]],

            [[-0.1013, -0.1346, -0.0561, -0.0085]],

            [[ 0.5364,  0.6285,  0.3696,  0.1818]],

            [[ 0.3704,  0.4747,  0.2201,  0.0596]]])]

由于我们使用 ``.sum().backward()``，因而损失传给最后一层输出脉冲的梯度全为1。


记录模块的输出梯度 :math:`\frac{\partial L}{\partial X}`
--------------------------------------------------------------------------------------
使用 :class:`spikingjelly.activation_based.monitor.GradInputMonitor` 可以轻松记录模块的输出梯度 :math:`\frac{\partial L}{\partial X}`。

让我们构建一个深度网络，调节替代函数的 ``alpha`` 并比较不同 ``alpha`` 下的梯度的幅值：

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import monitor, neuron, functional, layer, surrogate

    net = []
    for i in range(10):
        net.append(layer.Linear(8, 8))
        net.append(neuron.IFNode())

    net = nn.Sequential(*net)

    functional.set_step_mode(net, 'm')

    T = 4
    N = 1
    x_seq = torch.rand([T, N, 8])

    input_grad_monitor = monitor.GradInputMonitor(net, neuron.IFNode, function_on_grad_input=torch.norm)

    for alpha in [0.1, 0.5, 2, 4, 8]:
        for m in net.modules():
            if isinstance(m, surrogate.Sigmoid):
                m.alpha = alpha
        net(x_seq).sum().backward()
        print(f'alpha={alpha}, input_grad_monitor.records=\n{input_grad_monitor.records}\n')
        functional.reset_net(net)
        # zero grad
        for param in net.parameters():
            param.grad.zero_()

        input_grad_monitor.records.clear()


输出为：

.. code-block:: shell

    alpha=0.1, input_grad_monitor.records=
    [tensor(0.3868), tensor(0.0138), tensor(0.0003), tensor(9.1888e-06), tensor(1.0164e-07), tensor(1.9384e-09), tensor(4.0199e-11), tensor(8.6942e-13), tensor(1.3389e-14), tensor(2.7714e-16)]

    alpha=0.5, input_grad_monitor.records=
    [tensor(1.7575), tensor(0.2979), tensor(0.0344), tensor(0.0045), tensor(0.0002), tensor(1.5708e-05), tensor(1.6167e-06), tensor(1.6107e-07), tensor(1.1618e-08), tensor(1.1097e-09)]

    alpha=2, input_grad_monitor.records=
    [tensor(3.3033), tensor(1.2917), tensor(0.4673), tensor(0.1134), tensor(0.0238), tensor(0.0040), tensor(0.0008), tensor(0.0001), tensor(2.5466e-05), tensor(3.9537e-06)]

    alpha=4, input_grad_monitor.records=
    [tensor(3.5353), tensor(1.6377), tensor(0.7076), tensor(0.2143), tensor(0.0369), tensor(0.0069), tensor(0.0026), tensor(0.0006), tensor(0.0003), tensor(8.5736e-05)]

    alpha=8, input_grad_monitor.records=
    [tensor(4.3944), tensor(2.4396), tensor(0.8996), tensor(0.4376), tensor(0.0640), tensor(0.0122), tensor(0.0053), tensor(0.0016), tensor(0.0013), tensor(0.0005)]


降低内存占用
-------------------------------------------
如果我们需要记录大量数据，当被记录的数据是脉冲时，可以通过一些方法来降低内存占用。
为了能够进行浮点计算，尽管脉冲只含有0/1，但它们仍然被存储为浮点形式。因此，脉冲tensor的数据类型仍然为float32，或float16（如果使用混合精度训练）。

将float32转换为bool类型，可以降低内存占用。但由于C++中的bool类型实际上仍然是8比特，这种方式只能把内存降低为原来的1/4：

.. code-block:: python

    import torch

    def tensor_memory(x: torch.Tensor):
        return x.element_size() * x.numel()

    N = 1 << 10
    spike = torch.randint(0, 2, [N]).float()

    print('float32 size =', tensor_memory(spike))
    print('torch.bool size =', tensor_memory(spike.to(torch.bool)))

输出为：

.. code-block:: shell

    float32 size = 4096
    torch.bool size = 1024

在 :class:`spikingjelly.activation_based.tensor_cache` 中提供了将float32/float16类型的脉冲tensor压缩到uint8类型脉冲tensor的函数，其中uint8的tensor，每个\
元素使用8比特，保存8个脉冲，相当于是“真正的bool”类型。示例如下：

.. code-block:: python

    import torch

    def tensor_memory(x: torch.Tensor):
        return x.element_size() * x.numel()

    N = 1 << 10
    spike = torch.randint(0, 2, [N]).float()

    print('float32 size =', tensor_memory(spike))
    print('torch.bool size =', tensor_memory(spike.to(torch.bool)))

    from spikingjelly.activation_based import tensor_cache

    spike_b, s_dtype, s_shape, s_padding = tensor_cache.float_spike_to_bool(spike)


    print('bool size =', tensor_memory(spike_b))

    spike_recover = tensor_cache.bool_spike_to_float(spike_b, s_dtype, s_shape, s_padding)

    print('spike == spike_recover?', torch.equal(spike, spike_recover))

输出为：

.. code-block:: shell

    float32 size = 4096
    torch.bool size = 1024
    bool size = 128
    spike == spike_recover? True


与监视器结合使用，只需要将压缩函数增加到监视器的自定义函数中：

.. code-block:: python

    spike_seq_monitor = monitor.OutputMonitor(net, neuron.IFNode, function_on_output=tensor_cache.float_spike_to_bool)

在访问记录的数据时，再临时解压缩即可：


.. code-block:: python

    for item in spike_seq_monitor.records:
        print(tensor_cache.bool_spike_to_float(*item))

此外，对于稀疏的脉冲，还可以考虑使用 ``zlib`` 等库进行进一步的压缩。下面是对发放率为0.2的脉冲进行进一步压缩的例子：

.. code-block:: python

    import torch
    import zlib
    from spikingjelly.activation_based import tensor_cache

    def tensor_memory(x: torch.Tensor):
        return x.element_size() * x.numel()

    N = 1 << 20
    spike = (torch.rand([N]) > 0.8).float()

    spike_b, s_dtype, s_shape, s_padding = tensor_cache.float_spike_to_bool(spike)

    arr = spike_b.numpy()

    compressed_arr = zlib.compress(arr.tobytes())

    print("compressed ratio:", len(compressed_arr) / arr.nbytes * tensor_memory(spike_b) / tensor_memory(spike))

输出为：

.. code-block:: shell

    compressed ratio: 0.024264097213745117

如果想和监视器结合使用，仍然是放进自定义函数即可。完整的示例如下：

.. code-block:: python

    import torch
    import torch.nn as nn
    import zlib
    import numpy as np
    from spikingjelly.activation_based import monitor, neuron, functional, layer, tensor_cache

    def compress(spike: torch.Tensor):
        spike_b, s_dtype, s_shape, s_padding = tensor_cache.float_spike_to_bool(spike)
        spike_cb = zlib.compress(spike_b.cpu().numpy().tobytes())
        return spike_cb, s_dtype, s_shape, s_padding

    def decompress(spike_cb, s_dtype, s_shape, s_padding):
        spike_b = torch.frombuffer(zlib.decompress(spike_cb), dtype=torch.uint8)
        return tensor_cache.bool_spike_to_float(spike_b, s_dtype, s_shape, s_padding)

    net = nn.Sequential(
        layer.Linear(8, 4),
        neuron.IFNode(),
        layer.Linear(4, 2),
        neuron.IFNode()
    )

    for param in net.parameters():
        param.data.abs_()

    functional.set_step_mode(net, 'm')

    spike_seq_monitor = monitor.OutputMonitor(net, neuron.IFNode, function_on_output=compress)
    T = 4
    N = 1
    x_seq = torch.rand([T, N, 8])

    with torch.no_grad():
        net(x_seq)

    for item in spike_seq_monitor.records:
        print(decompress(*item))

需要注意的是，``zlib`` 的压缩只能在CPU上进行，如果原始数据在GPU上，则两边传输数据会大幅度拖慢运行速度。