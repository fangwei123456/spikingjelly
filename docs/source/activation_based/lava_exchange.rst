转换到Lava框架以进行Loihi部署
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

感谢 `AllenYolk <https://github.com/AllenYolk>`_ 和 `banzhuangonglxh <https://github.com/banzhuangonglxh>`_ 对 `lava_exchange` 模块的贡献


Lava框架简介
-------------------------------------------
`Lava <https://github.com/lava-nc/lava>`_ 是Intel主导开发的神经形态计算框架，支持Intel Loihi芯片的部署。Lava 提供了一个名为 `Lava DL <https://github.com/lava-nc/lava-dl>`_ 的深度学习子包，可以搭建和训练深度SNN。

若想将SNN部署到Loihi芯片运行，则需要使用Lava框架。SpikingJelly中提供了对应的转换模块，可以将SpikingJelly中的模块或训练的网络转换到Lava框架，以便将网络部署到Loihi芯片运行。\
其基本流程为：

``SpikingJelly -> Lava DL -> Lava -> Loihi``

与Lava相关的模块，都定义在 :class:`spikingjelly.activation_based.lava_exchange` 中。

基本转换
-------------------------------------------

数据格式转换
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Lava DL默认数据格式为 ``shape = [N, *, T]``，其中 ``N`` 是batch维度，``T`` 是time-step维度。而SpikingJelly中的模块在多步模式(``step_mode = 'm'``)下，使用的数据格式是\
``shape = [T, N, *]``。因此，``lava_exchange`` 提供了两种格式的相互转换函数，:class:`TNX_to_NXT <spikingjelly.activation_based.lava_exchange.TNX_to_NXT>` 和\
:class:`NXT_to_TNX <spikingjelly.activation_based.lava_exchange.NXT_to_TNX>`。示例如下：


.. code-block:: python

    import torch
    from spikingjelly.activation_based import lava_exchange

    T = 6
    N = 4
    C = 2

    x_seq = torch.rand([T, N, C])

    x_seq_la = lava_exchange.TNX_to_NXT(x_seq)
    print(f'x_seq_la.shape=[N, C, T]={x_seq_la.shape}')

    x_seq_sj = lava_exchange.NXT_to_TNX(x_seq_la)
    print(f'x_seq_sj.shape=[T, N, C]={x_seq_sj.shape}')

输出为：

.. code-block:: shell

    x_seq_la.shape=[N, C, T]=torch.Size([4, 2, 6])
    x_seq_sj.shape=[T, N, C]=torch.Size([6, 4, 2])


神经元转换
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SpikingJelly中的神经元可以直接转换为Lava DL中的神经元。由于开发者精力有限，目前仅支持最常用的IF神经元和LIF神经元，其他神经元将在视用户需求添加。

使用 :class:`to_lava_neuron <spikingjelly.activation_based.lava_exchange.to_lava_neuron>` 进行转换，示例如下：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import lava_exchange, neuron

    if_sj = neuron.IFNode(v_threshold=1., v_reset=0., step_mode='m')
    if_la = lava_exchange.to_lava_neuron(if_sj)

    T = 8
    N = 2
    C = 1

    x_seq_sj = torch.rand([T, N, C])
    x_seq_la = lava_exchange.TNX_to_NXT(x_seq_sj)

    print('output of sj(reshaped to NXT):\n', lava_exchange.TNX_to_NXT(if_sj(x_seq_sj)))
    print('output of lava:\n', if_la(x_seq_la))

输出为：

.. code-block:: shell

    output of sj(reshaped to NXT):
    tensor([[[0., 0., 1., 0., 1., 0., 0., 0.]],

            [[0., 1., 0., 1., 0., 1., 0., 1.]]])
    output of lava:
    tensor([[[0., 0., 1., 0., 1., 0., 0., 0.]],

            [[0., 1., 0., 1., 0., 1., 0., 1.]]])

使用LIF神经元的示例如下：


.. code-block:: python

    import torch
    from spikingjelly.activation_based import lava_exchange, neuron

    if_sj = neuron.LIFNode(tau=50., decay_input=False, v_threshold=1., v_reset=0., step_mode='m')
    if_la = lava_exchange.to_lava_neuron(if_sj)

    T = 8
    N = 2
    C = 1

    x_seq_sj = torch.rand([T, N, C])
    x_seq_la = lava_exchange.TNX_to_NXT(x_seq_sj)

    print('output of sj:\n', lava_exchange.TNX_to_NXT(if_sj(x_seq_sj)))
    print('output of lava:\n', if_la(x_seq_la))

输出为：

.. code-block:: shell

    output of sj:
    tensor([[[0., 1., 0., 1., 0., 0., 1., 0.]],

            [[0., 0., 1., 0., 0., 1., 0., 1.]]])
    output of lava:
    tensor([[[0., 1., 0., 1., 0., 0., 1., 0.]],

            [[0., 0., 1., 0., 0., 1., 0., 1.]]])

突触转换
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
常用的卷积、全连接、池化层都支持转换。需要注意的是：

* 不支持bias
* Lava只支持求和池化，相当于是平均池化不做平均

示例如下：

.. code-block:: python

    from spikingjelly.activation_based import lava_exchange, layer

    conv = layer.Conv2d(3, 4, kernel_size=3, stride=1, bias=False)
    fc = layer.Linear(4, 2, bias=False)
    ap = layer.AvgPool2d(2, 2)

    conv_la = lava_exchange.conv2d_to_lava_synapse_conv(conv)
    fc_la = lava_exchange.linear_to_lava_synapse_dense(fc)
    sp_la = lava_exchange.avgpool2d_to_lava_synapse_pool(ap)

    print(f'conv_la={conv_la}')
    print(f'fc_la={fc_la}')
    print(f'sp_la={sp_la}')

输出为：

.. code-block:: shell

    WARNING:root:The lava slayer pool layer applies sum pooling, rather than average pooling. `avgpool2d_to_lava_synapse_pool` will return a sum pooling layer.
    conv_la=Conv(3, 4, kernel_size=(3, 3, 1), stride=(1, 1, 1), bias=False)
    fc_la=Dense(4, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    sp_la=Pool(1, 1, kernel_size=(2, 2, 1), stride=(2, 2, 1), bias=False)

Lava DL中几乎所有突触都是由 :class:`torch.nn.Conv3d` 实现的，因此打印出来会显示含有3个元素的tuple的 ``kernel_size`` 和 ``stride``。


BlockContainer
-------------------------------------------
使用Lava DL的一般流程是：

1. 使用Lava DL框架中的 `Blocks <https://lava-nc.org/lava-lib-dl/slayer/block/modules.html>`_ 搭建并训练网络
2. 将网络导出为hdf5文件
3. 使用Lava框架读取hdf5文件，以Lava的格式重建网络，并使用Loihi或CPU仿真的Loihi进行推理

具体信息，请参考 `Lava: Deep Learning <https://lava-nc.org/dl.html#deep-learning>`_。

`Blocks <https://lava-nc.org/lava-lib-dl/slayer/block/modules.html>`_ 可以被视作突触和神经元组成的集合。例如，:class:`lava.lib.dl.slayer.block.cuba.Conv` 实际上就是由卷积\
突触和CUBA神经元组成的。

需要注意的是，为了进行网络部署，``Blocks`` 中的突触权重和神经元的神经动态都进行了量化，因此 ``Blocks`` 并不是简单的\
``synapse + neuron``，而是 ``quantize(synapse) + quantize(neuron)``。

SpikingJelly提供了 :class:`BlockContainer <spikingjelly.activation_based.lava_exchange.BlockContainer>` ，主要特点如下：

* 支持替代梯度训练
* 对突触和神经动态进行了量化，与 :class:`lava.lib.dl.slayer.block` 具有完全相同的输出
* 支持直接转换为一个 :class:`lava.lib.dl.slayer.block`

目前 ``BlockContainer`` 仅支持 :class:`lava_exchange.CubaLIFNode <spikingjelly.activation_based.lava_exchange.CubaLIFNode>`，但也支持\
自动将输入的 :class:`IFNode <spikingjelly.activation_based.neuron.IFNode>` 和 :class:`LIFNode <spikingjelly.activation_based.neuron.LIFNode>` 转换为 ``CubaLIFNode``。\
例如：

.. code-block:: python

    from spikingjelly.activation_based import lava_exchange, layer, neuron

    fc_block_sj = lava_exchange.BlockContainer(
        synapse=layer.Linear(8, 1, bias=False),
        neu=neuron.IFNode(),
        step_mode='m'
    )

    print('fc_block_sj=\n', fc_block_sj)

    fc_block_la = fc_block_sj.to_lava_block()
    print('fc_block_la=\n', fc_block_la)

输出为：

.. code-block:: shell

    fc_block_sj=
    BlockContainer(
    (synapse): Linear(in_features=8, out_features=1, bias=False)
    (neuron): CubaLIFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, step_mode=m, backend=torch
        (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
    )
    )
    fc_block_la=
    Dense(
    (neuron): Neuron()
    (synapse): Dense(8, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    )


MNIST CSNN示例
-------------------------------------------
最后，让我们训练一个用于分类MNIST的卷积SNN，并转换到Lava DL框架。

网络定义如下：

.. code-block:: python

    class MNISTNet(nn.Module):
        def __init__(self, channels: int = 16):
            super().__init__()
            self.conv_fc = nn.Sequential(
                lava_exchange.BlockContainer(
                    nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False),
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
                ),

                lava_exchange.BlockContainer(
                    nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False),
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
                ),
                # 14 * 14

                lava_exchange.BlockContainer(
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
                ),

                lava_exchange.BlockContainer(
                    nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False),
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
                ),

                # 7 * 7

                lava_exchange.BlockContainer(
                    nn.Flatten(),
                    None
                ),
                lava_exchange.BlockContainer(
                    nn.Linear(channels * 7 * 7, 128, bias=False),
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
                ),

                lava_exchange.BlockContainer(
                    nn.Linear(128, 10, bias=False),
                    neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
                ),
            )

        def forward(self, x):
            return self.conv_fc(x)

我们为其增加一个转换到Lava DL网络的转换函数，在训练完成后可以使用：

.. code-block:: python

    def to_lava(self):
        ret = []

        for i in range(self.conv_fc.__len__()):
            m = self.conv_fc[i]
            if isinstance(m, lava_exchange.BlockContainer):
                ret.append(m.to_lava_block())

        return nn.Sequential(*ret)

接下来，对这个网络进行训练即可。训练流程与普通网络区别不大，只是在 ``lava_exchange.BlockContainer`` 内部，突触和神经动态都做了量化，这会导致正确率低于普通网络。部分训练代码如下：

.. code-block:: python

    encoder = encoding.PoissonEncoder(step_mode='m')
    # ...
    for img, label in train_data_loader:
        optimizer.zero_grad()
        img = img.to(args.device)
        label = label.to(args.device)
        img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)

        fr = net(encoder(img)).mean(0)
        loss = F.cross_entropy(fr, label)
        loss.backward()
        optimizer.step()
        # ...

当我们训练完成后，将网络转换到Lava DL，并检查测试集的正确率：

.. code-block:: python

    net_ladl = net.to_lava().to(args.device)
    net_ladl.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for img, label in test_data_loader:
            img = img.to(args.device)
            label = label.to(args.device)
            img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
            img = encoder(img)
            img = lava_exchange.TNX_to_NXT(img)
            fr = net_ladl(img).mean(-1)
            loss = F.cross_entropy(fr, label)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (fr.argmax(1) == label).float().sum().item()

    test_loss /= test_samples
    test_acc /= test_samples

    print('test acc[lava dl] =', test_acc)

最后，我们将Lava DL的网络导出hdf5，这样之后可以使用Lava框架加载，并在Loihi或者CPU模拟的Loihi上进行推理。具体流程请参考 `Network Exchange (NetX) Library <https://lava-nc.org/dl.html#network-exchange-netx-library>`_。

导出部分的代码如下：

.. code-block:: python

    def export_hdf5(net, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(net):
            handle = layer.create_group(f'{i}')
            b.export_hdf5(handle)

    export_hdf5(net_ladl, os.path.join(args.out_dir, 'net_la.net'))

完整的代码位于 :class:`spikingjelly.activation_based.examples.lava_mnist`，命令行参数如下：

.. code-block:: shell

    (lava-env) wfang@mlg-ThinkStation-P920:~/tempdir/w1$ python -m spikingjelly.activation_based.examples.lava_mnist -h
    usage: lava_mnist.py [-h] [-T T] [-b B] [-device DEVICE] [-data-dir DATA_DIR]
                        [-channels CHANNELS] [-epochs EPOCHS] [-lr LR] [-out-dir OUT_DIR]

    options:
    -h, --help          show this help message and exit
    -T T                simulating time-steps
    -b B                batch size
    -device DEVICE      device
    -data-dir DATA_DIR  root dir of the MNIST dataset
    -channels CHANNELS  channels of CSNN
    -epochs EPOCHS      training epochs
    -lr LR              learning rate
    -out-dir OUT_DIR    path for saving weights


在启动后，会首先训练网络，然后转换到Lava DL并进行推理，最后将hdf5格式的网络导出：

.. code-block:: shell

    (lava-env) wfang@mlg-ThinkStation-P920:~/tempdir/w1$ python -m spikingjelly.activation_based.examples.lava_mnist -T 32 -device cuda:0 -b 128 -epochs 16 -data-dir /datasets/MNIST/ -lr 0.1 -channels 16
    Namespace(T=32, b=128, device='cuda:0', data_dir='/datasets/MNIST/', channels=16, epochs=16, lr=0.1, out_dir='./')
    Namespace(T=32, b=128, device='cuda:0', data_dir='/datasets/MNIST/', channels=16, epochs=16, lr=0.1, out_dir='./')
    epoch = 0, train_loss = 1.7607, train_acc = 0.7245, test_loss = 1.5243, test_acc = 0.9443, max_test_acc = 0.9443

    # ...

    Namespace(T=32, b=128, device='cuda:0', data_dir='/datasets/MNIST/', channels=16, epochs=16, lr=0.1, out_dir='./')
    epoch = 15, train_loss = 1.4743, train_acc = 0.9881, test_loss = 1.4760, test_acc = 0.9855, max_test_acc = 0.9860
    finish training
    test acc[sj] = 0.9855
    test acc[lava dl] = 0.9863
    save net.state_dict() to ./net.pt
    save net_ladl.state_dict() to ./net_ladl.pt
    export net_ladl to ./net_la.net

