Classify DVS Gesture
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

中文版： :doc:`../cn/classify_dvsg`

Translator: `Qiu Haonan <https://github.com/Maybe2022>`_, `fangwei123456 <https://github.com/fangwei123456>`_

In :doc:`./neuromorphic_datasets`, we have learned how to use neuromorphic datasets. Let's build a SNN to classify them.

Network Structure
-------------------------------------------
We will use the network defined in [#PLIF]_, which has the following structure:

.. image:: ../../_static/tutorials/classify_dvsg/network.png
    :width: 100%

All networks in [#PLIF]_ are defined in :class:`spikingjelly.activation_based.model.parametric_lif_net`. The network structure for DVS Gesture is:

.. code-block:: python

    # spikingjelly.activation_based.model.parametric_lif_net

    import torch
    import torch.nn as nn
    from .. import layer

    class DVSGestureNet(nn.Module):
        def __init__(self, channels=128, spiking_neuron: callable = None, *args, **kwargs):
            super().__init__()

            conv = []
            for i in range(5):
                if conv.__len__() == 0:
                    in_channels = 2
                else:
                    in_channels = channels

                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(*args, **kwargs))
                conv.append(layer.MaxPool2d(2, 2))


            self.conv_fc = nn.Sequential(
                *conv,

                layer.Flatten(),
                layer.Dropout(0.5),
                layer.Linear(channels * 4 * 4, 512),
                spiking_neuron(*args, **kwargs),

                layer.Dropout(0.5),
                layer.Linear(512, 110),
                spiking_neuron(*args, **kwargs),

                layer.VotingLayer(10)
            )

        def forward(self, x: torch.Tensor):
            return self.conv_fc(x)

Train
-------------------------------------------
The training loop follows :doc:`./conv_fashion_mnist`; this section shows the changes for DVS Gesture.

We use multi-step mode for faster training speed, and use automatic device-based neuron execution:

.. code-block:: python

    # spikingjelly.activation_based.examples.classify_dvsg

    import torch
    import sys
    import torch.nn.functional as F
    from torch.cuda import amp
    from spikingjelly.activation_based import functional, surrogate, neuron
    from spikingjelly.activation_based.model import parametric_lif_net
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    import time
    import os
    import argparse
    import datetime

    def main():
        # ...
        net = parametric_lif_net.DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

        functional.set_step_mode(net, 'm')
        # ...

Define the dataset:

.. code-block:: python

    # spikingjelly.activation_based.examples.classify_dvsg

    def main():
        # ...
        train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
        test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
        # ...

Note that dimension 0 is always the batch dimension for data packed by ``DataLoader``. So, the data we read from ``DataLoader`` has ``shape = [N, T, C, H, W]``. We need to reshape the data to ``shape = [T, N, C, H, W]`` for the multi-step mode:

.. code-block:: python

    # spikingjelly.activation_based.examples.classify_dvsg

    def main():
        # ...
        for epoch in range(start_epoch, args.epochs):
            for frame, label in train_data_loader:
                optimizer.zero_grad()
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                # ...

            with torch.no_grad():
                for frame, label in test_data_loader:
                    frame = frame.to(args.device)
                    frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                    # ...

        # ...

DVS Gesture has 11 classes:

.. code-block:: python

    # spikingjelly.activation_based.examples.classify_dvsg

    def main():
        # ...
        label_onehot = F.one_hot(label, 11).float()
        # ...

``DVSGestureNet`` outputs ``[T, N, 11]`` tensors. Its final ``VotingLayer`` averages each group of 10 neurons, so the per-step output is not binary. Averaging over time gives the classification scores.

.. code-block:: python

    # spikingjelly.activation_based.model.parametric_lif_net

    class DVSGestureNet(nn.Module):
        # ...
        def forward(self, x: torch.Tensor):
            return self.conv_fc(x)

Use the time-averaged classification scores to compute loss and accuracy:

.. code-block:: python

    # spikingjelly.activation_based.examples.classify_dvsg

    def main():
        # ...
        out_fr = net(frame).mean(0)
        loss = F.mse_loss(out_fr, label_onehot)
        # ...

Train the network:

.. code-block:: shell

    python -m spikingjelly.activation_based.examples.classify_dvsg -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir /datasets/DVSGesture/ -amp -opt adam -lr 0.001 -j 8

The following results come from an older version and have not been rerun on V2.

The outputs are:

.. code-block:: shell

    Namespace(T=16, device='cuda:0', b=16, epochs=64, j=8, data_dir='/datasets/DVSGesture/', out_dir='./logs', resume=None, amp=True, opt='adam', momentum=0.9, lr=0.001, channels=128)
    DVSGestureNet(
    (conv_fc): Sequential(...)
    ...
    )
    The directory [/datasets/DVSGesture/frames_number_16_split_by_number] already exists.
    The directory [/datasets/DVSGesture/frames_number_16_split_by_number] already exists.
    Mkdir ./logs/T16_b16_adam_lr0.001_c128_amp.
    Namespace(T=16, device='cuda:0', b=16, epochs=64, j=8, data_dir='/datasets/DVSGesture/', out_dir='./logs', resume=None, amp=True, opt='adam', momentum=0.9, lr=0.001, channels=128)
    ./logs/T16_b16_adam_lr0.001_c128_amp
    epoch = 0, train_loss = 0.0666, train_acc = 0.3964, test_loss = 0.0514, test_acc = 0.6042, max_test_acc = 0.6042
    train speed = 92.7646 images/s, test speed = 115.2935 images/s
    escape time = 2022-05-25 21:31:54

    Namespace(T=16, device='cuda:0', b=16, epochs=64, j=8, data_dir='/datasets/DVSGesture/', out_dir='./logs', resume=None, amp=True, opt='adam', momentum=0.9, lr=0.001, channels=128)
    ./logs/T16_b16_adam_lr0.001_c128_amp
    epoch = 1, train_loss = 0.0463, train_acc = 0.6036, test_loss = 0.0439, test_acc = 0.6319, max_test_acc = 0.6319
    train speed = 101.5938 images/s, test speed = 120.5184 images/s
    escape time = 2022-05-25 21:30:48

    ...

    Namespace(T=16, device='cuda:0', b=16, epochs=64, j=8, data_dir='/datasets/DVSGesture/', out_dir='./logs', resume=None, amp=True, opt='adam', momentum=0.9, lr=0.001, channels=128)
    ./logs/T16_b16_adam_lr0.001_c128_amp
    epoch = 63, train_loss = 0.0011, train_acc = 0.9991, test_loss = 0.0103, test_acc = 0.9375, max_test_acc = 0.9375
    train speed = 100.4324 images/s, test speed = 121.0402 images/s
    escape time = 2022-05-25 21:30:51

Finally, ``max_test_acc = 0.9375`` is achieved.

The following figure shows the accuracy curves during the training process:


.. image:: ../../_static/tutorials/classify_dvsg/dvsg_logs.*
    :width: 100%


.. [#PLIF] Fang, Wei, et al. "Incorporating learnable membrane time constant to enhance learning of spiking neural networks." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 2661-2671. `CVF <https://openaccess.thecvf.com/content/ICCV2021/html/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.html>`_
