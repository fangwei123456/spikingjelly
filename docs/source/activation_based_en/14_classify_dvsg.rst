Classify DVS128 Gesture
======================================

Author: `fangwei123456 <https://github.com/fangwei123456>`_

We have learned how to use neuromorphic datasets in last tutorial :doc:`Neuromorphic Datasets Processing <./13_neuromorphic_datasets>`.
Now, let us start to build a SNN to classify the DVS128 Gesture dataset. We will use the SNN from `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_ [#PLIF]_. We will use LIF
neurons and max pooling in this SNN.

The paper [#PLIF]_ uses an old version of SpikingJelly. The origin codes and logs are available at: `Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron <https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron>`_

In this tutorial, we will write codes by the new version of SpikingJelly, and our codes run faster than the origin codes.

Define The Network
-----------------------
The paper [#PLIF]_ use a general structure to build SNNs for different datasets, which is shown in the following figure:


.. image:: ../_static/tutorials/clock_driven/14_classify_dvsg/network.png
    :width: 100%

:math:`N_{conv}=1, N_{down}=5, N_{fc}=2` for the DVS128 Gesture dataset.

The detailed network structure is `{c128k3s1-BN-LIF-MPk2s2}*5-DP-FC512-LIF-DP-FC110-LIF-APk10s10}`, where `APk10s10` is
an additional voting layer.

The meanings of symbol are:

    `c128k3s1`: :code:`torch.nn.Conv2d(in_channels, out_channels=128, kernel_size=3, padding=1)`

    `BN`: :code:`torch.nn.BatchNorm2d(128)`

    `MPk2s2`: :code:`torch.nn.MaxPool2d(2, 2)`

    `DP`: :code:`spikingjelly.clock_driven.layer.Dropout(0.5)`

    `FC512`: :code:`torch.nn.Linear(in_features, out_features=512`

    `APk10s10`: :code:`torch.nn.AvgPool1d(2, 2)`

For simplicity, we firstly implement the network by the step-by-step mode:

.. code:: python

    class VotingLayer(nn.Module):
        def __init__(self, voter_num: int):
            super().__init__()
            self.voting = nn.AvgPool1d(voter_num, voter_num)
        def forward(self, x: torch.Tensor):
            # x.shape = [N, voter_num * C]
            # ret.shape = [N, C]
            return self.voting(x.unsqueeze(1)).squeeze(1)

    class PythonNet(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            conv = []
            conv.extend(PythonNet.conv3x3(2, channels))
            conv.append(nn.MaxPool2d(2, 2))
            for i in range(4):
                conv.extend(PythonNet.conv3x3(channels, channels))
                conv.append(nn.MaxPool2d(2, 2))
            self.conv = nn.Sequential(*conv)
            self.fc = nn.Sequential(
                nn.Flatten(),
                layer.Dropout(0.5),
                nn.Linear(channels * 4 * 4, channels * 2 * 2, bias=False),
                neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
                layer.Dropout(0.5),
                nn.Linear(channels * 2 * 2, 110, bias=False),
                neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
            )
            self.vote = VotingLayer(10)

        @staticmethod
        def conv3x3(in_channels: int, out_channels):
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
            ]

Forward and Loss
-----------------
We set simulating time-steps as ``T``, batch size as ``N``, then the frames ``x`` from ``DataLoader`` will have
``x.shape=[N, T, 2, 128, 128]``. We firstly convert ``x`` to ``shape=[T, N, 2, 128, 128]``.

Then, we send ``x[t]`` to the network, accumulate the output spikes and get the firing rate ``out_spikes / x.shape[0]``,
which is a tensor with ``shape=[N, 11]``.

.. code:: python

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        out_spikes = self.vote(self.fc(self.conv(x[0])))
        for t in range(1, x.shape[0]):
            out_spikes += self.vote(self.fc(self.conv(x[t])))
        return out_spikes / x.shape[0]

The loss is defined by the MSE between firing rate and the label in one hot format:

.. code:: python

    for frame, label in train_data_loader:
        optimizer.zero_grad()
        frame = frame.float().to(args.device)
        label = label.to(args.device)
        label_onehot = F.one_hot(label, 11).float()

        out_fr = net(frame)
        loss = F.mse_loss(out_fr, label_onehot)
        loss.backward()
        optimizer.step()

        functional.reset_net(net)

Accelerate by CUDA Neurons and Layer-by-layer
--------------------------------------------------
If the reader is not familiar with propagation pattern in SpikingJelly, please read the previous tutorials: :doc:`Propagation Pattern <./10_propagation_pattern>` and :doc:`Accelerate with CUDA-Enhanced Neuron and Layer-by-Layer Propagation <./11_cext_neuron_with_lbl>`.

We have built the net in the step-by-step model, whose codes are user-friendly but run slower. Now let us re-write the
net in the layer-by-layer mode with CUDA neurons:

.. code:: python

    import cupy

    class CextNet(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            conv = []
            conv.extend(CextNet.conv3x3(2, channels))
            conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            for i in range(4):
                conv.extend(CextNet.conv3x3(channels, channels))
                conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            self.conv = nn.Sequential(*conv)
            self.fc = nn.Sequential(
                nn.Flatten(2),
                layer.MultiStepDropout(0.5),
                layer.SeqToANNContainer(nn.Linear(channels * 4 * 4, channels * 2 * 2, bias=False)),
                neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'),
                layer.MultiStepDropout(0.5),
                layer.SeqToANNContainer(nn.Linear(channels * 2 * 2, 110, bias=False)),
                neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')
            )
            self.vote = VotingLayer(10)

        def forward(self, x: torch.Tensor):
            x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
            out_spikes = self.fc(self.conv(x))  # shape = [T, N, 110]
            return self.vote(out_spikes.mean(0))

        @staticmethod
        def conv3x3(in_channels: int, out_channels):
            return [
                layer.SeqToANNContainer(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                ),
                neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')
            ]

We can find that the two kind of models are similar. All stateless layers, e,g, ``Conv2d``, will be contained in
``layer.SeqToANNContainer``. The forward function is defined easily:

.. code:: python

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        out_spikes = self.fc(self.conv(x))  # shape = [T, N, 110]
        return self.vote(out_spikes.mean(0))

Code Details
-----------------
We add more arguments:

.. code:: python

    parser = argparse.ArgumentParser(description='Classify DVS128 Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-channels', default=128, type=int, help='channels of Conv2d in SNN')
    parser.add_argument('-data_dir', type=str, help='root dir of DVS128 Gesture dataset')
    parser.add_argument('-out_dir', type=str, help='root dir for saving logs and checkpoint')

    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use CUDA neuron and multi-step forward mode')


    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=32, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=32, type=int, help='T_max for CosineAnnealingLR')

Using automatic mixed precision (AMP) can accelerate training and reduce memory consumption:

.. code:: python

    if args.amp:
        with amp.autocast():
            out_fr = net(frame)
            loss = F.mse_loss(out_fr, label_onehot)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        out_fr = net(frame)
        loss = F.mse_loss(out_fr, label_onehot)
        loss.backward()
        optimizer.step()

We can also resume from a check point:

.. code:: python

    #...........
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
    # ...

    for epoch in range(start_epoch, args.epochs):
    # train...

    # test...

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        # ...

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

Star Training
----------------------
The complete codes are available at `spikingjelly.clock_driven.examples.classify_dvsg <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/classify_dvsg.py>`_.

We train the net in a linux server with `Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz` CPU and `GeForce RTX 2080 Ti` GPU. We
use almost the same hyper-parameters with those in the paper [#PLIF]_ with little difference, which is we use ``T=16``
because our `GeForce RTX 2080 Ti` only has 12GB memory, while the paper uses ``T=20``. Besides, we use AMP to accelerate,
which may cause slightly worse accuracy than the full precision training.

Let us try to train the step-by-step network:

.. code:: bash

    (test-env) root@de41f92009cf3011eb0ac59057a81652d2d0-fangw1714-0:/userhome/test# python -m spikingjelly.clock_driven.examples.classify_dvsg -data_dir /userhome/datasets/DVS128Gesture -out_dir ./logs -amp -opt Adam -device cuda:0 -lr_scheduler CosALR -T_max 64 -epochs 256
    Namespace(T=16, T_max=64, amp=True, b=16, cupy=False, channels=128, data_dir='/userhome/datasets/DVS128Gesture', device='cuda:0', epochs=256, gamma=0.1, j=4, lr=0.001, lr_scheduler='CosALR', momentum=0.9, opt='Adam', out_dir='./logs', resume=None, step_size=32)
    PythonNet(
      (conv): Sequential(
        (0): Conv2d(2, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LIFNode(
          v_threshold=1.0, v_reset=0.0, tau=2.0
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): LIFNode(
          v_threshold=1.0, v_reset=0.0, tau=2.0
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (10): LIFNode(
          v_threshold=1.0, v_reset=0.0, tau=2.0
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (12): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (13): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (14): LIFNode(
          v_threshold=1.0, v_reset=0.0, tau=2.0
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (16): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (17): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (18): LIFNode(
          v_threshold=1.0, v_reset=0.0, tau=2.0
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (19): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (fc): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Dropout(p=0.5)
        (2): Linear(in_features=2048, out_features=512, bias=False)
        (3): LIFNode(
          v_threshold=1.0, v_reset=0.0, tau=2.0
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (4): Dropout(p=0.5)
        (5): Linear(in_features=512, out_features=110, bias=False)
        (6): LIFNode(
          v_threshold=1.0, v_reset=0.0, tau=2.0
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
      )
      (vote): VotingLayer(
        (voting): AvgPool1d(kernel_size=(10,), stride=(10,), padding=(0,))
      )
    )
    The directory [/userhome/datasets/DVS128Gesture/frames_number_16_split_by_number] already exists.
    The directory [/userhome/datasets/DVS128Gesture/frames_number_16_split_by_number] already exists.
    Mkdir ./logs/T_16_b_16_c_128_Adam_lr_0.001_CosALR_64_amp.
    Namespace(T=16, T_max=64, amp=True, b=16, cupy=False, channels=128, data_dir='/userhome/datasets/DVS128Gesture', device='cuda:0', epochs=256, gamma=0.1, j=4, lr=0.001, lr_scheduler='CosALR', momentum=0.9, opt='Adam', out_dir='./logs', resume=None, step_size=32)
    epoch=0, train_loss=0.06680945929599134, train_acc=0.4032534246575342, test_loss=0.04891310722774102, test_acc=0.6180555555555556, max_test_acc=0.6180555555555556, total_time=27.759592294692993

It takes 27.76s to finish an epoch. We stop it and train the faster network:

.. code:: bash

    (test-env) root@de41f92009cf3011eb0ac59057a81652d2d0-fangw1714-0:/userhome/test# python -m spikingjelly.clock_driven.examples.classify_dvsg -data_dir /userhome/datasets/DVS128Gesture -out_dir ./logs -amp -opt Adam -device cuda:0 -lr_scheduler CosALR -T_max 64 -cupy -epochs 256
    Namespace(T=16, T_max=64, amp=True, b=16, cupy=True, channels=128, data_dir='/userhome/datasets/DVS128Gesture', device='cuda:0', epochs=256, gamma=0.1, j=4, lr=0.001, lr_scheduler='CosALR', momentum=0.9, opt='Adam', out_dir='./logs', resume=None, step_size=32)
    CextNet(
      (conv): Sequential(
        (0): SeqToANNContainer(
          (module): Sequential(
            (0): Conv2d(2, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): MultiStepLIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, surrogate_function=ATan, alpha=2.0 tau=2.0)
        (2): SeqToANNContainer(
          (module): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        (3): SeqToANNContainer(
          (module): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (4): MultiStepLIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, surrogate_function=ATan, alpha=2.0 tau=2.0)
        (5): SeqToANNContainer(
          (module): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        (6): SeqToANNContainer(
          (module): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (7): MultiStepLIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, surrogate_function=ATan, alpha=2.0 tau=2.0)
        (8): SeqToANNContainer(
          (module): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        (9): SeqToANNContainer(
          (module): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (10): MultiStepLIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, surrogate_function=ATan, alpha=2.0 tau=2.0)
        (11): SeqToANNContainer(
          (module): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        (12): SeqToANNContainer(
          (module): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (13): MultiStepLIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, surrogate_function=ATan, alpha=2.0 tau=2.0)
        (14): SeqToANNContainer(
          (module): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
      )
      (fc): Sequential(
        (0): Flatten(start_dim=2, end_dim=-1)
        (1): MultiStepDropout(p=0.5)
        (2): SeqToANNContainer(
          (module): Linear(in_features=2048, out_features=512, bias=False)
        )
        (3): MultiStepLIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, surrogate_function=ATan, alpha=2.0 tau=2.0)
        (4): MultiStepDropout(p=0.5)
        (5): SeqToANNContainer(
          (module): Linear(in_features=512, out_features=110, bias=False)
        )
        (6): MultiStepLIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, surrogate_function=ATan, alpha=2.0 tau=2.0)
      )
      (vote): VotingLayer(
        (voting): AvgPool1d(kernel_size=(10,), stride=(10,), padding=(0,))
      )
    )
    The directory [/userhome/datasets/DVS128Gesture/frames_number_16_split_by_number] already exists.
    The directory [/userhome/datasets/DVS128Gesture/frames_number_16_split_by_number] already exists.
    Mkdir ./logs/T_16_b_16_c_128_Adam_lr_0.001_CosALR_64_amp_cupy.
    Namespace(T=16, T_max=64, amp=True, b=16, cupy=True, channels=128, data_dir='/userhome/datasets/DVS128Gesture', device='cuda:0', epochs=256, gamma=0.1, j=4, lr=0.001, lr_scheduler='CosALR', momentum=0.9, opt='Adam', out_dir='./logs', resume=None, step_size=32)
    epoch=0, train_loss=0.06690179117738385, train_acc=0.4092465753424658, test_loss=0.049108295158172645, test_acc=0.6145833333333334, max_test_acc=0.6145833333333334, total_time=18.169376373291016

    ...

    Namespace(T=16, T_max=64, amp=True, b=16, cupy=True, channels=128, data_dir='/userhome/datasets/DVS128Gesture', device='cuda:0', epochs=256, gamma=0.1, j=4, lr=0.001, lr_scheduler='CosALR', momentum=0.9, opt='Adam', out_dir='./logs', resume=None, step_size=32)
    epoch=255, train_loss=0.00021228195577325645, train_acc=1.0, test_loss=0.008522209396485576, test_acc=0.9375, max_test_acc=0.9618055555555556, total_time=17.49005389213562

It takes 18.17s to finish an epoch, which is much faster. After 256 epochs, we will get the maximum accuracy 96.18%. The
logs curves during training are:

.. image:: ../_static/tutorials/clock_driven/14_classify_dvsg/train_loss.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/14_classify_dvsg/train_acc.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/14_classify_dvsg/test_loss.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/14_classify_dvsg/test_acc.*
    :width: 100%


.. [#PLIF] Fang, Wei, et al. "Incorporating learnable membrane time constant to enhance learning of spiking neural networks." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.