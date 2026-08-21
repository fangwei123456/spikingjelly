:orphan:

Train Large-Scale SNN
======================================

Author: `fangwei123456 <https://github.com/fangwei123456>`_

Use networks from :class:`spikingjelly.activation_based.model`
------------------------------------------------------------------
:class:`spikingjelly.activation_based.model` provides some classic networks. We use :class:`spikingjelly.activation_based.model.spiking_resnet` as the example to show how to use them.

Most of networks in :class:`spikingjelly.activation_based.model` have two version: single-step and multi-step. We can create
a single-step Spiking ResNet-18 [#ResNet]_ like this:

.. code:: python

    import torch
    import torch.nn.functional as F
    from spikingjelly.activation_based import neuron, surrogate, functional
    from spikingjelly.activation_based.model import spiking_resnet

    net = spiking_resnet.spiking_resnet18(pretrained=False, progress=True, single_step_neuron=neuron.IFNode, v_threshold=1., surrogate_function=surrogate.ATan())
    print(net)

As the arguments in ``spiking_resnet18(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs)``, ``single_step_neuron`` is the single-step neuron, and ``**kwargs`` are args for the neuron. If we set ``pretrained=True``,
the Spiking ResNet-18 will load pre-trained parameters from ResNet-18 (ANN, rather than SNN). The outputs are:

.. code:: shell

    SpikingResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (sn1): IFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False
        (surrogate_function): ATan(alpha=2.0, spiking=True)
      )
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): IFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )

The inputs of single-step network do not contain the time dimension. We need to give inputs at each time-step to the network:

.. code:: python

    net = spiking_resnet.spiking_resnet18(pretrained=False, progress=True, single_step_neuron=neuron.IFNode, v_threshold=1., surrogate_function=surrogate.ATan())
    T = 4
    N = 2
    x = torch.rand([T, N, 3, 224, 224])
    fr = 0.
    with torch.no_grad():
        for t in range(T):
            fr += net(x[t])
        fr /= T
    print('firing rate =', fr)


To build a multi-step network, we should use :class:`spikingjelly.activation_based.model.spiking_resnet.multi_step_spiking_resnet18`,
rather than :class:`spikingjelly.activation_based.model.spiking_resnet.spiking_resnet18`, and use the multi-step neuron:

.. code:: python

    net_ms = spiking_resnet.multi_step_spiking_resnet18(pretrained=False, progress=True, multi_step_neuron=neuron.MultiStepIFNode, v_threshold=1., surrogate_function=surrogate.ATan(), backend='torch')
    print(net_ms)

The outputs are:

.. code:: shell

    MultiStepSpikingResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (sn1): MultiStepIFNode(
        v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
        (surrogate_function): ATan(alpha=2.0, spiking=True)
      )
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): MultiStepBasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
        )
        (1): MultiStepBasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
        )
      )
      (layer2): Sequential(
        (0): MultiStepBasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): MultiStepBasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
        )
      )
      (layer3): Sequential(
        (0): MultiStepBasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): MultiStepBasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
        )
      )
      (layer4): Sequential(
        (0): MultiStepBasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): MultiStepBasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn1): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sn2): MultiStepIFNode(
            v_threshold=1.0, v_reset=0.0, detach_reset=False, backend=cupy
            (surrogate_function): ATan(alpha=2.0, spiking=True)
          )
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )


The inputs for multi-step network should have the time dimension:

.. code:: python

    net = spiking_resnet.spiking_resnet18(pretrained=False, progress=True, single_step_neuron=neuron.IFNode, v_threshold=1.,
                                          surrogate_function=surrogate.ATan())
    T = 4
    N = 2
    x = torch.rand([T, N, 3, 224, 224])
    fr = 0.
    with torch.no_grad():
        for t in range(T):
            fr += net(x[t])
        fr /= T

    net_ms = spiking_resnet.multi_step_spiking_resnet18(pretrained=False, progress=True, multi_step_neuron=neuron.MultiStepIFNode, v_threshold=1., surrogate_function=surrogate.ATan(), backend='torch')

    net_ms.load_state_dict(net.state_dict())
    with torch.no_grad():
        print('mse of single/multi step network outputs', F.mse_loss(net_ms(x).mean(0), fr))

However, this network also supports for inputs without time dimension, as long as we set ``T`` when building or after
building the network.

Setting ``T`` when building:

.. code:: python

    net_ms = spiking_resnet.multi_step_spiking_resnet18(pretrained=False, progress=True, T=4, multi_step_neuron=neuron.MultiStepIFNode, v_threshold=1., surrogate_function=surrogate.ATan(), backend='torch')

Or setting ``T`` after building:

.. code:: python

    net_ms = spiking_resnet.multi_step_spiking_resnet18(pretrained=False, progress=True, multi_step_neuron=neuron.MultiStepIFNode, v_threshold=1., surrogate_function=surrogate.ATan(), backend='torch')
    net_ms.T = 4

The network will repeat inputs in time dimension, which is same with we do it manually:

.. code:: python

    net_ms = spiking_resnet.multi_step_spiking_resnet18(pretrained=False, progress=True, multi_step_neuron=neuron.MultiStepIFNode, v_threshold=1., surrogate_function=surrogate.ATan(), backend='torch')
    T = 4
    N = 2

    with torch.no_grad():
        x = torch.rand([N, 3, 224, 224])
        y1 = net_ms(x.unsqueeze(0).repeat(T, 1, 1, 1, 1))
        functional.reset_net(net_ms)
        net_ms.T = T
        y2 = net_ms(x)
        print(F.mse_loss(y1, y2))

The outputs are:

.. code:: shell

    tensor(0.)

However, it is more efficient to let network to repeat. Refer to :doc:`Clock driven: Use convolutional SNN to identify Fashion-MNIST <./4_conv_fashion_mnist>` for the reason.

Training on ImageNet
---------------------------------------

The legacy training API has been removed. See the current
:doc:`large-scale SNN training tutorial <../../tutorials/en/train_large_scale_snn>`.

.. [#ResNet] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
