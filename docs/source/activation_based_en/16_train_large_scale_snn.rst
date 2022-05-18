Train Large-Scale SNN
======================================

Author: `fangwei123456 <https://github.com/fangwei123456>`_

Use networks from :class:`spikingjelly.clock_driven.model`
----------------------------------------------
:class:`spikingjelly.clock_driven.model` provides some classic networks. We use :class:`spikingjelly.clock_driven.model.spiking_resnet` as the example to show how to use them.

Most of networks in :class:`spikingjelly.clock_driven.model` have two version: single-step and multi-step. We can create
a single-step Spiking ResNet-18 [#ResNet]_ like this:

.. code:: python

    import torch
    import torch.nn.functional as F
    from spikingjelly.clock_driven import neuron, surrogate, functional
    from spikingjelly.clock_driven.model import spiking_resnet

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


To build a multi-step network, we should use :class:`spikingjelly.clock_driven.model.spiking_resnet.multi_step_spiking_resnet18`,
rather than :class:`spikingjelly.clock_driven.model.spiking_resnet.spiking_resnet18`, and use the multi-step neuron:

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
ImageNet [#ImageNet]_ is a popular baseline dataset for computer version, which is challenging for SNNs. SpikingJelly provides a code
example to train on ImageNet, which is available at `spikingjelly.clock_driven.model.train_imagenet <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/model/train_imagenet.py>`_ .The example is written
by referring `torchvision <https://github.com/pytorch/vision/blob/main/references/classification/train.py>`_ .We can use
it to train our network on ImageNet after we build the network, loss and how to calculate accuracy. Here is an example:

.. code:: python

    import torch
    import torch.nn.functional as F
    from spikingjelly.clock_driven.model import train_imagenet, spiking_resnet, train_classify
    from spikingjelly.clock_driven import neuron, surrogate

    def ce_loss(x_seq: torch.Tensor, label: torch.Tensor):
        # x_seq.shape = [T, N, C]
        return F.cross_entropy(input=x_seq.mean(0), target=label)

    def cal_acc1_acc5(output, target):
        return train_classify.default_cal_acc1_acc5(output.mean(0), target)


    if __name__ == '__main__':
        net = spiking_resnet.multi_step_spiking_resnet18(T=4, multi_step_neuron=neuron.MultiStepIFNode, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')
        args = train_imagenet.parse_args()
        train_imagenet.main(model=net, criterion=ce_loss, args=args, cal_acc1_acc5=cal_acc1_acc5)

Let us save these codes as `resnet18_imagenet.py`. The running arguments are:

.. code:: shell

    (pytorch-env) wfang@onebrain-dgx-a100-01:~/ssd/temp_dir$ python resnet18_imagenet.py -h

                                [--step-gamma STEP_GAMMA] [--cosa-tmax COSA_TMAX] [--momentum M] [--wd W] [--output-dir OUTPUT_DIR] [--resume RESUME] [--start-epoch N] [--cache-dataset]
                                [--sync-bn] [--amp] [--world-size WORLD_SIZE] [--dist-url DIST_URL] [--tb] [--T T] [--local_rank LOCAL_RANK]

    PyTorch Classification Training

    optional arguments:
      -h, --help            show this help message and exit
      --data-path DATA_PATH
                            dataset
      --device DEVICE       device
      -b BATCH_SIZE, --batch-size BATCH_SIZE
      --epochs N            number of total epochs to run
      -j N, --workers N     number of data loading workers (default: 16)
      --lr LR               initial learning rate
      --opt OPT             optimizer (sgd or adam)
      --lrs LRS             lr schedule (cosa(CosineAnnealingLR), step(StepLR)) or None
      --step-size STEP_SIZE
                            step_size for StepLR
      --step-gamma STEP_GAMMA
                            gamma for StepLR
      --cosa-tmax COSA_TMAX
                            T_max for CosineAnnealingLR. If none, it will be set to epochs
      --momentum M          Momentum for SGD
      --wd W, --weight-decay W
                            weight decay (default: 0)
      --output-dir OUTPUT_DIR
                            path where to save
      --resume RESUME       resume from checkpoint
      --start-epoch N       start epoch
      --cache-dataset       Cache the datasets for quicker initialization. It also serializes the transforms
      --sync-bn             Use sync batch norm
      --amp                 Use AMP training
      --world-size WORLD_SIZE
                            number of distributed processes
      --dist-url DIST_URL   url used to set up distributed training
      --tb                  Use TensorBoard to record logs
      --T T                 simulation steps
      --local_rank LOCAL_RANK


Training on a GPU:

.. code:: shell

    python resnet18_imagenet.py --data-path /raid/wfang/datasets/ImageNet --lr 0.1 --opt sgd --lrs cosa --amp --tb --device cuda:7

Training on multi-gpu:

.. code:: shell

    python -m torch.distributed.launch --nproc_per_node=8 resnet18_imagenet.py --data-path /raid/wfang/datasets/ImageNet --lr 0.1 --opt sgd --lrs cosa --amp --tb

.. [#ResNet] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

.. [#ImageNet] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. IEEE, 2009.