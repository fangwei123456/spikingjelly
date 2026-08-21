训练大规模SNN
======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

English version: :doc:`../en/train_large_scale_snn`

使用 ``activation_based.model``
----------------------------------------------
在 :class:`spikingjelly.activation_based.model` 中定义了一些经典网络模型，可以直接拿来使用，使用方法与 \
:class:`torchvision.models` 类似。以Spiking ResNet [#ResNet]_ 为例：

.. code-block:: python

  import torch
  import torch.nn as nn
  from spikingjelly.activation_based import surrogate, neuron, functional
  from spikingjelly.activation_based.model import spiking_resnet

  s_resnet18 = spiking_resnet.spiking_resnet18(pretrained=False, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

  print(f's_resnet18={s_resnet18}')

  with torch.no_grad():
      T = 4
      N = 1
      x_seq = torch.rand([T, N, 3, 224, 224])
      functional.set_step_mode(s_resnet18, 'm')
      y_seq = s_resnet18(x_seq)
      print(f'y_seq.shape={y_seq.shape}')
      functional.reset_net(s_resnet18)

输出为：

.. code-block:: shell

  s_resnet18=SpikingResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False, step_mode=s)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
    (sn1): IFNode(
      v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
      (surrogate_function): ATan(alpha=2.0, spiking=True)
    )
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False, step_mode=s)
    (layer1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn1): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn2): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn1): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn2): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
      )
    )
    (layer2): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, step_mode=s)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn1): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn2): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False, step_mode=s)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn1): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn2): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
      )
    )
    (layer3): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, step_mode=s)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn1): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn2): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False, step_mode=s)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn1): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn2): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
      )
    )
    (layer4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, step_mode=s)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn1): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn2): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False, step_mode=s)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn1): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, step_mode=s)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
        (sn2): IFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s, backend=torch
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
      )
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1), step_mode=s)
    (fc): Linear(in_features=512, out_features=1000, bias=True)
  )
  y_seq.shape=torch.Size([4, 1, 1000])


SpikingJelly按照 ``torchvision`` 中的ResNet结构搭建的Spiking ResNet，保持了 ``state_dict().keys()`` 相同，\
因此支持直接加载预训练权重，设置 ``pretrained=True`` 即可：

.. code-block:: python

  s_resnet18 = spiking_resnet.spiking_resnet18(pretrained=True, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

使用 ``activation_based.examples.common.train_classify``
----------------------------------------------------------------

:class:`spikingjelly.activation_based.examples.common.train_classify` 是根据 `torchvision 0.12 references <https://github.com/pytorch/vision/tree/release/0.12/references>`_ \
的分类代码进行改动而来，使用这个模块可以很方便的进行训练。

:class:`spikingjelly.activation_based.examples.common.train_classify.Trainer` 提供了较为灵活的训练方式，预留了一些接口给用户改动。\
例如， :class:`spikingjelly.activation_based.examples.common.train_classify.Trainer.set_optimizer` 定义了如何设置优化器，默认为：

.. code-block:: python

    # spikingjelly.activation_based.examples.common.train_classify
    class Trainer:
      # ...
      def set_optimizer(self, args, parameters):
          opt_name = args.opt.lower()
          if opt_name.startswith("sgd"):
              optimizer = torch.optim.SGD(
                  parameters,
                  lr=args.lr,
                  momentum=args.momentum,
                  weight_decay=args.weight_decay,
                  nesterov="nesterov" in opt_name,
              )
          elif opt_name == "rmsprop":
              optimizer = torch.optim.RMSprop(
                  parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
              )
          elif opt_name == "adamw":
              optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
          else:
              raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")
          return optimizer

      def main(self, args):
        # ...
        optimizer = self.set_optimizer(args, parameters)
        # ...

如果我们增加一个优化器，例如 ``Adamax`` ，只需要继承并重写此方法，例如：

.. code-block:: python

  class MyTrainer(train_classify.Trainer):
      def set_optimizer(self, args, parameters):
          opt_name = args.opt.lower()
          if opt_name.startswith("adamax"):
              optimizer = torch.optim.Adamax(parameters, lr=args.lr, weight_decay=args.weight_decay)
              return optimizer
          else:
              return super().set_optimizer(args, parameters)

默认的 :class:`Trainer.get_args_parser <spikingjelly.activation_based.examples.common.train_classify.Trainer.get_args_parser>` 已经包含了较多的参数设置：


.. code-block:: shell

  (pytorch-env) PS spikingjelly> python -m spikingjelly.activation_based.examples.common.train_classify -h

如果想增加参数，仍然可以通过继承的方式实现：

.. code-block:: python

  class MyTrainer(train_classify.Trainer):
      def get_args_parser(self, add_help=True):
          parser = super().get_args_parser()
          parser.add_argument('--do-something', type=str, help="do something")

:class:`Trainer <spikingjelly.activation_based.examples.common.train_classify.Trainer>` 的许多其他函数都可以进行补充修改或覆盖，方法类似，不再赘述。

对于 ``Trainer`` 及用户自己继承实现的子类，可以通过如下方式调用并进行训练：

.. code-block:: python

    trainer = Trainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)

``Trainer`` 在训练中会自动计算训练集、测试集的 ``Acc@1, Acc@5, loss`` 并使用 ``tensorboard`` 保存为日志文件，此外训练过程中的最新一个epoch的模型以及测试集性能最高的模型\
也会被保存下来。 ``Trainer`` 支持Distributed Data Parallel训练。

在ImageNet上训练
----------------------------------------------
 ``Trainer`` 默认的数据加载函数 :class:`load_ImageNet <spikingjelly.activation_based.examples.common.train_classify.Trainer.load_ImageNet>` 加载 ImageNet [#ImageNet]_ 数据集。\
 结合 :class:`Trainer <spikingjelly.activation_based.examples.common.train_classify.Trainer>` 和 :class:`spikingjelly.activation_based.model.spiking_resnet`，我们可以轻松训练\
 大型深度SNN，示例代码如下：

.. code-block:: python

  # spikingjelly.activation_based.examples.train_imagenet
  import torch
  from spikingjelly.activation_based import surrogate, neuron, functional
  from spikingjelly.activation_based.model import spiking_resnet
  from spikingjelly.activation_based.examples.common import train_classify


  class SResNetTrainer(train_classify.Trainer):
      def preprocess_train_sample(self, args, x: torch.Tensor):
          # define how to process train sample before send it to model
          return x.unsqueeze(0).expand(args.T, -1, -1, -1, -1)  # [N, C, H, W] -> [T, N, C, H, W]

      def preprocess_test_sample(self, args, x: torch.Tensor):
          # define how to process test sample before send it to model
          return x.unsqueeze(0).expand(args.T, -1, -1, -1, -1)  # [N, C, H, W] -> [T, N, C, H, W]

      def process_model_output(self, args, y: torch.Tensor):
          return y.mean(0)  # return firing rate

      def get_args_parser(self, add_help=True):
          parser = super().get_args_parser()
          parser.add_argument('--T', type=int, help="total time-steps")
          parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")
          return parser

      def get_tb_logdir_name(self, args):
          return super().get_tb_logdir_name(args) + f'_T{args.T}'

      def load_model(self, args, num_classes):
          if args.model in spiking_resnet.__all__:
              model = spiking_resnet.__dict__[args.model](pretrained=args.pretrained, spiking_neuron=neuron.IFNode,
                                                          surrogate_function=surrogate.ATan(), detach_reset=True)
              functional.set_step_mode(model, step_mode='m')
              if args.cupy:
                  functional.set_backend(model, 'cupy', neuron.IFNode)

              return model
          else:
              raise ValueError(f"args.model should be one of {spiking_resnet.__all__}")


  if __name__ == "__main__":
      trainer = SResNetTrainer()
      args = trainer.get_args_parser().parse_args()
      trainer.main(args)

代码位于 :class:`spikingjelly.activation_based.examples.train_imagenet`，可以直接运行。
在单卡上进行训练：

.. code-block:: shell

  python -m spikingjelly.activation_based.examples.train_imagenet --T 4 --model spiking_resnet18 --data-path /datasets/ImageNet0_03125 --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90

在多卡上进行训练：

.. code-block:: shell

  torchrun --nproc_per_node=2 -m spikingjelly.activation_based.examples.train_imagenet --T 4 --model spiking_resnet18 --data-path /datasets/ImageNet0_03125 --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90

.. [#ResNet] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

.. [#ImageNet] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. IEEE, 2009.
