训练大规模SNN
======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

English version: :doc:`../en/train_large_scale_snn`

使用 ``activation_based.model``
----------------------------------------------
先用 :mod:`spikingjelly.activation_based.model` 创建 Spiking ResNet [#ResNet]_，
再输入一个包含四个时间步的图像序列。模型工厂的用法与 :mod:`torchvision.models` 类似：

.. code-block:: python

  import torch
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

模型结构的输出省略了部分层。输出形状为 ``[T, N, classes]``，即四个时间步、
一张图像和 1000 个类别分数。处理下一个独立批次前，用 ``reset_net`` 清空神经元状态。

输出：

.. code-block:: shell

  s_resnet18=SpikingResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False, step_mode=s)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, step_mode=s)
    (sn1): IFNode(
      v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode=s
      (surrogate_function): ATan(alpha=2.0, spiking=True)
    )
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False, step_mode=s)
    ...
  )
  y_seq.shape=torch.Size([4, 1, 1000])

SpikingJelly按照 ``torchvision`` 中的ResNet结构搭建的Spiking ResNet，保持了 ``state_dict().keys()`` 相同，\
因此支持直接加载预训练权重，设置 ``pretrained=True`` 即可：

.. code-block:: python

  from spikingjelly.activation_based import neuron, surrogate
  from spikingjelly.activation_based.model import spiking_resnet

  s_resnet18 = spiking_resnet.spiking_resnet18(pretrained=True, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

使用 ``activation_based.examples.common.train_classify``
----------------------------------------------------------------

:mod:`spikingjelly.activation_based.examples.common.train_classify` 提供训练循环，
改编自 `torchvision 0.12 references <https://github.com/pytorch/vision/tree/release/0.12/references>`_。
你可以继承 ``Trainer``，修改其中某个训练步骤。例如，``Trainer.set_optimizer``
负责选择优化器。下面摘录了它支持的选择，以及训练循环调用它的位置：

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

要添加 ``Adamax``，重写 ``set_optimizer``，并保留继承的优化器选择：

.. code-block:: python

  import torch
  from spikingjelly.activation_based.examples.common import train_classify

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

  python -m spikingjelly.activation_based.examples.common.train_classify -h

如果想增加参数，仍然可以通过继承的方式实现：

.. code-block:: python

  from spikingjelly.activation_based.examples.common import train_classify

  class MyTrainer(train_classify.Trainer):
      def get_args_parser(self, add_help=True):
          parser = super().get_args_parser(add_help=add_help)
          parser.add_argument('--do-something', type=str, help="do something")
          return parser

解析命令行参数后，将它们传给你的训练器：

.. code-block:: python

    from spikingjelly.activation_based.examples.common import train_classify

    trainer = train_classify.Trainer()  # 或 MyTrainer() 使用自定义子类
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)

``Trainer`` 在训练中会自动计算训练集、测试集的 ``Acc@1, Acc@5, loss`` 并使用 ``tensorboard`` 保存为日志文件，此外训练过程中的最新一个epoch的模型以及测试集性能最高的模型\
也会被保存下来。 ``Trainer`` 支持Distributed Data Parallel训练。

在ImageNet上训练
----------------------------------------------
准备 ImageNet [#ImageNet]_，在数据目录下按 ``train/<class>`` 和 ``val/<class>``
组织图像。``Trainer.load_ImageNet`` 会读取这个目录结构。下面的子类把每张图像
重复 ``T`` 个时间步，并在计算分类损失前对输出分数取时间平均。继承的训练循环
会在每个批次结束后重置神经元状态。启动训练时，需要显式传入 ``--T``：

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
          return y.mean(0)  # 对时间维取平均的 logits

      def get_args_parser(self, add_help=True):
          parser = super().get_args_parser(add_help=add_help)
          parser.add_argument('--T', type=int, help="total time-steps")
          return parser

      def get_tb_logdir_name(self, args):
          return super().get_tb_logdir_name(args) + f'_T{args.T}'

      def load_model(self, args, num_classes):
          if args.model in spiking_resnet.__all__:
              model = spiking_resnet.__dict__[args.model](pretrained=args.pretrained, spiking_neuron=neuron.IFNode,
                                                          surrogate_function=surrogate.ATan(), detach_reset=True)
              functional.set_step_mode(model, step_mode='m')
              return model
          else:
              raise ValueError(f"args.model should be one of {spiking_resnet.__all__}")


  if __name__ == "__main__":
      trainer = SResNetTrainer()
      args = trainer.get_args_parser().parse_args()
      trainer.main(args)

代码位于 :mod:`spikingjelly.activation_based.examples.train_imagenet`，可以直接运行。
在单卡上进行训练：

.. code-block:: shell

  python -m spikingjelly.activation_based.examples.train_imagenet --T 4 --model spiking_resnet18 --data-path /datasets/ImageNet0_03125 --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90

在多卡上进行训练：

.. code-block:: shell

  torchrun --nproc_per_node=2 -m spikingjelly.activation_based.examples.train_imagenet --T 4 --model spiking_resnet18 --data-path /datasets/ImageNet0_03125 --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90

.. [#ResNet] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

.. [#ImageNet] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. IEEE, 2009.
