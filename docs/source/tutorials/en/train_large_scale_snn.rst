Train large-scale SNNs
======================================

Author: `fangwei123456 <https://github.com/fangwei123456>`_

中文版： :doc:`../cn/train_large_scale_snn`

Usage of ``activation_based.model``
----------------------------------------------
Build a Spiking ResNet [#ResNet]_ with :mod:`spikingjelly.activation_based.model`,
then pass a four-step image sequence through it. The model factories work like
:mod:`torchvision.models`:

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

The model structure is abbreviated below. The output has shape ``[T, N, classes]``:
four time steps, one image, and 1000 class scores. ``reset_net`` clears neuron
state before you process an independent batch.

Output:

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

Spiking ResNet in SpikingJelly has the same network structure as that in ``torchvision``. Their ``state_dict().keys()`` are identical and we can load \
pre-trained weights by setting ``pretrained=True``:

.. code-block:: python

  from spikingjelly.activation_based import neuron, surrogate
  from spikingjelly.activation_based.model import spiking_resnet

  s_resnet18 = spiking_resnet.spiking_resnet18(pretrained=True, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

Usage of ``activation_based.examples.common.train_classify``
--------------------------------------------------------------------------------
:mod:`spikingjelly.activation_based.examples.common.train_classify` provides a
training loop adapted from `torchvision 0.12 references <https://github.com/pytorch/vision/tree/release/0.12/references>`_.
Subclass ``Trainer`` to change individual training steps. For example,
``Trainer.set_optimizer`` selects the optimizer. This excerpt shows its choices
and where the training loop calls it:

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

To add ``Adamax``, override ``set_optimizer`` and keep the inherited choices:

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

:class:`Trainer.get_args_parser <spikingjelly.activation_based.examples.common.train_classify.Trainer.get_args_parser>` defines the args for training:

.. code-block:: shell

  python -m spikingjelly.activation_based.examples.common.train_classify -h

To add a command-line argument, override ``get_args_parser``:

.. code-block:: python

  from spikingjelly.activation_based.examples.common import train_classify

  class MyTrainer(train_classify.Trainer):
      def get_args_parser(self, add_help=True):
          parser = super().get_args_parser(add_help=add_help)
          parser.add_argument('--do-something', type=str, help="do something")
          return parser

Parse the command-line arguments, then pass them to your trainer:

.. code-block:: python

    from spikingjelly.activation_based.examples.common import train_classify

    trainer = train_classify.Trainer()  # or MyTrainer() to use your own subclass
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)

``Trainer`` will calculate ``Acc@1, Acc@5, loss`` on the training and test dataset, and save them by ``tensorboard``. The model weights of the latest epoch and the maximum test accuracy will also be saved.\ 
``Trainer`` also supports Distributed Data Parallel (DDP) training.

Training on ImageNet
----------------------------------------------
Prepare ImageNet [#ImageNet]_ with ``train/<class>`` and ``val/<class>`` folders
under your data path. ``Trainer.load_ImageNet`` reads this layout. The subclass
below repeats each image over ``T`` time steps and averages the output logits
before computing the classification loss. The inherited loop resets neuron
state after each batch. Pass ``--T`` explicitly when launching training:

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
          return y.mean(0)  # average logits over time

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

Run the example in :mod:`spikingjelly.activation_based.examples.train_imagenet`. Training on a single GPU:

.. code-block:: shell

  python -m spikingjelly.activation_based.examples.train_imagenet --T 4 --model spiking_resnet18 --data-path /datasets/ImageNet0_03125 --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90

Training with DDP on two GPUs:

.. code-block:: shell

  torchrun --nproc_per_node=2 -m spikingjelly.activation_based.examples.train_imagenet --T 4 --model spiking_resnet18 --data-path /datasets/ImageNet0_03125 --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90

.. [#ResNet] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

.. [#ImageNet] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. IEEE, 2009.
