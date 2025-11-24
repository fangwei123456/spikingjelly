Train large-scale SNNs
======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

Usage of ``activation_based.model``
----------------------------------------------
:class:`spikingjelly.activation_based.model` has defined some classic networks, which we can use as we use :class:`torchvision.models`. \
For example, we can build the Spiking ResNet [#ResNet]_ :

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

The outputs are:

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

Spiking ResNet in SpikingJelly has the same network structure as that in ``torchvision``. Their ``state_dict().keys()`` are identical and we can load \
pre-trained weights by setting ``pretrained=True``:

.. code-block:: python

  s_resnet18 = spiking_resnet.spiking_resnet18(pretrained=True, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

Usage of ``activation_based.model.train_classify``
----------------------------------------------
:class:`spikingjelly.activation_based.model.train_classify` is modified by `torchvision 0.12 references <https://github.com/pytorch/vision/tree/release/0.12/references>`_. \
We can use this module to train easily.

:class:`spikingjelly.activation_based.model.train_classify.Trainer` provides a flexible method to train. Users can change its functions to implement the desirable behaviors without too much 
efforts. For example, :class:`spikingjelly.activation_based.model.train_classify.Trainer.set_optimizer` defines how to set the optimizer:

.. code-block:: python

    # spikingjelly.activation_based.model.train_classify
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
              raise RuntimeError(f"Invalid optimizer
               {args.opt}. Only SGD, RMSprop and AdamW are supported.")
          return optimizer

      def main(self, args):
        # ...
        optimizer = self.set_optimizer(args, parameters)
        # ...

If we want to add an optimizer, e.g., ``Adamax``, we can inherit the class and override this function:

.. code-block:: python

  class MyTrainer(train_classify.Trainer):
      def set_optimizer(self, args, parameters):
          opt_name = args.opt.lower()
          if opt_name.startswith("adamax"):
              optimizer = torch.optim.Adamax(parameters, lr=args.lr, weight_decay=args.weight_decay)
              return optimizer
          else:
              return super().set_optimizer(args, parameters)

:class:`Trainer.get_args_parser <spikingjelly.activation_based.model.train_classify.Trainer.get_args_parser>` defines the args for training:

.. code-block:: shell

  (pytorch-env) PS spikingjelly> python -m spikingjelly.activation_based.model.train_classify -h

  usage: train_classify.py [-h] [--data-path DATA_PATH] [--model MODEL] [--device DEVICE] [-b BATCH_SIZE] [--epochs N] [-j N] [--opt OPT] [--lr LR] [--momentum M] [--wd W] [--norm-weight-decay NORM_WEIGHT_DECAY] [--label-smoothing LABEL_SMOOTHING]
                          [--mixup-alpha MIXUP_ALPHA] [--cutmix-alpha CUTMIX_ALPHA] [--lr-scheduler LR_SCHEDULER] [--lr-warmup-epochs LR_WARMUP_EPOCHS] [--lr-warmup-method LR_WARMUP_METHOD] [--lr-warmup-decay LR_WARMUP_DECAY]                     
                          [--lr-step-size LR_STEP_SIZE] [--lr-gamma LR_GAMMA] [--output-dir OUTPUT_DIR] [--resume RESUME] [--start-epoch N] [--cache-dataset] [--sync-bn] [--test-only] [--pretrained] [--auto-augment AUTO_AUGMENT]                  
                          [--random-erase RANDOM_ERASE] [--world-size WORLD_SIZE] [--dist-url DIST_URL] [--model-ema] [--model-ema-steps MODEL_EMA_STEPS] [--model-ema-decay MODEL_EMA_DECAY] [--interpolation INTERPOLATION]                         
                          [--val-resize-size VAL_RESIZE_SIZE] [--val-crop-size VAL_CROP_SIZE] [--train-crop-size TRAIN_CROP_SIZE] [--clip-grad-norm CLIP_GRAD_NORM] [--ra-sampler] [--ra-reps RA_REPS] [--prototype] [--weights WEIGHTS] [--seed SEED]
                          [--print-logdir] [--clean] [--disable-pinmemory] [--disable-amp] [--local_rank LOCAL_RANK] [--disable-uda]                                                                                                                  
                                                                                                                                                                                                                                                     
  PyTorch Classification Training                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                      
  optional arguments:                                                                                                                                                                                                                                  
    -h, --help            show this help message and exit                                                                                                                                                                                              
    --data-path DATA_PATH                                                                                                                                                                                                                              
                          dataset path                                                                                                                                                                                                                 
    --model MODEL         model name                                                                                                                                                                                                                   
    --device DEVICE       device (Use cuda or cpu Default: cuda)                                                                                                                                                                                       
    -b BATCH_SIZE, --batch-size BATCH_SIZE                                                                                                                                                                                                             
                          images per gpu, the total batch size is $NGPU x batch_size                                                                                                                                                                   
    --epochs N            number of total epochs to run                                                                                                                                                                                                
    -j N, --workers N     number of data loading workers (default: 16)                                                                                                                                                                                 
    --opt OPT             optimizer                                                                                                                                                                                                                    
    --lr LR               initial learning rate
    --momentum M          momentum
    --wd W, --weight-decay W
                          weight decay (default: 0.)
    --norm-weight-decay NORM_WEIGHT_DECAY
                          weight decay for Normalization layers (default: None, same value as --wd)
    --label-smoothing LABEL_SMOOTHING
                          label smoothing (default: 0.1)
    --mixup-alpha MIXUP_ALPHA
                          mixup alpha (default: 0.2)
    --cutmix-alpha CUTMIX_ALPHA
                          cutmix alpha (default: 1.0)
    --lr-scheduler LR_SCHEDULER
                          the lr scheduler (default: cosa)
    --lr-warmup-epochs LR_WARMUP_EPOCHS
                          the number of epochs to warmup (default: 5)
    --lr-warmup-method LR_WARMUP_METHOD
                          the warmup method (default: linear)
    --lr-warmup-decay LR_WARMUP_DECAY
                          the decay for lr
    --lr-step-size LR_STEP_SIZE
                          decrease lr every step-size epochs
    --lr-gamma LR_GAMMA   decrease lr by a factor of lr-gamma
    --output-dir OUTPUT_DIR
                          path to save outputs
    --resume RESUME       path of checkpoint. If set to 'latest', it will try to load the latest checkpoint
    --start-epoch N       start epoch
    --cache-dataset       Cache the datasets for quicker initialization. It also serializes the transforms
    --sync-bn             Use sync batch norm
    --test-only           Only test the model
    --pretrained          Use pre-trained models from the modelzoo
    --auto-augment AUTO_AUGMENT
                          auto augment policy (default: ta_wide)
    --random-erase RANDOM_ERASE
                          random erasing probability (default: 0.1)
    --world-size WORLD_SIZE
                          number of distributed processes
    --dist-url DIST_URL   url used to set up distributed training
    --model-ema           enable tracking Exponential Moving Average of model parameters
    --model-ema-steps MODEL_EMA_STEPS
                          the number of iterations that controls how often to update the EMA model (default: 32)
    --model-ema-decay MODEL_EMA_DECAY
                          decay factor for Exponential Moving Average of model parameters (default: 0.99998)
    --interpolation INTERPOLATION
                          the interpolation method (default: bilinear)
    --val-resize-size VAL_RESIZE_SIZE
                          the resize size used for validation (default: 232)
    --val-crop-size VAL_CROP_SIZE
                          the central crop size used for validation (default: 224)
    --train-crop-size TRAIN_CROP_SIZE
                          the random crop size used for training (default: 176)
    --clip-grad-norm CLIP_GRAD_NORM
                          the maximum gradient norm (default None)
    --ra-sampler          whether to use Repeated Augmentation in training
    --ra-reps RA_REPS     number of repetitions for Repeated Augmentation (default: 4)
    --prototype           Use prototype model builders instead those from main area
    --weights WEIGHTS     the weights enum name to load
    --seed SEED           the random seed
    --print-logdir        print the dirs for tensorboard logs and pt files and exit
    --clean               delete the dirs for tensorboard logs and pt files
    --disable-pinmemory   not use pin memory in dataloader, which can help reduce memory consumption
    --disable-amp         not use automatic mixed precision training
    --local_rank LOCAL_RANK
                          args for DDP, which should not be set by user
    --disable-uda         not set 'torch.use_deterministic_algorithms(True)', which can avoid the error raised by some functions that do not have a deterministic implementation


If we want to add some args, we can also inherit and override it:

.. code-block:: python

  class MyTrainer(train_classify.Trainer):
      def get_args_parser(self, add_help=True):
          parser = super().get_args_parser()
          parser.add_argument('--do-something', type=str, help="do something")

We can modify most functions in :class:`Trainer <spikingjelly.activation_based.model.train_classify.Trainer>`.

We can use the following codes to train with ``Trainer`` or the user-defined trainer:

.. code-block:: python

    trainer = Trainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)

``Trainer`` will calculate ``Acc@1, Acc@5, loss`` on the training and test dataset, and save them by ``tensorboard``. The model weights of the latest epoch and the maximum test accuracy will also be saved.\ 
``Trainer`` also supports Distributed Data Parallel (DDP) training.

Training on ImageNet
----------------------------------------------
The default data loading function :class:`load_data <spikingjelly.activation_based.model.train_classify.Trainer.load_data>` will load the ImageNet [#ImageNet]_ dataset. With :class:`Trainer <spikingjelly.activation_based.model.train_classify.Trainer>` and :class:`spikingjelly.activation_based.model.spiking_resnet`, \
we can train large-scale SNNs easily. Here are the example codes:

.. code-block:: python

  # spikingjelly.activation_based.model.train_imagenet_example
  import torch
  from spikingjelly.activation_based import surrogate, neuron, functional
  from spikingjelly.activation_based.model import spiking_resnet, train_classify


  class SResNetTrainer(train_classify.Trainer):
      def preprocess_train_sample(self, args, x: torch.Tensor):
          # define how to process train sample before send it to model
          return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

      def preprocess_test_sample(self, args, x: torch.Tensor):
          # define how to process test sample before send it to model
          return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

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

The codes are saved in :class:`spikingjelly.activation_based.model.train_imagenet_example`. Training on a single GPU:

.. code-block:: shell

  python -m spikingjelly.activation_based.model.train_imagenet_example --T 4 --model spiking_resnet18 --data-path /datasets/ImageNet0_03125 --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90

Training with DDP on two GPUs:

.. code-block:: shell

  python -m torch.distributed.launch --nproc_per_node=2 -m spikingjelly.activation_based.model.train_imagenet_example --T 4 --model spiking_resnet18 --data-path /datasets/ImageNet0_03125 --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90

.. [#ResNet] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

.. [#ImageNet] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. IEEE, 2009.