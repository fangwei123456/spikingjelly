import torch
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_resnet, train_classify


class SResNetTrainer(train_classify.Trainer):
    def cal_acc1_acc5(self, output, target):
        # output.shape = [T, N, C]
        return super().cal_acc1_acc5(output.mean(0), target)

    def preprocess_train_sample(self, args, x: torch.Tensor):
        # define how to process train sample before send it to model
        return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

    def preprocess_test_sample(self, args, x: torch.Tensor):
        # define how to process test sample before send it to model
        return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

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
    # python -m torch.distributed.launch --nproc_per_node=2 -m spikingjelly.activation_based.model.train_imagenet_example --T 4 --model spiking_resnet18 --data-path /datasets/ImageNet0_125 --batch-size 4 --lr 0.1 --lr-scheduler cosa --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 90 --random-erase 0.1 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --model-ema -j 4 --weight-decay 0.0 --val-resize-size 232
    trainer = SResNetTrainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)

