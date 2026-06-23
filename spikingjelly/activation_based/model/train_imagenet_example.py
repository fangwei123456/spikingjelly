import torch
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_resnet, train_classify


class SResNetTrainer(train_classify.Trainer):
    r"""
    **API Language** - :ref:`中文 <SResNetTrainer-cn>` | :ref:`English <SResNetTrainer-en>`

    ----

    .. _SResNetTrainer-cn:

    * **中文**

    ``SResNetTrainer`` 是一个用于在 ImageNet 数据集上训练脉冲 ResNet 模型的训练器类。
    它继承自 :class:`train_classify.Trainer` ，并重写了数据预处理、模型输出处理、模型加载等方法。

    主要功能：

    - 数据预处理：将 ``[N, C, H, W]`` 形状的输入扩展为 ``[T, N, C, H, W]`` ，其中 ``T`` 为总时间步数。
    - 模型输出处理：将 ``T`` 个时间步的输出沿时间维取均值，作为最终的预测结果（发放率）。
    - 模型加载：支持从 :mod:`spiking_resnet` 加载多种脉冲 ResNet 模型，并可选择 CuPy 后端加速。
    - 额外命令行参数：添加了 ``--T`` （时间步数）和 ``--cupy`` （是否使用 CuPy 后端）参数。

    ----

    .. _SResNetTrainer-en:

    * **English**

    ``SResNetTrainer`` is a trainer for training spiking ResNet models on the ImageNet dataset.
    It inherits from :class:`train_classify.Trainer` and overrides data preprocessing, model output
    processing, and model loading methods.

    Key features:

    - Data preprocessing: expands input from ``[N, C, H, W]`` to ``[T, N, C, H, W]``, where ``T``
      is the total number of time-steps.
    - Model output processing: averages outputs over ``T`` time-steps along the time dimension as
      the final prediction (firing rate).
    - Model loading: supports loading various spiking ResNet models from :mod:`spiking_resnet` with
      an optional CuPy backend for acceleration.
    - Extra CLI arguments: adds ``--T`` (number of time-steps) and ``--cupy`` (enable CuPy backend)
      arguments.
    """

    def preprocess_train_sample(self, args, x: torch.Tensor):
        # define how to process train sample before send it to model
        return x.unsqueeze(0).expand(
            args.T, -1, -1, -1, -1
        )  # [N, C, H, W] -> [T, N, C, H, W]

    def preprocess_test_sample(self, args, x: torch.Tensor):
        # define how to process test sample before send it to model
        return x.unsqueeze(0).expand(
            args.T, -1, -1, -1, -1
        )  # [N, C, H, W] -> [T, N, C, H, W]

    def process_model_output(self, args, y: torch.Tensor):
        return y.mean(0)  # return firing rate

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument("--T", type=int, help="total time-steps")
        parser.add_argument(
            "--cupy", action="store_true", help="set the neurons to use cupy backend"
        )
        return parser

    def get_tb_logdir_name(self, args):
        return super().get_tb_logdir_name(args) + f"_T{args.T}"

    def load_model(self, args, num_classes):
        if args.model in spiking_resnet.__all__:
            model = spiking_resnet.__dict__[args.model](
                pretrained=args.pretrained,
                spiking_neuron=neuron.IFNode,
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
            )
            functional.set_step_mode(model, step_mode="m")
            if args.cupy:
                functional.set_backend(model, "cupy", neuron.IFNode)

            return model
        else:
            raise ValueError(f"args.model should be one of {spiking_resnet.__all__}")


if __name__ == "__main__":
    # -m torch.distributed.launch --nproc_per_node=2 spikingjelly.activation_based.model.train_imagenet_example
    # python -m spikingjelly.activation_based.model.train_imagenet_example --T 4 --model spiking_resnet18 --data-path /datasets/ImageNet0_03125 --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90
    trainer = SResNetTrainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)
