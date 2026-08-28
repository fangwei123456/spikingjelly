"""Example training entrypoint for compiled FlexSN models.
This script mirrors ``train_imagenet.py`` but uses ``spiking_vgg`` with
``FlexSN(backend="triton")`` and defaults to ``torch.compile`` so users can
exercise the compiler-visible FlexSN training path end-to-end.
"""

import torch

from .. import functional, surrogate
from ..neuron.flexsn import FlexSN
from ..model import spiking_vgg
from .common import train_classify


class _FlexSNTrainer(train_classify.Trainer):
    def preprocess_train_sample(self, args, x: torch.Tensor):
        return x.unsqueeze(0).expand(args.T, -1, -1, -1, -1)

    def preprocess_test_sample(self, args, x: torch.Tensor):
        return x.unsqueeze(0).expand(args.T, -1, -1, -1, -1)

    def process_model_output(self, args, y: torch.Tensor):
        return y.mean(0)

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser(add_help=add_help)
        parser.add_argument("--T", type=int, default=4, help="total time-steps")
        parser.add_argument(
            "--surrogate-alpha",
            type=float,
            default=4.0,
            help="alpha for surrogate.Sigmoid used by the FlexSN core",
        )
        parser.set_defaults(compile=True, compile_eval=False, disable_uda=True)
        return parser

    def get_tb_logdir_name(self, args):
        return (
            super().get_tb_logdir_name(args)
            + f"_T{args.T}_flexsn_triton_compile_sa{args.surrogate_alpha}"
        )

    def load_model(self, args, num_classes):
        if args.model not in spiking_vgg.__all__:
            raise ValueError(f"args.model should be one of {spiking_vgg.__all__}")

        sg = surrogate.Sigmoid(alpha=args.surrogate_alpha)

        def lif_core_sg(x: torch.Tensor, v: torch.Tensor):
            tau, v_th = 2.0, 1.0
            h = v + (x - v) / tau
            s = sg(h - v_th)
            return s, h * (1.0 - s)

        def make_flexsn(**kwargs):
            return FlexSN(
                core=lif_core_sg,
                num_states=1,
                step_mode=kwargs.get("step_mode", "m"),
                backend="triton",
            )

        model = spiking_vgg.__dict__[args.model](
            pretrained=args.pretrained,
            spiking_neuron=make_flexsn,
            num_classes=num_classes,
        )
        functional.set_step_mode(model, step_mode="m")
        return model


if __name__ == "__main__":
    trainer = _FlexSNTrainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)
