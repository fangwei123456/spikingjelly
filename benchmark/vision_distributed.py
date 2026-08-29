from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from spikingjelly.activation_based.distributed import vision
from spikingjelly.activation_based.model.sew_resnet import SEWResNet34Config
from spikingjelly.activation_based.model.spikformer import (
    SpikformerCIFAR10Config,
    SpikformerConfig,
)
from spikingjelly.activation_based.precision import PrecisionConfig


class _SyntheticImages(Dataset):
    def __init__(
        self,
        samples: int,
        classes: int,
        image_size: int,
        time_steps: int,
        input_layout: str,
        seed: int,
    ) -> None:
        self.samples = samples
        self.classes = classes
        self.image_size = image_size
        self.time_steps = time_steps
        self.input_layout = input_layout
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        generator = torch.Generator().manual_seed(self.seed + index)
        shape = (3, self.image_size, self.image_size)
        if self.input_layout == "NTCHW":
            shape = (self.time_steps, *shape)
        image = torch.randn(*shape, generator=generator)
        return image, index % self.classes


def build_synthetic_datasets(
    samples: int,
    classes: int,
    image_size: int,
    time_steps: int,
    input_layout: str,
    seed: int = 1234,
) -> tuple[Dataset, Dataset]:
    return (
        _SyntheticImages(samples, classes, image_size, time_steps, input_layout, seed),
        _SyntheticImages(
            max(8, samples // 4),
            classes,
            image_size,
            time_steps,
            input_layout,
            seed + samples,
        ),
    )


def build_cifar10_datasets(
    root: str | Path,
    samples: int,
) -> tuple[Dataset, Dataset]:
    from torchvision import datasets, transforms

    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=9),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25),
        ]
    )
    validation_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train = datasets.CIFAR10(root, train=True, transform=train_transform)
    train = Subset(train, range(min(samples, len(train))))
    validation = datasets.CIFAR10(
        root,
        train=False,
        transform=validation_transform,
    )
    return train, validation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("sew-resnet34", "spikformer", "spikformer-cifar10"),
        required=True,
    )
    parser.add_argument(
        "--dataset", choices=("synthetic", "cifar10"), default="synthetic"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--data-parallel", choices=("ddp", "fsdp2"), default="ddp")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-microbatches", type=int, default=1)
    parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16", "fp8"), default="bf16"
    )
    parser.add_argument(
        "--fp8-recipe",
        choices=("auto", "delayed", "current", "block", "mxfp8"),
        default="auto",
    )
    parser.add_argument(
        "--triton-storage",
        choices=("fp32", "fp16", "bf16", "float8_e4m3fn", "float8_e5m2"),
    )
    parser.add_argument(
        "--triton-fwd", choices=("fp32", "fp16", "bf16", "fp8"), default="fp32"
    )
    parser.add_argument(
        "--triton-bwd", choices=("fp32", "fp16", "bf16", "fp8"), default="fp32"
    )
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--step-mode", choices=("s", "m"), default="m")
    parser.add_argument("--input-layout", choices=("NCHW", "NTCHW"), default="NCHW")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--scheduler-steps", type=int)
    parser.add_argument("--timing-warmup-steps", type=int, default=0)
    parser.add_argument("--memopt-level", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.scheduler_steps is not None and args.scheduler_steps <= 0:
        raise ValueError("scheduler steps must be positive.")
    classes = 10 if args.dataset == "cifar10" else args.classes
    if args.dataset == "cifar10" and args.input_layout != "NCHW":
        raise ValueError("CIFAR-10 requires --input-layout NCHW.")
    if args.dataset == "cifar10" and (
        args.model != "spikformer-cifar10" or args.image_size != 32
    ):
        raise ValueError("CIFAR-10 validation requires spikformer-cifar10 at 32×32.")
    model_parallel_size = args.tensor_parallel_size * args.pipeline_parallel_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size % model_parallel_size:
        raise ValueError("world size must be divisible by PP × TP.")
    dp_size = world_size // model_parallel_size
    batch_size = args.batch_size
    if args.global_batch_size is not None:
        if args.global_batch_size % dp_size:
            raise ValueError("global batch size must be divisible by DP size.")
        batch_size = args.global_batch_size // dp_size
    if args.model == "spikformer-cifar10":
        model = SpikformerCIFAR10Config(
            time_steps=args.time_steps,
            num_classes=classes,
            step_mode=args.step_mode,
            neuron_backend="triton",
        )
    elif args.model == "sew-resnet34":
        model = SEWResNet34Config(
            time_steps=args.time_steps,
            num_classes=classes,
            step_mode=args.step_mode,
            image_size=args.image_size,
        )
    else:
        model = SpikformerConfig(
            time_steps=args.time_steps,
            num_classes=classes,
            step_mode=args.step_mode,
            image_height=args.image_size,
            image_width=args.image_size,
        )
    if args.dataset == "cifar10":
        dataset_builder = f"{__name__}.build_cifar10_datasets"
        dataset_kwargs = {
            "root": args.data_root,
            "samples": args.samples,
        }
    else:
        dataset_builder = f"{__name__}.build_synthetic_datasets"
        dataset_kwargs = {
            "samples": args.samples,
            "classes": classes,
            "image_size": args.image_size,
            "time_steps": args.time_steps,
            "input_layout": args.input_layout,
        }
    cifar10 = args.dataset == "cifar10"
    config = vision.TrainingConfig(
        model=model,
        dataset_builder=dataset_builder,
        dataset_kwargs=dataset_kwargs,
        input_layout=args.input_layout,
        epochs=max(1, args.max_steps),
        batch_size=batch_size,
        workers=args.workers,
        optimizer="torch.optim.AdamW",
        optimizer_kwargs={
            "lr": args.learning_rate,
            "weight_decay": 0.06 if cifar10 else 0.0,
        },
        loss_function="torch.nn.functional.cross_entropy",
        loss_kwargs={"label_smoothing": 0.1} if cifar10 else {},
        mixup_alpha=0.5 if cifar10 else 0.0,
        scheduler="torch.optim.lr_scheduler.CosineAnnealingLR" if cifar10 else None,
        scheduler_kwargs={"T_max": args.scheduler_steps or args.max_steps}
        if cifar10
        else {},
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        pipeline_microbatches=args.pipeline_microbatches,
        data_parallel=args.data_parallel,
        precision=PrecisionConfig(
            mode=args.precision,
            fp8_recipe=args.fp8_recipe,
            triton_storage=args.triton_storage,
            triton_fwd=args.triton_fwd,
            triton_bwd=args.triton_bwd,
        ),
        memopt_level=args.memopt_level,
        max_steps=args.max_steps,
        timing_warmup_steps=args.timing_warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )
    metrics = vision.train_classification(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
