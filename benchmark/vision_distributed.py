from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from spikingjelly.activation_based.distributed import vision


class _SyntheticImages(Dataset):
    def __init__(self, samples: int, classes: int, image_size: int, seed: int) -> None:
        self.samples = samples
        self.classes = classes
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.randn(3, self.image_size, self.image_size, generator=generator)
        return image, index % self.classes


def build_synthetic_datasets(
    samples: int,
    classes: int,
    image_size: int,
    seed: int = 1234,
) -> tuple[Dataset, Dataset]:
    return (
        _SyntheticImages(samples, classes, image_size, seed),
        _SyntheticImages(max(8, samples // 4), classes, image_size, seed + samples),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("sew-resnet34", "spikformer"), required=True
    )
    parser.add_argument("--data-parallel", choices=("ddp", "fsdp2"), default="ddp")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-microbatches", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--timing-warmup-steps", type=int, default=0)
    parser.add_argument("--memopt-level", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
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
    if args.model == "sew-resnet34":
        model = vision.SEWResNet34Config(
            time_steps=args.time_steps,
            num_classes=args.classes,
            image_size=args.image_size,
        )
    else:
        model = vision.SpikformerConfig(
            time_steps=args.time_steps,
            num_classes=args.classes,
            image_height=args.image_size,
            image_width=args.image_size,
        )
    config = vision.TrainingConfig(
        model=model,
        dataset_builder=f"{__name__}.build_synthetic_datasets",
        dataset_kwargs={
            "samples": args.samples,
            "classes": args.classes,
            "image_size": args.image_size,
        },
        epochs=max(1, args.max_steps),
        batch_size=batch_size,
        workers=args.workers,
        optimizer="torch.optim.AdamW",
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        pipeline_microbatches=args.pipeline_microbatches,
        data_parallel=args.data_parallel,
        precision=args.precision,
        memopt_level=args.memopt_level,
        max_steps=args.max_steps,
        timing_warmup_steps=args.timing_warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )
    metrics = vision.train(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
