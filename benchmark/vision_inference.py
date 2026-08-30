from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from spikingjelly.activation_based.distributed import vision
from spikingjelly.activation_based.precision import PrecisionConfig


def _spread_indices(size: int, samples: int) -> list[int]:
    return torch.linspace(0, size - 1, samples, dtype=torch.int64).tolist()


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
        shape = (3, image_size, image_size)
        if input_layout == "NTCHW":
            shape = (time_steps, *shape)
        self.image = torch.rand(shape, generator=torch.Generator().manual_seed(seed))

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int):
        return self.image, index % self.classes


def build_synthetic_dataset(
    samples: int,
    classes: int,
    image_size: int,
    time_steps: int,
    input_layout: str,
    seed: int = 1234,
) -> Dataset:
    return _SyntheticImages(
        samples, classes, image_size, time_steps, input_layout, seed
    )


def build_imagefolder_dataset(data_dir: Path, image_size: int, samples: int) -> Dataset:
    from torch.utils.data import Subset
    from torchvision import datasets, transforms

    dataset = datasets.ImageFolder(
        data_dir,
        transforms.Compose(
            (
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            )
        ),
    )
    if samples > len(dataset):
        raise ValueError("samples cannot exceed the ImageFolder dataset size.")
    return Subset(dataset, _spread_indices(len(dataset), samples))


def build_cifar10_dataset(data_dir: Path, image_size: int, samples: int) -> Dataset:
    from torch.utils.data import Subset
    from torchvision import datasets, transforms

    dataset = datasets.CIFAR10(
        data_dir,
        train=False,
        download=True,
        transform=transforms.Compose(
            (transforms.Resize(image_size), transforms.ToTensor())
        ),
    )
    if samples > len(dataset):
        raise ValueError("samples cannot exceed the CIFAR10 test dataset size.")
    return Subset(dataset, _spread_indices(len(dataset), samples))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed vision inference")
    parser.add_argument("--artifact", required=True, type=Path)
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--data", type=Path)
    data.add_argument("--cifar10-data", type=Path)
    parser.add_argument("--export-checkpoint", type=Path)
    parser.add_argument(
        "--model",
        choices=("sew-resnet34", "spikformer", "spikformer-cifar10"),
        required=True,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-microbatches", type=int, default=1)
    parser.add_argument(
        "--data-parallel", choices=("replicate", "fsdp2"), default="replicate"
    )
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
    parser.add_argument("--input-layout", choices=("NCHW", "NTCHW"), default="NCHW")
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timing-warmup-batches", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "compile", "cuda_graph"),
        default="eager",
    )
    parser.add_argument("--cuda-graph-warmup-steps", type=int, default=3)
    args = parser.parse_args()
    if min(args.samples, args.classes, args.image_size, args.time_steps) <= 0:
        parser.error("samples, classes, image-size, and time-steps must be positive")
    if (
        args.data is not None or args.cifar10_data is not None
    ) and args.input_layout != "NCHW":
        parser.error("real-image inference requires --input-layout NCHW")
    return args


def main() -> None:
    args = _parse_args()
    if args.export_checkpoint is not None:
        vision.export_inference_artifact(args.export_checkpoint, args.artifact)
        return
    config_type = (
        vision.PredictionConfig if args.output is not None else vision.EvaluationConfig
    )
    dataset_builder = f"{__name__}.build_synthetic_dataset"
    dataset_kwargs = {
        "samples": args.samples,
        "classes": args.classes,
        "image_size": args.image_size,
        "time_steps": args.time_steps,
        "input_layout": args.input_layout,
        "seed": 1234,
    }
    if args.data is not None:
        dataset_builder = f"{__name__}.build_imagefolder_dataset"
        dataset_kwargs = {
            "data_dir": args.data,
            "image_size": args.image_size,
            "samples": args.samples,
        }
    elif args.cifar10_data is not None:
        dataset_builder = f"{__name__}.build_cifar10_dataset"
        dataset_kwargs = {
            "data_dir": args.cifar10_data,
            "image_size": args.image_size,
            "samples": args.samples,
        }
    config = config_type(
        artifact=args.artifact,
        dataset_builder=dataset_builder,
        dataset_kwargs=dataset_kwargs,
        input_layout=args.input_layout,
        batch_size=args.batch_size,
        workers=args.workers,
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
        execution_mode=args.execution_mode,
        cuda_graph_warmup_steps=args.cuda_graph_warmup_steps,
        **(
            {"timing_warmup_batches": args.timing_warmup_batches}
            if args.output is None
            else {}
        ),
    )
    if args.output is not None:
        vision.predict_classification(config, args.output)
        result = {"output": str(args.output)}
    else:
        result = vision.evaluate_classification(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
