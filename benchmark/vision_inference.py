from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from torch.utils.data import Dataset

from spikingjelly.activation_based.distributed import vision


class _SyntheticImages(Dataset):
    def __init__(self, samples: int, classes: int, image_size: int) -> None:
        import torch

        self.samples = samples
        self.classes = classes
        self.image = torch.zeros(3, image_size, image_size)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int):
        return self.image, index % self.classes


def build_synthetic_dataset(samples: int, classes: int, image_size: int) -> Dataset:
    return _SyntheticImages(samples, classes, image_size)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed vision inference")
    parser.add_argument("--artifact", required=True, type=Path)
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
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--input-layout", choices=("NCHW", "NTCHW"), default="NCHW")
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timing-warmup-batches", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.export_checkpoint is not None:
        vision.export_inference_artifact(args.export_checkpoint, args.artifact)
        return
    config_type = (
        vision.PredictionConfig if args.output is not None else vision.EvaluationConfig
    )
    config = config_type(
        artifact=args.artifact,
        dataset_builder=f"{__name__}.build_synthetic_dataset",
        dataset_kwargs={
            "samples": args.samples,
            "classes": args.classes,
            "image_size": args.image_size,
        },
        input_layout=args.input_layout,
        batch_size=args.batch_size,
        workers=args.workers,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        pipeline_microbatches=args.pipeline_microbatches,
        data_parallel=args.data_parallel,
        precision=args.precision,
        compile=args.compile,
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
