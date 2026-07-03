import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import ViT_B_16_Weights, vit_b_16

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import Converter, STATransformerRecipe


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate STATransformerRecipe on torchvision ViT-B/16 with an "
            "ImageNet validation directory."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to the ImageNet val directory readable by torchvision ImageFolder.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device, for example cuda:0. CPU execution is not supported.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--calib-samples", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--threshold-scale", type=float, default=0.5)
    return parser.parse_args()


def require_cuda(device):
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device such as cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this ImageNet ViT STA example.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"--device index {device.index} is out of range for "
            f"{torch.cuda.device_count()} CUDA device(s)."
        )
    torch.cuda.get_device_name(device)


def build_loaders(args, transform):
    data_root = Path(args.data_root)
    try:
        dataset = ImageFolder(data_root, transform=transform)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Could not open --data-root {data_root}.") from exc
    if args.calib_samples <= 0:
        raise ValueError("--calib-samples must be positive.")
    if args.calib_samples > len(dataset):
        raise ValueError("--calib-samples exceeds the dataset size.")
    if args.eval_samples is not None:
        if args.eval_samples <= 0:
            raise ValueError("--eval-samples must be positive when set.")
        if args.calib_samples + args.eval_samples > len(dataset):
            raise ValueError(
                "--calib-samples + --eval-samples exceeds the dataset size; "
                "use disjoint calibration and evaluation subsets."
            )

    calib_set = Subset(dataset, range(args.calib_samples))
    eval_set = dataset
    if args.eval_samples is not None:
        start = args.calib_samples
        eval_set = Subset(dataset, range(start, start + args.eval_samples))

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return (
        DataLoader(calib_set, **loader_kwargs),
        DataLoader(eval_set, **loader_kwargs),
        len(dataset),
    )


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum().item() for k in topk]


def evaluate(model, data_loader, device, name):
    model.eval().to(device)
    total = 0
    top1 = 0.0
    top5 = 0.0
    start = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            acc1, acc5 = accuracy(out, y)
            top1 += acc1
            top5 += acc5
            total += y.numel()
            if (i + 1) % 100 == 0:
                print(name, i + 1, total, top1 / total, top5 / total, flush=True)
    return {
        "top1": top1 / total,
        "top5": top5 / total,
        "total": total,
        "seconds": time.time() - start,
    }


def make_first_real_then_zero_sequence(x, time_steps):
    if time_steps <= 0:
        raise ValueError(f"time_steps must be positive, got {time_steps}.")
    x_seq = torch.zeros((time_steps, *x.shape), dtype=x.dtype, device=x.device)
    x_seq[0] = x
    return x_seq


def evaluate_sta(model, data_loader, device, name, time_steps):
    model.eval().to(device)
    functional.set_step_mode(model, "m")
    total = 0
    top1 = 0.0
    top5 = 0.0
    start = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            functional.reset_net(model)
            x_seq = make_first_real_then_zero_sequence(x, time_steps)
            out = model(x_seq).sum(dim=0)
            acc1, acc5 = accuracy(out, y)
            top1 += acc1
            top5 += acc5
            total += y.numel()
            if (i + 1) % 100 == 0:
                print(name, i + 1, total, top1 / total, top5 / total, flush=True)
    return {
        "top1": top1 / total,
        "top5": top5 / total,
        "total": total,
        "seconds": time.time() - start,
    }


def format_scale_label(scale):
    text = f"{scale:.8f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def main():
    args = parse_args()
    device = torch.device(args.device)
    require_cuda(device)

    weights = ViT_B_16_Weights.DEFAULT
    transform = weights.transforms()
    calib_loader, eval_loader, dataset_size = build_loaders(args, transform)
    try:
        model = vit_b_16(weights=weights).to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            "Could not load ViT-B/16 weights; check network access and the "
            "torchvision cache."
        ) from exc

    env = {
        "data_root": args.data_root,
        "dataset_size": dataset_size,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device),
        "model": "vit_b_16",
        "weights": "ViT_B_16_Weights.DEFAULT",
        "calib_samples": args.calib_samples,
        "eval_samples": args.eval_samples,
        "batch_size": args.batch_size,
        "time_steps": args.time_steps,
        "threshold_scale": args.threshold_scale,
    }
    print(json.dumps(env), flush=True)

    baseline = evaluate(model, eval_loader, device, "baseline")
    print("BASELINE", json.dumps(baseline), flush=True)

    recipe = STATransformerRecipe(
        dataloader=calib_loader,
        time_steps=args.time_steps,
        mode="spiking_encoder",
        threshold_mode="mse",
        threshold_scale=args.threshold_scale,
    )
    converted = Converter(recipe=recipe, device=device).convert(model).to(device).eval()
    sta_label = (
        f"STA_SPIKING_ENCODER_T{args.time_steps}_"
        f"S{format_scale_label(args.threshold_scale)}"
    )
    sta = evaluate_sta(
        converted,
        eval_loader,
        device,
        sta_label.lower(),
        args.time_steps,
    )
    print(sta_label, json.dumps(sta), flush=True)
    print("DROP", baseline["top1"] - sta["top1"], flush=True)


if __name__ == "__main__":
    main()
