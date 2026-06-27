import argparse
import json
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from spikingjelly.activation_based import ann2snn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ann2snn LocalThresholdBalancingRecipe on ImageNet ResNet-18."
    )
    parser.add_argument("--data-root", required=True, help="ImageNet root with train/val.")
    parser.add_argument("--calib-samples", type=int, default=50000)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--time-steps", type=int, nargs="+", default=[32, 64])
    parser.add_argument(
        "--delay-start",
        default="auto",
        help="SNN readout start timestep: none, auto, or an integer.",
    )
    parser.add_argument(
        "--recipes",
        nargs="+",
        choices=["ann", "robust", "robust_legacy", "ltb"],
        default=["ann", "robust", "ltb"],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--ltb-mode", default="99.9%")
    parser.add_argument(
        "--threshold-candidates",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.25, 1.5],
    )
    return parser.parse_args()


def build_loaders(args, transform):
    train_dir = Path(args.data_root) / "train"
    val_dir = Path(args.data_root) / "val"
    train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    val_set = torchvision.datasets.ImageFolder(val_dir, transform=transform)

    if args.calib_samples <= 0:
        raise ValueError("--calib-samples must be positive.")
    if args.calib_samples > len(train_set):
        raise ValueError("--calib-samples exceeds ImageNet train set size.")
    stride = max(len(train_set) // args.calib_samples, 1)
    indices = list(range(0, len(train_set), stride))[: args.calib_samples]
    calib_set = Subset(train_set, indices)
    if args.eval_samples is not None:
        if args.eval_samples <= 0:
            raise ValueError("--eval-samples must be positive when set.")
        if args.eval_samples > len(val_set):
            raise ValueError("--eval-samples exceeds ImageNet val set size.")
        stride = max(len(val_set) // args.eval_samples, 1)
        indices = list(range(0, len(val_set), stride))[: args.eval_samples]
        val_set = Subset(val_set, indices)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    calib_loader = DataLoader(calib_set, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    return calib_loader, val_loader


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, dim=1)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [
            correct[:k].reshape(-1).float().sum().item()
            for k in topk
        ]


def evaluate_ann(model, data_loader, device):
    model.eval().to(device)
    total = 0
    top1 = 0.0
    top5 = 0.0
    with torch.no_grad():
        for img, label in tqdm(data_loader):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            out = model(img)
            acc1, acc5 = accuracy(out, label)
            top1 += acc1
            top5 += acc5
            total += label.numel()
    return {"top1": top1 / total * 100.0, "top5": top5 / total * 100.0}


def reset_snn(model):
    for module in model.modules():
        if hasattr(module, "reset"):
            module.reset()


def resolve_delay_start(model, data_loader, device, time_steps, delay_start):
    if delay_start == "none":
        return 0
    if delay_start == "auto":
        return ann2snn.estimate_delay_start(model, data_loader, device, time_steps)
    value = int(delay_start)
    if value < 0 or value >= time_steps:
        raise ValueError("--delay-start must be none, auto, or in [0, time_steps).")
    return value


def evaluate_snn(model, data_loader, device, time_steps, delay_start=0):
    model.eval().to(device)
    total = 0
    top1 = 0.0
    top5 = 0.0
    with torch.no_grad():
        for img, label in tqdm(data_loader):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            reset_snn(model)
            out = None
            for t in range(time_steps):
                current = model(img)
                if t >= delay_start:
                    out = current if out is None else out + current
            if out is None:
                raise ValueError("delay_start leaves no timesteps for readout.")
            acc1, acc5 = accuracy(out, label)
            top1 += acc1
            top5 += acc5
            total += label.numel()
    return {"top1": top1 / total * 100.0, "top5": top5 / total * 100.0}


def make_model(weights):
    model = resnet18(weights=weights)
    model.eval()
    return model


def save_results(results, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


def main():
    args = parse_args()
    device = torch.device(args.device)
    weights = ResNet18_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    calib_loader, val_loader = build_loaders(args, transform)

    results = {
        "model": "resnet18",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "data_root": args.data_root,
        "calib_samples": args.calib_samples,
        "eval_samples": args.eval_samples,
        "batch_size": args.batch_size,
        "time_steps": args.time_steps,
        "delay_start": args.delay_start,
        "recipes": args.recipes,
        "metrics": {},
    }

    if "ann" in args.recipes:
        ann = make_model(weights)
        start = time.time()
        results["metrics"]["ann"] = evaluate_ann(ann, val_loader, device)
        results["metrics"]["ann"]["seconds"] = time.time() - start
        save_results(results, args.output)

    for t in args.time_steps:
        if "robust" in args.recipes:
            ann = make_model(weights)
            recipe = ann2snn.RateCodingRecipe(
                dataloader=calib_loader,
                mode="99.9%",
                fuse_flag=True,
                channel_wise=True,
                pre_spike_maxpool=True,
                half_threshold=True,
            )
            start = time.time()
            snn = ann2snn.Converter(recipe=recipe, device=device).convert(ann)
            start_t = resolve_delay_start(
                snn, calib_loader, device, t, args.delay_start
            )
            metrics = evaluate_snn(snn, val_loader, device, t, start_t)
            metrics["delay_start"] = start_t
            metrics["seconds"] = time.time() - start
            results["metrics"][f"robust_t{t}"] = metrics
            save_results(results, args.output)

        if "robust_legacy" in args.recipes:
            ann = make_model(weights)
            recipe = ann2snn.RateCodingRecipe(
                dataloader=calib_loader,
                mode="99.9%",
                fuse_flag=True,
            )
            start = time.time()
            snn = ann2snn.Converter(recipe=recipe, device=device).convert(ann)
            start_t = resolve_delay_start(
                snn, calib_loader, device, t, args.delay_start
            )
            metrics = evaluate_snn(snn, val_loader, device, t, start_t)
            metrics["delay_start"] = start_t
            metrics["seconds"] = time.time() - start
            results["metrics"][f"robust_legacy_t{t}"] = metrics
            save_results(results, args.output)

        if "ltb" in args.recipes:
            ann = make_model(weights)
            recipe = ann2snn.LocalThresholdBalancingRecipe(
                dataloader=calib_loader,
                time_steps=t,
                mode=args.ltb_mode,
                threshold_candidates=tuple(args.threshold_candidates),
                fuse_flag=True,
            )
            start = time.time()
            snn = ann2snn.Converter(recipe=recipe, device=device).convert(ann)
            start_t = resolve_delay_start(
                snn, calib_loader, device, t, args.delay_start
            )
            metrics = evaluate_snn(snn, val_loader, device, t, start_t)
            metrics["delay_start"] = start_t
            metrics["seconds"] = time.time() - start
            results["metrics"][f"ltb_t{t}"] = metrics
            save_results(results, args.output)

    save_results(results, args.output)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
