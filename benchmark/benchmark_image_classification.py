# SPDX-License-Identifier: Apache-2.0
"""CIFAR-100 MaxFormer/MaxResNet and Tiny ImageNet CLIF task benchmarks.

Max recipe: bic-L/MaxFormer, 6ddd4ccde2cc9c4b965059403868e57803b9b6b0,
cifar10-100/{cifar100.yaml,train.py,aa_snn.py}; timm 0.6.12.
CLIF recipe: HuuYuLong/Complementary-LIF,
d76af5918b5797be367150d6761d10946dcab30b, train.py and ICML 2024 Appendix I.
See benchmark/README.md for source/default discrepancies and measurement scope.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import spikingjelly
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import max_resnet18, maxformer_cifar


_MAX_REVISION = "6ddd4ccde2cc9c4b965059403868e57803b9b6b0"
_CLIF_REVISION = "d76af5918b5797be367150d6761d10946dcab30b"


class _MaxRandAugment:
    def __call__(self, image):
        # aa_snn.py deliberately restricts RandAugment to these five operations.
        op = np.random.choice(("equalize", "rotate", "shear", "x", "y"))
        if random.random() > 0.5:
            return image
        magnitude = min(10.0, max(0.0, random.gauss(9.0, 0.4))) / 10
        if op == "equalize":
            return ImageOps.equalize(image)
        magnitude *= -1 if random.random() > 0.5 else 1
        kwargs = dict(resample=Image.Resampling.BICUBIC, fillcolor=(129, 124, 112))
        if op == "rotate":
            return image.rotate(30 * magnitude, **kwargs)
        matrix = (
            (1, 0.3 * magnitude, 0, 0, 1, 0)
            if op == "shear"
            else (1, 0, 0.45 * magnitude * image.width, 0, 1, 0)
            if op == "x"
            else (1, 0, 0, 0, 1, 0.45 * magnitude * image.height)
        )
        return image.transform(image.size, Image.Transform.AFFINE, matrix, **kwargs)


class _MaxErase:
    def __call__(self, x):
        if random.random() > 0.25:
            return x
        _, height, width = x.shape
        for _ in range(10):
            area = random.uniform(0.02, 1 / 3) * height * width
            aspect = math.exp(random.uniform(math.log(0.3), math.log(1 / 0.3)))
            h, w = round(math.sqrt(area * aspect)), round(math.sqrt(area / aspect))
            if 0 < h < height and 0 < w < width:
                top, left = random.randint(0, height - h), random.randint(0, width - w)
                x[:, top : top + h, left : left + w] = 0
                break
        return x


class _TinyValidation(Dataset):
    def __init__(self, root, class_to_idx, transform):
        self.transform = transform
        self.samples = []
        for line in (root / "val_annotations.txt").read_text().splitlines():
            name, class_name, *_ = line.split("\t")
            if Path(name).name != name or class_name not in class_to_idx:
                raise ValueError(f"Invalid Tiny ImageNet annotation: {line}")
            path = root / "images" / name
            if not path.is_file():
                raise FileNotFoundError(path)
            self.samples.append((path, class_to_idx[class_name]))
        if not self.samples or len({p for p, _ in self.samples}) != len(self.samples):
            raise ValueError("Empty or duplicated Tiny ImageNet annotations.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        return self.transform(datasets.folder.default_loader(path)), target


class _SourceRectangle(surrogate.Rect):
    @staticmethod
    def spiking_function(x, alpha):
        # Author Rectangle uses inclusive endpoints; SJ Rect/paper use strict ones.
        proxy = x.clamp(-alpha / 2, alpha / 2) / alpha
        return surrogate.heaviside(x) + (proxy - proxy.detach())


def _clif_vgg():
    modules, channels = [], 3
    for output_channels in (64, 128, 256, 512, 512):
        for _ in range(2):
            modules.extend(
                [
                    layer.Conv2d(channels, output_channels, 3, padding=1),
                    layer.BatchNorm2d(output_channels),
                    neuron.ComplementaryLIFNode(
                        tau=2.0, surrogate_function=_SourceRectangle(alpha=1.0)
                    ),
                    layer.Dropout(p=0.0),
                ]
            )
            channels = output_channels
        modules.append(layer.AvgPool2d(2, 2))
    modules.extend(
        [
            layer.AdaptiveAvgPool2d((7, 7)),
            layer.Flatten(),
            layer.Linear(512 * 7 * 7, 200),
        ]
    )
    net = nn.Sequential(*modules)
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)
    return net


def _model(args):
    if args.case == "clif-tinyimagenet":
        return _clif_vgg()
    factory = maxformer_cifar if args.case == "maxformer-cifar100" else max_resnet18
    return factory(T=args.time_steps, num_classes=100)


def _datasets(args):
    tiny = args.case == "clif-tinyimagenet"
    mean = (0.4802, 0.4481, 0.3975) if tiny else (0.5071, 0.4867, 0.4408)
    std = (0.2302, 0.2265, 0.2262) if tiny else (0.2675, 0.2565, 0.2761)
    normalize = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if tiny:
        augmentation = [transforms.RandomCrop(64), transforms.RandomHorizontalFlip()]
        if args.recipe == "paper":
            augmentation.append(
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
            )
        train_transform = transforms.Compose(augmentation + normalize)
        eval_transform = transforms.Compose(normalize)
        train = datasets.ImageFolder(args.data_root / "train", train_transform)
        if len(train.classes) != 200:
            raise ValueError("Tiny ImageNet requires all 200 training classes.")
        root = args.data_root / "val"
        if (root / "val_annotations.txt").is_file():
            evaluation = _TinyValidation(root, train.class_to_idx, eval_transform)
        else:
            evaluation = datasets.ImageFolder(root, eval_transform)
            if evaluation.class_to_idx != train.class_to_idx:
                raise ValueError("Train/validation class mappings differ.")
        return train, evaluation
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                32,
                scale=(1, 1),
                ratio=(1, 1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            _MaxRandAugment(),
            transforms.PILToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(32),
            transforms.PILToTensor(),
        ]
    )
    return (
        datasets.CIFAR100(
            args.data_root, True, train_transform, download=args.download
        ),
        datasets.CIFAR100(
            args.data_root, False, eval_transform, download=args.download
        ),
    )


def _mixup(x, target, enabled):
    smooth = torch.full((len(target), 100), 0.001, device=target.device).scatter_(
        1, target[:, None], 0.901
    )
    if not enabled:
        return x, smooth
    if len(x) % 2:
        raise ValueError("Official batch Mixup requires an even batch size.")
    np.random.rand()  # Upstream batch-application probability is one.
    cutmix = np.random.rand() < 0.5
    lam = float(np.random.beta(0.5 if cutmix else 0.75, 0.5 if cutmix else 0.75))
    other = x.flip(0)
    if cutmix:
        height, width = x.shape[-2:]
        cut_h, cut_w = int(height * math.sqrt(1 - lam)), int(width * math.sqrt(1 - lam))
        cy, cx = np.random.randint(height), np.random.randint(width)
        y1, y2 = max(0, cy - cut_h // 2), min(height, cy + cut_h // 2)
        x1, x2 = max(0, cx - cut_w // 2), min(width, cx + cut_w // 2)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = other[:, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (height * width)
    else:
        x = (x.float() * lam + other.float() * (1 - lam)).round().to(torch.uint8)
        # FastCollateMixup rounds mixed raw pixels before normalization.
    return x, smooth * lam + smooth.flip(0) * (1 - lam)


def _prepare(x, args, training=False):
    if args.case == "clif-tinyimagenet":
        return x
    x = x.float()
    mean = x.new_tensor((0.5071, 0.4867, 0.4408)).view(1, 3, 1, 1) * 255
    std = x.new_tensor((0.2675, 0.2565, 0.2761)).view(1, 3, 1, 1) * 255
    x.sub_(mean).div_(std)
    if training:
        for sample in x:
            _MaxErase()(sample)
    return x


def _forward(net, x, args):
    if args.case == "clif-tinyimagenet":
        return torch.stack([net(x) for _ in range(args.time_steps)])
    return net(x)


def _loss(output, target, args):
    if args.case == "clif-tinyimagenet":
        logits = output.flatten(0, 1)
        return 0.95 * F.cross_entropy(
            logits, target.repeat(args.time_steps)
        ) + 0.05 * F.mse_loss(logits, torch.ones_like(logits))
    return F.cross_entropy(output, target)


def _scores(output, args):
    return output.mean(0) if args.case == "clif-tinyimagenet" else output


def _autocast(args, device, training=True):
    enabled = args.amp and (training or args.case != "clif-tinyimagenet")
    return torch.autocast(device.type, dtype=torch.float16, enabled=enabled)


def _batch(net, x, target, args, device, optimizer=None, scaler=None, check=True):
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    try:
        with _autocast(args, device, training=optimizer is not None):
            output = _forward(net, x, args)
            loss = _loss(output, target, args)
        if check and not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss.")
        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if check:
                gradients = [p.grad for p in net.parameters() if p.requires_grad]
                if any(g is None for g in gradients):
                    raise RuntimeError("Missing parameter gradient.")
                if not args.amp:
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), float("inf"), error_if_nonfinite=True
                    )
            scaler.step(optimizer)
            scaler.update()
        return output.detach(), loss.detach()
    finally:
        functional.reset_net(net)


def _epoch(net, loader, args, device, optimizer=None, scaler=None, epoch=0):
    training = optimizer is not None
    net.train(training)
    samples = correct1 = correct5 = 0
    loss_sum = 0.0
    skipped_updates = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for index, (x, target) in enumerate(loader):
            limit = args.train_batches if training else args.eval_batches
            if limit and index >= limit:
                break
            if training and args.case != "clif-tinyimagenet":
                x, target = _mixup(x, target, epoch < 250)
            x = _prepare(x.to(device), args, training)
            target = target.to(device)
            scale_before = scaler.get_scale() if training else None
            output, loss = _batch(net, x, target, args, device, optimizer, scaler)
            if training and scaler.get_scale() < scale_before:
                skipped_updates += 1
            scores = _scores(output, args)
            labels = target.argmax(1) if target.ndim == 2 else target
            correct1 += (scores.argmax(1) == labels).sum().item()
            correct5 += (
                (scores.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item()
            )
            samples += len(x)
            loss_sum += loss.item() * len(x)
    if not samples:
        raise ValueError("No samples processed; check batch size and dataset.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return dict(
        samples=samples,
        loss=loss_sum / samples,
        top1=100 * correct1 / samples,
        top5=100 * correct5 / samples,
        seconds=elapsed,
        end_to_end_samples_per_second=samples / elapsed,
        amp_skipped_updates=skipped_updates,
        precision="fp16"
        if args.amp and (training or args.case != "clif-tinyimagenet")
        else "fp32",
    )


def _optimizer(net, args):
    if args.case == "clif-tinyimagenet":
        return torch.optim.SGD(
            net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
    no_decay, decay = [], []
    for name, parameter in net.named_parameters():
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(
            parameter
        )
    return torch.optim.AdamW(
        [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": 0.06},
        ],
        lr=0.0015,
    )


def _learning_rate(args, epoch):
    if args.case == "clif-tinyimagenet":
        return 0.05 * (1 + math.cos(math.pi * epoch / 300))
    if epoch < 20:
        return 1e-5 + epoch * (0.0015 - 1e-5) / 20
    return 1e-5 + 0.5 * (0.0015 - 1e-5) * (
        1 + math.cos(math.pi * min(epoch, 400) / 400)
    )


class _CLIFPoolReference(nn.Module):
    def forward(self, x):
        if x.shape[-2:] != (2, 2):
            raise ValueError("CLIF correctness pooling expects 2x2 spike rates")
        bounds = ((0, 1),) * 3 + ((0, 2),) + ((1, 2),) * 3
        output = torch.stack(
            [
                torch.stack([x[..., a:b, c:d].mean((-2, -1)) for c, d in bounds], -1)
                for a, b in bounds
            ],
            -2,
        )
        with torch.no_grad():
            torch.testing.assert_close(
                output, F.adaptive_avg_pool2d(x, (7, 7)), rtol=0, atol=0
            )
        return output


def _check(net, x, target, args, device):
    state = copy.deepcopy(net.state_dict())
    mode = net.training
    x = x.detach().clone().requires_grad_()
    pool = net[-3] if args.case == "clif-tinyimagenet" else None
    try:
        if pool is not None:
            # CUDA adaptive-pool backward uses atomics; keep the strict oracle
            # deterministic without changing the timed or quality model.
            if not isinstance(pool, nn.AdaptiveAvgPool2d):
                raise ValueError("Unexpected CLIF pooling layer")
            net[-3] = _CLIFPoolReference()
        net.train()
        with _autocast(args, device):
            output = _forward(net, x, args)
            loss = _loss(output, target, args)
        if _scores(output, args).shape != (
            len(x),
            200 if args.case == "clif-tinyimagenet" else 100,
        ):
            raise RuntimeError("Incorrect output shape.")
        loss.backward()
        if (
            x.grad is None
            or not torch.isfinite(x.grad).all()
            or any(
                p.grad is None or not torch.isfinite(p.grad).all()
                for p in net.parameters()
            )
        ):
            raise RuntimeError("Invalid gradient.")
        packet = {
            "output": output.detach().cpu(),
            "input_gradient": x.grad.detach().cpu(),
            "parameter_gradients": {
                name: p.grad.detach().cpu() for name, p in net.named_parameters()
            },
            "states": {
                f"{name}.{key}": value[:2].detach().cpu()
                if value.ndim
                else value.detach().cpu()
                for name, module in net.named_modules()
                if isinstance(module, neuron.BaseNode)
                for key, value in module._memories.items()
                if isinstance(value, torch.Tensor)
            },
        }
        functional.reset_net(net)
        with torch.no_grad(), _autocast(args, device):
            torch.testing.assert_close(_forward(net, x, args), output.detach())
    finally:
        if pool is not None:
            net[-3] = pool
        functional.reset_net(net)
        net.load_state_dict(state)
        net.zero_grad(set_to_none=True)
        net.train(mode)
    return packet


def _clif_execution(args, device, training):
    if args.case != "clif-tinyimagenet":
        return None
    with torch.set_grad_enabled(training):
        selected = functional.neuron_implementation("clif", device)["implementation"]
    return {
        "requested": os.environ.get("SJ_CLIF_CUDA_IMPLEMENTATION", "auto"),
        "selected": selected,
        "state_dtype": "torch.float16" if training and args.amp else "torch.float32",
    }


def _benchmark(net, evaluation, args, device):
    if device.type != "cuda":
        raise ValueError("Performance measurements require CUDA.")
    if len(evaluation) < args.batch_size:
        raise ValueError(
            "Evaluation split is smaller than the requested benchmark batch."
        )
    indices = torch.linspace(0, len(evaluation) - 1, args.batch_size).long().tolist()
    x, target = next(
        iter(DataLoader(Subset(evaluation, indices), batch_size=args.batch_size))
    )
    if args.phase == "training" and args.case != "clif-tinyimagenet":
        _, target = _mixup(x, target, False)
    x, target = _prepare(x.to(device), args), target.to(device)
    packet = _check(net, x, target, args, device)
    correctness = args.output_dir / "correctness.pt"
    input_hash = hashlib.sha256(
        x.detach().cpu().contiguous().numpy().tobytes()
        + target.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    training = args.phase == "training"
    net.train(training)
    if not training:
        with torch.no_grad():
            expected, _ = _batch(net, x, target, args, device)
            repeated, _ = _batch(net, x, target, args, device)
            torch.testing.assert_close(repeated, expected)
            packet["inference_output"] = expected.cpu()
    torch.save(packet, correctness)
    optimizer = _optimizer(net, args) if training else None
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
    gpu_ms, wall_ms = [], []
    with torch.set_grad_enabled(training):
        for _ in range(args.warmup):
            _batch(net, x, target, args, device, optimizer, scaler, check=False)
        torch.cuda.synchronize(device)
        scale_start = scaler.get_scale()
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.steps):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            before = time.perf_counter()
            start.record()
            _batch(net, x, target, args, device, optimizer, scaler, check=False)
            end.record()
            end.synchronize()
            gpu_ms.append(start.elapsed_time(end))
            wall_ms.append((time.perf_counter() - before) * 1000)
    if scaler.get_scale() != scale_start:
        raise RuntimeError("AMP scale changed during timing; measurements are invalid")
    if any(not torch.isfinite(p).all() for p in net.parameters()):
        raise RuntimeError("Non-finite model parameters after measurement")
    if not all(math.isfinite(v) and v > 0 for v in gpu_ms + wall_ms):
        raise RuntimeError("Invalid timing samples.")
    execution = _clif_execution(args, device, training)
    if execution is not None:
        execution["correctness_state_dtypes"] = sorted(
            {str(v.dtype) for v in packet["states"].values()}
        )
    return dict(
        phase=args.phase,
        precision="fp16"
        if args.amp and (training or args.case != "clif-tinyimagenet")
        else "fp32",
        amp_scale_start=scale_start,
        amp_scale_end=scaler.get_scale(),
        scope="fixed real-input batch; excludes loader and augmentation",
        correctness="output/gradient/reset checked outside timing",
        correctness_pooling="deterministic CLIF window reference"
        if args.case == "clif-tinyimagenet"
        else "production",
        clif_execution=execution,
        correctness_file=correctness.name,
        correctness_batch_size=len(x),
        state_samples=min(2, len(x)),
        correctness_sha256=hashlib.sha256(correctness.read_bytes()).hexdigest(),
        input_sha256=input_hash,
        indices=indices,
        gpu_ms=gpu_ms,
        wall_ms=wall_ms,
        gpu_median_ms=statistics.median(gpu_ms),
        samples_per_second=1000 * len(x) / statistics.median(wall_ms),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("maxformer-cifar100", "maxresnet-cifar100", "clif-tinyimagenet"),
        required=True,
    )
    parser.add_argument(
        "--recipe",
        choices=("paper", "source"),
        default="paper",
        help="CLIF paper B256+AutoAugment, or README defaults B128 without AutoAugment",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("train", "evaluate", "benchmark"), default="train"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--time-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-batches", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=0)
    parser.add_argument(
        "--phase", choices=("training", "inference"), default="inference"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    tiny = args.case == "clif-tinyimagenet"
    if not tiny:
        args.recipe = "source"
    defaults = dict(
        time_steps=6 if tiny else 4,
        epochs=300 if tiny else 410,
        batch_size=(256 if args.recipe == "paper" else 128) if tiny else 64,
        workers=0 if tiny else 4,
        seed=2022 if tiny else 50425,
    )
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if min(args.time_steps, args.epochs, args.batch_size, args.steps) <= 0:
        parser.error("time-steps, epochs, batch-size and steps must be positive")
    if min(args.workers, args.warmup, args.train_batches, args.eval_batches) < 0:
        parser.error("workers, warmup and batch limits must be nonnegative")
    if not tiny and args.batch_size % 2:
        parser.error("CIFAR batch size must be even for the official Mixup recipe")
    if tiny and args.download:
        parser.error("Extract Tiny ImageNet beforehand")
    if args.mode != "train" and args.checkpoint is None:
        parser.error("evaluate/benchmark requires checkpoint")
    if args.epochs > defaults["epochs"]:
        parser.error("epochs cannot exceed the selected recipe schedule")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("device must be cpu or cuda")
    if args.amp and device.type != "cuda":
        parser.error("FP16 AMP requires CUDA; use --no-amp for CPU correctness checks")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
        torch.backends.cudnn.deterministic = tiny
        torch.backends.cudnn.benchmark = not tiny
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train, evaluation = _datasets(args)
    net = _model(args).to(device)
    optimizer = _optimizer(net, args)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
    config = dict(case=args.case, recipe=args.recipe, time_steps=args.time_steps)
    history, start_epoch, best_top1 = [], 0, -1.0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != config:
            raise ValueError("Checkpoint model/recipe/time steps differ")
        net.load_state_dict(checkpoint["model"])
        if args.mode == "train":
            expected = {
                k: getattr(args, k) for k in ("batch_size", "amp", "seed", "workers")
            }
            if checkpoint["training_config"] != expected:
                raise ValueError("Resume training configuration differs")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch, history, best_top1 = (
                checkpoint["epoch"],
                checkpoint["history"],
                checkpoint["best_top1"],
            )
            torch.set_rng_state(checkpoint["torch_rng"])
            random.setstate(checkpoint["python_rng"])
            state = checkpoint["numpy_rng"]
            np.random.set_state(
                (state[0], np.array(state[1], dtype=np.uint32), *state[2:])
            )
            if device.type == "cuda":
                if checkpoint["cuda_rng"] is None:
                    raise ValueError("CUDA resume requires CUDA RNG state")
                torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
    complete = (len(train), len(evaluation)) == (
        (100000, 10000) if tiny else (50000, 10000)
    )
    revision = None
    if shutil.which("git"):
        revision = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
    report = dict(
        configuration={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        upstream=dict(
            repository="HuuYuLong/Complementary-LIF" if tiny else "bic-L/MaxFormer",
            commit=_CLIF_REVISION if tiny else _MAX_REVISION,
        ),
        runner_commit=revision.stdout.strip()
        if revision and revision.returncode == 0
        else None,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            torch=torch.__version__,
            torchvision=torchvision.__version__,
            spikingjelly=importlib.metadata.version("spikingjelly"),
            spikingjelly_path=spikingjelly.__file__,
            cuda=torch.version.cuda,
            device=str(device),
            amp=args.amp,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        ),
        clif_execution=_clif_execution(
            args,
            device,
            args.mode == "train"
            or (args.mode == "benchmark" and args.phase == "training"),
        ),
        parameters=sum(p.numel() for p in net.parameters()),
        train_samples=len(train),
        evaluation_samples=len(evaluation),
        complete_dataset=complete,
        classes_sha256=hashlib.sha256(json.dumps(train.classes).encode()).hexdigest(),
        bounded=bool(
            not complete
            or args.train_batches
            or args.eval_batches
            or (args.mode == "train" and args.epochs != defaults["epochs"])
        ),
        recipe_overrides={
            k: getattr(args, k) for k, v in defaults.items() if getattr(args, k) != v
        },
        quality_protocol="final epoch and best-on-evaluation are separate; no independent holdout",
        metric_units=dict(top1="percent", top5="percent", timing="ms", memory="bytes"),
    )
    if revision and revision.returncode == 0:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent),
                "status",
                "--porcelain",
            ],
            capture_output=True,
            text=True,
        )
        report["runner_worktree_dirty"] = bool(status.stdout.strip())
    if device.type == "cuda":
        report["environment"]["gpu"] = torch.cuda.get_device_name(device)
        report["environment"]["gpu_uuid"] = str(
            torch.cuda.get_device_properties(device).uuid
        )
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        report["environment"]["driver_version"] = driver.stdout.splitlines()[0]
    if args.checkpoint:
        with args.checkpoint.open("rb") as stream:
            report["checkpoint_sha256"] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
        report["checkpoint_epoch"] = checkpoint["epoch"]
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if args.mode == "benchmark":
        report["measurements"] = _benchmark(net, evaluation, args, device)
    else:
        eval_loader = DataLoader(
            evaluation,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        if args.mode == "evaluate":
            report["evaluation"] = _epoch(net, eval_loader, args, device)
        else:
            if start_epoch >= args.epochs:
                raise ValueError("No training epochs remain")
            loader = DataLoader(
                train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                drop_last=not tiny,
            )
            for epoch in range(start_epoch, args.epochs):
                for group in optimizer.param_groups:
                    group["lr"] = _learning_rate(args, epoch)
                training = _epoch(net, loader, args, device, optimizer, scaler, epoch)
                evaluated = _epoch(net, eval_loader, args, device)
                history.append(
                    dict(
                        epoch=epoch + 1,
                        lr=optimizer.param_groups[0]["lr"],
                        training=training,
                        evaluation=evaluated,
                    )
                )
                is_best = evaluated["top1"] > best_top1
                best_top1 = max(best_top1, evaluated["top1"])
                np_state = np.random.get_state()
                checkpoint = dict(
                    model_config=config,
                    training_config={
                        k: getattr(args, k)
                        for k in ("batch_size", "amp", "seed", "workers")
                    },
                    model=net.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(),
                    epoch=epoch + 1,
                    history=history,
                    best_top1=best_top1,
                    torch_rng=torch.get_rng_state(),
                    python_rng=random.getstate(),
                    numpy_rng=(np_state[0], np_state[1].tolist(), *np_state[2:]),
                    cuda_rng=torch.cuda.get_rng_state(device)
                    if device.type == "cuda"
                    else None,
                )
                torch.save(checkpoint, args.output_dir / "checkpoint_last.pt")
                if is_best:
                    torch.save(checkpoint, args.output_dir / "checkpoint_best.pt")
                with (args.output_dir / "epochs.jsonl").open("a") as stream:
                    stream.write(json.dumps(history[-1], allow_nan=False) + "\n")
                print(json.dumps(history[-1], allow_nan=False), flush=True)
            report.update(
                history=history,
                best_evaluation_top1=best_top1,
                final_evaluation=history[-1]["evaluation"],
            )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    _main()
