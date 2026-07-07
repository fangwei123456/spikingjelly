from __future__ import annotations

import argparse
import json
import os
import pickle
import socket
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import safe_globals
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import ModuleConverter, SpikeZIPTFQANNRecipe
from spikingjelly.activation_based.neuron import STBIFNeuron


class SpikeZIPMyQuan(nn.Module):
    def __init__(self, level: int, sym: bool = False) -> None:
        super().__init__()
        self.level = int(level)
        self.sym = bool(sym)
        if sym:
            self.pos_max = torch.tensor(float(level // 2 - 1))
            self.neg_min = torch.tensor(float(-level // 2))
        else:
            self.pos_max = torch.tensor(float(level - 1))
            self.neg_min = torch.tensor(0.0)
        self.s = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_max = self.pos_max.to(device=x.device, dtype=x.dtype)
        neg_min = self.neg_min.to(device=x.device, dtype=x.dtype)
        return torch.clamp(torch.floor(x / self.s + 0.5), neg_min, pos_max) * self.s


class SpikeZIPQAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, level: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.level = int(level)
        self.is_softmax = True
        self.qkv = nn.Linear(dim, dim * 3)
        self.quan_q = SpikeZIPMyQuan(level, sym=True)
        self.quan_k = SpikeZIPMyQuan(level, sym=True)
        self.quan_v = SpikeZIPMyQuan(level, sym=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.quan_proj = SpikeZIPMyQuan(level, sym=True)
        self.proj_drop = nn.Dropout(0.0)
        self.attn_quan = SpikeZIPMyQuan(level, sym=False)
        self.after_attn_quan = SpikeZIPMyQuan(level, sym=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = self.quan_q(query) * self.scale
        key = self.quan_k(key)
        value = self.quan_v(value)
        attention = query @ key.transpose(-2, -1)
        attention = self.attn_quan(F.softmax(attention, dim=-1))
        attention = self.attn_drop(attention)
        x = self.after_attn_quan(attention @ value)
        x = x.transpose(1, 2).reshape(batch_size, seq_len, channels)
        x = self.proj_drop(self.proj(x))
        return self.quan_proj(x)


class SpikeZIPMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, level: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.Sequential(SpikeZIPMyQuan(level, sym=False), nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, dim), SpikeZIPMyQuan(level, sym=True)
        )
        self.drop1 = nn.Dropout(0.0)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class SpikeZIPViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, level: int) -> None:
        super().__init__()
        self.norm1 = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            SpikeZIPMyQuan(level, sym=True),
        )
        self.attn = SpikeZIPQAttention(dim=dim, num_heads=num_heads, level=level)
        self.norm2 = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            SpikeZIPMyQuan(level, sym=True),
        )
        self.mlp = SpikeZIPMLP(dim=dim, hidden_dim=dim * mlp_ratio, level=level)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SpikeZIPViTSmallQANN(nn.Module):
    def __init__(self, level: int = 32) -> None:
        super().__init__()
        dim = 384
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=16, stride=16),
            SpikeZIPMyQuan(level, sym=True),
        )
        self.patch_embed.num_patches = 196
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, dim))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.Sequential(
            *[
                SpikeZIPViTBlock(dim=dim, num_heads=6, mlp_ratio=4, level=level)
                for _ in range(12)
            ]
        )
        self.norm = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            SpikeZIPMyQuan(level, sym=True),
        )
        self.head = nn.Linear(dim, 1000)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def _load_checkpoint(
    path: Path,
    allow_namespace_checkpoint: bool = False,
) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        if not allow_namespace_checkpoint:
            raise RuntimeError(
                "Checkpoint requires argparse.Namespace in torch weights_only mode. "
                "Pass --allow-namespace-checkpoint only for trusted local checkpoints."
            ) from exc
        with safe_globals([argparse.Namespace]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["model"]


class _MetricAccumulator:
    def __init__(self, parity_atol: float) -> None:
        self.parity_atol = parity_atol
        self.samples = 0
        self.diff_sum = 0.0
        self.diff_count = 0
        self.diff_max = 0.0
        self.parity_pass = True
        self.qann_top1 = 0
        self.qann_top5 = 0
        self.snn_top1 = 0
        self.snn_top5 = 0
        self.agree_top1 = 0
        self.agree_top5_set_sum = 0.0

    def update(
        self,
        qann_logits: torch.Tensor,
        snn_logits: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> None:
        batch = qann_logits.shape[0]
        diff = (snn_logits - qann_logits).detach().abs()
        self.samples += batch
        self.diff_sum += diff.sum().item()
        self.diff_count += diff.numel()
        self.diff_max = max(self.diff_max, diff.max().item())
        self.parity_pass = self.parity_pass and bool(
            torch.allclose(snn_logits, qann_logits, atol=self.parity_atol, rtol=1e-5)
        )

        qann_top5 = qann_logits.topk(5, dim=1).indices
        snn_top5 = snn_logits.topk(5, dim=1).indices
        self.agree_top1 += qann_top5[:, 0].eq(snn_top5[:, 0]).sum().item()
        self.agree_top5_set_sum += (
            qann_top5.unsqueeze(2)
            .eq(snn_top5.unsqueeze(1))
            .any(dim=2)
            .float()
            .mean(dim=1)
            .sum()
            .item()
        )

        if labels is not None:
            self.qann_top1 += qann_top5[:, 0].eq(labels).sum().item()
            self.qann_top5 += (
                qann_top5.eq(labels.reshape(-1, 1)).any(dim=1).sum().item()
            )
            self.snn_top1 += snn_top5[:, 0].eq(labels).sum().item()
            self.snn_top5 += snn_top5.eq(labels.reshape(-1, 1)).any(dim=1).sum().item()

    def result(self, has_labels: bool) -> dict:
        result = {
            "samples": int(self.samples),
            "max_abs_diff": self.diff_max,
            "mean_abs_diff": self.diff_sum / max(self.diff_count, 1),
            "parity_pass": bool(self.parity_pass),
            "prediction_agreement": {
                "top1": self.agree_top1 * 100.0 / max(self.samples, 1),
                "top5_set": self.agree_top5_set_sum * 100.0 / max(self.samples, 1),
            },
        }
        if has_labels:
            result["qann_accuracy"] = {
                "acc1": self.qann_top1 * 100.0 / max(self.samples, 1),
                "acc5": self.qann_top5 * 100.0 / max(self.samples, 1),
            }
            result["snn_accuracy"] = {
                "acc1": self.snn_top1 * 100.0 / max(self.samples, 1),
                "acc5": self.snn_top5 * 100.0 / max(self.samples, 1),
            }
        return result


def _snapshot_step_modes(module: nn.Module) -> dict[nn.Module, str]:
    return {
        child: child.step_mode
        for child in module.modules()
        if hasattr(child, "step_mode")
    }


def _restore_step_modes(step_modes: dict[nn.Module, str]) -> None:
    for module, step_mode in step_modes.items():
        module.step_mode = step_mode


class _SequenceLoopModule(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x_seq: torch.Tensor):
        step_modes = _snapshot_step_modes(self.module)
        try:
            for child in step_modes:
                child.step_mode = "s"
            return torch.stack([self.module(x) for x in x_seq], dim=0)
        finally:
            _restore_step_modes(step_modes)


def _wrap_sequence_loop_bottlenecks(
    converted: nn.Module, bottlenecks: list[str]
) -> list[str]:
    if not bottlenecks:
        return []
    if not hasattr(converted, "blocks"):
        raise ValueError(
            "--sequence-loop-bottlenecks requires a converted ViT blocks model."
        )
    wrapped = []
    for block_index, block in enumerate(converted.blocks):
        if "attention" in bottlenecks and hasattr(block, "attn"):
            if not isinstance(block.attn, _SequenceLoopModule):
                block.attn = _SequenceLoopModule(block.attn)
            wrapped.append(f"blocks.{block_index}.attn")
        if "mlp" in bottlenecks and hasattr(block, "mlp"):
            if not isinstance(block.mlp, _SequenceLoopModule):
                block.mlp = _SequenceLoopModule(block.mlp)
            wrapped.append(f"blocks.{block_index}.mlp")
    return wrapped


def _iter_image_batches(args):
    if args.input is not None:
        images = torch.load(args.input, map_location="cpu", weights_only=True)
        count = (
            images.shape[0] if args.samples <= 0 else min(args.samples, images.shape[0])
        )
        for start in range(0, count, args.batch_size):
            yield images[start : start + args.batch_size], None
        return
    if args.imagenet_root is None:
        generator = torch.Generator().manual_seed(args.seed)
        count = args.batch_size if args.samples <= 0 else args.samples
        for start in range(0, count, args.batch_size):
            batch_size = min(args.batch_size, count - start)
            yield (
                torch.randn(
                    batch_size,
                    3,
                    224,
                    224,
                    generator=generator,
                ),
                None,
            )
        return

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(
        Path(args.imagenet_root) / "val", transform=transform
    )
    if args.samples > 0:
        dataset = Subset(dataset, range(min(args.samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    for images, labels in loader:
        yield images, labels


def _first_real_then_zero_sequence(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    x_seq = torch.zeros(
        time_steps,
        *x.shape,
        device=x.device,
        dtype=x.dtype,
    )
    x_seq[0] = x
    return x_seq


def _debug_qann_outputs(features) -> dict[int, list[torch.Tensor]]:
    outputs = {}
    for index, value in features:
        outputs.setdefault(index, []).append(value)
    return outputs


def _debug_snn_outputs(features, time_steps: int) -> dict[int, list[torch.Tensor]]:
    outputs = {}
    step_outputs = {}
    for index, value in features:
        if value.ndim >= 4 and value.shape[0] == time_steps:
            outputs.setdefault(index, []).append(value.sum(dim=0))
        else:
            values = step_outputs.setdefault(index, [])
            values.append(value)
            if len(values) == time_steps:
                outputs.setdefault(index, []).append(
                    torch.stack(values, dim=0).sum(dim=0)
                )
                step_outputs[index] = []
    return outputs


def _debug_block_diffs(
    qann_outputs: dict[int, list[torch.Tensor]],
    snn_outputs: dict[int, list[torch.Tensor]],
) -> list[dict]:
    result = []
    for index, qann_values in qann_outputs.items():
        snn_values = snn_outputs.get(index, [])
        if len(qann_values) != len(snn_values):
            raise RuntimeError(
                f"debug block {index} captured {len(qann_values)} QANN outputs "
                f"but {len(snn_values)} SNN outputs."
            )
        if not qann_values:
            continue
        diffs = [
            (snn_value - qann_value).abs()
            for qann_value, snn_value in zip(qann_values, snn_values)
        ]
        max_abs = max(diff.max().item() for diff in diffs)
        mean_abs = sum(diff.sum().item() for diff in diffs) / sum(
            diff.numel() for diff in diffs
        )
        result.append(
            {
                "block": index,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
            }
        )
    return result


def _forward_snn(converted, images: torch.Tensor, args):
    x_seq = _first_real_then_zero_sequence(images, args.time_steps)
    if args.step_mode == "m":
        chunk_size = args.snn_batch_size or images.shape[0]
        if chunk_size <= 0:
            raise ValueError("--snn-batch-size must be positive when it is set.")
        logits_chunks = []
        sequence_chunks = []
        for chunk in x_seq.split(chunk_size, dim=1):
            functional.reset_net(converted)
            if args.return_sequences:
                sequence = converted(chunk)
                logits = sequence.sum(dim=0)
                sequence_chunks.append(sequence)
            else:
                logits = converted(chunk).sum(dim=0)
            logits_chunks.append(logits)
        snn_logits = torch.cat(logits_chunks, dim=0)
        if args.return_sequences:
            sequence = torch.cat(sequence_chunks, dim=1)
            return snn_logits, args.time_steps, list(sequence.shape)
        return (
            snn_logits,
            args.time_steps,
            [
                args.time_steps,
                images.shape[0],
                snn_logits.shape[-1],
            ],
        )

    functional.reset_net(converted)
    step_logits = []
    for step in x_seq:
        step_logits.append(converted(step))
    snn_sequence = torch.stack(step_logits, dim=0)
    return snn_sequence.sum(dim=0), args.time_steps, list(snn_sequence.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--imagenet-root")
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--time-steps", type=int, default=64)
    parser.add_argument("--level", type=int, default=32)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--parity-atol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-blocks", action="store_true")
    parser.add_argument("--return-sequences", action="store_true")
    parser.add_argument("--step-mode", choices=("m", "s"), default="m")
    parser.add_argument("--stbif-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--snn-batch-size", type=int)
    parser.add_argument(
        "--sequence-loop-bottlenecks",
        nargs="*",
        choices=("attention", "mlp"),
        default=[],
    )
    parser.add_argument("--allow-namespace-checkpoint", action="store_true")
    parser.add_argument("--print-freq", type=int, default=0)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"--checkpoint {checkpoint_path} does not exist.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {args.device!r} requested but CUDA is not available."
        )
    if (
        args.debug_blocks
        and args.snn_batch_size is not None
        and args.snn_batch_size < args.batch_size
    ):
        raise ValueError("--debug-blocks requires --snn-batch-size >= --batch-size.")
    device = torch.device(args.device)
    model = SpikeZIPViTSmallQANN(level=args.level).eval()
    msg = model.load_state_dict(
        _load_checkpoint(
            checkpoint_path,
            allow_namespace_checkpoint=args.allow_namespace_checkpoint,
        ),
        strict=True,
    )
    model.to(device)
    converted = (
        ModuleConverter(
            recipe=SpikeZIPTFQANNRecipe(
                time_steps=args.time_steps,
                model_family="vit",
            ),
            device=device,
        )
        .convert(model)
        .eval()
    )
    wrapped_bottlenecks = _wrap_sequence_loop_bottlenecks(
        converted,
        args.sequence_loop_bottlenecks,
    )
    functional.set_step_mode(converted, args.step_mode)
    functional.set_backend(converted, args.stbif_backend, instance=STBIFNeuron)

    debug = {}
    qann_features = []
    snn_features = []
    qann_handles = []
    snn_handles = []
    if args.debug_blocks:
        for i, block in enumerate(model.blocks):
            qann_handles.append(
                block.register_forward_hook(
                    lambda _m, _inp, out, index=i: qann_features.append(
                        (index, out.detach().cpu())
                    )
                )
            )
        for i, block in enumerate(converted.blocks):
            snn_handles.append(
                block.register_forward_hook(
                    lambda _m, _inp, out, index=i: snn_features.append(
                        (index, out.detach().cpu())
                    )
                )
            )

    metrics = _MetricAccumulator(args.parity_atol)
    qann_seconds = 0.0
    snn_seconds = 0.0
    cuda_peak_memory_allocated = None
    cuda_peak_memory_reserved = None
    sequence_shape = None
    executed_steps = None
    has_labels = False
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(
            _iter_image_batches(args), start=1
        ):
            images = images.to(device)
            labels = None if labels is None else labels.to(device)
            has_labels = has_labels or labels is not None
            started = time.perf_counter()
            qann_logits = model(images)
            qann_seconds += time.perf_counter() - started
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            snn_logits, executed_steps, sequence_shape = _forward_snn(
                converted,
                images,
                args,
            )
            if device.type == "cuda":
                cuda_peak_memory_allocated = max(
                    cuda_peak_memory_allocated or 0,
                    torch.cuda.max_memory_allocated(device),
                )
                cuda_peak_memory_reserved = max(
                    cuda_peak_memory_reserved or 0,
                    torch.cuda.max_memory_reserved(device),
                )
            snn_seconds += time.perf_counter() - started
            metrics.update(qann_logits, snn_logits, labels)
            if args.print_freq > 0 and batch_index % args.print_freq == 0:
                print(
                    json.dumps(
                        {
                            "batch": batch_index,
                            "samples": metrics.samples,
                            "qann_seconds": qann_seconds,
                            "snn_seconds": snn_seconds,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    for handle in qann_handles + snn_handles:
        handle.remove()
    if args.debug_blocks:
        qann_outputs = _debug_qann_outputs(qann_features)
        snn_outputs = _debug_snn_outputs(snn_features, args.time_steps)
        debug["debug_batches"] = batch_index if "batch_index" in locals() else 0
        debug["block_max_abs_diff"] = _debug_block_diffs(qann_outputs, snn_outputs)

    result = {
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "allow_namespace_checkpoint": args.allow_namespace_checkpoint,
        "load_state_dict": str(msg),
        "device": str(device),
        "time_steps": args.time_steps,
        "executed_steps": executed_steps,
        "step_mode": args.step_mode,
        "stbif_backend": args.stbif_backend,
        "sequence_loop_bottlenecks": args.sequence_loop_bottlenecks,
        "wrapped_bottlenecks": wrapped_bottlenecks,
        "level": args.level,
        "qann_seconds": qann_seconds,
        "snn_seconds": snn_seconds,
        "parity_atol": args.parity_atol,
        "samples_requested": args.samples,
        "batch_size": args.batch_size,
        "snn_batch_size": args.snn_batch_size,
        "sequence_shape": sequence_shape,
    }
    if cuda_peak_memory_allocated is not None:
        result["cuda_peak_memory_allocated_mb"] = cuda_peak_memory_allocated / 1024**2
        result["cuda_peak_memory_reserved_mb"] = cuda_peak_memory_reserved / 1024**2
    result.update(metrics.result(has_labels))
    if debug:
        result["debug"] = debug

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
