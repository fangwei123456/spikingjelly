import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from benchmark.snn_llm import _spikegpt_pilot as pilot
from benchmark.snn_llm import _spikegpt_training as smoke
from spikingjelly.activation_based import distributed as sj_distributed
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.precision import (
    PrecisionConfig,
    prepare_model_for_precision,
)


VALIDATION_SAMPLE_METRIC = "validation_sample_bpc"
VALIDATION_SAMPLE_OFFSETS = (0, 2_490_694)
SUPPORTED_PRECISIONS = ("fp32", "bf16", "fp16", "fp8-probe")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 4 SpikeGPT distributed/precision smoke."
    )
    parser.add_argument("--spikegpt-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--precision", choices=SUPPORTED_PRECISIONS, required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def _rank_env() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def _log_stage(rank: int, message: str) -> None:
    print(f"rank={rank} stage={message}", flush=True)


def _require_distributed_training_world(*, world_size: int, precision: str) -> None:
    if precision != "fp8-probe" and world_size < 2:
        raise RuntimeError("Phase 4 distributed training requires world size >= 2.")


def _training_config(precision: str, world_size: int) -> dict:
    return {
        "kind": "spikegpt-enwik8-phase4-distributed-precision-v1",
        "backend": "cupy",
        "batch_size": pilot.BATCH_SIZE,
        "betas": smoke.BETAS,
        "context_length": pilot.CONTEXT_LENGTH,
        "data_md5": smoke.EXPECTED_DATA_MD5,
        "distributed_mode": "dp",
        "eps": smoke.EPS,
        "grad_norm_clip": smoke.GRAD_NORM_CLIP,
        "learning_rate": smoke.LEARNING_RATE,
        "model_type": pilot.MODEL_TYPE,
        "n_embd": pilot.N_EMBD,
        "n_layer": pilot.N_LAYER,
        "parameter_count": pilot.FULL_ENWIK8_PARAMETER_COUNT,
        "precision": precision,
        "reset": "after-forward-before-backward",
        "seed": smoke.SEED,
        "split": "enwik8-byte-90/5/5",
        "tokenizer": "utf8-character-sorted-full-enwik8-v1",
        "validation_metric": VALIDATION_SAMPLE_METRIC,
        "validation_offsets": VALIDATION_SAMPLE_OFFSETS,
        "vocab_size": pilot.FULL_ENWIK8_VOCAB_SIZE,
        "world_size": world_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _assert_checkpoint_unchanged(path: Path, expected: dict) -> None:
    actual = _checkpoint_identity(path)
    if actual != expected:
        raise RuntimeError(
            f"Resume source checkpoint changed: expected {expected}, got {actual}."
        )


def _rank_zero_paths(output_dir: Path, rank: int) -> tuple[Path | None, Path | None]:
    if rank != 0:
        return None, None
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    report = output_dir / "report.json"
    if checkpoint.exists():
        raise RuntimeError(f"Checkpoint exists; refusing to overwrite: {checkpoint}")
    if report.exists():
        raise RuntimeError(f"Report exists; refusing to overwrite: {report}")
    return checkpoint, report


def _rank_zero_report_path(output_dir: Path, rank: int) -> Path | None:
    if rank != 0:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    report = output_dir / "report.json"
    if checkpoint.exists():
        raise RuntimeError(f"Checkpoint exists in probe output: {checkpoint}")
    if report.exists():
        raise RuntimeError(f"Report exists; refusing to overwrite: {report}")
    return report


def _write_rank_zero_json(path: Path, payload: dict, rank: int) -> None:
    if rank != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _ensure_not_ddp(
    model: torch.nn.Module, ddp_type: type = DistributedDataParallel
) -> None:
    if isinstance(model, ddp_type):
        raise RuntimeError("Double DDP wrapping is not allowed.")


def _ensure_single_scaler(artifacts, external_scaler=None) -> None:
    if external_scaler is not None and artifacts.scaler is not None:
        raise RuntimeError("Double scaler ownership is not allowed.")


def _validate_lif_provenance(lif_nodes: list[torch.nn.Module]) -> tuple[str, ...]:
    modules = tuple(type(lif).__module__ for lif in lif_nodes)
    if any(module.startswith("src.spikingjelly") for module in modules):
        raise RuntimeError("Phase 4 runner used vendored SpikingJelly LIF nodes.")
    if not all(
        module.startswith("spikingjelly.activation_based") for module in modules
    ):
        raise RuntimeError(f"Unexpected LIF provenance: {modules}.")
    return modules


class _AutocastFp32LIF(torch.nn.Module):
    def __init__(self, lif: torch.nn.Module) -> None:
        super().__init__()
        self.lif = lif

    @property
    def v(self):
        return self.lif.v

    @property
    def v_seq(self):
        return self.lif.v_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            y = self.lif(x.float())
        return y.to(dtype=input_dtype)


def _wrap_lif_for_mixed_precision(model: torch.nn.Module) -> None:
    for block in model.blocks:
        block.lif1 = _AutocastFp32LIF(block.lif1)
        block.lif2 = _AutocastFp32LIF(block.lif2)


def _make_model(
    author_model, vocabulary_size: int, device: torch.device, precision: str = "fp32"
):
    model = pilot._make_model(author_model, vocabulary_size, device)
    if precision in {"bf16", "fp16"}:
        _wrap_lif_for_mixed_precision(model)
    _validate_lif_provenance(_lif_nodes(model))
    return model


def _precision_mode(precision: str) -> str:
    return "fp8-torchao" if precision == "fp8-probe" else precision


def _prepare_precision(model, device: torch.device, precision: str):
    config = PrecisionConfig(
        mode=_precision_mode(precision),
        strictness="warn" if precision == "fp8-probe" else "strict",
        device=str(device),
    )
    return prepare_model_for_precision(model, device, config)


def _make_distributed_runtime(model, world_size: int):
    analysis = sj_distributed.analyze(model, model_family="spikegpt")
    plan = sj_distributed.plan(
        analysis=analysis,
        objective="speed",
        topology={"dp": world_size},
        backend="cupy",
        batch_size=pilot.BATCH_SIZE,
        model_family="spikegpt",
        mode="dp",
    )
    return sj_distributed.apply(model=model, plan=plan, device_type="cuda")


def _optimizer_parameters(model: torch.nn.Module):
    return _unwrap_model(model).parameters()


def _backward_step(loss, artifacts, optimizer, model) -> float:
    parameters = list(_optimizer_parameters(model))
    scaler = artifacts.scaler
    if scaler is None:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, smoke.GRAD_NORM_CLIP)
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, smoke.GRAD_NORM_CLIP)
        scaler.step(optimizer)
        scaler.update()
    return float(grad_norm)


def _states_are_reset(model: torch.nn.Module) -> bool:
    return smoke._states_are_reset(_unwrap_model(model))


def _lif_nodes(model: torch.nn.Module) -> list[torch.nn.Module]:
    nodes = []
    for lif in smoke._lif_nodes(_unwrap_model(model)):
        nodes.append(lif.lif if isinstance(lif, _AutocastFp32LIF) else lif)
    return nodes


def _batch_with_vocab(
    text: str, vocabulary: tuple[str, ...], offset: int, device: torch.device
):
    return pilot._batch(text, vocabulary, offset, device)


def _validation_sample_bpc(
    model: torch.nn.Module,
    validation: str,
    vocabulary: tuple[str, ...],
    device: torch.device,
) -> float:
    target = _unwrap_model(model)
    target.eval()
    losses = []
    for offset in VALIDATION_SAMPLE_OFFSETS:
        inputs, targets = _batch_with_vocab(validation, vocabulary, offset, device)
        functional.reset_net(target)
        with torch.no_grad():
            loss = target(inputs, targets)
        functional.reset_net(target)
        losses.append(float(loss.item()))
    value = sum(losses) / len(losses) / math.log(2.0)
    if not math.isfinite(value):
        raise RuntimeError("Validation sample BPC is not finite.")
    return value


def _train_step(model, artifacts, optimizer, inputs, targets) -> dict:
    spike_sum = 0.0
    spike_count = 0

    def capture_spikes(module, module_inputs, output):
        nonlocal spike_sum, spike_count
        spike_sum += output.detach().float().sum().item()
        spike_count += output.numel()

    lif_nodes = _lif_nodes(model)
    handles = [lif.register_forward_hook(capture_spikes) for lif in lif_nodes]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with artifacts.autocast_context():
            loss = model(inputs, targets)
    finally:
        for handle in handles:
            handle.remove()
    voltages = [lif.v_seq.detach().float() for lif in lif_nodes]
    membrane_abs_sum = sum(value.abs().sum().item() for value in voltages)
    membrane_count = sum(value.numel() for value in voltages)
    membrane_abs_max = max(value.abs().max().item() for value in voltages)
    functional.reset_net(_unwrap_model(model))
    reset_ok = _states_are_reset(model)
    gradient_l2 = _backward_step(loss, artifacts, optimizer, model)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    result = {
        "loss": float(loss.item()),
        "gradient_l2": gradient_l2,
        "latency_ms": latency_ms,
        "spike_rate": spike_sum / spike_count,
        "membrane_abs_mean": membrane_abs_sum / membrane_count,
        "membrane_abs_max": membrane_abs_max,
        "reset_ok": reset_ok,
    }
    if (
        not smoke._training_metrics_are_finite(
            loss=result["loss"],
            gradient_l2=result["gradient_l2"],
            spike_rate=result["spike_rate"],
            membrane_abs_mean=result["membrane_abs_mean"],
            membrane_abs_max=result["membrane_abs_max"],
        )
        or not reset_ok
    ):
        raise RuntimeError(f"Distributed training step failed: {result}.")
    return result


def _all_gather_objects(value):
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def _all_reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def _training_direction(losses: list[float]) -> tuple[bool, float, float]:
    window_size = min(pilot.WINDOW_SIZE, len(losses) // 2)
    return pilot._training_direction(losses, window_size)


def _gpu_before_snapshot() -> list[str]:
    path = os.environ.get("SNN_PHASE4_GPU_BEFORE")
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8").splitlines()
    return []


def _build_report(
    *,
    source_revision: str,
    precision_requested: str,
    precision_effective: str,
    precision_report: dict,
    distributed_report: dict,
    experiment: dict,
    ranks: list[dict],
    metrics: dict,
    checkpoint: dict,
    gpu_before: list[str],
) -> dict:
    return {
        "source": {"spikegpt_revision": source_revision},
        "environment": {"hostname": socket.gethostname()},
        "distributed": distributed_report,
        "precision": {
            "requested": precision_requested,
            "effective": precision_effective,
            **precision_report,
        },
        "experiment": experiment,
        "ranks": ranks,
        "metrics": {
            **metrics,
            "performance": "shared_gpu_smoke_measurement",
        },
        "checkpoint": checkpoint,
        "gpu_before": gpu_before,
    }


def _save_checkpoint(
    path, model, optimizer, completed_steps, revision, vocabulary, config
):
    smoke._save_checkpoint(
        path,
        _unwrap_model(model),
        optimizer,
        completed_steps=completed_steps,
        next_batch_index=completed_steps,
        source_revision=revision,
        vocabulary=vocabulary,
        training_config=config,
    )


def _load_checkpoint(path, model, optimizer, device, revision, vocabulary, config):
    restored = smoke._load_checkpoint(
        path, _unwrap_model(model), optimizer, device, training_config=config
    )
    if restored["source_revision"] != revision or restored["vocabulary"] != vocabulary:
        raise RuntimeError("Phase 4 checkpoint source or vocabulary mismatch.")
    if restored["next_batch_index"] != restored["completed_steps"]:
        raise RuntimeError("Phase 4 checkpoint step and batch counters disagree.")
    return restored


def _run_fp8_probe(args, rank: int, local_rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{local_rank}")
    root = args.spikegpt_root.resolve()
    _log_stage(rank, "verify-source")
    revision = smoke._verify_source(root, args.source_revision)
    _log_stage(rank, "read-data")
    _, _, vocabulary = pilot._read_data(args.data.resolve())
    _log_stage(rank, "load-author-model")
    author_model = smoke._load_author_model(root, require_wkv=True)
    _log_stage(rank, "build-model")
    model = _make_model(author_model, len(vocabulary), device, precision="fp8-probe")
    _log_stage(rank, "prepare-precision")
    artifacts = _prepare_precision(model, device, "fp8-probe")
    _ensure_single_scaler(artifacts)
    report = _build_report(
        source_revision=revision,
        precision_requested="fp8-probe",
        precision_effective=artifacts.effective_config.mode,
        precision_report=artifacts.describe(),
        distributed_report={
            "world_size": world_size,
            "mode": "probe",
            "backend": "none",
        },
        experiment={
            "vocab_size": pilot.FULL_ENWIK8_VOCAB_SIZE,
            "parameter_count": pilot.FULL_ENWIK8_PARAMETER_COUNT,
            "spikingjelly_impl": "current-worktree",
            "author_wkv": "SpikeGPT-029f86f",
            "lif_backend": "current-cupy",
            "lif_modules": list(_validate_lif_provenance(smoke._lif_nodes(model))),
        },
        ranks=[{"rank": rank, "local_rank": local_rank}],
        metrics={"trained": False},
        checkpoint={},
        gpu_before=_gpu_before_snapshot(),
    )
    if rank == 0:
        report_path = _rank_zero_report_path(args.output_dir, rank=0)
        _write_rank_zero_json(report_path, report, rank=0)


def run(args: argparse.Namespace) -> None:
    rank, local_rank, world_size = _rank_env()
    _require_distributed_training_world(world_size=world_size, precision=args.precision)
    if args.max_steps <= 0 or args.checkpoint_every <= 0:
        raise RuntimeError("Phase 4 resource limits must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 4 SpikeGPT smoke.")

    torch.cuda.set_device(local_rank)
    if args.precision == "fp8-probe":
        _run_fp8_probe(args, rank, local_rank, world_size)
        return

    dist.init_process_group(backend="nccl")
    try:
        device = torch.device(f"cuda:{local_rank}")
        root = args.spikegpt_root.resolve()
        _log_stage(rank, "verify-source")
        revision = smoke._verify_source(root, args.source_revision)
        checkpoint_path, report_path = _rank_zero_paths(args.output_dir.resolve(), rank)
        source_identity = (
            _checkpoint_identity(args.resume_checkpoint.resolve())
            if args.resume_checkpoint is not None and rank == 0
            else None
        )
        _log_stage(rank, "read-data")
        training, validation, vocabulary = pilot._read_data(args.data.resolve())
        _log_stage(rank, "load-author-model")
        author_model = smoke._load_author_model(root, require_wkv=True)
        smoke._seed_everything()
        _log_stage(rank, "build-model")
        model = _make_model(
            author_model, len(vocabulary), device, precision=args.precision
        )
        _log_stage(rank, "prepare-precision")
        artifacts = _prepare_precision(model, device, args.precision)
        _ensure_single_scaler(artifacts)
        _ensure_not_ddp(artifacts.model)
        _log_stage(rank, "apply-distributed-dp")
        runtime = _make_distributed_runtime(artifacts.model, world_size)
        model = runtime.model
        _log_stage(rank, "build-optimizer")
        optimizer = smoke._make_optimizer(_unwrap_model(model))
        config = _training_config(args.precision, world_size)
        completed_steps = 0
        if args.resume_checkpoint is not None:
            _log_stage(rank, "load-resume-checkpoint")
            restored = _load_checkpoint(
                args.resume_checkpoint.resolve(),
                model,
                optimizer,
                device,
                revision,
                vocabulary,
                config,
            )
            completed_steps = restored["completed_steps"]
        if args.max_steps - completed_steps < 2:
            raise RuntimeError("Phase 4 smoke requires at least two remaining steps.")

        _log_stage(rank, "validation-sample-before")
        validation_before = _validation_sample_bpc(
            model, validation, vocabulary, device
        )
        _log_stage(rank, "train-loop")
        maximum_offset = len(training) - pilot.CONTEXT_LENGTH - 1
        losses = []
        results = []
        while completed_steps < args.max_steps:
            offset = int(
                np.random.default_rng(smoke.SEED + rank + completed_steps * world_size)
                .integers(0, maximum_offset)
                .item()
            )
            inputs, targets = _batch_with_vocab(training, vocabulary, offset, device)
            result = _train_step(model, artifacts, optimizer, inputs, targets)
            completed_steps += 1
            losses.append(result["loss"])
            results.append(result)
            print(
                f"rank={rank} step={completed_steps} loss={result['loss']:.8f} "
                f"gradient_l2={result['gradient_l2']:.8f}",
                flush=True,
            )
            if rank == 0 and (
                len(losses) == 1 or completed_steps % args.checkpoint_every == 0
            ):
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    completed_steps,
                    revision,
                    vocabulary,
                    config,
                )

        _log_stage(rank, "validation-sample-after")
        validation_after = _validation_sample_bpc(model, validation, vocabulary, device)
        if rank == 0:
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                completed_steps,
                revision,
                vocabulary,
                config,
            )
        if args.resume_checkpoint is not None and rank == 0:
            _assert_checkpoint_unchanged(
                args.resume_checkpoint.resolve(), source_identity
            )
        improved, start_bpc, end_bpc = _training_direction(losses)
        rank_summary = {
            "rank": rank,
            "local_rank": local_rank,
            "losses": losses,
            "training_start_bpc": start_bpc,
            "training_end_bpc": end_bpc,
            "training_improved": improved,
            VALIDATION_SAMPLE_METRIC: {
                "before": validation_before,
                "after": validation_after,
            },
            "last_step": results[-1],
        }
        ranks = _all_gather_objects(rank_summary)
        mean_loss_before = _all_reduce_mean(losses[0], device)
        mean_loss_after = _all_reduce_mean(losses[-1], device)
        if rank == 0:
            checkpoint = _checkpoint_identity(checkpoint_path)
            report = _build_report(
                source_revision=revision,
                precision_requested=args.precision,
                precision_effective=artifacts.effective_config.mode,
                precision_report={
                    "scaler_owner": (
                        "spikingjelly.precision"
                        if artifacts.scaler is not None
                        else "none"
                    ),
                    "artifacts": artifacts.describe(),
                },
                distributed_report={
                    "world_size": world_size,
                    "backend": "nccl",
                    "mode": runtime.plan.mode if runtime.plan else "dp",
                    "ddp_owner": "torch.distributed",
                    "mesh": str(runtime.mesh),
                },
                experiment={
                    "vocab_size": pilot.FULL_ENWIK8_VOCAB_SIZE,
                    "parameter_count": pilot.FULL_ENWIK8_PARAMETER_COUNT,
                    "spikingjelly_impl": "current-worktree",
                    "author_wkv": "SpikeGPT-029f86f",
                    "lif_backend": "current-cupy",
                    "lif_modules": list(_validate_lif_provenance(_lif_nodes(model))),
                },
                ranks=ranks,
                metrics={
                    "mean_loss_before": mean_loss_before,
                    "mean_loss_after": mean_loss_after,
                    "training_improved": all(
                        item["training_improved"] for item in ranks
                    ),
                },
                checkpoint=checkpoint,
                gpu_before=_gpu_before_snapshot(),
            )
            _write_rank_zero_json(report_path, report, rank=0)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    args = _parse_args()
    try:
        run(args)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
