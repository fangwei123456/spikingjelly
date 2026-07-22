"""Qwen2.5-3B public-recipe two-rank TDLinear tensor-parallel smoke."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import MethodType
from typing import Dict, List, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmark.snn_llm._reporting import write_rank0_report as _write_rank0_report
from spikingjelly.activation_based import distributed, functional
from spikingjelly.activation_based.ann2snn import (
    ModuleConverter,
    Qwen2SNNConfig,
    Qwen2SNNRecipe,
)
from spikingjelly.activation_based.precision import (
    PrecisionConfig,
    prepare_model_for_precision,
)

from benchmark.snn_llm.qwen_conversion._runtime import (
    ARTIFACT_LOCK,
    FIXED_PROMPTS,
    build_environment,
    cached_decode as _cached_decode,
    encode as _encode,
    hash_files as _hash_files,
    load_calibration as _load_calibration,
    load_lock as _load_lock,
    load_model as _load_model,
    relative_l2 as _relative_l2,
    shifted_loss as _loss,
    validate_calibration_config as _validate_calibration_config,
)


SCHEMA_VERSION = 1
CONTRACT_KIND = "qwen2-snn-scaleout-tensor-parallel"
EXPECTED_WORLD_SIZE = 2
TIME_STEP_CHOICES = (16, 32, 64, 128, 160, 192, 256, 512)


def _qwen_tdlinear_tp_plan(layer_count: int) -> Dict[str, str]:
    plan: Dict[str, str] = {}
    for index in range(layer_count):
        prefix = f"layers.{index}"
        for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            plan[f"{prefix}.{name}"] = "td_colwise_replicated"
        for name in ("o_proj", "down_proj"):
            plan[f"{prefix}.{name}"] = "td_rowwise_replicated"
    return plan


def _validate_distributed_environment() -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen tensor parallelism requires CUDA.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"Qwen tensor parallelism requires world_size=2, got {world_size}."
        )
    if os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise RuntimeError("g-series multi-GPU runs require NCCL_P2P_DISABLE=1.")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank} does not select a visible GPU.")
    return world_size, local_rank


def _storage_summary(
    model: torch.nn.Module, explicit_plan: Mapping[str, str]
) -> Dict[str, object]:
    local_bytes = 0
    global_bytes = 0
    sharded_modules = []
    for parameter in model.parameters():
        global_bytes += parameter.numel() * parameter.element_size()
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        local_bytes += local.numel() * local.element_size()
    for name in explicit_plan:
        weight = getattr(model.get_submodule(name), "weight", None)
        placements = getattr(weight, "placements", ())
        if any(type(placement).__name__ == "Shard" for placement in placements):
            sharded_modules.append(name)
    return {
        "local_parameter_storage_bytes": local_bytes,
        "single_device_parameter_bytes": global_bytes,
        "local_to_single_device_ratio": local_bytes / global_bytes,
        "sharded_module_names": sharded_modules,
    }


def _materialize_full_tensor(value):
    if not hasattr(value, "full_tensor"):
        return value
    # PyTorch 2.7 Parameter(DTensor).full_tensor() dispatches aten.detach_, which
    # has no DTensor sharding rule. This runner is frozen and inference-only.
    source = value.data if getattr(value, "_is_param", False) else value
    return source.full_tensor()


def _install_gathered_rowwise_execution(
    model: torch.nn.Module, explicit_plan: Mapping[str, str]
) -> None:
    def ann_forward(self, value):
        weight = _materialize_full_tensor(self.weight)
        bias = self.bias
        if bias is not None:
            bias = _materialize_full_tensor(bias)
        return F.linear(value, weight, bias)

    for name, style in explicit_plan.items():
        if style == "td_rowwise_replicated":
            module = model.get_submodule(name)
            if hasattr(module.weight, "full_tensor") and not dist.is_initialized():
                raise RuntimeError(
                    "Gathered rowwise DTensor execution requires initialized distributed ranks."
                )
            # Every rank installs and invokes the same collectives in model order.
            module.ann_forward = MethodType(ann_forward, module)


def _model_output(model, input_ids, attention_mask, autocast_context, mode):
    functional.reset_net(model)
    with torch.inference_mode(), autocast_context():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoding_mode=mode,
        ).logits
    functional.reset_net(model)
    with torch.inference_mode(), autocast_context():
        replay = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoding_mode=mode,
        ).logits
    cached = _cached_decode(
        model,
        input_ids[:1, :16],
        mode=mode,
        autocast_context=autocast_context,
    )
    return {
        "logits": logits.detach().float().cpu(),
        "loss": _loss(logits, input_ids, attention_mask),
        "cache_relative_l2": cached["max_relative_l2"],
        "token_ids": cached["new_token_ids"],
        "reset_error": float((replay - logits).abs().max()),
    }


def _compare_outputs(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    label: str,
) -> Dict[str, object]:
    reference_logits = torch.as_tensor(reference["logits"])
    candidate_logits = torch.as_tensor(candidate["logits"])
    if reference["token_ids"] != candidate["token_ids"]:
        raise ValueError("Tensor-parallel generation token IDs differ from reference.")
    metrics = {
        "logits_relative_l2": _relative_l2(candidate_logits, reference_logits),
        "loss_delta": abs(float(candidate["loss"]) - float(reference["loss"])),
        "cache_relative_l2": float(candidate["cache_relative_l2"]),
        "cache_relative_l2_delta": abs(
            float(candidate["cache_relative_l2"])
            - float(reference["cache_relative_l2"])
        ),
        "reset_error": float(candidate["reset_error"]),
        "generation_token_ids_equal": True,
    }
    if not math.isfinite(float(metrics["cache_relative_l2_delta"])):
        raise ValueError("Tensor-parallel cache relative-L2 delta must be finite.")
    limits = {
        "logits_relative_l2": 0.02,
        "loss_delta": 0.05,
        "cache_relative_l2": 0.20,
        "reset_error": 1e-6,
    }
    for name, limit in limits.items():
        value = float(metrics[name])
        if not math.isfinite(value) or value > limit:
            raise ValueError(
                f"Tensor-parallel {label} {name}={value!r} exceeds {limit!r}."
            )
    return metrics


def _signed_quality(
    signed: Mapping[str, object], exact: Mapping[str, object]
) -> Dict[str, object]:
    signed_logits = torch.as_tensor(signed["logits"])
    exact_logits = torch.as_tensor(exact["logits"])
    metrics = {
        "logits_relative_l2": _relative_l2(signed_logits, exact_logits),
        "loss_delta": abs(float(signed["loss"]) - float(exact["loss"])),
        "top1_agreement": float(
            (signed_logits.argmax(-1) == exact_logits.argmax(-1)).float().mean()
        ),
        "cache_relative_l2": float(signed["cache_relative_l2"]),
        "reset_error": float(signed["reset_error"]),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("TP signed quality metrics must be finite.")
    if metrics["logits_relative_l2"] > 0.50:
        raise ValueError("TP signed logits relative L2 exceeds 0.50.")
    if metrics["loss_delta"] > 0.75:
        raise ValueError("TP signed loss delta exceeds 0.75.")
    if metrics["top1_agreement"] < 0.50:
        raise ValueError("TP signed top-1 agreement is below 0.50.")
    if metrics["cache_relative_l2"] > 0.20:
        raise ValueError("TP signed cache relative L2 exceeds 0.20.")
    if metrics["reset_error"] > 1e-6:
        raise ValueError("TP signed reset error exceeds 1e-6.")
    return metrics


def _run(args: argparse.Namespace) -> tuple[Dict[str, object], int]:
    world_size, local_rank = _validate_distributed_environment()
    torch.cuda.set_device(local_rank)
    distributed.ensure_distributed_initialized(backend="nccl")
    rank = dist.get_rank()
    device = f"cuda:{local_rank}"
    lock = _load_lock()
    record = lock["models"]["3b"]
    tokenizer, source = _load_model(args.model_root, device)
    if sum(parameter.numel() for parameter in source.parameters()) != int(
        record["parameter_count"]
    ):
        raise ValueError("Loaded Qwen2.5-3B parameter count does not match lock.")
    if args.precision == "fp32":
        source.float()
    precision = prepare_model_for_precision(
        source,
        "cuda",
        PrecisionConfig(
            mode=args.precision, strictness="strict", report=True, device="cuda"
        ),
    )
    calibration, calibration_sha256 = _load_calibration(args.calibration_artifact)
    _validate_calibration_config(
        calibration,
        time_steps=args.time_steps,
        calibration_levels=args.calibration_levels,
        calibration_quantile=args.calibration_quantile,
        calibration_reservoir_size=args.calibration_reservoir_size,
        calibration_seed=20260719,
    )
    config = Qwen2SNNConfig(
        time_steps=args.time_steps,
        calibration_levels=args.calibration_levels,
        calibration_quantile=args.calibration_quantile,
        calibration_reservoir_size=args.calibration_reservoir_size,
        neuron_backend="triton",
    )
    converted = ModuleConverter(Qwen2SNNRecipe(calibration, config)).convert(
        precision.model
    )
    if args.precision == "fp32":
        converted.float()
    input_ids, attention_mask = _encode(tokenizer, [FIXED_PROMPTS[2]], device)
    autocast_context = precision.autocast_context
    reference_signed = _model_output(
        converted, input_ids, attention_mask, autocast_context, "signed_if"
    )
    reference_exact = _model_output(
        converted, input_ids, attention_mask, autocast_context, "exact_td"
    )
    del source, precision
    gc.collect()
    torch.cuda.empty_cache()

    layer_count = int(record["layer_count"])
    explicit_plan = _qwen_tdlinear_tp_plan(layer_count)
    analysis = distributed.analyze(converted)
    execution_plan = distributed.plan(
        analysis=analysis,
        objective="memory",
        topology={"tp": world_size},
        backend="torch",
        batch_size=1,
        model_family="generic",
        mode="tp",
        tensor_parallel_plan=explicit_plan,
    )
    runtime = distributed.apply(
        model=converted, plan=execution_plan, device_type="cuda"
    )
    candidate_exact = _model_output(
        runtime.model, input_ids, attention_mask, autocast_context, "exact_td"
    )
    if args.rowwise_execution == "gathered-exact":
        _install_gathered_rowwise_execution(runtime.model, explicit_plan)
    candidate_signed = _model_output(
        runtime.model, input_ids, attention_mask, autocast_context, "signed_if"
    )
    metrics = {
        "exact_sharding": _compare_outputs(
            reference_exact, candidate_exact, label="exact sharding"
        ),
        "signed_sharding": _compare_outputs(
            reference_signed, candidate_signed, label="signed sharding"
        ),
        "tp_signed_quality": _signed_quality(candidate_signed, candidate_exact),
    }
    storage = _storage_summary(runtime.model, explicit_plan)
    if len(storage["sharded_module_names"]) != len(explicit_plan):
        raise ValueError("Not all explicit Qwen TDLinear modules were sharded.")
    if float(storage["local_to_single_device_ratio"]) > 0.70:
        raise ValueError(
            "Local TP parameter storage exceeds 70% of single-device bytes."
        )
    local = {
        "rank": rank,
        "device": device,
        "metrics": metrics,
        "storage": storage,
        "token_ids": candidate_signed["token_ids"],
    }
    gathered: List[Optional[Dict[str, object]]] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if any(value is None for value in gathered):
        raise RuntimeError("Tensor-parallel rank gather was incomplete.")
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "source": {
            "worktree_revision": args.worktree_revision,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "artifact_lock_sha256": hashlib.sha256(
                ARTIFACT_LOCK.read_bytes()
            ).hexdigest(),
            "model_files": _hash_files(args.model_root),
        },
        "model": {"key": "3b", **record},
        "environment": {
            **build_environment("cuda"),
            "backend": dist.get_backend(),
            "world_size": world_size,
            "nccl_p2p_disable": os.environ["NCCL_P2P_DISABLE"],
        },
        "configuration": {
            "mode": "tp",
            "precision": args.precision,
            "time_steps": args.time_steps,
            "calibration_levels": args.calibration_levels,
            "calibration_quantile": args.calibration_quantile,
            "explicit_tdlinear_shards": len(explicit_plan),
            "colwise_shards": 5 * layer_count,
            "rowwise_shards": 2 * layer_count,
            "activation_layout": "replicated",
            "exact_rowwise_execution": "distributed",
            "signed_rowwise_execution": args.rowwise_execution,
            "rowwise_compute_speedup_claimed": False,
        },
        "conversion": {
            "temporal_layout": runtime.model.temporal_layout,
            "execution_schedule": runtime.model.execution_schedule,
            "online_inference": runtime.model.online_inference,
            "calibration_sha256": calibration_sha256,
        },
        "ranks": gathered,
        "acceptance": {"passed": True},
    }
    json.dumps(report, allow_nan=False)
    return report, rank


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--calibration-artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda",))
    parser.add_argument("--worktree-revision", required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--time-steps", required=True, type=int, choices=TIME_STEP_CHOICES
    )
    parser.add_argument("--calibration-levels", required=True, type=int)
    parser.add_argument("--calibration-quantile", required=True, type=float)
    parser.add_argument("--calibration-reservoir-size", type=int, default=4096)
    parser.add_argument(
        "--rowwise-execution",
        choices=("distributed", "gathered-exact"),
        default="gathered-exact",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = _parse_args(argv)
        if args.calibration_levels > args.time_steps:
            raise ValueError("calibration-levels must not exceed time-steps.")
        if (args.output_dir / "report.json").exists():
            raise FileExistsError(
                f"Refusing to overwrite {args.output_dir / 'report.json'}."
            )
        report, rank = _run(args)
        path = _write_rank0_report(args.output_dir, report, rank=rank)
        if path is not None:
            print(f"report_path={path.resolve()}")
        dist.barrier()
        dist.destroy_process_group()
        return 0
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        if dist.is_initialized():
            dist.destroy_process_group()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
