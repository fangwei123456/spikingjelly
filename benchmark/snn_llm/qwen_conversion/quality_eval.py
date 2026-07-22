"""Qwen2.5 dense/SNN WikiText-2 and zero-shot quality evaluation."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F

from benchmark.snn_llm._reporting import write_report as _write_report
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import (
    ModuleConverter,
    Qwen2SNNCalibration,
    Qwen2SNNConfig,
    Qwen2SNNRecipe,
    calibrate_qwen2_snn,
)
from spikingjelly.activation_based.precision import (
    PrecisionConfig,
    prepare_model_for_precision,
)

from benchmark.snn_llm.qwen_conversion._quality import (
    TASKS,
    validate_quality as _validate_quality,
)
from benchmark.snn_llm.qwen_conversion._runtime import (
    ARTIFACT_LOCK,
    build_environment,
    hash_files as _hash_files,
    load_calibration as _load_calibration,
    load_lock as _load_lock,
    load_model as _load_model,
    validate_calibration_config as _validate_calibration_config,
)


SCHEMA_VERSION = 1
CONTRACT_KIND = "qwen2-snn-paper-quality"
PRIMARY_METRICS = {
    "lambada_openai": "acc",
    "piqa": "acc_norm",
    "hellaswag": "acc_norm",
    "winogrande": "acc",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
}
TASK_DATASET_KEYS = {
    "lambada_openai": "lambada_openai",
    "piqa": "piqa",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "arc_easy": "arc",
    "arc_challenge": "arc",
}
if set(TASK_DATASET_KEYS) != set(TASKS):
    raise RuntimeError("Every fixed zero-shot task requires one dataset lock key.")
CALIBRATION_WINDOW = 512
CALIBRATION_WINDOWS = 128
PPL_CONTEXT = 2048
PPL_STRIDE = 512
PPL_CACHE_CHUNK = 128
LM_EVAL_VERSION = "0.4.12"
STATISTICS_PROBE_TEXT = "SpikingJelly Qwen SNN quality statistics probe."
RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DISTRIBUTED_TASK_TIMEOUT = timedelta(hours=24)


def _require_lm_eval_version() -> str:
    try:
        actual = metadata.version("lm_eval")
    except metadata.PackageNotFoundError as exc:
        raise ImportError("Qwen quality evaluation requires lm-eval.") from exc
    if actual != LM_EVAL_VERSION:
        raise RuntimeError(
            f"Qwen quality evaluation requires lm-eval=={LM_EVAL_VERSION}, got {actual}."
        )
    return actual


def _validate_distributed_task_args(args: argparse.Namespace) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive.")
    if world_size == 1:
        return 1
    if not args.skip_ppl or not args.run_tasks:
        raise ValueError("Distributed quality runs support task-only evaluation.")
    if args.calibration_artifact is None:
        raise ValueError("Distributed task evaluation requires --calibration-artifact.")
    if os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise ValueError("g-series distributed runs require NCCL_P2P_DISABLE=1.")
    return world_size


def _initialize_distributed_task_device(args: argparse.Namespace) -> tuple[str, int]:
    world_size = _validate_distributed_task_args(args)
    if world_size == 1:
        return args.device, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError(f"LOCAL_RANK={local_rank} does not select a visible GPU.")
    torch.cuda.set_device(local_rank)
    if not torch_dist.is_initialized():
        torch_dist.init_process_group(
            backend="nccl",
            timeout=DISTRIBUTED_TASK_TIMEOUT,
        )
    return f"cuda:{local_rank}", torch_dist.get_rank()


@contextmanager
def _pin_dataset_revisions(lock: Mapping[str, Mapping[str, object]]):
    import datasets

    original = datasets.load_dataset
    records = []

    def load_dataset(*args, **kwargs):
        repository = args[0] if args else kwargs.get("path")
        matches = [
            value for value in lock.values() if value["repository"] == repository
        ]
        if matches:
            kwargs["revision"] = matches[0]["revision"]
        result = original(*args, **kwargs)
        if matches:
            if hasattr(result, "items"):
                fingerprints = {
                    str(name): getattr(split, "_fingerprint", None)
                    for name, split in result.items()
                }
            else:
                fingerprints = {"selected": getattr(result, "_fingerprint", None)}
            if not all(
                isinstance(value, str) and value for value in fingerprints.values()
            ):
                raise RuntimeError(
                    f"Dataset {repository!r} did not expose stable fingerprints."
                )
            records.append(
                {
                    "repository": repository,
                    "revision": matches[0]["revision"],
                    "fingerprints": fingerprints,
                }
            )
        return result

    datasets.load_dataset = load_dataset
    try:
        yield records
    finally:
        datasets.load_dataset = original


def _load_wikitext(cache_dir: Path, split: str):
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Qwen quality evaluation requires datasets.") from exc
    record = _load_lock()["datasets"]["wikitext"]
    return datasets.load_dataset(
        record["repository"],
        record["config"],
        split=split,
        revision=record["revision"],
        cache_dir=str(cache_dir),
    )


def _token_stream(tokenizer, dataset) -> torch.Tensor:
    text = "\n\n".join(value for value in dataset["text"] if value.strip())
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    tokens = encoded["input_ids"].squeeze(0)
    if tokens.numel() <= CALIBRATION_WINDOW:
        raise ValueError("WikiText token stream is unexpectedly short.")
    return tokens


def _calibration_batches(
    tokens: torch.Tensor, device: str, count: int
) -> List[Dict[str, torch.Tensor]]:
    required = count * CALIBRATION_WINDOW
    if tokens.numel() < required:
        raise ValueError(
            f"Calibration requires {required} tokens, got {tokens.numel()}."
        )
    result = []
    for index in range(count):
        window = tokens[
            index * CALIBRATION_WINDOW : (index + 1) * CALIBRATION_WINDOW
        ].unsqueeze(0)
        result.append(
            {
                "input_ids": window.to(device),
                "attention_mask": torch.ones_like(window, device=device),
            }
        )
    return result


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return float(loss.detach()), int(valid.sum())


def _chunked_nll(
    *,
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    autocast_context,
    encoding_mode: Optional[str],
) -> tuple[float, int]:
    total_nll = 0.0
    token_count = 0
    cache = None
    previous_last_logits = None
    for start in range(0, input_ids.shape[1], PPL_CACHE_CHUNK):
        end = min(start + PPL_CACHE_CHUNK, input_ids.shape[1])
        functional.reset_net(model)
        kwargs = {
            "input_ids": input_ids[:, start:end],
            "attention_mask": torch.ones(
                input_ids.shape[0],
                end,
                dtype=input_ids.dtype,
                device=input_ids.device,
            ),
            "past_key_values": cache,
            "use_cache": True,
        }
        if encoding_mode is not None:
            kwargs["encoding_mode"] = encoding_mode
        with torch.inference_mode(), autocast_context():
            output = model(**kwargs)
        logits = output.logits
        cache = output.past_key_values
        if previous_last_logits is not None:
            boundary_label = labels[:, start]
            valid = boundary_label.ne(-100)
            if valid.any():
                total_nll += float(
                    F.cross_entropy(
                        previous_last_logits.float(),
                        boundary_label,
                        ignore_index=-100,
                        reduction="sum",
                    ).detach()
                )
                token_count += int(valid.sum())
        if end - start > 1:
            value, count = _nll(logits, labels[:, start:end])
            total_nll += value
            token_count += count
        previous_last_logits = logits[:, -1]
    return total_nll, token_count


def _rolling_ppl(
    *,
    dense_model,
    snn_model,
    tokens: torch.Tensor,
    device: str,
    autocast_context,
    max_windows: Optional[int],
    encoding_mode: str,
    shard_index: int = 0,
    shard_count: int = 1,
) -> Dict[str, object]:
    dense_nll = 0.0
    snn_nll = 0.0
    token_count = 0
    previous_end = 0
    global_window_count = 0
    processed_window_indices = []
    for begin_target in range(0, int(tokens.numel()), PPL_STRIDE):
        end = min(begin_target + PPL_STRIDE, int(tokens.numel()))
        begin = max(end - PPL_CONTEXT, 0)
        target_length = end - previous_end
        window_index = global_window_count
        if max_windows is not None and window_index >= max_windows:
            break
        previous_end = end
        global_window_count += 1
        if window_index % shard_count != shard_index:
            if end == tokens.numel():
                break
            continue
        input_ids = tokens[begin:end].unsqueeze(0).to(device)
        if input_ids.shape[1] < 2:
            break
        labels = input_ids.clone()
        labels[:, :-target_length] = -100
        dense_value, dense_tokens = _chunked_nll(
            model=dense_model,
            input_ids=input_ids,
            labels=labels,
            autocast_context=autocast_context,
            encoding_mode=None,
        )
        snn_value, snn_tokens = _chunked_nll(
            model=snn_model,
            input_ids=input_ids,
            labels=labels,
            autocast_context=autocast_context,
            encoding_mode=encoding_mode,
        )
        if dense_tokens != snn_tokens:
            raise RuntimeError("Dense and SNN PPL token counts disagree.")
        dense_nll += dense_value
        snn_nll += snn_value
        token_count += dense_tokens
        processed_window_indices.append(window_index)
        if end == tokens.numel():
            break
    if token_count == 0:
        raise ValueError("PPL evaluation produced no target tokens.")
    dense_ppl = math.exp(dense_nll / token_count)
    snn_ppl = math.exp(snn_nll / token_count)
    return {
        "dense_ppl": dense_ppl,
        "snn_ppl": snn_ppl,
        "dense_nll": dense_nll,
        "snn_nll": snn_nll,
        "relative_degradation": snn_ppl / dense_ppl - 1.0,
        "token_count": token_count,
        "processed_window_count": len(processed_window_indices),
        "window_count": len(processed_window_indices),
        "processed_window_indices": processed_window_indices,
        "global_window_count": global_window_count,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "context_length": PPL_CONTEXT,
        "stride": PPL_STRIDE,
        "cache_chunk_length": PPL_CACHE_CHUNK,
    }


def _metric_value(results: Mapping[str, object], task: str, metric: str) -> float:
    task_results = results["results"][task]
    matches = [
        float(value)
        for name, value in task_results.items()
        if name == metric or name.startswith(f"{metric},")
    ]
    if len(matches) != 1 or not math.isfinite(matches[0]):
        raise ValueError(f"Task {task!r} did not produce one finite {metric!r} metric.")
    return matches[0]


def _configure_task_adapters(dense_lm, snn_lm) -> None:
    dense_lm.enable_distributed_requests()
    snn_lm.enable_distributed_requests()
    snn_lm.reset_before_call = True


def _is_task_report_rank() -> bool:
    return not torch_dist.is_initialized() or torch_dist.get_rank() == 0


def _zero_shot(
    *,
    dense_model,
    snn_model,
    tokenizer,
    limit: Optional[int],
    batch_size: int,
    tasks: tuple[str, ...] = TASKS,
) -> Optional[Dict[str, object]]:
    lm_eval_version = _require_lm_eval_version()
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Qwen quality evaluation requires lm-eval[hf].") from exc

    class ResettingHFLM(HFLM):
        reset_before_call = False

        def _model_call(self, inps, attn_mask=None, labels=None):
            if self.reset_before_call:
                functional.reset_net(self.model)
            return super()._model_call(inps, attn_mask=attn_mask, labels=labels)

        def enable_distributed_requests(self) -> None:
            if not torch_dist.is_initialized():
                return
            self.rank = torch_dist.get_rank()
            self.world_size = torch_dist.get_world_size()

        def all_gather(self, tensor):
            if self.world_size <= 1:
                return tensor
            gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
            torch_dist.all_gather(gathered, tensor)
            return torch.stack(gathered)

        def barrier(self):
            if self.world_size > 1:
                torch_dist.barrier()

    dense_lm = ResettingHFLM(
        pretrained=dense_model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=PPL_CONTEXT,
        logits_cache=True,
    )
    snn_lm = ResettingHFLM(
        pretrained=snn_model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=PPL_CONTEXT,
        logits_cache=True,
    )
    _configure_task_adapters(dense_lm, snn_lm)
    lock = _load_lock()["datasets"]

    def evaluate(model):
        with _pin_dataset_revisions(lock) as records:
            result = lm_eval.simple_evaluate(
                model=model,
                tasks=list(tasks),
                num_fewshot=0,
                limit=limit,
                batch_size=batch_size,
                log_samples=False,
            )
        expected = {lock[TASK_DATASET_KEYS[name]]["repository"] for name in tasks}
        observed = {record["repository"] for record in records}
        if observed != expected:
            raise RuntimeError(
                f"lm-eval dataset provenance incomplete: expected {sorted(expected)}, "
                f"observed {sorted(observed)}."
            )
        return result, records

    dense_raw, dense_datasets = evaluate(dense_lm)
    snn_raw, snn_datasets = evaluate(snn_lm)
    if dense_datasets != snn_datasets:
        raise RuntimeError("Dense and SNN lm-eval dataset fingerprints differ.")
    if not _is_task_report_rank():
        return None
    if dense_raw is None or snn_raw is None:
        raise RuntimeError("lm-eval rank 0 did not return task results.")
    per_task = {}
    drops = []
    for task in tasks:
        metric = PRIMARY_METRICS[task]
        dense = _metric_value(dense_raw, task, metric)
        snn = _metric_value(snn_raw, task, metric)
        drop = 100.0 * (dense - snn)
        drops.append(drop)
        per_task[task] = {
            "metric": metric,
            "dense": dense,
            "snn": snn,
            "drop_percentage_points": drop,
        }
    return {
        "lm_eval_version": lm_eval_version,
        "num_fewshot": 0,
        "batch_size": batch_size,
        "world_size": dense_lm.world_size,
        "limit": limit,
        "tasks": per_task,
        "mean_drop_percentage_points": sum(drops) / len(drops),
        "max_drop_percentage_points": max(drops),
        "snn_encoding_mode": "signed_if",
        "datasets": dense_datasets,
    }


def _encoder_summary(converted) -> Dict[str, object]:
    records = list(converted.encoder_statistics())
    if not records:
        raise RuntimeError("Converted Qwen reported no signed encoders.")
    return {
        "count": len(records),
        "max_local_relative_l2": max(
            float(record["local_relative_l2"]) for record in records
        ),
        "mean_local_relative_l2": sum(
            float(record["local_relative_l2"]) for record in records
        )
        / len(records),
        "mean_positive_spike_rate": sum(
            float(record["positive_spike_rate"]) for record in records
        )
        / len(records),
        "mean_negative_spike_rate": sum(
            float(record["negative_spike_rate"]) for record in records
        )
        / len(records),
        "max_boundary_correction_fraction": max(
            float(record["boundary_correction_fraction"]) for record in records
        ),
        "worst_local_encoder": max(
            records, key=lambda record: float(record["local_relative_l2"])
        )["name"],
    }


def _capture_encoder_summary(
    converted, tokenizer, device: str, autocast_context
) -> Dict[str, object]:
    encoded = tokenizer(
        STATISTICS_PROBE_TEXT,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)
    converted.set_collect_statistics(True)
    functional.reset_net(converted)
    try:
        with torch.inference_mode(), autocast_context():
            converted(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoding_mode="signed_if",
            )
        return _encoder_summary(converted)
    finally:
        functional.reset_net(converted)
        converted.set_collect_statistics(False)


def _save_calibration(path: Path, calibration: Qwen2SNNCalibration) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(calibration.state_dict(), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_full_evaluation(
    *,
    split: str,
    max_ppl_windows: Optional[int],
    run_tasks: bool,
    task_limit: Optional[int],
    skip_ppl: bool = False,
    ppl_shard_count: int = 1,
    tasks: tuple[str, ...] = TASKS,
) -> bool:
    return (
        split == "test"
        and max_ppl_windows is None
        and not skip_ppl
        and ppl_shard_count == 1
        and run_tasks
        and task_limit is None
        and tasks == TASKS
    )


def _run(args: argparse.Namespace) -> Dict[str, object]:
    runtime_device, task_rank = _initialize_distributed_task_device(args)
    args.device = runtime_device
    lock = _load_lock()
    model_record = lock["models"][args.model_key]
    tokenizer, source_model = _load_model(args.model_root, args.device)
    parameter_count = sum(value.numel() for value in source_model.parameters())
    if parameter_count != int(model_record["parameter_count"]):
        raise ValueError("Loaded Qwen parameter count does not match artifact lock.")
    if args.precision == "fp32":
        source_model.float()
    precision = prepare_model_for_precision(
        source_model,
        args.device,
        PrecisionConfig(
            mode=args.precision,
            strictness="strict",
            report=True,
            device=args.device,
        ),
    )
    config = Qwen2SNNConfig(
        time_steps=args.time_steps,
        calibration_levels=args.calibration_levels,
        calibration_quantile=args.calibration_quantile,
        calibration_reservoir_size=args.calibration_reservoir_size,
        calibration_seed=20260719,
        neuron_backend="triton",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.calibration_artifact is None:
        train = _load_wikitext(args.hf_cache, "train")
        train_tokens = _token_stream(tokenizer, train)
        batches = _calibration_batches(
            train_tokens, args.device, args.calibration_windows
        )
        with precision.autocast_context():
            calibration = calibrate_qwen2_snn(precision.model, batches, config)
        calibration_path = args.output_dir / "calibration.pt"
        calibration_sha256 = _save_calibration(calibration_path, calibration)
        calibration_origin = "generated"
    else:
        calibration_path = args.calibration_artifact
        calibration, calibration_sha256 = _load_calibration(calibration_path)
        _validate_calibration_config(
            calibration,
            time_steps=config.time_steps,
            calibration_levels=config.calibration_levels,
            calibration_quantile=config.calibration_quantile,
            calibration_reservoir_size=config.calibration_reservoir_size,
            calibration_seed=config.calibration_seed,
        )
        calibration_origin = "reused"
    converted = ModuleConverter(Qwen2SNNRecipe(calibration, config)).convert(
        precision.model
    )
    encoder_summary = _capture_encoder_summary(
        converted,
        tokenizer,
        args.device,
        precision.autocast_context,
    )
    ppl = None
    if not args.skip_ppl:
        evaluation = _load_wikitext(args.hf_cache, args.wikitext_split)
        evaluation_tokens = _token_stream(tokenizer, evaluation)
        ppl = _rolling_ppl(
            dense_model=precision.model,
            snn_model=converted,
            tokens=evaluation_tokens,
            device=args.device,
            autocast_context=precision.autocast_context,
            max_windows=args.max_ppl_windows,
            encoding_mode="signed_if",
            shard_index=args.ppl_shard_index,
            shard_count=args.ppl_shard_count,
        )
    zero_shot = None
    if args.run_tasks:
        zero_shot = _zero_shot(
            dense_model=precision.model,
            snn_model=converted,
            tokenizer=tokenizer,
            limit=args.task_limit,
            batch_size=args.task_batch_size,
            tasks=tuple(args.tasks),
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "run": {
            "argv": sys.argv,
            "hostname": os.uname().nodename,
            "utc_unix_time": time.time(),
            "task_rank": task_rank,
        },
        "source": {
            "worktree_revision": args.worktree_revision,
            "runner_sha256": RUNNER_SHA256,
            "artifact_lock_sha256": hashlib.sha256(
                ARTIFACT_LOCK.read_bytes()
            ).hexdigest(),
        },
        "model": {
            "key": args.model_key,
            **model_record,
            "files": _hash_files(args.model_root),
        },
        "data": lock["datasets"],
        "environment": build_environment(args.device),
        "configuration": {
            "precision": args.precision,
            "time_steps": config.time_steps,
            "calibration_levels": config.calibration_levels,
            "calibration_quantile": config.calibration_quantile,
            "calibration_reservoir_size": config.calibration_reservoir_size,
            "calibration_seed": config.calibration_seed,
            "calibration_windows_requested": (
                args.calibration_windows if args.calibration_artifact is None else None
            ),
            "calibration_valid_token_count": calibration.valid_token_count,
            "wikitext_split": args.wikitext_split,
            "full_evaluation": _is_full_evaluation(
                split=args.wikitext_split,
                max_ppl_windows=args.max_ppl_windows,
                run_tasks=args.run_tasks,
                task_limit=args.task_limit,
                skip_ppl=args.skip_ppl,
                ppl_shard_count=args.ppl_shard_count,
                tasks=tuple(args.tasks),
            ),
            "skip_ppl": args.skip_ppl,
            "ppl_shard_index": args.ppl_shard_index,
            "ppl_shard_count": args.ppl_shard_count,
            "tasks": list(args.tasks) if args.run_tasks else [],
            "max_ppl_windows": args.max_ppl_windows,
            "task_limit": args.task_limit,
            "task_batch_size": args.task_batch_size,
            "task_world_size": (
                torch_dist.get_world_size() if torch_dist.is_initialized() else 1
            ),
            "statistics_enabled_during_quality": False,
            "evaluation_mode": "multistep_signed_if",
        },
        "precision": precision.describe(),
        "conversion": {
            "temporal_layout": converted.temporal_layout,
            "execution_schedule": converted.execution_schedule,
            "online_inference": converted.online_inference,
            "structure": converted.structure_summary(),
            "calibration_path": str(calibration_path.resolve()),
            "calibration_sha256": calibration_sha256,
            "calibration_origin": calibration_origin,
            "encoder_summary": encoder_summary,
            "statistics_probe_sha256": hashlib.sha256(
                STATISTICS_PROBE_TEXT.encode("utf-8")
            ).hexdigest(),
        },
        "quality": {"wikitext": ppl, "zero_shot": zero_shot},
        "acceptance": {
            "formal_phase_gate_passed": None,
            "ppl_gate_passed": None,
            "zero_shot_gate_passed": None,
        },
    }
    report["environment"]["distributed_backend"] = (
        torch_dist.get_backend() if torch_dist.is_initialized() else None
    )
    report["environment"]["distributed_timeout_seconds"] = int(
        DISTRIBUTED_TASK_TIMEOUT.total_seconds()
    )
    full_ppl = (
        args.wikitext_split == "test"
        and args.max_ppl_windows is None
        and not args.skip_ppl
        and args.ppl_shard_count == 1
    )
    if full_ppl:
        _validate_quality(report)
        report["acceptance"]["ppl_gate_passed"] = True
        if args.run_tasks and args.task_limit is None and tuple(args.tasks) == TASKS:
            report["acceptance"]["zero_shot_gate_passed"] = True
            report["acceptance"]["formal_phase_gate_passed"] = True
    json.dumps(report, allow_nan=False)
    return report


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True, choices=("0.5b", "1.5b", "3b"))
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hf-cache", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda",))
    parser.add_argument("--worktree-revision", required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--time-steps",
        required=True,
        type=int,
        choices=(16, 32, 64, 128, 160, 192, 256, 512),
    )
    parser.add_argument("--calibration-levels", required=True, type=int)
    parser.add_argument("--calibration-quantile", required=True, type=float)
    parser.add_argument("--calibration-reservoir-size", type=int, default=4096)
    parser.add_argument("--calibration-windows", type=int, default=CALIBRATION_WINDOWS)
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument(
        "--wikitext-split", choices=("validation", "test"), required=True
    )
    parser.add_argument(
        "--max-ppl-windows",
        type=int,
        help="Global pre-sharding PPL window limit for preflight runs.",
    )
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--ppl-shard-index", type=int, default=0)
    parser.add_argument("--ppl-shard-count", type=int, default=1)
    parser.add_argument("--run-tasks", action="store_true")
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--task-batch-size", type=int, default=1)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = _parse_args(argv)
        _validate_distributed_task_args(args)
        if args.calibration_levels > args.time_steps:
            raise ValueError("calibration-levels must not exceed time-steps.")
        if args.calibration_windows <= 0:
            raise ValueError("calibration-windows must be positive.")
        if args.max_ppl_windows is not None and args.max_ppl_windows <= 0:
            raise ValueError("max-ppl-windows must be positive.")
        if args.wikitext_split != "test" and args.run_tasks:
            raise ValueError("Zero-shot tasks are only allowed with the test split.")
        if args.task_limit is not None and not args.run_tasks:
            raise ValueError("task-limit requires --run-tasks.")
        if args.task_batch_size <= 0:
            raise ValueError("task-batch-size must be positive.")
        if args.tasks != list(TASKS) and not args.run_tasks:
            raise ValueError("tasks requires --run-tasks.")
        if args.skip_ppl and not args.run_tasks:
            raise ValueError("skip-ppl requires --run-tasks.")
        if (
            args.ppl_shard_count <= 0
            or not 0 <= args.ppl_shard_index < args.ppl_shard_count
        ):
            raise ValueError("PPL shard index must lie in [0, ppl-shard-count).")
        if args.ppl_shard_count > 1 and args.skip_ppl:
            raise ValueError("PPL sharding cannot be combined with skip-ppl.")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen quality evaluation requires CUDA.")
        if (args.output_dir / "report.json").exists():
            raise FileExistsError(
                f"Refusing to overwrite {args.output_dir / 'report.json'}."
            )
        report = _run(args)
        rank = torch_dist.get_rank() if torch_dist.is_initialized() else 0
        target = _write_report(args.output_dir, report) if rank == 0 else None
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
        return 2
    finally:
        if torch_dist.is_initialized():
            torch_dist.destroy_process_group()
    if target is not None:
        print(f"report_path={target.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
