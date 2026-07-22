"""Qwen2.5 0.5B-3B public-recipe correctness runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import torch
from benchmark.snn_llm._reporting import write_report as _write_report
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import (
    ModuleConverter,
    Qwen2SNNConfig,
    Qwen2SNNRecipe,
    calibrate_qwen2_snn,
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
    load_calibration_artifact as _load_calibration_artifact,
    load_lock as _load_lock,
    load_model as _load_model,
    relative_l2 as _relative_l2,
    shifted_loss as _loss,
)


SCHEMA_VERSION = 1
CONTRACT_KIND = "qwen2-snn-scaleout-correctness"
TIME_STEP_CHOICES = (16, 32, 64, 128, 160, 192, 256, 512)
PREFILL_LENGTH = 16


def _expected_structure(layer_count: int) -> Dict[str, int]:
    return {
        "converted_decoder_count": layer_count,
        "td_rms_norm_count": 2 * layer_count + 1,
        "td_linear_count": 7 * layer_count + 1,
        "td_silu_count": layer_count,
        "td_elementwise_product_count": layer_count,
        "td_sdpa_count": layer_count,
        "signed_if_encoder_count": 4 * layer_count + 1,
        "activation_aware_if_node_count": 2 * (4 * layer_count + 1),
    }


def _validate_report(report: Mapping[str, object]) -> None:
    metrics = report["metrics"]
    limits = {
        "exact_logits_relative_l2": 0.02,
        "exact_loss_delta": 0.05,
        "signed_logits_relative_l2": 0.50,
        "signed_loss_delta": 0.75,
        "reset_replay_max_abs_error": 1e-6,
        "exact_cached_decode_max_relative_l2": 0.02,
        "signed_cached_decode_max_relative_l2": 0.20,
    }
    for name, limit in limits.items():
        value = float(metrics[name])
        if not math.isfinite(value) or value > limit:
            raise ValueError(f"Metric {name}={value!r} exceeds {limit!r}.")
    top1_agreement = float(metrics["signed_top1_agreement"])
    if not math.isfinite(top1_agreement) or top1_agreement < 0.50:
        raise ValueError("signed_top1_agreement is below 0.50.")
    conversion = report["conversion"]
    expected = _expected_structure(int(report["model"]["layer_count"]))
    for name, value in expected.items():
        if int(conversion["structure"][name]) != value:
            raise ValueError(f"Structure {name} does not match model config.")
    if conversion["temporal_layout"] != "[T,B,S,H]":
        raise ValueError("Converted Qwen model lost its explicit temporal layout.")
    if conversion["execution_schedule"] != "layerwise_offline_multistep":
        raise ValueError("Converted Qwen execution schedule is not offline multistep.")


def _run(args: argparse.Namespace) -> Dict[str, object]:
    lock = _load_lock()
    model_record = lock["models"][args.model_key]
    tokenizer, source_model = _load_model(args.model_root, args.device)
    parameter_count = sum(value.numel() for value in source_model.parameters())
    if parameter_count != int(model_record["parameter_count"]):
        raise ValueError(
            f"Parameter count {parameter_count} does not match artifact lock."
        )
    evaluation_ids, evaluation_mask = _encode(
        tokenizer, list(FIXED_PROMPTS[2:]), args.device
    )
    precision = prepare_model_for_precision(
        source_model,
        args.device,
        PrecisionConfig(
            mode="bf16",
            strictness="strict",
            report=True,
            device=args.device,
        ),
    )
    with torch.inference_mode(), precision.autocast_context():
        dense_logits = precision.model(
            input_ids=evaluation_ids,
            attention_mask=evaluation_mask,
            use_cache=False,
        ).logits
    dense_loss = _loss(dense_logits, evaluation_ids, evaluation_mask)
    config = Qwen2SNNConfig(
        time_steps=args.time_steps,
        calibration_levels=args.calibration_levels,
        calibration_quantile=args.calibration_quantile,
        neuron_backend=args.neuron_backend,
    )
    if args.calibration_artifact is None:
        calibration_ids, calibration_mask = _encode(
            tokenizer, list(FIXED_PROMPTS[:2]), args.device
        )
        with precision.autocast_context():
            calibration = calibrate_qwen2_snn(
                precision.model,
                [
                    {
                        "input_ids": calibration_ids,
                        "attention_mask": calibration_mask,
                    }
                ],
                config,
            )
        calibration_origin = "fixed_prompt_smoke"
        calibration_sha256 = None
    else:
        calibration, calibration_sha256 = _load_calibration_artifact(
            args.calibration_artifact, config
        )
        calibration_origin = "reused_artifact"
    converted = ModuleConverter(Qwen2SNNRecipe(calibration, config)).convert(
        precision.model
    )
    converted.eval()
    with torch.inference_mode(), precision.autocast_context():
        functional.reset_net(converted)
        exact = converted(
            input_ids=evaluation_ids,
            attention_mask=evaluation_mask,
            encoding_mode="exact_td",
        ).logits
        functional.reset_net(converted)
        signed = converted(
            input_ids=evaluation_ids,
            attention_mask=evaluation_mask,
            encoding_mode="signed_if",
        ).logits
        functional.reset_net(converted)
        replay = converted(
            input_ids=evaluation_ids,
            attention_mask=evaluation_mask,
            encoding_mode="signed_if",
        ).logits
    valid = evaluation_mask.to(torch.bool)
    exact_loss = _loss(exact, evaluation_ids, evaluation_mask)
    signed_loss = _loss(signed, evaluation_ids, evaluation_mask)
    prompt = evaluation_ids[:1, :PREFILL_LENGTH]
    exact_cache = _cached_decode(
        converted,
        prompt,
        mode="exact_td",
        autocast_context=precision.autocast_context,
    )
    signed_cache = _cached_decode(
        converted,
        prompt,
        mode="signed_if",
        autocast_context=precision.autocast_context,
    )
    metrics = {
        "dense_loss": dense_loss,
        "exact_loss": exact_loss,
        "signed_loss": signed_loss,
        "exact_loss_delta": abs(exact_loss - dense_loss),
        "signed_loss_delta": abs(signed_loss - dense_loss),
        "exact_logits_relative_l2": _relative_l2(exact[valid], dense_logits[valid]),
        "signed_logits_relative_l2": _relative_l2(signed[valid], dense_logits[valid]),
        "signed_top1_agreement": float(
            signed.argmax(-1)[valid].eq(dense_logits.argmax(-1)[valid]).float().mean()
        ),
        "reset_replay_max_abs_error": float((replay - signed).abs().max()),
        "exact_cached_decode_max_relative_l2": exact_cache["max_relative_l2"],
        "signed_cached_decode_max_relative_l2": signed_cache["max_relative_l2"],
    }
    structure = converted.structure_summary()
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "run": {
            "argv": sys.argv,
            "utc_unix_time": time.time(),
            "hostname": os.uname().nodename,
        },
        "source": {
            "worktree_revision": args.worktree_revision,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "artifact_lock_sha256": hashlib.sha256(
                ARTIFACT_LOCK.read_bytes()
            ).hexdigest(),
        },
        "model": {
            "key": args.model_key,
            **model_record,
            "files": _hash_files(args.model_root),
        },
        "environment": build_environment(args.device),
        "configuration": {
            "precision": "bf16",
            "time_steps": args.time_steps,
            "calibration_levels": args.calibration_levels,
            "calibration_quantile": args.calibration_quantile,
            "neuron_backend": args.neuron_backend,
        },
        "precision": precision.describe(),
        "conversion": {
            "parameter_count_before": parameter_count,
            "parameter_count_after": sum(
                value.numel() for value in converted.parameters()
            ),
            "temporal_layout": converted.temporal_layout,
            "execution_schedule": converted.execution_schedule,
            "online_inference": converted.online_inference,
            "structure": structure,
            "encoder_statistics": converted.encoder_statistics(),
            "calibration_origin": calibration_origin,
            "calibration_sha256": calibration_sha256,
            "calibration_path": (
                None
                if args.calibration_artifact is None
                else str(args.calibration_artifact.resolve())
            ),
        },
        "metrics": metrics,
        "generation": {
            "exact": exact_cache,
            "signed": signed_cache,
        },
        "acceptance": {"passed": True},
    }
    _validate_report(report)
    json.dumps(report, allow_nan=False)
    return report


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True, choices=("0.5b", "1.5b", "3b"))
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda",))
    parser.add_argument("--worktree-revision", required=True)
    parser.add_argument(
        "--time-steps", required=True, type=int, choices=TIME_STEP_CHOICES
    )
    parser.add_argument("--calibration-levels", required=True, type=int)
    parser.add_argument("--calibration-quantile", required=True, type=float)
    parser.add_argument("--neuron-backend", required=True, choices=("torch", "triton"))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = _parse_args(argv)
        if args.calibration_levels > args.time_steps:
            raise ValueError("calibration-levels must not exceed time-steps.")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen scale-out smoke requires CUDA.")
        report = _run(args)
        target = _write_report(args.output_dir, report)
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
    print(f"report_path={target.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
