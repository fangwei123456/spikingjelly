"""Private Phase 6.0 GPT-2 dense DynamicCache contract + equivalence smoke.

This runner captures the Hugging Face GPT-2 ``DynamicCache`` semantics that
Phase 6.0 needs to support.  It performs a single dense forward and several
cached prefill + decode scenarios, comparing the cached logits to the dense
forward within the tolerance the runner measures.  It does **not** call
``model.generate()``, train, persist checkpoints, run a full PPL benchmark,
or interact with RoPE / GQA.  Phase 7 will start a Qwen adapter only after
this contract passes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from .conversion_contract import (
        DEFAULT_MODEL_NAME,
        EXPECTED_REVISION,
        GPT2ModelPaths,
    )
    from .dense_baseline import (
        DEFAULT_MAX_SAMPLES,
        FIXED_PROMPTS,
        MAX_LENGTH,
        REPORT_FILENAME,
        _import_huggingface,
        build_environment,
        hash_files,
        validate_model_root,
    )
except ImportError:  # pragma: no cover - script entry path
    from conversion_contract import (
        DEFAULT_MODEL_NAME,
        EXPECTED_REVISION,
        GPT2ModelPaths,
    )
    from dense_baseline import (
        DEFAULT_MAX_SAMPLES,
        FIXED_PROMPTS,
        MAX_LENGTH,
        REPORT_FILENAME,
        _import_huggingface,
        build_environment,
        hash_files,
        validate_model_root,
    )


CACHE_SCHEMA_VERSION = 1
CONTRACT_KIND = "gpt2-dense-dynamic-cache-equivalence"
VALID_DEVICES = ("cpu", "cuda")
DEFAULT_MAX_LENGTH = 64
DEFAULT_PREFILL_LENGTH = 16
DEFAULT_DECODE_STEPS = 8
CACHE_OWNER = "huggingface_gpt2_dense"
CACHE_MUTATION = "decode_mutates_cache_in_place"

_UNSUPPORTED = (
    "no SNN conversion",
    "no Qwen",
    "no RoPE/GQA",
    "no generate()",
    "no checkpoint",
    "no full PPL",
)


def _cache_class_name(cache: object) -> str:
    return type(cache).__name__


def _cache_seq_length(cache: object) -> int:
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    raise TypeError("Cache object does not expose get_seq_length().")


def _cache_layer_summaries(cache: object) -> List[Dict[str, Any]]:
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError("Cache object does not expose a layers attribute.")
    summaries: List[Dict[str, Any]] = []
    for layer in layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("Cache layer must expose tensor .keys and .values.")
        summaries.append(
            {
                "key_shape": list(keys.shape),
                "value_shape": list(values.shape),
                "dtype": str(keys.dtype),
                "device": str(keys.device),
            }
        )
    return summaries


def _clone_cache(cache: object) -> object:
    return copy.deepcopy(cache)


def _reorder_cache(cache: object, indices: torch.Tensor) -> object:
    """Reorder a cache along its batch dimension and return the (now-reordered) cache.

    :class:`transformers.cache_utils.DynamicCache.reorder_cache` mutates the
    cache in place and returns ``None`` in ``transformers`` 5.x; older caches
    expose ``batch_select_indices`` instead and return the new cache. The
    helper accepts either shape so the runner is not pinned to a specific
    Hugging Face release.
    """

    reorder = getattr(cache, "reorder_cache", None)
    if callable(reorder):
        result = reorder(indices)
        return cache if result is None else result
    batch_select = getattr(cache, "batch_select_indices", None)
    if callable(batch_select):
        result = batch_select(indices)
        return cache if result is None else result
    raise TypeError("Cache object does not support reorder or batch_select_indices.")


def _compute_shifted_ce_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    labels = input_ids.clone()
    if attention_mask is not None:
        labels[attention_mask == 0] = -100
    loss_tensor = F.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
        labels[..., 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    value = float(loss_tensor.detach().float().cpu())
    if not math.isfinite(value):
        raise ValueError(f"Computed shifted CE loss is not finite: {value!r}.")
    return value


def _abs_error(actual: torch.Tensor, reference: torch.Tensor) -> Tuple[float, float]:
    diff = (actual - reference).detach()
    if not torch.isfinite(diff).all():
        raise ValueError(
            "Non-finite values encountered while computing equivalence error."
        )
    return float(diff.abs().max().cpu()), float(diff.abs().mean().cpu())


def _load_gpt2(
    paths: GPT2ModelPaths,
    *,
    device: str,
) -> Tuple[object, object]:
    transformers = _import_huggingface(lambda: __import__("transformers"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(paths.root), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token and no eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(paths.root), local_files_only=True
    )
    model.eval()
    model.to(device)
    return tokenizer, model


def _encode_inputs(
    tokenizer: object,
    prompts: Sequence[str],
    *,
    max_length: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        list(prompts),
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return input_ids, attention_mask


def _full_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, object]:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    logits = outputs.logits
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Model output did not expose tensor logits.")
    return logits, outputs.past_key_values


def _decode_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: object,
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    return outputs.logits


def _run_cached_decode(
    model: torch.nn.Module,
    prefill_ids: torch.Tensor,
    prefill_mask: torch.Tensor,
    decode_ids: torch.Tensor,
    decode_mask: torch.Tensor,
    *,
    decode_steps: int,
) -> Tuple[List[torch.Tensor], int, int]:
    """Run a single prefill followed by ``decode_steps`` one-token decodes.

    :returns: ``(logits_per_step, prefill_seq_length, final_seq_length)``.
    """

    prefill_logits, prefill_cache = _full_forward(model, prefill_ids, prefill_mask)
    prefill_seq_length = _cache_seq_length(prefill_cache)
    decode_cache = _clone_cache(prefill_cache)
    del prefill_cache, prefill_logits  # ensure we only use the clone.

    step_logits: List[torch.Tensor] = []
    for step in range(decode_steps):
        step_input = decode_ids[:, step : step + 1]
        step_mask = decode_mask[:, : prefill_seq_length + step + 1]
        logits = _decode_step(model, step_input, step_mask, decode_cache)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Decode output did not expose tensor logits.")
        step_logits.append(logits[:, 0])
    final_seq_length = _cache_seq_length(decode_cache)
    return step_logits, prefill_seq_length, final_seq_length


def compute_scenarios(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefill_length: int,
    decode_steps: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Run the full set of Phase 6.0 cache scenarios."""

    if prefill_length < 2:
        raise ValueError(
            "prefill_length must be at least 2 for segmented prefill; "
            f"got {prefill_length}."
        )
    if decode_steps <= 0:
        raise ValueError(f"decode_steps must be positive; got {decode_steps}.")
    if input_ids.shape[0] < 2:
        raise ValueError(
            "At least 2 samples are required to exercise cache reorder; "
            f"got batch size {input_ids.shape[0]}."
        )
    if prefill_length + decode_steps > input_ids.shape[1]:
        raise ValueError(
            "prefill_length + decode_steps must be <= max_length "
            f"({input_ids.shape[1]}); got {prefill_length + decode_steps}."
        )

    full_logits, full_cache = _full_forward(model, input_ids, attention_mask)
    dense_loss = _compute_shifted_ce_loss(
        full_logits, input_ids, attention_mask=attention_mask
    )
    cache_layer_summaries = _cache_layer_summaries(full_cache)
    cache_layer_count = len(cache_layer_summaries)

    prefill_ids = input_ids[:, :prefill_length]
    prefill_mask = attention_mask[:, :prefill_length]
    decode_ids = input_ids[:, prefill_length : prefill_length + decode_steps]
    decode_mask = attention_mask[:, : prefill_length + decode_steps]

    (
        single_logits,
        prefill_seq_length,
        _single_final_seq,
    ) = _run_cached_decode(
        model,
        prefill_ids,
        prefill_mask,
        decode_ids,
        decode_mask,
        decode_steps=decode_steps,
    )
    single_cached_max, single_cached_mean = _abs_error(
        torch.stack(single_logits, dim=1),
        full_logits[:, prefill_length : prefill_length + decode_steps],
    )

    segmented_cache = _clone_cache(
        _full_forward(
            model,
            input_ids[:, : prefill_length // 2],
            attention_mask[:, : prefill_length // 2],
        )[1]
    )
    with torch.no_grad():
        segmented_outputs = model(
            input_ids=input_ids[:, prefill_length // 2 : prefill_length],
            attention_mask=attention_mask[:, :prefill_length],
            past_key_values=segmented_cache,
            use_cache=True,
            return_dict=True,
        )
    del segmented_outputs

    if decode_steps > 0:
        seg_decode_cache = _clone_cache(segmented_cache)
        seg_step_logits: List[torch.Tensor] = []
        for step in range(decode_steps):
            step_input = decode_ids[:, step : step + 1]
            step_mask = decode_mask[:, : prefill_length + step + 1]
            seg_step_logits.append(
                _decode_step(model, step_input, step_mask, seg_decode_cache)[:, 0]
            )
        seg_full = torch.stack(seg_step_logits, dim=1)
        seg_max, seg_mean = _abs_error(
            seg_full,
            full_logits[:, prefill_length : prefill_length + decode_steps],
        )
    else:
        seg_max, seg_mean = 0.0, 0.0
        seg_decode_cache = segmented_cache
    segmented_decode_final_seq = _cache_seq_length(seg_decode_cache)
    segmented_compare = {
        "max": seg_max,
        "mean": seg_mean,
    }

    prefill_logits, prefill_cache = _full_forward(model, prefill_ids, prefill_mask)
    prefill_cache_len_before = _cache_seq_length(prefill_cache)
    branch_cache = _clone_cache(prefill_cache)
    branch_decoded = _decode_step(
        model,
        decode_ids[:, 0:1],
        decode_mask[:, : prefill_length + 1],
        branch_cache,
    )
    del branch_decoded
    prefill_cache_len_after = _cache_seq_length(prefill_cache)
    clone_independent = prefill_cache_len_before == prefill_cache_len_after
    if not clone_independent:
        raise ValueError("Cache clone is not independent from the source cache.")

    batch_size = prefill_ids.shape[0]
    if batch_size >= 2:
        if batch_size >= 4:
            indices = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        else:
            indices = torch.arange(batch_size - 1, -1, -1, dtype=torch.long)
        reorder_prefill_cache = _clone_cache(prefill_cache)
        reordered = _reorder_cache(reorder_prefill_cache, indices)
        # Reorder both the cache and the decode inputs so that batch i of the
        # reordered set is exactly equivalent to old sample ``indices[i]``;
        # only then does the dense forward at ``prefill_length`` (with the
        # same token appended) provide a valid reference for the logits.
        reorder_logits = _decode_step(
            model,
            decode_ids[indices, 0:1],
            decode_mask[indices, : prefill_length + 1],
            reordered,
        )[:, 0]
        reference_logits = full_logits[indices, prefill_length]
        reorder_max, reorder_mean = _abs_error(reorder_logits, reference_logits)
    else:
        raise ValueError(
            "At least 2 samples are required to exercise cache reorder; "
            f"got batch size {batch_size}."
        )

    fresh_prefill_logits, _fresh_prefill_cache = _full_forward(
        model, prefill_ids, prefill_mask
    )
    reset_max, reset_mean = _abs_error(fresh_prefill_logits, prefill_logits)

    contract: Dict[str, object] = {
        "owner": CACHE_OWNER,
        "cache_class": _cache_class_name(full_cache),
        "layer_summaries": cache_layer_summaries,
        "mutation": CACHE_MUTATION,
        "clone_required_for_branching": True,
        "snn_state_separate": True,
        "reset_supported": True,
        "clone_supported": True,
        "reorder_supported": True,
        "serialized_cache": False,
    }

    metrics = {
        "dense_loss": dense_loss,
        "cache_layer_count": cache_layer_count,
        "cache_seq_length_after_prefill": prefill_seq_length,
        # Total decoded tokens across the batch: decode steps per sample
        # times the batch size. Computed from the input parameters rather
        # than the cache seq length so the metric tracks what the runner
        # actually ran, independent of model-level cache semantics.
        "decode_token_count": decode_steps * prefill_ids.shape[0],
        "cached_decode_max_abs_error": single_cached_max,
        "cached_decode_mean_abs_error": single_cached_mean,
        "segmented_prefill_max_abs_error": segmented_compare["max"],
        "segmented_prefill_mean_abs_error": segmented_compare["mean"],
        "reorder_max_abs_error": reorder_max,
        "reorder_mean_abs_error": reorder_mean,
        "reset_prefill_max_abs_error": reset_max,
        "reset_prefill_mean_abs_error": reset_mean,
        "clone_independence_preserved": bool(clone_independent),
        "segmented_cache_seq_length_after_decode": segmented_decode_final_seq,
    }
    return contract, metrics


def build_cache_report(
    *,
    paths: GPT2ModelPaths,
    max_samples: int,
    max_length: int,
    prefill_length: int,
    decode_steps: int,
    cache_contract: Mapping[str, object],
    metrics: Mapping[str, object],
    environment: Mapping[str, object],
) -> Dict[str, object]:
    report = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "model": {
            "name": DEFAULT_MODEL_NAME,
            "revision": EXPECTED_REVISION,
        },
        "environment": dict(environment),
        "input": {
            "max_samples": max_samples,
            "max_length": max_length,
            "prefill_length": prefill_length,
            "decode_steps": decode_steps,
            "prompts": list(FIXED_PROMPTS[:max_samples]),
        },
        "cache_contract": dict(cache_contract),
        "scenarios": {
            "full_forward": True,
            "single_prefill_cached_decode": True,
            "segmented_prefill_cached_decode": True,
            "clone_independence": True,
            "reorder_cache": True,
            "reset": True,
        },
        "metrics": dict(metrics),
        "files": hash_files(paths.root),
        "unsupported": list(_UNSUPPORTED),
    }
    json.dumps(report, allow_nan=False)
    return report


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 6.0 GPT-2 dense DynamicCache contract and "
            "equivalence scenarios."
        )
    )
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=VALID_DEVICES, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--prefill-length", type=int, default=DEFAULT_PREFILL_LENGTH)
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    return parser.parse_args(argv)


def _emit(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _check_revision(declared: str) -> None:
    if declared != EXPECTED_REVISION:
        raise SystemExit(
            f"--source-revision must equal {EXPECTED_REVISION}; got {declared!r}."
        )


def _check_report_available(output_dir: Path) -> None:
    target = output_dir / REPORT_FILENAME
    if target.exists():
        raise SystemExit(
            f"Refusing to overwrite existing report at {target}. "
            "Choose a new --output-dir or delete the previous result."
        )


def _validate_inputs(args: argparse.Namespace) -> None:
    if args.max_samples < 2 or args.max_samples > len(FIXED_PROMPTS):
        raise ValueError(
            f"--max-samples must be between 2 and {len(FIXED_PROMPTS)}; "
            f"got {args.max_samples}."
        )
    if args.max_length <= 1:
        raise ValueError(f"--max-length must be greater than 1; got {args.max_length}.")
    if args.prefill_length < 2:
        raise ValueError(
            "--prefill-length must be at least 2 for segmented prefill; "
            f"got {args.prefill_length}."
        )
    if args.decode_steps <= 0:
        raise ValueError(f"--decode-steps must be positive; got {args.decode_steps}.")
    if args.prefill_length + args.decode_steps > args.max_length:
        raise ValueError(
            "--prefill-length + --decode-steps must be <= --max-length; "
            f"got {args.prefill_length + args.decode_steps} > {args.max_length}."
        )


def _write_report(output_dir: Path, report: Mapping[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_report_available(output_dir)
    target = output_dir / REPORT_FILENAME
    target.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return target


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        _check_revision(args.source_revision)
    except SystemExit as exc:
        _emit(str(exc))
        return 2

    try:
        _validate_inputs(args)
    except ValueError as exc:
        _emit(str(exc))
        return 2

    try:
        _check_report_available(args.output_dir)
    except SystemExit as exc:
        _emit(str(exc))
        return 2

    try:
        paths = validate_model_root(args.model_root, require_all=True)
    except FileNotFoundError as exc:
        _emit(str(exc))
        return 3

    if args.device == "cuda" and not torch.cuda.is_available():
        _emit("CUDA was requested, but torch.cuda.is_available() is false.")
        return 4

    try:
        tokenizer, model = _load_gpt2(paths, device=args.device)
        prompts = list(FIXED_PROMPTS[: args.max_samples])
        input_ids, attention_mask = _encode_inputs(
            tokenizer, prompts, max_length=args.max_length, device=args.device
        )
        cache_contract, metrics = compute_scenarios(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefill_length=args.prefill_length,
            decode_steps=args.decode_steps,
        )
        report = build_cache_report(
            paths=paths,
            max_samples=args.max_samples,
            max_length=args.max_length,
            prefill_length=args.prefill_length,
            decode_steps=args.decode_steps,
            cache_contract=cache_contract,
            metrics=metrics,
            environment=build_environment(args.device),
        )
        target = _write_report(args.output_dir, report)
    except ImportError as exc:
        _emit(str(exc))
        return 5
    except OSError as exc:
        _emit(f"Failed to load GPT-2 from {paths.root}: {exc}")
        return 6
    except (TypeError, ValueError, RuntimeError) as exc:
        _emit(str(exc))
        return 7

    sys.stdout.write(f"report_path={target.resolve()}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
