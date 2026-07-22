"""Shared runtime for the final private Qwen conversion runners."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
from pathlib import Path
from typing import Dict, List, Mapping

import torch
import torch.nn.functional as F

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import (
    Qwen2SNNCalibration,
    Qwen2SNNConfig,
)


ARTIFACT_LOCK = Path(__file__).with_name("artifacts.json")
MAX_LENGTH = 64
DECODE_STEPS = 2
FIXED_PROMPTS = (
    "Qwen dense cache equivalence uses a deliberately long deterministic prompt "
    "so that the prefill and decode comparison window contains real tokens only, "
    "while the report records RoPE settings, grouped query attention metadata, "
    "cache tensor shapes, clone behavior, reset behavior, and reorder behavior.",
    "This second fixed prompt repeats the same contract in different words: cached "
    "decoding must match a full dense forward pass over the exact same token "
    "window, segmented prefill must preserve the cache semantics, and unsupported "
    "generation or SNN claims must remain absent from the JSON report.",
    "A third prompt describes the engineering boundary for Phase seven point zero: "
    "the runner loads Qwen locally, never downloads from the network, never writes "
    "checkpoints, never evaluates full perplexity, and only measures small dense "
    "cache correctness on the selected CUDA device.",
    "The fourth prompt ensures batch reordering is exercised with enough real text "
    "tokens before any padding appears, making the comparison independent of Qwen "
    "padding position semantics and focused on the DynamicCache behavior that the "
    "later adapter work will need.",
    "The fifth prompt is intentionally verbose because short sentences would place "
    "padding inside the measured prefill and decode region; that would test padding "
    "policy rather than cache equivalence, so this prompt stays long and explicit.",
    "The sixth prompt is extra capacity for narrowing the prompt bank while still "
    "keeping deterministic language, fixed tokenizer behavior, local file loading, "
    "and a clear separation between dense Qwen validation and future spiking work.",
)


def load_lock() -> Dict[str, object]:
    with ARTIFACT_LOCK.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if lock.get("schema_version") != 1 or set(lock.get("models", {})) != {
        "0.5b",
        "1.5b",
        "3b",
    }:
        raise ValueError("Qwen artifact lock is invalid or incomplete.")
    return lock


def load_calibration_artifact(
    path: Path, config: Qwen2SNNConfig
) -> tuple[Qwen2SNNCalibration, str]:
    calibration, digest = load_calibration(path)
    expected = {
        "time_steps": config.time_steps,
        "calibration_levels": config.calibration_levels,
        "calibration_quantile": config.calibration_quantile,
        "calibration_reservoir_size": config.calibration_reservoir_size,
        "calibration_seed": config.calibration_seed,
    }
    for name, value in expected.items():
        if getattr(calibration, name) != value:
            raise ValueError(
                f"Calibration {name} does not match correctness configuration."
            )
    return calibration, digest


def load_calibration(path: Path) -> tuple[Qwen2SNNCalibration, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration artifact does not exist: {path}.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("Calibration artifact must contain a mapping.")
    calibration = Qwen2SNNCalibration.from_state_dict(state)
    return calibration, hashlib.sha256(path.read_bytes()).hexdigest()


def validate_calibration_config(
    calibration: Qwen2SNNCalibration,
    *,
    time_steps: int,
    calibration_levels: int,
    calibration_quantile: float,
    calibration_reservoir_size: int,
    calibration_seed: int,
) -> None:
    expected = {
        "time_steps": time_steps,
        "calibration_levels": calibration_levels,
        "calibration_quantile": calibration_quantile,
        "calibration_reservoir_size": calibration_reservoir_size,
        "calibration_seed": calibration_seed,
    }
    for name, value in expected.items():
        actual = getattr(calibration, name)
        if actual != value:
            raise ValueError(
                f"Calibration {name}={actual!r} does not match requested {value!r}."
            )


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    value = float(
        (torch.linalg.vector_norm((actual - reference).float()) / denominator).detach()
    )
    if not math.isfinite(value):
        raise ValueError("Relative L2 must be finite.")
    return value


def shifted_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> float:
    labels = input_ids.clone()
    labels[~attention_mask.to(torch.bool)] = -100
    value = float(
        F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        ).detach()
    )
    if not math.isfinite(value):
        raise ValueError("Cross-entropy loss must be finite.")
    return value


def encode(tokenizer, prompts: List[str], device: str):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires a pad or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def load_model(model_root: Path, device: str):
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Qwen scale-out requires the transformers package.") from exc
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (model_root / name).is_file()]
    if missing or not tuple(model_root.glob("*.safetensors")):
        raise FileNotFoundError(
            f"Incomplete local Qwen artifact {model_root}: missing {missing!r}."
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    return tokenizer, model.eval()


def hash_files(root: Path) -> Dict[str, str]:
    result = {}
    for path in sorted(root.iterdir()):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            result[path.name] = digest.hexdigest()
    return result


def build_environment(device: str) -> Dict[str, object]:
    return {
        "host": socket.gethostname(),
        "slug": socket.gethostname().split(".")[0],
        "device": device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def cached_decode(
    model,
    prompt: torch.Tensor,
    *,
    mode: str,
    autocast_context,
) -> Dict[str, object]:
    generated = prompt.clone()
    cache = None
    errors = []
    tokens = []
    cache_summary = None
    for step in range(DECODE_STEPS):
        current = generated if step == 0 else generated[:, -1:]
        mask = torch.ones_like(generated)
        functional.reset_net(model)
        with torch.inference_mode(), autocast_context():
            cached = model(
                input_ids=current,
                attention_mask=mask,
                encoding_mode=mode,
                past_key_values=cache,
                use_cache=True,
            )
        cache = cached.past_key_values
        functional.reset_net(model)
        with torch.inference_mode(), autocast_context():
            full = model(
                input_ids=generated,
                attention_mask=mask,
                encoding_mode=mode,
            )
        errors.append(relative_l2(cached.logits[:, -1], full.logits[:, -1]))
        token = cached.logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(int(token.item()))
        generated = torch.cat((generated, token), dim=1)
    if cache is not None and hasattr(cache, "storage_summary"):
        cache_summary = cache.storage_summary()
    return {
        "max_relative_l2": max(errors),
        "new_token_ids": tokens,
        "cache": cache_summary,
    }
