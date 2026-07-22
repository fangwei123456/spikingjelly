import argparse
import hashlib
import logging
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from benchmark.snn_llm import _spikegpt_training as smoke
from spikingjelly.activation_based import functional


CONTEXT_LENGTH = 1024
BATCH_SIZE = 1
N_LAYER = 12
N_EMBD = 512
MODEL_TYPE = "RWKV"
TRAINING_BYTES = 90_000_000
VALIDATION_BYTES = 5_000_000
WINDOW_SIZE = 10

# Pilot-specific constants for the full Enwik8 validation protocol. The
# full-file vocabulary makes complete validation well-defined; the previous
# train-only vocab cannot encode every validation character.
FULL_ENWIK8_VOCAB_SIZE = 6064
FULL_ENWIK8_PARAMETER_COUNT = 47_173_632
VALIDATION_CHARS = 4_982_413
VALIDATION_TARGET_TOKENS = 4_982_412
VALIDATION_BATCHES = 4866
TEST_CHARS = 4_983_534


def _bpc_from_nll(nll: float) -> float:
    return nll / math.log(2.0)


def _training_direction(
    losses: list[float], window_size: int
) -> tuple[bool, float, float]:
    if window_size <= 0:
        raise ValueError("Training direction window size must be positive.")
    if len(losses) < window_size * 2:
        raise ValueError("Training direction requires two complete windows.")
    start_nll = sum(losses[:window_size]) / window_size
    end_nll = sum(losses[-window_size:]) / window_size
    start_bpc = _bpc_from_nll(start_nll)
    end_bpc = _bpc_from_nll(end_nll)
    return end_bpc < start_bpc, start_bpc, end_bpc


def _decode_splits(
    raw: bytes,
    *,
    training_bytes: int = TRAINING_BYTES,
    validation_bytes: int = VALIDATION_BYTES,
) -> tuple[str, str, str]:
    """Decode the 90M training, 5M validation, and 5M test byte splits."""
    training = raw[:training_bytes].decode("utf-8")
    validation = raw[training_bytes : training_bytes + validation_bytes].decode("utf-8")
    test = raw[training_bytes + validation_bytes :].decode("utf-8")
    return training, validation, test


def _build_full_vocabulary(*texts: str) -> tuple[str, ...]:
    """Return a deterministically sorted char vocabulary covering every input."""
    union: set[str] = set()
    for text in texts:
        union |= set(text)
    return tuple(sorted(union))


def _validation_windows(text_length: int) -> tuple[tuple[int, int], ...]:
    """Yield ``(start_offset, target_tokens)`` covering ``text_length - 1``."""
    target_tokens = text_length - 1
    if target_tokens <= 0:
        raise RuntimeError("Validation split is too short.")
    windows = []
    for start in range(0, target_tokens, CONTEXT_LENGTH):
        length = min(CONTEXT_LENGTH, target_tokens - start)
        windows.append((start, length))
    return tuple(windows)


def _variable_length_batch(
    text: str,
    character_to_id: dict[str, int],
    offset: int,
    target_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample = text[offset : offset + target_tokens + 1]
    if len(sample) != target_tokens + 1:
        raise ValueError(
            f"Sample at offset {offset} length {len(sample)} "
            f"does not match target tokens {target_tokens}."
        )
    tokens = torch.tensor(
        [[character_to_id[character] for character in sample]],
        dtype=torch.long,
        device=device,
    )
    return tokens[:, :-1], tokens[:, 1:]


def _training_config() -> dict:
    return {
        "kind": "spikegpt-enwik8-context1024-pilot-v1",
        "backend": "cupy",
        "batch_size": BATCH_SIZE,
        "betas": smoke.BETAS,
        "context_length": CONTEXT_LENGTH,
        "data_md5": smoke.EXPECTED_DATA_MD5,
        "eps": smoke.EPS,
        "grad_norm_clip": smoke.GRAD_NORM_CLIP,
        "learning_rate": smoke.LEARNING_RATE,
        "model_type": MODEL_TYPE,
        "n_embd": N_EMBD,
        "n_layer": N_LAYER,
        "parameter_count": FULL_ENWIK8_PARAMETER_COUNT,
        "precision": "float32",
        "reset": "after-forward-before-backward",
        "seed": smoke.SEED,
        "split": "enwik8-byte-90/5/5",
        "tokenizer": "utf8-character-sorted-full-enwik8-v1",
        "validation_batches": VALIDATION_BATCHES,
        "validation_bytes": VALIDATION_BYTES,
        "validation_chars": VALIDATION_CHARS,
        "validation_target_tokens": VALIDATION_TARGET_TOKENS,
        "vocab_size": FULL_ENWIK8_VOCAB_SIZE,
    }


def _read_data(path: Path) -> tuple[str, str, tuple[str, ...]]:
    raw = path.read_bytes()
    if len(raw) != smoke.EXPECTED_DATA_BYTES:
        raise RuntimeError(
            f"Expected {smoke.EXPECTED_DATA_BYTES} Enwik8 bytes, got {len(raw)}."
        )
    if hashlib.md5(raw).hexdigest() != smoke.EXPECTED_DATA_MD5:
        raise RuntimeError("Enwik8 MD5 mismatch.")
    if hashlib.sha1(raw).hexdigest() != smoke.EXPECTED_DATA_SHA1:
        raise RuntimeError("Enwik8 SHA-1 mismatch.")
    training, validation, test = _decode_splits(raw)
    vocabulary = _build_full_vocabulary(training, validation, test)
    if len(vocabulary) != FULL_ENWIK8_VOCAB_SIZE:
        raise RuntimeError(
            f"Expected {FULL_ENWIK8_VOCAB_SIZE} full Enwik8 characters, "
            f"got {len(vocabulary)}."
        )
    if len(test) != TEST_CHARS:
        raise RuntimeError(f"Expected {TEST_CHARS} test characters, got {len(test)}.")
    if len(validation) != VALIDATION_CHARS:
        raise RuntimeError(
            f"Expected {VALIDATION_CHARS} validation characters, got {len(validation)}."
        )
    if len(validation) - 1 != VALIDATION_TARGET_TOKENS:
        raise RuntimeError(
            f"Expected {VALIDATION_TARGET_TOKENS} validation target tokens, "
            f"got {len(validation) - 1}."
        )
    unseen = set(validation) - set(vocabulary)
    if unseen:
        raise RuntimeError(
            f"Full Enwik8 vocabulary does not cover validation split: "
            f"{len(unseen)} unseen characters."
        )
    return training, validation, vocabulary


def _make_model(author_model, vocabulary_size: int, device: torch.device):
    config = author_model.GPTConfig(
        vocab_size=vocabulary_size,
        ctx_len=CONTEXT_LENGTH,
        model_type=MODEL_TYPE,
        n_layer=N_LAYER,
        n_embd=N_EMBD,
    )
    model = author_model.GPT(config)
    for block in model.blocks:
        block.lif1 = smoke.make_current_lif("cupy")
        block.lif2 = smoke.make_current_lif("cupy")
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != FULL_ENWIK8_PARAMETER_COUNT:
        raise RuntimeError(
            f"Expected {FULL_ENWIK8_PARAMETER_COUNT} parameters, got {parameter_count}."
        )
    return model


def _batch(
    text: str,
    vocabulary: tuple[str, ...],
    offset: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = smoke._make_shifted_batch(
        text, vocabulary, (offset,), CONTEXT_LENGTH
    )
    return inputs.to(device), targets.to(device)


def _evaluate_bpc(
    model: torch.nn.Module,
    validation: str,
    vocabulary: tuple[str, ...],
    device: torch.device,
) -> float:
    model.eval()
    character_to_id = {character: index for index, character in enumerate(vocabulary)}
    total_nll = 0.0
    total_targets = 0
    for offset, target_tokens in _validation_windows(len(validation)):
        inputs, targets = _variable_length_batch(
            validation,
            character_to_id,
            offset,
            target_tokens,
            device,
        )
        functional.reset_net(model)
        with torch.no_grad():
            loss = model(inputs, targets)
        functional.reset_net(model)
        total_nll += loss.item() * target_tokens
        total_targets += target_tokens
    value = total_nll / total_targets / math.log(2.0)
    if not math.isfinite(value):
        raise RuntimeError("Validation BPC is not finite.")
    return value


def _save_pilot_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    training_wall_seconds: float,
    revision: str,
    vocabulary: tuple[str, ...],
) -> None:
    smoke._save_checkpoint(
        path,
        model,
        optimizer,
        completed_steps=completed_steps,
        next_batch_index=completed_steps,
        source_revision=revision,
        vocabulary=vocabulary,
        training_config=_training_config(),
        extra_state={"training_wall_seconds": training_wall_seconds},
    )


def _load_pilot_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    revision: str,
    vocabulary: tuple[str, ...],
) -> tuple[int, float]:
    restored = smoke._load_checkpoint(
        path,
        model,
        optimizer,
        device,
        training_config=_training_config(),
    )
    if restored["source_revision"] != revision or restored["vocabulary"] != vocabulary:
        raise RuntimeError("Pilot checkpoint source or vocabulary mismatch.")
    if restored["next_batch_index"] != restored["completed_steps"]:
        raise RuntimeError("Pilot checkpoint step and batch counters disagree.")
    training_wall_seconds = restored["extra_state"].get("training_wall_seconds")
    if (
        isinstance(training_wall_seconds, bool)
        or not isinstance(training_wall_seconds, (int, float))
        or not math.isfinite(training_wall_seconds)
        or training_wall_seconds < 0
    ):
        raise RuntimeError("Pilot checkpoint training wall time is invalid.")
    return restored["completed_steps"], float(training_wall_seconds)


def run(
    spikegpt_root: Path,
    data_path: Path,
    output_dir: Path,
    declared_revision: str | None,
    resume_checkpoint: Path | None,
    max_steps: int,
    max_minutes: float,
    checkpoint_every: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SpikeGPT training pilot.")
    try:
        import cupy  # noqa: F401
    except ImportError as exc:
        logging.info("SpikeGPT training pilot requires CuPy: %s", exc)
        raise ImportError("CuPy is required for the SpikeGPT training pilot.") from exc

    root = spikegpt_root.resolve()
    revision = smoke._verify_source(root, declared_revision)
    checkpoint_path = smoke._prepare_checkpoint_path(output_dir.resolve())
    training, validation, vocabulary = _read_data(data_path.resolve())
    author_model = smoke._load_author_model(root, require_wkv=True)
    device = torch.device("cuda")
    smoke._seed_everything()
    model = _make_model(author_model, len(vocabulary), device)
    optimizer = smoke._make_optimizer(model)
    completed_steps = 0
    training_wall_seconds = 0.0
    if resume_checkpoint is not None:
        completed_steps, training_wall_seconds = _load_pilot_checkpoint(
            resume_checkpoint.resolve(),
            model,
            optimizer,
            device,
            revision,
            vocabulary,
        )
    if max_steps - completed_steps < 2:
        raise RuntimeError(
            "Pilot requires at least two remaining steps to measure training direction."
        )

    validation_bpc_before = _evaluate_bpc(model, validation, vocabulary, device)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    max_seconds = max_minutes * 60.0
    losses = []
    results = []
    maximum_offset = len(training) - CONTEXT_LENGTH - 1
    while completed_steps < max_steps:
        if training_wall_seconds + time.monotonic() - start >= max_seconds:
            break
        offset = int(np.random.randint(0, maximum_offset))
        inputs, targets = _batch(training, vocabulary, offset, device)
        result = smoke._train_step(model, optimizer, inputs, targets)
        completed_steps += 1
        losses.append(result["loss"])
        results.append(result)
        print(
            f"step={completed_steps} loss={result['loss']:.8f} "
            f"bpc={_bpc_from_nll(result['loss']):.8f} "
            f"gradient_l2={result['gradient_l2']:.8f} "
            f"latency_ms={result['latency_ms']:.2f}",
            flush=True,
        )
        if len(losses) == 1 or completed_steps % checkpoint_every == 0:
            _save_pilot_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                completed_steps,
                training_wall_seconds + time.monotonic() - start,
                revision,
                vocabulary,
            )

    segment_training_seconds = time.monotonic() - start
    training_wall_seconds += segment_training_seconds
    validation_bpc_after = _evaluate_bpc(model, validation, vocabulary, device)
    _save_pilot_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        completed_steps,
        training_wall_seconds,
        revision,
        vocabulary,
    )
    if len(losses) < 2:
        raise RuntimeError("Pilot stopped before training direction could be measured.")
    window_size = min(WINDOW_SIZE, len(losses) // 2)
    improved, start_bpc, end_bpc = _training_direction(losses, window_size)
    peak_memory_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    measured_seconds = sum(result["latency_ms"] for result in results) / 1000.0
    if not improved:
        raise RuntimeError(
            f"Pilot training BPC did not improve: {start_bpc} -> {end_bpc}."
        )

    print(f"source_revision={revision}")
    print(
        f"model=SpikeGPT parameter_count={FULL_ENWIK8_PARAMETER_COUNT} "
        f"layers={N_LAYER} hidden={N_EMBD} context={CONTEXT_LENGTH} "
        f"backend=current-cupy precision=float32 "
        f"tokenizer=utf8-character-sorted-full-enwik8-v1 vocab_size={len(vocabulary)}"
    )
    print(
        f"training segment_steps={len(losses)} completed_steps={completed_steps} "
        f"window={window_size} start_bpc={start_bpc:.8f} "
        f"end_bpc={end_bpc:.8f} improved={improved}"
    )
    print(
        f"validation full_batches={VALIDATION_BATCHES} "
        f"target_tokens={VALIDATION_TARGET_TOKENS} "
        f"bpc_before={validation_bpc_before:.8f} "
        f"bpc_after={validation_bpc_after:.8f}"
    )
    print(
        f"state spike_rate={results[-1]['spike_rate']:.8f} "
        f"membrane_abs_mean={results[-1]['membrane_abs_mean']:.8f} "
        f"membrane_abs_max={results[-1]['membrane_abs_max']:.8f} "
        f"reset_ok={all(result['reset_ok'] for result in results)}"
    )
    print(
        f"checkpoint={checkpoint_path} "
        f"resumed_from={resume_checkpoint.resolve() if resume_checkpoint else 'none'}"
    )
    print(
        "performance=pilot_measurement "
        f"tokens_per_second={len(losses) * CONTEXT_LENGTH / measured_seconds:.2f} "
        f"mean_step_ms={measured_seconds * 1000 / len(losses):.2f} "
        f"peak_memory_mib={peak_memory_mib:.2f} "
        f"segment_training_wall_seconds={segment_training_seconds:.2f} "
        f"cumulative_training_wall_seconds={training_wall_seconds:.2f} "
        f"stop={'max_steps' if completed_steps == max_steps else 'max_minutes'}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded SpikeGPT 47M Enwik8 context-1024 pilot."
    )
    parser.add_argument("--spikegpt-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not math.isfinite(args.max_minutes):
        print("Pilot max-minutes must be finite.", file=sys.stderr)
        return 1
    if args.max_steps <= 0 or args.max_minutes <= 0 or args.checkpoint_every <= 0:
        print("Pilot resource limits must be positive.", file=sys.stderr)
        return 1
    try:
        run(
            args.spikegpt_root,
            args.data,
            args.output_dir,
            args.source_revision,
            args.resume_checkpoint,
            args.max_steps,
            args.max_minutes,
            args.checkpoint_every,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
