import argparse
import hashlib
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from benchmark.snn_llm._spikegpt_author import (
    SPIKEGPT_REVISION,
    _load_author_model,
    _verify_source,
    make_current_lif,
)
from spikingjelly.activation_based import functional


SEED = 42
CONTEXT_LENGTH = 32
N_LAYER = 12
N_EMBD = 512
MODEL_TYPE = "RWKV"
BATCH_SIZE = 1
LEARNING_RATE = 6e-4
BETAS = (0.9, 0.99)
EPS = 4e-9
GRAD_NORM_CLIP = 1.0
OVERFIT_STEPS = 8
SHORT_TRAIN_STEPS = 2
EXPECTED_DATA_BYTES = 100_000_000
EXPECTED_DATA_MD5 = "a1fa5ffddb56f4953e226637dabbb36a"
EXPECTED_DATA_SHA1 = "57b8363b814821dc9d47aa4d41f58733519076b2"
EXPECTED_VOCAB_SIZE = 5458
EXPECTED_PARAMETER_COUNT = 46_553_088


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed SpikeGPT 46M Enwik8 training smoke."
    )
    parser.add_argument(
        "--spikegpt-root",
        type=Path,
        required=True,
        help="Path to the fixed SpikeGPT source tree.",
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to the official enwik8 file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the resumable checkpoint.",
    )
    parser.add_argument(
        "--source-revision",
        help="Required when the SpikeGPT source tree has no .git directory.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Continue a prior smoke checkpoint into the new output directory.",
    )
    return parser.parse_args()


def _build_vocabulary(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(text)))


def _make_shifted_batch(
    text: str,
    vocabulary: tuple[str, ...],
    offsets: tuple[int, ...],
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    character_to_id = {character: index for index, character in enumerate(vocabulary)}
    sequences = []
    for offset in offsets:
        sample = text[offset : offset + context_length + 1]
        if len(sample) != context_length + 1:
            raise ValueError(f"Sample at offset {offset} is shorter than the context.")
        sequences.append([character_to_id[character] for character in sample])
    tokens = torch.tensor(sequences, dtype=torch.long)
    return tokens[:, :-1], tokens[:, 1:]


def _make_config(author_model, vocab_size: int):
    return author_model.GPTConfig(
        vocab_size=vocab_size,
        ctx_len=CONTEXT_LENGTH,
        model_type=MODEL_TYPE,
        n_layer=N_LAYER,
        n_embd=N_EMBD,
    )


def _checkpoint_config() -> dict:
    return {
        "backend": "cupy",
        "batch_size": BATCH_SIZE,
        "betas": BETAS,
        "context_length": CONTEXT_LENGTH,
        "eps": EPS,
        "grad_norm_clip": GRAD_NORM_CLIP,
        "learning_rate": LEARNING_RATE,
        "model_type": MODEL_TYPE,
        "n_embd": N_EMBD,
        "n_layer": N_LAYER,
        "reset": "after-forward-before-backward",
        "seed": SEED,
    }


def _rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["keys"].cpu().numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    completed_steps: int,
    next_batch_index: int,
    source_revision: str,
    vocabulary: tuple[str, ...],
    training_config: dict | None = None,
    extra_state: dict | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_step": getattr(model, "step", None),
        "completed_steps": completed_steps,
        "next_batch_index": next_batch_index,
        "source_revision": source_revision,
        "vocabulary": vocabulary,
        "config": _checkpoint_config() if training_config is None else training_config,
        "extra_state": {} if extra_state is None else extra_state,
        "rng": _rng_state(),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    training_config: dict | None = None,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint payload must be a mapping.")
    required_keys = {
        "model",
        "optimizer",
        "model_step",
        "completed_steps",
        "next_batch_index",
        "source_revision",
        "vocabulary",
        "config",
        "rng",
    }
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        raise RuntimeError(f"Checkpoint is missing keys: {sorted(missing_keys)}")
    if not isinstance(checkpoint["model"], dict) or not isinstance(
        checkpoint["optimizer"], dict
    ):
        raise RuntimeError("Checkpoint model and optimizer states must be mappings.")
    if not isinstance(checkpoint["rng"], dict):
        raise RuntimeError("Checkpoint RNG state must be a mapping.")
    extra_state = checkpoint.get("extra_state", {})
    if not isinstance(extra_state, dict):
        raise RuntimeError("Checkpoint extra state must be a mapping.")
    if (
        type(checkpoint["completed_steps"]) is not int
        or type(checkpoint["next_batch_index"]) is not int
    ):
        raise RuntimeError("Checkpoint step counters must be integers.")
    if checkpoint["completed_steps"] < 0 or checkpoint["next_batch_index"] < 0:
        raise RuntimeError("Checkpoint step counters must be nonnegative.")
    if (
        checkpoint["model_step"] is not None
        and type(checkpoint["model_step"]) is not int
    ):
        raise RuntimeError("Checkpoint model step must be an integer or null.")
    if not isinstance(checkpoint["source_revision"], str):
        raise RuntimeError("Checkpoint source revision must be a string.")
    if not isinstance(checkpoint["vocabulary"], (tuple, list)) or not all(
        isinstance(character, str) for character in checkpoint["vocabulary"]
    ):
        raise RuntimeError("Checkpoint vocabulary must be a character sequence.")
    expected_config = (
        _checkpoint_config() if training_config is None else training_config
    )
    if checkpoint["config"] != expected_config:
        raise RuntimeError(
            "Checkpoint training configuration does not match the smoke."
        )
    model.load_state_dict(checkpoint["model"])
    if checkpoint["model_step"] is not None:
        model.step = checkpoint["model_step"]
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    try:
        _restore_rng_state(checkpoint["rng"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Checkpoint RNG state is invalid.") from exc
    return {
        "completed_steps": checkpoint["completed_steps"],
        "next_batch_index": checkpoint["next_batch_index"],
        "source_revision": checkpoint["source_revision"],
        "vocabulary": tuple(checkpoint["vocabulary"]),
        "extra_state": extra_state,
    }


def _prepare_checkpoint_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    if checkpoint.exists():
        raise RuntimeError(f"Checkpoint exists; refusing to overwrite: {checkpoint}")
    return checkpoint


def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _read_training_text(path: Path) -> tuple[str, tuple[str, ...]]:
    raw = path.read_bytes()
    if len(raw) != EXPECTED_DATA_BYTES:
        raise RuntimeError(
            f"Expected {EXPECTED_DATA_BYTES} Enwik8 bytes, got {len(raw)}."
        )
    if hashlib.md5(raw).hexdigest() != EXPECTED_DATA_MD5:
        raise RuntimeError("Enwik8 MD5 mismatch.")
    if hashlib.sha1(raw).hexdigest() != EXPECTED_DATA_SHA1:
        raise RuntimeError("Enwik8 SHA-1 mismatch.")
    training_text = raw[:90_000_000].decode("utf-8")
    vocabulary = _build_vocabulary(training_text)
    if len(vocabulary) != EXPECTED_VOCAB_SIZE:
        actual_size = len(vocabulary)
        raise RuntimeError(
            f"Expected {EXPECTED_VOCAB_SIZE} training characters, got {actual_size}."
        )
    return training_text, vocabulary


def _make_model(author_model, vocabulary_size: int, device: torch.device):
    model = author_model.GPT(_make_config(author_model, vocabulary_size))
    for block in model.blocks:
        block.lif1 = make_current_lif("cupy")
        block.lif2 = make_current_lif("cupy")
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETER_COUNT} parameters, got {parameter_count}."
        )
    return model


def _make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPS)


def _lif_nodes(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [lif for block in model.blocks for lif in (block.lif1, block.lif2)]


def _states_are_reset(model: torch.nn.Module) -> bool:
    for lif in _lif_nodes(model):
        value = lif.v
        if isinstance(value, torch.Tensor):
            if not torch.equal(value, torch.zeros_like(value)):
                return False
        elif value != 0.0:
            return False
    return True


def _evaluate_loss(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> float:
    model.eval()
    functional.reset_net(model)
    with torch.no_grad():
        loss = model(inputs, targets)
    functional.reset_net(model)
    return loss.item()


def _reset_replay_matches(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> bool:
    model.eval()
    functional.reset_net(model)
    with torch.no_grad():
        first = model(inputs, targets)
    functional.reset_net(model)
    with torch.no_grad():
        replay = model(inputs, targets)
    functional.reset_net(model)
    return torch.equal(first, replay) and _states_are_reset(model)


def _training_metrics_are_finite(
    *,
    loss: float,
    gradient_l2: float,
    spike_rate: float,
    membrane_abs_mean: float,
    membrane_abs_max: float,
) -> bool:
    return all(
        math.isfinite(value)
        for value in (
            loss,
            gradient_l2,
            spike_rate,
            membrane_abs_mean,
            membrane_abs_max,
        )
    )


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict:
    spike_sum = 0.0
    spike_count = 0

    def capture_spikes(module, module_inputs, output):
        nonlocal spike_sum, spike_count
        spike_sum += output.detach().float().sum().item()
        spike_count += output.numel()

    handles = [lif.register_forward_hook(capture_spikes) for lif in _lif_nodes(model)]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        loss = model(inputs, targets)
    finally:
        for handle in handles:
            handle.remove()

    voltages = [lif.v_seq.detach().float() for lif in _lif_nodes(model)]
    membrane_abs_sum = sum(value.abs().sum().item() for value in voltages)
    membrane_count = sum(value.numel() for value in voltages)
    membrane_abs_max = max(value.abs().max().item() for value in voltages)
    functional.reset_net(model)
    reset_ok = _states_are_reset(model)
    loss.backward()
    gradient_l2 = torch.nn.utils.clip_grad_norm_(
        model.parameters(), GRAD_NORM_CLIP
    ).item()
    optimizer.step()
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    spike_rate = spike_sum / spike_count
    membrane_abs_mean = membrane_abs_sum / membrane_count
    finite = _training_metrics_are_finite(
        loss=loss.item(),
        gradient_l2=gradient_l2,
        spike_rate=spike_rate,
        membrane_abs_mean=membrane_abs_mean,
        membrane_abs_max=membrane_abs_max,
    )
    if not finite or not reset_ok:
        raise RuntimeError(
            f"Training step failed: finite={finite}, reset_ok={reset_ok}."
        )
    return {
        "loss": loss.item(),
        "gradient_l2": gradient_l2,
        "latency_ms": latency_ms,
        "spike_rate": spike_rate,
        "membrane_abs_mean": membrane_abs_mean,
        "membrane_abs_max": membrane_abs_max,
        "reset_ok": reset_ok,
    }


def _state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _state_dicts_equal(
    expected: dict[str, torch.Tensor], model: torch.nn.Module
) -> bool:
    actual = model.state_dict()
    return expected.keys() == actual.keys() and all(
        torch.equal(value, actual[name].detach().cpu())
        for name, value in expected.items()
    )


def _resume_training(
    resume_checkpoint: Path,
    output_checkpoint: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    revision: str,
    vocabulary: tuple[str, ...],
) -> tuple[dict, int, int]:
    restored = _load_checkpoint(resume_checkpoint, model, optimizer, device)
    if restored["source_revision"] != revision or restored["vocabulary"] != vocabulary:
        raise RuntimeError("Resume checkpoint source or vocabulary mismatch.")
    batch_index = restored["next_batch_index"]
    if not 0 <= batch_index < len(batches):
        raise RuntimeError(f"No smoke batch is defined for index {batch_index}.")
    result = _train_step(model, optimizer, *batches[batch_index])
    completed_steps = restored["completed_steps"] + 1
    _save_checkpoint(
        output_checkpoint,
        model,
        optimizer,
        completed_steps=completed_steps,
        next_batch_index=batch_index + 1,
        source_revision=revision,
        vocabulary=vocabulary,
    )
    return result, completed_steps, batch_index


def run(
    spikegpt_root: Path,
    data_path: Path,
    output_dir: Path,
    declared_revision: str | None,
    resume_checkpoint: Path | None = None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SpikeGPT training smoke.")
    try:
        import cupy  # noqa: F401
    except ImportError as exc:
        logging.info("SpikeGPT training smoke requires CuPy: %s", exc)
        raise ImportError("CuPy is required for the SpikeGPT training smoke.") from exc

    revision = _verify_source(spikegpt_root.resolve(), declared_revision)
    checkpoint_path = _prepare_checkpoint_path(output_dir.resolve())
    training_text, vocabulary = _read_training_text(data_path.resolve())
    author_model = _load_author_model(spikegpt_root.resolve(), require_wkv=True)
    device = torch.device("cuda")
    fixed_inputs, fixed_targets = _make_shifted_batch(
        training_text, vocabulary, (0,), CONTEXT_LENGTH
    )
    fixed_inputs = fixed_inputs.to(device)
    fixed_targets = fixed_targets.to(device)
    offsets = (
        len(training_text) // 4,
        len(training_text) // 2,
        3 * len(training_text) // 4,
    )
    batches = [
        tuple(
            value.to(device)
            for value in _make_shifted_batch(
                training_text, vocabulary, (offset,), CONTEXT_LENGTH
            )
        )
        for offset in offsets
    ]

    if resume_checkpoint is not None:
        _seed_everything()
        torch.cuda.reset_peak_memory_stats()
        model = _make_model(author_model, len(vocabulary), device)
        optimizer = _make_optimizer(model)
        result, completed_steps, batch_index = _resume_training(
            resume_checkpoint.resolve(),
            checkpoint_path,
            model,
            optimizer,
            batches,
            device,
            revision,
            vocabulary,
        )
        peak_memory_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"source_revision={revision}")
        print(
            f"resumed_from={resume_checkpoint.resolve()} batch_index={batch_index} "
            f"loss={result['loss']:.8f} gradient_l2={result['gradient_l2']:.8f}"
        )
        print(f"checkpoint={checkpoint_path} completed_steps={completed_steps}")
        print(
            "performance=smoke_measurement "
            f"step_ms={result['latency_ms']:.2f} "
            f"peak_memory_mib={peak_memory_mib:.2f}"
        )
        return

    _seed_everything()
    model = _make_model(author_model, len(vocabulary), device)
    optimizer = _make_optimizer(model)
    replay_matches = _reset_replay_matches(model, fixed_inputs, fixed_targets)
    overfit_loss_before = _evaluate_loss(model, fixed_inputs, fixed_targets)
    overfit_steps = [
        _train_step(model, optimizer, fixed_inputs, fixed_targets)
        for _ in range(OVERFIT_STEPS)
    ]
    overfit_loss_after = _evaluate_loss(model, fixed_inputs, fixed_targets)
    if not replay_matches or not overfit_loss_after < overfit_loss_before:
        raise RuntimeError(
            "Overfit gate failed: "
            f"reset_replay={replay_matches}, "
            f"loss_before={overfit_loss_before}, loss_after={overfit_loss_after}."
        )
    del model, optimizer
    torch.cuda.empty_cache()

    _seed_everything()
    torch.cuda.reset_peak_memory_stats()
    model = _make_model(author_model, len(vocabulary), device)
    optimizer = _make_optimizer(model)
    short_steps = [
        _train_step(model, optimizer, *batches[index])
        for index in range(SHORT_TRAIN_STEPS)
    ]
    _save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        completed_steps=SHORT_TRAIN_STEPS,
        next_batch_index=SHORT_TRAIN_STEPS,
        source_revision=revision,
        vocabulary=vocabulary,
    )
    expected_continuation = _train_step(model, optimizer, *batches[SHORT_TRAIN_STEPS])
    expected_parameters = _state_dict_to_cpu(model)
    del model, optimizer
    torch.cuda.empty_cache()

    model = _make_model(author_model, len(vocabulary), device)
    optimizer = _make_optimizer(model)
    restored = _load_checkpoint(checkpoint_path, model, optimizer, device)
    restored_continuation = _train_step(
        model, optimizer, *batches[restored["next_batch_index"]]
    )
    exact_resume = (
        restored["source_revision"] == revision
        and restored["vocabulary"] == vocabulary
        and expected_continuation["loss"] == restored_continuation["loss"]
        and _state_dicts_equal(expected_parameters, model)
    )
    peak_memory_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    del model, optimizer
    torch.cuda.empty_cache()
    if not exact_resume:
        raise RuntimeError("Checkpoint continuation was not bitwise exact.")

    measured_steps = overfit_steps + short_steps + [expected_continuation]
    measured_tokens = len(measured_steps) * BATCH_SIZE * CONTEXT_LENGTH
    measured_seconds = sum(step["latency_ms"] for step in measured_steps) / 1000.0
    print(f"source_revision={revision}")
    print(
        f"data=enwik8 bytes={EXPECTED_DATA_BYTES} split=90/5/5 "
        f"tokenizer=utf8-character-sorted-train-v1 vocab_size={len(vocabulary)}"
    )
    print(
        f"model=SpikeGPT parameter_count={EXPECTED_PARAMETER_COUNT} layers={N_LAYER} "
        f"hidden={N_EMBD} context={CONTEXT_LENGTH} backend=current-cupy"
    )
    print(
        f"overfit loss_before={overfit_loss_before:.8f} "
        f"loss_after={overfit_loss_after:.8f} steps={OVERFIT_STEPS} "
        f"reset_replay={replay_matches}"
    )
    print(
        f"short_train losses={[step['loss'] for step in short_steps]} "
        f"grad_l2={[step['gradient_l2'] for step in short_steps]}"
    )
    print(
        f"state spike_rate={measured_steps[-1]['spike_rate']:.8f} "
        f"membrane_abs_mean={measured_steps[-1]['membrane_abs_mean']:.8f} "
        f"membrane_abs_max={measured_steps[-1]['membrane_abs_max']:.8f} "
        f"reset_ok={all(step['reset_ok'] for step in measured_steps)}"
    )
    print(
        f"checkpoint={checkpoint_path} completed_steps={SHORT_TRAIN_STEPS} "
        f"exact_resume={exact_resume}"
    )
    print(
        "performance=smoke_measurement "
        f"tokens_per_second={measured_tokens / measured_seconds:.2f} "
        f"mean_step_ms={measured_seconds * 1000 / len(measured_steps):.2f} "
        f"peak_memory_mib={peak_memory_mib:.2f}"
    )


def main() -> int:
    args = _parse_args()
    try:
        run(
            args.spikegpt_root,
            args.data,
            args.output_dir,
            args.source_revision,
            args.resume_checkpoint,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
