import argparse
import hashlib
import importlib
import io
import os
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import torch

from benchmark.snn_llm._spikegpt_author import (
    SPIKEGPT_REVISION,
    _verify_source,
)
from spikingjelly.activation_based import neuron


CHECKPOINT_REPOSITORY = "ridger/SpikeGPT-OpenWebText-216M"
CHECKPOINT_REVISION = "4039295cca3da1df0e5871f4bc7727b227496132"
CHECKPOINT_SHA256 = "024d2dab5f71b6bd8d4f3ef38bf9bd61c54b2ea8f9efe4b57517bf9b6b0328de"
TOKENIZER_SHA256 = "56ac4821e129d2c520fdaba60abd920fa852ada51b45c0dd52bbb6bd8c985ade"
N_LAYER = 18
N_EMBD = 768
VOCAB_SIZE = 50277
CONTEXT_LENGTH = 1024
PARAMETER_COUNT = 215399424
GENERATION_STEPS = 16
PROMPT = (
    "Prehistoric man sketched an incredible array of prehistoric beasts on the "
    "rough limestone walls of a cave."
)
LOGIT_ATOL = 1e-4
LOGIT_RTOL = 1e-3


def _make_current_lif() -> neuron.LIFNode:
    lif = neuron.LIFNode(
        tau=2.0,
        decay_input=True,
        v_threshold=1.0,
        v_reset=0.0,
        detach_reset=False,
        step_mode="s",
        backend="torch",
    )
    lif.eval()
    return lif


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SpikeGPT 216M legacy and current LIF inference."
    )
    parser.add_argument(
        "--spikegpt-root",
        type=Path,
        required=True,
        help="Path to the fixed SpikeGPT source checkout or synchronized tree.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the fixed SpikeGPT-216M.pth checkpoint.",
    )
    parser.add_argument(
        "--source-revision",
        help="Required when the SpikeGPT source tree has no .git directory.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifacts(
    spikegpt_root: Path, checkpoint: Path, declared_revision: str | None
) -> tuple[Path, Path, str]:
    root = spikegpt_root.resolve()
    revision = _verify_source(root, declared_revision)
    tokenizer = root / "20B_tokenizer.json"
    required_files = (root / "src" / "model_run.py", tokenizer, checkpoint)
    for path in required_files:
        if not path.is_file():
            raise RuntimeError(f"Required SpikeGPT artifact not found: {path}")
    if checkpoint.suffix != ".pth":
        raise RuntimeError("SpikeGPT checkpoint must use the .pth suffix.")

    checkpoint_digest = _sha256(checkpoint)
    if checkpoint_digest != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: expected {CHECKPOINT_SHA256}, "
            f"got {checkpoint_digest}."
        )
    tokenizer_digest = _sha256(tokenizer)
    if tokenizer_digest != TOKENIZER_SHA256:
        raise RuntimeError(
            f"Tokenizer SHA-256 mismatch: expected {TOKENIZER_SHA256}, "
            f"got {tokenizer_digest}."
        )
    return root, tokenizer, revision


def _load_author_runtime(root: Path):
    existing_src = sys.modules.get("src")
    if existing_src is not None:
        src_paths = [
            Path(path).resolve() for path in getattr(existing_src, "__path__", [])
        ]
        if (root / "src").resolve() not in src_paths:
            raise RuntimeError(
                "A different top-level 'src' package is already imported."
            )

    root_string = str(root)
    previous_jit = os.environ.get("RWKV_JIT_ON")
    os.environ["RWKV_JIT_ON"] = "0"
    sys.path.insert(0, root_string)
    try:
        importlib.invalidate_caches()
        model_run = importlib.import_module("src.model_run")
        author_utils = importlib.import_module("src.utils")
    finally:
        sys.path.remove(root_string)
        if previous_jit is None:
            os.environ.pop("RWKV_JIT_ON", None)
        else:
            os.environ["RWKV_JIT_ON"] = previous_jit

    expected_model = (root / "src" / "model_run.py").resolve()
    if Path(model_run.__file__).resolve() != expected_model:
        raise RuntimeError(
            f"Imported unexpected SpikeGPT runtime: {model_run.__file__}"
        )
    return model_run, author_utils


def _model_args(checkpoint: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        RUN_DEVICE="cuda",
        FLOAT_MODE="fp32",
        MODEL_NAME=str(checkpoint.resolve())[: -len(checkpoint.suffix)],
        n_layer=N_LAYER,
        n_embd=N_EMBD,
        ctx_len=CONTEXT_LENGTH,
        vocab_size=VOCAB_SIZE,
        head_qk=0,
        pre_ffn=0,
        grad_cp=0,
        my_pos_emb=0,
    )


def _make_state(lif_factory, device: torch.device):
    state = torch.zeros(N_LAYER * 5, N_EMBD, device=device, dtype=torch.float32)
    state[4::5].fill_(-1e30)
    return (
        state,
        [lif_factory() for _ in range(N_LAYER)],
        [lif_factory() for _ in range(N_LAYER)],
    )


def _selection_token(logits: torch.Tensor) -> int:
    selectable = logits.clone()
    selectable[0] = -torch.inf
    return int(selectable.argmax().item())


def _spike_hooks(attention_lifs, ffn_lifs):
    spike_sums = [[None, None] for _ in range(N_LAYER)]
    element_counts = [[0, 0] for _ in range(N_LAYER)]
    handles = []

    def hook(layer_id, kind):
        def record(module, inputs, output):
            value = output.detach().sum()
            current = spike_sums[layer_id][kind]
            spike_sums[layer_id][kind] = value if current is None else current + value
            element_counts[layer_id][kind] += output.numel()

        return record

    for layer_id, (att_lif, ffn_lif) in enumerate(zip(attention_lifs, ffn_lifs)):
        handles.append(att_lif.register_forward_hook(hook(layer_id, 0)))
        handles.append(ffn_lif.register_forward_hook(hook(layer_id, 1)))
    return spike_sums, element_counts, handles


def _spike_rates(spike_sums, element_counts) -> list[dict]:
    return [
        {
            "layer": layer_id,
            "attention": spike_sums[layer_id][0].item() / element_counts[layer_id][0],
            "ffn": spike_sums[layer_id][1].item() / element_counts[layer_id][1],
        }
        for layer_id in range(N_LAYER)
    ]


def _state_bytes(state: torch.Tensor, attention_lifs, ffn_lifs) -> int:
    size = state.numel() * state.element_size()
    for lif in (*attention_lifs, *ffn_lifs):
        if isinstance(lif.v, torch.Tensor):
            size += lif.v.numel() * lif.v.element_size()
    return size


def _execute(
    model,
    prompt_tokens: list[int],
    lif_factory,
    forced_tokens: list[int] | None = None,
) -> dict:
    device = torch.device("cuda")
    state, attention_lifs, ffn_lifs = _make_state(lif_factory, device)
    spike_sums, element_counts, handles = _spike_hooks(attention_lifs, ffn_lifs)
    context = []
    logits_history = []
    predicted_tokens = []
    try:
        with torch.no_grad():
            for token in prompt_tokens:
                context.append(token)
                logits, state, attention_lifs, ffn_lifs = model.forward(
                    context, state, attention_lifs, ffn_lifs
                )
                logits_history.append(logits.detach().cpu())

            for step in range(GENERATION_STEPS):
                predicted = _selection_token(logits)
                predicted_tokens.append(predicted)
                token = predicted if forced_tokens is None else forced_tokens[step]
                context.append(token)
                logits, state, attention_lifs, ffn_lifs = model.forward(
                    context, state, attention_lifs, ffn_lifs
                )
                logits_history.append(logits.detach().cpu())
    finally:
        for handle in handles:
            handle.remove()

    return {
        "logits": logits_history,
        "tokens": predicted_tokens,
        "spike_rates": _spike_rates(spike_sums, element_counts),
        "finite": all(torch.isfinite(value).all().item() for value in logits_history),
    }


def _benchmark(
    model, prompt_tokens: list[int], generated_tokens: list[int], lif_factory
):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    state, attention_lifs, ffn_lifs = _make_state(lif_factory, device)
    context = []

    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for token in prompt_tokens:
            context.append(token)
            logits, state, attention_lifs, ffn_lifs = model.forward(
                context, state, attention_lifs, ffn_lifs
            )
        end.record()
        end.synchronize()
        prefill_ms = start.elapsed_time(end)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for token in generated_tokens:
            context.append(token)
            logits, state, attention_lifs, ffn_lifs = model.forward(
                context, state, attention_lifs, ffn_lifs
            )
        end.record()
        end.synchronize()
        decode_ms = start.elapsed_time(end)

    peak_memory_delta = torch.cuda.max_memory_allocated() - baseline_memory
    return {
        "resident_model_mib": baseline_memory / (1024 * 1024),
        "prefill_ms": prefill_ms,
        "prefill_tokens_per_second": len(prompt_tokens) * 1000.0 / prefill_ms,
        "decode_ms": decode_ms,
        "decode_tokens_per_second": len(generated_tokens) * 1000.0 / decode_ms,
        "state_mib": _state_bytes(state, attention_lifs, ffn_lifs) / (1024 * 1024),
        "peak_memory_delta_mib": peak_memory_delta / (1024 * 1024),
    }


def _compare_runs(reference: dict, candidate: dict) -> dict:
    reference_logits = torch.stack(reference["logits"])
    candidate_logits = torch.stack(candidate["logits"])
    difference = (reference_logits - candidate_logits).abs()
    return {
        "logits_close": torch.allclose(
            reference_logits,
            candidate_logits,
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        ),
        "tokens_match": reference["tokens"] == candidate["tokens"],
        "logit_max_abs": difference.max().item(),
    }


def _print_report(
    revision: str,
    checkpoint: Path,
    prompt_tokens: list[int],
    generated_text: str,
    reference: dict,
    candidate: dict,
    comparison: dict,
    replay_comparison: dict,
    performance: dict[str, dict],
) -> None:
    dense_mac_per_token = 13 * N_LAYER * N_EMBD * N_EMBD + VOCAB_SIZE * N_EMBD
    print(f"source_revision={revision}")
    print(
        f"checkpoint_repository={CHECKPOINT_REPOSITORY} "
        f"checkpoint_revision={CHECKPOINT_REVISION}"
    )
    print(f"checkpoint_path={checkpoint.resolve()}")
    print(f"checkpoint_sha256={CHECKPOINT_SHA256}")
    print(f"tokenizer_sha256={TOKENIZER_SHA256}")
    print(
        f"config=layers:{N_LAYER},hidden:{N_EMBD},vocab:{VOCAB_SIZE},"
        f"context:{CONTEXT_LENGTH},parameters:{PARAMETER_COUNT},dtype:float32"
    )
    print(
        f"prompt={PROMPT!r} prompt_tokens={len(prompt_tokens)} "
        f"generation_steps={GENERATION_STEPS} decoding=greedy"
    )
    print(
        f"comparison=logit_max_abs:{comparison['logit_max_abs']:.3e},"
        f"logits_close:{comparison['logits_close']},"
        f"tokens_match:{comparison['tokens_match']}"
    )
    print(
        f"fresh_state_replay=logit_max_abs:{replay_comparison['logit_max_abs']:.3e},"
        f"logits_close:{replay_comparison['logits_close']},"
        f"tokens_match:{replay_comparison['tokens_match']}"
    )
    print(f"generated_token_ids={reference['tokens']}")
    print(f"generated_text={generated_text!r}")
    for name, result in performance.items():
        print(
            f"{name}=resident_model_mib:{result['resident_model_mib']:.3f},"
            f"prefill_ms:{result['prefill_ms']:.3f},"
            f"prefill_tokens_per_second:{result['prefill_tokens_per_second']:.2f},"
            f"decode_ms:{result['decode_ms']:.3f},"
            f"decode_tokens_per_second:{result['decode_tokens_per_second']:.2f},"
            f"state_mib:{result['state_mib']:.3f},"
            f"peak_memory_delta_mib:{result['peak_memory_delta_mib']:.3f}"
        )
    print("current_spike_rates:")
    for value in candidate["spike_rates"]:
        print(
            f"  layer={value['layer']} attention={value['attention']:.6f} "
            f"ffn={value['ffn']:.6f}"
        )
    print(
        f"operation_assumption=dense_mac_per_token:{dense_mac_per_token},"
        "excludes:layernorm_and_elementwise,sparse_ac:not_estimated,"
        "reason:author_runtime_uses_dense_torch_matmul"
    )


def run(spikegpt_root: Path, checkpoint: Path, declared_revision: str | None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SpikeGPT 216M comparison.")
    checkpoint = checkpoint.resolve()
    root, tokenizer_path, revision = _verify_artifacts(
        spikegpt_root, checkpoint, declared_revision
    )
    model_run, author_utils = _load_author_runtime(root)
    print(f"loading_checkpoint={checkpoint}", flush=True)
    with redirect_stdout(io.StringIO()):
        model = model_run.RWKV_RNN(_model_args(checkpoint))
        tokenizer = author_utils.TOKENIZER(
            [str(tokenizer_path), str(tokenizer_path)], UNKNOWN_CHAR=None
        )
    prompt_tokens = tokenizer.tokenizer.encode(PROMPT)
    if not prompt_tokens or len(prompt_tokens) >= CONTEXT_LENGTH - GENERATION_STEPS:
        raise RuntimeError(
            "Fixed prompt tokenization is outside the supported context."
        )
    if tokenizer.tokenizer.decode([187]) != "\n":
        raise RuntimeError("Unexpected tokenizer newline token.")

    legacy = _execute(model, prompt_tokens, model_run.neuron.LIFNode)
    current = _execute(
        model,
        prompt_tokens,
        _make_current_lif,
        forced_tokens=legacy["tokens"],
    )
    comparison = _compare_runs(legacy, current)
    current_replay = _execute(
        model,
        prompt_tokens,
        _make_current_lif,
        forced_tokens=legacy["tokens"],
    )
    replay_comparison = _compare_runs(current, current_replay)
    performance = {
        "legacy": _benchmark(
            model, prompt_tokens, legacy["tokens"], model_run.neuron.LIFNode
        ),
        "current": _benchmark(
            model, prompt_tokens, legacy["tokens"], _make_current_lif
        ),
    }
    generated_text = tokenizer.tokenizer.decode(legacy["tokens"])
    _print_report(
        revision,
        checkpoint,
        prompt_tokens,
        generated_text,
        legacy,
        current,
        comparison,
        replay_comparison,
        performance,
    )
    if (
        not legacy["finite"]
        or not current["finite"]
        or not comparison["logits_close"]
        or not comparison["tokens_match"]
        or not replay_comparison["logits_close"]
        or not replay_comparison["tokens_match"]
    ):
        raise RuntimeError("SpikeGPT 216M checkpoint comparison acceptance failed.")
    print("acceptance=PASS")


def main() -> None:
    args = _parse_args()
    run(args.spikegpt_root, args.checkpoint, args.source_revision)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
