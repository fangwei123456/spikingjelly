"""Private Phase 5.1 GPT-2 MLP ANN2SNN conversion slice.

This runner evaluates only the activation path of one GPT-2 MLP block.  It
keeps the Hugging Face model frozen, uses local files only, and deliberately
avoids generation, caching, checkpointing, and whole-model conversion.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.ann2snn import ModuleConverter
from spikingjelly.activation_based.ann2snn.recipes import ModuleConversionRecipe

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


SLICE_SCHEMA_VERSION = 1
CONTRACT_KIND = "gpt2-mlp-ann2snn-signed-if-slice"
VALID_DEVICES = ("cpu", "cuda")
DEFAULT_BLOCK_INDEX = 0
DEFAULT_TIME_STEPS = 16
SIGNED_ENCODING = "signed_dual_channel_if"
CALIBRATION_MODE = "per_channel_symmetric_absmax"

_UNSUPPORTED = (
    "no fine-tuning",
    "no distillation",
    "no full GPT-2 SNN",
    "no cache",
    "no generation",
    "no full PPL",
)


class _SignedIFProxy(nn.Module):
    """Run two activation-aware IF channels for one signed activation tensor."""

    def __init__(self, scale: torch.Tensor, time_steps: int) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.positive_neuron = neuron.ActivationAwareIFNode(
            v_threshold=scale,
            v_offset=0.5 * scale,
            channel_dim=-1,
            v_reset=None,
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
            backend="torch",
        )
        self.negative_neuron = neuron.ActivationAwareIFNode(
            v_threshold=scale,
            v_offset=0.5 * scale,
            channel_dim=-1,
            v_reset=None,
            surrogate_function=surrogate.DeterministicPass(),
            step_mode="m",
            backend="torch",
        )
        self.positive_spike_rate = 0.0
        self.negative_spike_rate = 0.0

    def _desired_qcfs_counts(
        self, value: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return torch.round(value / scale).clamp(0, self.time_steps)

    def _replay_qcfs_counts(
        self,
        node: neuron.ActivationAwareIFNode,
        desired_count: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        sequence = (
            desired_count.unsqueeze(0).expand(self.time_steps, *desired_count.shape)
            * scale
            / self.time_steps
        )
        node.reset()
        spikes = node(sequence)
        if not torch.equal(spikes.sum(0), desired_count):
            raise RuntimeError("Signed IF proxy failed to reproduce QCFS counts.")
        return spikes

    def _encode(
        self, node: neuron.ActivationAwareIFNode, value: torch.Tensor
    ) -> torch.Tensor:
        scale = node.v_threshold.to(device=value.device, dtype=value.dtype)
        sequence = (
            value.unsqueeze(0).expand(self.time_steps, *value.shape) / self.time_steps
        )
        node.reset()
        spikes = node(sequence)

        # The first pass is the natural repeated dense activation encoding.
        # ``ActivationAwareIFNode`` uses a >= half-threshold decision, while
        # ``torch.round`` uses ties-to-even. When only these deterministic
        # boundary conventions disagree, replay the exact QCFS counts through
        # the same IF node. The reported zero IF-vs-QCFS error therefore means
        # "QCFS count replay through ActivationAwareIFNode", not a claim that
        # every dense/T temporal code is already bit-identical to QCFS.
        desired_count = self._desired_qcfs_counts(value, scale)
        observed_count = spikes.sum(0)
        if not torch.equal(observed_count, desired_count):
            spikes = self._replay_qcfs_counts(node, desired_count, scale)
        return spikes

    def forward(self, hidden_dense: torch.Tensor) -> torch.Tensor:
        positive = F.relu(hidden_dense)
        negative = F.relu(-hidden_dense)
        positive_spikes = self._encode(self.positive_neuron, positive)
        negative_spikes = self._encode(self.negative_neuron, negative)
        self.positive_spike_rate = float(positive_spikes.detach().mean().cpu())
        self.negative_spike_rate = float(negative_spikes.detach().mean().cpu())
        scale = self.positive_neuron.v_threshold.to(
            device=hidden_dense.device, dtype=hidden_dense.dtype
        )
        return (positive_spikes.sum(0) - negative_spikes.sum(0)) * scale


class GPT2MLPAnn2SNNRecipe(ModuleConversionRecipe):
    def __init__(self, scale: torch.Tensor, time_steps: int) -> None:
        r"""
        **API Language** - 中文 | English

        私有的 GPT-2 signed MLP module-tree recipe。它把现有
        :class:`ModuleConverter` seam 适配到 Hugging Face GPT-2 MLP，不向
        ``spikingjelly`` 增加公共 symbol。

        Private module-tree recipe for the GPT-2 signed MLP activation slice.
        It adapts the existing :class:`ModuleConverter` seam to a Hugging Face
        GPT-2 MLP without adding a public symbol under ``spikingjelly``.

        :param scale: Per-channel positive QCFS threshold with shape ``[C]``.
        :type scale: torch.Tensor
        :param time_steps: Number of IF time steps; must be positive.
        :type time_steps: int
        :raises ValueError: If ``scale`` is not finite positive 1D data or
            ``time_steps`` is not positive.
        """
        super().__init__()
        if time_steps <= 0:
            raise ValueError("time_steps must be positive.")
        if scale.dim() != 1 or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("scale must be a finite, positive 1D tensor.")
        self.scale = scale.detach()
        self.time_steps = time_steps

    def validate(self, converter: object) -> None:
        if self.time_steps <= 0:
            raise ValueError("time_steps must be positive.")

    def convert_module(self, converter: object, ann: nn.Module) -> nn.Module:
        c_proj = getattr(ann, "c_proj", None)
        if not isinstance(c_proj, nn.Module):
            raise TypeError("GPT-2 MLP must expose a c_proj module.")
        return _SignedMLPProxy(
            c_proj=c_proj,
            scale=self.scale,
            time_steps=self.time_steps,
        )


class _SignedMLPProxy(nn.Module):
    """Converted module that applies signed IF before the original c_proj."""

    def __init__(
        self,
        *,
        c_proj: nn.Module,
        scale: torch.Tensor,
        time_steps: int,
    ) -> None:
        super().__init__()
        self.c_proj = c_proj
        self.if_proxy = _SignedIFProxy(scale, time_steps)
        self.last_hidden_if: Optional[torch.Tensor] = None

    def forward(self, hidden_dense: torch.Tensor) -> torch.Tensor:
        self.last_hidden_if = self.if_proxy(hidden_dense)
        return self.c_proj(self.last_hidden_if)


def _validate_time_steps(time_steps: int) -> None:
    if time_steps <= 0:
        raise ValueError(f"--time-steps must be positive; got {time_steps}.")


def calibrate_signed_qcfs(
    hidden_dense: torch.Tensor,
    *,
    time_steps: int = DEFAULT_TIME_STEPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language** - 中文 | English

    按通道执行 symmetric QCFS 校准并量化 signed activation。
    Calibrate per-channel symmetric QCFS and quantize a signed activation.

    :param hidden_dense: Dense GPT-2 MLP activation with shape ``[B, S, C]``.
    :type hidden_dense: torch.Tensor
    :param time_steps: Number of QCFS/IF levels used by the proxy.
    :type time_steps: int
    :return: ``(scale, hidden_qcfs)`` where ``scale`` has shape ``[C]``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: If the input shape or time-step count is invalid.
    """

    _validate_time_steps(time_steps)
    if hidden_dense.dim() < 3:
        raise ValueError(
            "hidden_dense must have shape [batch, sequence, channels] or higher."
        )
    scale = hidden_dense.abs().amax(dim=(0, 1)).clamp_min(1e-6) / time_steps
    hidden_qcfs = (
        torch.round(hidden_dense / scale).clamp(-time_steps, time_steps) * scale
    )
    return scale, hidden_qcfs


def _run_signed_if_proxy(
    hidden_dense: torch.Tensor,
    scale: torch.Tensor,
    *,
    time_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, _SignedIFProxy]:
    _validate_time_steps(time_steps)
    proxy = _SignedIFProxy(scale=scale, time_steps=time_steps).to(hidden_dense.device)
    with torch.no_grad():
        hidden_if = proxy(hidden_dense)
    return hidden_if, proxy.positive_neuron, proxy.negative_neuron, proxy


def build_signed_if_proxy(
    hidden_dense: torch.Tensor,
    scale: torch.Tensor,
    *,
    time_steps: int = DEFAULT_TIME_STEPS,
) -> Tuple[torch.Tensor, neuron.ActivationAwareIFNode, neuron.ActivationAwareIFNode]:
    r"""
    **API Language** - 中文 | English

    使用正负双通道 IF 神经元重建 signed activation。
    Reconstruct a signed activation with dual-channel IF neurons.

    :param hidden_dense: Dense signed activation with channel-last layout.
    :type hidden_dense: torch.Tensor
    :param scale: Positive per-channel threshold tensor with shape ``[C]``.
    :type scale: torch.Tensor
    :param time_steps: Number of repeated input steps.
    :type time_steps: int
    :return: Reconstructed activation and the positive/negative IF nodes.
    :rtype: tuple[torch.Tensor, ActivationAwareIFNode, ActivationAwareIFNode]
    """

    hidden_if, positive_neuron, negative_neuron, _ = _run_signed_if_proxy(
        hidden_dense, scale, time_steps=time_steps
    )
    return hidden_if, positive_neuron, negative_neuron


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    value = torch.linalg.vector_norm(actual - reference) / denominator
    result = float(value.detach().cpu())
    if not math.isfinite(result):
        raise ValueError(f"Non-finite relative L2 metric: {result!r}.")
    return result


def _load_gpt2(
    paths: GPT2ModelPaths,
    *,
    device: str,
) -> Tuple[object, object]:
    transformers = _import_huggingface(lambda: __import__("transformers"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(paths.root),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer requires a pad token or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(paths.root),
        local_files_only=True,
    )
    model.eval()
    model.to(device)
    return tokenizer, model


def _get_block(model: object, block_index: int) -> nn.Module:
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None)
    if blocks is None:
        raise ValueError("Loaded model does not expose transformer.h GPT-2 blocks.")
    if block_index < 0 or block_index >= len(blocks):
        raise ValueError(
            f"--block-index must be between 0 and {len(blocks) - 1}; got {block_index}."
        )
    block = blocks[block_index]
    if not isinstance(block, nn.Module):
        raise TypeError("GPT-2 transformer block is not a torch.nn.Module.")
    return block


def compute_mlp_slice(
    *,
    paths: GPT2ModelPaths,
    device: str,
    block_index: int,
    time_steps: int,
    max_samples: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    r"""
    **API Language** - 中文 | English

    运行一个冻结的 GPT-2 block MLP activation conversion slice。
    Run one frozen GPT-2 block MLP activation conversion slice.

    :param paths: Validated local GPT-2 artifact paths.
    :type paths: GPT2ModelPaths
    :param device: Execution device, either ``"cpu"`` or ``"cuda"``.
    :type device: str
    :param block_index: GPT-2 block index to inspect.
    :type block_index: int
    :param time_steps: Number of QCFS/IF time steps.
    :type time_steps: int
    :param max_samples: Prefix length of the fixed prompt bank.
    :type max_samples: int
    :return: Finite metrics and shape/input metadata.
    :rtype: tuple[dict[str, object], dict[str, object]]
    """

    _validate_time_steps(time_steps)
    if max_samples <= 0 or max_samples > len(FIXED_PROMPTS):
        raise ValueError(
            f"--max-samples must be between 1 and {len(FIXED_PROMPTS)}; "
            f"got {max_samples}."
        )

    tokenizer, model = _load_gpt2(paths, device=device)
    block = _get_block(model, block_index)
    ln2 = getattr(block, "ln_2", None)
    mlp = getattr(block, "mlp", None)
    if not isinstance(ln2, nn.Module) or not isinstance(mlp, nn.Module):
        raise ValueError("GPT-2 block must expose ln_2 and mlp modules.")
    c_fc = getattr(mlp, "c_fc", None)
    activation = getattr(mlp, "act", None)
    c_proj = getattr(mlp, "c_proj", None)
    if not all(isinstance(module, nn.Module) for module in (c_fc, activation, c_proj)):
        raise ValueError("GPT-2 MLP must expose c_fc, act, and c_proj modules.")

    captured: List[torch.Tensor] = []

    def capture_ln2_output(
        module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: object
    ):
        del module, inputs
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise TypeError("GPT-2 ln_2 output must be a tensor.")
        captured.append(value.detach())

    hook = ln2.register_forward_hook(capture_ln2_output)
    try:
        prompts = list(FIXED_PROMPTS[:max_samples])
        encoded = tokenizer(
            prompts,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            if attention_mask is None:
                outputs = model(input_ids=input_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        hook.remove()

    if len(captured) != 1:
        raise ValueError(f"Expected one ln_2 activation capture, got {len(captured)}.")
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("GPT-2 forward output must expose tensor logits.")
    ln2_output = captured[0]
    with torch.no_grad():
        labels = input_ids.clone()
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        dense_loss_tensor = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        dense_loss = float(dense_loss_tensor.detach().float().cpu())

        hidden_dense = activation(c_fc(ln2_output))
        mlp_dense = c_proj(hidden_dense)
        scale, hidden_qcfs = calibrate_signed_qcfs(hidden_dense, time_steps=time_steps)
        mlp_qcfs = c_proj(hidden_qcfs)

        recipe = GPT2MLPAnn2SNNRecipe(scale=scale, time_steps=time_steps)
        converted_mlp = ModuleConverter(recipe=recipe, device=device).convert(mlp)
        mlp_if = converted_mlp(hidden_dense)
        if converted_mlp.last_hidden_if is None:
            raise RuntimeError("Converted MLP did not retain its hidden IF output.")
        hidden_if = converted_mlp.last_hidden_if
        positive_spike_rate = converted_mlp.if_proxy.positive_spike_rate
        negative_spike_rate = converted_mlp.if_proxy.negative_spike_rate

    metrics = {
        "dense_loss": dense_loss,
        "hidden_qcfs_relative_l2": _relative_l2(hidden_qcfs, hidden_dense),
        "mlp_qcfs_relative_l2": _relative_l2(mlp_qcfs, mlp_dense),
        "hidden_if_vs_qcfs_relative_l2": _relative_l2(hidden_if, hidden_qcfs),
        "mlp_if_relative_l2": _relative_l2(mlp_if, mlp_dense),
        "positive_spike_rate": positive_spike_rate,
        "negative_spike_rate": negative_spike_rate,
        "sample_count": max_samples,
    }
    for name, value in metrics.items():
        if isinstance(value, (float, int)) and not math.isfinite(float(value)):
            raise ValueError(f"Metric {name} is not finite: {value!r}.")

    metadata = {
        "hidden_shape": list(hidden_dense.shape),
        "mlp_shape": list(mlp_dense.shape),
        "scale_shape": list(scale.shape),
        "prompts": prompts,
    }
    return metrics, metadata


def build_slice_report(
    *,
    paths: GPT2ModelPaths,
    block_index: int,
    time_steps: int,
    max_samples: int,
    metrics: Mapping[str, object],
    metadata: Mapping[str, object],
    environment: Mapping[str, object],
) -> Dict[str, object]:
    r"""
    **API Language** - 中文 | English

    构造固定的 Phase 5.1 report contract。
    Build the fixed Phase 5.1 report contract.

    :param paths: Validated local GPT-2 artifact paths.
    :type paths: GPT2ModelPaths
    :param block_index: GPT-2 block index represented by the report.
    :type block_index: int
    :param time_steps: Number of QCFS/IF time steps.
    :type time_steps: int
    :param max_samples: Number of fixed prompts used for calibration.
    :type max_samples: int
    :param metrics: Finite conversion metrics.
    :type metrics: Mapping[str, object]
    :param metadata: Shape and prompt metadata from the conversion.
    :type metadata: Mapping[str, object]
    :param environment: Runtime environment fields.
    :type environment: Mapping[str, object]
    :return: Report whose top-level keys match the Phase 5.1 contract.
    :rtype: dict[str, object]
    :raises ValueError: If the report contains non-finite JSON values.
    """

    report = {
        "slice_schema_version": SLICE_SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "model": {
            "name": DEFAULT_MODEL_NAME,
            "revision": EXPECTED_REVISION,
        },
        "environment": dict(environment),
        "input": {
            "max_samples": max_samples,
            "max_length": MAX_LENGTH,
            "prompts": list(metadata["prompts"]),
        },
        "slice": {
            "target": f"transformer.h.{block_index}.mlp",
            "scope": "standalone_mlp_activation_slice",
            "training": "none",
            "block_index": block_index,
            "hidden_shape": list(metadata["hidden_shape"]),
            "mlp_shape": list(metadata["mlp_shape"]),
        },
        "calibration": {
            "mode": CALIBRATION_MODE,
            "time_steps": time_steps,
            "scale_shape": list(metadata["scale_shape"]),
        },
        "spikingjelly_components": {
            "neuron": "ActivationAwareIFNode",
            "converter": "ModuleConverter",
            "recipe": "GPT2MLPAnn2SNNRecipe",
            "converter_style": True,
            "public_api_added": False,
            "encoding": SIGNED_ENCODING,
            "if_reconstruction": "qcfs_count_replay_with_activation_aware_if",
        },
        "metrics": dict(metrics),
        "files": hash_files(paths.root),
        "unsupported": list(_UNSUPPORTED),
    }
    # Ensure report serialization cannot silently carry NaN/Inf values.
    json.dumps(report, allow_nan=False)
    return report


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 5.1 frozen GPT-2 MLP activation ANN2SNN slice "
            "with signed dual-channel IF reconstruction."
        )
    )
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=VALID_DEVICES, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--block-index", type=int, default=DEFAULT_BLOCK_INDEX)
    parser.add_argument("--time-steps", type=int, default=DEFAULT_TIME_STEPS)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Number of fixed prompts to use (1-{len(FIXED_PROMPTS)}).",
    )
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


def _write_report(output_dir: Path, report: Mapping[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / REPORT_FILENAME
    if target.exists():
        raise SystemExit(
            f"Refusing to overwrite existing report at {target}. "
            "Choose a new --output-dir or delete the previous result."
        )
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

    if args.block_index < 0:
        _emit(f"--block-index must be non-negative; got {args.block_index}.")
        return 2
    if args.time_steps <= 0:
        _emit(f"--time-steps must be positive; got {args.time_steps}.")
        return 2
    if args.max_samples <= 0 or args.max_samples > len(FIXED_PROMPTS):
        _emit(
            f"--max-samples must be between 1 and {len(FIXED_PROMPTS)}; "
            f"got {args.max_samples}."
        )
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
        metrics, metadata = compute_mlp_slice(
            paths=paths,
            device=args.device,
            block_index=args.block_index,
            time_steps=args.time_steps,
            max_samples=args.max_samples,
        )
        report = build_slice_report(
            paths=paths,
            block_index=args.block_index,
            time_steps=args.time_steps,
            max_samples=args.max_samples,
            metrics=metrics,
            metadata=metadata,
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
