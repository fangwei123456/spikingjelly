from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import ModuleConverter, SpikeZIPTFQANNRecipe
from spikingjelly.activation_based.neuron import STBIFNeuron


class SpikeZIPQuantizer(nn.Module):
    def __init__(self, level: int = 8, sym: bool = True, scale: float = 0.25) -> None:
        super().__init__()
        self.level = int(level)
        self.sym = bool(sym)
        self.s = nn.Parameter(torch.tensor(float(scale)))
        if self.sym:
            pos_max = self.level // 2 - 1
            neg_min = -self.level // 2
        else:
            pos_max = self.level - 1
            neg_min = 0
        self.register_buffer("pos_max", torch.tensor(float(pos_max)))
        self.register_buffer("neg_min", torch.tensor(float(neg_min)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.floor(x / self.s + 0.5)
        q = torch.clamp(q, min=float(self.neg_min), max=float(self.pos_max))
        return q * self.s


class TinyQRobertaSelfAttention(nn.Module):
    def __init__(self, hidden_size: int = 16, num_heads: int = 4, level: int = 8):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.query_quan = SpikeZIPQuantizer(level=level, sym=True)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.key_quan = SpikeZIPQuantizer(level=level, sym=True)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.value_quan = SpikeZIPQuantizer(level=level, sym=True)
        self.attn_quan = SpikeZIPQuantizer(level=level, sym=False, scale=0.125)
        self.after_attn_quan = SpikeZIPQuantizer(level=level, sym=True)
        self.dropout = nn.Dropout(0.0)
        self.position_embedding_type = "absolute"
        self.is_decoder = False

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions: bool = False,
    ):
        del encoder_hidden_states, encoder_attention_mask, past_key_value
        query_layer = self.transpose_for_scores(
            self.query_quan(self.query(hidden_states))
        )
        key_layer = self.transpose_for_scores(self.key_quan(self.key(hidden_states)))
        value_layer = self.transpose_for_scores(
            self.value_quan(self.value(hidden_states))
        )
        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / (self.attention_head_size**0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        attention_probs = F.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_probs = self.attn_quan(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context = torch.matmul(attention_probs, value_layer)
        context = self.after_attn_quan(context)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        return (context, attention_probs) if output_attentions else (context,)


class TinyQRobertaClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 16,
        num_heads: int = 4,
        level: int = 8,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = TinyQRobertaSelfAttention(hidden_size, num_heads, level)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.embedding(tokens)
        hidden = self.attention(hidden, attention_mask=attention_mask)[0]
        hidden = self.norm(hidden)
        return self.classifier(hidden[:, 0])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a synthetic SpikeZIP-compatible RoBERTa QANN to SNN conversion "
            "parity check."
        )
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--time-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--level", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parity-atol", type=float, default=1e-3)
    parser.add_argument("--qann-checkpoint", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def write_output(path, payload):
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def collect_stbif_state(model: nn.Module):
    values = set()
    min_accumulated = None
    max_accumulated = None
    for module in model.modules():
        if not isinstance(module, STBIFNeuron) or module.cur_output is None:
            continue
        values.update(float(x) for x in torch.unique(module.cur_output).cpu())
        accumulated = module.accumulated.detach()
        current_min = float(accumulated.min().cpu())
        current_max = float(accumulated.max().cpu())
        min_accumulated = (
            current_min
            if min_accumulated is None
            else min(min_accumulated, current_min)
        )
        max_accumulated = (
            current_max
            if max_accumulated is None
            else max(max_accumulated, current_max)
        )
    if not values and min_accumulated is None and max_accumulated is None:
        raise RuntimeError(
            "collect_stbif_state found no STBIFNeuron with cur_output set; "
            "the SpikeZIPTFQANNRecipe conversion likely did not run."
        )
    return {
        "last_step_spike_values": sorted(values),
        "min_accumulated": min_accumulated,
        "max_accumulated": max_accumulated,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = (
        TinyQRobertaClassifier(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            level=args.level,
        )
        .to(device)
        .eval()
    )
    if args.qann_checkpoint is not None:
        state = torch.load(
            args.qann_checkpoint,
            map_location=device,
            weights_only=True,
        )
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "QANN checkpoint does not match the configured "
                "TinyQRobertaClassifier. Re-run with the same "
                "--vocab-size/--hidden-size/--num-heads/--level as the "
                f"checkpoint. Missing: {result.missing_keys}; "
                f"unexpected: {result.unexpected_keys}."
            )

    tokens = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
    )
    attention_mask = torch.zeros(args.batch_size, 1, 1, args.seq_len, device=device)
    if args.seq_len > 2:
        attention_mask[:, :, :, -1] = -10000.0

    start = time.time()
    with torch.no_grad():
        qann_logits = model(tokens, attention_mask=attention_mask)
        converted = (
            ModuleConverter(
                recipe=SpikeZIPTFQANNRecipe(
                    time_steps=args.time_steps,
                    model_family="roberta",
                ),
                device=device,
            )
            .convert(model)
            .eval()
        )
        functional.set_step_mode(converted, "s")
        functional.reset_net(converted)
        accumulated = None
        for _ in range(args.time_steps):
            step_logits = converted(tokens, attention_mask=attention_mask)
            accumulated = (
                step_logits if accumulated is None else accumulated + step_logits
            )
        snn_logits = accumulated
        accumulated_sequence_shape = [args.time_steps, *snn_logits.shape]
    diff = (qann_logits - snn_logits).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    stbif_state = collect_stbif_state(converted)
    payload = {
        "env": {
            "device": str(device),
            "cuda_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "seed": args.seed,
            "time_steps": args.time_steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "level": args.level,
            "parity_atol": args.parity_atol,
            "qann_checkpoint": args.qann_checkpoint,
        },
        "recipe": "SpikeZIPTFQANNRecipe",
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "accumulated_sequence_shape": accumulated_sequence_shape,
        "stbif_state": stbif_state,
        "seconds": time.time() - start,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if max_abs_diff > args.parity_atol:
        raise RuntimeError(
            "SpikeZIP QANN parity failed: "
            f"max_abs_diff={max_abs_diff} > parity_atol={args.parity_atol}."
        )
    if not set(stbif_state["last_step_spike_values"]).issubset({-1.0, 0.0, 1.0}):
        raise RuntimeError(
            f"Unexpected ST-BIF spike values: {stbif_state['last_step_spike_values']}."
        )
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
