import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based.ann2snn import Converter, SpikeZIPTFQANNRecipe
from spikingjelly.activation_based.ann2snn.recipes.spikezip_qann import STBIFNeuron


class SpikeZIPQuantizer(nn.Module):
    def __init__(self, level: int = 8, sym: bool = True, scale: float = 0.25) -> None:
        super().__init__()
        self.level = int(level)
        self.sym = bool(sym)
        self.s = nn.Parameter(torch.tensor(float(scale)))
        if self.sym:
            self.pos_max = torch.tensor(self.level // 2 - 1)
            self.neg_min = torch.tensor(-self.level // 2)
        else:
            self.pos_max = torch.tensor(self.level - 1)
            self.neg_min = torch.tensor(0)

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
        scores = scores / (self.attention_head_size ** 0.5)
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
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--level", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
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
            current_min if min_accumulated is None else min(min_accumulated, current_min)
        )
        max_accumulated = (
            current_max if max_accumulated is None else max(max_accumulated, current_max)
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
    model = TinyQRobertaClassifier(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        level=args.level,
    ).to(device).eval()
    if args.qann_checkpoint is not None:
        state = torch.load(args.qann_checkpoint, map_location=device)
        model.load_state_dict(state)

    tokens = torch.randint(0, 32, (args.batch_size, args.seq_len), device=device)
    attention_mask = torch.zeros(args.batch_size, 1, 1, args.seq_len, device=device)
    if args.seq_len > 2:
        attention_mask[:, :, :, -1] = -10000.0

    start = time.time()
    with torch.no_grad():
        qann_logits = model(tokens, attention_mask=attention_mask)
        converted = Converter(
            recipe=SpikeZIPTFQANNRecipe(
                time_steps=args.time_steps,
                model_family="roberta",
            ),
            device=device,
        ).convert(model).eval()
        snn_logits, sequence = converted(
            tokens,
            attention_mask=attention_mask,
            return_sequences=True,
        )
    max_abs_diff = (qann_logits - snn_logits).abs().max().item()
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
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "level": args.level,
            "qann_checkpoint": args.qann_checkpoint,
        },
        "recipe": "SpikeZIPTFQANNRecipe",
        "max_abs_diff": max_abs_diff,
        "sequence_shape": list(sequence.shape),
        "stbif_state": stbif_state,
        "seconds": time.time() - start,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if max_abs_diff > 1e-5:
        raise RuntimeError(f"SpikeZIP QANN parity failed: max_abs_diff={max_abs_diff}.")
    if not set(stbif_state["last_step_spike_values"]).issubset({-1.0, 0.0, 1.0}):
        raise RuntimeError(
            "Unexpected ST-BIF spike values: "
            f"{stbif_state['last_step_spike_values']}."
        )
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
