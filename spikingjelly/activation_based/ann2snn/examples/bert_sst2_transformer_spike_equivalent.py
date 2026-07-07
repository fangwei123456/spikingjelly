import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import (
    Converter,
    TransformerSpikeEquivalentRecipe,
)


def import_huggingface():
    try:
        from datasets import load_dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "This example requires optional Hugging Face dependencies. Install "
            "them with: uv pip install transformers datasets"
        ) from exc
    return load_dataset, AutoModelForSequenceClassification, AutoTokenizer


class BertSST2FromEmbeddings(nn.Module):
    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.encoder = FXFriendlyBertEncoder(hf_model.bert.encoder, hf_model.config)
        self.pooler = hf_model.bert.pooler
        self.dropout = hf_model.dropout
        self.classifier = hf_model.classifier

    def forward(
        self,
        embedding_output: torch.Tensor,
        extended_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence_output = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


class FXFriendlyBertSelfAttention(nn.Module):
    def __init__(self, source: nn.Module, config) -> None:
        super().__init__()
        self.query = source.query
        self.key = source.key
        self.value = source.value
        self.dropout = source.dropout
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(
            x.shape[0],
            x.shape[1],
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        extended_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size**0.5)
        attention_scores = attention_scores + extended_attention_mask
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.all_head_size,
        )
        return context_layer


class FXFriendlyBertAttention(nn.Module):
    def __init__(self, source: nn.Module, config) -> None:
        super().__init__()
        self.self = FXFriendlyBertSelfAttention(source.self, config)
        self.output = source.output

    def forward(
        self,
        hidden_states: torch.Tensor,
        extended_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_output = self.self(hidden_states, extended_attention_mask)
        return self.output(self_output, hidden_states)


class FXFriendlyBertLayer(nn.Module):
    def __init__(self, source: nn.Module, config) -> None:
        super().__init__()
        self.attention = FXFriendlyBertAttention(source.attention, config)
        self.intermediate = source.intermediate
        self.output = source.output

    def forward(
        self,
        hidden_states: torch.Tensor,
        extended_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.attention(hidden_states, extended_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class FXFriendlyBertEncoder(nn.Module):
    def __init__(self, source: nn.Module, config) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            FXFriendlyBertLayer(layer, config) for layer in source.layer
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        extended_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TransformerSpikeEquivalentRecipe on a BERT SST-2 classifier "
            "classification through an embedding-output conversion boundary."
        )
    )
    parser.add_argument(
        "--model-name-or-path",
        default="textattack/bert-base-uncased-SST-2",
    )
    parser.add_argument("--dataset-name", default="nyu-mll/glue")
    parser.add_argument("--dataset-config", default="sst2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--parity-batches", type=int, default=2)
    parser.add_argument("--parity-atol", type=float, default=1e-5)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def collate_tokenized(batch, tokenizer, max_length):
    texts = [item["sentence"] for item in batch]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    encoded["labels"] = labels
    return encoded


def build_loader(args, tokenizer):
    load_dataset, _, _ = import_huggingface()
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    if args.eval_samples is not None:
        if args.eval_samples <= 0:
            raise ValueError("--eval-samples must be positive when set.")
        dataset = dataset.select(range(min(args.eval_samples, len(dataset))))
    return (
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_tokenized(
                batch,
                tokenizer,
                args.max_length,
            ),
        ),
        len(dataset),
    )


def make_embedding_batch(hf_model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    labels = batch["labels"].to(device)
    embedding_output = hf_model.bert.embeddings(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
    )
    extended_attention_mask = hf_model.get_extended_attention_mask(
        attention_mask,
        input_ids.shape,
    )
    return embedding_output, extended_attention_mask, labels


def make_first_real_then_zero_sequence(x, time_steps):
    if time_steps <= 0:
        raise ValueError(f"time_steps must be positive, got {time_steps}.")
    x_seq = torch.zeros((time_steps, *x.shape), dtype=x.dtype, device=x.device)
    x_seq[0] = x
    return x_seq


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).sum().item()


def evaluate_ann(wrapper, hf_model, loader, device, name):
    wrapper.eval().to(device)
    hf_model.eval().to(device)
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            embedding_output, extended_attention_mask, labels = make_embedding_batch(
                hf_model,
                batch,
                device,
            )
            logits = wrapper(embedding_output, extended_attention_mask)
            correct += accuracy(logits, labels)
            total += labels.numel()
            if (index + 1) % 50 == 0:
                print(name, index + 1, total, correct / total, flush=True)
    return {"accuracy": correct / total, "total": total, "seconds": time.time() - start}


def check_hf_wrapper_parity(wrapper, hf_model, loader, device, max_batches, atol):
    if max_batches <= 0:
        return {"checked_batches": 0, "max_abs_diff": None}
    wrapper.eval().to(device)
    hf_model.eval().to(device)
    max_abs_diff = 0.0
    checked_batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            embedding_output, extended_attention_mask, _ = make_embedding_batch(
                hf_model,
                batch,
                device,
            )
            hf_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                hf_kwargs["token_type_ids"] = token_type_ids
            hf_logits = hf_model(**hf_kwargs).logits
            wrapper_logits = wrapper(embedding_output, extended_attention_mask)
            diff = (hf_logits - wrapper_logits).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            checked_batches += 1
            if checked_batches >= max_batches:
                break
    if max_abs_diff > atol:
        raise RuntimeError(
            "BERT wrapper parity check failed: "
            f"max_abs_diff={max_abs_diff} > atol={atol}."
        )
    return {
        "checked_batches": checked_batches,
        "max_abs_diff": max_abs_diff,
        "atol": atol,
    }


def evaluate_transformer_spike_equivalent(
    converted,
    hf_model,
    loader,
    device,
    time_steps,
    name,
):
    converted.eval().to(device)
    hf_model.eval().to(device)
    functional.set_step_mode(converted, "m")
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            embedding_output, extended_attention_mask, labels = make_embedding_batch(
                hf_model,
                batch,
                device,
            )
            functional.reset_net(converted)
            embedding_seq = make_first_real_then_zero_sequence(
                embedding_output,
                time_steps,
            )
            logits = converted(embedding_seq, extended_attention_mask).sum(dim=0)
            correct += accuracy(logits, labels)
            total += labels.numel()
            if (index + 1) % 50 == 0:
                print(name, index + 1, total, correct / total, flush=True)
    return {"accuracy": correct / total, "total": total, "seconds": time.time() - start}


def write_output(path, payload):
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main():
    args = parse_args()
    device = torch.device(args.device)
    load_dataset, model_cls, tokenizer_cls = import_huggingface()
    _ = load_dataset
    tokenizer = tokenizer_cls.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    hf_model = (
        model_cls.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
        .to(device)
        .eval()
    )
    loader, dataset_size = build_loader(args, tokenizer)
    wrapper = BertSST2FromEmbeddings(hf_model).to(device).eval()

    env = {
        "model_name_or_path": args.model_name_or_path,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "dataset_size": dataset_size,
        "eval_samples": args.eval_samples,
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "time_steps": args.time_steps,
        "recipe": "TransformerSpikeEquivalentRecipe",
        "conversion_boundary": "bert embeddings output",
        "parity_batches": args.parity_batches,
        "parity_atol": args.parity_atol,
    }
    print(json.dumps(env), flush=True)

    parity = check_hf_wrapper_parity(
        wrapper,
        hf_model,
        loader,
        device,
        args.parity_batches,
        args.parity_atol,
    )
    print("HF_WRAPPER_PARITY", json.dumps(parity), flush=True)

    baseline = evaluate_ann(wrapper, hf_model, loader, device, "baseline")
    print("BASELINE", json.dumps(baseline), flush=True)

    recipe = TransformerSpikeEquivalentRecipe(time_steps=args.time_steps)
    converted = (
        Converter(recipe=recipe, device=device).convert(wrapper).to(device).eval()
    )
    converted_result = evaluate_transformer_spike_equivalent(
        converted,
        hf_model,
        loader,
        device,
        args.time_steps,
        f"transformer_spike_equivalent_t{args.time_steps}",
    )
    drop = baseline["accuracy"] - converted_result["accuracy"]
    print("TRANSFORMER_SPIKE_EQUIVALENT", json.dumps(converted_result), flush=True)
    print("DROP", drop, flush=True)

    payload = {
        "env": env,
        "hf_wrapper_parity": parity,
        "baseline": baseline,
        "transformer_spike_equivalent": converted_result,
        "drop": drop,
    }
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
