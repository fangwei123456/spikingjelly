from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch

from spikingjelly.activation_based.distributed import llm


def _prompts(
    *,
    count: int,
    input_length: int,
    shared_prefix_length: int,
    vocab_size: int,
    seed: int,
) -> list[list[int]]:
    if not 0 <= shared_prefix_length <= input_length:
        raise ValueError("shared_prefix_length must lie in [0, input_length].")
    generator = torch.Generator().manual_seed(seed)
    prefix = torch.randint(
        0, vocab_size, (shared_prefix_length,), generator=generator
    ).tolist()
    prompts = []
    for index in range(count):
        length = max(
            shared_prefix_length,
            input_length - index % max(1, input_length // 8),
        )
        private = torch.randint(
            0,
            vocab_size,
            (length - shared_prefix_length,),
            generator=generator,
        ).tolist()
        prompts.append(prefix + private)
    return prompts


def _load_prompts(path: Path, count: int, input_length: int) -> list[list[int]]:
    import numpy as np

    prompts = np.load(path, mmap_mode="r")
    if prompts.ndim != 2 or prompts.shape[0] < count or prompts.shape[1] < input_length:
        raise ValueError(
            f"prompts_npy shape must be at least ({count}, {input_length}), "
            f"got {prompts.shape}."
        )
    if not np.issubdtype(prompts.dtype, np.integer):
        raise ValueError("prompts_npy must contain integer token IDs.")
    return prompts[:count, :input_length].astype("int64").tolist()


async def _run_requests(engine, prompts, output_length):
    async def generate(input_ids):
        started = time.perf_counter()
        first_token = None
        output_ids = []
        stream = await engine.async_generate(
            input_ids=input_ids,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": output_length,
                "ignore_eos": True,
            },
            stream=True,
        )
        async for chunk in stream:
            if first_token is None and chunk.get("output_ids"):
                first_token = time.perf_counter()
            chunk_ids = chunk.get("output_ids", [])
            completion_tokens = int(
                chunk.get("meta_info", {}).get("completion_tokens", len(chunk_ids))
            )
            if len(chunk_ids) == completion_tokens:
                output_ids = chunk_ids
            elif len(output_ids) + len(chunk_ids) == completion_tokens:
                output_ids.extend(chunk_ids)
            else:
                raise RuntimeError("SGLang stream returned inconsistent token counts.")
        finished = time.perf_counter()
        if not output_ids or first_token is None:
            raise RuntimeError("SGLang request returned no generated token.")
        tokens = len(output_ids)
        return {
            "tokens": tokens,
            "output_ids": output_ids,
            "ttft_seconds": first_token - started,
            "e2e_seconds": finished - started,
            "tpot_seconds": (
                (finished - first_token) / (tokens - 1) if tokens > 1 else 0.0
            ),
        }

    started = time.perf_counter()
    requests = await asyncio.gather(*(generate(prompt) for prompt in prompts))
    return requests, time.perf_counter() - started


def _gpu_memory(stop: threading.Event, peaks: list[int], errors: list[str]) -> None:
    while True:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            errors.append(str(error))
            return
        values = [
            int(value.strip())
            for value in completed.stdout.splitlines()
            if value.strip().isdigit()
        ]
        if completed.returncode or not values:
            errors.append(completed.stderr.strip() or "nvidia-smi returned no values")
            return
        peaks[0] = max(peaks[0], max(values))
        if stop.wait(0.05):
            return


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(len(values) - 1, int(len(values) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description="SGLang offline Engine benchmark")
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--external-model-package",
        default="benchmark.snn_llm.sglang_models",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--shared-prefix-length", type=int, default=0)
    parser.add_argument("--prompts-npy", type=Path)
    parser.add_argument("--include-outputs", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.requests, args.input_length, args.output_length, args.repeats) <= 0:
        parser.error("requests, lengths, and repeats must be positive")
    if not 0 <= args.shared_prefix_length <= args.input_length:
        parser.error("shared_prefix_length must lie in [0, input_length]")
    model_config = json.loads(
        (args.artifact / "config.json").read_text(encoding="utf-8")
    )
    if args.prompts_npy is None:
        prompts = _prompts(
            count=args.requests,
            input_length=args.input_length,
            shared_prefix_length=args.shared_prefix_length,
            vocab_size=int(model_config["vocab_size"]),
            seed=args.seed,
        )
    else:
        prompts = _load_prompts(args.prompts_npy, args.requests, args.input_length)
    if args.shared_prefix_length and any(
        prompt[: args.shared_prefix_length] != prompts[0][: args.shared_prefix_length]
        for prompt in prompts[1:]
    ):
        raise ValueError("Prompts do not share the declared prefix.")
    config = llm.SGLangEngineConfig(
        artifact=args.artifact,
        external_model_package=args.external_model_package,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        memory_fraction=args.memory_fraction,
        seed=args.seed,
        tokenizer=args.tokenizer,
    )
    runs = []
    with llm.open_sglang_engine(config) as engine:
        engine.loop.run_until_complete(
            _run_requests(engine, prompts[:1], min(8, args.output_length))
        )
        for _ in range(args.repeats):
            engine.flush_cache()
            stop = threading.Event()
            peak_memory_mib = [0]
            memory_errors: list[str] = []
            monitor = threading.Thread(
                target=_gpu_memory,
                args=(stop, peak_memory_mib, memory_errors),
                daemon=True,
            )
            monitor.start()
            try:
                requests, elapsed = engine.loop.run_until_complete(
                    _run_requests(engine, prompts, args.output_length)
                )
            finally:
                stop.set()
                monitor.join()
            if memory_errors:
                raise RuntimeError(f"GPU memory polling failed: {memory_errors[0]}")
            output_tokens = sum(request["tokens"] for request in requests)
            run = {
                "elapsed_seconds": elapsed,
                "request_throughput": len(requests) / elapsed,
                "input_tokens_per_second": sum(map(len, prompts)) / elapsed,
                "output_tokens_per_second": output_tokens / elapsed,
                "ttft_median_ms": statistics.median(
                    request["ttft_seconds"] * 1000 for request in requests
                ),
                "ttft_p99_ms": _percentile(
                    [request["ttft_seconds"] * 1000 for request in requests],
                    0.99,
                ),
                "tpot_median_ms": statistics.median(
                    request["tpot_seconds"] * 1000 for request in requests
                ),
                "tpot_p99_ms": _percentile(
                    [request["tpot_seconds"] * 1000 for request in requests],
                    0.99,
                ),
                "e2e_p99_ms": _percentile(
                    [request["e2e_seconds"] * 1000 for request in requests],
                    0.99,
                ),
                "peak_gpu_memory_mib": peak_memory_mib[0],
            }
            if args.include_outputs:
                run["outputs"] = [request["output_ids"] for request in requests]
            runs.append(run)
    report = {
        "artifact": str(args.artifact),
        "topology": {
            "tp": args.tensor_parallel_size,
            "pp": args.pipeline_parallel_size,
            "dp": args.data_parallel_size,
        },
        "workload": {
            "requests": args.requests,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "shared_prefix_length": args.shared_prefix_length,
            "seed": args.seed,
        },
        "cuda_graph": False,
        "runs": runs,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
