from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from spikingjelly.activation_based.distributed import llm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SGLang offline SNN generation")
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--prefill-context-parallel-size", type=int, default=1)
    parser.add_argument("--decode-context-parallel-size", type=int, default=1)
    parser.add_argument("--prompt-count", type=int, default=4)
    parser.add_argument("--prompt-counts", type=int, nargs="+")
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--memory-fraction", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stream-results", action="store_true")
    args = parser.parse_args()
    if args.warmups <= 0 or args.repeats <= 0:
        parser.error("warmups and repeats must be positive")
    if (
        min(
            args.tensor_parallel_size,
            args.pipeline_parallel_size,
            args.data_parallel_size,
        )
        <= 0
    ):
        parser.error("parallel sizes must be positive")
    prompt_counts = args.prompt_counts or [args.prompt_count]
    if any(count <= 0 for count in prompt_counts):
        parser.error("prompt counts must be positive")
    if any(count % args.data_parallel_size for count in prompt_counts):
        parser.error("prompt counts must be divisible by data-parallel-size")
    return args


def _device_memory_used_bytes(gpu_count: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_ids = (
        visible.split(",")[:gpu_count]
        if visible
        else [str(index) for index in range(gpu_count)]
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={','.join(device_ids)}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(int(value) for value in completed.stdout.split()) * 1024**2


def main() -> None:
    args = _parse_args()
    tokens = np.load(args.data / "valid.npy", mmap_mode="r")
    prompt_counts = args.prompt_counts or [args.prompt_count]
    if len(tokens) == 0:
        raise ValueError("dataset must be non-empty.")
    config = llm.SGLangGenerationConfig(
        artifact=args.artifact,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        prefill_context_parallel_size=args.prefill_context_parallel_size,
        decode_context_parallel_size=args.decode_context_parallel_size,
        memory_fraction=args.memory_fraction,
        external_model_package="benchmark.snn_llm.sglang_models",
        disable_radix_cache=True,
    )
    engine = llm.create_sglang_engine(config)
    sampling = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    }
    try:
        measurements = []
        gpu_count = (
            args.tensor_parallel_size
            * args.pipeline_parallel_size
            * args.data_parallel_size
        )

        def emit() -> None:
            print(
                json.dumps(
                    {
                        "measurements": measurements,
                        "tensor_parallel_size": args.tensor_parallel_size,
                        "pipeline_parallel_size": args.pipeline_parallel_size,
                        "data_parallel_size": args.data_parallel_size,
                        "prefill_context_parallel_size": (
                            args.prefill_context_parallel_size
                        ),
                        "decode_context_parallel_size": (
                            args.decode_context_parallel_size
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        for prompt_count in prompt_counts:
            prompts = torch.from_numpy(
                np.array(
                    tokens[np.arange(prompt_count) % len(tokens), : args.prompt_length],
                    copy=True,
                )
            ).long()
            prompt_list = prompts.tolist()
            for _ in range(args.warmups):
                engine.generate(input_ids=prompt_list, sampling_params=sampling)
            elapsed_samples = []
            peak_memory_bytes = 0
            results = None
            for _ in range(args.repeats):
                started = time.perf_counter()
                results = engine.generate(
                    input_ids=prompt_list, sampling_params=sampling
                )
                elapsed_samples.append(time.perf_counter() - started)
                peak_memory_bytes = max(
                    peak_memory_bytes, _device_memory_used_bytes(gpu_count)
                )
            generated_tokens = sum(len(result["output_ids"]) for result in results)
            elapsed = statistics.median(elapsed_samples)
            measurements.append(
                {
                    "prompt_count": prompt_count,
                    "outputs": [result["output_ids"] for result in results],
                    "inference_seconds": elapsed,
                    "inference_seconds_samples": elapsed_samples,
                    "generated_tokens_per_second": generated_tokens / elapsed,
                    "peak_device_memory_bytes": peak_memory_bytes,
                }
            )
            if args.stream_results:
                emit()
    finally:
        engine.shutdown()
    if not args.stream_results:
        emit()


if __name__ == "__main__":
    main()
