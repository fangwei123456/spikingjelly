"""Benchmark public sparse Linear strategies with correctness gates.

The sparse timing includes row-index construction and weight transposition.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cupy
import torch

from spikingjelly.activation_based.cuda_kernel import sparse_linear


def _sync():
    torch.cuda.synchronize()


def time_fn(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sparse_kernel_only.json")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")

    device = torch.device("cuda")
    torch.manual_seed(0)
    results = []
    cases = [
        (M, K, N, density)
        for density in [0.02, 0.05, 0.10, 0.20]
        for M, K, N in [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ]
    ]

    for M, K, N, density in cases:
        s = (torch.rand(M, K, device=device) < density).float()
        W = torch.randn(N, K, device=device)

        def dense():
            return torch.nn.functional.linear(s, W)

        def torch_linear():
            return sparse_linear(s, W, strategy="torch")

        def sparse():
            return sparse_linear(s, W, strategy="sparse")

        reference = dense()
        correctness_atol = 1e-5 if density <= 0.05 else 5e-5
        torch.testing.assert_close(torch_linear(), reference, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            sparse(), reference, rtol=1e-4, atol=correctness_atol
        )

        times = {
            "dense_ms": time_fn(dense, args.warmup, args.iters),
            "torch_ms": time_fn(torch_linear, args.warmup, args.iters),
            "sparse_ms": time_fn(sparse, args.warmup, args.iters),
        }
        nnz = int(s.to(torch.bool).sum().item())
        speedups = {
            "torch": times["dense_ms"] / times["torch_ms"],
            "sparse": times["dense_ms"] / times["sparse_ms"],
        }
        case = {
            "M": M,
            "K": K,
            "N": N,
            "density": density,
            "correctness_atol": correctness_atol,
            "nnz": nnz,
            "times_ms": times,
            "speedup_vs_dense": speedups,
        }
        print(
            f"M={M:5d} K={K:5d} N={N:5d} d={density:.2f} nnz={nnz:7d}  "
            f"dense={times['dense_ms']:6.3f}ms "
            f"torch={times['torch_ms']:6.3f}ms ({speedups['torch']:.2f}x) "
            f"sparse={times['sparse_ms']:6.3f}ms ({speedups['sparse']:.2f}x)"
        )
        results.append(case)

    properties = torch.cuda.get_device_properties(device)
    source_path = Path(sparse_linear.__code__.co_filename)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    output = {
        "metadata": {
            "source_revision": os.environ.get("SPIKINGJELLY_COMMIT"),
            "source_file_sha256": source_sha256,
            "evidence_kind": "single_process_exploratory",
            "process_runs": 1,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cupy_version": cupy.__version__,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "warmup": args.warmup,
            "iterations": args.iters,
            "sparse_includes_row_index_build": True,
            "correctness_rtol": 1e-4,
            "strict_low_density_atol": 1e-5,
            "exploratory_high_density_atol": 5e-5,
        },
        "cases": results,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
