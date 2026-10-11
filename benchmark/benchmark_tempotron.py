"""Run with python -m benchmark.benchmark_tempotron, in three fresh processes."""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import statistics
import subprocess
import time
from pathlib import Path

import torch

from spikingjelly._ops import tempotron as tempotron_ops
from spikingjelly.timing_based import functional
from spikingjelly.timing_based.encoding import GaussianTuning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mnist-input", type=Path, required=True)
    parser.add_argument("--legacy-source", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--case-indices", type=int, nargs="+", choices=range(10))
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    legacy_type = None
    if args.legacy_source is not None:
        spec = importlib.util.spec_from_file_location(
            "legacy_tempotron_baseline", args.legacy_source
        )
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        legacy_type = legacy.Tempotron

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def check_gpu_owner():
        if args.gpu_uuid is None:
            return
        listing = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for line in listing.splitlines():
            uuid, pid = [part.strip() for part in line.split(",")]
            if uuid == args.gpu_uuid and int(pid) != os.getpid():
                raise RuntimeError(
                    "Another compute process is using the benchmark GPU."
                )

    cases = [
        (1, 500, 1, 1, 100.0),
        (8, 500, 10, 1, 100.0),
        (8, 500, 10, 3, 100.0),
        (8, 500, 10, 1, 1000.0),
        (64, 12544, 10, 1, 100.0),
        (1, 500, 1, 3, 100.0),
        (1, 500, 10, 1, 100.0),
        (1, 500, 10, 3, 100.0),
        (8, 500, 1, 1, 100.0),
        (8, 500, 1, 3, 100.0),
    ]
    if args.case_indices is not None:
        cases = [cases[index] for index in args.case_indices]
    records = []
    for batch, inputs, outputs, events_per_input, duration in cases:
        check_gpu_owner()
        torch.manual_seed(20261007)
        dense_times = torch.rand(batch, inputs, events_per_input, device=device) * 100
        if inputs == 12544:
            images = torch.load(args.mnist_input, weights_only=True)["images"].to(
                device
            )
            encoder = GaussianTuning(1, 16, torch.zeros(1), torch.ones(1)).to(device)
            times, indices, offsets = encoder(images, duration)
        else:
            times = dense_times.flatten()
            indices = (
                torch.arange(inputs, device=device)
                .repeat_interleave(events_per_input)
                .repeat(batch)
            )
            offsets = torch.arange(batch + 1, device=device) * inputs * events_per_input
        times = times.detach().requires_grad_()
        weight = (torch.randn(outputs, inputs, device=device) * 0.001).requires_grad_()
        saved = []

        def pack(tensor):
            saved.append(
                {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bytes": tensor.numel() * tensor.element_size(),
                }
            )
            return tensor

        def objective(result, mode):
            if mode == "v_max":
                return result[0].sum()
            if mode == "t_peak":
                return result[1].sum()
            spike = torch.where(torch.isfinite(result[2]), result[2], 0).sum()
            return (
                spike
                if mode == "t_spike"
                else result[0].sum() + result[1].sum() + spike
            )

        modes = ("v_max", "t_peak", "t_spike", "all")
        for _ in range(3):
            result = functional.tempotron(times, indices, offsets, weight, duration)
            for mode in modes:
                times.grad = weight.grad = None
                objective(result, mode).backward(retain_graph=True)
            del result
        times.grad = weight.grad = None
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            initial_memory = torch.cuda.memory_allocated(device)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            result = functional.tempotron(times, indices, offsets, weight, duration)
        objective(result, "all").backward()
        synchronize()
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device)
        del result
        forward_samples = []
        backward_samples = {mode: [] for mode in modes}
        for _ in range(5):
            times.grad = weight.grad = None
            synchronize()
            start = time.perf_counter()
            result = functional.tempotron(times, indices, offsets, weight, duration)
            synchronize()
            forward_samples.append((time.perf_counter() - start) * 1000)
            for mode in modes:
                times.grad = weight.grad = None
                synchronize()
                start = time.perf_counter()
                objective(result, mode).backward(retain_graph=True)
                synchronize()
                backward_samples[mode].append((time.perf_counter() - start) * 1000)
                assert (
                    torch.isfinite(times.grad).all()
                    and torch.isfinite(weight.grad).all()
                )
            del result
        forward_ms = statistics.median(forward_samples)
        backward = {
            mode: statistics.median(samples)
            for mode, samples in backward_samples.items()
        }
        record = {
            "B": batch,
            "I": inputs,
            "O": outputs,
            "events_per_input": events_per_input,
            "duration": duration,
            "event_count": times.numel(),
            "source": "MNIST test images 0:64"
            if inputs == 12544
            else "seeded synthetic events, fixed [0,100]",
            "forward_ms": forward_ms,
            "forward_samples_ms": forward_samples,
            "backward_ms": backward,
            "backward_samples_ms": backward_samples,
            "forward_backward_ms": {
                mode: statistics.median(
                    [a + b for a, b in zip(forward_samples, samples)]
                )
                for mode, samples in backward_samples.items()
            },
            "saved_tensors": saved,
            "saved_tensor_bytes": sum(x["bytes"] for x in saved),
        }
        if device.type == "cuda":
            record["peak_allocated_bytes"] = peak_memory
            record["peak_extra_allocated_bytes"] = (
                record["peak_allocated_bytes"] - initial_memory
            )
        record["process_peak_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
        del times
        weight.grad = None
        if legacy_type is not None:
            model = legacy_type(inputs * events_per_input, outputs, int(duration)).to(
                device
            )
            with torch.no_grad():
                model.model.summation_layer.weight.copy_(
                    weight.detach().repeat_interleave(events_per_input, 1)
                )
            old_times = dense_times.reshape(batch, -1)
            if inputs == 12544:
                # This is the frozen historical rounded-time encoder convention.
                old_times = torch.round(
                    encoder.mu.new_tensor(duration)
                    * (
                        1
                        - torch.exp(
                            -(
                                images.unsqueeze(-1) - encoder.mu[None, :, None, :]
                            ).square()
                            / (2 * encoder.sigma2[None, :, None, :])
                        )
                    )
                ).reshape(batch, -1)
                old_times[old_times >= duration] = -1
            old_times = old_times.detach().requires_grad_()
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                old_times.grad = None
                model(old_times, "v_max").sum().backward()
            model.zero_grad(set_to_none=True)
            old_times.grad = None
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                legacy_initial_memory = torch.cuda.memory_allocated(device)
            historical = model(old_times, "v_max")
            historical.sum().backward()
            synchronize()
            if device.type == "cuda":
                legacy_peak_memory = torch.cuda.max_memory_allocated(device)
            del historical
            legacy_forward, legacy_backward = [], []
            for _ in range(5):
                model.zero_grad(set_to_none=True)
                old_times.grad = None
                synchronize()
                start = time.perf_counter()
                historical = model(old_times, "v_max")
                synchronize()
                legacy_forward.append((time.perf_counter() - start) * 1000)
                synchronize()
                start = time.perf_counter()
                historical.sum().backward()
                synchronize()
                legacy_backward.append((time.perf_counter() - start) * 1000)
                del historical
            record["legacy_forward_ms"] = statistics.median(legacy_forward)
            record["legacy_backward_ms"] = statistics.median(legacy_backward)
            record["legacy_forward_samples_ms"] = legacy_forward
            record["legacy_backward_samples_ms"] = legacy_backward
            if device.type == "cuda":
                record["legacy_peak_allocated_bytes"] = legacy_peak_memory
                record["legacy_peak_extra_allocated_bytes"] = (
                    legacy_peak_memory - legacy_initial_memory
                )
        records.append(record)
        check_gpu_owner()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "device": str(device),
                    "gpu_uuid": args.gpu_uuid,
                    "device_name": torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else platform.machine(),
                    "torch": torch.__version__,
                    "platform": platform.platform(),
                    "threads": 1,
                    "dtype": "float32; new internal float64",
                    "seed": 20261007,
                    "warmup": 3,
                    "repeats": 5,
                    "source_sha256": {
                        "timing_based/functional.py": hashlib.sha256(
                            Path(functional.__file__).read_bytes()
                        ).hexdigest(),
                        **{
                            "ops/tempotron/" + path.name: hashlib.sha256(
                                path.read_bytes()
                            ).hexdigest()
                            for path in sorted(
                                Path(tempotron_ops.__file__).parent.glob("*.py")
                            )
                        },
                    },
                    "benchmark_source_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "encoder_source_sha256": hashlib.sha256(
                        Path(functional.__file__).with_name("encoding.py").read_bytes()
                    ).hexdigest(),
                    "mnist_input_sha256": hashlib.sha256(
                        args.mnist_input.read_bytes()
                    ).hexdigest(),
                    "legacy_backward_scope": "weights and input times",
                    "legacy_is_equivalent": False,
                    "cpu_rss_is_cumulative": True,
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps(
                {k: v for k, v in record.items() if k not in ("saved_tensors",)}
            ),
            flush=True,
        )
        del weight
        if legacy_type is not None:
            del model, old_times


if __name__ == "__main__":
    main()
