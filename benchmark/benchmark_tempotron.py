"""Run with python -m benchmark.benchmark_tempotron, in three fresh processes."""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import torch

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

    # Initialize the device/allocator without repeating the largest workload.
    functional.tempotron(
        torch.tensor([0.3], device=device),
        torch.tensor([0], device=device),
        torch.tensor([0, 1], device=device),
        torch.ones(1, 1, device=device),
        5.0,
    )
    synchronize()
    if legacy_type is not None:
        warm_model = legacy_type(1, 1, 5).to(device)
        warm_model(torch.tensor([[0.3]], device=device), "v_max").sum().backward()
        synchronize()
        del warm_model
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

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            initial_memory = torch.cuda.memory_allocated(device)
        synchronize()
        start = time.perf_counter()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            result = functional.tempotron(times, indices, offsets, weight, duration)
        synchronize()
        forward_ms = (time.perf_counter() - start) * 1000
        backward = {}
        for mode in ("v_max", "t_peak", "t_spike", "all"):
            times.grad = None
            weight.grad = None
            synchronize()
            start = time.perf_counter()
            terms = [
                result[0].sum(),
                result[1].sum(),
                torch.where(torch.isfinite(result[2]), result[2], 0).sum(),
            ]
            loss = (
                sum(terms)
                if mode == "all"
                else terms[("v_max", "t_peak", "t_spike").index(mode)]
            )
            loss.backward(retain_graph=True)
            synchronize()
            backward[mode] = (time.perf_counter() - start) * 1000
            assert (
                torch.isfinite(times.grad).all() and torch.isfinite(weight.grad).all()
            )
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
            "backward_ms": backward,
            "forward_backward_ms": {k: forward_ms + v for k, v in backward.items()},
            "saved_tensors": saved,
            "saved_tensor_bytes": sum(x["bytes"] for x in saved),
        }
        if device.type == "cuda":
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            record["peak_extra_allocated_bytes"] = (
                record["peak_allocated_bytes"] - initial_memory
            )
        record["process_peak_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
        del result, times, terms, loss
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
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                legacy_initial_memory = torch.cuda.memory_allocated(device)
            synchronize()
            start = time.perf_counter()
            historical = model(old_times, "v_max")
            synchronize()
            record["legacy_forward_ms"] = (time.perf_counter() - start) * 1000
            synchronize()
            start = time.perf_counter()
            historical.sum().backward()
            synchronize()
            record["legacy_backward_ms"] = (time.perf_counter() - start) * 1000
            if device.type == "cuda":
                record["legacy_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                    device
                )
                record["legacy_peak_extra_allocated_bytes"] = (
                    record["legacy_peak_allocated_bytes"] - legacy_initial_memory
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
                    "source_sha256": hashlib.sha256(
                        Path(functional.__file__).read_bytes()
                    ).hexdigest(),
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
            del model, historical, old_times


if __name__ == "__main__":
    main()
