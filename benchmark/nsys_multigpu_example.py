"""Independent two-GPU DP/DDP/PP capture and correctness workload.

Run DP with Python and DDP/PP with torchrun (two workers). Manual control uses
ready/go/done/exit files in --gate-dir so the shell controls the NSYS session.
No profiling interface is added to SpikingJelly distributed training.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch import nn

from spikingjelly import nsys
from spikingjelly.activation_based import functional, neuron


class _SmallStage(nn.Module):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.linear = nn.Linear(64 if stage == 0 else 128, 128 if stage == 0 else 16)
        self.lif = neuron.LIFNode(step_mode="m", v_threshold=0.5)

    def forward(self, x):
        x = self.linear(x)
        if self.stage == 0:
            x = self.lif(x.transpose(0, 1)).transpose(0, 1)
            functional.reset_net(self)
        else:
            x = x.mean(1)
        return x


class _ImageStage(nn.Module):
    def __init__(self, model, stage):
        super().__init__()
        self.stage = stage
        self.layers = (
            nn.Sequential(model.patch_embed, *model.blocks[:3])
            if stage == 0
            else nn.Sequential(*model.blocks[3:])
        )
        self.head = model.head if stage == 1 else None

    def forward(self, x):
        x = (
            x.unsqueeze(0).repeat(4, 1, 1, 1, 1)
            if self.stage == 0
            else x.transpose(0, 1)
        )
        x = self.layers(x)
        result = (
            x.transpose(0, 1).contiguous()
            if self.stage == 0
            else self.head(x.flatten(3).mean(-1)).mean(0)
        )
        functional.reset_net(self)
        return result


def _wait(path, timeout=120):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parallel", choices=("dp", "ddp", "pp", "graph"), required=True
    )
    parser.add_argument(
        "--phase", choices=("training", "inference"), default="training"
    )
    parser.add_argument("--model", choices=("small", "spikformer_s"), default="small")
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16, help="Global batch")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--control", choices=("api", "manual", "none"), default="api")
    parser.add_argument("--markers", choices=("on", "off", "none"), default="on")
    parser.add_argument("--gate-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Compare every step with a single-device reference",
    )
    parser.add_argument("--delay-rank", type=int, default=-1)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--fail-rank", type=int, default=-1)
    args = parser.parse_args()
    if args.validate and (args.model != "small" or args.parallel == "graph"):
        parser.error("--validate requires --model small and --parallel dp, ddp or pp")
    distributed = args.parallel in ("ddp", "pp")
    rank = int(os.environ["RANK"]) if distributed else 0
    world_size = int(os.environ["WORLD_SIZE"]) if distributed else 1
    if (distributed and world_size != 2) or torch.cuda.device_count() < 2:
        raise ValueError(
            "this example requires two visible GPUs and two torchrun workers for DDP/PP"
        )
    if args.batch_size % (args.microbatches if args.parallel == "pp" else 2):
        raise ValueError(
            "global batch must divide evenly between workers or microbatches"
        )
    if args.control == "manual" and args.gate_dir is None:
        raise ValueError("manual control requires --gate-dir")
    device = rank if distributed else 0
    torch.cuda.set_device(device)
    devices = [device] if distributed else [0, 1]
    control_group = None
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(seconds=45))
        control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=45))

    def barrier():
        if distributed:
            dist.barrier(group=control_group)

    def synchronize():
        for index in devices:
            torch.cuda.synchronize(index)

    def region(name, **metadata):
        return (
            nullcontext()
            if args.markers == "none"
            else nsys.region(name, args.markers == "on", **metadata)
        )

    torch.manual_seed(17)
    if args.model == "small":
        whole = nn.Sequential(_SmallStage(0), _SmallStage(1)).to(device)
        input_shape, activation_shape, classes = (4, 64), (4, 128), 16
    else:
        from spikingjelly.activation_based.model.spikformer import spikformer_s

        model = spikformer_s(backend="torch")
        whole = nn.Sequential(_ImageStage(model, 0), _ImageStage(model, 1)).to(device)
        input_shape, activation_shape, classes = (3, 224, 224), (4, 384, 14, 14), 1000
    training = args.phase == "training" and args.parallel != "graph"
    whole.train(training)
    reference = None
    if args.validate:
        import copy

        reference = copy.deepcopy(whole)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.01, momentum=0.9
        )
    module = whole[rank] if args.parallel == "pp" else whole
    if args.parallel == "dp":
        module = nn.DataParallel(module, device_ids=[0, 1])
    elif args.parallel == "ddp":
        module = nn.parallel.DistributedDataParallel(module, device_ids=[device])
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01, momentum=0.9)
    generator = torch.Generator().manual_seed(123)
    images = torch.randn(args.batch_size, *input_shape, generator=generator).to(device)
    targets = torch.randn(args.batch_size, classes, generator=generator).to(device)
    local_images = images.chunk(2)[rank] if args.parallel == "ddp" else images
    local_targets = targets.chunk(2)[rank] if args.parallel == "ddp" else targets
    graph_pairs = []
    if args.parallel == "graph":
        for index in devices:
            with torch.cuda.device(index):
                value = torch.ones(1024, device=index)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        value.sin()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with (
                    torch.cuda.graph(graph, stream=stream),
                    region("forward", stage=index),
                ):
                    result = value.sin() + value
                graph_pairs.append((graph, result))

    def run_step(index, marked):
        step_scope = (
            nsys.step(index, args.phase, marked, rank=rank, world_size=world_size)
            if args.markers != "none"
            else nullcontext()
        )
        with step_scope:
            if marked and rank == args.fail_rank and index == 2:
                raise RuntimeError("intentional worker failure")
            output = None
            with region("zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            if args.parallel == "graph":
                with region("graph_runner"):
                    for device_index, (graph, _) in zip(devices, graph_pairs):
                        with torch.cuda.device(device_index):
                            graph.replay()
                return
            if args.parallel == "pp":
                activations, outputs = [], []
                chunks = local_images.chunk(args.microbatches)
                labels = targets.chunk(args.microbatches)
                for microbatch in range(args.microbatches):
                    metadata = {"stage": rank, "microbatch": microbatch}
                    if rank == 0:
                        with region("forward", **metadata):
                            activation = module(chunks[microbatch])
                        activations.append(activation)
                        with region("send", **metadata):
                            dist.send(activation.detach().contiguous(), dst=1)
                    else:
                        activation = torch.empty(
                            args.batch_size // args.microbatches,
                            *activation_shape,
                            device=device,
                        )
                        with region("recv", **metadata):
                            dist.recv(activation, src=0)
                        activation.requires_grad_(training)
                        activations.append(activation)
                        with region("forward", **metadata):
                            outputs.append(module(activation))
                if rank == 1:
                    output = torch.cat(outputs)
                if training:
                    for microbatch in reversed(range(args.microbatches)):
                        metadata = {"stage": rank, "microbatch": microbatch}
                        if rank == 1:
                            with region("loss", **metadata):
                                loss = (
                                    nn.functional.mse_loss(
                                        outputs[microbatch], labels[microbatch]
                                    )
                                    / args.microbatches
                                )
                            with region("backward", **metadata):
                                loss.backward()
                            with region("send", **metadata):
                                dist.send(
                                    activations[microbatch].grad.contiguous(), dst=0
                                )
                        else:
                            gradient = torch.empty(
                                activations[microbatch].shape, device=device
                            )
                            with region("recv", **metadata):
                                dist.recv(gradient, src=1)
                            with region("backward", **metadata):
                                activations[microbatch].backward(gradient)
            else:
                with region("forward"):
                    output = module(local_images)
                if training:
                    with region("loss"):
                        loss = nn.functional.mse_loss(output, local_targets)
                    with region("backward"):
                        loss.backward()
            if reference is not None:
                reference_optimizer.zero_grad(set_to_none=True)
                expected = reference(images)
                if output is not None:
                    torch.testing.assert_close(
                        output,
                        expected.chunk(2)[rank] if args.parallel == "ddp" else expected,
                        rtol=1e-4,
                        atol=1e-6,
                    )
                if training:
                    nn.functional.mse_loss(expected, targets).backward()
                    ref_part = reference[rank] if args.parallel == "pp" else reference
                    for actual, expected_parameter in zip(
                        module.parameters(), ref_part.parameters(), strict=True
                    ):
                        torch.testing.assert_close(
                            actual.grad, expected_parameter.grad, rtol=1e-4, atol=1e-6
                        )
            if training:
                with region("optimizer"):
                    optimizer.step()
                if reference is not None:
                    reference_optimizer.step()
                    for actual, expected_parameter in zip(
                        module.parameters(), ref_part.parameters(), strict=True
                    ):
                        torch.testing.assert_close(
                            actual, expected_parameter, rtol=1e-4, atol=1e-6
                        )
                        torch.testing.assert_close(
                            optimizer.state[actual]["momentum_buffer"],
                            reference_optimizer.state[expected_parameter][
                                "momentum_buffer"
                            ],
                            rtol=1e-4,
                            atol=1e-6,
                        )
            with region("reset"):
                functional.reset_net(whole)
            if reference is not None:
                functional.reset_net(reference)
                assert all(
                    not isinstance(m.v, torch.Tensor)
                    for m in whole.modules()
                    if isinstance(m, neuron.LIFNode)
                )

    try:
        with torch.enable_grad() if training else torch.no_grad():
            for index in range(args.warmup):
                run_step(index, False)
            synchronize()
            barrier()
            if rank == args.delay_rank:
                time.sleep(args.delay_seconds)
            if args.control == "manual":
                args.gate_dir.mkdir(parents=True, exist_ok=True)
                (args.gate_dir / f"ready-{rank}").touch()
                _wait(args.gate_dir / "go")
            with nsys.capture(args.control == "api", devices=devices):
                barrier()
                started = time.perf_counter()
                for index in range(args.steps):
                    run_step(index, args.markers == "on")
                if rank == args.delay_rank:
                    time.sleep(args.delay_seconds)
                synchronize()
                elapsed = time.perf_counter() - started
                barrier()
            result = {
                "case": vars(args) | {"rank": rank, "world_size": world_size},
                "mean_step_wall_ms": elapsed * 1000 / args.steps,
                "validated_steps": args.steps if args.validate else 0,
                "pid": os.getpid(),
                "devices": [torch.cuda.get_device_name(i) for i in devices],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_name(f"{args.output.stem}-rank{rank}.json").write_text(
                json.dumps(result, default=str, indent=2)
            )
            if args.control == "manual":
                (args.gate_dir / f"done-{rank}").touch()
                _wait(args.gate_dir / "exit")
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
