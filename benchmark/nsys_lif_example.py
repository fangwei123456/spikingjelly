"""Small runnable SNN workload for the Nsight Systems tutorial."""

import argparse
import sys

import torch
from torch import nn

from spikingjelly import nsys
from spikingjelly.activation_based import functional, neuron


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "serve"), default="train")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires one CUDA GPU")
    torch.manual_seed(42)
    model = nn.Sequential(
        neuron.LIFNode(step_mode="m"),
        nn.Linear(128, 10),
    ).cuda()
    x = torch.randn(4, 16, 128, device="cuda")
    target = torch.randint(10, (16,), device="cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def run_step(index: int, training: bool, mark: bool) -> None:
        phase = "training" if training else "inference"
        with nsys.step(index, phase, mark):
            if training:
                with nsys.region("zero_grad", mark):
                    optimizer.zero_grad(set_to_none=True)
            with nsys.region("forward", mark):
                output = model(x).mean(0)
            if training:
                with nsys.region("loss", mark):
                    loss = nn.functional.cross_entropy(output, target)
                with nsys.region("backward", mark):
                    loss.backward()
                with nsys.region("optimizer", mark):
                    optimizer.step()
            with nsys.region("reset", mark):
                functional.reset_net(model)
            if not training:
                torch.cuda.synchronize()

    training = args.mode == "train"
    model.train(training)
    with torch.enable_grad() if training else torch.inference_mode():
        for index in range(args.warmup):
            run_step(index, training, False)
    torch.cuda.synchronize()
    if training:
        with nsys.capture(args.profile):
            for index in range(args.steps):
                run_step(index, True, args.profile)
            torch.cuda.synchronize()
    else:
        print("ready: enter 'run', 'profile', or 'quit'", flush=True)
        for line in sys.stdin:
            action = line.strip()
            if action == "quit":
                break
            if action in ("run", "profile"):
                profiling = action == "profile"
                with nsys.capture(profiling), torch.inference_mode():
                    run_step(0, False, profiling)
                print("done", flush=True)


if __name__ == "__main__":
    main()
