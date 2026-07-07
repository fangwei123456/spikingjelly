from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import torch

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.neuron import STBIFNeuron


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _first_real_then_zero_sequence(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    x_seq = torch.zeros(time_steps, *x.shape, device=x.device, dtype=x.dtype)
    x_seq[0] = x
    return x_seq


def _run_old_loop(neuron: STBIFNeuron, x_seq: torch.Tensor) -> torch.Tensor:
    functional.reset_net(neuron)
    outputs = []
    for x in x_seq:
        outputs.append(neuron.single_step_forward(x))
    return torch.stack(outputs, dim=0)


def _time_call(device: torch.device, repeat: int, fn):
    seconds = []
    result = None
    for _ in range(repeat):
        _sync(device)
        start = time.perf_counter()
        result = fn()
        _sync(device)
        seconds.append(time.perf_counter() - start)
    return min(seconds), result


def _state(neuron: STBIFNeuron) -> dict[str, torch.Tensor]:
    return {
        "q": neuron.q.detach().clone(),
        "acc_q": neuron.acc_q.detach().clone(),
        "cur_output": neuron.cur_output.detach().clone(),
    }


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).detach().abs().max().item())


def _compare_state(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]
) -> dict[str, float]:
    return {key: _max_abs_diff(a[key], b[key]) for key in a}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--time-steps", type=int, default=64)
    parser.add_argument("--numel", type=int, default=1048576)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--level", type=int, default=32)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.time_steps <= 0:
        raise ValueError("--time-steps must be positive.")
    if args.numel <= 0:
        raise ValueError("--numel must be positive.")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {args.device!r} requested but CUDA is not available."
        )
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    x = (
        torch.randn(args.numel, generator=generator, dtype=torch.float32)
        * args.scale
        * 2
    )
    x = x.to(device=device, dtype=dtype)
    x_seq = _first_real_then_zero_sequence(x, args.time_steps)

    reference = STBIFNeuron(
        args.scale,
        level=args.level,
        sym=args.sym,
    ).to(device)
    torch_opt = STBIFNeuron(
        args.scale,
        level=args.level,
        sym=args.sym,
    ).to(device)
    functional.set_step_mode(torch_opt, "m")

    with torch.inference_mode():
        loop_seconds, loop_out = _time_call(
            device,
            args.repeat,
            lambda: _run_old_loop(reference, x_seq),
        )
        loop_state = _state(reference)

        torch_opt.backend = "torch"
        torch_seconds, torch_out = _time_call(
            device,
            args.repeat,
            lambda: (functional.reset_net(torch_opt), torch_opt(x_seq))[1],
        )
        torch_state = _state(torch_opt)

        result = {
            "host": socket.gethostname(),
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "dtype": args.dtype,
            "shape": list(x_seq.shape),
            "time_steps": args.time_steps,
            "numel": args.numel,
            "level": args.level,
            "scale": args.scale,
            "sym": args.sym,
            "loop_seconds": loop_seconds,
            "torch_seconds": torch_seconds,
            "torch_speedup_vs_loop": loop_seconds / torch_seconds,
            "torch_max_abs_diff": _max_abs_diff(loop_out, torch_out),
            "torch_state_max_abs_diff": _compare_state(loop_state, torch_state),
        }

        if device.type == "cuda":
            triton_neuron = STBIFNeuron(
                args.scale,
                level=args.level,
                sym=args.sym,
            ).to(device)
            functional.set_step_mode(triton_neuron, "m")
            triton_neuron.backend = "triton"
            triton_seconds, triton_out = _time_call(
                device,
                args.repeat,
                lambda: (functional.reset_net(triton_neuron), triton_neuron(x_seq))[1],
            )
            triton_state = _state(triton_neuron)
            result.update(
                {
                    "triton_seconds": triton_seconds,
                    "triton_speedup_vs_loop": loop_seconds / triton_seconds,
                    "triton_speedup_vs_torch": torch_seconds / triton_seconds,
                    "triton_max_abs_diff": _max_abs_diff(loop_out, triton_out),
                    "triton_state_max_abs_diff": _compare_state(
                        loop_state, triton_state
                    ),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
