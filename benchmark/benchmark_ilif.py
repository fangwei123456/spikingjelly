import argparse

import torch

from spikingjelly.activation_based import neuron


def _run_once(
    node: neuron.ILIFNode,
    x: torch.Tensor,
    backward: bool,
) -> None:
    if backward and isinstance(node.v, torch.Tensor):
        node.v = node.v.detach()
    if x.grad is not None:
        x.grad = None
    y = node(x)
    if backward:
        y.sum().backward()


def _measure(
    node: neuron.ILIFNode,
    x: torch.Tensor,
    backward: bool,
    warmup: int,
    repeat: int,
) -> tuple[float, float]:
    node.reset()
    for _ in range(warmup):
        _run_once(node, x, backward)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        _run_once(node, x, backward)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat, torch.cuda.max_memory_allocated() / 2**20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--numel", type=int, default=1 << 18)
    parser.add_argument("--step-mode", choices=("s", "m"), default="m")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--mode", choices=("eval", "train"), default="train")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    dtype = getattr(torch, args.dtype)
    backward = args.mode == "train"
    shape = (args.T, args.numel) if args.step_mode == "m" else (args.numel,)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=backward)
    results = {}
    for backend in ("torch", "triton"):
        node = neuron.ILIFNode(
            step_mode=args.step_mode,
            backend=backend,
            store_v_seq=False,
        ).to(device="cuda", dtype=dtype)
        node.train(backward)
        if backward:
            context = torch.enable_grad()
        else:
            context = torch.inference_mode()
        with context:
            results[backend] = _measure(
                node,
                x,
                backward,
                args.warmup,
                args.repeat,
            )

    torch_ms, torch_memory = results["torch"]
    triton_ms, triton_memory = results["triton"]
    print(
        f"mode={args.mode} step_mode={args.step_mode} dtype={args.dtype} "
        f"T={args.T if args.step_mode == 'm' else 1} numel={args.numel} "
        f"torch={torch_ms:.3f}ms/{torch_memory:.1f}MiB "
        f"triton={triton_ms:.3f}ms/{triton_memory:.1f}MiB "
        f"speedup={torch_ms / triton_ms:.2f}x"
    )


if __name__ == "__main__":
    main()
