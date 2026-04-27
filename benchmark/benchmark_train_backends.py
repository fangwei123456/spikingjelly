"""
Benchmark training performance and memory for 4 modes:

1) torch backend
2) triton backend
3) torch backend + torch.compile
4) triton backend + torch.compile

The benchmark uses a multi-step SNN ConvNet with BN:
    x[T, B, C, H, W]
      -> Conv2d -> BN -> LIF -> Pool
      -> Conv2d -> BN -> LIF -> Pool
      -> Flatten -> Linear -> LIF -> Linear
      -> mean over T -> logits[B, num_classes]
"""

import time
from dataclasses import dataclass
from importlib.util import find_spec

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate

# ---------------------------------------------------------------------------
# Fixed benchmark config
# Edit these constants directly if you want different settings.
# ---------------------------------------------------------------------------

T = 16
BATCH_SIZE = 64
IN_CHANNELS = 3
IMG_SIZE = 64
CONV1_CHANNELS = 64
CONV2_CHANNELS = 128
FC_HIDDEN_DIM = 1024
NUM_CLASSES = 10
WARMUP = 20
ITERS = 100
LR = 1e-3
SG_ALPHA = 4.0
SEED = 20260420


@dataclass(frozen=True)
class BenchMode:
    name: str
    backend: str
    use_compile: bool


class SmallSNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: int,
        conv1_channels: int,
        conv2_channels: int,
        fc_hidden_dim: int,
        num_classes: int,
        backend: str,
        sg_alpha: float,
    ):
        super().__init__()
        sg = surrogate.Sigmoid(alpha=sg_alpha)

        self.conv1 = layer.Conv2d(
            in_channels=in_channels,
            out_channels=conv1_channels,
            kernel_size=3,
            padding=1,
            step_mode="m",
        )
        self.bn1 = layer.BatchNorm2d(conv1_channels, step_mode="m")
        self.sn1 = neuron.LIFNode(
            tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=sg,
            detach_reset=False,
            step_mode="m",
            backend=backend,
        )
        self.pool1 = layer.MaxPool2d(kernel_size=2, stride=2, step_mode="m")

        self.conv2 = layer.Conv2d(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            kernel_size=3,
            padding=1,
            step_mode="m",
        )
        self.bn2 = layer.BatchNorm2d(conv2_channels, step_mode="m")
        self.sn2 = neuron.LIFNode(
            tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=sg,
            detach_reset=False,
            step_mode="m",
            backend=backend,
        )
        self.pool2 = layer.MaxPool2d(kernel_size=2, stride=2, step_mode="m")

        feat_hw = img_size // 4
        feat_dim = conv2_channels * feat_hw * feat_hw
        self.flat = layer.Flatten(step_mode="m")
        self.fc1 = layer.Linear(feat_dim, fc_hidden_dim, step_mode="m")
        self.sn3 = neuron.LIFNode(
            tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=sg,
            detach_reset=False,
            step_mode="m",
            backend=backend,
        )
        self.fc2 = layer.Linear(fc_hidden_dim, num_classes, step_mode="m")

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x_seq = self.conv1(x_seq)
        x_seq = self.bn1(x_seq)
        x_seq = self.sn1(x_seq)
        x_seq = self.pool1(x_seq)

        x_seq = self.conv2(x_seq)
        x_seq = self.bn2(x_seq)
        x_seq = self.sn2(x_seq)
        x_seq = self.pool2(x_seq)

        x_seq = self.flat(x_seq)
        x_seq = self.fc1(x_seq)
        x_seq = self.sn3(x_seq)
        x_seq = self.fc2(x_seq)
        return x_seq.mean(0)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def try_compile_model(model: nn.Module, backend: str) -> nn.Module:
    if backend == "triton":
        # Keep compile settings conservative for Triton backend to improve
        # cross-version stability.
        try:
            return torch.compile(
                model,
                backend="inductor",
                options={
                    "triton.cudagraphs": False,
                    "triton.cudagraph_trees": False,
                },
            )
        except RuntimeError:
            # Some PyTorch versions reject/ignore these options.
            return torch.compile(model, backend="inductor")

    # For torch backend, use compile default behavior.
    return torch.compile(model, backend="inductor")


def run_one_mode(
    mode: BenchMode,
    model_state: dict,
    x_seq: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> None:
    model = SmallSNN(
        in_channels=IN_CHANNELS,
        img_size=IMG_SIZE,
        conv1_channels=CONV1_CHANNELS,
        conv2_channels=CONV2_CHANNELS,
        fc_hidden_dim=FC_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        backend=mode.backend,
        sg_alpha=SG_ALPHA,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.train()

    if mode.use_compile:
        model = try_compile_model(model, mode.backend)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Warmup
    for _ in range(WARMUP):
        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_seq)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    sync_if_cuda(device)

    # Timed run
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(ITERS):
        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_seq)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    sync_if_cuda(device)
    elapsed = time.perf_counter() - start

    avg_step_ms = elapsed / ITERS * 1e3
    samples_per_sec = BATCH_SIZE * ITERS / elapsed
    max_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else float("nan")
    )

    mem_str = f"{max_mem_mb:.1f} MB" if device.type == "cuda" else "N/A (CPU)"
    print(
        f"{mode.name:<30s}  "
        f"avg_step={avg_step_ms:8.3f} ms  "
        f"throughput={samples_per_sec:10.1f} samples/s  "
        f"max_mem={mem_str}"
    )


def main() -> None:
    has_cuda = torch.cuda.is_available()
    has_triton_pkg = find_spec("triton") is not None
    has_compile = hasattr(torch, "compile")

    device = torch.device("cuda" if has_cuda else "cpu")
    print(f"Device: {device}")
    if has_cuda:
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(
        f"Env   : triton_pkg={has_triton_pkg}, torch.compile={has_compile}, "
        f"torch={torch.__version__}"
    )
    print(
        f"Shape : T={T}, B={BATCH_SIZE}, C={IN_CHANNELS}, H=W={IMG_SIZE}, "
        f"conv=({CONV1_CHANNELS},{CONV2_CHANNELS}), fc={FC_HIDDEN_DIM}, cls={NUM_CLASSES}"
    )
    print("-" * 120)

    torch.manual_seed(SEED)
    if has_cuda:
        torch.cuda.manual_seed_all(SEED)

    x_seq = torch.randn(T, BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    y = torch.randint(NUM_CLASSES, (BATCH_SIZE,), device=device)

    # Build a reference model once, then share weights across all modes for fairness.
    ref_model = SmallSNN(
        in_channels=IN_CHANNELS,
        img_size=IMG_SIZE,
        conv1_channels=CONV1_CHANNELS,
        conv2_channels=CONV2_CHANNELS,
        fc_hidden_dim=FC_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        backend="torch",
        sg_alpha=SG_ALPHA,
    ).to(device)
    ref_state = ref_model.state_dict()

    all_modes = [
        BenchMode("torch backend", backend="torch", use_compile=False),
        BenchMode("triton backend", backend="triton", use_compile=False),
        BenchMode("torch + torch.compile", backend="torch", use_compile=True),
        BenchMode("triton + torch.compile", backend="triton", use_compile=True),
    ]

    for mode in all_modes:
        if mode.backend == "triton":
            if device.type != "cuda":
                print(f"{mode.name:<30s}  skipped (requires CUDA)")
                continue
            if not has_triton_pkg:
                print(f"{mode.name:<30s}  skipped (triton package not found)")
                continue

        if mode.use_compile and not has_compile:
            print(f"{mode.name:<30s}  skipped (torch.compile unavailable)")
            continue

        try:
            run_one_mode(mode, ref_state, x_seq, y, device)
        except Exception as e:
            print(f"{mode.name:<30s}  failed ({type(e).__name__}: {e})")

    print("-" * 120)
    print("Done.")


if __name__ == "__main__":
    main()
