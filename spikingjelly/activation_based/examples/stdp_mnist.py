"""Shallow MNIST STDP with current LIF and local neuronal adaptation.

浅层 MNIST STDP 与电流 LIF、层内循环抑制；标签仅用于学习后的类别指派。
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from spikingjelly.activation_based import base, functional, layer, learning, neuron


class _AdaptiveLIFNode(neuron.SimpleLIFNode):
    def __init__(
        self,
        neurons: int,
        homeostasis_gain: float,
        v_threshold: float,
        refractory_steps: int,
        bias_initial: float,
        steps: int,
    ):
        super().__init__(tau=100.0, decay_input=True, v_threshold=v_threshold)
        bias = (
            -torch.rand(neurons) / (1 / bias_initial)
            if bias_initial
            else torch.zeros(neurons)
        )
        self.register_buffer("bias_current", bias)
        self.homeostasis_gain = homeostasis_gain
        self.target_rate = 120 / (neurons * steps)
        self.refractory_steps = refractory_steps
        self.register_memory("adaptation", 0.0)
        self.register_memory("refractory", 0.0)

    def neuronal_charge(self, x: torch.Tensor) -> None:
        super().neuronal_charge(x + self.bias_current - self.adaptation)
        self.v = torch.where(
            torch.as_tensor(self.refractory, device=x.device) > 0, 0.0, self.v
        )

    def neuronal_fire(self) -> torch.Tensor:
        return (
            (self.v >= self.v_threshold)
            & (torch.as_tensor(self.refractory, device=self.v.device) <= 0)
        ).to(self.v)

    def neuronal_reset(self, spike: torch.Tensor) -> None:
        super().neuronal_reset(spike)
        previous = torch.as_tensor(
            self.refractory, device=spike.device, dtype=spike.dtype
        )
        self.refractory = torch.where(
            spike.bool(), self.refractory_steps, (previous - 1).clamp_min(0)
        )
        self.adaptation = math.exp(-1 / 200) * self.adaptation + 0.01 * spike
        if self.training:
            self.bias_current.add_(
                self.target_rate - spike.mean(0), alpha=self.homeostasis_gain
            )


class _CurrentFilter(base.MemoryModule):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
        self.register_memory("current", 0.0)

    def single_step_functional_forward(self, inputs, states, **kwargs):
        current = states[0] + (inputs[0] - states[0]) / self.tau
        return (current,), (current,)


class _PopulationCurrent(nn.Module):
    def __init__(self, neurons, inhibition):
        super().__init__()
        self.inhibition = inhibition
        self.register_buffer("input_gain", torch.ones(neurons))

    def forward(self, previous, current):
        if previous.shape != current.shape:
            raise ValueError("Recurrent batch shape changed; reset before reuse.")
        return self.input_gain * (
            current - current.mean(-1, keepdim=True)
        ) - self.inhibition * (previous.sum(-1, keepdim=True) - previous)


class _SNN(nn.Module):
    def __init__(
        self,
        neurons: int,
        homeostasis_gain: float,
        inhibition: float,
        v_threshold: float = 0.01,
        inhibition_tau: float = 5.0,
        refractory_steps: int = 5,
        bias_initial: float = 0.01,
        *,
        steps: int,
        covariance: torch.Tensor,
        device: str,
    ):
        super().__init__()
        self.synapse = nn.Linear(784, neurons, bias=False)
        with torch.no_grad():
            self.synapse.weight.uniform_(0, 1)
            self.synapse.weight.mul_(78.4 / self.synapse.weight.sum(1, keepdim=True))
        self.sn = layer.ElementWiseRecurrentContainer(
            nn.Sequential(
                _CurrentFilter(inhibition_tau),
                _AdaptiveLIFNode(
                    neurons,
                    homeostasis_gain,
                    v_threshold,
                    refractory_steps,
                    bias_initial,
                    steps,
                ),
            ),
            _PopulationCurrent(neurons, inhibition),
        )
        self.requires_grad_(False)
        self.to(device)
        self.register_buffer(
            "pixel_covariance",
            covariance.to(device=device, dtype=self.synapse.weight.dtype),
        )
        self.register_buffer("reference_input_variance", self.input_variance().clone())
        self.refresh_gain()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sn(self.synapse(x))

    @torch.no_grad()
    def input_variance(self):
        w = self.synapse.weight
        field = w - w.mean(0, keepdim=True)
        return (field @ self.pixel_covariance * field).sum(1)

    @torch.no_grad()
    def refresh_gain(self):
        variance = self.input_variance()
        if not torch.isfinite(variance).all() or (variance <= 0).any():
            raise RuntimeError("Degenerate projected input variance; stop training.")
        self.sn.element_wise_function.input_gain.copy_(
            (self.reference_input_variance / variance).sqrt()
        )


@torch.no_grad()
def _training_covariance(pixels, learning_indices):
    x = pixels[learning_indices].flatten(1).float() / 255
    x = x - x.mean(0)
    return x.T @ x / (len(x) - 1)


def _predict(counts: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    scores = counts.new_zeros((counts.shape[0], 10))
    for label in range(10):
        selected = assignment == label
        if selected.any():
            scores[:, label] = counts[:, selected].mean(1)
    prediction = scores.argmax(1)
    return prediction.masked_fill(scores.max(1).values == 0, -1)


def _assign_labels(counts: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    means = torch.stack(
        [
            counts[labels == label].sum(0) / (labels == label).sum().clamp_min(1)
            for label in range(10)
        ]
    )
    assignment = means.argmax(0)
    return assignment.masked_fill(means.max(0).values == 0, -1)


@torch.no_grad()
def _run_batch(model, images, steps, generator, learner=None, baseline=None):
    counts = images.new_zeros((len(images), model.synapse.out_features))
    baseline_counts = torch.zeros_like(counts) if baseline is not None else None
    probability = images.flatten(1) * 0.064
    encoder = neuron.SimpleIFNode(v_threshold=1.0, v_reset=None).to(images.device)
    encoder.v = torch.rand(probability.shape, device=images.device, generator=generator)
    for _ in range(steps):
        spikes = encoder(probability)
        counts.add_(model(spikes))
        if baseline is not None:
            baseline_counts.add_(baseline(spikes))
        if learner is not None:
            dw = learner.step(on_grad=False, scale=1 / len(images))
            weight = model.synapse.weight
            weight.add_(dw).clamp_(0, 1)
            weight.mul_(78.4 / weight.sum(1, keepdim=True).clamp_min(1e-12))
            weight.clamp_(0, 1)
    if learner is not None:
        model.refresh_gain()
        if baseline is not None:
            baseline.refresh_gain()
        learner.reset()
    functional.reset_net(model.sn)
    if baseline is not None:
        functional.reset_net(baseline.sn)
    return counts, baseline_counts


@torch.no_grad()
def _features(model, baseline, images, args, seed):
    model.eval()
    baseline.eval()
    generator = torch.Generator(device=images.device).manual_seed(seed)
    learned, frozen = [], []
    for offset in range(0, len(images), args.batch_size):
        counts, control = _run_batch(
            model,
            images[offset : offset + args.batch_size],
            args.steps,
            generator,
            baseline=baseline,
        )
        learned.append(counts)
        frozen.append(control)
    return torch.cat(learned), torch.cat(frozen)


def _save_checkpoint(
    path, model, baseline, assignment, baseline_assignment, config, epoch
):
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "baseline": {k: v.cpu() for k, v in baseline.state_dict().items()},
            "assignment": assignment.cpu(),
            "baseline_assignment": baseline_assignment.cpu(),
            "config": config,
            "epoch": epoch,
        },
        temporary,
    )
    temporary.replace(path)


def _load_models(checkpoint, device):
    config = checkpoint["config"]
    if (
        config.get("model_kind") != "current_stdp_mnist"
        or config.get("input_encoding") != "regular_if_uniform_phase"
    ):
        raise ValueError(
            "Current STDP MNIST checkpoint required; use archived source for old experiments."
        )
    model = _SNN(
        config["neurons"],
        config["homeostasis_gain"],
        config["inhibition"],
        config["v_threshold"],
        config["inhibition_tau"],
        config["refractory_steps"],
        config["bias_initial"],
        steps=config["steps"],
        covariance=checkpoint["model"]["pixel_covariance"],
        device=device,
    )
    baseline = copy.deepcopy(model)
    model.load_state_dict(checkpoint["model"])
    baseline.load_state_dict(checkpoint["baseline"])
    return model.eval(), baseline.eval()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=Path("logs/plasticity/data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("logs/plasticity/current/mnist")
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neurons", type=int, default=3200)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--ltp", type=float, default=0.00229619779275505)
    parser.add_argument("--homeostasis-gain", type=float, default=0.007774997243242785)
    parser.add_argument("--inhibition-tau", type=float, default=5.0)
    parser.add_argument("--refractory-steps", type=int, default=5)
    parser.add_argument("--bias-initial", type=float, default=0.01)
    parser.add_argument(
        "--v-threshold",
        type=float,
        default=0.01,
        help="Base membrane firing threshold in native voltage units.",
    )
    parser.add_argument(
        "--inhibition",
        type=float,
        default=0.09978118161925603,
        help="Fixed negative recurrent current strength; zero disables feedback.",
    )
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--calibration-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=10000)
    parser.add_argument("--select-best", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path, help="Evaluate a frozen checkpoint on the test set."
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=43200,
        help="Stop at a batch boundary; an interrupted run is not accepted.",
    )
    return parser


def main() -> None:
    r"""
    **API Language** - :ref:`中文 <stdp-mnist-main-cn>` | :ref:`English <stdp-mnist-main-en>`

    ----

    .. _stdp-mnist-main-cn:

    * **中文**

    从命令行训练或评估浅层 MNIST STDP 网络。训练不使用标签或梯度；
    训练集标签仅用于学习后固定的神经元类别指派。``--checkpoint`` 只评估
    冻结模型；其余参数和数据划分见 STDP 教程。输出 checkpoint 和 JSON 指标。
    循环抑制电流延迟一个时间步，不应期只约束自身，允许同时多神经元发放。
    输入使用随机像素相位的普通 IF 编码；每批清空膜电位、快适应电流、滤波器
    和 STDP trace，保留慢适应电流 B 与输入增益。评估冻结权重、B 和增益；
    训练数据的像素协方差只用于无标签输入增益约束，随 checkpoint 保存。

    :raises ValueError: 命令行样本数量、网络规模、系数或预算不合法。
    :raises TimeoutError: 训练超过时间上限；保存未完成状态，但不视为验收通过。
    :raises RuntimeError: 投影输入方差退化或非有限。

    ----

    .. _stdp-mnist-main-en:

    * **English**

    Train or evaluate a shallow MNIST STDP network from the command line.
    Weight learning uses neither labels nor gradients. Training-set labels only
    assign fixed neuron classes after learning. ``--checkpoint`` evaluates a
    frozen model; see the STDP tutorial for configuration and data splits.
    Writes checkpoints and JSON metrics.
    Recurrent inhibitory current has a one-step delay and individual refractory
    periods; multiple simultaneous spikes are allowed. Input uses ordinary IF
    coding with random pixel phases. Batches clear voltage, fast adaptation,
    filter state and STDP traces, retaining slow intrinsic current B and input
    gain. Evaluation freezes weights, B and gain. Training-only pixel covariance
    provides an unlabeled gain constraint and is saved with the checkpoint.

    :raises ValueError: Invalid sample counts, model sizes, coefficients or budget.
    :raises TimeoutError: Training exceeds its time limit; saves incomplete states
        without claiming acceptance.
    :raises RuntimeError: Projected input variance is degenerate or nonfinite.
    """
    args = _parser().parse_args()
    for name, maximum in (
        ("train_samples", 50000),
        ("calibration_samples", 5000),
        ("validation_samples", 5000),
        ("test_samples", 10000),
    ):
        if not 0 < getattr(args, name) <= maximum:
            raise ValueError(f"{name} must be between 1 and {maximum}.")
    if min(args.steps, args.batch_size, args.epochs) <= 0:
        raise ValueError("Steps, batch size and epochs must be positive.")
    if args.neurons < 2 or args.inhibition_tau < 1:
        raise ValueError("At least two neurons and current-filter tau >= 1 required.")
    if not all(
        math.isfinite(v) and v > 0
        for v in (
            args.ltp,
            args.homeostasis_gain,
            args.v_threshold,
            args.inhibition_tau,
            args.max_seconds,
        )
    ):
        raise ValueError(
            "Learning coefficients and time budget must be finite and positive."
        )
    if not math.isfinite(args.inhibition) or args.inhibition < 0:
        raise ValueError("Inhibition must be finite and non-negative.")
    if (
        args.refractory_steps < 0
        or not math.isfinite(args.bias_initial)
        or args.bias_initial < 0
    ):
        raise ValueError(
            "Refractory steps and initial bias range must be non-negative."
        )
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": args.device,
        "cuda": torch.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(args.device)
        if str(args.device).startswith("cuda")
        else None,
    }
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model, baseline = _load_models(checkpoint, args.device)
        # Evaluation uses the saved encoding and shape configuration.
        args.steps = checkpoint["config"]["steps"]
        args.batch_size = checkpoint["config"]["batch_size"]
        args.seed = checkpoint["config"]["seed"]
        dataset = torchvision.datasets.MNIST(
            args.data_dir, train=False, download=args.download
        )
        images = (
            dataset.data[: args.test_samples].to(args.device, dtype=torch.float32) / 255
        )
        labels = dataset.targets[: args.test_samples].to(args.device)
        counts, control = _features(model, baseline, images, args, args.seed + 30000)
        assignment = checkpoint["assignment"].to(args.device)
        baseline_assignment = checkpoint["baseline_assignment"].to(args.device)
        correct = int((_predict(counts, assignment) == labels).sum())
        baseline_correct = int((_predict(control, baseline_assignment) == labels).sum())
        metrics = {
            "seed": args.seed,
            "epoch": checkpoint["epoch"],
            "test_samples": len(labels),
            "correct": correct,
            "baseline_correct": baseline_correct,
            "accuracy": correct / len(labels),
            "baseline_accuracy": baseline_correct / len(labels),
            "silent_sample_fraction": (counts.sum(1) == 0).float().mean().item(),
            "environment": environment,
            "seconds": time.perf_counter() - started,
        }
        (args.output_dir / "test.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics), flush=True)
        return
    dataset = torchvision.datasets.MNIST(
        args.data_dir, train=True, download=args.download
    )
    split = torch.randperm(60000, generator=torch.Generator().manual_seed(42))
    train_idx = split[: args.train_samples]
    calibration_idx = split[50000 : 50000 + args.calibration_samples]
    validation_idx = split[55000 : 55000 + args.validation_samples]
    images = dataset.data.to(args.device, dtype=torch.float32) / 255
    train_images = images[train_idx.to(args.device)]
    calibration_images = images[calibration_idx.to(args.device)]
    validation_images = images[validation_idx.to(args.device)]
    calibration_labels = dataset.targets[calibration_idx].to(args.device)
    validation_labels = dataset.targets[validation_idx].to(args.device)
    covariance = _training_covariance(dataset.data, split[:50000])
    model = _SNN(
        args.neurons,
        args.homeostasis_gain,
        args.inhibition,
        args.v_threshold,
        args.inhibition_tau,
        args.refractory_steps,
        args.bias_initial,
        steps=args.steps,
        covariance=covariance,
        device=args.device,
    )
    baseline = copy.deepcopy(model)
    initial_weight = model.synapse.weight.clone()
    learner = learning.STDPLearner(
        "s",
        model.synapse,
        model.sn,
        20.0,
        20.0,
        f_pre=lambda w: args.ltp * w,
        f_post=lambda w: args.ltp * (1 - w),
    )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        split_seed=42,
        model_kind="current_stdp_mnist",
        input_encoding="regular_if_uniform_phase",
        gain_statistics=dict(samples=50000, split_seed=42, labels_used=False),
    )
    history = []
    best_accuracy = -1.0
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            baseline.train()
            learner.enable()
            generator = torch.Generator(device=args.device).manual_seed(
                args.seed + epoch
            )
            order = torch.randperm(
                len(train_images),
                generator=torch.Generator().manual_seed(args.seed + epoch),
            )
            for batch, offset in enumerate(range(0, len(order), args.batch_size)):
                if time.perf_counter() - started >= args.max_seconds:
                    raise TimeoutError(
                        "Experiment budget exhausted; no task acceptance claimed."
                    )
                batch_images = train_images[
                    order[offset : offset + args.batch_size].to(args.device)
                ]
                _run_batch(
                    model, batch_images, args.steps, generator, learner, baseline
                )
                if batch % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "processed": offset,
                                "seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
            learner.disable()
            counts, control = _features(
                model, baseline, calibration_images, args, args.seed + 10000
            )
            assignment = _assign_labels(counts, calibration_labels)
            baseline_assignment = _assign_labels(control, calibration_labels)
            counts, control = _features(
                model, baseline, validation_images, args, args.seed + 20000
            )
            metrics = {
                "epoch": epoch,
                "validation_accuracy": (
                    _predict(counts, assignment) == validation_labels
                )
                .float()
                .mean()
                .item(),
                "baseline_validation_accuracy": (
                    _predict(control, baseline_assignment) == validation_labels
                )
                .float()
                .mean()
                .item(),
                "silent_neuron_fraction": (assignment == -1).float().mean().item(),
                "saturated_weight_fraction": (
                    (model.synapse.weight == 0) | (model.synapse.weight == 1)
                )
                .float()
                .mean()
                .item(),
                "weight_change_l1": (model.synapse.weight - initial_weight)
                .abs()
                .mean()
                .item(),
                "seconds": time.perf_counter() - started,
            }
            history.append(metrics)
            _save_checkpoint(
                args.output_dir / f"epoch-{epoch}.pt",
                model,
                baseline,
                assignment,
                baseline_assignment,
                config,
                epoch,
            )
            if not args.select_best or metrics["validation_accuracy"] > best_accuracy:
                _save_checkpoint(
                    args.output_dir / "selected.pt",
                    model,
                    baseline,
                    assignment,
                    baseline_assignment,
                    config,
                    epoch,
                )
                best_accuracy = metrics["validation_accuracy"]
            result = {"config": config, "environment": environment, "history": history}
            (args.output_dir / "validation.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(json.dumps(metrics), flush=True)
            assert all(p.grad is None for p in model.parameters())
    except TimeoutError:
        torch.save(
            {
                "model": model.state_dict(),
                "baseline": baseline.state_dict(),
                "config": config,
                "epoch": epoch,
                "processed": offset,
            },
            args.output_dir / "interrupted.pt",
        )
        raise
    finally:
        learner.close()


if __name__ == "__main__":
    main()
