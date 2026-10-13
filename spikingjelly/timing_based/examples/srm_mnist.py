import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .. import functional, neuron


class _RowWeights(nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.gain = gain

    def forward(self, weight):
        return self.gain * weight.flatten(1).softmax(1).reshape_as(weight)


class _SRMNet(nn.Module):
    def __init__(self, hidden_features=128):
        super().__init__()
        self.hidden = neuron.SRM(784, hidden_features, duration=2.0)
        self.output = neuron.SRM(hidden_features, 10, duration=2.0)
        with torch.no_grad():
            for layer in (self.hidden, self.output):
                layer.weight.normal_(0.0, 0.5)
        # Differentiable row gain maintains firing without changing SRM's Jacobian.
        for layer, gain in ((self.hidden, 8.0), (self.output, 2.0)):
            parametrize.register_parametrization(layer, "weight", _RowWeights(gain))

    def forward(self, images):
        batch_count = images.shape[0]
        times = (1 - images.flatten(1)).flatten()
        indices = torch.arange(784, device=images.device).repeat(batch_count)
        offsets = torch.arange(batch_count + 1, device=images.device) * 784
        events = self.hidden(times, indices, offsets)
        times, indices, offsets = self.output(*events)
        batches = torch.repeat_interleave(
            torch.arange(batch_count, device=images.device),
            offsets.diff(),
            output_size=times.numel(),
        )
        scores = times.new_zeros(batch_count * 10).index_add(
            0, batches * 10 + indices, self.output.duration - times
        )
        return 5 * scores.reshape(batch_count, 10)


def _evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            correct += (logits.argmax(1) == labels.to(device)).sum().item()
    return correct / len(loader.dataset), correct


def main() -> None:
    r"""
    **API Language** - :ref:`中文 <timing-srm-mnist-main-cn>` | :ref:`English <timing-srm-mnist-main-en>`

    ----

    .. _timing-srm-mnist-main-cn:

    * **中文**

    下载 MNIST，使用 55000/5000 训练/验证划分训练两个 SRM 层，最后评估
    完整 10000 测试样本。结构为 784→128→10（隐藏宽度可配置）；像素编码
    为 ``1-pixel`` 时刻，层间传紧凑多发事件，无均匀时间网格。每个输出事件
    对类别分数贡献 ``2-time``，logits 为累计贡献的 5 倍，使用交叉熵。
    用 Adam 训练原始权重，示例通过 PyTorch parametrization 将其可微映射为
    ``gain*softmax(weight)``，两层行增益分别为 8、2，保持发放并限制事件数量。
    SRM 本身允许负权重且不内置恢复。
    验证集选择 checkpoint/早停，测试集仅最终评估。保存配置、JSONL 指标、
    state_dict、源码 SHA256 和测试正确数。默认 10 轮、patience 3、seed 0。

    ----

    .. _timing-srm-mnist-main-en:

    * **English**

    Download MNIST, train two SRM layers on a 55000/5000 train/validation split,
    then evaluate all 10000 test examples. Architecture: 784→128→10 with configurable
    hidden width. Pixels become ``1-pixel`` times; compact multi-spike events pass
    between layers without a uniform grid. Each output event contributes ``2-time``
    to its class score; logits are five times the accumulated contribution and the
    loss is cross-entropy. Adam trains raw weights; the demo's PyTorch parametrization
    differentiably maps them to ``gain*softmax(weight)`` with row gains 8 and 2.
    This maintains firing and bounds event counts; SRM itself permits negative
    weights and has no recovery rule. Validation
    selects checkpoints/early stopping; test is evaluated only at the end. Save
    configuration, JSONL metrics, state_dict, source SHA256 and test correct counts.
    Defaults: 10 epochs, patience 3, seed 0.
    """
    _train(_SRMNet, "srm-mnist")


def _train(model_factory, output_name, batch_size=128, hidden_features=128):
    parser = argparse.ArgumentParser(description="Two-layer continuous-event SRM MNIST")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-dir", default="./data")
    parser.add_argument("--output-dir", default="./" + output_name)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--hidden-features", type=int, default=hidden_features)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.batch_size, args.hidden_features, args.epochs, args.patience) < 1:
        parser.error("batch size, hidden width, epochs and patience must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        torch_version=str(torch.__version__),
        device_name=torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(functional.__file__),
                Path(neuron.__file__),
                Path(__file__),
                Path(inspect.getfile(model_factory)),
                Path(functional.solver.__file__),
            )
        },
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    full_train = datasets.MNIST(
        args.dataset_dir, train=True, transform=transforms.ToTensor(), download=True
    )
    train_set, validation_set = random_split(
        full_train, [55000, 5000], generator=torch.Generator().manual_seed(args.seed)
    )
    test_set = datasets.MNIST(
        args.dataset_dir, train=False, transform=transforms.ToTensor(), download=True
    )
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, args.batch_size)
    test_loader = DataLoader(test_set, args.batch_size)
    model = model_factory(args.hidden_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_accuracy, stale_epochs = -1.0, 0
    with (output_dir / "metrics.jsonl").open("w") as metrics:
        for epoch in range(1, args.epochs + 1):
            model.train()
            start = time.perf_counter()
            correct, total_loss = 0, 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                correct += (logits.argmax(1) == labels).sum().item()
                total_loss += loss.item() * labels.numel()
            accuracy, _ = _evaluate(model, validation_loader, device)
            record = dict(
                epoch=epoch,
                train_loss=total_loss / len(train_set),
                train_accuracy=correct / len(train_set),
                validation_accuracy=accuracy,
                seconds=time.perf_counter() - start,
            )
            print(json.dumps(record), flush=True)
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            if accuracy > best_accuracy:
                best_accuracy, stale_epochs = accuracy, 0
                torch.save(
                    dict(state_dict=model.state_dict(), epoch=epoch, config=config),
                    output_dir / "best.pt",
                )
            else:
                stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    checkpoint = torch.load(
        output_dir / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    accuracy, correct = _evaluate(model, test_loader, device)
    result = dict(
        selected_epoch=checkpoint["epoch"],
        validation_accuracy=best_accuracy,
        test_accuracy=accuracy,
        test_correct=correct,
        test_count=len(test_set),
    )
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
