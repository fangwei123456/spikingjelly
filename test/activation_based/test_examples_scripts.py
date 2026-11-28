import sys

import pytest
import torch

pytest.importorskip("tensorboard")

from spikingjelly.activation_based.examples import (
    classify_dvsg,
    conv_fashion_mnist,
    rsnn_sequential_fmnist,
    train_flexsn_compile,
    train_imagenet,
)
from spikingjelly.activation_based.examples.common import train_classify


class _FakeDataset(torch.utils.data.Dataset):
    def __init__(self, *args, shape, num_classes, **kwargs):
        generator = torch.Generator().manual_seed(0)
        self.x = torch.rand((4, *shape), generator=generator)
        self.y = torch.arange(4) % num_classes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def _fake(shape, num_classes):
    return lambda *args, **kwargs: _FakeDataset(shape=shape, num_classes=num_classes)


@pytest.mark.parametrize(
    "script, patch, extra",
    [
        (
            conv_fashion_mnist,
            ("torchvision.datasets", "FashionMNIST", _fake((1, 28, 28), 10)),
            ["-T", "2", "-channels", "4"],
        ),
        (
            rsnn_sequential_fmnist,
            ("torchvision.datasets", "FashionMNIST", _fake((1, 28, 28), 10)),
            ["-model", "plain"],
        ),
        (
            classify_dvsg,
            (None, "DVS128Gesture", _fake((2, 2, 128, 128), 11)),
            ["-T", "2", "-channels", "4"],
        ),
    ],
    ids=["conv_fashion_mnist", "rsnn_sequential_fmnist", "classify_dvsg"],
)
def test_classification_scripts_run_without_opt(
    script, patch, extra, tmp_path, monkeypatch
):
    target, name, fake = patch
    owner = script if target is None else script.torchvision.datasets
    monkeypatch.setattr(owner, name, fake)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            script.__name__,
            "-device",
            "cpu",
            "-b",
            "2",
            "-epochs",
            "1",
            "-j",
            "0",
            "-data-dir",
            str(tmp_path / "data"),
            "-out-dir",
            str(tmp_path / "out"),
            *extra,
        ],
    )

    script.main()

    (checkpoint,) = tmp_path.glob("out/*_sgd_*/checkpoint_latest.pth")
    assert torch.load(checkpoint, weights_only=True)["epoch"] == 0


def _trainer_args(trainer, *extra):
    return trainer.get_args_parser().parse_args(["--device", "cpu", *extra])


def test_train_imagenet_model_uses_dataset_num_classes():
    trainer = train_imagenet.SResNetTrainer()
    args = _trainer_args(trainer, "--model", "spiking_resnet18", "--T", "2")
    model = trainer.load_model(args, num_classes=7)
    assert model.fc.out_features == 7


def test_top5_accuracy_with_fewer_than_five_classes():
    output = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    target = torch.tensor([0, 0, 1])
    acc1, acc5 = train_classify.utils.accuracy(output, target, topk=(1, 5))
    assert acc1.item() == pytest.approx(100.0 / 3)
    assert acc5.item() == 100.0


def test_train_flexsn_compile_builds_its_model():
    trainer = train_flexsn_compile._FlexSNTrainer()
    args = _trainer_args(trainer, "--model", "spiking_vgg11")
    model = trainer.load_model(args, num_classes=7)
    with torch.no_grad():
        y = model(torch.rand(2, 1, 3, 32, 32))
    assert y.shape == (2, 1, 7)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--lr-scheduler", "cosine", "--lr-warmup-epochs", "0"], "cosine"),
        (["--lr-scheduler", "cosine", "--lr-warmup-epochs", "2"], "cosine"),
        (["--lr-warmup-method", "linaer", "--lr-warmup-epochs", "2"], "linaer"),
    ],
)
def test_trainer_rejects_unknown_lr_scheduler_names(extra, message):
    trainer = train_classify.Trainer()
    args = _trainer_args(trainer, *extra)
    optimizer = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.1)
    with pytest.raises(RuntimeError, match=message):
        trainer.set_lr_scheduler(args, optimizer)
