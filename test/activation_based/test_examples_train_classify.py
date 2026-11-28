import hashlib
import math
import os
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("tensorboard")
Image = pytest.importorskip("PIL.Image")

from spikingjelly.activation_based.examples.common import train_classify

_CROP = 8


class _Interrupted(Exception):
    pass


class _TinyTrainer(train_classify.Trainer):
    def __init__(self, cache_dir, test_acc1=(), stop_at_epoch=None):
        self.cache_dir = cache_dir
        self.test_acc1 = list(test_acc1)
        self.stop_at_epoch = stop_at_epoch

    def _get_cache_path(self, filepath):
        digest = hashlib.sha1(filepath.encode()).hexdigest()[:10]
        return os.path.join(self.cache_dir, digest + ".pt")

    def load_model(self, args, num_classes):
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * _CROP * _CROP, num_classes))

    def train_one_epoch(
        self, model, criterion, optimizer, data_loader, device, epoch, *args
    ):
        if epoch == self.stop_at_epoch:
            raise _Interrupted
        return super().train_one_epoch(
            model, criterion, optimizer, data_loader, device, epoch, *args
        )

    def evaluate(self, args, model, criterion, data_loader, device, log_suffix=""):
        loss, acc1, acc5 = super().evaluate(
            args, model, criterion, data_loader, device, log_suffix
        )
        if self.test_acc1:
            acc1 = self.test_acc1.pop(0)
        return loss, acc1, acc5


@pytest.fixture
def image_folder(tmp_path):
    generator = torch.Generator().manual_seed(0)
    for split in ("train", "val"):
        for label in range(5):
            folder = tmp_path / "data" / split / f"c{label}"
            folder.mkdir(parents=True)
            for i in range(2):
                pixels = torch.randint(
                    0, 256, (12, 12, 3), generator=generator, dtype=torch.uint8
                )
                Image.fromarray(pixels.numpy()).save(folder / f"{i}.png")
    return tmp_path / "data"


@pytest.fixture(autouse=True)
def _keep_global_determinism(monkeypatch):
    # The trainer enables torch.use_deterministic_algorithms process-wide.
    monkeypatch.setattr(train_classify, "set_deterministic", lambda *a, **k: None)


def _parse(trainer, data, out, *extra):
    return trainer.get_args_parser().parse_args(
        [
            "--data-path",
            str(data),
            "--output-dir",
            str(out),
            "--device",
            "cpu",
            "--disable-amp",
            "--workers",
            "0",
            "--batch-size",
            "4",
            "--lr-warmup-epochs",
            "0",
            "--mixup-alpha",
            "0",
            "--cutmix-alpha",
            "0",
            "--train-crop-size",
            str(_CROP),
            "--val-resize-size",
            str(_CROP),
            "--val-crop-size",
            str(_CROP),
            *extra,
        ]
    )


def _load(out: Path, name: str):
    (path,) = out.glob(f"pt/*/{name}")
    return torch.load(path, map_location="cpu", weights_only=True)


def _interrupt_then_resume(tmp_path, image_folder, first_acc1=(), resumed_acc1=()):
    trainer = _TinyTrainer(tmp_path / "cache", first_acc1, stop_at_epoch=1)
    with pytest.raises(_Interrupted):
        trainer.main(_parse(trainer, image_folder, tmp_path / "out", "--epochs", "2"))
    assert _load(tmp_path / "out", "checkpoint_latest.pth")["epoch"] == 0

    trainer = _TinyTrainer(tmp_path / "cache", resumed_acc1)
    trainer.main(
        _parse(
            trainer,
            image_folder,
            tmp_path / "out",
            "--epochs",
            "2",
            "--resume",
            "latest",
        )
    )


def test_trainer_resumes_an_interrupted_run(tmp_path, image_folder):
    _interrupt_then_resume(tmp_path, image_folder)

    checkpoint = _load(tmp_path / "out", "checkpoint_latest.pth")
    assert checkpoint["epoch"] == 1
    assert checkpoint["args"]["resume"] == "latest"


def test_trainer_resume_keeps_the_best_test_accuracy(tmp_path, image_folder):
    _interrupt_then_resume(tmp_path, image_folder, [80.0], [50.0])

    assert _load(tmp_path / "out", "checkpoint_latest.pth")["max_test_acc1"] == 80.0
    best = _load(tmp_path / "out", "checkpoint_max_test_acc1.pth")
    assert best["epoch"] == 0


def test_trainer_resumes_with_more_epochs(tmp_path, image_folder):
    trainer = _TinyTrainer(tmp_path / "cache")
    trainer.main(_parse(trainer, image_folder, tmp_path / "a", "--epochs", "1"))
    (first,) = (tmp_path / "a").glob("pt/*/checkpoint_latest.pth")

    trainer.main(
        _parse(
            trainer,
            image_folder,
            tmp_path / "b",
            "--epochs",
            "2",
            "--resume",
            str(first),
        )
    )
    assert _load(tmp_path / "b", "checkpoint_latest.pth")["epoch"] == 1


def test_trainer_resume_with_more_epochs_follows_the_new_schedule(
    tmp_path, image_folder
):
    trainer = _TinyTrainer(tmp_path / "cache")
    trainer.main(_parse(trainer, image_folder, tmp_path / "a", "--epochs", "1"))
    (first,) = (tmp_path / "a").glob("pt/*/checkpoint_latest.pth")

    trainer = _TinyTrainer(tmp_path / "cache", stop_at_epoch=2)
    with pytest.raises(_Interrupted):
        trainer.main(
            _parse(
                trainer,
                image_folder,
                tmp_path / "b",
                "--epochs",
                "3",
                "--resume",
                str(first),
            )
        )

    checkpoint = _load(tmp_path / "b", "checkpoint_latest.pth")
    assert checkpoint["epoch"] == 1
    # Cosine annealing planned for 3 epochs, after 2 of them.
    expected = 0.1 * (1 + math.cos(math.pi * 2 / 3)) / 2
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == pytest.approx(expected)


def test_trainer_resume_with_more_epochs_replays_the_saved_scheduler_steps(
    tmp_path, image_folder
):
    step = ("--lr-scheduler", "step", "--lr-step-size", "2")
    trainer = _TinyTrainer(tmp_path / "cache")
    trainer.main(
        _parse(
            trainer,
            image_folder,
            tmp_path / "a",
            "--start-epoch",
            "3",
            "--epochs",
            "4",
            *step,
        )
    )
    (first,) = (tmp_path / "a").glob("pt/*/checkpoint_latest.pth")

    trainer = _TinyTrainer(tmp_path / "cache", stop_at_epoch=5)
    with pytest.raises(_Interrupted):
        trainer.main(
            _parse(
                trainer,
                image_folder,
                tmp_path / "b",
                "--epochs",
                "6",
                "--resume",
                str(first),
                *step,
            )
        )

    # One scheduler step in the first run, one after epoch 4: one decay.
    checkpoint = _load(tmp_path / "b", "checkpoint_latest.pth")
    assert checkpoint["epoch"] == 4
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.01)


def test_trainer_resumes_a_legacy_namespace_checkpoint(
    tmp_path, image_folder, monkeypatch
):
    trainer = _TinyTrainer(tmp_path / "cache")
    args = _parse(trainer, image_folder, tmp_path / "a", "--epochs", "1")
    trainer.main(args)
    (first,) = (tmp_path / "a").glob("pt/*/checkpoint_latest.pth")
    # Checkpoints written before args became a dict hold the Namespace itself;
    # they load only with weights_only disabled.
    checkpoint = torch.load(first, map_location="cpu", weights_only=True)
    checkpoint["args"] = args
    torch.save(checkpoint, first)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    trainer.main(
        _parse(
            trainer,
            image_folder,
            tmp_path / "b",
            "--epochs",
            "2",
            "--resume",
            str(first),
        )
    )
    assert _load(tmp_path / "b", "checkpoint_latest.pth")["epoch"] == 1


def test_trainer_reloads_its_dataset_cache(tmp_path, image_folder, monkeypatch):
    trainer = _TinyTrainer(tmp_path / "cache")
    extra = ("--epochs", "1", "--cache-dataset")
    trainer.main(_parse(trainer, image_folder, tmp_path / "a", *extra))
    assert len(list((tmp_path / "cache").glob("*.pt"))) == 2

    def _must_use_cache(*args, **kwargs):
        raise AssertionError("the dataset should be loaded from the cache")

    monkeypatch.setattr(
        train_classify.torchvision.datasets, "ImageFolder", _must_use_cache
    )
    trainer.main(_parse(trainer, image_folder, tmp_path / "b", *extra))
    assert _load(tmp_path / "b", "checkpoint_latest.pth")["epoch"] == 0
