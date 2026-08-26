import torch
import torch.nn as nn
from data_module import CIFAR10DVSDataModule
from lightning.pytorch import callbacks
from lightning.pytorch.cli import LightningCLI
from lightning_callbacks import (
    GlobalMeanBatchTimeCallback,
    PeakMemoryTillNowCallback,
    SamplePerSecondCallback,
)
from lightning_modules import ClassificationLightningModule
from models import VGGBlock
from spikingjelly.activation_based import memopt, neuron


class CIFAR10DVSLightningModule(ClassificationLightningModule):
    def __init__(
        self, net: nn.Module, T: int, level: int, compress_x: bool, criterion: nn.Module
    ):
        super().__init__(net, criterion, num_classes=10, y_with_T=True)
        self.T = T
        self.memopt_level = level
        self.memopt_compress_inputs = compress_x

    def on_fit_start(self) -> None:
        dummy = torch.zeros(32, self.T, 2, 48, 48, device=self.device)
        memopt.optimize_memory(
            self.net,
            VGGBlock,
            lambda current: current(dummy),
            level=self.memopt_level,
            compress=self.memopt_compress_inputs,
            split_fn=lambda module: (module.proj_bn, module.neuron),
            can_chunk=lambda module: isinstance(module, neuron.BaseNode),
        )


def main():
    cli = LightningCLI(
        CIFAR10DVSLightningModule,
        CIFAR10DVSDataModule,
        run=False,
        trainer_defaults={
            "logger": {
                "class_path": "CSVLogger",
                "init_args": {"save_dir": "./logs", "name": "CIFAR10DVS"},
            },
            "enable_model_summary": False,
            "enable_checkpointing": False,
        },
    )
    assert cli.model.T == cli.datamodule.T
    cli.trainer.callbacks += [
        callbacks.ModelSummary(max_depth=-1),
        callbacks.ModelCheckpoint(
            filename="best-{epoch}-{train_acc:.4f}-{val_acc:.4f}",
            save_top_k=1,
            monitor="val_acc",
            mode="max",
        ),
        GlobalMeanBatchTimeCallback(reset_per_epoch=True),
        SamplePerSecondCallback(),
        PeakMemoryTillNowCallback(),
    ]
    if cli.trainer.is_global_zero:
        print(cli.model)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
