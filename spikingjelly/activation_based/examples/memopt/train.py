import torch.nn as nn
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import callbacks
from spikingjelly.activation_based import functional, memopt

from data_module import CIFAR10DVSDataModule
from lightning_modules import ClassificationLightningModule
from lightning_callbacks import *
from models import VGGBlock



class CIFAR10DVSLightningModule(ClassificationLightningModule):
    def __init__(self, net: nn.Module, T: int, level: int, compress_x: bool, criterion: nn.Module):
        net = memopt.memory_optimization(
            net,
            (VGGBlock,),
            dummy_input=(torch.zeros(32, T, 2, 48, 48),),
            compress_x=compress_x,
            level=level,
            verbose=True,
        )
        super().__init__(net, criterion, num_classes=10, y_with_T=True)
        self.T = T


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
