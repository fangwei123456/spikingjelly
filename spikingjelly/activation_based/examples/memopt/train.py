import torch.nn as nn
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import callbacks
from spikingjelly.activation_based import functional, memopt

from data_module import SCIFARDataModule
from lightning_modules import ClassificationLightningModule
from lightning_callbacks import *
from models import ConvBNNeuron, AvgPoolFlattenLinearNeuron



class SCIFARLightningModule(ClassificationLightningModule):
    def __init__(self, net: nn.Module, level: int, compress_x: bool, criterion: nn.Module):
        functional.set_step_mode(net, "m")
        net = memopt.memory_optimization(
            net,
            (ConvBNNeuron, AvgPoolFlattenLinearNeuron),
            dummy_input=(torch.rand(128, 3, 32, 32),),
            compress_x=compress_x,
            level=level,
            verbose=True,
        )
        super().__init__(
            net, criterion, num_classes=net.num_classes, y_with_T=False
        )


def main():
    cli = LightningCLI(
        SCIFARLightningModule,
        SCIFARDataModule,
        run=False,
        trainer_defaults={
            "logger": {
                "class_path": "CSVLogger",
                "init_args": {"save_dir": "./logs", "name": "SCIFAR"},
            },
            "enable_model_summary": False,
            "enable_checkpointing": False,
        },
    )
    assert cli.model.num_classes == cli.datamodule.num_classes
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
