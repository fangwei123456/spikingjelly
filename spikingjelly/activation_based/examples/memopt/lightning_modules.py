from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy
import torch.nn as nn


class ClassificationLightningModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        num_classes: int,
        y_with_T: bool = False,  # for computing accuracy
        **kwargs,
    ):
        super().__init__()
        self.y_with_T = y_with_T
        self.num_classes = num_classes
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.net = net
        self.criterion = criterion

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, label = batch[0].float(), batch[1]
        y = self(x)
        batch_loss = self.criterion(y, label)  # must properly handle the sizes!
        if self.y_with_T:
            y = y.mean(dim=0)
        if label.ndim > 1:
            label = label.argmax(dim=1)
        self.train_acc.update(y, label)
        self.train_loss.update(batch_loss.data)
        self.log("train_loss", self.train_loss.compute(), prog_bar=True)
        self.log("train_acc", self.train_acc.compute() * 100, prog_bar=True)
        return batch_loss

    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()
        train_loss = self.train_loss.compute()
        self.log("train_loss", train_loss, on_epoch=True, sync_dist=True)
        self.log("train_acc", train_acc * 100, on_epoch=True, sync_dist=True)
        self.train_acc.reset()
        self.train_loss.reset()

        if self.global_rank == 0:
            print(
                f"Epoch {self.current_epoch}/{self.trainer.max_epochs}: "
                f"train_loss={train_loss:.2f}, "
                f"train_acc={train_acc * 100:.2f}%"
            )

    def validation_step(self, batch, batch_idx):
        x, label = batch[0].float(), batch[1]
        y = self(x)
        batch_loss = self.criterion(y, label)  # must properly handle the sizes!
        if self.y_with_T:
            y = y.mean(dim=0)
        if label.ndim > 1:
            label = label.argmax(dim=1)
        self.val_acc.update(y, label)
        self.val_loss.update(batch_loss.data)
        return batch_loss

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_loss = self.val_loss.compute()
        self.log("val_acc", val_acc * 100, on_epoch=True, sync_dist=True)
        self.log("val_loss", val_loss, on_epoch=True, sync_dist=True)
        self.val_acc.reset()
        self.val_loss.reset()

        if self.global_rank == 0:
            print(
                f"Epoch {self.current_epoch}/{self.trainer.max_epochs}: "
                f"val_loss={val_loss:.2f}, val_acc={val_acc * 100:.2f}%"
            )
