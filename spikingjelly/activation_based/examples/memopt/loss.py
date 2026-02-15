import torch
import torch.nn as nn
import torch.nn.functional as F


class TMeanCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, y_seq, label):
        y = y_seq.mean(dim=0)
        return super().forward(y, label)


class TETLoss(nn.Module):
    def __init__(
        self,
        base_criterion=nn.CrossEntropyLoss(),
        mean: float = 1.0,
        tet_lambda: float = 1e-3,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.mean = mean
        self.tet_lambda = tet_lambda

        if tet_lambda == 0:
            self.regularization_loss = self._regularization_loss_0
        else:
            self.regularization_loss = self._regularization_loss

    def base_criterion_loss(self, y, label):
        T = y.shape[0]
        l = 0
        for t in range(T):
            l += self.base_criterion(y[t], label)
        return l / T

    def _regularization_loss(self, y):
        reg = torch.full_like(y, self.mean)
        return F.mse_loss(y, reg)

    def _regularization_loss_0(self, y):
        return 0.0

    def forward(self, y, label):
        base_loss = self.base_criterion_loss(y, label)
        re_loss = self.regularization_loss(y)
        return (1.0 - self.tet_lambda) * base_loss + self.tet_lambda * re_loss
