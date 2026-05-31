import torch.nn as nn


class CNN(nn.Module):
    r"""
    **API Language:**
    :ref:`中文 <CNN-cn>` | :ref:`English <CNN-en>`

    ----

    .. _CNN-cn:
    * **中文**

    * **中文**

    用于 MNIST 分类的简单 ANN 卷积网络。可作为 ANN-to-SNN 转换的示例模型。

    ----

    .. _CNN-en:
    * **English**

    * **English**

    A simple ANN CNN for MNIST classification. Can be used as an example model for ANN-to-SNN conversion.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.network(x)
        return x
