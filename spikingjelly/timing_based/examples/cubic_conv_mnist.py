import torch
from torch import nn
from torch.nn.utils import parametrize

from .. import neuron
from .srm_mnist import _RowWeights, _train


class _CubicConvNet(nn.Module):
    def __init__(self, hidden_features=8):
        super().__init__()
        self.hidden = neuron.CubicConv2d(1, hidden_features, 5, 4.0, stride=2)
        self.output = neuron.CubicSRM(hidden_features * 12 * 12, 10, 4.0)
        for layer, gain in ((self.hidden, 8.0), (self.output, 2.0)):
            with torch.no_grad():
                layer.weight.normal_(0.0, 0.5)
            parametrize.register_parametrization(layer, "weight", _RowWeights(gain))

    def forward(self, images):
        batch_count = images.shape[0]
        times = (1 - images.flatten(1)).flatten()
        indices = torch.arange(784, device=images.device).repeat(batch_count)
        offsets = torch.arange(batch_count + 1, device=images.device) * 784
        events = self.hidden(times, indices, offsets, (28, 28))
        times, indices, offsets = self.output(*events)
        batches = torch.repeat_interleave(
            torch.arange(batch_count, device=images.device),
            offsets.diff(),
            output_size=times.numel(),
        )
        scores = times.new_zeros(batch_count * 10).index_add(
            0, batches * 10 + indices, self.output.duration - times
        )
        return 50 * scores.reshape(batch_count, 10)


def main() -> None:
    r"""
    **API Language** - :ref:`中文 <timing-cubic-conv-mnist-cn>` | :ref:`English <timing-cubic-conv-mnist-en>`

    ----

    .. _timing-cubic-conv-mnist-cn:

    * **中文**

    完整MNIST两层连续事件卷积示例：像素编码为 ``1-pixel``，
    ``1×28×28 → CubicConv2d(1→8,k5,stride2) → 8×12×12 → CubicSRM(1152→10)``。
    ``--hidden-features`` 指卷积输出通道数，默认8；无bias/pooling/BN。
    两层窗口4、支撑2、阈值1；参数按每个输出行使用可微 ``gain*softmax(raw)``，
    卷积/分类增益8/2。每输出最多一发，logits为 ``50*(4-time)``，沉默类别0。
    FP32求解/一阶时刻梯度，CE+Adam默认lr 0.01，无替代梯度。
    复用55000/5000/10000划分、验证选checkpoint/早停和最终一次测试，保存配置、
    源码指纹、指标、权重。默认batch32、10轮、patience3、seed0。

    ----

    .. _timing-cubic-conv-mnist-en:

    * **English**

    Complete two-layer continuous-event convolution on MNIST. Encode pixels as
    ``1-pixel`` and use ``1×28×28 → CubicConv2d(1→8,k5,stride2) → 8×12×12
    → CubicSRM(1152→10)``. ``--hidden-features`` is the convolution output-channel
    count, default 8; no bias/pooling/BN. Both windows are 4, supports 2 and thresholds 1.
    Differentiably normalize each output weight row to ``gain*softmax(raw)`` with
    convolution/classifier gains 8/2. Each output fires at most once;
    logits are ``50*(4-time)``, zero for silent classes. FP32 solving and first-order
    time gradients use cross-entropy and Adam (default lr 0.01), without a surrogate.
    Reuse the 55000/5000/10000 split, validation checkpoints/early stopping and one
    final test, saving configuration, source fingerprints, metrics and weights.
    Defaults: batch 32, 10 epochs, patience 3, seed 0.
    """
    _train(_CubicConvNet, "cubic-conv-mnist", batch_size=32, hidden_features=8)


if __name__ == "__main__":
    main()
