import torch
from torch import nn
from torch.nn.utils import parametrize

from .. import neuron
from .srm_mnist import _RowWeights, _train


class _CubicNet(nn.Module):
    def __init__(self, hidden_features=128):
        super().__init__()
        self.hidden = neuron.CubicSRM(784, hidden_features, duration=4.0, support=2.0)
        self.output = neuron.CubicSRM(hidden_features, 10, duration=4.0, support=2.0)
        for layer in (self.hidden, self.output):
            with torch.no_grad():
                layer.weight.normal_(0.0, 0.5)
            parametrize.register_parametrization(layer, "weight", _RowWeights(2.0))

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
        return 10 * scores.reshape(batch_count, 10)


def main() -> None:
    r"""
    **API Language** - :ref:`中文 <timing-cubic-mnist-cn>` | :ref:`English <timing-cubic-mnist-en>`

    ----

    .. _timing-cubic-mnist-cn:

    * **中文**

    用通用求解器训练两个单发 CubicSRM 层，MNIST 784→128→10。
    像素编码为 ``1-pixel`` 时刻，层间传紧凑事件。两层 PSP 支撑 2、窗口 4、
    阈值 1；示例使用 ``2*softmax(weight)`` 可微行约束。每输出至多一发，
    logits 为 ``10*(4-time)``，沉默类别为零；更早发放得分更高。
    loss 为交叉熵，Adam 默认 lr=0.01，无替代梯度。共享 SRM 示例的训练流程：
    完整 55000/5000/10000 划分、验证选 checkpoint/早停、最终一次测试，
    保存配置、源码指纹、指标及权重。默认 batch=32、10 epochs、patience=3、seed=0。
    这是固定三次核示例，通用接口的其他核需要自行提供完整单调分段。

    ----

    .. _timing-cubic-mnist-en:

    * **English**

    Train two single-spike CubicSRM layers through the general solver on MNIST,
    784→128→10. Pixels become ``1-pixel`` times and layers exchange compact events.
    Both PSP supports are 2, windows 4 and thresholds 1; the demo differentiably
    constrains rows to ``2*softmax(weight)``. Each output emits at most once;
    logits are ``10*(4-time)``, zero for silent classes, favoring earlier spikes.
    Cross-entropy and Adam (default lr=0.01) use no surrogate. Reuse the SRM demo's
    complete 55000/5000/10000 split, validation checkpoint/early stopping and one
    final test evaluation, saving configuration, source fingerprints, metrics and
    weights. Defaults: batch=32, 10 epochs, patience=3, seed=0. This fixed cubic
    kernel example does not supply complete partitions for other kernels.
    """
    _train(_CubicNet, "cubic-srm-mnist", batch_size=32)


if __name__ == "__main__":
    main()
