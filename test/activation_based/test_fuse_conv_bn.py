import torch
from torch import nn

from spikingjelly.activation_based import functional, layer


class _StepBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = layer.Conv2d(
            3, 8, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"
        )
        self.bn = layer.BatchNorm2d(8, step_mode="m")
        self.tail = layer.Conv2d(
            8, 8, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.tail(x)
        return x


def test_fuse_conv_bn_eval_modules_matches_step_block():
    torch.manual_seed(0)
    model = _StepBlock().eval()
    x = torch.randn(4, 2, 3, 16, 16)

    with torch.no_grad():
        y_ref = model(x)
        fused = functional.fuse_conv_bn_eval_modules(model)
        y_fused = fused(x)

    torch.testing.assert_close(y_fused, y_ref, atol=1e-5, rtol=1e-4)
    assert not any(isinstance(m, layer.BatchNorm2d) for m in fused.modules())
