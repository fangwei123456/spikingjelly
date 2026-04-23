import copy

import torch
from torch import nn

from spikingjelly.activation_based import functional, layer
from spikingjelly.activation_based.functional.misc import _TrainConvBnWrapper


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


class _NativeStepBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.tail = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)

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


def test_fuse_conv_bn_eval_modules_matches_native_block():
    torch.manual_seed(0)
    model = _NativeStepBlock().eval()
    x = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        y_ref = model(x)
        fused = functional.fuse_conv_bn_eval_modules(copy.deepcopy(model))
        y_fused = fused(x)

    torch.testing.assert_close(y_fused, y_ref, atol=1e-5, rtol=1e-4)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in fused.modules())


def test_pack_conv_bn_train_modules_matches_step_block():
    torch.manual_seed(0)
    model = _StepBlock().train()
    packed = functional.pack_conv_bn_train_modules(copy.deepcopy(model))
    x = torch.randn(4, 2, 3, 16, 16, requires_grad=True)
    x_packed = x.detach().clone().requires_grad_(True)

    y_ref = model(x)
    y_packed = packed(x_packed)
    torch.testing.assert_close(y_packed, y_ref, atol=1e-5, rtol=1e-4)

    loss_ref = y_ref.square().mean()
    loss_packed = y_packed.square().mean()
    loss_ref.backward()
    loss_packed.backward()

    torch.testing.assert_close(x_packed.grad, x.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        packed.conv.conv.weight.grad, model.conv.weight.grad, atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        packed.conv.bn.weight.grad, model.bn.weight.grad, atol=1e-5, rtol=1e-4
    )
    assert any(isinstance(m, _TrainConvBnWrapper) for m in packed.modules())


def test_pack_conv_bn_train_modules_matches_native_block():
    torch.manual_seed(0)
    model = _NativeStepBlock().train()
    packed = functional.pack_conv_bn_train_modules(copy.deepcopy(model))
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    x_packed = x.detach().clone().requires_grad_(True)

    y_ref = model(x)
    y_packed = packed(x_packed)
    torch.testing.assert_close(y_packed, y_ref, atol=1e-5, rtol=1e-4)

    loss_ref = y_ref.square().mean()
    loss_packed = y_packed.square().mean()
    loss_ref.backward()
    loss_packed.backward()

    torch.testing.assert_close(x_packed.grad, x.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        packed.conv.conv.weight.grad, model.conv.weight.grad, atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        packed.conv.bn.weight.grad, model.bn.weight.grad, atol=1e-5, rtol=1e-4
    )
    assert any(isinstance(m, _TrainConvBnWrapper) for m in packed.modules())
