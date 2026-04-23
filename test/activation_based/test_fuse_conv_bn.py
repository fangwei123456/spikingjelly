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


class _NoRunningStatsBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8, track_running_stats=False)

    def forward(self, x):
        return self.bn(self.conv(x))


def test_fuse_conv_bn_eval_modules_matches_step_block():
    torch.manual_seed(0)
    model = _StepBlock().eval()
    x = torch.randn(4, 2, 3, 16, 16)

    with torch.no_grad():
        y_ref = model(x)
        fused = functional.fuse_conv_bn_eval_modules(copy.deepcopy(model))
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


def test_fuse_conv_bn_eval_modules_rejects_missing_running_stats():
    model = _NoRunningStatsBlock().eval()

    try:
        functional.fuse_conv_bn_eval_modules(copy.deepcopy(model))
    except ValueError as e:
        assert "track running stats" in str(e)
    else:
        raise AssertionError("Expected ValueError for BatchNorm without running stats")


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


def test_train_conv_bn_wrapper_packs_native_multistep_inputs():
    torch.manual_seed(0)
    model = _NativeStepBlock().train()
    wrapped = _TrainConvBnWrapper(copy.deepcopy(model.conv), copy.deepcopy(model.bn)).train()
    x = torch.randn(4, 2, 3, 16, 16, requires_grad=True)
    x_packed = x.detach().clone().requires_grad_(True)

    y_packed = wrapped(x_packed)

    x_ref = x.flatten(0, 1)
    y_ref = model.bn(model.conv(x_ref)).view_as(y_packed)
    torch.testing.assert_close(y_packed, y_ref, atol=1e-5, rtol=1e-4)

    loss_ref = y_ref.square().mean()
    loss_packed = y_packed.square().mean()
    loss_ref.backward()
    loss_packed.backward()

    torch.testing.assert_close(x_packed.grad, x.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        wrapped.conv.weight.grad, model.conv.weight.grad, atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        wrapped.bn.weight.grad, model.bn.weight.grad, atol=1e-5, rtol=1e-4
    )


def test_train_conv_bn_wrapper_keeps_conv_hooks_active():
    torch.manual_seed(0)
    model = _StepBlock().train()
    wrapped = _TrainConvBnWrapper(copy.deepcopy(model.conv), copy.deepcopy(model.bn)).train()
    x = torch.randn(4, 2, 3, 16, 16)
    hook_calls = []

    def hook(module, args, output):
        hook_calls.append((args[0].shape, output.shape))

    handle = wrapped.conv.register_forward_hook(hook)
    try:
        wrapped(x)
    finally:
        handle.remove()

    assert hook_calls == [((8, 3, 16, 16), (8, 8, 16, 16))]
