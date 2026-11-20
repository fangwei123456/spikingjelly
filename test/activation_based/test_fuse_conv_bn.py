import copy

import pytest
import torch
from torch import nn

from spikingjelly.activation_based import layer
from spikingjelly.activation_based.functional.conv_bn_fusion import (
    _TrainConvBnWrapper,
    fuse_conv_bn_eval_modules,
    pack_conv_bn_train_modules,
)


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


class _SharedEvalConvBnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        a = self.bn(self.conv(x))
        b = self.bn(self.conv(x))
        return a + b


class _SharedTrainConvBnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        a = self.bn(self.conv(x))
        b = self.bn(self.conv(x))
        return a + b


class _SharedConvDifferentBnEvalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)

    def forward(self, x):
        return self.bn1(self.conv(x)) + self.bn2(self.conv(x))


class _SharedConvDifferentBnTrainBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)

    def forward(self, x):
        return self.bn1(self.conv(x)) + self.bn2(self.conv(x))


class _PartialConvBnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        return self.bn(self.conv(x)) + self.conv(x)


def test_fuse_conv_bn_eval_modules_matches_step_block():
    torch.manual_seed(0)
    model = _StepBlock().eval()
    x = torch.randn(4, 2, 3, 16, 16)

    with torch.no_grad():
        y_ref = model(x)
        fused = fuse_conv_bn_eval_modules(copy.deepcopy(model))
        y_fused = fused(x)

    torch.testing.assert_close(y_fused, y_ref, atol=1e-5, rtol=1e-4)
    assert not any(isinstance(m, layer.BatchNorm2d) for m in fused.modules())


def test_fuse_conv_bn_eval_modules_treats_step_modules_as_leaves():
    torch.manual_seed(0)
    model = nn.Sequential(
        layer.Conv2d(3, 8, kernel_size=3, padding=1, bias=False, step_mode="m"),
        layer.BatchNorm2d(8, step_mode="m"),
        layer.MaxPool2d(2, step_mode="m"),
    ).eval()
    x = torch.randn(4, 2, 3, 16, 16)

    with torch.no_grad():
        y_ref = model(x)
        fused = fuse_conv_bn_eval_modules(copy.deepcopy(model))
        y_fused = fused(x)

    torch.testing.assert_close(y_fused, y_ref, atol=1e-5, rtol=1e-4)
    assert not any(isinstance(m, layer.BatchNorm2d) for m in fused.modules())


def test_fuse_conv_bn_eval_modules_matches_native_block():
    torch.manual_seed(0)
    model = _NativeStepBlock().eval()
    x = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        y_ref = model(x)
        fused = fuse_conv_bn_eval_modules(copy.deepcopy(model))
        y_fused = fused(x)

    torch.testing.assert_close(y_fused, y_ref, atol=1e-5, rtol=1e-4)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in fused.modules())


def test_fuse_conv_bn_eval_modules_rejects_missing_running_stats():
    model = _NoRunningStatsBlock().eval()

    with pytest.raises(ValueError, match="track running stats"):
        fuse_conv_bn_eval_modules(copy.deepcopy(model))


def test_fuse_conv_bn_eval_modules_handles_shared_conv_bn_pairs():
    torch.manual_seed(0)
    model = _SharedEvalConvBnBlock().eval()
    x = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        y_ref = model(x)
        fused = fuse_conv_bn_eval_modules(copy.deepcopy(model))
        y_fused = fused(x)

    torch.testing.assert_close(y_fused, y_ref, atol=1e-5, rtol=1e-4)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in fused.modules())


def test_fuse_conv_bn_eval_modules_skips_shared_conv_with_different_bn():
    torch.manual_seed(0)
    model = _SharedConvDifferentBnEvalBlock().eval()
    x = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        y_ref = model(x)
        fused = fuse_conv_bn_eval_modules(copy.deepcopy(model))
        y_fused = fused(x)

    torch.testing.assert_close(y_fused, y_ref, atol=1e-5, rtol=1e-4)
    assert sum(isinstance(m, nn.BatchNorm2d) for m in fused.modules()) == 2


def test_fuse_conv_bn_eval_modules_skips_partial_conv_bn_callsites():
    torch.manual_seed(0)
    model = _PartialConvBnBlock().eval()
    x = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        y_ref = model(x)
        fused = fuse_conv_bn_eval_modules(copy.deepcopy(model))
        y_fused = fused(x)

    torch.testing.assert_close(y_fused, y_ref, atol=1e-5, rtol=1e-4)
    assert any(isinstance(m, nn.BatchNorm2d) for m in fused.modules())


def test_pack_conv_bn_train_modules_matches_step_block():
    torch.manual_seed(0)
    model = _StepBlock().train()
    packed = pack_conv_bn_train_modules(copy.deepcopy(model))
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
    packed = pack_conv_bn_train_modules(copy.deepcopy(model))
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
    wrapped = _TrainConvBnWrapper(
        copy.deepcopy(model.conv), copy.deepcopy(model.bn)
    ).train()
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
    wrapped = _TrainConvBnWrapper(
        copy.deepcopy(model.conv), copy.deepcopy(model.bn)
    ).train()
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


def test_pack_conv_bn_train_modules_handles_shared_conv_bn_pairs():
    torch.manual_seed(0)
    model = _SharedTrainConvBnBlock().train()
    packed = pack_conv_bn_train_modules(copy.deepcopy(model))
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


def test_pack_conv_bn_train_modules_skips_shared_conv_with_different_bn():
    torch.manual_seed(0)
    model = _SharedConvDifferentBnTrainBlock().train()
    packed = pack_conv_bn_train_modules(copy.deepcopy(model))
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
    assert not any(isinstance(m, _TrainConvBnWrapper) for m in packed.modules())


def test_pack_conv_bn_train_modules_skips_partial_conv_bn_callsites():
    torch.manual_seed(0)
    model = _PartialConvBnBlock().train()
    packed = pack_conv_bn_train_modules(copy.deepcopy(model))
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
    assert not any(isinstance(m, _TrainConvBnWrapper) for m in packed.modules())


@pytest.mark.parametrize(
    "step_mode, native_bn", [("s", False), ("m", False), ("s", True)]
)
def test_eval_fusion_preserves_weight_standardized_conv(step_mode, native_bn):
    torch.manual_seed(0)
    conv = layer.WSConv2d(2, 3, 3, padding=1, step_mode=step_mode)
    bn = nn.BatchNorm2d(3) if native_bn else layer.BatchNorm2d(3, step_mode=step_mode)
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([0.5, 2.0, 3.0]))
        bn.bias.copy_(torch.tensor([-1.0, 0.5, 1.0]))
        bn.running_mean.copy_(torch.tensor([0.25, -0.5, 1.0]))
        bn.running_var.copy_(torch.tensor([0.5, 1.5, 2.0]))
    model = nn.Sequential(conv, bn).eval()
    fused = fuse_conv_bn_eval_modules(copy.deepcopy(model))
    shape = (2, 2, 8, 8) if step_mode == "s" else (4, 2, 2, 8, 8)
    x = torch.randn(shape)
    torch.testing.assert_close(fused(x), model(x))
    assert any(isinstance(m, type(bn)) for m in fused.modules())


class _ShiftedBatchNorm(layer.BatchNorm2d):
    def forward(self, x):
        return super().forward(x) + 2.0


def test_eval_fusion_preserves_custom_batchnorm_forward():
    model = nn.Sequential(layer.Conv2d(2, 3, 3), _ShiftedBatchNorm(3)).eval()
    x = torch.randn(2, 2, 8, 8)
    fused = fuse_conv_bn_eval_modules(copy.deepcopy(model))
    torch.testing.assert_close(fused(x), model(x))


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("native_conv", [False, True])
def test_eval_fusion_keeps_mixed_plain_layers_supported(dimension, native_conv):
    conv_cls = getattr(nn if native_conv else layer, f"Conv{dimension}d")
    bn_cls = getattr(layer if native_conv else nn, f"BatchNorm{dimension}d")
    model = nn.Sequential(conv_cls(2, 3, 3), bn_cls(3)).eval()
    x = torch.randn(2, 2, *([5] * dimension))
    fused = fuse_conv_bn_eval_modules(copy.deepcopy(model))
    torch.testing.assert_close(fused(x), model(x))
    assert not any(isinstance(m, bn_cls) for m in fused.modules())
