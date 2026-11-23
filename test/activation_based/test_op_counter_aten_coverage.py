from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional, layer, neuron, op_counter
from spikingjelly.activation_based.model.masnn import MASNN, AttMSResNet
from spikingjelly.activation_based.model.maxformer import MaxFormer
from spikingjelly.activation_based.model.ms_resnet import MSResNet
from spikingjelly.activation_based.model.parametric_lif_net import CIFAR10Net
from spikingjelly.activation_based.model.qkformer import QKFormer
from spikingjelly.activation_based.model.sew_resnet import BasicBlock as SEWBlock
from spikingjelly.activation_based.model.sew_resnet import SEWResNet
from spikingjelly.activation_based.model.snas_net import SNASNet
from spikingjelly.activation_based.model.spike_dhs import SpikeDHS
from spikingjelly.activation_based.model.spike_driven_transformer import (
    SpikeDrivenTransformer,
)
from spikingjelly.activation_based.model.spikformer import Spikformer
from spikingjelly.activation_based.model.spiking_resnet import BasicBlock, SpikingResNet
from spikingjelly.activation_based.model.spiking_vgg import SpikingVGG
from spikingjelly.activation_based.model.spiking_vggws_ottt import OTTTSpikingVGG

T = 2


def _multi_step(net):
    functional.set_step_mode(net, "m")
    return net


def _spiking_vgg_bn():
    net = SpikingVGG(
        cfg=[8, "M", 512, "M"],
        batch_norm=True,
        num_classes=5,
        init_weights=False,
        spiking_neuron=neuron.IFNode,
    )
    # Keep the classifier structure (Linear, IF, Dropout) but shrink its width so
    # that the test does not allocate the 4096-wide VGG head.
    net.classifier = nn.Sequential(
        layer.Linear(512 * 7 * 7, 8),
        neuron.IFNode(),
        layer.Dropout(),
        layer.Linear(8, 5),
    )
    return _multi_step(net)


def _snas_net(celltype):
    args = SimpleNamespace(
        dataset="cifar10",
        timestep=T,
        second_avgpooling=2,
        celltype=celltype,
        threshold=1.0,
        tau=2.0,
    )
    # A searched cell that uses every candidate operation: 0 zero, 1 identity,
    # 2 conv 1x1, 3 conv 3x3 and 4 average pooling.
    con_mat = torch.tensor([[0, 3, 2, 4], [0, 0, 1, 3], [0, 0, 0, 0], [0, 0, 0, 0]])
    return SNASNet(args, con_mat)


def _spike_dhs():
    # Feature-network settings of the CIFAR SpikeDHS architecture (576 channels
    # before the classifier, which SpikeDHS hard-codes).
    args = SimpleNamespace(
        fea_step=3, fea_num_layers=8, fea_block_multiplier=3, fea_filter_multiplier=48
    )
    return _multi_step(SpikeDHS(init_channels=3, args=args))


# One small configuration per file of ``spikingjelly.activation_based.model``.
FAMILIES = {
    "spiking_resnet": (
        lambda: _multi_step(
            SpikingResNet(
                BasicBlock, [1, 1, 1, 1], num_classes=5, spiking_neuron=neuron.IFNode
            )
        ),
        (T, 1, 3, 16, 16),
    ),
    "sew_resnet": (
        lambda: _multi_step(
            SEWResNet(
                SEWBlock,
                [1, 1, 1, 1],
                num_classes=5,
                cnf="ADD",
                spiking_neuron=neuron.IFNode,
            )
        ),
        (T, 1, 3, 16, 16),
    ),
    "spiking_vgg": (_spiking_vgg_bn, (T, 1, 3, 16, 16)),
    "spiking_vggws_ottt": (
        lambda: OTTTSpikingVGG(
            cfg=[8, "M", 512], num_classes=5, spiking_neuron=neuron.OTTTLIFNode
        ),
        (1, 3, 8, 8),
    ),
    "parametric_lif_net": (
        lambda: _multi_step(CIFAR10Net(channels=8, spiking_neuron=neuron.LIFNode)),
        (T, 1, 3, 32, 32),
    ),
    "ms_resnet": (
        partial(
            MSResNet,
            T=T,
            num_classes=5,
            layers=(1, 1, 1, 1),
            base_channels=8,
            stem_kernel_size=3,
            stem_stride=1,
            stem_pool=False,
        ),
        (1, 3, 16, 16),
    ),
    "masnn": (
        partial(
            MASNN,
            T=4,
            in_channels=2,
            num_classes=5,
            input_size=(16, 16),
            channels=(8, 16, 16),
            pools=(1, 2, 2),
            fc_hidden=12,
            reduction_t=2,
            reduction_c=4,
        ),
        (1, 2, 16, 16),
    ),
    "masnn_att_ms_resnet": (
        partial(
            AttMSResNet,
            T=T,
            num_classes=5,
            layers=(1, 1, 1, 1),
            base_channels=8,
            stem_kernel_size=3,
            stem_stride=1,
            stem_pool=False,
            reduction_c=4,
        ),
        (1, 3, 16, 16),
    ),
    "maxformer": (
        partial(MaxFormer, T=T, num_classes=5, embed_dims=32, depths=(1, 1, 1)),
        (1, 3, 32, 32),
    ),
    "qkformer": (
        partial(
            QKFormer,
            T=T,
            num_classes=5,
            embed_dims=32,
            num_heads=(1, 2, 4),
            depths=(1, 1, 1),
        ),
        (1, 3, 32, 32),
    ),
    "spike_driven_transformer": (
        partial(
            SpikeDrivenTransformer,
            T=T,
            num_classes=5,
            embed_dims=32,
            num_heads=4,
            depths=1,
            pooling_stat="1010",
        ),
        (1, 3, 32, 32),
    ),
    "snas_net_forward_cell": (partial(_snas_net, "forward"), (1, 3, 32, 32)),
    "snas_net_backward_cell": (partial(_snas_net, "backward"), (1, 3, 32, 32)),
    "spike_dhs": (_spike_dhs, (1, 3, 32, 32)),
    "spikformer": (
        partial(
            Spikformer,
            T=T,
            num_classes=5,
            img_size_h=32,
            img_size_w=32,
            patch_size=4,
            embed_dims=32,
            num_heads=4,
            depths=1,
        ),
        (1, 3, 32, 32),
    ),
}


@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize(
    "counter_cls", [op_counter.FlopCounter, op_counter.MemoryAccessCounter]
)
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_model_families_are_covered_in_strict_mode(family, counter_cls, train):
    torch.manual_seed(0)
    make_net, input_shape = FAMILIES[family]
    net = make_net().train(train)
    # Created outside the counter so that ``aten.rand`` is not counted.
    x = torch.rand(input_shape)
    counter = counter_cls()

    with op_counter.DispatchCounterMode([counter], strict=True):
        if train:
            out = net(x)
            # SpikeDHS also returns the auxiliary-head logits (None here).
            out = out[0] if isinstance(out, tuple) else out
            out.sum().backward()
        else:
            with torch.no_grad():
                net(x)

    assert counter.get_total() > 0


def _count(counter_cls, fn, *args):
    counter = counter_cls()
    with op_counter.DispatchCounterMode([counter], strict=True):
        out = fn(*args)
    return counter.get_total(), out


def _count_flops_and_memory(fn, *args):
    flops = op_counter.FlopCounter()
    memory = op_counter.MemoryAccessCounter()
    with op_counter.DispatchCounterMode([flops, memory], strict=True):
        out = fn(*args)
    return flops.get_total(), memory.get_total(), out


def _bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is unavailable"
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("train,p", [(False, 0.5), (None, 0.5), (True, 1.0)])
def test_native_dropout_forward_counts(device, train, p):
    x = torch.ones(4, device=device)
    dropout = torch.ops.aten.native_dropout.default
    zero_fill = device == "cuda" and p == 1
    flops, memory, out = _count_flops_and_memory(dropout, x, p, train)
    assert flops == (0 if train is False or zero_fill else 8)
    assert memory == _bytes(*out) + (0 if zero_fill else _bytes(x))


@pytest.mark.parametrize("device", DEVICES)
def test_native_dropout_backward_counts(device):
    grad = torch.ones(4, device=device)
    mask = torch.tensor([True, False, True, False], device=device)
    backward = torch.ops.aten.native_dropout_backward.default
    flops, memory, out = _count_flops_and_memory(backward, grad, mask, 2.0)
    assert flops == 8
    assert memory == _bytes(grad, mask, out)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "train,p,expected_flops",
    [
        (False, 0.5, 0),
        (True, 0.0, 0),
        (True, 0.5, 8),
        (True, 1.0, 4),
    ],
)
def test_dropout_in_inference_mode(device, inplace, train, p, expected_flops):
    x = torch.ones(4, device=device)
    dropout = nn.Dropout(p, inplace=inplace).train(train)
    expected_bytes = {
        0: 0,
        4: 40,  # input and output plus the zero scalar write and read
        8: 36 if device == "cuda" and not inplace else 96,
    }
    with torch.inference_mode():
        flops, memory, _ = _count_flops_and_memory(dropout, x)
    assert flops == expected_flops
    assert memory == expected_bytes[expected_flops]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "output_size,scales,cpu_flops,cuda_flops",
    [
        ((2, 2), (), 4, 4),
        ((2, 8), (), 16, 16),
        ((4, 4), (), 16, 0),
        ((8, 8), (0.5, None), 64, 16),
    ],
)
def test_nearest_upsample_backward_flops(
    device, output_size, scales, cpu_flops, cuda_flops
):
    grad = torch.ones(1, 1, *output_size, device=device)
    flops, _ = _count(
        op_counter.FlopCounter,
        torch.ops.aten.upsample_nearest2d_backward.default,
        grad,
        output_size,
        (1, 1, 4, 4),
        *scales,
    )
    assert flops == (cuda_flops if device == "cuda" else cpu_flops)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ndim", [2, 3])
def test_adaptive_max_pool_backward_flops(device, ndim):
    x = torch.zeros(1, 1, *([3] * ndim), device=device)
    x[(0, 0) + (1,) * ndim] = 1
    op = getattr(torch.ops.aten, f"adaptive_max_pool{ndim}d")
    y, indices = op(x, (2,) * ndim)
    grad = torch.ones_like(y)
    backward = getattr(torch.ops.aten, f"adaptive_max_pool{ndim}d_backward")
    flops, grad_input = _count(op_counter.FlopCounter, backward, grad, x, indices)
    assert grad_input[(0, 0) + (1,) * ndim].item() == 2**ndim
    assert flops == 2**ndim


@pytest.mark.parametrize("affine", [False, True])
def test_native_batch_norm_backward_memory(affine):
    aten = torch.ops.aten
    x = torch.randn(2, 3, 4, 4)
    weight = torch.ones(3) if affine else None
    grad = torch.ones_like(x)
    _, mean, invstd = aten.native_batch_norm(
        x, weight, None, None, None, True, 0.1, 1e-5
    )
    memory, gradients = _count(
        op_counter.MemoryAccessCounter,
        aten.native_batch_norm_backward.default,
        grad,
        x,
        weight,
        None,
        None,
        mean,
        invstd,
        True,
        1e-5,
        (True, affine, affine),
    )
    assert memory == (1212 if affine else 1176)
    assert _bytes(*(t for t in gradients if t is not None)) == (408 if affine else 384)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.backends.cudnn.is_available(),
    reason="CUDA with cuDNN is unavailable",
)
def test_cudnn_batch_norm_counts():
    aten = torch.ops.aten
    x = torch.randn(2, 3, 4, 4, device="cuda")
    grad = torch.ones_like(x)
    weight = torch.ones(3, device="cuda")
    bias, mean, var = (
        torch.zeros_like(weight),
        torch.zeros_like(weight),
        torch.ones_like(weight),
    )
    forward_args = (x, weight, bias, mean, var, True, 0.1, 1e-5)
    flops, memory, (_, saved_mean, invstd, reserve) = _count_flops_and_memory(
        aten.cudnn_batch_norm.default,
        *forward_args,
    )
    assert flops == 702
    assert memory == 840 + _bytes(reserve)

    backward_args = (x, grad, weight, mean, var, saved_mean, invstd, 1e-5, reserve)
    flops, memory, _ = _count_flops_and_memory(
        aten.cudnn_batch_norm_backward.default,
        *backward_args,
    )
    assert flops == 960
    assert memory == 1212 + _bytes(reserve)


def test_adaptive_avg_pool2d_matches_avg_pool2d_on_divisible_input():
    x = torch.randn(2, 3, 8, 8)
    for counter_cls in (op_counter.FlopCounter, op_counter.MemoryAccessCounter):
        adaptive, _ = _count(counter_cls, F.adaptive_avg_pool2d, x, (4, 4))
        fixed, _ = _count(counter_cls, F.avg_pool2d, x, 2)
        assert adaptive == fixed

    grad = torch.ones(2, 3, 2, 2)
    x = torch.randn(2, 3, 4, 4)
    aten = torch.ops.aten
    fixed, _ = _count(
        op_counter.FlopCounter,
        aten.avg_pool2d_backward.default,
        grad,
        x,
        (2, 2),
        (2, 2),
        (0, 0),
        False,
        True,
        None,
    )
    adaptive, _ = _count(
        op_counter.FlopCounter,
        aten._adaptive_avg_pool2d_backward.default,
        grad,
        x,
    )
    assert adaptive == fixed == 120


def test_adaptive_pool2d_flops_count_overlapping_windows():
    # Length 5 pooled to 3 gives the windows [0, 2), [1, 4) and [3, 5), i.e. 7
    # input positions per axis and 7 * 7 per channel.
    x = torch.randn(2, 3, 5, 5)
    flops, _ = _count(op_counter.FlopCounter, F.adaptive_avg_pool2d, x, (3, 3))
    assert flops == 2 * 3 * 7 * 7
    # Max pooling needs one comparison fewer than the window size per output.
    flops, _ = _count(op_counter.FlopCounter, F.adaptive_max_pool2d, x, (3, 3))
    assert flops == 2 * 3 * (7 * 7 - 3 * 3)


def test_adaptive_avg_pool2d_is_counted_without_strict():
    x = torch.randn(1, 2, 4, 4)
    counter = op_counter.FlopCounter()
    with op_counter.DispatchCounterMode([counter]):
        layer.AdaptiveAvgPool2d((2, 2))(x)
    assert counter.get_total() == 2 * 4 * 4


def test_views_and_allocations_are_free_in_flops():
    x = torch.randn(2, 3, 4)
    for fn in (
        lambda: x.transpose(0, 1).reshape(3, 8),  # transpose + _unsafe_view
        lambda: x.permute(2, 0, 1),
        lambda: x.split(1, dim=0),
        lambda: x[:, 1:],
        lambda: torch.empty_like(x),
        lambda: torch.zeros_like(x),
        lambda: x.new_zeros(5),
        lambda: torch.cat([x, x]),
        lambda: x.repeat(2, 1, 1),
    ):
        flops, _ = _count(op_counter.FlopCounter, fn)
        assert flops == 0


def test_data_movement_memory_rules():
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)

    memory, out = _count(op_counter.MemoryAccessCounter, torch.cat, [x, y])
    assert memory == _bytes(x, y, out)

    memory, out = _count(op_counter.MemoryAccessCounter, x.repeat, 2, 1, 1)
    assert memory == _bytes(x, out)

    z = torch.empty_like(x)
    memory, _ = _count(op_counter.MemoryAccessCounter, z.copy_, x)
    assert memory == _bytes(x, z)

    memory, (values, indices) = _count(op_counter.MemoryAccessCounter, x.max, 1)
    assert memory == _bytes(x, values, indices)

    memory, out = _count(op_counter.MemoryAccessCounter, torch.zeros_like, x)
    assert memory == _bytes(out)


def test_element_wise_activation_rules():
    x = torch.randn(2, 3, 4)
    for fn in (torch.abs, torch.relu, partial(torch.pow, exponent=2)):
        flops, out = _count(op_counter.FlopCounter, fn, x)
        assert flops == out.numel()
        memory, out = _count(op_counter.MemoryAccessCounter, fn, x)
        assert memory == _bytes(x, out)
