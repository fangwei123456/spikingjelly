from functools import partial

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


# One small configuration per file of ``spikingjelly.activation_based.model``,
# except the NAS models (``snas_net``, ``spike_dhs``), which need search outputs.
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
            net(x).sum().backward()
        else:
            with torch.no_grad():
                net(x)

    assert counter.get_total() > 0


def _count(counter_cls, fn, *args):
    counter = counter_cls()
    with op_counter.DispatchCounterMode([counter], strict=True):
        out = fn(*args)
    return counter.get_total(), out


def _bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def test_adaptive_avg_pool2d_matches_avg_pool2d_on_divisible_input():
    x = torch.randn(2, 3, 8, 8)
    for counter_cls in (op_counter.FlopCounter, op_counter.MemoryAccessCounter):
        adaptive, _ = _count(counter_cls, F.adaptive_avg_pool2d, x, (4, 4))
        fixed, _ = _count(counter_cls, F.avg_pool2d, x, 2)
        assert adaptive == fixed


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
