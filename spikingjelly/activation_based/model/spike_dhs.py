import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import *
from torch.autograd import Function
from torch import Tensor
from collections import namedtuple
from ...activation_based import layer
from ..neuron import LIFNode
from torch.nn.functional import interpolate
from ..surrogate import SurrogateFunctionBase, heaviside
from math import tanh
from torch.jit import script

import numpy as np
import argparse


### Components ###
def network_layer_to_space(net_arch):
    """
    :param net_arch: network level sample rate
        0: down 1: None 2: Up
    :type net_arch: numpy.ndarray
    :return: network level architecture
        network_space[layer][level][sample]:
        layer: 0 - 8
        level: sample_level {0: 1, 1: 2, 2: 4, 3: 8}
        sample: 0: down 1: None 2: Up
    :rtype: numpy.ndarray

    Convert network level sample rate like [0,0,1,1,1,2,2,2] to network architecture.
    """
    for i, layer in enumerate(net_arch):
        if i == 0:
            space = np.zeros((1, 4, 3))
            space[0][layer][0] = 1
            prev = layer
        else:
            if layer == prev + 1:
                sample = 0
            elif layer == prev:
                sample = 1
            elif layer == prev - 1:
                sample = 2
            space1 = np.zeros((1, 4, 3))
            space1[0][layer][sample] = 1
            space = np.concatenate([space, space1], axis=0)
            prev = layer
    return space


Genotype = namedtuple('Genotype_2D', 'cell cell_concat')
PRIMITIVES = [
    'skip_connect',
    'snn_b3',
    'snn_b5'
]

class Identity(nn.Module):
    def __init__(self, C_in, C_out, signal):
        super(Identity, self).__init__()
        self.signal = signal

    def forward(self, x):
        return x

OPS = {
    'skip_connect': lambda Cin, Cout, stride, signal: (
        SpikingConv2d(Cin, Cout, stride=stride, padding=1, spiking=False) if signal == 1 else Identity(Cin, Cout,
                                                                                                       signal)),
    'snn_b3': lambda Cin, Cout, stride, signal: SpikingConv2d(Cin, Cout, kernel_size=3, stride=stride, padding=1,
                                                              spiking=False),
    'snn_b5': lambda Cin, Cout, stride, signal: SpikingConv2d(Cin, Cout, kernel_size=3, stride=stride, padding=1,
                                                              spiking=False)
}





@script
def dSpike_backward(grad_output: Tensor, x: Tensor, alpha: float):
    mask = x.abs() > 0.5
    const = alpha / (2. * tanh(alpha / 2.))
    grad_x = (grad_output * const / (alpha * x).cosh_().square_()
              ).masked_fill_(mask, 0)
    return grad_x, None


class dSpike(Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return dSpike_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class DSpike(SurrogateFunctionBase):
    def __init__(self, alpha: float = 3, spiking=True):
        super().__init__(alpha, spiking)
        assert alpha > 0, 'alpha must be lager than 0.'

    @staticmethod
    def spiking_function(x: Tensor, alpha: float):
        return dSpike.apply(x, alpha)


class save_v_LIFNode(LIFNode):
    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        self.v_before_spike = (self.v - self.v_threshold).mean()
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v_before_spike)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)


def getSpikingNode(v_threshold=0.5):
    return LIFNode(tau=1.25, decay_input=False, v_threshold=v_threshold, detach_reset=True, surrogate_function=DSpike())


def get_save_v_SpikingNode(v_threshold=0.5):
    return save_v_LIFNode(tau=1.25, decay_input=False, v_threshold=v_threshold, detach_reset=True,
                          surrogate_function=DSpike())


class SpikingConv2d(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3, spiking=True, v_threshold=0.5):
        super(SpikingConv2d, self).__init__()
        self.conv = layer.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = layer.BatchNorm2d(output_c)
        self.spiking = spiking
        if self.spiking:
            self.spike = getSpikingNode(v_threshold=v_threshold)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.spiking:
            x = self.spike(x)
        return x


class SearchSpikingConv2d_stem(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3, spiking=True, v_threshold=0.5):
        super(SearchSpikingConv2d_stem, self).__init__()
        self.conv_m = layer.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_b = layer.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_s = layer.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn_m = layer.BatchNorm2d(output_c)
        self.bn_b = layer.BatchNorm2d(output_c)
        self.bn_s = layer.BatchNorm2d(output_c)

        self.spike_m = get_save_v_SpikingNode()
        self.spike_b = get_save_v_SpikingNode()
        self.spike_s = get_save_v_SpikingNode()

        self.is_DGS = False

        self.dgs_alpha = nn.Parameter(1e-3 * torch.ones(3).cuda(), requires_grad=True)
        self.dgs_step = 0.2

    def dgs_init_stage(self):
        self.is_DGS = True
        self.conv_s.load_state_dict(self.conv_m.state_dict())
        self.conv_b.load_state_dict(self.conv_m.state_dict())

        self.bn_s.load_state_dict(self.bn_m.state_dict())
        self.bn_b.load_state_dict(self.bn_m.state_dict())

        self.spike_s.surrogate_function.alpha = self.spike_m.surrogate_function.alpha - self.dgs_step
        self.spike_b.surrogate_function.alpha = self.spike_m.surrogate_function.alpha + self.dgs_step

        self.dgs_alpha = nn.Parameter(1e-3 * torch.ones(3).cuda(), requires_grad=True)

        for name, value in self.named_parameters():
            value.requires_grad_(True)

    def dgs_finish_stage(self, dgs_direction):
        self.is_DGS = False
        value_list = [-self.dgs_step, 0, self.dgs_step]
        value = value_list[dgs_direction]
        self.spike_m.surrogate_function.alpha += value
        if self.spike_m.surrogate_function.alpha < 0.2:
            self.spike_m.surrogate_function.alpha = 0.2

    def forward(self, x):
        if self.is_DGS:
            n_a = F.softmax(self.dgs_alpha, dim=0)
            x = n_a[0] * self.spike_s(self.bn_s(self.conv_s(x))) + \
                n_a[1] * self.spike_m(self.bn_m(self.conv_m(x))) + \
                n_a[2] * self.spike_b(self.bn_b(self.conv_b(x)))
        else:
            x = self.spike_m(self.bn_m(self.conv_m(x)))
        return x


class SearchSpikingConv2d_cell(nn.Module):
    def __init__(self, io_c, kernel_size=3, stride=1, padding=1, b=3, spiking=True, v_threshold=0.5):
        super(SearchSpikingConv2d_cell, self).__init__()
        [input_c1, output_c1, primitive1], [input_c2, output_c2, primitive2] = io_c

        self.conv1_m = layer.Conv2d(input_c1, output_c1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1_b = layer.Conv2d(input_c1, output_c1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1_s = layer.Conv2d(input_c1, output_c1, kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn1_m = layer.BatchNorm2d(output_c1)
        self.bn1_b = layer.BatchNorm2d(output_c1)
        self.bn1_s = layer.BatchNorm2d(output_c1)

        self.conv2_m = layer.Conv2d(input_c2, output_c2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_b = layer.Conv2d(input_c2, output_c2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_s = layer.Conv2d(input_c2, output_c2, kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn2_m = layer.BatchNorm2d(output_c2)
        self.bn2_b = layer.BatchNorm2d(output_c2)
        self.bn2_s = layer.BatchNorm2d(output_c2)

        self.spike_m = get_save_v_SpikingNode()
        self.spike_b = get_save_v_SpikingNode()
        self.spike_s = get_save_v_SpikingNode()

        self.is_DGS = False

        self.dgs_alpha = nn.Parameter(1e-3 * torch.ones(3).cuda(), requires_grad=True)
        self.dgs_step = 0.2

    def dgs_init_stage(self):
        self.is_DGS = True
        self.conv1_s.load_state_dict(self.conv1_m.state_dict())
        self.conv1_b.load_state_dict(self.conv1_m.state_dict())

        self.bn1_s.load_state_dict(self.bn1_m.state_dict())
        self.bn1_b.load_state_dict(self.bn1_m.state_dict())

        self.conv2_s.load_state_dict(self.conv2_m.state_dict())
        self.conv2_b.load_state_dict(self.conv2_m.state_dict())

        self.bn2_s.load_state_dict(self.bn2_m.state_dict())
        self.bn2_b.load_state_dict(self.bn2_m.state_dict())

        self.spike_s.surrogate_function.alpha = self.spike_m.surrogate_function.alpha - self.dgs_step
        self.spike_b.surrogate_function.alpha = self.spike_m.surrogate_function.alpha + self.dgs_step

        self.dgs_alpha = nn.Parameter(1e-3 * torch.ones(3).cuda(), requires_grad=True)

        for name, value in self.named_parameters():
            value.requires_grad_(True)

    def dgs_finish_stage(self, dgs_direction):
        self.is_DGS = False
        value_list = [-self.dgs_step, 0, self.dgs_step]
        value = value_list[dgs_direction]
        self.spike_m.surrogate_function.alpha += value
        if self.spike_m.surrogate_function.alpha < 0.2:
            self.spike_m.surrogate_function.alpha = 0.2

    def forward(self, x1, x2):
        if self.is_DGS:
            n_a = F.softmax(self.dgs_alpha, dim=0)
            x = n_a[0] * self.spike_s(self.bn1_s(self.conv1_s(x1)) + self.bn2_s(self.conv2_s(x2))) + \
                n_a[1] * self.spike_m(self.bn1_m(self.conv1_m(x1)) + self.bn2_m(self.conv2_m(x2))) + \
                n_a[2] * self.spike_b(self.bn1_b(self.conv1_b(x1)) + self.bn2_b(self.conv2_b(x2)))
        else:
            x = self.spike_m(self.bn1_m(self.conv1_m(x1)) + self.bn2_m(self.conv2_m(x2)))
        return x


class SpikingLinear(nn.Module):
    def __init__(self, input_c, output_c, spiking=True):
        super(SpikingLinear, self).__init__()
        self.linear = layer.Linear(input_c, output_c)
        self.bn = layer.SeqToANNContainer(nn.BatchNorm1d(output_c))
        self.spiking = spiking
        if self.spiking:
            self.spike = getSpikingNode()

    def forward(self, x):
        x = self.bn(self.linear(x))
        if self.spiking:
            x = self.spike(x)
        return x


class SpikingAvgPool2d(nn.Module):
    def __init__(self, kernel_size=5, stride=3, padding=0, b=3, spiking=True):
        super(SpikingAvgPool2d, self).__init__()
        self.pooling = layer.SeqToANNContainer(
            nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, count_include_pad=False))
        self.spike = getSpikingNode()

    def forward(self, x):
        return self.spike(self.pooling(x))


class SpikingAdaptiveAvgPool2d(nn.Module):
    def __init__(self, dimension, b=3, spiking=True):
        super(SpikingAdaptiveAvgPool2d, self).__init__()
        self.pooling = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d(dimension))
        self.spike = getSpikingNode()

    def forward(self, x):
        return self.spike(self.pooling(x))


class Nearest(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self._shape = shape

    def forward(self, x):
        return interpolate(x, self._shape, mode="nearest")


### Model ###

class Cell(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch, network_arch,
                 filter_multiplier, downup_sample, args=None):
        """
        :param steps: number of nodes
        :type steps: int
        :param block_multiplier: The change factor for the channel for current node
        :type block_multiplier: int
        :param prev_prev_fmultiplier: The change factor for the channel for previous previous node
        :type prev_prev_fmultiplier: int
        :param prev_filter_multiplier: The change factor for the channel for previous node
        :type prev_filter_multiplier: int
        :param cell_arch: cell level architecture
        :type cell_arch: numpy.ndarray
        :param network_arch: layer level architecture
        :type network_arch: numpy.ndarray
        :param filter_multiplier: filter channel multiplier
        :type filter_multiplier: int
        :param downup_sample: sample rate, -1:downsample, 1:upsample, 0: no change
        :type downup_sample: int
        :param args: additional arguments

        A cell is defined as a repeated and searchable unit, which is a directed acyclic graph with N nodes.
        """
        super(Cell, self).__init__()

        self.cell_arch = cell_arch
        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2

        self.cell_arch = torch.sort(self.cell_arch, dim=0)[0].to(torch.uint8)

        C_out = self.C_out
        ops_channel = []
        for i, x in enumerate(self.cell_arch):
            primitive = PRIMITIVES[x[1]]

            if x[0] in [0, 2, 5]:
                C_in = self.C_prev_prev
            elif x[0] in [1, 3, 6]:
                C_in = self.C_prev
            else:
                C_in = self.C_out

            ops_channel.append([C_in, C_out, primitive])
            if i % 2 == 1:
                op = SearchSpikingConv2d_cell(io_c=ops_channel)
                self._ops.append(op)
                ops_channel = []

        self.spikes = nn.ModuleList([getSpikingNode()
                                     for _ in range(self._steps)])

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[3], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[4], self.scale)
            interpolate = layer.SeqToANNContainer(Nearest([feature_size_h, feature_size_w]))
            s1 = interpolate(s1)
        if (s0.shape[3] != s1.shape[3]) or (s0.shape[4] != s1.shape[4]):
            interpolate = layer.SeqToANNContainer(Nearest([s1.shape[3], s1.shape[4]]))

            s0 = interpolate(s0)
        device = prev_input.device

        states = [s0, s1]

        # Cell structure is fixed, it can be change.
        spike = self._ops[0](states[0], states[1])
        states.append(spike)

        spike = self._ops[1](states[0], states[1])
        states.append(spike)

        spike = self._ops[2](states[2], states[3])
        states.append(spike)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=2)
        return prev_input, concat_feature


class newFeature(nn.Module):
    def __init__(self, frame_rate, network_arch, cell_arch, cell=Cell, args=None):
        """
        :param frame_rate: input channel
        :type frame_rate: int
        :param network_arch: layer level architecture
        :type network_arch: numpy.ndarray
        :param cell_arch: cell level architecture
        :type cell_arch: numpy.ndarray
        :param cell: choice the type of cell, defaults to Cell
        :type cell: Cell class
        :param args: additional arguments

        newFeature is used to extract feature.
        """
        super(newFeature, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self._step = args.fea_step
        self._num_layers = args.fea_num_layers
        self._block_multiplier = args.fea_block_multiplier
        self._filter_multiplier = args.fea_filter_multiplier

        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)
        self._num_end = self._filter_multiplier * self._block_multiplier
        self.stem0 = SearchSpikingConv2d_stem(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1,
                                              padding=1, b=3)
        self.auxiliary_head = AuxiliaryHeadCIFAR(576, 100)

        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}

        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()

            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             self._filter_multiplier,
                             self.cell_arch, self.network_arch[i],
                             int(self._filter_multiplier * filter_param_dict[level]),
                             downup_sample, self.args)

            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 self._filter_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]),
                                 downup_sample, self.args)

                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_prev_level]),
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]), downup_sample, self.args)

            self.cells += [_cell]

    def forward(self, x):
        stem0 = self.stem0(x)
        stem1 = stem0
        out = (stem0, stem1)

        for i in range(self._num_layers):
            out = self.cells[i](out[0], out[1])
            '''
            cell torch.Size([50, 144, 32, 32])
            cell torch.Size([50, 144, 32, 32])
            cell torch.Size([50, 288, 16, 16])
            cell torch.Size([50, 288, 16, 16])
            cell torch.Size([50, 288, 16, 16])
            cell torch.Size([50, 576, 8, 8] -> auxiliary [50, 10]
            cell torch.Size([50, 576, 8, 8])
            cell torch.Size([50, 576, 8, 8])
            '''
            if i == 2 * 8 // 3:
                if self.training:
                    logits_aux = self.auxiliary_head(out[-1])

        last_output = out[-1]

        if self.training:
            return last_output, logits_aux
        else:
            return last_output, None

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.pooling = SpikingAvgPool2d(kernel_size=5, stride=3, padding=0)
        self.conv1 = SpikingConv2d(C, 128, 1, padding=0, b=3)
        self.conv2 = SpikingConv2d(128, 768, 2, padding=0, b=3)
        self.classifier = SpikingLinear(768, num_classes, spiking=False)

    def forward(self, x):
        x = self.pooling(x)
        spike1 = self.conv1(x)
        spike2 = self.conv2(spike1)
        shape = spike2.shape[:2]
        result = self.classifier(spike2.view(*shape, -1))
        return result


class SpikeDHS(nn.Module):
    def __init__(self, init_channels=3, args=None):
        """
        :param init_channels: channel size, defaults to 3
        :type init_channels: int
        :param args: additional arguments

        The SpikeDHS `Auto-Spikformer: Spikformer Architecture Search <https://arxiv.org/abs/2306.00807>`_ implementation by Spikingjelly.

        """
        super(SpikeDHS, self).__init__()
        p = 0.0
        network_path_fea = [0, 0, 1, 1, 1, 2, 2, 2]
        network_path_fea = np.array(network_path_fea)
        network_arch_fea = network_layer_to_space(network_path_fea)

        cell_arch_fea = [[1, 1],
                         [0, 1],
                         [3, 2],
                         [2, 1],
                         [7, 1],
                         [8, 1]]
        cell_arch_fea = np.array(cell_arch_fea)

        self.feature = newFeature(init_channels, network_arch_fea, cell_arch_fea, args=args)
        self.global_pooling = SpikingAdaptiveAvgPool2d(1)
        self.classifier = SpikingLinear(576, 100, spiking=False)
        self._time_step = 6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters()]

    def forward(self, input):
        input = input.expand(self._time_step, -1, -1, -1, -1)
        shape = input.shape[:2]

        feature_out, logits_aux = self.feature(input)
        pooling_out = self.global_pooling(feature_out)
        shape = pooling_out.shape[:2]
        logits_buf = self.classifier(pooling_out.view(*shape, -1))

        logits = logits_buf.mean(0)
        if logits_aux is not None:
            logits_aux = logits_aux.mean(0)

        if self.training:
            return logits, logits_aux
        else:
            return logits, None

    def dgs_freeze_weights(self):
        for name, value in self.named_parameters():
            value.requires_grad_(False)

    def dgs_unfreeze_weights(self):
        for name, value in self.named_parameters():
            value.requires_grad_(True)


if __name__ == "__main__":
    ### Example ###

    parser = argparse.ArgumentParser("cifar")

    parser.add_argument('--layers', type=int, default=8, help='total number of layers')

    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
    parser.add_argument('--fea_num_layers', type=int, default=8)
    parser.add_argument('--fea_filter_multiplier', type=int, default=48)
    parser.add_argument('--fea_block_multiplier', type=int, default=3)
    parser.add_argument('--fea_step', type=int, default=3)
    parser.add_argument('--net_arch_fea', default=None, type=str)
    parser.add_argument('--cell_arch_fea', default=None, type=str)
    args = parser.parse_args()
    spikedhs = SpikeDHS(init_channels=3, args=args)