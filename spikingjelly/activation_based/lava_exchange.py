import torch
import torch.nn as nn
import logging
from . import neuron
from typing import Iterable
try:
    import lava.lib.dl.slayer as slayer

except BaseException as e:
    logging.info(f'spikingjelly.activation_based.lava_exchange: {e}')
    slayer = None

# ----------------------------------------
# data reshape function

def TNX_to_NXT(x_seq: torch.Tensor):
    # x_seq.shape = [T, N, *]
    permute_args = list(range(1, x_seq.dim()))
    permute_args.append(0)
    return x_seq.permute(permute_args)

def NXT_to_TNX(x_seq: torch.Tensor):
    # x_seq.shape = [N, *, T]
    permute_args = list(range(x_seq.dim() - 1))
    permute_args.insert(0, x_seq.dim() - 1)
    return x_seq.permute(permute_args)


def lava_neuron_forward(lava_neuron: nn.Module, x_seq: torch.Tensor, v: torch.Tensor or float):
    # x_seq.shape = [T, N, *]
    # lave uses shape = [*, T], while SJ uses shape = [T, *]
    unsqueeze_flag = False
    if x_seq.dim() == 2:
        x_seq = x_seq.unsqueeze(1)
        # lave needs input with shape [N, ... ,T]
        unsqueeze_flag = True

    if isinstance(v, float):
        v_init = v
        v = torch.zeros_like(x_seq[0])
        if v_init != 0.:
            torch.fill_(v, v_init)

    x_seq_shape = x_seq.shape
    x_seq = x_seq.flatten(2).permute(1, 2, 0)
    # [T, N, *] -> [N, *, T]

    lava_neuron.voltage_state = v
    spike = lava_neuron(x_seq).permute(2, 0, 1)

    v = lava_neuron.voltage_state.reshape(x_seq_shape[1:])
    spike = spike.reshape(x_seq_shape)
    if unsqueeze_flag:
        v = v.squeeze(1)
        spike = spike.squeeze(1)

    return spike, v

# ----------------------------------------
# quantize function

class _step_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step):
        return torch.round(x / step) * step

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def step_quantize(x: torch.Tensor, step: float = 1.):
    """
    :param x: the input tensor
    :type x: torch.Tensor
    :param step: the quantize step
    :type step: float
    :return: quantized tensor
    :rtype: torch.Tensor

    The step quantize function. Here is an example:

    .. code-block:: python

        # plt.style.use(['science', 'muted', 'grid'])
        fig = plt.figure(dpi=200, figsize=(6, 4))
        x = torch.arange(-4, 4, 0.001)
        plt.plot(x, lava_exchange.step_quantize(x, 2.), label='quantize(x, step=2)')
        plt.plot(x, x, label='y=x', ls='-.')
        plt.legend()
        plt.grid(ls='--')
        plt.title('step quantize')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.savefig('./docs/source/_static/API/activation_based/lava_exchange/step_quantize.svg')
        plt.savefig('./docs/source/_static/API/activation_based/lava_exchange/step_quantize.pdf')

    .. image:: ./_static/API/activation_based/lava_exchange/step_quantize.*
        :width: 100%

    """
    return _step_quantize.apply(x, step)


def quantize_8bit(x: torch.Tensor, scale, descale=False):
    if descale:
        return step_quantize(x, 2. / scale).clamp(-256. / scale, 255. / scale) * scale
    else:
        return step_quantize(x, 2. / scale).clamp(-256. / scale, 255. / scale)

# ----------------------------------------
# convert function

def check_instance(m, instance):
    if not isinstance(m, instance):
        raise ValueError(
            f'expected {m} with type {instance}, but got {m} with type {type(m)}!')


def check_no_bias(m):
    if m.bias is not None:
        raise ValueError(f'lava does not support for {type(m)} with bias!')

def to_lava_neuron_param_dict(sj_ms_neuron: nn.Module):
    if isinstance(sj_ms_neuron, neuron.IFNode):
        if sj_ms_neuron.v_reset != 0.:
            raise ValueError('lava only supports for v_reset == 0!')
        return {
            'threshold': sj_ms_neuron.v_threshold,
            'current_decay': 1.,
            'voltage_decay': 0.,
            'tau_grad': 1, 'scale_grad': 1, 'scale': sj_ms_neuron.lava_s_cale,
            'norm': None, 'dropout': None,
            'shared_param': True, 'persistent_state': True, 'requires_grad': False,
            'graded_spike': False
        }

    elif isinstance(sj_ms_neuron, neuron.LIFNode):
        if sj_ms_neuron.v_reset != 0.:
            raise ValueError('lava only supports for v_reset == 0!')
        if sj_ms_neuron.decay_input:
            raise ValueError('lava only supports for decay_input == False!')
        return {
            'threshold': sj_ms_neuron.v_threshold,
            'current_decay': 1.,
            'voltage_decay': 1. / sj_ms_neuron.tau,
            'tau_grad': 1, 'scale_grad': 1, 'scale': sj_ms_neuron.lava_s_cale,
            'norm': None, 'dropout': None,
            'shared_param': True, 'persistent_state': True, 'requires_grad': False,
            'graded_spike': False
        }
    else:
        raise NotImplementedError(sj_ms_neuron)


def to_lava_neuron(sj_ms_neuron: nn.Module):
    if isinstance(sj_ms_neuron, (neuron.IFNode, neuron.LIFNode)):
        return slayer.neuron.cuba.Neuron(
            **to_lava_neuron_param_dict(sj_ms_neuron)
        )
    else:
        raise NotImplementedError(sj_ms_neuron)

def linear_to_lava_synapse_dense(fc: nn.Linear):
    """
    :param fc: a pytorch linear layer without bias
    :type fc: nn.Linear
    :return: a lava slayer dense synapse
    :rtype: slayer.synapse.Dense

    Codes example:

    .. code-block:: python

        T = 4
        N = 2
        layer_nn = nn.Linear(8, 4, bias=False)
        layer_sl = lava_exchange.linear_to_lava_synapse_dense(layer_nn)
        x_seq = torch.rand([T, N, 8])
        with torch.no_grad():
            y_nn = functional.seq_to_ann_forward(x_seq, layer_nn)
            y_sl = lava_exchange.NXT_to_TNX(layer_sl(lava_exchange.TNX_to_NXT(x_seq)))
            print('max error:', (y_nn - y_sl).abs().max())
    """
    check_instance(fc, nn.Linear)
    check_no_bias(fc)

    slayer_dense = slayer.synapse.Dense(fc.in_features, fc.out_features)

    # `slayer_dense` is a `torch.torch.nn.Conv3d`. Its weight has shape [out_features, in_features, 1, 1, 1]
    slayer_dense.weight.data[:, :, 0, 0, 0] = fc.weight.data.clone()

    return slayer_dense

def conv2d_to_lava_synapse_conv(conv2d_nn: nn.Conv2d):
    """
    :param conv2d_nn: a pytorch conv2d layer without bias
    :type conv2d_nn: nn.Conv2d
    :return: a lava slayer conv synapse
    :rtype: slayer.synapse.Conv

    Codes example:

    .. code-block:: python

        T = 4
        N = 2
        layer_nn = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        layer_sl = lava_exchange.conv2d_to_lava_synapse_conv(layer_nn)
        x_seq = torch.rand([T, N, 3, 28, 28])
        with torch.no_grad():
            y_nn = functional.seq_to_ann_forward(x_seq, layer_nn)
            y_sl = lava_exchange.NXT_to_TNX(layer_sl(lava_exchange.TNX_to_NXT(x_seq)))
            print('max error:', (y_nn - y_sl).abs().max())
    """
    check_instance(conv2d_nn, nn.Conv2d)
    check_no_bias(conv2d_nn)

    slayer_conv = slayer.synapse.Conv(in_features=conv2d_nn.in_channels, out_features=conv2d_nn.out_channels, kernel_size=conv2d_nn.kernel_size, stride=conv2d_nn.stride, padding=conv2d_nn.padding, dilation=conv2d_nn.dilation, groups=conv2d_nn.groups)
    # `slayer_conv` is a `torch.torch.nn.Conv3d`.
    slayer_conv.weight.data[:, :, :, :, 0] = conv2d_nn.weight.data.clone()

    return slayer_conv

def avgpool2d_to_lava_synapse_pool(pool2d_nn: nn.AvgPool2d):
    """
    :param pool2d_nn: a pytorch AvgPool2d layer
    :type pool2d_nn: nn.AvgPool2d
    :return: a lava slayer pool layer
    :rtype: slayer.synapse.Pool

    .. admonition:: Warning
        :class: warning

        The lava slayer pool layer applies sum pooling, rather than average pooling.

    .. code-block:: python

        T = 4
        N = 2
        layer_nn = nn.AvgPool2d(kernel_size=2, stride=2)
        layer_sl = lava_exchange.avgpool2d_to_lava_synapse_pool(layer_nn)
        x_seq = torch.rand([T, N, 3, 28, 28])
        with torch.no_grad():
            y_nn = functional.seq_to_ann_forward(x_seq, layer_nn)
            y_sl = lava_exchange.NXT_to_TNX(layer_sl(lava_exchange.TNX_to_NXT(x_seq))) / 4.
            print('max error:', (y_nn - y_sl).abs().max())
    """
    check_instance(pool2d_nn, nn.AvgPool2d)
    logging.warning('The lava slayer pool layer applies sum pooling, rather than average pooling. `avgpool2d_to_lava_synapse_pool` will return a sum pooling layer.')

    return slayer.synapse.Pool(pool2d_nn.kernel_size, pool2d_nn.stride, pool2d_nn.padding)

def to_lava_block_dense(fc: nn.Linear, sj_ms_neuron: nn.Module, quantize_to_8bit: bool = True):

    check_instance(fc, nn.Linear)
    check_no_bias(fc)

    neuron_params = to_lava_neuron_param_dict(sj_ms_neuron)
    if isinstance(sj_ms_neuron, (neuron.IFNode, neuron.LIFNode)):
        block_init = slayer.block.cuba.Dense
    else:
        raise NotImplementedError(sj_ms_neuron)


    if quantize_to_8bit:
        # if 'pre_hook_fx' not in kwargs.keys(), then `pre_hook_fx` will be set to `quantize_8bit` by default
        lava_block = block_init(neuron_params, fc.in_features, fc.out_features, delay_shift=False)
    else:
        lava_block = block_init(neuron_params, fc.in_features, fc.out_features, delay_shift=False, pre_hook_fx=None)

    lava_block.synapse.weight.data[:, :, 0, 0, 0] = fc.weight.data.clone()

    return lava_block


def to_lava_block_conv(conv2d_nn: nn.Conv2d, sj_ms_neuron: nn.Module, quantize_to_8bit: bool = True):

    check_instance(conv2d_nn, nn.Conv2d)
    check_no_bias(conv2d_nn)

    neuron_params = to_lava_neuron_param_dict(sj_ms_neuron)
    if isinstance(sj_ms_neuron, (neuron.IFNode, neuron.LIFNode)):
        block_init = slayer.block.cuba.Conv
    else:
        raise NotImplementedError(sj_ms_neuron)

    if quantize_to_8bit:
        # if 'pre_hook_fx' not in kwargs.keys(), then `pre_hook_fx` will be set to `quantize_8bit` by default
        lava_block = block_init(neuron_params, in_features=conv2d_nn.in_channels, out_features=conv2d_nn.out_channels, kernel_size=conv2d_nn.kernel_size, stride=conv2d_nn.stride, padding=conv2d_nn.padding, dilation=conv2d_nn.dilation, groups=conv2d_nn.groups, delay_shift=False)
    else:
        lava_block = block_init(neuron_params, in_features=conv2d_nn.in_channels, out_features=conv2d_nn.out_channels, kernel_size=conv2d_nn.kernel_size, stride=conv2d_nn.stride, padding=conv2d_nn.padding, dilation=conv2d_nn.dilation, groups=conv2d_nn.groups, delay_shift=False, pre_hook_fx=None)

    lava_block.synapse.weight.data[:, :, :, :, 0] = conv2d_nn.weight.data.clone()

    return lava_block

def to_lava_block_pool(pool2d_nn: nn.AvgPool2d, sj_ms_neuron: nn.Module, quantize_to_8bit: bool = True):

    check_instance(pool2d_nn, nn.AvgPool2d)

    neuron_params = to_lava_neuron_param_dict(sj_ms_neuron)
    if isinstance(sj_ms_neuron, (neuron.IFNode, neuron.LIFNode)):
        block_init = slayer.block.cuba.Pool
    else:
        raise NotImplementedError(sj_ms_neuron)

    if quantize_to_8bit:
        # if 'pre_hook_fx' not in kwargs.keys(), then `pre_hook_fx` will be set to `quantize_8bit` by default
        lava_block = block_init(neuron_params, pool2d_nn.kernel_size, pool2d_nn.stride, pool2d_nn.padding, delay_shift=False)
    else:
        lava_block = block_init(neuron_params, pool2d_nn.kernel_size, pool2d_nn.stride, pool2d_nn.padding, delay_shift=False, pre_hook_fx=None)

    logging.warning('The lava slayer pool layer applies sum pooling, rather than average pooling. `avgpool2d_to_lava_synapse_pool` will return a sum pooling layer.')

    return lava_block

def to_lava_block_flatten(flatten_nn: nn.Flatten):
    check_instance(flatten_nn, nn.Flatten)
    if flatten_nn.start_dim != 1:
        raise ValueError('lava only supports for flatten_nn.start_dim == 1!')
    return slayer.block.cuba.Flatten()



def to_lava_blocks(net: Iterable):
    # https://lava-nc.org/lava-lib-dl/netx/netx.html
    '''
    Supported layer types
    input  : {shape, type}
    flatten: {shape, type}
    average: {shape, type}
    concat : {shape, type, layers}
    dense  : {shape, type, neuron, inFeatures, outFeatures, weight, delay(if available)}
    pool   : {shape, type, neuron, kernelSize, stride, padding, dilation, weight}
    conv   : {shape, type, neuron, inChannels, outChannels, kernelSize, stride,
                            |      padding, dilation, groups, weight, delay(if available)}
                            |
                            |-> this is the description of the compartment parameters
                            |-> {iDecay, vDecay, vThMant, refDelay, ... (other additional params)}
    '''
    blocks = []
    length = net.__len__()
    i = 0
    while True:
        if isinstance(net[i], nn.Linear):
            if i + 1 < length and isinstance(net[i + 1], (neuron.IFNode, neuron.LIFNode)):
                blocks.append(to_lava_block_dense(net[i], net[i + 1]))
                i += 2
            else:
                raise ValueError(type(net[i]))

        elif isinstance(net[i], nn.Conv2d):
            if i + 1 < length and isinstance(net[i + 1], (neuron.IFNode, neuron.LIFNode)):
                blocks.append(to_lava_block_conv(net[i], net[i + 1]))
                i += 2
            else:
                raise ValueError(type(net[i]))

        elif isinstance(net[i], nn.AvgPool2d):
            if i + 1 < length and isinstance(net[i + 1], (neuron.IFNode, neuron.LIFNode)):
                blocks.append(to_lava_block_pool(net[i], net[i + 1]))
                i += 2
            else:
                raise ValueError(type(net[i]))


        elif isinstance(net[i], nn.Flatten):
            blocks.append(to_lava_block_flatten(net[i]))
            i += 1

        else:
            raise ValueError(type(net[i]))

        if i == length:
            break

    return blocks
