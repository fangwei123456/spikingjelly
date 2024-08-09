import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Union, Callable, Optional
from . import neuron, base, surrogate

_hw_bits = 12


@torch.jit.script
def step_quantize_forward(x: torch.Tensor, step: float):
    x = x / step
    torch.round_(x)
    return x * step


class step_quantize_atgf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, step: float = 1.):
        return step_quantize_forward(x, step)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def step_quantize(x: torch.Tensor, step: float = 1.):
    """
    :param x: a float tensor whose range is ``0 <= x <= 1``.
    :type x: torch.Tensor

    :param step: the quantization step
    :type step: float

    :return: ``y = round(x / step) * step``
    :rtype: torch.Tensor

    The step quantizer defined in `Lava`.

    Denote ``k`` as an ``int``, ``x[i]`` will be quantized to the nearest ``k * step``.
    """
    return step_quantize_atgf.apply(x, step)


def quantize_8b(x, scale, descale=False):
    """
    Denote ``k`` as an ``int``, ``x[i]`` will be quantized to the nearest ``2 * k / scale``, \
    and ``k = {-128, -127, ..., 126, 127}``.
    """
    if not descale:
        return step_quantize(x, step=2 / scale).clamp(-256 / scale, 255 / scale)
    else:
        return step_quantize(x, step=2 / scale).clamp(-256 / scale, 255 / scale) * scale


@torch.jit.script
def right_shift_to_zero(x: torch.Tensor, bits: int):
    dtype = x.dtype
    assert dtype in (torch.int32, torch.int64)
    return (torch.sign(x) * (torch.abs(x) >> bits)).to(dtype)


@torch.jit.script
def _listep_forward(x: torch.Tensor, decay: torch.Tensor, state: torch.Tensor, w_scale: int,
                    dtype: torch.dtype = torch.int32, hw_bits: int = 12):
    # y = (state * w_scale * ((1 << hw_bits) - decay) / (1 << hw_bits) + w_scale * x) / w_scale
    # y = state * (1 - decay / (1 << hw_bits)) + x
    scaled_state = (state * w_scale).to(dtype=dtype)
    decay_int = (1 << hw_bits) - decay.to(dtype=dtype)
    output = right_shift_to_zero(scaled_state * decay_int, hw_bits) + (w_scale * x).to(dtype=dtype)
    return output / w_scale


@torch.jit.script
def _listep_backward(grad_output: torch.Tensor, decay: torch.Tensor, state: torch.Tensor, hw_bits: int = 12):
    grad_state = (1 - decay / (1 << hw_bits)) * grad_output
    grad_decay = - state / (1 << hw_bits) * grad_output

    grad_decay = grad_decay.sum()

    return grad_output, grad_decay, grad_state
    # x, decay, state

class BatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-05, momentum: float = 0.1,
                 track_running_stats: bool = True, weight_exp_bits: int = 3, pre_hook_fx: Callable = lambda x: x):
        super().__init__()
        # lava.lib.dl.slayer.neuron.norm.WgtScaleBatchNorm
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.weight_exp_bits = weight_exp_bits

        self.pre_hook_fx = pre_hook_fx
        self.register_buffer(
            'running_mean',
            torch.zeros(num_features)
        )
        self.register_buffer(
            'running_var',
            torch.zeros(1)
        )

    def to_lava(self):
        bn = slayer.neuron.norm.WgtScaleBatchNorm(num_features=self.num_features, momentum=self.momentum, weight_exp_bits=self.weight_exp_bits, eps=self.eps, pre_hook_fx=self.pre_hook_fx)
        bn.load_state_dict(self.state_dict())
        print(self.state_dict())
        return bn

    def forward(self, x: torch.Tensor):
        if self.track_running_stats and self.training:
            x_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            x_var = torch.var(x, unbiased=False)
            numel = x.numel() / self.num_features
            with torch.no_grad():
                self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * x_mean.squeeze()
                self.running_var = (1. - self.momentum) * self.running_var + self.momentum * x_var * numel / (numel + 1)

        else:
            x_mean = self.running_mean.view(1, -1, 1, 1)
            x_var = self.running_var.view(1, -1, 1, 1)

        x_std = torch.sqrt(x_var + self.eps)


        x_std = torch.pow(2., torch.ceil(torch.log2(x_std)).clamp(
            -self.weight_exp_bits, self.weight_exp_bits
        ))

        return (x - self.pre_hook_fx(x_mean)) / x_std

class LeakyIntegratorStep(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, x, decay, state, w_scale):
        output = _listep_forward(x, decay, state, w_scale, dtype=torch.int64, hw_bits=_hw_bits)
        if x.requires_grad or state.requires_grad:
            ctx.save_for_backward(decay, state)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        decay, state = ctx.saved_tensors
        grad_input, grad_decay, grad_state = _listep_backward(
            grad_output, decay, state, hw_bits=_hw_bits
        )
        return grad_input, grad_decay, grad_state, None




class CubaLIFNode(neuron.BaseNode):
    def __init__(
            self, current_decay: Union[float, torch.Tensor], voltage_decay: Union[float, torch.Tensor],
            v_threshold: float = 1., v_reset: float = 0.,
            scale=1 << 6,
            requires_grad=False,
            surrogate_function: Callable = surrogate.Sigmoid(),
            norm: BatchNorm2d = None,
            detach_reset=False,
            step_mode="s", backend="torch",
            store_v_seq: bool = False, store_i_seq: bool = False,
    ):
        # author: https://github.com/AllenYolk
        """
        * :ref:`API in English <CubaLIFNode.__init__-en>`

        .. _CubaLIFNode.__init__-cn:

        :param current_decay: 电流衰减常数
        :type current_decay: Union[float, torch.Tensor]

        :param voltage_decay: 电压衰减常数
        :type voltage_decay: Union[float, torch.Tensor]

        :param v_threshold: 神经元阈值电压。默认为1。
        :type v_threshold: float

        :param v_reset: 重置电压，默认为0
        :type v_reset: float, None


        :param scale: 量化参数，控制神经元的量化精度（参考了lava-dl的cuba.Neuron）。默认为 ``1<<6`` 。
            等效于``w_scale=int(scale)``, ``s_scale=int(scale * (1<<6))``, ``p_scale=1<<12``。
        :type scale: float

        :param requires_grad: 指明 ``current_decay`` 和 ``voltage_decay`` 两个神经元参数是否可学习（是否需要梯度），默认为 ``False`` 。
        :type requires_grad: bool

        :param detach_reset: 是否将reset的计算图分离，默认为 ``False`` 。
        :type detach_reset: bool

        :param step_mode: 步进模式，可以为 `'s'` （单步）或 `'m'` （多步），默认为 `'s'` 。
        :type step_mode: str

        :param backend: 使用哪种后端。不同的 ``step_mode`` 可能会带有不同的后端。可以通过打印 ``self.supported_backends`` 查看当前
            使用的步进模式支持的后端。目前只支持torch
        :type backend: str

        :param store_v_seq: 在使用 ``step_mode = 'm'`` 时，给与 ``shape = [T, N, *]`` 的输入后，是否保存中间过程的 ``shape = [T, N, *]``
            的各个时间步的电压值 ``self.v_seq`` 。设置为 ``False`` 时计算完成后只保留最后一个时刻的电压，即 ``shape = [N, *]`` 的 ``self.voltage_state`` 。
            通常设置成 ``False`` ，可以节省内存。
        :type store_v_seq: bool

        :param store_i_seq: 在使用 ``step_mode = 'm'`` 时，给与 ``shape = [T, N, *]`` 的输入后，是否保存中间过程的 ``shape = [T, N, *]``
            的各个时间步的电流值 ``self.i_seq`` 。设置为 ``False`` 时计算完成后只保留最后一个时刻的电流，即 ``shape = [N, *]`` 的 ``self.current_state`` 。
            通常设置成 ``False`` ，可以节省内存。
        :type store_i_seq: bool

        .. math::
            I[t] = (1 - \\alpha_{I})I[t-1] + X[t]
            V[t] = (1 - \\alpha_{V})V[t-1] + I[t]


        * :ref:`中文API <CubaLIFNode.__init__-cn>`

        .. _CubaLIFNode.__init__-en:

        :param current_decay: current decay constant
        :type current_decay: Union[float, torch.Tensor]

        :param voltage_decay: voltage decay constant
        :type voltage_decay: Union[float, torch.Tensor]

        :param v_threshold: threshold of the the neurons in this layer. Default to 1.
        :type v_threshold: float

        :param v_reset: reset potential of the neurons in this layer, 0 by default
        :type v_reset: float

        :param scale: quantization precision (ref: lava-dl cuba.Neuron). Default to ``1<<6`` .
            Equivalent to ``w_scale=int(scale)``, ``s_scale=int(scale * (1<<6))``, ``p_scale=1<<12``.
        :type scale: float

        :param requires_grad: whether ``current_decay`` and ``voltage_decay`` are learnable. Default to ``False`` .
        :type requires_grad: bool


        :param detach_reset: whether to detach the computational graph of reset in backward pass. Default to ``False`` .
        :type detach_reset: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step). Default to `'s'` .
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
        print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. Only `torch` is supported.
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.voltage_state`` with ``shape = [N, *]``, which can reduce the
            memory consumption. Default to ``False`` .
        :type store_v_seq: bool

        :param store_i_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the current at each time-step to ``self.i_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the current at last time-step will be stored to ``self.current_state`` with ``shape = [N, *]``, which can reduce the
            memory consumption. Default to ``False`` .
        :type store_i_seq: bool
        .. math::
            I[t] = (1 - \\alpha_{I})I[t-1] + X[t]
            V[t] = (1 - \\alpha_{V})V[t-1] + I[t]

        """
        self.lava_cuba_neuron_params = {
            'threshold': v_threshold,
            'current_decay': current_decay,
            'voltage_decay': voltage_decay,
            'scale': scale
        }

        super().__init__(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function,
                         detach_reset=detach_reset, step_mode=step_mode, backend=backend, store_v_seq=store_v_seq)

        self.store_i_seq = store_i_seq
        assert v_reset == 0., 'CubaLIFNode only supports for hard reset with v_reset = 0. !'
        self.requires_grad = requires_grad

        # the default quantization parameter setting in lava
        self._scale = int(scale)
        self._s_scale = int(scale * (1 << 6))
        self._p_scale = 1 << _hw_bits
        # Which is equivalent to:
        # self.p_scale = 1<<12
        # self.w_scale = int(scale)
        # self.s_scale = int(scale * (1<<6))

        self._v_threshold = int(v_threshold * self.scale) / self.scale
        # ``_v_threshold`` is the nearest and no more than ``k / scale`` to ``v_threshold`` where ``k`` is an ``int``

        self.v_threshold_eps = 0.01 / self.s_scale
        # loihi use s[t] = v[t] > v_th, but we use s[t] = v[t] >= v_th. Thus, we use v[t] + eps >= v_th to approximate

        current_decay = torch.tensor(self.p_scale * current_decay, dtype=torch.float32)
        voltage_decay = torch.tensor(self.p_scale * voltage_decay, dtype=torch.float32)

        if requires_grad:
            self.current_decay = nn.Parameter(current_decay)
            self.voltage_decay = nn.Parameter(voltage_decay)
        else:
            self.register_buffer('current_decay', current_decay)
            self.register_buffer('voltage_decay', voltage_decay)

        self.register_memory('current_state', 0.)
        self.register_memory('voltage_state', 0.)

        self.clamp_decay_parameters()

        self.norm = norm

        if self.norm is not None:

            if isinstance(self.norm, BatchNorm2d):
                self.norm.pre_hook_fx = self.quantize_8bit

            else:
                raise NotImplementedError(self.norm)






    def quantize_8bit(self, x, descale=False):
        return quantize_8b(x, scale=self.scale, descale=descale)

    def clamp_decay_parameters(self):
        with torch.no_grad():
            self.current_decay.data.clamp_(min=0., max=self.p_scale)
            self.voltage_decay.data.clamp_(min=0., max=self.p_scale)

    @property
    def scale(self):
        """Read-only attribute: scale"""
        return self._scale

    @property
    def s_scale(self):
        """Read-only attribute: s_scale"""
        return self._s_scale

    @property
    def p_scale(self):
        """Read-only attribute: s_scale"""
        return self._p_scale

    @property
    def store_i_seq(self):
        return self._store_i_seq

    @store_i_seq.setter
    def store_i_seq(self, value: bool):
        self._store_i_seq = value
        if value:
            if not hasattr(self, "i_seq"):
                self.register_memory("i_seq", None)

    @property
    def supported_backends(self):
        if self.step_mode == "m" or self.step_mode == "s":
            return ("torch",)
        else:
            raise ValueError(
                f"self.step_mode should be 's' or 'm', "
                f"but get {self.step_mode} instead."
            )

    # computation process
    def state_initialization(self, x: torch.Tensor):
        if isinstance(self.current_state, float):
            self.current_state = torch.zeros_like(x.data)

        if isinstance(self.voltage_state, float):
            self.voltage_state = torch.zeros_like(x.data)

    def neuronal_charge(self, x: torch.Tensor):
        if self.requires_grad:
            self.clamp_decay_parameters()

        current = LeakyIntegratorStep.apply(
            x,
            step_quantize(self.current_decay),
            self.current_state.contiguous(),
            self.s_scale,
        )

        if self.norm is not None:
            current = self.norm(current)

        voltage = LeakyIntegratorStep.apply(
            current,
            step_quantize(self.voltage_decay),
            self.voltage_state.contiguous(),
            self.s_scale,
        )

        self.current_state = current
        self.voltage_state = voltage

    def neuronal_fire(self):
        return self.surrogate_function(self.voltage_state - (self.v_threshold + self.v_threshold_eps))

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        self.voltage_state = self.jit_hard_reset(self.voltage_state, spike_d, self.v_reset)

    def single_step_forward(self, x):
        self.state_initialization(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        if self.store_i_seq:
            i_seq = []

        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.voltage_state)
            if self.store_i_seq:
                i_seq.append(self.current_state)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        if self.store_i_seq:
            self.i_seq = torch.stack(i_seq)

        return torch.stack(y_seq)


try:
    import lava.lib.dl.slayer as slayer


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


    def lava_neuron_forward(lava_neuron: nn.Module, x_seq: torch.Tensor, v: Union[torch.Tensor, float]):
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

        .. image:: ../_static/API/activation_based/lava_exchange/step_quantize.*
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

        slayer_conv = slayer.synapse.Conv(in_features=conv2d_nn.in_channels, out_features=conv2d_nn.out_channels,
                                          kernel_size=conv2d_nn.kernel_size, stride=conv2d_nn.stride,
                                          padding=conv2d_nn.padding, dilation=conv2d_nn.dilation,
                                          groups=conv2d_nn.groups)
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
        logging.warning(
            'The lava slayer pool layer applies sum pooling, rather than average pooling. `avgpool2d_to_lava_synapse_pool` will return a sum pooling layer.')

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
            lava_block = block_init(neuron_params, in_features=conv2d_nn.in_channels,
                                    out_features=conv2d_nn.out_channels, kernel_size=conv2d_nn.kernel_size,
                                    stride=conv2d_nn.stride, padding=conv2d_nn.padding, dilation=conv2d_nn.dilation,
                                    groups=conv2d_nn.groups, delay_shift=False)
        else:
            lava_block = block_init(neuron_params, in_features=conv2d_nn.in_channels,
                                    out_features=conv2d_nn.out_channels, kernel_size=conv2d_nn.kernel_size,
                                    stride=conv2d_nn.stride, padding=conv2d_nn.padding, dilation=conv2d_nn.dilation,
                                    groups=conv2d_nn.groups, delay_shift=False, pre_hook_fx=None)

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
            lava_block = block_init(neuron_params, pool2d_nn.kernel_size, pool2d_nn.stride, pool2d_nn.padding,
                                    delay_shift=False)
        else:
            lava_block = block_init(neuron_params, pool2d_nn.kernel_size, pool2d_nn.stride, pool2d_nn.padding,
                                    delay_shift=False, pre_hook_fx=None)

        logging.warning(
            'The lava slayer pool layer applies sum pooling, rather than average pooling. `avgpool2d_to_lava_synapse_pool` will return a sum pooling layer.')

        return lava_block


    def to_lava_block_flatten(flatten_nn: nn.Flatten):
        check_instance(flatten_nn, nn.Flatten)
        if flatten_nn.start_dim != 1:
            raise ValueError('lava only supports for flatten_nn.start_dim == 1!')
        return slayer.block.cuba.Flatten()


    def to_lava_blocks(net: Union[list, tuple, nn.Sequential]):
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
        k = None
        while True:
            if isinstance(net[i], nn.Linear):
                if k is not None:
                    if isinstance(net[i], (nn.Conv2d, nn.Linear)):
                        net[i].weight.data /= k
                    else:
                        raise NotImplementedError(type(net[i]))

                    k = None
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
                    if isinstance(net[i].kernel_size, int):
                        k = float(net[i].kernel_size * net[i].kernel_size)
                    else:
                        k = float(net[i].kernel_size[0] * net[i].kernel_size[1])
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


    class SumPool2d(nn.Module):
        def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
            """
            .. code-block:: python

                x = torch.rand([4, 2, 4, 16, 16])

                with torch.no_grad():
                    sp_sj = SumPool2d(kernel_size=2, stride=2)
                    y_sj = functional.seq_to_ann_forward(x, sp_sj)

                    sp_la = slayer.synapse.Pool(kernel_size=2, stride=2)
                    y_la = lava_exchange.NXT_to_TNX(sp_la(lava_exchange.TNX_to_NXT(x)))
                    print((y_sj - y_la).abs().sum())
            """
            super().__init__()
            temp_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride,
                                  padding=padding,
                                  dilation=dilation, bias=False)

            self.weight = torch.ones_like(temp_conv.weight.data)
            self.kernel_size = temp_conv.kernel_size
            self.stride = temp_conv.stride
            self.padding = temp_conv.padding
            self.dilation = temp_conv.dilation
            del temp_conv

        def forward(self, x: torch.Tensor):
            # x.shape = [N, C, H, W]
            if self.dilation == (1, 1):
                return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding) * self.weight.numel()
            else:
                N, C, H, W = x.shape
                x = x.view(N * C, 1, H, W)
                x = F.conv2d(x, weight=self.weight, bias=None, stride=self.stride, padding=self.padding,
                             dilation=self.dilation)
                x = x.view(N, C, x.shape[2], x.shape[3])

                return x



    class BlockContainer(nn.Module, base.StepModule):

        @property
        def step_mode(self):
            return self._step_mode

        @step_mode.setter
        def step_mode(self, value: str):
            if value not in self.supported_step_mode():
                raise ValueError(f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!')
            self._step_mode = value
            if isinstance(self.neuron, base.StepModule):
                self.neuron.step_mode = value

            if isinstance(self.synapse, base.StepModule):
                self.synapse.step_mode = value

        def __init__(self, synapse: Union[nn.Conv2d, nn.Linear, nn.AvgPool2d, nn.Flatten], neu: Optional[CubaLIFNode],
                     step_mode: str = 's'):
            super().__init__()
            if isinstance(synapse, nn.Flatten):
                assert neu is None
                self.synapse = synapse
                self.neuron = None
                if synapse.start_dim != 1:
                    raise ValueError('lava only supports for torch.nn.Flatten with start_dim == 1!')
            else:

                if isinstance(neu, neuron.IFNode):
                    if neu.v_reset != 0.:
                        raise ValueError('lava only supports for v_reset == 0!')
                    neu = CubaLIFNode(current_decay=1., voltage_decay=0., v_threshold=neu.v_threshold, scale=neu.lava_s_cale)

                elif isinstance(neu, neuron.LIFNode):
                    if neu.v_reset != 0.:
                        raise ValueError('lava only supports for v_reset == 0!')
                    if neu.decay_input:
                        raise ValueError('lava only supports for decay_input == False!')
                    neu = CubaLIFNode(current_decay=1., voltage_decay=1. / neu.tau, v_threshold=neu.v_threshold,
                                      scale=neu.lava_s_cale)
                else:
                    assert isinstance(neu, CubaLIFNode)
                self.synapse = synapse
                self.neuron = neu
                if isinstance(self.synapse, (nn.Conv2d, nn.Linear)):
                    assert self.synapse.bias is None

            self.step_mode = step_mode

        def forward(self, x: torch.Tensor):
            if self.step_mode == 'm':
                T = x.shape[0]
                N = x.shape[1]
                x = x.flatten(0, 1)

            if isinstance(self.synapse, (nn.Conv2d, nn.Linear)):
                weight = self.neuron.quantize_8bit(self.synapse.weight)
                # 量化到 2k / self.neuron.scale, k = -128, -127, ..., 127，共有256个取值

                if isinstance(self.synapse, nn.Conv2d):
                    x = F.conv2d(x, weight=weight, bias=self.synapse.bias, stride=self.synapse.stride,
                                 padding=self.synapse.padding, dilation=self.synapse.dilation,
                                 groups=self.synapse.groups)

                elif isinstance(self.synapse, nn.Linear):
                    x = F.linear(x, weight, self.synapse.bias)

            elif isinstance(self.synapse, (SumPool2d, nn.Flatten)):
                x = self.synapse(x)

            else:
                raise NotImplementedError(type(self.synapse))

            if self.step_mode == 'm':
                x = x.view([T, N, *x.shape[1:]])

            if self.neuron is not None:
                x = self.neuron(x)

            return x

        def to_lava_block(self):


            if isinstance(self.synapse, nn.Linear):
                lava_block = slayer.block.cuba.Dense(self.neuron.lava_cuba_neuron_params, self.synapse.in_features,
                                                     self.synapse.out_features, delay_shift=False)
                lava_block.synapse.weight.data[:, :, 0, 0, 0] = self.synapse.weight.data.clone()

            elif isinstance(self.synapse, nn.Conv2d):
                lava_block = slayer.block.cuba.Conv(self.neuron.lava_cuba_neuron_params,
                                                    in_features=self.synapse.in_channels,
                                                    out_features=self.synapse.out_channels,
                                                    kernel_size=self.synapse.kernel_size,
                                                    stride=self.synapse.stride, padding=self.synapse.padding,
                                                    dilation=self.synapse.dilation,
                                                    groups=self.synapse.groups, delay_shift=False)
                lava_block.synapse.weight.data[:, :, :, :, 0] = self.synapse.weight.data.clone()

            elif isinstance(self.synapse, SumPool2d):
                lava_block = slayer.block.cuba.Pool(self.neuron.lava_cuba_neuron_params, self.synapse.kernel_size,
                                                    self.synapse.stride, self.synapse.padding, self.synapse.dilation,
                                                    delay_shift=False)

            elif isinstance(self.synapse, nn.Flatten):
                return slayer.block.cuba.Flatten()
            else:
                raise NotImplementedError


            # 补上norm
            if self.neuron.norm is not None:
                lava_block.neuron.norm = self.neuron.norm.to_lava()
            return lava_block








except BaseException as e:
    logging.info(f'spikingjelly.activation_based.lava_exchange: {e}')
    slayer = None
