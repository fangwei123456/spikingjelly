from typing import Union, Callable

import numpy as np
import torch
import torch.nn as nn

from .quantize import step_quantize, quantize_8b, right_shift_to_zero
from .base import MemoryModule
from . import surrogate


hw_bits = 12


def _listep_forward(x: torch.Tensor, decay: torch.Tensor, state: torch.Tensor, w_scale: int, dtype=torch.int32):
    # y = (state * w_scale * ((1 << hw_bits) - decay) / (1 << hw_bits) + w_scale * x) / w_scale
    # y = state * (1 - decay / (1 << hw_bits)) + x
    scaled_state = (state * w_scale).to(dtype=dtype)
    decay_int = (1 << hw_bits) - decay.to(dtype=dtype)
    output = right_shift_to_zero(scaled_state * decay_int, hw_bits) + (w_scale * x).to(dtype=dtype)
    return output / w_scale


def _listep_backward(grad_output: torch.Tensor, decay: torch.Tensor, state: torch.Tensor):
    grad_state = (1 - decay / (1 << hw_bits)) * grad_output
    grad_decay = - state / (1 << hw_bits) * grad_output

    grad_decay = grad_decay.sum()

    return grad_output, grad_decay, grad_state
    # x, decay, state


class LeakyIntegratorStep(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, x, decay, state, w_scale):
        output = _listep_forward(x, decay, state, w_scale, dtype=torch.int64)
        if x.requires_grad or state.requires_grad:
            ctx.save_for_backward(decay, state)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        decay, state = ctx.saved_tensors
        grad_input, grad_decay, grad_state = _listep_backward(
            grad_output, decay, state
        )
        return grad_input, grad_decay, grad_state, None


class CubaLIFNode(MemoryModule):
    def __init__(
            self, current_decay: Union[float, torch.Tensor], voltage_decay: Union[float, torch.Tensor],
            v_threshold: float = 1., v_reset: float = 0.,
            scale=1 << 6,
            requires_grad=False,
            surrogate_function: Callable = surrogate.Sigmoid(),
            detach_reset=False,
            step_mode="s", backend="torch",
            store_v_seq: bool = False, store_i_seq: bool = False,
    ):
        """
        * :ref:`API in English <CubaLIFNode.__init__-en>`
        
        .. _CubaLIFNode.__init__-cn:

        :param current_decay: 电流衰减常数
        :type current_decay: float | torch.Tensor

        :param voltage_decay: 电压衰减常数
        :type voltage_decay: float | torch.Tensor

        :param v_threshold: 神经元阈值电压。默认为1。
        :type v_threshold: float

        :param v_reset: 重置电压，默认为0。若为 ``None``，则必为软·重置；
            若不为 ``None``，则取决于 ``soft_reset``的值来进行软/硬重置
        :type v_reset: float, None


        :param scale: 量化参数，控制神经元的量化精度（参考了lava-dl的cuba.Neuron）。默认为 ``1<<6`` 。
            等效于``w_scale=int(scale)``, ``s_scale=int(scale * (1<<6))``, ``p_scale=1<<12``。
        :type scale: float

        :param requires_grad: 指明 ``current_decay`` 和 ``voltage_decay`` 两个神经元参数是否可学习（是否需要梯度），默认为 ``False`` 。
        :type requires_grad: bool

        :param detach_reset: 是否将reset的计算图分离，默认为 ``False`` 。
        :type detach_reset: bool

        :param soft_reset: 是否进行软重制，默认为 ``False`` 。
            注意：即使这个值为 ``False`` ，倘若 ``v_reset`` 为 ``None`` 的话，也会强行进行软重制。
        :type soft_reset: bool

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

        .. CubaLIFNode.__init__-en:

        :param current_decay: current decay constant
        :type current_decay: float | torch.Tensor

        :param voltage_decay: voltage decay constant
        :type voltage_decay: float | torch.Tensor

        :param v_threshold: threshold of the the neurons in this layer. Default to 1.
        :type v_threshold: float

        :param v_reset: reset potential of the neurons in this layer, 0 by default.
            If ``None`` , do soft reset.
            If not ``None`` , the type of reset depends on the value of ``soft_reset``.
        :type v_reset: float, None

        :param scale: quantization precision (ref: lava-dl cuba.Neuron). Default to ``1<<6`` .
            Equivalent to ``w_scale=int(scale)``, ``s_scale=int(scale * (1<<6))``, ``p_scale=1<<12``.
        :type scale: float

        :param requires_grad: whether ``current_decay`` and ``voltage_decay`` are learnable. Default to ``False`` .
        :type requires_grad: bool


        :param detach_reset: whether to detach the computational graph of reset in backward pass. Default to ``False`` .
        :type detach_reset: bool

        :param soft_reset: whether to do soft reset. Default to ``False`` .
            Notice: even if the value is ``False`` , soft reset will be done if ``v_reset`` is ``None`` .
        :type soft_reset: bool

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
        super().__init__()
        self.detach_reset = detach_reset
        self.step_mode = step_mode
        self.backend = backend
        self.store_v_seq = store_v_seq
        self.store_i_seq = store_i_seq
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.requires_grad = requires_grad

        # the default quantization parameter setting in lava
        self._scale = int(scale)
        self._s_scale = int(scale * (1 << 6))
        # Which is equivalent to:
        # self.p_scale = 1<<12
        # self.w_scale = int(scale)
        # self.s_scale = int(scale * (1<<6))

        self._v_threshold = int(v_threshold * self.scale) / self.scale
        # ``_v_threshold`` is the nearest and no more than ``k / scale`` to ``v_threshold`` where ``k`` is an ``int``

        self.v_threshold_eps = 0.01 / self.s_scale
        # loihi use s[t] = v[t] > v_th, but we use s[t] = v[t] >= v_th. Thus, we use v[t] + eps >= v_th to approximate

        current_decay = torch.tensor((1 << hw_bits) * current_decay, dtype=torch.float32)
        voltage_decay = torch.tensor((1 << hw_bits) * voltage_decay, dtype=torch.float32)

        if requires_grad:
            self.current_decay = nn.Parameter(current_decay)
            self.voltage_decay = nn.Parameter(voltage_decay)
        else:
            self.register_buffer('current_decay', current_decay)
            self.register_buffer('voltage_decay', voltage_decay)

        self.register_memory('current_state', 0.)
        self.register_memory('voltage_state', 0.)

        self.clamp_decay_parameters()

    def quantize_8bit(self, x, descale=False):
        return quantize_8b(x, scale=self.scale, descale=descale)

    def clamp_decay_parameters(self):
        with torch.no_grad():
            self.current_decay.data.clamp_(min=0., max=1 << hw_bits)
            self.voltage_decay.data.clamp_(min=0., max=1 << hw_bits)

    # properties
    @property
    def device(self):
        return self.current_decay.device

    @property
    def scale(self):
        """Read-only attribute: scale"""
        return self._scale

    @property
    def s_scale(self):
        """Read-only attribute: s_scale"""
        return self._s_scale

    @property
    def v_threshold(self):
        """Read-only attribute: v_threshold"""
        return self._v_threshold

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, "v_seq"):
                self.register_memory("v_seq", None)

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

    # static jit methods
    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    # computation process
    def state_initialization(self, x: torch.Tensor):
        self.current_state = torch.zeros_like(x.data)
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
        if isinstance(self.current_state, float) or isinstance(self.voltage_state, float):
            self.state_initialization(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if isinstance(self.current_state, float) or isinstance(self.voltage_state, float):
            self.state_initialization(x_seq[0])
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        if self.store_i_seq:
            i_seq = []

        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y.unsqueeze(0))
            if self.store_v_seq:
                v_seq.append(self.voltage_state)
            if self.store_i_seq:
                i_seq.append(self.current_state)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        if self.store_i_seq:
            self.i_seq = torch.stack(i_seq)

        return torch.cat(y_seq, dim=0)
