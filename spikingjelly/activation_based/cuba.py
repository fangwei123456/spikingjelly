from typing import Union

import numpy as np
import torch

from .quantize import step_quantize, quantize_8b, right_shift_to_zero
from .base import MemoryModule
from .surrogate import cuba_spike


SCALE_GRAD_MULT = 0.1
TAU_GRAD_MULT = 100


def _listep_forward(input, decay, state, w_scale, dtype = torch.int32):
    device = input.device
    scaled_state = (state * w_scale).clone().detach().to(
        dtype = dtype, device = device
    )
    decay_int = (1<<12) - decay.clone().detach().to(
        dtype = dtype, device = device
    )
    output = right_shift_to_zero(scaled_state * decay_int, 12) \
        + (w_scale * input).to(dtype = dtype)
    return output / w_scale


def _listep_backward_correct(grad_output, decay, state,):
    with torch.no_grad():
        decay_factor = 1 - decay / (1<<12)
        grad_input = grad_output
        grad_state = grad_output * decay_factor
        grad_decay = -grad_output * state
        if torch.numel(decay) == 1:
            grad_decay = torch.sum(grad_decay).unsqueeze(dim = 0)
        else:
            grad_decay = torch.sum(grad_decay, dim = 0)
    return grad_input, grad_decay, grad_state


def _listep_backward_lava(grad_output, decay, state, last_state_before_spike,):
    with torch.no_grad():
        decay_factor = 1 - decay / (1 << 12)
        grad_input = grad_output
        grad_state = grad_output * decay_factor
        if last_state_before_spike is not None:
            grad_decay = grad_output * last_state_before_spike
        else:
            grad_decay = grad_output * state
        if torch.numel(decay) == 1:
            grad_decay = torch.sum(grad_decay).unsqueeze(dim = 0)
        else:
            grad_decay = torch.sum(grad_decay, dim = 0) 
    return grad_input, grad_decay, grad_state


class LeakyIntegratorStep(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, decay, state, w_scale, 
        lava_charge_bp, last_state_before_spike
    ):
        output = _listep_forward(input, decay, state, w_scale, dtype = torch.int64)
        ctx.lava_charge_bp = lava_charge_bp
        if lava_charge_bp:
            ctx.save_for_backward(decay, state, last_state_before_spike)
        else:
            ctx.save_for_backward(decay, state)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.lava_charge_bp:
            decay, state, last_state_before_spike = ctx.saved_tensors
            grad_input, grad_decay, grad_state = _listep_backward_lava(
                grad_output, decay, state, last_state_before_spike
            )
        else:
            decay, state = ctx.saved_tensors
            grad_input, grad_decay, grad_state = _listep_backward_correct(
                grad_output, decay, state
            )
        return grad_input, grad_decay, grad_state, None, None, None,


class CubaNeuronReset(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, spike, v_reset):
        v_after = (1. - spike) * v + spike * v_reset
        return v_after

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class CubaLIFNode(MemoryModule):
    def __init__(
        self, current_decay: Union[float, torch.Tensor], voltage_decay: Union[float, torch.Tensor],
        v_threshold: float = 1., v_reset: float = 0.,
        tau_grad = 1, scale_grad = 1, scale = 1 << 6,
        requires_grad = False, graded_spike = False,  
        normalization = None, lava_style = True,
        detach_reset = False, soft_reset = False,
        step_mode = "s", backend = "torch",
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

        :param tau_grad: 控制梯度替代函数的陡峭程度，默认为1。
        :type tau_grad: float

        :param scale_grad: 控制梯度替代函数的幅度，默认为1。
        :type scale_grad: float

        :param scale: 量化参数，控制神经元的量化精度（参考了lava-dl的cuba.Neuron）。默认为 ``1<<6`` 。
            等效于``w_scale=int(scale)``, ``s_scale=int(scale * (1<<6))``, ``p_scale=1<<12``。
        :type scale: float

        :param requires_grad: 指明 ``current_decay`` 和 ``voltage_decay`` 两个神经元参数是否可学习（是否需要梯度），默认为 ``False`` 。
        :type requires_grad: bool

        :param graded_spike: 指明输出的形式，默认为 ``False`` 。
            若为 ``False`` ，则为常规脉冲输出形式；输出取值为0或1。
            若为 ``True`` ，则输出为0（若不发放脉冲）或脉冲前电压（若发放脉冲）。
        :type graded_spike: bool

        :param normalization: 对电流的normalization函数，默认为 ``None`` 。
        :type normalization: Callable

        :param lava_style: 是否严格按照lava-dl中 ``cuba.Neuron`` 的机制进行计算，默认为 ``True`` 。
            若为 ``True`` ，则 ``detach_reset, soft_reset``这两个自定义选项均将被无视。
        :type lava_style: bool

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

        :param tau_grad: control the steepness of the surrogate gradient function. Default to 1.
        :type tau_grad: float

        :param scale_grad: control the scale of the surrogate gradient function. Default to 1.
        :type scale_grad: float

        :param scale: quantization precision (ref: lava-dl cuba.Neuron). Default to ``1<<6`` .
            Equivalent to ``w_scale=int(scale)``, ``s_scale=int(scale * (1<<6))``, ``p_scale=1<<12``.
        :type scale: float

        :param requires_grad: whether ``current_decay`` and ``voltage_decay`` are learnable. Default to ``False`` .
        :type requires_grad: bool

        :param graded_spike: the form of spike output. Default to ``False`` .
            If ``False`` , the spike output takes value from 0 and 1.
            If ``True`` , the spike output is 0 if a spike is not emitted and takes the value of the pre-spike voltage if there's a spike.
        :type graded_spike: bool

        :param normalization: normalization function acting on neuronal current. Default to ``None`` .
        :type normalization: Callable

        :param lava_style: whether to strictly follow the computation of ``cuba.Neuron`` . Default to ``True`` .
            If ``True``, optional arguments ``detach_reset, soft_reset`` will have no effect.
        :type lava_style: bool

        :param detach_reset: whether to detach the computational graph of reset in backward pass. Default to ``False`` .
        :type detach_reset: bool

        :param soft_reset: whether to do soft reset. Default to ``False`` .
            Notice: even if the value is ``False`` , soft reset will be done if ``v_reset`` is ``None`` .
        :type soft_reset: bool

        :param step_mode: 可the step mode, which can be `s` (single-step) or `m` (multi-step). Default to `'s'` .
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
        self.lava_style = lava_style
        self.detach_reset = detach_reset
        self.soft_reset = soft_reset
        self.step_mode = step_mode
        self.backend = backend
        self.store_v_seq = store_v_seq
        self.store_i_seq = store_i_seq
        self.v_reset = v_reset

        # the default quantization parameter setting in lava
        self._scale = int(scale)
        self._s_scale = int(scale * (1<<6))
        # Which is equivalent to:
        # self.p_scale = 1<<12
        # self.w_scale = int(scale)
        # self.s_scale = int(scale * (1<<6))

        self._v_threshold = int(v_threshold*self.scale) / self.scale
        self.tau_grad = tau_grad * self.v_threshold
        self.scale_grad = scale_grad
        self.requires_grad = requires_grad
        self.graded_spike = graded_spike
        self.v_threshold_eps = 0.01 / self.s_scale

        self.normalization = normalization
        if self.normalization is not None and hasattr(self.normalization, "pre_hook_fx"):
            self.normalization.pre_hook_fx = self.quantize_8bit

        self.current_decay = torch.nn.Parameter(
            torch.tensor(
                [(1<<12) * current_decay] 
                    if np.isscalar(current_decay) 
                    else (1<<12) * current_decay,
                dtype = torch.float32,
            ),
            requires_grad = self.requires_grad,
        )
        self.voltage_decay = torch.nn.Parameter(
            torch.tensor(
                [(1<<12) * voltage_decay] 
                    if np.isscalar(voltage_decay) 
                    else (1<<12) * voltage_decay, 
                dtype = torch.float32,
            ),
            requires_grad = self.requires_grad,
        )

        self.register_memory(
            "current_state",
            torch.zeros(1, dtype = torch.float)
        )
        self.register_memory(
            "voltage_state",
            torch.zeros(1, dtype = torch.float)
        )
        self.register_memory("shape", None)
        self.register_memory("num_neuron", None)
        self.register_memory("last_voltage_before_spike", None)

        self.clamp_decay_parameters()

    def quantize_8bit(self, x, descale = False):
        return quantize_8b(x, scale = self.scale, descale = descale)

    def clamp_decay_parameters(self):
        with torch.no_grad():
            self.current_decay.data.clamp_(min = 0., max = 1<<12)
            self.voltage_decay.data.clamp_(min = 0., max = 1<<12)

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
    def device_params(self):
        self.clamp_decay_parameters()
        cd_val = step_quantize(self.current_decay).cpu().data.numpy().astype(int)
        vd_val = step_quantize(self.voltage_decay).cpu().data.numpy().astype(int)
        return {
            'type': 'CUBA',
            'iDecay': cd_val[0] if len(cd_val) == 1 else cd_val,
            'vDecay': vd_val[0] if len(vd_val) == 1 else vd_val,
            'vThMant': int(self.v_threshold * self.scale),
            'refDelay': 1,
            'gradedSpike': self.graded_spike,
        }

    # static methods
    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    # computation process
    def neuronal_charge_1st(self, x: torch.Tensor):
        self.shape = x.shape[1:] # x.shape = [batch_size, ......]
        if len(self.shape) <= 0:
            raise ValueError(
                "x.shape of neuronal_charge() should be"
                "[batch_size, ......]"
            )
        self.num_neurons = int(np.prod(self.shape))

        self.current_state = self.current_state * torch.ones(x.shape).to(
            dtype = self.current_state.dtype,
            device = self.current_state.device,
        )
        self.voltage_state = self.voltage_state * torch.ones(x.shape).to(
            dtype = self.voltage_state.dtype,
            device = self.voltage_state.device,
        )

    def neuronal_charge(self, x: torch.Tensor):
        if self.shape is None: # if neuronal_charge() is called for the first time
            self.neuronal_charge_1st(x)
        else:
            if x.shape[1:] != self.shape:
                raise AssertionError(
                    f"x.shape of neuronal_charge should be"
                    f"{self.shape}"
                )
        if self.requires_grad:
            self.clamp_decay_parameters()

        current = LeakyIntegratorStep.apply(
            x,
            step_quantize(self.current_decay),
            self.current_state.contiguous(),
            self.s_scale,
            self.lava_style, None
        )
        if self.normalization is not None:
            current = self.normalization(current)
        voltage = LeakyIntegratorStep.apply(
            current,
            step_quantize(self.voltage_decay),
            self.voltage_state.contiguous(),
            self.s_scale,
            self.lava_style, self.last_voltage_before_spike
        )

        self.current_state = current
        self.voltage_state = voltage

    def neuronal_fire(self):
        return cuba_spike.apply(
            self.voltage_state,
            self.v_threshold + self.v_threshold_eps,
            self.tau_grad * TAU_GRAD_MULT,
            self.scale_grad * SCALE_GRAD_MULT,
            self.graded_spike,
        )

    def neuronal_reset(self, spike):
        if self.graded_spike:
            spike = (spike >= self.v_threshold).to(dtype = torch.float32)

        if self.lava_style:
            spike_d = spike.detach()
            v_after = CubaNeuronReset.apply(self.voltage_state, spike_d, self.v_reset)
            self.voltage_state = v_after

        else:
            if self.detach_reset:
                spike_d = spike.detach()
            else:
                spike_d = spike

            if (self.v_reset is None) or self.soft_reset: 
                self.voltage_state = self.jit_soft_reset(self.voltage_state, spike_d, self.v_threshold+self.v_threshold_eps)
            else:
                self.voltage_state = self.jit_hard_reset(self.voltage_state, spike_d, self.v_reset)

    def single_step_forward(self, x):
        self.neuronal_charge(x)
        if self.lava_style:
            self.last_voltage_before_spike = self.voltage_state
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
            y_seq.append(y.unsqueeze(0))
            if self.store_v_seq:
                v_seq.append(self.voltage_state.unsqueeze(0))
            if self.store_i_seq:
                i_seq.append(self.current_state.unsqueeze(0))

        if self.store_v_seq:
            self.v_seq = torch.cat(v_seq, dim = 0)
        if self.store_i_seq:
            self.i_seq = torch.cat(i_seq, dim = 0)

        return torch.cat(y_seq, dim = 0)