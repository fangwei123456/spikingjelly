from abc import abstractmethod
from typing import Callable, Optional

import torch
import torch.nn as nn

from .. import surrogate, base


__all__ = [
    "SimpleBaseNode",
    "BaseNode",
    "NonSpikingBaseNode",
]


class SimpleBaseNode(base.MemoryModule):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
    ):
        """
        **API Language:**
        :ref:`中文 <SimpleBaseNode.__init__-cn>` | :ref:`English <SimpleBaseNode.__init__-en>`

        ----

        .. _SimpleBaseNode.__init__-cn:

        * **中文**

        :class:`BaseNode` 的简化版，便于用户修改或扩展神经元。

        ----

        .. _SimpleBaseNode.__init__-en:

        * **English**

        A simple version of :class:`BaseNode`. Users can modify this neuron easily.
        """
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.detach_reset = detach_reset
        self.step_mode = step_mode
        self.register_memory(name="v", value=0.0)

    def single_step_forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - self.v_threshold * spike_d

        else:
            # hard reset
            self.v = spike_d * self.v_reset + (1.0 - spike_d) * self.v


class BaseNode(base.MemoryModule):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        """
        **API Language:**
        :ref:`中文 <BaseNode.__init__-cn>` | :ref:`English <BaseNode.__init__-en>`

        ----

        .. _BaseNode.__init__-cn:

        * **中文**

        可微分SNN神经元的基类神经元。

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，当神经元释放脉冲后，电压会被减去 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        :param backend: 使用哪种后端。不同的 ``step_mode`` 可能会带有不同的后端。可以通过打印 ``self.supported_backends`` 查看当前
            使用的步进模式支持的后端。在支持的情况下，使用 ``'cupy'`` 或 ``'triton'`` 后端速度更快。
        :type backend: str

        :param store_v_seq: 在使用 ``step_mode = 'm'`` 时，给与 ``shape = [T, N, *]`` 的输入后，是否保存中间过程的 ``shape = [T, N, *]``
            的各个时间步的电压值 ``self.v_seq`` 。设置为 ``False`` 时计算完成后只保留最后一个时刻的电压，即 ``shape = [N, *]`` 的 ``self.v`` 。
            通常设置成 ``False`` ，可以节省内存
        :type store_v_seq: bool

        ----

        .. _BaseNode.__init__-en:

        * **English**

        This class is the base class of differentiable spiking neurons.

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset in backward
        :type detach_reset: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
            print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
            using ``'cupy'`` or ``'triton'`` backend will be faster
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool
        """
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory("v", 0.0)
        else:
            self.register_memory("v", v_reset)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq

        # used in lava_exchange
        self.lava_s_cale = 1 << 6

        # used for cupy backend
        self.forward_kernel = None
        self.backward_kernel = None

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, "v_seq"):
                self.register_memory("v_seq", None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1.0 - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        """
        **API Language:**
        :ref:`中文 <BaseNode.neuronal_charge-cn>` | :ref:`English <BaseNode.neuronal_charge-en>`

        ----

        .. _BaseNode.neuronal_charge-cn:

        * **中文**

        定义神经元的充电差分方程。子类必须实现这个函数。

        ----

        .. _BaseNode.neuronal_charge-en:

        * **English**

        Define the charge difference equation. The sub-class must implement this function.
        """
        raise NotImplementedError

    def neuronal_fire(self):
        """
        **API Language:**
        :ref:`中文 <BaseNode.neuronal_fire-cn>` | :ref:`English <BaseNode.neuronal_fire-en>`

        ----

        .. _BaseNode.neuronal_fire-cn:

        * **中文**

        根据当前神经元的电压、阈值，计算输出脉冲。

        ----

        .. _BaseNode.neuronal_fire-en:

        * **English**

        Calculate out spikes of neurons by their current membrane potential and threshold voltage.
        """
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        """
        **API Language:**
        :ref:`中文 <BaseNode.neuronal_reset-cn>` | :ref:`English <BaseNode.neuronal_reset-en>`

        ----

        .. _BaseNode.neuronal_reset-cn:

        * **中文**

        根据当前神经元释放的脉冲，对膜电位进行重置。

        ----

        .. _BaseNode.neuronal_reset-en:

        * **English**

        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}"

    def single_step_forward(self, x: torch.Tensor):
        """
        **API Language:**
        :ref:`中文 <BaseNode.single_step_forward-cn>` | :ref:`English <BaseNode.single_step_forward-en>`

        ----

        .. _BaseNode.single_step_forward-cn:

        * **中文**

        按照充电、放电、重置的顺序进行前向传播。

        :param x: 输入到神经元的电压增量
        :type x: torch.Tensor

        :return: 神经元的输出脉冲
        :rtype: torch.Tensor

        ----

        .. _BaseNode.single_step_forward-en:

        * **English**

        Forward by the order of ``neuronal_charge``, ``neuronal_fire``, and ``neuronal_reset``.

        :param x: increment of voltage inputted to neurons
        :type x: torch.Tensor

        :return: out spikes of neurons
        :rtype: torch.Tensor
        """
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
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
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class NonSpikingBaseNode(nn.Module, base.MultiStepModule):
    def __init__(self, decode: Optional[str] = None):
        super().__init__()
        self.decode = decode

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(self, x_seq: torch.Tensor):
        self.v = torch.full_like(x_seq[0].data, fill_value=0.0)

        T = x_seq.shape[0]
        v_seq = []

        for t in range(T):
            self.neuronal_charge(x_seq[t])
            v_seq.append(self.v)

        if self.decode == "max-mem":
            return torch.max(torch.stack(v_seq, 0), 0).values
        elif self.decode == "max-abs-mem":
            v_stack = torch.stack(v_seq, 0)
            max_mem = torch.max(v_stack, 0).values
            min_mem = torch.min(v_stack, 0).values
            mem = max_mem * (max_mem.abs() > min_mem.abs()) + min_mem * (
                max_mem.abs() <= min_mem.abs()
            )
            return mem
        elif self.decode == "mean-mem":
            return torch.mean(torch.stack(v_seq, 0), 0)
        elif self.decode == "last_mem":
            return v_seq[-1]
        else:
            return v_seq
