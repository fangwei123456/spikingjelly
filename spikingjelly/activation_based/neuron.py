from abc import abstractmethod
from typing import Callable, overload
import torch
import torch.nn as nn
from . import surrogate, base, lava_exchange
from .. import configure
import math
import numpy as np
import logging
try:
    import cupy
    from . import neuron_kernel, cuda_utils
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.neuron: {e}')
    cupy = None
    neuron_kernel = None
    cuda_utils = None

try:
    import lava.lib.dl.slayer as slayer

except BaseException as e:
    logging.info(f'spikingjelly.activation_based.neuron: {e}')
    slayer = None

class BaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <BaseNode.__init__-en>`

        .. _BaseNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        可微分SNN神经元的基类神经元。

        * :ref:`中文API <BaseNode.__init__-cn>`

        .. _BaseNode.__init__-en:

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        This class is the base class of differentiable spiking neurons.
        """
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.register_memory('v_threshold', v_threshold)
        self.register_memory('v_reset', v_reset)

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def js_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def js_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        """
         * :ref:`API in English <BaseNode.neuronal_charge-en>`

        .. _BaseNode.neuronal_charge-cn:

        定义神经元的充电差分方程。子类必须实现这个函数。

        * :ref:`中文API <BaseNode.neuronal_charge-cn>`

        .. _BaseNode.neuronal_charge-en:


        Define the charge difference equation. The sub-class must implement this function.
        """
        raise NotImplementedError

    def neuronal_fire(self):
        """
        * :ref:`API in English <BaseNode.neuronal_fire-en>`

        .. _BaseNode.neuronal_fire-cn:

        根据当前神经元的电压、阈值，计算输出脉冲。

        * :ref:`中文API <BaseNode.neuronal_fire-cn>`

        .. _BaseNode.neuronal_fire-en:


        Calculate out spikes of neurons by their current membrane potential and threshold voltage.
        """

        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        """
        * :ref:`API in English <BaseNode.neuronal_reset-en>`

        .. _BaseNode.neuronal_reset-cn:

        根据当前神经元释放的脉冲，对膜电位进行重置。

        * :ref:`中文API <BaseNode.neuronal_reset-cn>`

        .. _BaseNode.neuronal_reset-en:


        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.js_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.js_hard_reset(self.v, spike_d, self.v_reset)

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}'

    def single_step_forward(self, x: torch.Tensor):
        """

        * :ref:`API in English <BaseNode.single_step_forward-en>`

        .. _BaseNode.single_step_forward-cn:

        :param x: 输入到神经元的电压增量
        :type x: torch.Tensor

        :return: 神经元的输出脉冲
        :rtype: torch.Tensor

        按照充电、放电、重置的顺序进行前向传播。

        * :ref:`中文API <BaseNode.single_step_forward-cn>`

        .. _BaseNode.single_step_forward-en:

        :param x: increment of voltage inputted to neurons
        :type x: torch.Tensor

        :return: out spikes of neurons
        :rtype: torch.Tensor

        Forward by the order of `neuronal_charge`, `neuronal_fire`, and `neuronal_reset`.

        """
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.zeros_like(x.data)
            if v_init != 0.:
                torch.fill_(self.v, v_init)

class AdaptiveBaseNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 v_rest: float = 0., w_rest: float = 0, tau_w: float = 2., a: float = 0., b: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        # b: jump amplitudes
        # a: subthreshold coupling
        assert isinstance(w_rest, float)
        assert isinstance(v_rest, float)
        assert isinstance(tau_w, float)
        assert isinstance(a, float)
        assert isinstance(b, float)

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.register_memory('w', w_rest)

        self.w_rest = w_rest
        self.v_rest = v_rest
        self.tau_w = tau_w
        self.a = a
        self.b = b

    @staticmethod
    @torch.jit.script
    def js_neuronal_adaptation(w: float, tau_w: float, a: float, b: float, v_rest: float, spike: torch.Tensor, v: torch.Tensor):
        return w + 1. / tau_w * (a * (v - v_rest) - w) + b * spike

    def neuronal_adaptation(self, spike):
        self.w = self.js_neuronal_adaptation(self.w, self.tau_w, self.a, self.b, self.v_rest, spike, self.v)

    def extra_repr(self):
        return super().extra_repr() + f', v_rest={self.v_rest}, w_rest={self.w_rest}, tau_w={self.tau_w}, a={self.a}, b={self.b}'

    def single_step_forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_adaptation(spike)
        self.neuronal_reset(spike)
        return spike

class IFNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False, single_step_cupy_fp32_inference=False):
        """
        * :ref:`API in English <IFNode.__init__-en>`

        .. _IFNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param cupy_fp32_inference: 若为 `True`，在 `eval` 模式下，使用float32，却在GPU上运行，并且 `cupy` 已经安装，则会自动使用 `cupy` 进行加速
        :type cupy_fp32_inference: bool

        Integrate-and-Fire 神经元模型，可以看作理想积分器，无输入时电压保持恒定，不会像LIF神经元那样衰减。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + X[t]

        * :ref:`中文API <IFNode.__init__-cn>`

        .. _IFNode.__init__-en:

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param cupy_fp32_inference: If `True`, if this module is in `eval` mode, using float32, running on GPU, and `cupy` is installed, then this
            module will use `cupy` to accelerate
        :type cupy_fp32_inference: bool

        The Integrate-and-Fire neuron, which can be seen as a ideal integrator. The voltage of the IF neuron will not decay
        as that of the LIF neuron. The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + X[t]

        """
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.single_step_cupy_fp32_inference = single_step_cupy_fp32_inference

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', )
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = neuron_kernel.MultiStepIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.v_threshold, self.v_reset, self.detach_reset,
                self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)

    def single_step_forward(self, x: torch.Tensor):
        if self.single_step_cupy_fp32_inference and cupy is not None and not self.training and x.dtype == torch.float32 and x.get_device() >= 0:
            return self.single_step_cupy_fp32_inference_forward(x)
        else:
            return super().single_step_forward(x)

    def single_step_cupy_fp32_inference_forward(self, x: torch.Tensor):
        # cupy is installed && eval mode && fp32
        device_id = x.get_device()
        # use cupy to accelerate
        if isinstance(self.v, float):
            v = torch.zeros_like(x)
            if self.v != 0.:
                torch.fill_(v, self.v)
            self.v = v

        if self.v_reset is None:
            hard_reset = False
        else:
            hard_reset = True

        code = rf'''
            extern "C" __global__
            void IFNode_{'hard' if hard_reset else 'soft'}_reset_inference_forward(
            const float * x, const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
            float * spike, float * v,
            const int & numel)
        '''

        code += r'''
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < numel)
                {
                    v[index] += x[index];
                    spike[index] = (float) (v[index] >= v_threshold);
        '''

        code += rf'''
                    {'v[index] = (1.0f - spike[index]) * v[index] + spike[index] * v_reset;' if hard_reset else 'v[index] -= spike[index] * v_threshold;'}
        '''

        code += r'''
                }
            }
        '''
        if hasattr(self, 'cp_kernel'):
            if self.cp_kernel.code != code:
                # replace codes
                del self.cp_kernel
                self.cp_kernel = cupy.RawKernel(code, f"IFNode_{'hard' if hard_reset else 'soft'}_reset_inference_forward", options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)
        else:
            self.cp_kernel = cupy.RawKernel(code, f"IFNode_{'hard' if hard_reset else 'soft'}_reset_inference_forward", options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)

        with cuda_utils.DeviceEnvironment(device_id):
            numel = x.numel()
            threads = configure.cuda_threads
            blocks = cuda_utils.cal_blocks(numel)
            cp_numel = cupy.asarray(numel)
            cp_v_threshold = cupy.asarray(self.v_threshold, dtype=np.float32)
            if hard_reset:
                cp_v_reset = cupy.asarray(self.v_reset, dtype=np.float32)

            spike = torch.zeros_like(x)
            if hard_reset:
                x, cp_v_threshold, cp_v_reset, spike, self.v, cp_numel = cuda_utils.get_contiguous(x, cp_v_threshold, cp_v_reset, spike, self.v, cp_numel)
                kernel_args = [x, cp_v_threshold, cp_v_reset, spike, self.v, cp_numel]
            else:
                x, cp_v_threshold, spike, self.v, cp_numel = cuda_utils.get_contiguous(x, cp_v_threshold, spike, self.v, cp_numel)
                kernel_args = [x, cp_v_threshold, spike, self.v, cp_numel]
            self.cp_kernel(
                (blocks,), (threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device_id,
                    *kernel_args
                )
            )
            return spike

class LIFNode(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False, single_step_cupy_fp32_inference=False):
        """
        * :ref:`API in English <LIFNode.__init__-en>`

        .. _LIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param decay_input: 输入是否会衰减
        :type decay_input: bool

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param cupy_fp32_inference: 若为 `True`，在 `eval` 模式下，使用float32，却在GPU上运行，并且 `cupy` 已经安装，则会自动使用 `cupy` 进行加速
        :type cupy_fp32_inference: bool

        Leaky Integrate-and-Fire 神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        若 ``decay_input == True``:

            .. math::
                V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        若 ``decay_input == False``:

            .. math::
                V[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        .. tip::

            在 `eval` 模式下，使用float32，却在GPU上运行，并且 `cupy` 已经安装，则会自动使用 `cupy` 进行加速。

        * :ref:`中文API <LIFNode.__init__-cn>`

        .. _LIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param cupy_fp32_inference: If `True`, if this module is in `eval` mode, using float32, running on GPU, and `cupy` is installed, then this
            module will use `cupy` to accelerate
        :type cupy_fp32_inference: bool

        The Leaky Integrate-and-Fire neuron, which can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        IF ``decay_input == True``:

            .. math::
                V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        IF ``decay_input == False``:

            .. math::
                V[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        .. admonition:: Tip
            :class: tip

            If this module is in `eval` mode, using float32, running on GPU, and `cupy` is installed, then this
            module will use `cupy` to accelerate.

        """
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.single_step_cupy_fp32_inference = single_step_cupy_fp32_inference

        self.tau = tau
        self.decay_input = decay_input

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(v: torch.Tensor, x: torch.Tensor, tau: float):
        v = v + (x - v) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(v: torch.Tensor, x: torch.Tensor, tau: float, v_reset: float):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input_reset0(v: torch.Tensor, x: torch.Tensor, tau: float):
        v = v * (1. - 1. / tau) + x
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input(v: torch.Tensor, x: torch.Tensor, tau: float, v_reset: float):
        v = v - (v - v_reset) / tau + x
        return v

    def neuronal_charge(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_decay_input_reset0(self.v, x, self.tau)
            else:
                self.v = self.neuronal_charge_decay_input(self.v, x, self.tau, self.v_reset)

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_no_decay_input_reset0(self.v, x, self.tau)
            else:
                self.v = self.neuronal_charge_no_decay_input(self.v, x, self.tau, self.v_reset)

    def single_step_cupy_fp32_inference_forward(self, x: torch.Tensor):
        # cupy is installed && eval mode && fp32
        device_id = x.get_device()

        # use cupy to accelerate
        if isinstance(self.v, float):
            v = torch.zeros_like(x)
            if self.v != 0.:
                torch.fill_(v, self.v)
            self.v = v

        if self.v_reset is None:
            hard_reset = False
        else:
            hard_reset = True

        code = rf'''
            extern "C" __global__
            void LIFNode_{'hard' if hard_reset else 'soft'}_reset_decayInput_{self.decay_input}_inference_forward(
            const float * x, const float & v_threshold, {'const float & v_reset,' if hard_reset else ''} const float & tau,
            float * spike, float * v,
            const int & numel)
        '''

        code += r'''
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < numel)
                {
                    
        '''

        if self.decay_input:
            if hard_reset:
                code += r'''
                            v[index] += (x[index] - (v[index] - v_reset)) / tau;
                        '''
            else:
                code += r'''
                            v[index] += (x[index] - v[index]) / tau;
                '''
        else:
            if hard_reset:
                code += r'''
                            v[index] = x[index] + v[index] - (v[index] - v_reset) / tau;
                        '''
            else:
                code += r'''
                            v[index] = x[index] + v[index] * (1.0f - 1.0f / tau);
                '''

        code += rf'''
                    spike[index] = (float) (v[index] >= v_threshold);
                    {'v[index] = (1.0f - spike[index]) * v[index] + spike[index] * v_reset;' if hard_reset else 'v[index] -= spike[index] * v_threshold;'}
        '''

        code += r'''
                }
            }
        '''
        if hasattr(self, 'cp_kernel'):
            if self.cp_kernel.code != code:
                # replace codes
                del self.cp_kernel
                self.cp_kernel = cupy.RawKernel(code, f"LIFNode_{'hard' if hard_reset else 'soft'}_reset_decayInput_{self.decay_input}_inference_forward", options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)
        else:
            self.cp_kernel = cupy.RawKernel(code, f"LIFNode_{'hard' if hard_reset else 'soft'}_reset_decayInput_{self.decay_input}_inference_forward", options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)

        with cuda_utils.DeviceEnvironment(device_id):
            numel = x.numel()
            threads = configure.cuda_threads
            blocks = cuda_utils.cal_blocks(numel)
            cp_numel = cupy.asarray(numel)
            cp_v_threshold = cupy.asarray(self.v_threshold, dtype=np.float32)
            if hard_reset:
                cp_v_reset = cupy.asarray(self.v_reset, dtype=np.float32)
            cp_tau = cupy.asarray(self.tau, dtype=np.float32)
            spike = torch.zeros_like(x)
            if hard_reset:
                x, cp_v_threshold, cp_v_reset, cp_tau, spike, self.v, cp_numel = cuda_utils.get_contiguous(x, cp_v_threshold, cp_v_reset, cp_tau, spike, self.v, cp_numel)
                kernel_args = [x, cp_v_threshold, cp_v_reset, cp_tau, spike, self.v, cp_numel]
            else:
                x, cp_v_threshold, cp_tau, spike, self.v, cp_numel = cuda_utils.get_contiguous(x, cp_v_threshold, cp_tau, spike, self.v, cp_numel)
                kernel_args = [x, cp_v_threshold, cp_tau, spike, self.v, cp_numel]

            self.cp_kernel(
                (blocks,), (threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device_id,
                    *kernel_args
                )
            )
            return spike

    def single_step_forward(self, x: torch.Tensor):
        if self.single_step_cupy_fp32_inference and cupy is not None and not self.training and x.dtype == torch.float32 and x.get_device() >= 0:
            return self.single_step_cupy_fp32_inference_forward(x)
        else:
            return super().single_step_forward(x)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = neuron_kernel.MultiStepLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.decay_input, self.tau, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)

class ParametricLIFNode(BaseNode):
    def __init__(self, init_tau: float = 2.0, decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <ParametricLIFNode.__init__-en>`

        .. _ParametricLIFNode.__init__-cn:

        :param init_tau: 膜电位时间常数的初始值
        :type init_tau: float

        :param decay_input: 输入是否会衰减
        :type decay_input: bool

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_
        提出的 Parametric Leaky Integrate-and-Fire (PLIF)神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        若 ``decay_input == True``:

            .. math::
                V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        若 ``decay_input == False``:

            .. math::
                V[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        其中 :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`，:math:`w` 是可学习的参数。

        * :ref:`中文API <ParametricLIFNode.__init__-cn>`

        .. _ParametricLIFNode.__init__-en:

        :param init_tau: the initial value of membrane time constant
        :type init_tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        The Parametric Leaky Integrate-and-Fire (PLIF) neuron, which is proposed by `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_ and can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        IF ``decay_input == True``:

            .. math::
                V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        IF ``decay_input == False``:

            .. math::
                V[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        where :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`, :math:`w` is a learnable parameter.
        """

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.decay_input = decay_input
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', )
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - self.w.sigmoid()) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * self.w.sigmoid() + x

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = neuron_kernel.MultiStepParametricLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.w.sigmoid(), self.decay_input, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)

class QIFNode(BaseNode):
    def __init__(self, tau: float = 2., v_c: float = 0.8, a0: float = 1., v_threshold: float = 1., v_rest: float = 0., v_reset: float = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <QIFNode.__init__-en>`

        .. _QIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param v_c: 关键电压
        :type v_c: float

        :param a0: 
        :type a0: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_rest: 静息电位
        :type v_rest: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool


        Quadratic Integrate-and-Fire 神经元模型，一种非线性积分发放神经元模型，也是指数积分发放神经元(Exponential Integrate-and-Fire)的近似版本。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] + a_0 (V[t-1] - V_{rest})(V[t-1] - V_c))

        * :ref:`中文API <QIFNode.__init__-cn>`

        .. _QIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param v_c: critical voltage
        :type v_c: float

        :param a0: 
        :type a0: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_rest: resting potential
        :type v_rest: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        The Quadratic Integrate-and-Fire neuron is a kind of nonlinear integrate-and-fire models and also an approximation of the Exponential Integrate-and-Fire model.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] + a_0 (V[t-1] - V_{rest})(V[t-1] - V_c))
        """
                 
        assert isinstance(tau, float) and tau > 1.
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert a0 > 0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.tau = tau
        self.v_c = v_c
        self.v_rest = v_rest
        self.a0 = a0

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, v_c={self.v_c}, a0={self.a0}, v_rest={self.v_rest}'

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x + self.a0 * (self.v - self.v_rest) * (self.v - self.v_c)) / self.tau

class EIFNode(BaseNode):
    def __init__(self, tau: float = 2., delta_T: float = 1., theta_rh: float = .8, v_threshold: float = 1., v_rest: float = 0., v_reset: float = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <EIFNode.__init__-en>`

        .. _EIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param delta_T: 陡峭度参数
        :type delta_T: float

        :param theta_rh: 基强度电压阈值
        :type theta_rh: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_rest: 静息电位
        :type v_rest: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool


        Exponential Integrate-and-Fire 神经元模型，一种非线性积分发放神经元模型，是由HH神经元模型(Hodgkin-Huxley model)简化后推导出的一维模型。在 :math:`\\Delta_T\\to 0` 时退化为LIF模型。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}\\left(X[t] - (V[t-1] - V_{rest}) + \\Delta_T\\exp\\left(\\frac{V[t-1] - \\theta_{rh}}{\\Delta_T}\\right)\\right)

        * :ref:`中文API <EIFNode.__init__-cn>`

        .. _EIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param delta_T: sharpness parameter
        :type delta_T: float

        :param theta_rh: rheobase threshold
        :type theta_rh: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_rest: resting potential
        :type v_rest: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        The Exponential Integrate-and-Fire neuron is a kind of nonlinear integrate-and-fire models and also an one-dimensional model derived from the Hodgkin-Huxley model. It degenerates to the LIF model when :math:`\\Delta_T\\to 0`.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}\\left(X[t] - (V[t-1] - V_{rest}) + \\Delta_T\\exp\\left(\\frac{V[t-1] - \\theta_{rh}}{\\Delta_T}\\right)\\right)
        """
                 
        assert isinstance(tau, float) and tau > 1.
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert delta_T > 0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.tau = tau
        self.delta_T = delta_T
        self.v_rest = v_rest
        self.theta_rh = theta_rh

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, delta_T={self.delta_T}, theta_rh={self.theta_rh}'

    def neuronal_charge(self, x: torch.Tensor):
        with torch.no_grad():
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.as_tensor(self.v, device=x.device)
        
        self.v = self.v + (x + self.v_rest - self.v + self.delta_T * torch.exp((self.v - self.theta_rh) / self.delta_T)) / self.tau

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', )
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = neuron_kernel.MultiStepEIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.tau, self.v_threshold, self.v_reset, self.v_rest,
                self.theta_rh, self.delta_T, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = self.v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)

class LIAFNode(LIFNode):
    def __init__(self, act: Callable, threshold_related: bool, *args, **kwargs):
        """
        :param act: the activation function
        :type act: Callable
        :param threshold_related: whether the neuron uses threshold related (TR mode). If true, `y = act(h - v_th)`,
            otherwise `y = act(h)`
        :type threshold_related: bool

        Other parameters in `*args, **kwargs` are same with :class:`LIFNode`.

        The LIAF neuron proposed in `LIAF-Net: Leaky Integrate and Analog Fire Network for Lightweight and Efficient Spatiotemporal Information Processing <https://arxiv.org/abs/2011.06176>`_.

        .. admonition:: Warning
            :class: warning

            The outputs of this neuron are not binary spikes.

        """
        super().__init__(*args, **kwargs)
        self.act = act
        self.threshold_related = threshold_related

    @property
    def supported_backends(self):
        return ('torch', )

    def single_step_forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        if self.threshold_related:
            y = self.act(self.v - self.v_threshold)
        else:
            y = self.act(self.v)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return y


