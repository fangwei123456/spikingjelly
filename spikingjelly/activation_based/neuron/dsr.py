import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .. import base


__all__ = ["DSRIFNode", "DSRLIFNode"]


class DSRIFNode(base.MemoryModule):
    def __init__(
        self,
        T: int = 20,
        v_threshold: float = 6.0,
        alpha: float = 0.5,
        v_threshold_training: bool = True,
        v_threshold_grad_scaling: float = 1.0,
        v_threshold_lower_bound: float = 0.01,
        step_mode="m",
        backend="torch",
        **kwargs,
    ):
        """
        **API Language:**
        :ref:`中文 <DSRIFNode.__init__-cn>` | :ref:`English <DSRIFNode.__init__-en>`

        ----

        .. _DSRIFNode.__init__-cn:

        * **中文**

        DSR IF 神经元，由
        `Training High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representation
        <https://arxiv.org/pdf/2205.00459.pdf>`_ 提出。
        该模型基于对脉冲表示的可微建模，用于低时延、高性能脉冲神经网络训练。

        :param T: 时间步数
        :type T: int

        :param v_threshold: 神经元阈值电压的初始值
        :type v_threshold: float

        :param alpha: 阈值电压的缩放因子
        :type alpha: float

        :param v_threshold_training: 是否将阈值电压设为可学习参数，默认为 ``True``
        :type v_threshold_training: bool

        :param v_threshold_grad_scaling: 对阈值电压梯度进行缩放的系数
        :type v_threshold_grad_scaling: float

        :param v_threshold_lower_bound: 训练过程中阈值电压允许的最小值
        :type v_threshold_lower_bound: float

        :param step_mode: 步进模式，仅支持 ``'m'`` （多步）
        :type step_mode: str

        :param backend: 使用的后端。不同 ``step_mode`` 支持的后端可能不同。
            可通过 ``self.supported_backends`` 查看当前步进模式支持的后端。
            DSR-IF 仅支持 ``'torch'`` 后端
        :type backend: str

        ----

        .. _DSRIFNode.__init__-en:

        * **English**

        DSR IF neuron, proposed in
        `Training High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representation
        <https://arxiv.org/pdf/2205.00459.pdf>`_.
        This model enables low-latency and high-performance SNN training via differentiable spike representations.

        :param T: number of time-steps
        :type T: int

        :param v_threshold: initial membrane potential threshold
        :type v_threshold: float

        :param alpha: scaling factor of the membrane potential threshold
        :type alpha: float

        :param v_threshold_training: whether the membrane potential threshold is learnable, default: ``True``
        :type v_threshold_training: bool

        :param v_threshold_grad_scaling: scaling factor applied to the gradient of the membrane potential threshold
        :type v_threshold_grad_scaling: float

        :param v_threshold_lower_bound: minimum allowable membrane potential threshold during training
        :type v_threshold_lower_bound: float

        :param step_mode: step mode, only `'m'` (multi-step) is supported
        :type step_mode: str

        :param backend: backend of this neuron layer. Supported backends depend on ``step_mode``.
            Users can print ``self.supported_backends`` to check availability.
            DSR-IF only supports the ``'torch'`` backend
        :type backend: str
        """
        assert isinstance(T, int) and T is not None
        assert isinstance(v_threshold, float) and v_threshold >= v_threshold_lower_bound
        assert isinstance(alpha, float) and alpha > 0.0 and alpha <= 1.0
        assert (
            isinstance(v_threshold_lower_bound, float) and v_threshold_lower_bound > 0.0
        )
        assert step_mode == "m"

        super().__init__()
        self.backend = backend
        self.step_mode = step_mode
        self.T = T
        if v_threshold_training:
            self.v_threshold = nn.Parameter(torch.tensor(v_threshold))
        else:
            self.v_threshold = torch.tensor(v_threshold)
        self.alpha = alpha
        self.v_threshold_lower_bound = v_threshold_lower_bound
        self.v_threshold_grad_scaling = v_threshold_grad_scaling

    @property
    def supported_backends(self):
        return "torch"

    def extra_repr(self):
        with torch.no_grad():
            T = self.T
            v_threshold = self.v_threshold
            alpha = self.alpha
            v_threshold_lower_bound = self.v_threshold_lower_bound
            v_threshold_grad_scaling = self.v_threshold_grad_scaling
        return (
            f", T={T}"
            + f", init_vth={v_threshold}"
            + f", alpha={alpha}"
            + f", vth_bound={v_threshold_lower_bound}"
            + f", vth_g_scale={v_threshold_grad_scaling}"
        )

    def multi_step_forward(self, x_seq: torch.Tensor):
        with torch.no_grad():
            self.v_threshold.copy_(
                F.relu(self.v_threshold - self.v_threshold_lower_bound)
                + self.v_threshold_lower_bound
            )
        iffunc = self.DSRIFFunction.apply
        y_seq = iffunc(
            x_seq, self.T, self.v_threshold, self.alpha, self.v_threshold_grad_scaling
        )
        return y_seq

    class DSRIFFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, inp, T=10, v_threshold=1.0, alpha=0.5, v_threshold_grad_scaling=1.0
        ):
            ctx.save_for_backward(inp)

            mem_potential = torch.zeros_like(inp[0]).to(inp.device)
            spikes = []

            for t in range(inp.size(0)):
                mem_potential = mem_potential + inp[t]
                spike = (
                    (mem_potential >= alpha * v_threshold).float() * v_threshold
                ).float()
                mem_potential = mem_potential - spike
                spikes.append(spike)
            output = torch.stack(spikes)

            ctx.T = T
            ctx.v_threshold = v_threshold
            ctx.v_threshold_grad_scaling = v_threshold_grad_scaling
            return output

        @staticmethod
        def backward(ctx, grad_output):
            with torch.no_grad():
                inp = ctx.saved_tensors[0]
                T = ctx.T
                v_threshold = ctx.v_threshold
                v_threshold_grad_scaling = ctx.v_threshold_grad_scaling

                input_rate_coding = torch.mean(inp, 0)
                grad_output_coding = torch.mean(grad_output, 0) * T

                input_grad = grad_output_coding.clone()
                input_grad[
                    (input_rate_coding < 0) | (input_rate_coding > v_threshold)
                ] = 0
                input_grad = torch.stack([input_grad for _ in range(T)]) / T

                v_threshold_grad = grad_output_coding.clone()
                v_threshold_grad[input_rate_coding <= v_threshold] = 0
                v_threshold_grad = (
                    torch.sum(v_threshold_grad) * v_threshold_grad_scaling
                )
                if v_threshold_grad.is_cuda and torch.cuda.device_count() != 1:
                    try:
                        dist.all_reduce(v_threshold_grad, op=dist.ReduceOp.SUM)
                    except:
                        raise RuntimeWarning(
                            "Something wrong with the `all_reduce` operation when summing up the gradient of v_threshold from multiple gpus. Better check the gpu status and try DistributedDataParallel."
                        )

                return input_grad, None, v_threshold_grad, None, None


class DSRLIFNode(base.MemoryModule):
    def __init__(
        self,
        T: int = 20,
        v_threshold: float = 1.0,
        tau: float = 2.0,
        delta_t: float = 0.05,
        alpha: float = 0.3,
        v_threshold_training: bool = True,
        v_threshold_grad_scaling: float = 1.0,
        v_threshold_lower_bound: float = 0.1,
        step_mode="m",
        backend="torch",
        **kwargs,
    ):
        """
        **API Language:**
        :ref:`中文 <DSRLIFNode.__init__-cn>` | :ref:`English <DSRLIFNode.__init__-en>`

        ----

        .. _DSRLIFNode.__init__-cn:

        * **中文**

        DSR LIF 神经元，由
        `Training High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representation
        <https://arxiv.org/pdf/2205.00459.pdf>`_ 提出。该模型通过对脉冲表示进行可微建模，实现低时延、高性能的脉冲神经网络训练。

        :param T: 时间步数
        :type T: int

        :param v_threshold: 神经元阈值电压的初始值
        :type v_threshold: float

        :param tau: 膜电位时间常数
        :type tau: float

        :param delta_t: 对连续时间 LIF 微分方程进行离散化的时间步长
        :type delta_t: float

        :param alpha: 阈值电压的缩放因子
        :type alpha: float

        :param v_threshold_training: 是否将阈值电压设为可学习参数，默认为 ``True``
        :type v_threshold_training: bool

        :param v_threshold_grad_scaling: 对阈值电压梯度进行缩放的系数
        :type v_threshold_grad_scaling: float

        :param v_threshold_lower_bound: 训练过程中阈值电压允许的最小值
        :type v_threshold_lower_bound: float

        :param step_mode: 步进模式，仅支持 ``'m'`` （多步）
        :type step_mode: str

        :param backend: 使用的后端。不同 ``step_mode`` 支持的后端可能不同。
            可通过 ``self.supported_backends`` 查看当前步进模式支持的后端。
            DSR-LIF 仅支持 ``'torch'`` 后端
        :type backend: str

        ----

        .. _DSRLIFNode.__init__-en:

        * **English**

        DSR LIF neuron, proposed in
        `Training High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representation
        <https://arxiv.org/pdf/2205.00459.pdf>`_.
        This model enables low-latency and high-performance SNN training by differentiating spike representations.

        :param T: number of time-steps
        :type T: int

        :param v_threshold: initial membrane potential threshold
        :type v_threshold: float

        :param tau: membrane time constant
        :type tau: float

        :param delta_t: discretization step for the continuous-time LIF differential equation
        :type delta_t: float

        :param alpha: scaling factor of the membrane potential threshold
        :type alpha: float

        :param v_threshold_training: whether the membrane potential threshold is learnable, default: ``True``
        :type v_threshold_training: bool

        :param v_threshold_grad_scaling: scaling factor applied to the gradient of the membrane potential threshold
        :type v_threshold_grad_scaling: float

        :param v_threshold_lower_bound: minimum allowable membrane potential threshold during training
        :type v_threshold_lower_bound: float

        :param step_mode: step mode, only `'m'` (multi-step) is supported
        :type step_mode: str

        :param backend: backend of this neuron layer. Supported backends depend on ``step_mode``.
            Users can print ``self.supported_backends`` to check availability.
            DSR-LIF only supports the ``'torch'`` backend
        :type backend: str
        """
        assert isinstance(T, int) and T is not None
        assert isinstance(v_threshold, float) and v_threshold >= v_threshold_lower_bound
        assert isinstance(alpha, float) and alpha > 0.0 and alpha <= 1.0
        assert (
            isinstance(v_threshold_lower_bound, float) and v_threshold_lower_bound > 0.0
        )
        assert step_mode == "m"

        super().__init__()
        self.backend = backend
        self.step_mode = step_mode
        self.T = T
        if v_threshold_training:
            self.v_threshold = nn.Parameter(torch.tensor(v_threshold))
        else:
            self.v_threshold = torch.tensor(v_threshold)
        self.tau = tau
        self.delta_t = delta_t
        self.alpha = alpha
        self.v_threshold_lower_bound = v_threshold_lower_bound
        self.v_threshold_grad_scaling = v_threshold_grad_scaling

    @property
    def supported_backends(self):
        return "torch"

    def extra_repr(self):
        with torch.no_grad():
            T = self.T
            v_threshold = self.v_threshold
            tau = self.tau
            delta_t = self.delta_t
            alpha = self.alpha
            v_threshold_lower_bound = self.v_threshold_lower_bound
            v_threshold_grad_scaling = self.v_threshold_grad_scaling
        return (
            f", T={T}"
            + f", init_vth={v_threshold}"
            + f", tau={tau}"
            + f", dt={delta_t}"
            + f", alpha={alpha}"
            + f", vth_bound={v_threshold_lower_bound}"
            + f", vth_g_scale={v_threshold_grad_scaling}"
        )

    def multi_step_forward(self, x_seq: torch.Tensor):
        with torch.no_grad():
            self.v_threshold.copy_(
                F.relu(self.v_threshold - self.v_threshold_lower_bound)
                + self.v_threshold_lower_bound
            )
        liffunc = self.DSRLIFFunction.apply
        y_seq = liffunc(
            x_seq,
            self.T,
            self.v_threshold,
            self.tau,
            self.delta_t,
            self.alpha,
            self.v_threshold_grad_scaling,
        )
        return y_seq

    @classmethod
    def weight_rate_spikes(cls, data, tau, delta_t):
        T = data.shape[0]
        chw = data.size()[2:]
        data_reshape = data.permute(list(range(1, len(chw) + 2)) + [0])
        weight = torch.tensor(
            [
                math.exp(-1 / tau * (delta_t * T - ii * delta_t))
                for ii in range(1, T + 1)
            ]
        ).to(data_reshape.device)
        return (weight * data_reshape).sum(dim=len(chw) + 1) / weight.sum()

    class DSRLIFFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            inp,
            T,
            v_threshold,
            tau,
            delta_t=0.05,
            alpha=0.3,
            v_threshold_grad_scaling=1.0,
        ):
            ctx.save_for_backward(inp)

            mem_potential = torch.zeros_like(inp[0]).to(inp.device)
            beta = math.exp(-delta_t / tau)

            spikes = []
            for t in range(inp.size(0)):
                mem_potential = beta * mem_potential + (1 - beta) * inp[t]
                spike = (
                    (mem_potential >= alpha * v_threshold).float() * v_threshold
                ).float()
                mem_potential = mem_potential - spike
                spikes.append(spike / delta_t)
            output = torch.stack(spikes)

            ctx.T = T
            ctx.v_threshold = v_threshold
            ctx.tau = tau
            ctx.delta_t = delta_t
            ctx.v_threshold_grad_scaling = v_threshold_grad_scaling
            return output

        @staticmethod
        def backward(ctx, grad_output):
            inp = ctx.saved_tensors[0]
            T = ctx.T
            v_threshold = ctx.v_threshold
            delta_t = ctx.delta_t
            tau = ctx.tau
            v_threshold_grad_scaling = ctx.v_threshold_grad_scaling

            input_rate_coding = DSRLIFNode.weight_rate_spikes(inp, tau, delta_t)
            grad_output_coding = (
                DSRLIFNode.weight_rate_spikes(grad_output, tau, delta_t) * T
            )

            indexes = (input_rate_coding > 0) & (
                input_rate_coding < v_threshold / delta_t * tau
            )
            input_grad = torch.zeros_like(grad_output_coding)
            input_grad[indexes] = grad_output_coding[indexes].clone() / tau
            input_grad = torch.stack([input_grad for _ in range(T)]) / T

            v_threshold_grad = grad_output_coding.clone()
            v_threshold_grad[input_rate_coding <= v_threshold / delta_t * tau] = 0
            v_threshold_grad = (
                torch.sum(v_threshold_grad) * delta_t * v_threshold_grad_scaling
            )
            if v_threshold_grad.is_cuda and torch.cuda.device_count() != 1:
                try:
                    dist.all_reduce(v_threshold_grad, op=dist.ReduceOp.SUM)
                except:
                    raise RuntimeWarning(
                        "Something wrong with the `all_reduce` operation when summing up the gradient of v_threshold from multiple gpus. Better check the gpu status and try DistributedDataParallel."
                    )

            return input_grad, None, v_threshold_grad, None, None, None, None
