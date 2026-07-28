from typing import Optional

import torch
import torch.nn as nn

from .. import functional, surrogate
from .base_node import BaseNode
from .lif import LIFNode

__all__ = ["MPBNBaseNode", "MPBNLIFNode"]


class MPBNBaseNode(BaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
        mpbn: bool = True,
        out_features=None,
        out_channels=None,
        learnable_vth: bool = False,
        bn_momentum: float = 0.1,
        bn_decay_momentum: float = 0.94,
        bn_min_momentum: float = 0.005,
    ):
        r"""
        **API Language** - :ref:`中文 <MPBNBaseNode.__init__-cn>` | :ref:`English <MPBNBaseNode.__init__-en>`

        ----

        .. _MPBNBaseNode.__init__-cn:

        * **中文**

        该基类神经元实现了 `Membrane Potential Batch Normalization for Spiking Neural Networks <https://arxiv.org/abs/2308.08359>`_ 中提出的膜电压批量归一化方法，并在 `Threshold Modulation for Online Test-Time Adaptation of Spiking Neural Networks <https://arxiv.org/abs/2505.05375>`_ 的基础上引入阈值调制模块，用于测试时适应任务并降低能耗。

        神经动力学方程如下：

        .. math::
            :nowrap:

            \begin{align*}
            H'[t] &= \mathbf{BN}(H[t]), & \text{（训练时）} \\
            (\tilde{V}_{th})_{i} &= \frac{(V_{th}-\beta_{i})\sqrt{\sigma_{i}^{2}}}{\gamma_{i}}+\mu_{i}, & \text{（测试时适应）}
            \end{align*}

        :param mpbn: 是否启用 MPBN
        :type mpbn: bool

        :param out_features: 特征维度，用于线性层后
        :type out_features: int

        :param out_channels: 特征通道数，用于 2D 卷积层后
        :type out_channels: int

        :param learnable_vth: 阈值是否可训练
        :type learnable_vth: bool

        :param bn_momentum: 阈值重参数化后，更新统计量时使用的动量
        :type bn_momentum: float

        :param bn_decay_momentum: 阈值重参数化后，更新统计量时使用的动量衰减
        :type bn_decay_momentum: float

        :param bn_min_momentum: 阈值重参数化后，更新统计量时使用的最小动量
        :type bn_min_momentum: float

        其余参数与 :class:`BaseNode` 相同。

        ----

        .. _MPBNBaseNode.__init__-en:

        * **English**

        Base class of neuron with membrane potential batch normalization proposed in `Membrane Potential Batch Normalization for Spiking Neural Networks <https://arxiv.org/abs/2308.08359>`_.
        `Threshold Modulation for Online Test-Time Adaptation of Spiking Neural Networks <https://arxiv.org/abs/2505.05375>`_ further introduces a Threshold Modulation module after threshold re-parameterization to enable test-time adaptation and reduce energy consumption.

        The neuronal dynamics are described as:

        .. math::
            :nowrap:

            \begin{align*}
            H'[t] &= \mathbf{BN}(H[t]), & \text{(training)} \\
            (\tilde{V}_{th})_{i} &= \frac{(V_{th}-\beta_{i})\sqrt{\sigma_{i}^{2}}}{\gamma_{i}}+\mu_{i}, & \text{(test-time adaptation)}
            \end{align*}

        :param mpbn: whether to enable MPBN
        :type mpbn: bool

        :param out_features: feature dimension, when used after `Linear`
        :type out_features: int

        :param out_channels: number of channels, when used after `Conv2d`
        :type out_channels: int

        :param learnable_vth: whether to train a (positive) threshold
        :type learnable_vth: bool

        :param bn_momentum: the momentum used in statistics update after threshold re-parameterization
        :type bn_momentum: float

        :param bn_decay_momentum: the momentum decay used in statistics update after threshold re-parameterization
        :type bn_decay_momentum: float

        :param bn_min_momentum: the minimum momentum used in statistics update after threshold re-parameterization
        :type bn_min_momentum: float

        Other parameters are the same as :class:`BaseNode`.
        """
        super().__init__(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )
        if (out_features is None) == (out_channels is None):
            raise ValueError("Specify exactly one of out_features or out_channels.")
        feature_count = out_channels if out_channels is not None else out_features
        if mpbn:
            self.vbn = (
                nn.LazyBatchNorm2d()
                if out_channels is not None
                else nn.LazyBatchNorm1d()
            )
        else:
            self.vbn = nn.Identity()

        self.register_buffer("mu", None)
        self.register_buffer("sigma2", None)
        self.gamma = None
        self.beta = None
        self.eps = None

        self.fold_bn = False
        self.normalize_residual = False
        self.running_stats = False

        self.bn_momentum = bn_momentum
        self.bn_decay_momentum = bn_decay_momentum
        self.bn_min_momentum = bn_min_momentum

        self.learnable_vth = learnable_vth
        if learnable_vth:  # force the threshold to be positive
            self.a = nn.Parameter(torch.zeros(feature_count))

    def compute_running_stats(
        self, v: torch.Tensor
    ):  # you can disable this completely by overiding it in subclasses
        if v.ndim not in (2, 4):
            raise NotImplementedError(
                f"Only 2D and 4D tensor are supported, but got {v.ndim}D tensor."
            )
        if v.ndim == 2 and v.shape[0] == 1:
            return

        reduce_dims = 0 if v.ndim == 2 else (0, 2, 3)
        mu = torch.mean(v, dim=reduce_dims).detach()
        sigma2 = torch.var(v, dim=reduce_dims, unbiased=True).detach()
        if self.running_stats and self.mu is not None:
            self.mu = self.mu.detach() * (1 - self.bn_momentum) + mu * self.bn_momentum
            self.sigma2 = (
                self.sigma2.detach() * (1 - self.bn_momentum)
                + sigma2 * self.bn_momentum
            )
            self.bn_momentum = max(
                self.bn_momentum * self.bn_decay_momentum, self.bn_min_momentum
            )
        else:
            self.mu = mu
            self.sigma2 = sigma2

    def pre_charge(self, x: torch.Tensor):
        raise NotImplementedError(
            "This method should be implemented in subclasses, e.g. the charging function of LIF neuron."
        )

    def neuronal_charge(self, x: torch.Tensor):
        self.pre_charge(x)
        self.v = self.vbn(self.v)
        if self.fold_bn and not self.learnable_vth and self.training:
            self.compute_running_stats(self.v)

    def neuronal_fire(self):
        if self.v.ndim not in (2, 4):
            raise NotImplementedError(
                f"Only 2D and 4D tensors are supported, but got {self.v.ndim}D tensors."
            )
        if self.fold_bn and not self.learnable_vth:
            threshold = (self.v_threshold - self.beta) * torch.sqrt(
                self.sigma2 + self.eps
            ) / self.gamma + self.mu
        elif self.learnable_vth:
            threshold = torch.exp(self.a)
        else:
            threshold = self.v_threshold
        threshold = torch.as_tensor(
            threshold, device=self.v.device, dtype=self.v.dtype
        ).expand(self.v.shape[1])
        spike, self.v = functional.mpbn_fire(
            self.v,
            threshold,
            self.surrogate_function,
            self.normalize_residual,
            self.gamma,
            self.mu,
            self.beta,
            self.sigma2,
            self.eps,
        )
        return spike

    def single_step_forward(self, x: torch.Tensor):
        """
        **API Language** - :ref:`中文 <MPBNBaseNode.single_step_forward-cn>` | :ref:`English <MPBNBaseNode.single_step_forward-en>`

        ----

        .. _MPBNBaseNode.single_step_forward-cn:

        * **中文**

        :param x: 当前时间步输入张量（2D 或 4D）
        :type x: torch.Tensor
        :return: 当前时间步输出脉冲
        :rtype: torch.Tensor
        :raises NotImplementedError: 当输入维度不是 2D 或 4D 时，内部放电逻辑会抛出异常

        ----

        .. _MPBNBaseNode.single_step_forward-en:

        * **English**

        :param x: Input tensor at current time step (2D or 4D)
        :type x: torch.Tensor
        :return: Output spike at current time step
        :rtype: torch.Tensor
        :raises NotImplementedError: Raised by internal firing logic when input rank is neither 2D nor 4D
        """
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def re_parameterize_v_threshold(
        self, normalize_residual: bool = False, running_stats: bool = False
    ):
        # "re-parameterize" threshold to enable TTA capability
        if isinstance(self.vbn, nn.Identity):
            return
        self.fold_bn = True
        if self.learnable_vth:  # if self.a is learned during training:
            self.v_threshold = torch.exp(self.a.detach())
            del self.a
            self.learnable_vth = False
        self.normalize_residual = normalize_residual
        self.running_stats = running_stats
        self.mu = self.vbn.running_mean
        self.sigma2 = self.vbn.running_var
        self.gamma = self.vbn.weight
        self.beta = self.vbn.bias
        self.eps = self.vbn.eps
        self.vbn = nn.Identity()


class MPBNLIFNode(MPBNBaseNode):
    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
        mpbn: bool = True,
        out_features=None,
        out_channels=None,
        learnable_vth: bool = False,
        bn_momentum: float = 0.1,
        bn_decay_momentum: float = 0.94,
        bn_min_momentum: float = 0.005,
    ):
        r"""
        **API Language** - :ref:`中文 <MPBNLIFNode.__init__-cn>` | :ref:`English <MPBNLIFNode.__init__-en>`

        ----

        .. _MPBNLIFNode.__init__-cn:

        * **中文**

        该神经元模型在 `Membrane Potential Batch Normalization for Spiking Neural Networks <https://arxiv.org/abs/2308.08359>`_ 中对膜电压进行了批量归一化，并在 `Threshold Modulation for Online Test-Time Adaptation of Spiking Neural Networks <https://arxiv.org/abs/2505.05375>`_ 的基础上引入阈值调制模块，用于测试时适应任务并降低能耗。

        神经动力学方程如下：

        .. math::
            :nowrap:

            \begin{align*}
            H'[t] &= \mathbf{BN}(H[t]), & \text{（训练时）} \\
            (\tilde{V}_{th})_{i} &= \frac{(V_{th}-\beta_{i})\sqrt{\sigma_{i}^{2}}}{\gamma_{i}}+\mu_{i}, & \text{（测试时适应）}
            \end{align*}

        :param tau: LIF中的时间常数
        :type tau: float

        :param decay_input: 输入是否参与衰减
        :type decay_input: bool

        其余参数与 :class:`MPBNBaseNode` 相同。

        ----

        .. _MPBNLIFNode.__init__-en:

        * **English**

        This neuron model applies membrane potential batch normalization as in `Membrane Potential Batch Normalization for Spiking Neural Networks <https://arxiv.org/abs/2308.08359>`_.
        `Threshold Modulation for Online Test-Time Adaptation of Spiking Neural Networks <https://arxiv.org/abs/2505.05375>`_ further introduces a Threshold Modulation module for test-time adaptation and energy efficiency.

        The neuronal dynamics are described as:

        .. math::
            :nowrap:

            \begin{align*}
            H'[t] &= \mathbf{BN}(H[t]), & \text{(training)} \\
            (\tilde{V}_{th})_{i} &= \frac{(V_{th}-\beta_{i})\sqrt{\sigma_{i}^{2}}}{\gamma_{i}}+\mu_{i}, & \text{(test-time adaptation)}
            \end{align*}

        :param tau: time constant in LIF
        :type tau: float

        :param decay_input: whether the input current is decayed
        :type decay_input: bool

        Other parameters are the same as :class:`MPBNBaseNode`.
        """
        assert isinstance(tau, float) and tau > 1.0
        super().__init__(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
            mpbn,
            out_features,
            out_channels,
            learnable_vth,
            bn_momentum,
            bn_decay_momentum,
            bn_min_momentum,
        )

        self.tau = tau
        self.decay_input = decay_input

    @property
    def supported_backends(self):
        return "torch"

    def pre_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.0:
                self.v = LIFNode.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = LIFNode.neuronal_charge_decay_input(
                    x, self.v, self.v_reset, self.tau
                )
        else:
            if self.v_reset is None or self.v_reset == 0.0:
                self.v = LIFNode.neuronal_charge_no_decay_input_reset0(
                    x, self.v, self.tau
                )
            else:
                self.v = LIFNode.neuronal_charge_no_decay_input(
                    x, self.v, self.v_reset, self.tau
                )
