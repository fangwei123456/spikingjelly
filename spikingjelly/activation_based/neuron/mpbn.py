from typing import Callable, Optional

import torch
import torch.nn as nn

from .. import surrogate
from .base_node import BaseNode


__all__ = ["MPBNBaseNode", "MPBNLIFNode"]


class MPBNBaseNode(BaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
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
        * :ref:`API in English <MPBNBaseNode.__init__-en>`

        ----

        .. _MPBNBaseNode.__init__-cn:

        * **中文**

        该基类神经元实现了 `Membrane Potential Batch Normalization for Spiking Neural Networks <https://arxiv.org/abs/2308.08359>`_ 中提出的膜电压批量归一化方法，并在 `Threshold Modulation for Online Test-Time Adaptation of Spiking Neural Networks <https://arxiv.org/abs/2505.05375>`_ 的基础上引入阈值调制模块，用于测试时适应任务并降低能耗。

        神经动力学方程如下：

        .. math::
            H'[t] &= \mathbf{BN}(H[t]), & \text{（训练时）} \\
            (\tilde{V}_{th})_{i} &= \frac{(V_{th}-\beta_{i})\sqrt{\sigma_{i}^{2}}}{\gamma_{i}}+\mu_{i}, & \text{（测试时适应）}

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
            H'[t] &= \mathbf{BN}(H[t]), & \text{(training)} \\
            (\tilde{V}_{th})_{i} &= \frac{(V_{th}-\beta_{i})\sqrt{\sigma_{i}^{2}}}{\gamma_{i}}+\mu_{i}, & \text{(test-time adaptation)}

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
        assert (
            out_features is None
            and out_channels is not None
            or out_features is not None
            and out_channels is None
        ), "One of out_features or out_channels should be specified."
        self.out_features = out_features
        self.out_channels = out_channels
        self.mpbn = mpbn
        if mpbn:
            if out_features is None and out_channels is not None:
                self.vbn = nn.LazyBatchNorm2d()
            else:
                self.vbn = nn.LazyBatchNorm1d()
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
            self.a = nn.Parameter(torch.full((out_channels or out_features,), 0.0))

        self.register_memory("vth", v_threshold)

    def init_vth(self, x: torch.Tensor):
        if isinstance(self.vth, float):
            if isinstance(self.v_threshold, float):
                self.vth = torch.full(
                    (x.shape[1],), self.v_threshold, device=x.device, dtype=x.dtype
                )
            else:
                self.vth = self.v_threshold
            self.vth_ = self.vth

    def compute_running_stats(
        self, v: torch.Tensor
    ):  # you can disable this completely by overiding it in subclasses
        if v.ndim == 2:
            if v.shape[0] == 1:
                return
            mu = torch.mean(v, dim=0).detach()
            sigma2 = torch.var(v, dim=0, unbiased=True).detach()
            if self.running_stats:
                if self.mu is None or self.sigma2 is None:
                    self.mu = mu
                    self.sigma2 = sigma2
                else:
                    self.mu = (
                        self.mu.detach() * (1 - self.bn_momentum)
                        + mu * self.bn_momentum
                    )
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
        elif v.ndim == 4:
            mu = torch.mean(v, dim=(0, 2, 3)).detach()
            sigma2 = torch.var(v, dim=(0, 2, 3), unbiased=True).detach()
            if self.running_stats:
                if self.mu is None or self.sigma2 is None:
                    self.mu = mu
                    self.sigma2 = sigma2
                else:
                    self.mu = (
                        self.mu.detach() * (1 - self.bn_momentum)
                        + mu * self.bn_momentum
                    )
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
        else:
            raise NotImplementedError(
                f"Only 2D and 4D tensor are supported, but got {v.ndim}D tensor."
            )

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
        if self.v.ndim == 2:
            if self.fold_bn and not self.learnable_vth:
                self.vth = (self.vth_ - self.beta) * torch.sqrt(
                    self.sigma2 + self.eps
                ) / self.gamma + self.mu
            if self.learnable_vth:
                self.vth = torch.exp(self.a)
            diff = self.v - self.vth.view(1, self.vth.shape[0])
            spike = self.surrogate_function(diff)
            if self.normalize_residual:
                mask = diff <= 0
                gamma = self.gamma.unsqueeze(0).expand_as(mask)
                mu = self.mu.unsqueeze(0).expand_as(mask)
                beta = self.beta.unsqueeze(0).expand_as(mask)
                sigma = torch.sqrt(self.sigma2 + self.eps).unsqueeze(0).expand_as(mask)
                normalized_residual = (self.v[mask] - mu[mask]) / sigma[mask] * gamma[
                    mask
                ] + beta[mask]
                self.v.masked_scatter_(mask, normalized_residual)
        elif self.v.ndim == 4:
            if self.fold_bn and not self.learnable_vth:
                self.vth = (self.vth_ - self.beta) * torch.sqrt(
                    self.sigma2 + self.eps
                ) / self.gamma + self.mu
            if self.learnable_vth:
                self.vth = torch.exp(self.a)
            diff = self.v - self.vth.view(1, self.vth.shape[0], 1, 1)
            spike = self.surrogate_function(diff)
            if self.normalize_residual:
                mask = diff <= 0
                gamma = self.gamma.view(1, -1, 1, 1).expand_as(mask)
                mu = self.mu.view(1, -1, 1, 1).expand_as(mask)
                beta = self.beta.view(1, -1, 1, 1).expand_as(mask)
                sigma = (
                    torch.sqrt(self.sigma2 + self.eps).view(1, -1, 1, 1).expand_as(mask)
                )
                normalized_residual = (self.v[mask] - mu[mask]) / sigma[mask] * gamma[
                    mask
                ] + beta[mask]
                self.v.masked_scatter_(mask, normalized_residual)
        else:
            raise NotImplementedError(
                f"Only 2D and 4D tensors are supported, but got {self.v.ndim}D tensors."
            )
        return spike

    def single_step_forward(self, x: torch.Tensor):
        self.init_vth(x)
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
            if self.fold_bn == True:
                print(
                    f"Re-parameterization has already been done in this neuron, skipping..."
                )
            else:
                print(f"MPBN is not enabled in this neuron, skipping...")
            return
        self.fold_bn = True
        if self.learnable_vth:  # if self.a is learned during training:
            with torch.no_grad():
                self.v_threshold = torch.exp(self.a)
            self.learnable_vth = False
        self.normalize_residual = normalize_residual
        self.running_stats = running_stats
        self.mu = self.vbn.running_mean
        self.sigma2 = self.vbn.running_var
        self.gamma = nn.Parameter(self.vbn.weight)
        self.beta = nn.Parameter(self.vbn.bias)
        self.eps = self.vbn.eps
        self.vbn = nn.Identity()


class MPBNLIFNode(MPBNBaseNode):
    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
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
        * :ref:`API in English <MPBNLIFNode.__init__-en>`

        ----

        .. _MPBNLIFNode.__init__-cn:

        * **中文**

        该神经元模型在 `Membrane Potential Batch Normalization for Spiking Neural Networks <https://arxiv.org/abs/2308.08359>`_ 中对膜电压进行了批量归一化，并在 `Threshold Modulation for Online Test-Time Adaptation of Spiking Neural Networks <https://arxiv.org/abs/2505.05375>`_ 的基础上引入阈值调制模块，用于测试时适应任务并降低能耗。

        神经动力学方程如下：

        .. math::
            H'[t] &= \mathbf{BN}(H[t]), & \text{（训练时）} \\
            (\tilde{V}_{th})_{i} &= \frac{(V_{th}-\beta_{i})\sqrt{\sigma_{i}^{2}}}{\gamma_{i}}+\mu_{i}, & \text{（测试时适应）}

        :param tau: LIF中的时间常数
        :type tau: float

        :param decay_input: 输入是否参与衰减
        :type decay_input: bool

        其余参数与 :class:`MPBNBaseNode <spikingjelly.activation_based.neuron.base_node.MPBNBaseNode>` 相同。

        ----

        .. _MPBNLIFNode.__init__-en:

        * **English**

        This neuron model applies membrane potential batch normalization as in `Membrane Potential Batch Normalization for Spiking Neural Networks <https://arxiv.org/abs/2308.08359>`_.
        `Threshold Modulation for Online Test-Time Adaptation of Spiking Neural Networks <https://arxiv.org/abs/2505.05375>`_ further introduces a Threshold Modulation module for test-time adaptation and energy efficiency.

        The neuronal dynamics are described as:

        .. math::
            H'[t] &= \mathbf{BN}(H[t]), & \text{(training)} \\
            (\tilde{V}_{th})_{i} &= \frac{(V_{th}-\beta_{i})\sqrt{\sigma_{i}^{2}}}{\gamma_{i}}+\mu_{i}, & \text{(test-time adaptation)}

        :param tau: time constant in LIF
        :type tau: float

        :param decay_input: whether the input current is decayed
        :type decay_input: bool

        Other parameters are the same as :class:`MPBNBaseNode <spikingjelly.activation_based.neuron.base_node.MPBNBaseNode>`.
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
                self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_decay_input(
                    x, self.v, self.v_reset, self.tau
                )
        else:
            if self.v_reset is None or self.v_reset == 0.0:
                self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_no_decay_input(
                    x, self.v, self.v_reset, self.tau
                )

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(
        x: torch.Tensor, v: torch.Tensor, tau: float
    ):
        v = v + (x - v) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(
        x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float
    ):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input_reset0(
        x: torch.Tensor, v: torch.Tensor, tau: float
    ):
        v = v * (1.0 - 1.0 / tau) + x
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input(
        x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float
    ):
        v = v - (v - v_reset) / tau + x
        return v
