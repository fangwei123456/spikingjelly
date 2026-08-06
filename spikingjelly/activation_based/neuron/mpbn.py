from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import functional, surrogate
from .base_node import BaseNode

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
                nn.BatchNorm2d(feature_count)
                if out_channels is not None
                else nn.BatchNorm1d(feature_count)
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

    def _normalization_state(self):
        return {
            "running_mean": (
                self.vbn.running_mean.clone()
                if not isinstance(self.vbn, nn.Identity)
                else None
            ),
            "running_var": (
                self.vbn.running_var.clone()
                if not isinstance(self.vbn, nn.Identity)
                else None
            ),
            "num_batches_tracked": (
                self.vbn.num_batches_tracked.clone()
                if not isinstance(self.vbn, nn.Identity)
                else None
            ),
            "mu": self.mu,
            "sigma2": self.sigma2,
            "bn_momentum": self.bn_momentum,
        }

    def forward(self, *args, **kwargs):
        stats = self._normalization_state()
        kwargs["_mpbn_stats"] = stats
        output = super().forward(*args, **kwargs)
        if self.training:
            if not isinstance(self.vbn, nn.Identity):
                self.vbn.running_mean.copy_(stats["running_mean"])
                self.vbn.running_var.copy_(stats["running_var"])
                self.vbn.num_batches_tracked.copy_(stats["num_batches_tracked"])
            elif self.fold_bn and not self.learnable_vth:
                self.mu = stats["mu"]
                self.sigma2 = stats["sigma2"]
                self.bn_momentum = stats["bn_momentum"]
        return output

    def multi_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        if "_mpbn_stats" not in kwargs:
            kwargs["_mpbn_stats"] = self._normalization_state()
        return super().multi_step_functional_forward(inputs, states, **kwargs)

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        v = states[0]
        stats = kwargs.get("_mpbn_stats")
        if stats is None:
            stats = self._normalization_state()
        v = functional.lif_charge(x, v, self.tau, self.decay_input, self.v_reset)

        if not isinstance(self.vbn, nn.Identity):
            running_mean = stats["running_mean"]
            running_var = stats["running_var"]
            batches_tracked = stats["num_batches_tracked"]
            exponential_average_factor = (
                0.0 if self.vbn.momentum is None else self.vbn.momentum
            )
            if self.training:
                batches_tracked.add_(1)
                if self.vbn.momentum is None:
                    exponential_average_factor = 1.0 / float(batches_tracked)
            v = F.batch_norm(
                v,
                running_mean,
                running_var,
                self.vbn.weight,
                self.vbn.bias,
                self.training,
                exponential_average_factor,
                self.vbn.eps,
            )

        mu = stats["mu"]
        sigma2 = stats["sigma2"]
        bn_momentum = stats["bn_momentum"]
        if (
            self.fold_bn
            and not self.learnable_vth
            and self.training
            and not (v.ndim == 2 and v.shape[0] == 1)
        ):
            reduce_dims = 0 if v.ndim == 2 else (0, 2, 3)
            batch_mu = torch.mean(v, dim=reduce_dims).detach()
            batch_sigma2 = torch.var(v, dim=reduce_dims, unbiased=True).detach()
            if self.running_stats:
                mu = mu.detach() * (1 - bn_momentum) + batch_mu * bn_momentum
                sigma2 = (
                    sigma2.detach() * (1 - bn_momentum) + batch_sigma2 * bn_momentum
                )
                bn_momentum = max(
                    bn_momentum * self.bn_decay_momentum,
                    self.bn_min_momentum,
                )
            else:
                mu = batch_mu
                sigma2 = batch_sigma2
            stats["mu"] = mu
            stats["sigma2"] = sigma2
            stats["bn_momentum"] = bn_momentum

        feature_shape = (1, -1) if v.ndim == 2 else (1, -1, 1, 1)
        if self.fold_bn and not self.learnable_vth:
            threshold = (self.v_threshold - self.beta) * torch.sqrt(
                sigma2 + self.eps
            ) / self.gamma + mu
        elif self.learnable_vth:
            threshold = torch.exp(self.a)
        else:
            threshold = self.v_threshold
        if self.fold_bn or self.learnable_vth:
            threshold = threshold.view(feature_shape)
        diff = v - threshold
        spike = self.surrogate_function(diff)

        if self.normalize_residual:
            mask = diff <= 0
            gamma = self.gamma.view(feature_shape).expand_as(mask)
            mu = mu.view(feature_shape).expand_as(mask)
            beta = self.beta.view(feature_shape).expand_as(mask)
            sigma = torch.sqrt(sigma2 + self.eps).view(feature_shape).expand_as(mask)
            normalized_residual = (v[mask] - mu[mask]) / sigma[mask] * gamma[
                mask
            ] + beta[mask]
            v = v.masked_scatter(mask, normalized_residual)

        v = functional.voltage_reset(
            v, spike, self.v_threshold, self.v_reset, self.detach_reset
        )
        return (spike,), (v, *states[1:])
