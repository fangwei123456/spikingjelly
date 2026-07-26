import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Tempotron"]


class Tempotron(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        T: int,
        tau: float = 15.0,
        tau_s: float = 15.0 / 4,
        v_threshold: float = 1.0,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <Tempotron.__init__-cn>` | :ref:`English <Tempotron.__init__-en>`

        ----

        .. _Tempotron.__init__-cn:

        * **中文**

        实现 Tempotron 脉冲时序分类神经元。输入中的每个值表示对应输入神经元
        的脉冲时刻，负值表示未发放。

        :param in_features: 输入神经元数量
        :type in_features: int
        :param out_features: 输出神经元数量
        :type out_features: int
        :param T: 仿真时间窗口
        :type T: int
        :param tau: 膜电位时间常数
        :type tau: float
        :param tau_s: 突触电流时间常数
        :type tau_s: float
        :param v_threshold: 发放阈值
        :type v_threshold: float
        :raises ValueError: 数量、时间常数或阈值非正，或两个时间常数相等

        ----

        .. _Tempotron.__init__-en:

        * **English**

        Implement a Tempotron neuron for spike-timing classification. Each input
        value is the spike time of one input neuron; a negative value means that
        neuron did not spike.

        :param in_features: Number of input neurons
        :type in_features: int
        :param out_features: Number of output neurons
        :type out_features: int
        :param T: Simulation time window
        :type T: int
        :param tau: Membrane-potential time constant
        :type tau: float
        :param tau_s: Synaptic-current time constant
        :type tau_s: float
        :param v_threshold: Firing threshold
        :type v_threshold: float
        :raises ValueError: If a size, time constant, or threshold is not positive,
            or if the two time constants are equal
        """
        super().__init__()
        if in_features <= 0 or out_features <= 0 or T <= 0:
            raise ValueError("in_features, out_features, and T must be positive.")
        if tau <= 0 or tau_s <= 0 or v_threshold <= 0 or tau == tau_s:
            raise ValueError(
                "tau and tau_s must be positive and different; v_threshold must be positive."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.T = T
        self.tau = tau
        self.tau_s = tau_s
        self.v_threshold = v_threshold
        # Preserve the public checkpoint key ``model.summation_layer.weight``.
        self.model = nn.Module()
        self.model.summation_layer = nn.Linear(in_features, out_features, bias=False)

        t_max = tau * tau_s * math.log(tau / tau_s) / (tau - tau_s)
        self.v0 = v_threshold / (math.exp(-t_max / tau) - math.exp(-t_max / tau_s))

    @staticmethod
    def _psp_kernel(t: torch.Tensor, tau: float, tau_s: float) -> torch.Tensor:
        t = t.clamp_min(0)
        return torch.exp(-t / tau) - torch.exp(-t / tau_s)

    def mse_loss(self, v_max: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <Tempotron.mse_loss-cn>` | :ref:`English <Tempotron.mse_loss-en>`

        ----

        .. _Tempotron.mse_loss-cn:

        * **中文**

        :param v_max: 各输出神经元在时间窗口内的最大电压，形状为
            ``[batch_size, out_features]``
        :type v_max: torch.Tensor
        :param label: 分类标签，形状为 ``[batch_size]``
        :type label: torch.Tensor
        :return: 仅计算错误发放神经元的均方误差
        :rtype: torch.Tensor

        ----

        .. _Tempotron.mse_loss-en:

        * **English**

        :param v_max: Maximum voltage of each output neuron with shape
            ``[batch_size, out_features]``
        :type v_max: torch.Tensor
        :param label: Class labels with shape ``[batch_size]``
        :type label: torch.Tensor
        :return: Mean squared error over incorrectly firing neurons
        :rtype: torch.Tensor
        """
        wrong = (
            (v_max >= self.v_threshold).to(v_max.dtype)
            != F.one_hot(label, self.out_features)
        ).to(v_max.dtype)
        return ((v_max - self.v_threshold) * wrong).square().sum() / label.shape[0]

    def forward(self, in_spikes: torch.Tensor, ret_type: str) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <Tempotron.forward-cn>` | :ref:`English <Tempotron.forward-en>`

        ----

        .. _Tempotron.forward-cn:

        * **中文**

        :param in_spikes: 输入脉冲时刻，形状为 ``[batch_size, in_features]``
        :type in_spikes: torch.Tensor
        :param ret_type: ``"v"``、``"v_max"`` 或 ``"spikes"``
        :type ret_type: str
        :return: 完整电压轨迹、最大电压或输出脉冲时刻
        :rtype: torch.Tensor
        :raises ValueError: ``ret_type`` 不受支持

        ----

        .. _Tempotron.forward-en:

        * **English**

        :param in_spikes: Input spike times with shape
            ``[batch_size, in_features]``
        :type in_spikes: torch.Tensor
        :param ret_type: ``"v"``, ``"v_max"``, or ``"spikes"``
        :type ret_type: str
        :return: Full voltage trace, maximum voltage, or output spike times
        :rtype: torch.Tensor
        :raises ValueError: If ``ret_type`` is unsupported
        """
        times = torch.arange(self.T, device=in_spikes.device).view(1, 1, self.T)
        spike_times = in_spikes.unsqueeze(-1)
        voltage = (
            self.v0
            * self._psp_kernel(times - spike_times, self.tau, self.tau_s)
            * (spike_times >= 0).float()
        )
        voltage = self.model.summation_layer(voltage.permute(0, 2, 1)).permute(0, 2, 1)

        if ret_type == "v":
            return voltage
        if ret_type == "v_max":
            return voltage.max(dim=2).values
        if ret_type == "spikes":
            max_index = voltage.argmax(dim=2)
            times = times.expand(in_spikes.shape[0], self.out_features, -1)
            soft_index = (F.softmax(voltage * self.T, dim=2) * times).sum(dim=2)
            fired = (voltage.max(dim=2).values >= self.v_threshold).to(voltage.dtype)
            sign = fired * 2 - 1
            max_index = max_index * sign
            soft_index = soft_index * sign
            return soft_index + (max_index - soft_index).detach()
        raise ValueError(
            f"Invalid out_voltage_type: {ret_type}. Must be 'v', 'v_max', or 'spikes'"
        )
