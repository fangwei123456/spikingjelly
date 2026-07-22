from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, surrogate

__all__ = ["SignedQCFSSequenceEncoder"]


class SignedQCFSSequenceEncoder(nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        time_steps: int,
        *,
        neuron_backend: str = "torch",
        channel_dim: int = -1,
        collect_statistics: bool = True,
        name: str = "activation",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <SignedQCFSSequenceEncoder.__init__-cn>` | :ref:`English <SignedQCFSSequenceEncoder.__init__-en>`

        ----

        .. _SignedQCFSSequenceEncoder.__init__-cn:

        * **中文**

        将 signed 浮点激活编码为正负双通道 QCFS 脉冲序列。模块使用两个
        :class:`~spikingjelly.activation_based.neuron.ActivationAwareIFNode`，
        输出保留显式时间维 ``[T, ...]``，其时间和等于
        ``round(x / scale).clamp(-T, T) * scale``。当 PyTorch round-to-even 与
        神经元半阈值边界约定不一致时，模块通过同一神经元重放目标脉冲计数。

        :param scale: 有限、正值的一维逐通道 QCFS 步长张量。
        :type scale: torch.Tensor
        :param time_steps: 输出脉冲序列的时间步数，必须为正整数。
        :type time_steps: int
        :param neuron_backend: ``ActivationAwareIFNode`` 后端。
        :type neuron_backend: str
        :param channel_dim: 输入中的通道维；其长度必须等于 ``scale.numel()``。
        :type channel_dim: int
        :param collect_statistics: 是否采集 spike rate、计数、局部误差和边界修正比例。
        :type collect_statistics: bool
        :param name: 统计报告中的模块名称。
        :type name: str
        :raises TypeError: 当 ``scale``、``time_steps`` 或 ``channel_dim`` 类型不合法。
        :raises ValueError: 当 scale、时间步数或后端配置不合法。

        ----

        .. _SignedQCFSSequenceEncoder.__init__-en:

        * **English**

        Encode signed floating-point activations as a dual-channel QCFS spike
        sequence. The module uses two
        :class:`~spikingjelly.activation_based.neuron.ActivationAwareIFNode`
        instances and preserves an explicit ``[T, ...]`` time dimension. The
        temporal sum equals ``round(x / scale).clamp(-T, T) * scale``. If
        PyTorch round-to-even and the neuron's half-threshold convention differ
        at a boundary, the desired spike count is replayed through the same
        neuron.

        :param scale: Finite, positive, one-dimensional per-channel QCFS step.
        :type scale: torch.Tensor
        :param time_steps: Positive number of output time steps.
        :type time_steps: int
        :param neuron_backend: Backend used by ``ActivationAwareIFNode``.
        :type neuron_backend: str
        :param channel_dim: Input channel dimension; its size must equal
            ``scale.numel()``.
        :type channel_dim: int
        :param collect_statistics: Whether to collect spike rates, counts, local
            error, and the boundary-correction fraction.
        :type collect_statistics: bool
        :param name: Module name included in statistics.
        :type name: str
        :raises TypeError: If ``scale``, ``time_steps``, or ``channel_dim`` has an
            invalid type.
        :raises ValueError: If the scale, time-step count, or backend is invalid.
        """
        super().__init__()
        if not isinstance(scale, torch.Tensor):
            raise TypeError("scale must be a torch.Tensor.")
        if scale.dim() != 1 or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("scale must be a finite positive one-dimensional tensor.")
        if not isinstance(time_steps, int) or isinstance(time_steps, bool):
            raise TypeError("time_steps must be an integer.")
        if time_steps <= 0:
            raise ValueError("time_steps must be positive.")
        if not isinstance(channel_dim, int) or isinstance(channel_dim, bool):
            raise TypeError("channel_dim must be an integer.")
        if neuron_backend not in ("torch", "triton"):
            raise ValueError(
                f"Unsupported ActivationAwareIFNode backend={neuron_backend!r}."
            )
        self.name = str(name)
        self.time_steps = time_steps
        self.channel_dim = channel_dim
        self.collect_statistics = bool(collect_statistics)
        kwargs = {
            "v_threshold": scale,
            "v_offset": 0.5 * scale,
            "channel_dim": channel_dim,
            "v_reset": None,
            "surrogate_function": surrogate.DeterministicPass(),
            "step_mode": "m",
            "backend": neuron_backend,
        }
        self.positive_neuron = neuron.ActivationAwareIFNode(**kwargs)
        self.negative_neuron = neuron.ActivationAwareIFNode(**kwargs)
        self.metric_scope = "activation"
        self._clear_statistics()

    def _clear_statistics(self) -> None:
        self.positive_spike_rate = 0.0
        self.negative_spike_rate = 0.0
        self.positive_spike_count = 0
        self.negative_spike_count = 0
        self.spike_value_count = 0
        self.local_relative_l2 = 0.0
        self.boundary_correction_fraction = 0.0

    def _replay_mismatches(
        self,
        node: neuron.ActivationAwareIFNode,
        spikes: torch.Tensor,
        desired: torch.Tensor,
        mismatch: torch.Tensor,
        scale: torch.Tensor,
        *,
        verify: bool,
    ) -> torch.Tensor:
        original_membrane = node.v
        time = torch.arange(self.time_steps, device=desired.device)
        time = time.reshape(self.time_steps, *([1] * desired.dim()))
        # Exact threshold pulses avoid BF16 accumulation drift from replaying
        # desired * scale / T while still exercising the stateful IF neuron.
        replay = (time < desired.unsqueeze(0)).to(scale.dtype) * scale
        node.reset()
        replayed_spikes = node(replay)
        replayed_membrane = node.v
        spikes = torch.where(mismatch.unsqueeze(0), replayed_spikes, spikes)
        if not isinstance(original_membrane, torch.Tensor) or not isinstance(
            replayed_membrane, torch.Tensor
        ):
            raise RuntimeError("Signed QCFS encoding requires tensor membrane state.")
        node.v = torch.where(mismatch, replayed_membrane, original_membrane)
        if verify and not torch.equal(spikes.sum(0), desired):
            raise RuntimeError("Signed QCFS encoding failed to replay target counts.")
        return spikes

    def _encode_nonnegative(
        self, node: neuron.ActivationAwareIFNode, value: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        scale = node.v_threshold.to(device=value.device, dtype=value.dtype)
        shape = [1] * value.dim()
        shape[self.channel_dim % value.dim()] = scale.numel()
        scale = scale.reshape(shape)
        sequence = (
            value.unsqueeze(0).expand(self.time_steps, *value.shape) / self.time_steps
        )
        node.reset()
        spikes = node(sequence)
        desired = torch.round(value / scale).clamp(0, self.time_steps)
        mismatch = spikes.sum(0).ne(desired)
        if not self.collect_statistics:
            if mismatch.any():
                spikes = self._replay_mismatches(
                    node, spikes, desired, mismatch, scale, verify=False
                )
            return spikes, 0
        corrected = int(mismatch.count_nonzero().detach().cpu())
        if corrected:
            spikes = self._replay_mismatches(
                node, spikes, desired, mismatch, scale, verify=True
            )
        return spikes, corrected

    def encode(
        self, value: torch.Tensor, metric_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <SignedQCFSSequenceEncoder.encode-cn>` | :ref:`English <SignedQCFSSequenceEncoder.encode-en>`

        ----

        .. _SignedQCFSSequenceEncoder.encode-cn:

        * **中文**

        将浮点激活编码为显式 signed QCFS 脉冲序列。输入通道维必须与逐通道
        scale 对齐，输出第 0 维为时间维，形状 ``[T, *value.shape]``。mask 仅影响
        统计，不改变编码结果。

        :param value: 浮点激活张量。
        :type value: torch.Tensor
        :param metric_mask: 可选的非通道维布尔 mask，仅用于排除统计元素。
        :type metric_mask: Optional[torch.Tensor]
        :return: 形状 ``[T, *value.shape]`` 的 signed spike sequence。
        :rtype: torch.Tensor
        :raises ValueError: 输入不是至少一维的浮点张量、通道数与 scale 不匹配，
            或统计 mask 的形状无效或未选择任何元素。

        ----

        .. _SignedQCFSSequenceEncoder.encode-en:

        * **English**

        Encode a floating-point activation as an explicit signed QCFS spike sequence.
        The input channel dimension must match the per-channel scale. The leading
        output dimension is time, producing shape ``[T, *value.shape]``. The mask
        affects statistics only, not encoding.

        :param value: Floating-point activation tensor.
        :type value: torch.Tensor
        :param metric_mask: Optional boolean mask over non-channel dimensions;
            masked elements are excluded from statistics only.
        :type metric_mask: Optional[torch.Tensor]
        :return: Signed spike sequence with shape ``[T, *value.shape]``.
        :rtype: torch.Tensor
        :raises ValueError: If the input is not a floating-point tensor with at
            least one dimension, its channel count does not match the scale, or
            the statistics mask has an invalid shape or selects no elements.
        """
        if not torch.is_floating_point(value):
            raise ValueError("Signed QCFS encoding requires a floating-point tensor.")
        if value.dim() == 0:
            raise ValueError("Signed QCFS encoding requires at least one dimension.")
        channel_dim = self.channel_dim % value.dim()
        if value.shape[channel_dim] != self.positive_neuron.v_threshold.numel():
            raise ValueError("Input channel size does not match the QCFS scale.")
        positive_spikes, positive_corrections = self._encode_nonnegative(
            self.positive_neuron, F.relu(value)
        )
        negative_spikes, negative_corrections = self._encode_nonnegative(
            self.negative_neuron, F.relu(-value)
        )
        scale = self.positive_neuron.v_threshold.to(value)
        shape = [1] * value.dim()
        shape[channel_dim] = scale.numel()
        sequence = (positive_spikes - negative_spikes) * scale.reshape(shape)
        if self.collect_statistics:
            positive_metric = positive_spikes.detach()
            negative_metric = negative_spikes.detach()
            reconstructed = sequence.sum(0)
            reference = value
            if metric_mask is not None:
                mask = metric_mask.to(device=value.device, dtype=torch.bool)
                expected_shape = (
                    value.shape[:channel_dim] + value.shape[channel_dim + 1 :]
                )
                if mask.shape != expected_shape:
                    raise ValueError(
                        "metric_mask shape must match value dimensions excluding "
                        f"channel_dim: expected {tuple(expected_shape)}, got {tuple(mask.shape)}."
                    )
                if not mask.any():
                    raise ValueError("metric_mask must select at least one element.")
                expanded_mask = mask.unsqueeze(channel_dim).expand(value.shape)
                positive_metric = positive_metric[:, expanded_mask]
                negative_metric = negative_metric[:, expanded_mask]
                reconstructed = reconstructed[expanded_mask]
                reference = reference[expanded_mask]
            self.positive_spike_rate = float(positive_metric.mean().cpu())
            self.negative_spike_rate = float(negative_metric.mean().cpu())
            self.positive_spike_count = int(positive_metric.count_nonzero().cpu())
            self.negative_spike_count = int(negative_metric.count_nonzero().cpu())
            self.spike_value_count = int(positive_metric.numel())
            self.boundary_correction_fraction = (
                positive_corrections + negative_corrections
            ) / (2 * value.numel())
            denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
            relative_l2 = (
                torch.linalg.vector_norm((reconstructed - reference).float())
                / denominator
            )
            self.local_relative_l2 = float(relative_l2.detach().cpu())
            if not math.isfinite(self.local_relative_l2):
                raise ValueError("Signed QCFS local relative L2 must be finite.")
        return sequence

    def reconstruct(self, value: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <SignedQCFSSequenceEncoder.reconstruct-cn>` | :ref:`English <SignedQCFSSequenceEncoder.reconstruct-en>`

        ----

        .. _SignedQCFSSequenceEncoder.reconstruct-cn:

        * **中文**

        直接计算 :meth:`encode` 所生成多步脉冲序列的时间和，不执行神经元时间
        循环。该方法使用完全相同的逐通道 scale、round-to-even 和计数裁剪规则，
        用于验证或加速只依赖时间聚合值的离线逐层推理；返回值本身不是脉冲序列。

        :param value: 浮点激活张量，通道维必须与 scale 对齐。
        :type value: torch.Tensor
        :return: 与 ``encode(value).sum(0)`` 等价的 QCFS 重建值。
        :rtype: torch.Tensor
        :raises ValueError: 输入不是至少一维的浮点张量，或通道数与 scale 不匹配。

        ----

        .. _SignedQCFSSequenceEncoder.reconstruct-en:

        * **English**

        Compute the temporal sum of the multi-step spike sequence produced by
        :meth:`encode` without running the neuron time loop. This method uses the
        same per-channel scale, round-to-even operation, and count clipping. It
        supports validation or acceleration of offline layerwise inference that
        depends only on temporal aggregates; the returned value is not itself a
        spike sequence.

        :param value: Floating-point activation whose channel dimension matches
            the scale.
        :type value: torch.Tensor
        :return: QCFS reconstruction equivalent to ``encode(value).sum(0)``.
        :rtype: torch.Tensor
        :raises ValueError: If the input is not a floating-point tensor with at
            least one dimension or its channel count does not match the scale.
        """
        if not torch.is_floating_point(value):
            raise ValueError("Signed QCFS reconstruction requires floating point.")
        if value.dim() == 0:
            raise ValueError(
                "Signed QCFS reconstruction requires at least one dimension."
            )
        channel_dim = self.channel_dim % value.dim()
        scale = self.positive_neuron.v_threshold.to(value)
        if value.shape[channel_dim] != scale.numel():
            raise ValueError("Input channel size does not match the QCFS scale.")
        shape = [1] * value.dim()
        shape[channel_dim] = scale.numel()
        scale = scale.reshape(shape)
        positive = torch.round(F.relu(value) / scale).clamp(0, self.time_steps)
        negative = torch.round(F.relu(-value) / scale).clamp(0, self.time_steps)
        return (positive - negative) * scale

    def forward(
        self, value_seq: torch.Tensor, metric_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <SignedQCFSSequenceEncoder.forward-cn>` | :ref:`English <SignedQCFSSequenceEncoder.forward-en>`

        ----

        .. _SignedQCFSSequenceEncoder.forward-cn:

        * **中文**

        聚合输入时间维后重新编码为 signed QCFS 脉冲。``value_seq`` 的第 0 维是
        时间维；模块先求和，再生成新的 ``T`` 步序列，因此是离线多步编码而非
        在线逐步流水。

        :param value_seq: 形状 ``[T_in, ...]`` 的非空输入序列。
        :type value_seq: torch.Tensor
        :param metric_mask: 可选统计 mask。
        :type metric_mask: Optional[torch.Tensor]
        :return: 形状 ``[self.time_steps, ...]`` 的编码序列。
        :rtype: torch.Tensor
        :raises ValueError: 输入缺少非空前导时间维，或聚合值不能被 :meth:`encode`
            编码。

        ----

        .. _SignedQCFSSequenceEncoder.forward-en:

        * **English**

        Aggregate the input temporal dimension and re-encode it as signed QCFS
        spikes. The leading dimension is summed before a new ``T``-step sequence is
        generated, so this is offline multi-step encoding rather than an online
        stepwise pipeline.

        :param value_seq: Non-empty input sequence with shape ``[T_in, ...]``.
        :type value_seq: torch.Tensor
        :param metric_mask: Optional statistics mask.
        :type metric_mask: Optional[torch.Tensor]
        :return: Encoded sequence with shape ``[self.time_steps, ...]``.
        :rtype: torch.Tensor
        :raises ValueError: If the input lacks a non-empty leading time dimension or
            the aggregate cannot be encoded by :meth:`encode`.
        """
        if value_seq.dim() < 2 or value_seq.shape[0] <= 0:
            raise ValueError("value_seq must have a non-empty leading time dimension.")
        return self.encode(value_seq.sum(0), metric_mask)

    def statistics(self) -> Dict[str, object]:
        r"""
        **API Language** - :ref:`中文 <SignedQCFSSequenceEncoder.statistics-cn>` | :ref:`English <SignedQCFSSequenceEncoder.statistics-en>`

        ----

        .. _SignedQCFSSequenceEncoder.statistics-cn:

        * **中文**

        返回最近一次编码的脉冲率、脉冲计数、局部误差、边界修正比例和最终膜电位。

        :return: 编码统计 mapping。
        :rtype: Dict[str, object]

        ----

        .. _SignedQCFSSequenceEncoder.statistics-en:

        * **English**

        Return spike rates, spike counts, local error, boundary-correction fraction,
        and final membrane magnitude from the most recent encoding.

        :return: Encoding statistics mapping.
        :rtype: Dict[str, object]
        """
        membranes = []
        for node in (self.positive_neuron, self.negative_neuron):
            value = node.v
            membranes.append(
                float(value.detach().abs().max().cpu())
                if isinstance(value, torch.Tensor)
                else abs(float(value))
            )
        return {
            "name": self.name,
            "backend": self.positive_neuron.backend,
            "positive_spike_rate": self.positive_spike_rate,
            "negative_spike_rate": self.negative_spike_rate,
            "positive_spike_count": self.positive_spike_count,
            "negative_spike_count": self.negative_spike_count,
            "spike_value_count": self.spike_value_count,
            "local_relative_l2": self.local_relative_l2,
            "metric_scope": self.metric_scope,
            "boundary_correction_fraction": self.boundary_correction_fraction,
            "membrane_abs_max": max(membranes),
        }
