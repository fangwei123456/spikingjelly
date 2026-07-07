from typing import Optional

import torch
import torch.nn as nn

from .. import base

__all__ = ["STBIFNeuron"]


def _as_scalar_tensor(value, reference: Optional[torch.Tensor] = None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.tensor(value)
    if reference is not None:
        tensor = tensor.to(device=reference.device, dtype=reference.dtype)
    return tensor.detach().clone()


def _quantizer_attr(module: nn.Module, name: str):
    if not hasattr(module, name):
        raise ValueError(
            f"STBIFNeuron.from_quantizer requires quantizer {module!r} "
            f"to expose {name!r}."
        )
    return getattr(module, name)


class STBIFNeuron(base.MemoryModule):
    def __init__(
        self,
        q_threshold,
        level: int,
        sym: bool = False,
        pos_max=None,
        neg_min=None,
        step_mode: str = "s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <STBIFNeuron.__init__-cn>` | :ref:`English <STBIFNeuron.__init__-en>`

        ----

        .. _STBIFNeuron.__init__-cn:

        * **中文**

        SpikeZIP QANN-to-SNN 转换使用的 signed ternary BIF 神经元。它从
        QANN quantizer 的尺度 ``q_threshold`` 和量化边界构造状态动力学。
        当前步内部输出 ``cur_output`` 取值为 ``-1``、``0`` 或 ``1``，最终
        返回值为 ``cur_output * q_threshold``，因此通常不是二值 ``0/1``
        spike。``acc_q`` 记录已累计释放的量化整数，并被限制在
        ``[neg_min, pos_max]`` 内；``q`` 是量化残差，初始化为 ``0.5`` 以对应
        round-to-nearest 的量化偏置。

        该神经元仅用于推理阶段的 QANN-to-SNN 转换结果执行，不支持训练。
        ``single_step_forward`` 和 torch multi-step 路径包含 ``detach``、
        ``round`` 和离散状态更新；Triton multi-step kernel 也没有实现
        backward。因此不要把该神经元用于端到端梯度训练或微调。

        :param q_threshold: QANN quantizer scale。
        :type q_threshold: float or torch.Tensor
        :param level: 量化级数。
        :type level: int
        :param sym: 是否使用有符号对称量化边界。
        :type sym: bool
        :param pos_max: 正向累计量化上界。为 ``None`` 时由 ``level`` 和
            ``sym`` 推断。
        :type pos_max: float or torch.Tensor or None
        :param neg_min: 负向累计量化下界。为 ``None`` 时由 ``level`` 和
            ``sym`` 推断。
        :type neg_min: float or torch.Tensor or None
        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。
        :type step_mode: str

        ----

        .. _STBIFNeuron.__init__-en:

        * **English**

        Signed ternary BIF neuron used by SpikeZIP QANN-to-SNN conversion. It
        builds its state dynamics from the QANN quantizer scale
        ``q_threshold`` and quantization bounds. The internal current output
        ``cur_output`` is ``-1``, ``0`` or ``1``. The returned value is
        ``cur_output * q_threshold``, so it is generally not a binary ``0/1``
        spike. ``acc_q`` stores the accumulated released quantized integer and
        is clamped by the effective ``[neg_min, pos_max]`` bounds. ``q`` is the
        quantized residual and starts from ``0.5`` to match round-to-nearest
        quantization.

        This neuron is inference-only and is intended for executing converted
        QANN-to-SNN models. ``single_step_forward`` and the torch multi-step path
        contain ``detach``, ``round`` and discrete state updates; the Triton
        multi-step kernel also does not implement backward. Do not use this
        neuron for end-to-end gradient training or fine-tuning.

        :param q_threshold: QANN quantizer scale.
        :type q_threshold: float or torch.Tensor
        :param level: Number of quantization levels.
        :type level: int
        :param sym: Whether to use signed symmetric quantization bounds.
        :type sym: bool
        :param pos_max: Positive accumulated quantization bound. If ``None``,
            it is inferred from ``level`` and ``sym``.
        :type pos_max: float or torch.Tensor or None
        :param neg_min: Negative accumulated quantization bound. If ``None``,
            it is inferred from ``level`` and ``sym``.
        :type neg_min: float or torch.Tensor or None
        :param step_mode: Step mode, ``"s"`` or ``"m"``.
        :type step_mode: str
        """
        super().__init__()
        self.register_buffer("q_threshold", _as_scalar_tensor(q_threshold).float())
        self.register_buffer(
            "pos_max",
            _as_scalar_tensor(
                (level // 2 - 1 if sym else level - 1) if pos_max is None else pos_max
            ).float(),
        )
        self.register_buffer(
            "neg_min",
            _as_scalar_tensor(
                (-level // 2 if sym else 0) if neg_min is None else neg_min
            ).float(),
        )
        self.level = int(level)
        self.sym = bool(sym)
        self.step_mode = step_mode
        self.reset()

    @classmethod
    def from_quantizer(cls, quantizer: nn.Module) -> "STBIFNeuron":
        scale = _quantizer_attr(quantizer, "s")
        sym = bool(_quantizer_attr(quantizer, "sym"))
        pos_max = _quantizer_attr(quantizer, "pos_max")
        neg_min = _quantizer_attr(quantizer, "neg_min")
        default_level = (
            int(_as_scalar_tensor(pos_max).item())
            - int(_as_scalar_tensor(neg_min).item())
            + 1
        )
        level = int(getattr(quantizer, "level", default_level))
        if level <= 0:
            raise ValueError("SpikeZIP quantizer level must be positive.")
        return cls(scale, level=level, sym=sym, pos_max=pos_max, neg_min=neg_min)

    @property
    def supported_backends(self) -> tuple[str, ...]:
        return ("torch", "triton")

    def reset(self) -> None:
        self.q = None
        self.acc_q = None
        self.cur_output = None
        self.is_work = False

    def _init_state(self, x: torch.Tensor) -> None:
        if (
            self.cur_output is None
            or self.acc_q is None
            or self.q is None
            or self.cur_output.shape != x.shape
        ):
            self.cur_output = torch.zeros_like(x)
            self.acc_q = torch.zeros_like(x)
            self.q = torch.full_like(x, 0.5)
        elif self.cur_output.device != x.device or self.cur_output.dtype != x.dtype:
            self.cur_output = self.cur_output.to(device=x.device, dtype=x.dtype)
            self.acc_q = self.acc_q.to(device=x.device, dtype=x.dtype)
            self.q = self.q.to(device=x.device, dtype=x.dtype)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        q_threshold = self.q_threshold.to(device=x.device, dtype=x.dtype)
        normalized = x / q_threshold
        self._init_state(normalized)

        self.q = self.q + normalized.detach()
        self.acc_q = torch.round(self.acc_q)
        pos_max = self.pos_max.to(device=x.device, dtype=x.dtype)
        neg_min = self.neg_min.to(device=x.device, dtype=x.dtype)
        spike_position = (self.q - 1 >= 0) & (self.acc_q < pos_max)
        neg_spike_position = (self.q < 0) & (self.acc_q > neg_min)

        self.cur_output.zero_()
        self.cur_output[spike_position] = 1
        self.cur_output[neg_spike_position] = -1
        self.acc_q = self.acc_q + self.cur_output
        self.q[spike_position] = self.q[spike_position] - 1
        self.q[neg_spike_position] = self.q[neg_spike_position] + 1
        self.is_work = bool((normalized != 0).any() | (self.cur_output != 0).any())
        return self.cur_output * q_threshold

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.device.type == "cuda" and self.backend == "triton":
            return self._multi_step_forward_triton(x_seq)
        return self._multi_step_forward_torch_optimized(x_seq)

    def _multi_step_forward_torch_optimized(self, x_seq: torch.Tensor) -> torch.Tensor:
        q_threshold = self.q_threshold.to(device=x_seq.device, dtype=x_seq.dtype)
        pos_max = self.pos_max.to(device=x_seq.device, dtype=x_seq.dtype)
        neg_min = self.neg_min.to(device=x_seq.device, dtype=x_seq.dtype)
        self._init_state(x_seq[0])
        q = self.q
        acc_q = self.acc_q
        out_seq = torch.empty_like(x_seq)
        for t in range(x_seq.shape[0]):
            normalized = (x_seq[t] / q_threshold).detach()
            q = q + normalized
            acc_q = torch.round(acc_q)
            spike_position = (q - 1 >= 0) & (acc_q < pos_max)
            neg_spike_position = (q < 0) & (acc_q > neg_min)
            cur_output = spike_position.to(x_seq.dtype) - neg_spike_position.to(
                x_seq.dtype
            )
            acc_q = acc_q + cur_output
            q = torch.where(spike_position, q - 1, q)
            q = torch.where(neg_spike_position, q + 1, q)
            out_seq[t] = cur_output * q_threshold
        self.q = q
        self.acc_q = acc_q
        self.cur_output = cur_output
        self.is_work = bool((x_seq != 0).any() | (out_seq != 0).any())
        return out_seq

    def _multi_step_forward_triton(self, x_seq: torch.Tensor) -> torch.Tensor:
        from spikingjelly.activation_based.triton_kernel import spikezip_kernel

        q_threshold = self.q_threshold.to(device=x_seq.device, dtype=x_seq.dtype)
        pos_max = self.pos_max.to(device=x_seq.device, dtype=x_seq.dtype)
        neg_min = self.neg_min.to(device=x_seq.device, dtype=x_seq.dtype)
        self._init_state(x_seq[0])
        out_seq, q, acc_q, cur_output, work_flag = spikezip_kernel.multi_step_stbif(
            x_seq,
            self.q,
            self.acc_q,
            q_threshold,
            pos_max,
            neg_min,
        )
        self.q = q
        self.acc_q = acc_q
        self.cur_output = cur_output
        self.is_work = bool(work_flag.item())
        return out_seq

    @property
    def accumulated(self) -> torch.Tensor:
        if self.acc_q is None:
            raise RuntimeError("STBIFNeuron has no accumulated state before forward.")
        return self.acc_q * self.q_threshold.to(
            device=self.acc_q.device,
            dtype=self.acc_q.dtype,
        )
