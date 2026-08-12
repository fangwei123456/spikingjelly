import torch
import torch.nn as nn

from .. import base, functional

__all__ = ["STBIFNeuron"]


def _as_scalar_tensor(value) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.tensor(value)
    return tensor.detach().clone()


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

        该神经元仅用于推理阶段的 QANN-to-SNN 转换结果执行，不支持训练：
        Torch 路径包含 ``detach``、``round`` 和离散状态更新；Triton 单步和
        多步 kernel 也没有实现 backward。

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

        This neuron is intended only for executing QANN-to-SNN conversion results
        during inference and does not support training: the Torch path contains
        ``detach``, ``round``, and discrete state updates; the Triton single-step
        and multi-step kernels do not implement backward.

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
        self.level = int(level)
        if isinstance(level, bool) or self.level < 2:
            raise ValueError("SpikeZIP quantizer level must be >= 2.")
        self.sym = bool(sym)
        self.register_buffer("q_threshold", _as_scalar_tensor(q_threshold).float())
        self.register_buffer(
            "pos_max",
            _as_scalar_tensor(
                (self.level // 2 - 1 if self.sym else self.level - 1)
                if pos_max is None
                else pos_max
            ).float(),
        )
        self.register_buffer(
            "neg_min",
            _as_scalar_tensor(
                (-self.level // 2 if self.sym else 0) if neg_min is None else neg_min
            ).float(),
        )
        self.step_mode = step_mode
        self.register_memory("q", None)
        self.register_memory("acc_q", None)
        self.register_memory("cur_output", None)

    @classmethod
    def from_quantizer(cls, quantizer: nn.Module) -> "STBIFNeuron":
        scale = quantizer.s
        sym = bool(quantizer.sym)
        pos_max = quantizer.pos_max
        neg_min = quantizer.neg_min
        default_level = (
            int(_as_scalar_tensor(pos_max).item())
            - int(_as_scalar_tensor(neg_min).item())
            + 1
        )
        level = int(getattr(quantizer, "level", default_level))
        return cls(scale, level=level, sym=sym, pos_max=pos_max, neg_min=neg_min)

    @property
    def supported_backends(self) -> tuple[str, ...]:
        return ("torch", "triton")

    def materialize_states(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        step_mode: str,
    ) -> tuple[object, ...]:
        x = inputs[0] if step_mode == "s" else inputs[0][0]
        q, acc_q, cur_output = states
        if q is None or q.shape != x.shape:
            q = torch.full_like(x, 0.5)
            acc_q = torch.zeros_like(x)
            cur_output = torch.zeros_like(x)
        else:
            q = q.to(x)
            acc_q = acc_q.to(x)
            cur_output = cur_output.to(x)
        return q, acc_q, cur_output

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        q, acc_q, cur_output = states
        if self.backend == "triton":
            out, q, acc_q, cur_output = functional.stbif_single_step_triton(
                x,
                q,
                acc_q,
                self.q_threshold,
                self.pos_max,
                self.neg_min,
            )
            return (out,), (q, acc_q, cur_output)
        out, q, acc_q, cur_output = functional.stbif_step(
            x,
            q,
            acc_q,
            self.q_threshold,
            self.pos_max,
            self.neg_min,
        )
        return (out,), (q, acc_q, cur_output)

    def multi_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x_seq = inputs[0]
        q, acc_q, cur_output = states
        if self.backend == "triton" and x_seq.device.type != "cuda":
            raise RuntimeError("STBIFNeuron backend='triton' requires a CUDA tensor.")
        if self.backend == "triton":
            from spikingjelly.activation_based.triton_kernel.neuron_kernel import stbif

            out_seq, q, acc_q, cur_output = stbif.multi_step_stbif(
                x_seq,
                q,
                acc_q,
                self.q_threshold.to(dtype=x_seq.dtype),
                self.pos_max.to(dtype=x_seq.dtype),
                self.neg_min.to(dtype=x_seq.dtype),
            )
        else:
            out_seq = torch.empty_like(x_seq)
            for t in range(x_seq.shape[0]):
                out_seq[t], q, acc_q, cur_output = functional.stbif_step(
                    x_seq[t],
                    q,
                    acc_q,
                    self.q_threshold,
                    self.pos_max,
                    self.neg_min,
                )
        return (out_seq,), (q, acc_q, cur_output)

    @property
    def accumulated(self) -> torch.Tensor:
        if self.acc_q is None:
            raise RuntimeError("STBIFNeuron has no accumulated state before forward.")
        return self.acc_q * self.q_threshold.to(dtype=self.acc_q.dtype)
