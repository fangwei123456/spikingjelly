from typing import Sequence, Union, Optional

import torch
import torch.nn as nn

from .. import base, surrogate


__all__ = [
    "FewSpikeTable",
    "FewSpikeNode",
    "OutlierAwareThresholdNode",
    "HGNode",
]


TensorLike1D = Union[torch.Tensor, Sequence[float]]


def _as_float_1d_tensor(name: str, value: TensorLike1D) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.dim() != 1:
        shape = tuple(tensor.shape)
        raise ValueError(
            f"{name} must be a 1-D tensor or sequence, but got shape {shape}."
        )
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
    if torch.is_complex(tensor):
        raise TypeError(f"{name} must be a real-valued tensor or sequence.")
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(torch.get_default_dtype())
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor.detach().clone()


def _check_table_length(name: str, table: "FewSpikeTable", K: int):
    if not isinstance(table, FewSpikeTable):
        raise TypeError(
            f"{name} must be FewSpikeTable, but got {type(table).__name__}."
        )
    if table.K != K:
        raise ValueError(f"{name}.K must be {K}, but got {table.K}.")


class FewSpikeTable:
    def __init__(
        self,
        theta: TensorLike1D,
        h: TensorLike1D,
        d: TensorLike1D,
    ):
        r"""
        **API Language** - :ref:`中文 <FewSpikeTable.__init__-cn>` | :ref:`English <FewSpikeTable.__init__-en>`

        ----

        .. _FewSpikeTable.__init__-cn:

        * **中文**

        Few-Spike 神经元的编码表。该对象只保存一维浮点配置，不是
        :class:`torch.nn.Module`。当它传入 :class:`FewSpikeNode` 或其子类时，
        ``theta``、``h``、``d`` 会被注册为 buffer，因此会跟随模块的
        ``device`` 和 ``dtype`` 迁移。

        :param theta: 长度为 ``K`` 的阈值序列，形状为 ``[K]``。
        :type theta: Union[torch.Tensor, Sequence[float]]
        :param h: 长度为 ``K`` 的膜电位扣减序列，形状为 ``[K]``。
        :type h: Union[torch.Tensor, Sequence[float]]
        :param d: 长度为 ``K`` 的输出权重序列，形状为 ``[K]``。
        :type d: Union[torch.Tensor, Sequence[float]]
        :raises ValueError: 当任一参数不是一维、为空、shape 不一致或包含非有限值时抛出。
        :raises TypeError: 当任一参数为复数张量或复数序列时抛出。

        ----

        .. _FewSpikeTable.__init__-en:

        * **English**

        Coding table for Few-Spike neurons. This object only stores 1-D floating
        point configuration and is not a :class:`torch.nn.Module`. When passed to
        :class:`FewSpikeNode` or its subclasses, ``theta``, ``h`` and ``d`` are
        registered as buffers, so they follow module ``device`` and ``dtype``
        conversions.

        :param theta: Threshold sequence with length ``K`` and shape ``[K]``.
        :type theta: Union[torch.Tensor, Sequence[float]]
        :param h: Membrane subtraction sequence with length ``K`` and shape ``[K]``.
        :type h: Union[torch.Tensor, Sequence[float]]
        :param d: Output weight sequence with length ``K`` and shape ``[K]``.
        :type d: Union[torch.Tensor, Sequence[float]]
        :raises ValueError: Raised when any argument is not 1-D, is empty,
            has inconsistent shape, or contains non-finite values.
        :raises TypeError: Raised when any argument is complex-valued.
        """
        self.theta = _as_float_1d_tensor("theta", theta)
        self.h = _as_float_1d_tensor("h", h)
        self.d = _as_float_1d_tensor("d", d)
        if self.theta.shape != self.h.shape or self.theta.shape != self.d.shape:
            raise ValueError(
                "theta, h and d must have the same shape, but got "
                f"{tuple(self.theta.shape)}, {tuple(self.h.shape)}, "
                f"{tuple(self.d.shape)}."
            )

    @property
    def K(self) -> int:
        return self.theta.numel()


class FewSpikeNode(nn.Module, base.StepModule):
    r"""
    **API Language** - :ref:`中文 <FewSpikeNode-cn>` | :ref:`English <FewSpikeNode-en>`

    ----

    .. _FewSpikeNode-cn:

    * **中文**

    Memoryless Few-Spike 神经元。该模块支持 ``step_mode="s"`` 和
    ``step_mode="m"``，但不继承 :class:`MemoryModule` 或 ``BaseNode``，也不保存
    跨 ``forward`` 的膜电位状态。

    单步模式输入 ``x`` 的形状为 ``[...]``，表示已经聚合好的 gate input / 初始膜电位；
    输出形状为 ``[...]``。多步模式输入 ``x_seq`` 的形状必须为 ``[K, ...]``，第 0 维
    长度必须等于编码表长度 ``K``；模块先计算 ``x_seq.sum(0)`` 作为初始膜电位，再输出
    weighted spike sequence，形状为 ``[K, ...]``。多步模式不会在输出端自动聚合。

    输入必须为浮点 tensor。编码表参数注册为 buffer，会随模块 ``to(device/dtype)`` 迁移；
    若输入 dtype 与 buffer dtype 不同，遵循 PyTorch 的 dtype promotion 规则。

    ----

    .. _FewSpikeNode-en:

    * **English**

    A memoryless Few-Spike neuron. This module supports ``step_mode="s"`` and
    ``step_mode="m"``, but does not inherit :class:`MemoryModule` or ``BaseNode``
    and does not store membrane potential across ``forward`` calls.

    In single-step mode, input ``x`` has shape ``[...]`` and is interpreted as an
    already accumulated gate input / initial membrane potential; output shape is
    ``[...]``. In multi-step mode, input ``x_seq`` must have shape ``[K, ...]``
    whose leading dimension equals the coding table length ``K``; the module first
    uses ``x_seq.sum(0)`` as the initial membrane potential and returns a weighted
    spike sequence with shape ``[K, ...]``. Multi-step mode does not aggregate the
    output sequence.

    Inputs must be floating point tensors. Coding table parameters are registered
    as buffers and follow module ``to(device/dtype)`` conversions. If the input
    dtype differs from the buffer dtype, PyTorch dtype promotion rules apply.
    """

    def __init__(
        self,
        table: FewSpikeTable,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        step_mode: str = "s",
    ):
        r"""
        :param table: Few-Spike 编码表 / Few-Spike coding table.
        :type table: FewSpikeTable
        :param surrogate_function: 替代梯度函数 / surrogate function for spike generation.
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param step_mode: 步进模式，``"s"`` 或 ``"m"`` / step mode, ``"s"`` or ``"m"``.
        :type step_mode: str
        :raises TypeError: 当 ``table`` 不是 :class:`FewSpikeTable` 时抛出 /
            raised when ``table`` is not :class:`FewSpikeTable`.
        :raises ValueError: 当 ``step_mode`` 非法时抛出 /
            raised when ``step_mode`` is invalid.
        """
        super().__init__()
        if not isinstance(table, FewSpikeTable):
            raise TypeError(
                f"table must be FewSpikeTable, but got {type(table).__name__}."
            )
        self.register_buffer("theta", table.theta.clone())
        self.register_buffer("h", table.h.clone())
        self.register_buffer("d", table.d.clone())
        self.surrogate_function = surrogate_function
        self.step_mode = step_mode

    @property
    def K(self) -> int:
        return self.theta.numel()

    def _check_input(self, x: torch.Tensor, name: str):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if not torch.is_floating_point(x):
            raise TypeError(f"{name} must be a floating point tensor.")

    def _check_multi_step_input(self, x_seq: torch.Tensor):
        self._check_input(x_seq, "x_seq")
        if x_seq.dim() < 1:
            raise ValueError(
                "x_seq must have at least one dimension and shape [K, ...]."
            )
        if x_seq.shape[0] != self.K:
            raise ValueError(
                f"x_seq.shape[0] must equal K={self.K}, but got {x_seq.shape[0]}."
            )

    def _run_table(
        self,
        gate: torch.Tensor,
        theta: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor,
        return_sequence: bool,
    ) -> torch.Tensor:
        v = gate
        y_seq = []
        y = None
        for k in range(theta.numel()):
            z = self.surrogate_function(v - theta[k])
            weighted_spike = d[k] * z
            if return_sequence:
                y_seq.append(weighted_spike)
            else:
                if y is None:
                    y = torch.zeros_like(gate)
                y = y + weighted_spike
            v = v - h[k] * z
        if return_sequence:
            return torch.stack(y_seq, dim=0)
        return y

    def _forward_from_gate(
        self, gate: torch.Tensor, return_sequence: bool
    ) -> torch.Tensor:
        return self._run_table(gate, self.theta, self.h, self.d, return_sequence)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <FewSpikeNode.single_step_forward-cn>` | :ref:`English <FewSpikeNode.single_step_forward-en>`

        ----

        .. _FewSpikeNode.single_step_forward-cn:

        * **中文**

        单步前向传播。``x`` 的形状为 ``[...]``，表示聚合后的 gate input；
        返回形状为 ``[...]`` 的数值输出。

        :param x: 浮点输入张量，形状为 ``[...]``。
        :type x: torch.Tensor
        :return: 单步输出张量，形状为 ``[...]``。
        :rtype: torch.Tensor
        :raises TypeError: 当 ``x`` 不是浮点 tensor 时抛出。

        ----

        .. _FewSpikeNode.single_step_forward-en:

        * **English**

        Single-step forward. ``x`` has shape ``[...]`` and is interpreted as the
        accumulated gate input; returns a numeric output with shape ``[...]``.

        :param x: Floating point input tensor with shape ``[...]``.
        :type x: torch.Tensor
        :return: Single-step output tensor with shape ``[...]``.
        :rtype: torch.Tensor
        :raises TypeError: Raised when ``x`` is not a floating point tensor.
        """
        self._check_input(x, "x")
        return self._forward_from_gate(x, return_sequence=False)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <FewSpikeNode.multi_step_forward-cn>` | :ref:`English <FewSpikeNode.multi_step_forward-en>`

        ----

        .. _FewSpikeNode.multi_step_forward-cn:

        * **中文**

        多步前向传播。``x_seq`` 的形状必须为 ``[K, ...]``。模块先对时间维求和得到
        初始膜电位，再返回形状为 ``[K, ...]`` 的 weighted spike sequence。返回值不是
        binary raw spikes，也不会在输出端自动 ``sum(0)``。

        :param x_seq: 浮点输入序列，形状为 ``[K, ...]``。
        :type x_seq: torch.Tensor
        :return: weighted spike sequence，形状为 ``[K, ...]``。
        :rtype: torch.Tensor
        :raises TypeError: 当 ``x_seq`` 不是浮点 tensor 时抛出。
        :raises ValueError: 当 ``x_seq.shape[0] != K`` 时抛出。

        ----

        .. _FewSpikeNode.multi_step_forward-en:

        * **English**

        Multi-step forward. ``x_seq`` must have shape ``[K, ...]``. The module
        first sums the time dimension to get the initial membrane potential and
        then returns a weighted spike sequence with shape ``[K, ...]``. The return
        value is not binary raw spikes and is not automatically aggregated by
        ``sum(0)``.

        :param x_seq: Floating point input sequence with shape ``[K, ...]``.
        :type x_seq: torch.Tensor
        :return: Weighted spike sequence with shape ``[K, ...]``.
        :rtype: torch.Tensor
        :raises TypeError: Raised when ``x_seq`` is not a floating point tensor.
        :raises ValueError: Raised when ``x_seq.shape[0] != K``.
        """
        self._check_multi_step_input(x_seq)
        return self._forward_from_gate(x_seq.sum(dim=0), return_sequence=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self.single_step_forward(x)
        elif self.step_mode == "m":
            return self.multi_step_forward(x)
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self) -> str:
        return f"K={self.K}, step_mode={self.step_mode}"


class OutlierAwareThresholdNode(FewSpikeNode):
    def __init__(
        self,
        table: FewSpikeTable,
        outlier_table: FewSpikeTable,
        split_threshold: float,
        clamp_value: Optional[float] = None,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        step_mode: str = "s",
    ):
        r"""
        **API Language** - :ref:`中文 <OutlierAwareThresholdNode.__init__-cn>` | :ref:`English <OutlierAwareThresholdNode.__init__-en>`

        ----

        .. _OutlierAwareThresholdNode.__init__-cn:

        * **中文**

        通用 outlier-aware thresholding Few-Spike 节点。输入 gate 先按 ``clamp_value`` 截断
        （若提供），再分解为 ``sign`` 和幅值。幅值 ``<= split_threshold`` 的元素使用
        ``table``，幅值 ``> split_threshold`` 的元素使用 ``outlier_table``，最后恢复
        ``sign``；因此 gate 为 0 的元素输出为 0。单步模式输入输出形状为 ``[...]``；
        多步模式输入形状必须为 ``[K, ...]``，输出形状为 ``[K, ...]``。两个 table 的
        ``K`` 必须相同。

        该类复现 LAS 中 OAT 分支的 sign / clamp / split 结构，但不内置 LAS 的
        ``mtn`` 表生成或 fast floor-quantized 路径。若需要与 LAS 数值完全一致，应显式
        传入对应的 :class:`FewSpikeTable`。

        :param table: normal 分支编码表。
        :type table: FewSpikeTable
        :param outlier_table: outlier 分支编码表。
        :type outlier_table: FewSpikeTable
        :param split_threshold: normal 与 outlier 分支的非负幅值分界。
        :type split_threshold: float
        :param clamp_value: 可选的对称截断幅值。若不为 ``None``，必须大于等于 ``split_threshold``。
        :type clamp_value: Optional[float]
        :param surrogate_function: 替代梯度函数。
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。
        :type step_mode: str
        :raises ValueError: 当 table 长度不一致或阈值参数非法时抛出。

        ----

        .. _OutlierAwareThresholdNode.__init__-en:

        * **English**

        Generic outlier-aware thresholding Few-Spike node. The input gate is first clamped
        by ``clamp_value`` when provided, then decomposed into ``sign`` and
        magnitude. Elements with magnitude ``<= split_threshold`` use ``table``;
        elements with magnitude ``> split_threshold`` use ``outlier_table``; the
        sign is restored afterward, so zero-valued gate elements map to zero output.
        Single-step input and output shape is ``[...]``; multi-step input must have
        shape ``[K, ...]`` and output shape is ``[K, ...]``. Both tables must have
        the same ``K``.

        This class reproduces the sign / clamp / split structure of LAS OAT
        branches, but does not include LAS ``mtn`` table generation or the fast
        floor-quantized path. Pass matching :class:`FewSpikeTable` objects explicitly
        when exact LAS numeric behavior is required.

        :param table: Coding table for the normal branch.
        :type table: FewSpikeTable
        :param outlier_table: Coding table for the outlier branch.
        :type outlier_table: FewSpikeTable
        :param split_threshold: Non-negative magnitude threshold between normal
            and outlier branches.
        :type split_threshold: float
        :param clamp_value: Optional symmetric clamp magnitude. If not ``None``,
            it must be no smaller than ``split_threshold``.
        :type clamp_value: Optional[float]
        :param surrogate_function: Surrogate function.
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param step_mode: Step mode, ``"s"`` or ``"m"``.
        :type step_mode: str
        :raises ValueError: Raised when table lengths are inconsistent or threshold
            arguments are invalid.
        """
        _check_table_length("outlier_table", outlier_table, table.K)
        if split_threshold < 0:
            raise ValueError("split_threshold must be non-negative.")
        if clamp_value is not None:
            if clamp_value <= 0:
                raise ValueError("clamp_value must be positive when it is not None.")
            if clamp_value < split_threshold:
                raise ValueError("clamp_value must be no smaller than split_threshold.")
        super().__init__(table, surrogate_function, step_mode)
        self.register_buffer("outlier_theta", outlier_table.theta.clone())
        self.register_buffer("outlier_h", outlier_table.h.clone())
        self.register_buffer("outlier_d", outlier_table.d.clone())
        self.split_threshold = float(split_threshold)
        self.clamp_value = None if clamp_value is None else float(clamp_value)

    def _forward_from_gate(
        self, gate: torch.Tensor, return_sequence: bool
    ) -> torch.Tensor:
        if self.clamp_value is not None:
            gate = gate.clamp(min=-self.clamp_value, max=self.clamp_value)
        signs = torch.sign(gate).detach()
        magnitude = gate.abs()
        normal = self._run_table(magnitude, self.theta, self.h, self.d, return_sequence)
        outlier = self._run_table(
            magnitude,
            self.outlier_theta,
            self.outlier_h,
            self.outlier_d,
            return_sequence,
        )
        mask = magnitude <= self.split_threshold
        if return_sequence:
            mask = mask.unsqueeze(0)
            signs = signs.unsqueeze(0)
        return torch.where(mask, normal, outlier) * signs

    def extra_repr(self) -> str:
        return super().extra_repr() + (
            f", split_threshold={self.split_threshold}, clamp_value={self.clamp_value}"
        )


class HGNode(FewSpikeNode):
    def __init__(
        self,
        tables: Sequence[FewSpikeTable],
        gate_thresholds: TensorLike1D,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        step_mode: str = "s",
    ):
        r"""
        **API Language** - :ref:`中文 <HGNode.__init__-cn>` | :ref:`English <HGNode.__init__-en>`

        ----

        .. _HGNode.__init__-cn:

        * **中文**

        通用 hierarchically-gated Few-Spike 节点。``gate_thresholds`` 按升序把 gate input
        划分为 ``len(tables)`` 个区间：第一个 table 处理 ``x <= gate_thresholds[0]``，
        中间 table 处理相邻阈值之间的元素，最后一个 table 处理
        ``x > gate_thresholds[-1]``。所有 table 的 ``K`` 必须一致。单步模式输入输出形状
        为 ``[...]``；多步模式输入形状必须为 ``[K, ...]``，输出形状为 ``[K, ...]``。

        该类提供区域路由形式的通用层级门控，不声明与 LAS 中某个具体 fitted activation
        模块完全等价。需要复现特定 LAS 非线性时，应使用与该非线性对应的 table 和阈值。

        :param tables: 每个 gate 区间对应的编码表序列，长度至少为 1。
        :type tables: Sequence[FewSpikeTable]
        :param gate_thresholds: 升序一维阈值序列，长度必须为 ``len(tables) - 1``。
        :type gate_thresholds: Union[torch.Tensor, Sequence[float]]
        :param surrogate_function: 替代梯度函数。
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。
        :type step_mode: str
        :raises ValueError: 当 table 数量、阈值数量、阈值顺序或 table 长度非法时抛出。
        :raises TypeError: 当 ``tables`` 中存在非 :class:`FewSpikeTable` 对象时抛出。

        ----

        .. _HGNode.__init__-en:

        * **English**

        Generic hierarchically-gated Few-Spike node. ``gate_thresholds`` partitions the
        gate input into ``len(tables)`` regions in ascending order: the first table
        handles ``x <= gate_thresholds[0]``, middle tables handle adjacent
        threshold intervals, and the last table handles ``x > gate_thresholds[-1]``.
        All tables must have the same ``K``. Single-step input and output shape is
        ``[...]``; multi-step input must have shape ``[K, ...]`` and output shape is
        ``[K, ...]``.

        This class provides a generic region-routing hierarchical gate and does not
        claim exact equivalence to a specific LAS fitted activation module. To
        reproduce a specific LAS nonlinearity, use tables and thresholds matching
        that nonlinearity.

        :param tables: Coding tables for gate regions, with length at least 1.
        :type tables: Sequence[FewSpikeTable]
        :param gate_thresholds: Ascending 1-D threshold sequence with length
            ``len(tables) - 1``.
        :type gate_thresholds: Union[torch.Tensor, Sequence[float]]
        :param surrogate_function: Surrogate function.
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param step_mode: Step mode, ``"s"`` or ``"m"``.
        :type step_mode: str
        :raises ValueError: Raised when table count, threshold count, threshold
            order, or table lengths are invalid.
        :raises TypeError: Raised when an item in ``tables`` is not
            :class:`FewSpikeTable`.
        """
        if len(tables) == 0:
            raise ValueError("tables must contain at least one FewSpikeTable.")
        for i, table in enumerate(tables):
            if not isinstance(table, FewSpikeTable):
                raise TypeError(
                    f"tables[{i}] must be FewSpikeTable, "
                    f"but got {type(table).__name__}."
                )
        K = tables[0].K
        for i, table in enumerate(tables[1:], start=1):
            _check_table_length(f"tables[{i}]", table, K)

        if len(tables) == 1 and torch.as_tensor(gate_thresholds).numel() == 0:
            thresholds = torch.empty(
                0,
                dtype=tables[0].theta.dtype,
                device=tables[0].theta.device,
            )
        else:
            thresholds = _as_float_1d_tensor("gate_thresholds", gate_thresholds)
        if thresholds.numel() != len(tables) - 1:
            raise ValueError(
                "gate_thresholds must have length len(tables) - 1, but got "
                f"{thresholds.numel()} for {len(tables)} tables."
            )
        if thresholds.numel() > 1 and not torch.all(thresholds[1:] > thresholds[:-1]):
            raise ValueError("gate_thresholds must be strictly increasing.")

        super().__init__(tables[0], surrogate_function, step_mode)
        self.register_buffer(
            "region_theta", torch.stack([table.theta for table in tables])
        )
        self.register_buffer("region_h", torch.stack([table.h for table in tables]))
        self.register_buffer("region_d", torch.stack([table.d for table in tables]))
        self.register_buffer("gate_thresholds", thresholds)

    def _forward_from_gate(
        self, gate: torch.Tensor, return_sequence: bool
    ) -> torch.Tensor:
        region_ids = torch.bucketize(gate.detach(), self.gate_thresholds)
        y = self._run_table(
            gate,
            self.region_theta[0],
            self.region_h[0],
            self.region_d[0],
            return_sequence,
        )
        for i in range(1, self.region_theta.shape[0]):
            region_y = self._run_table(
                gate,
                self.region_theta[i],
                self.region_h[i],
                self.region_d[i],
                return_sequence,
            )
            mask = region_ids == i
            if return_sequence:
                mask = mask.unsqueeze(0)
            y = torch.where(mask, region_y, y)
        return y

    def extra_repr(self) -> str:
        return super().extra_repr() + (
            f", regions={self.region_theta.shape[0]}, "
            f"gate_thresholds={self.gate_thresholds}"
        )
