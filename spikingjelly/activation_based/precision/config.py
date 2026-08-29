from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class PrecisionConfig:
    mode: Literal["fp32", "fp16", "bf16", "fp8"] = "fp32"
    fp8_recipe: Literal["auto", "delayed", "current", "block", "mxfp8"] = "auto"
    triton_storage: Optional[
        Literal[
            "fp32",
            "fp16",
            "bf16",
            "float8_e4m3fn",
            "float8_e5m2",
        ]
    ] = None
    triton_fwd: Literal["fp8", "fp16", "bf16", "fp32"] = "fp32"
    triton_bwd: Literal["fp8", "fp16", "bf16", "fp32"] = "fp32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", str(self.mode).lower())
        object.__setattr__(self, "fp8_recipe", str(self.fp8_recipe).lower())
        if self.triton_storage is not None:
            object.__setattr__(
                self,
                "triton_storage",
                str(self.triton_storage).lower().removeprefix("torch."),
            )
        object.__setattr__(self, "triton_fwd", str(self.triton_fwd).lower())
        object.__setattr__(self, "triton_bwd", str(self.triton_bwd).lower())
        if self.mode not in {"fp32", "fp16", "bf16", "fp8"}:
            raise ValueError("mode must be 'fp32', 'fp16', 'bf16', or 'fp8'.")
        if self.fp8_recipe not in {"auto", "delayed", "current", "block", "mxfp8"}:
            raise ValueError("Unsupported fp8_recipe.")
        if self.mode != "fp8" and self.fp8_recipe != "auto":
            raise ValueError("fp8_recipe is only valid when mode='fp8'.")
        if self.triton_storage is None and (
            self.triton_fwd != "fp32" or self.triton_bwd != "fp32"
        ):
            raise ValueError(
                "triton_fwd and triton_bwd require triton_storage to be set."
            )
        if self.triton_storage is not None and self.triton_storage not in {
            "fp32",
            "fp16",
            "bf16",
            "float8_e4m3fn",
            "float8_e5m2",
        }:
            raise ValueError("Unsupported triton_storage.")
        if self.triton_fwd not in {"fp8", "fp16", "bf16", "fp32"}:
            raise ValueError("Unsupported triton_fwd.")
        if self.triton_bwd not in {"fp8", "fp16", "bf16", "fp32"}:
            raise ValueError("Unsupported triton_bwd.")
        if (
            self.triton_storage is not None
            and "fp8" in {self.triton_fwd, self.triton_bwd}
            and not self.triton_storage.startswith("float8_")
        ):
            raise ValueError("FP8 Triton compute requires FP8 Triton storage.")

    @classmethod
    def from_any(
        cls,
        config: "PrecisionConfig | str | dict | None",
    ) -> "PrecisionConfig":
        r"""
        **API Language** - :ref:`中文 <PrecisionConfig.from_any-cn>` | :ref:`English <PrecisionConfig.from_any-en>`

        ----

        .. _PrecisionConfig.from_any-cn:

        * **中文**

        将 ``None``、mode 字符串、字典或现有配置规范化为
        :class:`PrecisionConfig`。不接受已移除的字段或 mode。

        :param config: 精度配置输入。
        :type config: PrecisionConfig | str | dict | None
        :return: 规范化配置。
        :rtype: PrecisionConfig
        :raises TypeError: 输入类型或字典字段不受支持。
        :raises ValueError: 配置组合无效。

        ----

        .. _PrecisionConfig.from_any-en:

        * **English**

        Normalize ``None``, a mode string, a dictionary, or an existing
        configuration into :class:`PrecisionConfig`. Removed fields and modes
        are rejected.

        :param config: Precision configuration input.
        :type config: PrecisionConfig | str | dict | None
        :return: Normalized configuration.
        :rtype: PrecisionConfig
        :raises TypeError: If the input type or a dictionary field is unsupported.
        :raises ValueError: If the configuration is invalid.
        """
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if isinstance(config, str):
            return cls(mode=config.lower())
        if isinstance(config, dict):
            return cls(**dict(config))

        raise TypeError(
            "PrecisionConfig.from_any() expects None, PrecisionConfig, str, or dict."
        )


PrecisionConfig.__init__.__doc__ = r"""Configure model and Triton-neuron precision.

**API Language** - :ref:`中文 <PrecisionConfig.__init__-cn>` | :ref:`English <PrecisionConfig.__init__-en>`

----

.. _PrecisionConfig.__init__-cn:

* **中文**

``mode`` 控制普通模型算子的精度；``fp8`` 使用 Transformer Engine。
``triton_storage`` 独立启用已有 multi-step Triton IF/LIF/PLIF 节点的 mixed-precision
路径，``triton_fwd`` 和 ``triton_bwd`` 分别控制其前向与反向算术。配置不自动切换
神经元 backend，也不会静默降级。

:param mode: 模型精度模式。
:type mode: Literal["fp32", "fp16", "bf16", "fp8"]
:param fp8_recipe: Transformer Engine FP8 recipe；仅 ``mode="fp8"`` 有效。
:type fp8_recipe: Literal["auto", "delayed", "current", "block", "mxfp8"]
:param triton_storage: Triton 神经元状态 storage dtype；``None`` 禁用 mixed path。
:type triton_storage: Optional[Literal["fp32", "fp16", "bf16",
    "float8_e4m3fn", "float8_e5m2"]]
:param triton_fwd: Triton 神经元前向算术 dtype。
:type triton_fwd: Literal["fp8", "fp16", "bf16", "fp32"]
:param triton_bwd: Triton 神经元反向算术 dtype。
:type triton_bwd: Literal["fp8", "fp16", "bf16", "fp32"]
:raises ValueError: mode、recipe 或 Triton dtype 组合无效。

----

.. _PrecisionConfig.__init__-en:

* **English**

``mode`` controls regular model-operation precision; ``fp8`` uses Transformer
Engine. ``triton_storage`` independently enables the mixed-precision path for
existing multi-step Triton IF/LIF/PLIF nodes, while ``triton_fwd`` and
``triton_bwd`` select forward and backward arithmetic. The configuration neither
changes neuron backends nor silently falls back.

:param mode: Model precision mode.
:type mode: Literal["fp32", "fp16", "bf16", "fp8"]
:param fp8_recipe: Transformer Engine FP8 recipe, valid only for ``mode="fp8"``.
:type fp8_recipe: Literal["auto", "delayed", "current", "block", "mxfp8"]
:param triton_storage: Triton neuron-state storage dtype; ``None`` disables the
    mixed path.
:type triton_storage: Optional[Literal["fp32", "fp16", "bf16",
    "float8_e4m3fn", "float8_e5m2"]]
:param triton_fwd: Triton-neuron forward arithmetic dtype.
:type triton_fwd: Literal["fp8", "fp16", "bf16", "fp32"]
:param triton_bwd: Triton-neuron backward arithmetic dtype.
:type triton_bwd: Literal["fp8", "fp16", "bf16", "fp32"]
:raises ValueError: If a mode, recipe, or Triton dtype combination is invalid.
"""
