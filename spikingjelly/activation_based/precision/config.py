from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class PrecisionConfig:
    mode: Literal["fp32", "fp16", "bf16", "fp8"] = "fp32"
    fp8_recipe: Literal["auto", "delayed", "current", "block", "mxfp8"] = "auto"
    neuron_storage: Optional[
        Literal[
            "fp32",
            "fp16",
            "bf16",
            "float8_e4m3fn",
            "float8_e5m2",
        ]
    ] = None
    neuron_fwd: Literal["fp8", "fp16", "bf16", "fp32"] = "fp32"
    neuron_bwd: Literal["fp8", "fp16", "bf16", "fp32"] = "fp32"
    fp8_fallback_dtype: Literal["fp32", "fp16", "bf16"] = "bf16"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", str(self.mode).lower())
        object.__setattr__(self, "fp8_recipe", str(self.fp8_recipe).lower())
        object.__setattr__(
            self, "fp8_fallback_dtype", str(self.fp8_fallback_dtype).lower()
        )
        if self.neuron_storage is not None:
            object.__setattr__(
                self,
                "neuron_storage",
                str(self.neuron_storage).lower().removeprefix("torch."),
            )
        object.__setattr__(self, "neuron_fwd", str(self.neuron_fwd).lower())
        object.__setattr__(self, "neuron_bwd", str(self.neuron_bwd).lower())
        if self.mode not in {"fp32", "fp16", "bf16", "fp8"}:
            raise ValueError("mode must be 'fp32', 'fp16', 'bf16', or 'fp8'.")
        if self.fp8_recipe not in {"auto", "delayed", "current", "block", "mxfp8"}:
            raise ValueError("Unsupported fp8_recipe.")
        if self.mode != "fp8" and self.fp8_recipe != "auto":
            raise ValueError("fp8_recipe is only valid when mode='fp8'.")
        if self.fp8_fallback_dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Unsupported fp8_fallback_dtype.")
        if self.mode != "fp8" and self.fp8_fallback_dtype != "bf16":
            raise ValueError("fp8_fallback_dtype is only valid when mode='fp8'.")
        if self.neuron_storage is None and (
            self.neuron_fwd != "fp32" or self.neuron_bwd != "fp32"
        ):
            raise ValueError(
                "neuron_fwd and neuron_bwd require neuron_storage to be set."
            )
        if self.neuron_storage is not None and self.neuron_storage not in {
            "fp32",
            "fp16",
            "bf16",
            "float8_e4m3fn",
            "float8_e5m2",
        }:
            raise ValueError("Unsupported neuron_storage.")
        if self.neuron_fwd not in {"fp8", "fp16", "bf16", "fp32"}:
            raise ValueError("Unsupported neuron_fwd.")
        if self.neuron_bwd not in {"fp8", "fp16", "bf16", "fp32"}:
            raise ValueError("Unsupported neuron_bwd.")
        if (
            self.neuron_storage is not None
            and "fp8" in {self.neuron_fwd, self.neuron_bwd}
            and not self.neuron_storage.startswith("float8_")
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
``neuron_storage`` 为 multi-step IF/LIF/PLIF 节点启用显式混合精度路径，
``neuron_fwd`` 和 ``neuron_bwd`` 分别控制前向与反向算术。该精度路径要求 CUDA
Triton；普通神经元执行仍按设备自动选择。

:param mode: 模型精度模式。
:type mode: Literal["fp32", "fp16", "bf16", "fp8"]
:param fp8_recipe: Transformer Engine FP8 recipe；仅 ``mode="fp8"`` 有效。
:type fp8_recipe: Literal["auto", "delayed", "current", "block", "mxfp8"]
:param neuron_storage: 神经元状态 storage dtype；``None`` 禁用 mixed path。显式精度要求 CUDA Triton。
:type neuron_storage: Optional[Literal["fp32", "fp16", "bf16",
    "float8_e4m3fn", "float8_e5m2"]]
:param neuron_fwd: 神经元前向算术 dtype。
:type neuron_fwd: Literal["fp8", "fp16", "bf16", "fp32"]
:param neuron_bwd: 神经元反向算术 dtype。
:type neuron_bwd: Literal["fp8", "fp16", "bf16", "fp32"]
:param fp8_fallback_dtype: 未由 Transformer Engine 转换的普通 CUDA 算子使用的
    fallback autocast dtype，默认为 ``bf16``；``fp32`` 表示不启用外层 autocast。
:type fp8_fallback_dtype: Literal["fp32", "fp16", "bf16"]
:raises ValueError: mode、recipe 或 Triton dtype 组合无效。

----

.. _PrecisionConfig.__init__-en:

* **English**

``mode`` controls regular model-operation precision; ``fp8`` uses Transformer
Engine. ``neuron_storage`` enables an explicit mixed-precision path for
multi-step IF/LIF/PLIF nodes, while ``neuron_fwd`` and ``neuron_bwd`` select
forward and backward arithmetic. This precision path requires CUDA Triton;
ordinary neuron execution remains automatically selected from the device.

:param mode: Model precision mode.
:type mode: Literal["fp32", "fp16", "bf16", "fp8"]
:param fp8_recipe: Transformer Engine FP8 recipe, valid only for ``mode="fp8"``.
:type fp8_recipe: Literal["auto", "delayed", "current", "block", "mxfp8"]
:param neuron_storage: Neuron-state storage dtype; ``None`` disables the mixed path. Explicit
    precision requires CUDA Triton.
:type neuron_storage: Optional[Literal["fp32", "fp16", "bf16",
    "float8_e4m3fn", "float8_e5m2"]]
:param neuron_fwd: Neuron forward arithmetic dtype.
:type neuron_fwd: Literal["fp8", "fp16", "bf16", "fp32"]
:param neuron_bwd: Neuron backward arithmetic dtype.
:type neuron_bwd: Literal["fp8", "fp16", "bf16", "fp32"]
:param fp8_fallback_dtype: Fallback autocast dtype for ordinary CUDA operations
    not converted by Transformer Engine. The default is ``bf16``; ``fp32``
    disables the outer autocast.
:type fp8_fallback_dtype: Literal["fp32", "fp16", "bf16"]
:raises ValueError: If a mode, recipe, or Triton dtype combination is invalid.
"""
