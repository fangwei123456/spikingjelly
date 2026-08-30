from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Iterable

import torch
from torch.distributed import ProcessGroup

from spikingjelly.logger import logger

from .config import PrecisionConfig
from .convert import _configure_triton_neurons
from .policy import BF16Policy, FP16Policy, FP32Policy

if TYPE_CHECKING:
    from .policy import PrecisionPolicy


def _resolve_policy(config: PrecisionConfig, device: torch.device) -> PrecisionPolicy:
    device_type = device.type
    if config.mode == "fp32":
        return FP32Policy()
    if config.mode == "fp16":
        return FP16Policy(device_type=device_type)
    if config.mode == "bf16":
        return BF16Policy(device_type=device_type)
    from .float8_te import Float8TransformerEnginePolicy

    return Float8TransformerEnginePolicy(
        device_type=device_type,
        fp8_recipe=config.fp8_recipe,
        fp8_fallback_dtype=config.fp8_fallback_dtype,
    )


@dataclass
class PrecisionArtifacts:
    config: PrecisionConfig
    _policy: PrecisionPolicy
    model: torch.nn.Module
    scaler: torch.amp.GradScaler | None = None
    triton_report: dict | None = None

    def autocast_context(
        self, group: ProcessGroup | None = None
    ) -> AbstractContextManager:
        r"""
        **API Language** - :ref:`中文 <PrecisionArtifacts.autocast_context-cn>` | :ref:`English <PrecisionArtifacts.autocast_context-en>`

        ----

        .. _PrecisionArtifacts.autocast_context-cn:

        * **中文**

        返回前向精度上下文。分布式 FP8 runtime 通过 ``group`` 指定用于同步
        Transformer Engine scaling metadata 的进程组。

        :param group: 可选分布式进程组。
        :type group: Optional[torch.distributed.ProcessGroup]
        :return: 前向上下文管理器。
        :rtype: contextlib.AbstractContextManager

        ----

        .. _PrecisionArtifacts.autocast_context-en:

        * **English**

        Return the forward precision context. Distributed FP8 runtimes use
        ``group`` to select the process group that synchronizes Transformer
        Engine scaling metadata.

        :param group: Optional distributed process group.
        :type group: Optional[torch.distributed.ProcessGroup]
        :return: Forward context manager.
        :rtype: contextlib.AbstractContextManager
        """
        return self._policy.autocast_context(group)

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad_norm: float | None = None,
        parameters: Iterable[torch.nn.Parameter] | None = None,
        step_optimizer: bool = True,
    ) -> float | None:
        r"""
        **API Language** - :ref:`中文 <PrecisionArtifacts.backward-cn>` | :ref:`English <PrecisionArtifacts.backward-en>`

        ----

        .. _PrecisionArtifacts.backward-cn:

        * **中文**

        按配置的 GradScaler 执行反向传播，可选梯度裁剪和 optimizer step。

        :param loss: 标量 loss。
        :type loss: torch.Tensor
        :param optimizer: Optimizer。
        :type optimizer: torch.optim.Optimizer
        :param clip_grad_norm: 可选梯度范数上限。
        :type clip_grad_norm: Optional[float]
        :param parameters: 要裁剪的参数；``None`` 使用 prepared model。
        :type parameters: Optional[Iterable[torch.nn.Parameter]]
        :param step_optimizer: 是否执行 optimizer step。
        :type step_optimizer: bool
        :return: 裁剪前梯度范数；未裁剪时为 ``None``。
        :rtype: Optional[float]
        :raises ValueError: GradScaler 生效时请求不执行 optimizer step 的梯度裁剪。

        ----

        .. _PrecisionArtifacts.backward-en:

        * **English**

        Run backward with the configured GradScaler, optional gradient clipping,
        and an optional optimizer step.

        :param loss: Scalar loss.
        :type loss: torch.Tensor
        :param optimizer: Optimizer.
        :type optimizer: torch.optim.Optimizer
        :param clip_grad_norm: Optional gradient-norm limit.
        :type clip_grad_norm: Optional[float]
        :param parameters: Parameters to clip; ``None`` uses the prepared model.
        :type parameters: Optional[Iterable[torch.nn.Parameter]]
        :param step_optimizer: Whether to step the optimizer.
        :type step_optimizer: bool
        :return: Pre-clipping gradient norm, or ``None`` when clipping is disabled.
        :rtype: Optional[float]
        :raises ValueError: If clipping without an optimizer step is requested
            with GradScaler.
        """
        if clip_grad_norm is not None and parameters is None:
            parameters = self.model.parameters()

        grad_norm = None
        if self.scaler is None:
            loss.backward()
            if clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
            if step_optimizer:
                optimizer.step()
        else:
            self.scaler.scale(loss).backward()
            if clip_grad_norm is not None:
                if not step_optimizer:
                    raise ValueError(
                        "clip_grad_norm with step_optimizer=False is not supported "
                        "when a grad scaler is active."
                    )
                self.scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
            if step_optimizer:
                self.scaler.step(optimizer)
                self.scaler.update()

        return float(grad_norm) if grad_norm is not None else None

    def describe(self) -> dict[str, object]:
        r"""
        **API Language** - :ref:`中文 <PrecisionArtifacts.describe-cn>` | :ref:`English <PrecisionArtifacts.describe-en>`

        ----

        .. _PrecisionArtifacts.describe-cn:

        * **中文**

        返回可序列化的配置、model policy、capability、model conversion 与
        Triton neuron conversion 报告。

        :return: 精度诊断。
        :rtype: dict[str, object]

        ----

        .. _PrecisionArtifacts.describe-en:

        * **English**

        Return serializable configuration, model-policy, capability,
        model-conversion, and Triton-neuron conversion reports.

        :return: Precision diagnostics.
        :rtype: dict[str, object]
        """
        return {
            "config": asdict(self.config),
            "policy": self._policy.describe(),
            "capability_report": self._policy.capability_report(),
            "conversion_report": self._policy.conversion_report(),
            "triton_neurons": self.triton_report
            or {"converted_modules": [], "unsupported_modules": []},
        }


def prepare_model_for_precision(
    model: torch.nn.Module,
    device: torch.device | str,
    config: PrecisionConfig | str | dict,
) -> PrecisionArtifacts:
    r"""
    **API Language** - :ref:`中文 <prepare_model_for_precision-cn>` | :ref:`English <prepare_model_for_precision-en>`

    ----

    .. _prepare_model_for_precision-cn:

    * **中文**

    检查 capability，转换模型级 FP8 模块，并配置已有 multi-step Triton
    IF/LIF/PLIF 节点。该函数可能替换模型模块，必须在创建 optimizer 之前调用。

    :param model: 要准备的模型。
    :type model: torch.nn.Module
    :param device: 模型参数与 buffer 所在设备。
    :type device: torch.device | str
    :param config: 精度配置或受支持的简写。
    :type config: PrecisionConfig | str | dict
    :return: Prepared model、contexts、scaler 与报告。
    :rtype: PrecisionArtifacts
    :raises RuntimeError: 依赖、硬件、转换目标或 Triton 组合不可用。

    ----

    .. _prepare_model_for_precision-en:

    * **English**

    Check capabilities, convert model-level FP8 modules, and configure existing
    multi-step Triton IF/LIF/PLIF nodes. This function may replace model modules
    and must run before optimizer construction.

    :param model: Model to prepare.
    :type model: torch.nn.Module
    :param device: Device containing model parameters and buffers.
    :type device: torch.device | str
    :param config: Precision configuration or shorthand.
    :type config: PrecisionConfig | str | dict
    :return: Prepared model, contexts, scaler, and reports.
    :rtype: PrecisionArtifacts
    :raises RuntimeError: If a dependency, hardware capability, conversion target,
        or Triton combination is unavailable.
    """
    start_time = time.perf_counter()
    device = torch.device(device)
    requested = PrecisionConfig.from_any(config)
    policy = _resolve_policy(requested, device)
    policy.check_capability(model, device)
    prepared_model = policy.prepare_model(model)
    conversion_report = policy.conversion_report()
    if requested.mode == "fp8" and not (
        conversion_report["converted_modules"]
        or conversion_report["converted_patterns"]
    ):
        raise RuntimeError("precision='fp8' did not convert any model modules.")
    triton_report = _configure_triton_neurons(prepared_model, requested, device)
    scaler = policy.create_grad_scaler()
    artifacts = PrecisionArtifacts(
        config=requested,
        _policy=policy,
        model=prepared_model,
        scaler=scaler,
        triton_report=triton_report,
    )
    logger.info(
        "Preparation completed: mode={} device={} converted_modules={} triton_neurons={} unsupported_modules={} grad_scaler={} elapsed_ms={:.3f}",
        requested.mode,
        device,
        len(conversion_report.get("converted_modules", ())),
        len(triton_report.get("converted_modules", ())),
        len(conversion_report.get("unsupported_modules", ())),
        scaler is not None,
        (time.perf_counter() - start_time) * 1000.0,
    )
    return artifacts


PrecisionArtifacts.__init__.__doc__ = r"""Hold a prepared model and precision runtime.

**API Language** - :ref:`中文 <PrecisionArtifacts.__init__-cn>` | :ref:`English <PrecisionArtifacts.__init__-en>`

----

.. _PrecisionArtifacts.__init__-cn:

* **中文**

由 :func:`prepare_model_for_precision` 创建；调用方使用 ``model``、
``autocast_context()``、``backward()`` 和 ``describe()``，不直接依赖内部 policy。

:param config: 生效配置。
:type config: PrecisionConfig
:param _policy: 内部 precision policy。
:type _policy: PrecisionPolicy
:param model: Prepared model.
:type model: torch.nn.Module
:param scaler: 可选 GradScaler。
:type scaler: Optional[torch.amp.GradScaler]
:param triton_report: Triton 神经元配置报告。
:type triton_report: Optional[dict]

----

.. _PrecisionArtifacts.__init__-en:

* **English**

Created by :func:`prepare_model_for_precision`. Callers use ``model``,
``autocast_context()``, ``backward()``, and ``describe()`` without depending on
the internal policy.

:param config: Effective configuration.
:type config: PrecisionConfig
:param _policy: Internal precision policy.
:type _policy: PrecisionPolicy
:param model: Prepared model.
:type model: torch.nn.Module
:param scaler: Optional GradScaler.
:type scaler: Optional[torch.amp.GradScaler]
:param triton_report: Triton-neuron configuration report.
:type triton_report: Optional[dict]
"""
