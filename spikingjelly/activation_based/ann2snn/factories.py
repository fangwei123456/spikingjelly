from typing import Optional, Type, Union

import torch.nn as nn

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.ann2snn.modules import VoltageHook


class NeuronFactory:
    def __init__(
        self,
        neuron_type: Type[nn.Module] = neuron.IFNode,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        **kwargs,
    ):
        """
        **API Language** - :ref:`中文 <NeuronFactory.__init__-cn>` | :ref:`English <NeuronFactory.__init__-en>`

        ----

        .. _NeuronFactory.__init__-cn:

        * **中文**

        用于创建替换激活函数的脉冲神经元模块。默认创建
        :class:`spikingjelly.activation_based.neuron.IFNode`，并使用
        ``v_threshold=1.0`` 与 ``v_reset=None`` 保持原有 ANN2SNN 行为。默认转换会通过
        :class:`VoltageScaler` 处理激活尺度，因此默认工厂不会把 ``scale`` 直接写入
        神经元阈值；自定义工厂可读取 ``scale`` 派生阈值或其他参数。

        :param neuron_type: 神经元类，必须接受 ``v_threshold`` 与 ``v_reset`` 关键字参数。
            默认为 :class:`spikingjelly.activation_based.neuron.IFNode`。
        :type neuron_type: Type[nn.Module]
        :param v_threshold: 神经元发放阈值，传递给神经元构造函数。
        :type v_threshold: float
        :param v_reset: 膜电位复位值。``None`` 表示软复位（减法复位），默认为 ``None``。
        :type v_reset: Optional[float]
        :param kwargs: 透传给神经元构造函数的其他关键字参数。

        ----

        .. _NeuronFactory.__init__-en:

        * **English**

        Factory that creates spiking-neuron modules used to replace ANN activation
        functions. By default it instantiates
        :class:`spikingjelly.activation_based.neuron.IFNode` with
        ``v_threshold=1.0`` and ``v_reset=None`` to preserve the original
        ANN2SNN behaviour. The default conversion handles the activation scale
        with :class:`VoltageScaler`, so the default factory does not copy
        ``scale`` into the neuron threshold. Custom factories may derive
        thresholds or other neuron parameters from ``scale``.

        :param neuron_type: Neuron class to instantiate. Must accept
            ``v_threshold`` and ``v_reset`` keyword arguments. Defaults to
            :class:`spikingjelly.activation_based.neuron.IFNode`.
        :type neuron_type: Type[nn.Module]
        :param v_threshold: Firing threshold passed to the neuron constructor.
        :type v_threshold: float
        :param v_reset: Membrane reset value. ``None`` means soft reset
            (subtractive reset). Defaults to ``None``.
        :type v_reset: Optional[float]
        :param kwargs: Additional keyword arguments forwarded to the neuron
            constructor.
        """
        if not isinstance(neuron_type, type) or not issubclass(neuron_type, nn.Module):
            raise TypeError(
                "neuron_type must be an nn.Module subclass, "
                f"but got {neuron_type!r}."
            )
        if not isinstance(v_threshold, (int, float)) or isinstance(v_threshold, bool):
            raise TypeError(
                "v_threshold must be a real number, "
                f"got {type(v_threshold).__name__}."
            )
        if not (v_threshold > 0):
            raise ValueError(f"v_threshold must be positive, got {v_threshold}.")
        if (
            v_reset is not None
            and (
                not isinstance(v_reset, (int, float))
                or isinstance(v_reset, bool)
            )
        ):
            raise TypeError(
                "v_reset must be None or a real number, "
                f"got {v_reset!r}."
            )
        reserved = self.neuron_kwargs_reserved_keys() & kwargs.keys()
        if reserved:
            names = ", ".join(sorted(reserved))
            raise TypeError(
                f"neuron kwargs contain reserved key(s): {names}. "
                "Use NeuronFactory parameters instead."
            )
        self.neuron_type = neuron_type
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.neuron_kwargs = kwargs

    @staticmethod
    def neuron_kwargs_reserved_keys() -> set[str]:
        return {"v_threshold", "v_reset"}

    def create(self, scale: float) -> nn.Module:
        r"""
        **API Language** - :ref:`中文 <NeuronFactory.create-cn>` | :ref:`English <NeuronFactory.create-en>`

        ----

        .. _NeuronFactory.create-cn:

        * **中文**

        根据工厂配置创建一个脉冲神经元模块实例。``scale`` 为当前层校准得到的激活
        尺度，默认实现不直接使用该值，但子类可据此派生阈值或其他参数。

        :param scale: 当前层的校准尺度。
        :type scale: float
        :return: 配置完成的脉冲神经元模块。
        :rtype: nn.Module

        ----

        .. _NeuronFactory.create-en:

        * **English**

        Instantiate a spiking-neuron module with the configured parameters.
        ``scale`` is the calibrated activation scale of the current layer; the
        default implementation does not use it directly, but subclasses can
        derive thresholds or other neuron parameters from it.

        :param scale: Calibration scale for the layer.
        :type scale: float
        :return: A spiking-neuron module.
        :rtype: nn.Module
        """
        return self.neuron_type(
            v_threshold=self.v_threshold,
            v_reset=self.v_reset,
            **self.neuron_kwargs,
        )


class HookFactory:
    def __init__(self, mode: Union[str, float] = "Max", momentum: float = 0.1):
        """
        **API Language** - :ref:`中文 <HookFactory.__init__-cn>` | :ref:`English <HookFactory.__init__-en>`

        ----

        .. _HookFactory.__init__-cn:

        * **中文**

        用于创建校准阶段使用的 :class:`VoltageHook` 实例。每个匹配到的激活节点会获得
        独立的 hook 实例。

        :param mode: 校准模式，传递给 :class:`VoltageHook`。``"Max"`` 记录激活最大值；
            ``"99.9%"`` 记录 99.9 分位点；``(0, 1]`` 区间的 float 表示
            ``max * mode``。
        :type mode: str, float
        :param momentum: :class:`VoltageHook` 的 EMA 动量。
        :type momentum: float

        ----

        .. _HookFactory.__init__-en:

        * **English**

        Factory that creates :class:`VoltageHook` instances used during
        calibration. Each matched activation node receives an independent hook
        instance.

        :param mode: Calibration mode forwarded to :class:`VoltageHook`.
            ``"Max"`` records the maximum activation; ``"99.9%"`` records the
            99.9-th percentile; a float in ``(0, 1]`` records ``max * mode``.
        :type mode: str, float
        :param momentum: EMA momentum for :class:`VoltageHook`.
        :type momentum: float
        """
        if isinstance(mode, str):
            if not mode:
                raise ValueError("mode must be 'Max', a percentile string, or a float.")
            if mode[-1] == "%":
                try:
                    percentile = float(mode[:-1])
                except ValueError as exc:
                    raise ValueError(
                        "mode percentile string must contain a numeric value."
                    ) from exc
                if not (0.0 < percentile <= 100.0):
                    raise ValueError(
                        f"mode percentile must lie in (0, 100], got {mode!r}."
                    )
            elif mode.lower() != "max":
                raise ValueError(
                    f"mode string must be 'Max' or a percentile string, got {mode!r}."
                )
        elif isinstance(mode, (int, float)) and not isinstance(mode, bool):
            if not (0.0 < float(mode) <= 1.0):
                raise ValueError(f"mode float must lie in (0, 1], got {mode!r}.")
        else:
            raise TypeError(
                "mode must be a string or a float in (0, 1], "
                f"got {type(mode).__name__}."
            )
        if not isinstance(momentum, (int, float)) or isinstance(momentum, bool):
            raise TypeError(
                "momentum must be a real number, "
                f"got {type(momentum).__name__}."
            )
        if not (0.0 <= float(momentum) <= 1.0):
            raise ValueError(f"momentum must lie in [0, 1], got {momentum!r}.")
        self.mode = mode
        self.momentum = momentum

    def create(self) -> VoltageHook:
        r"""
        **API Language** - :ref:`中文 <HookFactory.create-cn>` | :ref:`English <HookFactory.create-en>`

        ----

        .. _HookFactory.create-cn:

        * **中文**

        创建一个新的 :class:`VoltageHook` 实例。

        :return: 配置完成的 :class:`VoltageHook`。
        :rtype: VoltageHook

        ----

        .. _HookFactory.create-en:

        * **English**

        Create a new :class:`VoltageHook` instance.

        :return: A configured :class:`VoltageHook`.
        :rtype: VoltageHook
        """
        return VoltageHook(momentum=self.momentum, mode=self.mode)
