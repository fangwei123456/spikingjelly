import math
from typing import Optional, Type

import torch.nn as nn

from spikingjelly.activation_based import neuron


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
                f"neuron_type must be an nn.Module subclass, but got {neuron_type!r}."
            )
        if not isinstance(v_threshold, (int, float)) or isinstance(v_threshold, bool):
            raise TypeError(
                f"v_threshold must be a real number, got {type(v_threshold).__name__}."
            )
        if not math.isfinite(float(v_threshold)) or not (v_threshold > 0):
            raise ValueError(
                f"v_threshold must be finite and positive, got {v_threshold}."
            )
        if v_reset is not None and (
            not isinstance(v_reset, (int, float)) or isinstance(v_reset, bool)
        ):
            raise TypeError(f"v_reset must be None or a real number, got {v_reset!r}.")
        if v_reset is not None and not math.isfinite(float(v_reset)):
            raise ValueError(f"v_reset must be finite, got {v_reset}.")
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
