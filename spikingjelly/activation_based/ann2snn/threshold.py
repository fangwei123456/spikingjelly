from spikingjelly.activation_based.ann2snn.modules import VoltageHook


class ThresholdOptimizer:
    def __init__(self, strategy: str = "fixed"):
        """
        **API Language:**
        :ref:`中文 <ThresholdOptimizer.__init__-cn>` | :ref:`English <ThresholdOptimizer.__init__-en>`

        ----

        .. _ThresholdOptimizer.__init__-cn:

        * **中文**

        阈值优化器。根据 :class:`VoltageHook` 在校准阶段记录的 ``scale`` 计算当前层的
        神经元阈值。当前内置策略：

        * ``"fixed"``: 阈值等于校准 ``scale`` （默认，等价于 SpikingJelly 原有行为）。

        其他策略需通过子类化并重写 :meth:`compute_threshold` 实现；基类可接受任意策略
        名，但只有 ``"fixed"`` 在基类中真正生效。

        :param strategy: 阈值计算策略名称。
        :type strategy: str

        ----

        .. _ThresholdOptimizer.__init__-en:

        * **English**

        Threshold optimizer. Computes the neuron threshold for a layer from the
        ``scale`` recorded by :class:`VoltageHook` during calibration. Built-in
        strategy:

        * ``"fixed"``: threshold equals the calibrated ``scale`` (default,
          matches the original SpikingJelly behaviour).

        Additional strategies should be implemented by subclassing and overriding
        :meth:`compute_threshold`. The base class accepts any strategy name but
        only implements ``"fixed"`` itself.

        :param strategy: Name of the threshold computation strategy.
        :type strategy: str
        """
        self.strategy = strategy

    def compute_threshold(self, hook: VoltageHook) -> float:
        r"""
        **API Language:**
        :ref:`中文 <ThresholdOptimizer.compute_threshold-cn>` | :ref:`English <ThresholdOptimizer.compute_threshold-en>`

        ----

        .. _ThresholdOptimizer.compute_threshold-cn:

        * **中文**

        返回当前层对应的脉冲神经元阈值。当前仅在 ``strategy="fixed"`` 时直接返回
        hook 中记录的 ``scale``；其他策略由子类实现。

        :param hook: 已完成校准的 :class:`VoltageHook`，其 ``scale`` 属性保存激活
            范围统计量。
        :type hook: VoltageHook
        :return: 神经元阈值。
        :rtype: float
        :raises NotImplementedError: 当 ``strategy`` 不是已实现策略时抛出。
        :raises AttributeError: 当 ``hook`` 不包含 ``scale`` 属性时抛出。

        ----

        .. _ThresholdOptimizer.compute_threshold-en:

        * **English**

        Return the spiking-neuron threshold for the layer represented by
        *hook*. With ``strategy="fixed"`` this returns the ``scale`` stored
        in the hook; other strategies should be implemented by subclasses.

        :param hook: A calibrated :class:`VoltageHook` whose ``scale``
            attribute holds the activation range statistic.
        :type hook: VoltageHook
        :return: The neuron threshold.
        :rtype: float
        :raises NotImplementedError: If ``strategy`` is not implemented.
        :raises AttributeError: If ``hook`` does not expose a ``scale``
            attribute.
        """
        if self.strategy == "fixed":
            return hook.scale.item()
        raise NotImplementedError(f"Strategy {self.strategy!r} is not implemented.")
