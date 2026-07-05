from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import fx

if TYPE_CHECKING:
    from spikingjelly.activation_based.ann2snn.converter import Converter


__all__ = ["ConversionRecipe"]


class ConversionRecipe:
    r"""
    **API Language** - :ref:`中文 <ConversionRecipe-cn>` | :ref:`English <ConversionRecipe-en>`

    ----

    .. _ConversionRecipe-cn:

    * **中文**

    ANN2SNN 转换 recipe 基类。Recipe 是策略对象，只定义
    :class:`~spikingjelly.activation_based.ann2snn.converter.Converter`
    在固定转换模板中每一步应该做什么；Recipe 本身不提供 ``convert``、
    ``run`` 或 ``__call__`` 执行入口。

    子类可以覆盖 :meth:`validate`、:meth:`before_trace`、
    :meth:`after_trace`、:meth:`insert_observers`、:meth:`calibrate`、
    :meth:`replace` 和 :meth:`finalize`。``before_trace`` 接收原始 ANN；
    图步骤接收同一个 ``Converter`` 和当前 ``fx.GraphModule``。步骤可以
    原地修改对象，也必须返回下一步要继续使用的对象。

    ----

    .. _ConversionRecipe-en:

    * **English**

    Base class for ANN2SNN conversion recipes. A recipe is a strategy object
    that defines what each step in the fixed
    :class:`~spikingjelly.activation_based.ann2snn.converter.Converter`
    pipeline should do; the recipe itself does not expose a ``convert``,
    ``run`` or ``__call__`` execution entrypoint.

    Subclasses can override :meth:`validate`, :meth:`before_trace`,
    :meth:`after_trace`, :meth:`insert_observers`, :meth:`calibrate`,
    :meth:`replace` and :meth:`finalize`. ``before_trace`` receives the original
    ANN. Graph steps receive the same ``Converter`` and the current
    ``fx.GraphModule``. They may mutate the object in-place, and must return the
    object that the next step should use.
    """

    def requires_fx_trace(self) -> bool:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.requires_fx_trace-cn>` | :ref:`English <ConversionRecipe.requires_fx_trace-en>`

        ----

        .. _ConversionRecipe.requires_fx_trace-cn:

        * **中文**

        返回该 recipe 是否使用 ``Converter`` 的 FX tracing 图转换模板。
        默认返回 ``True``。不基于 FX 的 recipe 可以返回 ``False``，此时
        ``Converter`` 只执行 ``validate``、``before_trace`` 和 ``finalize``。

        :return: 是否需要 FX tracing。
        :rtype: bool

        ----

        .. _ConversionRecipe.requires_fx_trace-en:

        * **English**

        Return whether this recipe uses the ``Converter`` FX tracing graph
        conversion template. The default is ``True``. Non-FX recipes can return
        ``False``; then ``Converter`` only runs ``validate``, ``before_trace``
        and ``finalize``.

        :return: Whether FX tracing is required.
        :rtype: bool
        """
        return True

    def validate(self, converter: "Converter") -> None:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.validate-cn>` | :ref:`English <ConversionRecipe.validate-en>`

        ----

        .. _ConversionRecipe.validate-cn:

        * **中文**

        校验当前 recipe 的前置条件。默认实现不做任何检查。该方法由
        ``Converter`` 在每次转换开始时调用一次，子类不应在这里执行图转换。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: Converter

        ----

        .. _ConversionRecipe.validate-en:

        * **English**

        Validate this recipe's prerequisites. The default implementation checks
        nothing. ``Converter`` calls this method once at the beginning of each
        conversion; subclasses should not perform graph conversion here.

        :param converter: Converter that executes this recipe.
        :type converter: Converter
        """
        return None

    def before_trace(self, converter: "Converter", ann: nn.Module) -> nn.Module:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.before_trace-cn>` | :ref:`English <ConversionRecipe.before_trace-en>`

        ----

        .. _ConversionRecipe.before_trace-cn:

        * **中文**

        FX tracing 之前运行的步骤。默认直接返回 ``ann``。子类可在此设置
        训练/推理模式，或执行必须发生在 tracing 前的模型准备。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: Converter
        :param ann: 待 trace 的原始 ANN。
        :type ann: torch.nn.Module
        :return: 后续 tracing 使用的 ANN。
        :rtype: torch.nn.Module

        ----

        .. _ConversionRecipe.before_trace-en:

        * **English**

        Step executed before FX tracing. The default implementation returns
        ``ann`` unchanged. Subclasses can set training/eval mode or perform
        model preparation that must happen before tracing.

        :param converter: Converter that executes this recipe.
        :type converter: Converter
        :param ann: Original ANN to be traced.
        :type ann: torch.nn.Module
        :return: ANN used by FX tracing.
        :rtype: torch.nn.Module
        """
        return ann

    def after_trace(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.after_trace-cn>` | :ref:`English <ConversionRecipe.after_trace-en>`

        ----

        .. _ConversionRecipe.after_trace-cn:

        * **中文**

        FX tracing 和 device 转移之后运行的步骤。默认直接返回
        ``fx_model``。子类可在此执行 Conv-BN 融合或做其他 tracing 后预处理；
        影响 FX tracing 的训练/推理模式应在 :meth:`before_trace` 中设置。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: Converter
        :param fx_model: 已 trace 并移动到目标 device 的 ``GraphModule``。
        :type fx_model: torch.fx.GraphModule
        :return: 后续步骤使用的 ``GraphModule``。
        :rtype: torch.fx.GraphModule

        ----

        .. _ConversionRecipe.after_trace-en:

        * **English**

        Step executed after FX tracing and device transfer. The default
        implementation returns ``fx_model`` unchanged. Subclasses can fuse
        Conv-BN modules or perform other post-tracing preprocessing here;
        training/eval mode that affects FX tracing should be set in
        :meth:`before_trace`.

        :param converter: Converter that executes this recipe.
        :type converter: Converter
        :param fx_model: ``GraphModule`` after tracing and device transfer.
        :type fx_model: torch.fx.GraphModule
        :return: ``GraphModule`` used by later steps.
        :rtype: torch.fx.GraphModule
        """
        return fx_model

    def insert_observers(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.insert_observers-cn>` | :ref:`English <ConversionRecipe.insert_observers-en>`

        ----

        .. _ConversionRecipe.insert_observers-cn:

        * **中文**

        插入校准 observer / hook 的步骤。默认不插入任何模块并直接返回
        ``fx_model``。需要校准数据的 recipe 可在此修改 FX 图。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: Converter
        :param fx_model: 当前 ``GraphModule``。
        :type fx_model: torch.fx.GraphModule
        :return: 后续步骤使用的 ``GraphModule``。
        :rtype: torch.fx.GraphModule

        ----

        .. _ConversionRecipe.insert_observers-en:

        * **English**

        Insert calibration observers or hooks. The default implementation
        inserts nothing and returns ``fx_model`` unchanged. Recipes that need
        calibration data can mutate the FX graph here.

        :param converter: Converter that executes this recipe.
        :type converter: Converter
        :param fx_model: Current ``GraphModule``.
        :type fx_model: torch.fx.GraphModule
        :return: ``GraphModule`` used by later steps.
        :rtype: torch.fx.GraphModule
        """
        return fx_model

    def calibrate(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.calibrate-cn>` | :ref:`English <ConversionRecipe.calibrate-en>`

        ----

        .. _ConversionRecipe.calibrate-cn:

        * **中文**

        运行校准数据的步骤。默认不运行 dataloader 并直接返回 ``fx_model``。
        需要校准的子类应自行决定是否使用 ``torch.no_grad()``、如何解析 batch，
        以及如何更新已插入的 observer / hook。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: Converter
        :param fx_model: 当前 ``GraphModule``。
        :type fx_model: torch.fx.GraphModule
        :return: 后续步骤使用的 ``GraphModule``。
        :rtype: torch.fx.GraphModule

        ----

        .. _ConversionRecipe.calibrate-en:

        * **English**

        Run calibration data. The default implementation does not iterate over
        the dataloader and returns ``fx_model`` unchanged. Subclasses that need
        calibration should decide whether to use ``torch.no_grad()``, how to
        parse batches, and how to update inserted observers or hooks.

        :param converter: Converter that executes this recipe.
        :type converter: Converter
        :param fx_model: Current ``GraphModule``.
        :type fx_model: torch.fx.GraphModule
        :return: ``GraphModule`` used by later steps.
        :rtype: torch.fx.GraphModule
        """
        return fx_model

    def replace(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.replace-cn>` | :ref:`English <ConversionRecipe.replace-en>`

        ----

        .. _ConversionRecipe.replace-cn:

        * **中文**

        执行核心替换的步骤，例如将 activation 替换为 spiking neuron，或将 ANN
        module 替换为 TD operator。默认直接返回 ``fx_model``。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: Converter
        :param fx_model: 当前 ``GraphModule``。
        :type fx_model: torch.fx.GraphModule
        :return: 替换后的 ``GraphModule``。
        :rtype: torch.fx.GraphModule

        ----

        .. _ConversionRecipe.replace-en:

        * **English**

        Perform the core replacement step, such as replacing activations with
        spiking neurons or replacing ANN modules with TD operators. The default
        implementation returns ``fx_model`` unchanged.

        :param converter: Converter that executes this recipe.
        :type converter: Converter
        :param fx_model: Current ``GraphModule``.
        :type fx_model: torch.fx.GraphModule
        :return: Replaced ``GraphModule``.
        :rtype: torch.fx.GraphModule
        """
        return fx_model

    def finalize(self, converter: "Converter", fx_model: fx.GraphModule) -> nn.Module:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.finalize-cn>` | :ref:`English <ConversionRecipe.finalize-en>`

        ----

        .. _ConversionRecipe.finalize-cn:

        * **中文**

        转换结束前的收尾步骤。默认直接返回 ``fx_model``。子类可在此做最终
        graph lint、清理临时模块、恢复状态，或包装最终返回的
        :class:`torch.nn.Module`。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: Converter
        :param fx_model: 当前 ``GraphModule``。
        :type fx_model: torch.fx.GraphModule
        :return: 最终转换结果。
        :rtype: torch.nn.Module

        ----

        .. _ConversionRecipe.finalize-en:

        * **English**

        Final step before returning the converted model. The default
        implementation returns ``fx_model`` unchanged. Subclasses can perform
        final graph linting, clean temporary modules, restore state, or wrap the
        final returned :class:`torch.nn.Module`.

        :param converter: Converter that executes this recipe.
        :type converter: Converter
        :param fx_model: Current ``GraphModule``.
        :type fx_model: torch.fx.GraphModule
        :return: Final converted model.
        :rtype: torch.nn.Module
        """
        return fx_model
