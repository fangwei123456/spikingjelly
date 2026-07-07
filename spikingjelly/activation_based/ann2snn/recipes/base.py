from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import fx

if TYPE_CHECKING:
    from spikingjelly.activation_based.ann2snn.converter import (
        FXConverter,
        ModuleConverter,
    )


__all__ = [
    "ConversionRecipe",
    "FXConversionRecipe",
    "ModuleConversionRecipe",
]


class FXConversionRecipe:
    r"""
    **API Language** - :ref:`中文 <ConversionRecipe-cn>` | :ref:`English <ConversionRecipe-en>`

    ----

    .. _ConversionRecipe-cn:

    * **中文**

    FX graph 路径的 ANN2SNN 转换 recipe 基类。兼容名
    ``ConversionRecipe`` 等价于 ``FXConversionRecipe``。Recipe 是策略对象，
    只定义 :class:`~spikingjelly.activation_based.ann2snn.converter.FXConverter`
    在固定 FX 转换模板中每一步应该做什么；Recipe 本身不提供 ``convert``、
    ``run`` 或 ``__call__`` 执行入口。

    子类可以覆盖 :meth:`validate`、:meth:`before_trace`、
    :meth:`after_trace`、:meth:`insert_observers`、:meth:`calibrate`、
    :meth:`replace` 和 :meth:`finalize`。``before_trace`` 接收原始 ANN；
    图步骤接收同一个 ``FXConverter`` 和当前 ``fx.GraphModule``。步骤可以
    原地修改对象，也必须返回下一步要继续使用的对象。

    ----

    .. _ConversionRecipe-en:

    * **English**

    Base class for FX graph ANN2SNN conversion recipes. The compatibility name
    ``ConversionRecipe`` is equivalent to ``FXConversionRecipe``. A recipe is a
    strategy object that defines what each step in the fixed
    :class:`~spikingjelly.activation_based.ann2snn.converter.FXConverter`
    pipeline should do; the recipe itself does not expose a ``convert``,
    ``run`` or ``__call__`` execution entrypoint.

    Subclasses can override :meth:`validate`, :meth:`before_trace`,
    :meth:`after_trace`, :meth:`insert_observers`, :meth:`calibrate`,
    :meth:`replace` and :meth:`finalize`. ``before_trace`` receives the original
    ANN. Graph steps receive the same ``FXConverter`` and the current
    ``fx.GraphModule``. They may mutate the object in-place, and must return the
    object that the next step should use.
    """

    def validate(self, converter: "FXConverter") -> None:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.validate-cn>` | :ref:`English <ConversionRecipe.validate-en>`

        ----

        .. _ConversionRecipe.validate-cn:

        * **中文**

        校验当前 recipe 的前置条件。默认实现不做任何检查。该方法由
        ``FXConverter`` / ``Converter`` 在每次转换开始时调用一次，子类不应
        在这里执行图转换。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: FXConverter

        ----

        .. _ConversionRecipe.validate-en:

        * **English**

        Validate this recipe's prerequisites. The default implementation checks
        nothing. ``FXConverter`` / ``Converter`` calls this method once at the
        beginning of each conversion; subclasses should not perform graph
        conversion here.

        :param converter: Converter that executes this recipe.
        :type converter: FXConverter
        """
        return None

    def before_trace(self, converter: "FXConverter", ann: nn.Module) -> nn.Module:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.before_trace-cn>` | :ref:`English <ConversionRecipe.before_trace-en>`

        ----

        .. _ConversionRecipe.before_trace-cn:

        * **中文**

        FX tracing 之前运行的步骤。默认直接返回 ``ann``。子类可在此设置
        训练/推理模式，或执行必须发生在 tracing 前的模型准备。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: FXConverter
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
        :type converter: FXConverter
        :param ann: Original ANN to be traced.
        :type ann: torch.nn.Module
        :return: ANN used by FX tracing.
        :rtype: torch.nn.Module
        """
        return ann

    def after_trace(
        self, converter: "FXConverter", fx_model: fx.GraphModule
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
        :type converter: FXConverter
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
        :type converter: FXConverter
        :param fx_model: ``GraphModule`` after tracing and device transfer.
        :type fx_model: torch.fx.GraphModule
        :return: ``GraphModule`` used by later steps.
        :rtype: torch.fx.GraphModule
        """
        return fx_model

    def insert_observers(
        self, converter: "FXConverter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.insert_observers-cn>` | :ref:`English <ConversionRecipe.insert_observers-en>`

        ----

        .. _ConversionRecipe.insert_observers-cn:

        * **中文**

        插入校准 observer / hook 的步骤。默认不插入任何模块并直接返回
        ``fx_model``。需要校准数据的 recipe 可在此修改 FX 图。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: FXConverter
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
        :type converter: FXConverter
        :param fx_model: Current ``GraphModule``.
        :type fx_model: torch.fx.GraphModule
        :return: ``GraphModule`` used by later steps.
        :rtype: torch.fx.GraphModule
        """
        return fx_model

    def calibrate(
        self, converter: "FXConverter", fx_model: fx.GraphModule
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
        :type converter: FXConverter
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
        :type converter: FXConverter
        :param fx_model: Current ``GraphModule``.
        :type fx_model: torch.fx.GraphModule
        :return: ``GraphModule`` used by later steps.
        :rtype: torch.fx.GraphModule
        """
        return fx_model

    def replace(
        self, converter: "FXConverter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.replace-cn>` | :ref:`English <ConversionRecipe.replace-en>`

        ----

        .. _ConversionRecipe.replace-cn:

        * **中文**

        执行核心替换的步骤，例如将 activation 替换为 spiking neuron，或将 ANN
        module 替换为 TD operator。默认直接返回 ``fx_model``。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: FXConverter
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
        :type converter: FXConverter
        :param fx_model: Current ``GraphModule``.
        :type fx_model: torch.fx.GraphModule
        :return: Replaced ``GraphModule``.
        :rtype: torch.fx.GraphModule
        """
        return fx_model

    def finalize(self, converter: "FXConverter", fx_model: fx.GraphModule) -> nn.Module:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.finalize-cn>` | :ref:`English <ConversionRecipe.finalize-en>`

        ----

        .. _ConversionRecipe.finalize-cn:

        * **中文**

        转换结束前的收尾步骤。默认直接返回 ``fx_model``。子类可在此做最终
        graph lint、清理临时模块、恢复状态，或包装最终返回的
        :class:`torch.nn.Module`。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: FXConverter
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
        :type converter: FXConverter
        :param fx_model: Current ``GraphModule``.
        :type fx_model: torch.fx.GraphModule
        :return: Final converted model.
        :rtype: torch.nn.Module
        """
        return fx_model


ConversionRecipe = FXConversionRecipe


class ModuleConversionRecipe:
    r"""
    **API Language** - :ref:`中文 <ModuleConversionRecipe-cn>` | :ref:`English <ModuleConversionRecipe-en>`

    ----

    .. _ModuleConversionRecipe-cn:

    * **中文**

    直接 ``nn.Module`` tree 转换 recipe 基类。该路径不执行 FX tracing，
    只由 :class:`~spikingjelly.activation_based.ann2snn.converter.ModuleConverter`
    调用 :meth:`validate` 和 :meth:`convert_module`。适用于 SpikeZIP 这类
    需要按 module tree 替换子模块、但不改写 FX graph 的转换。该基类没有
    ``before_trace``、``after_trace``、``insert_observers``、``calibrate``、
    ``replace`` 或 ``finalize`` 生命周期。

    ----

    .. _ModuleConversionRecipe-en:

    * **English**

    Base class for direct ``nn.Module`` tree conversion recipes. This path does
    not run FX tracing. :class:`~spikingjelly.activation_based.ann2snn.converter.ModuleConverter`
    only calls :meth:`validate` and :meth:`convert_module`. It is intended for
    conversions such as SpikeZIP that replace submodules in a module tree
    without rewriting an FX graph. This base class has no ``before_trace``,
    ``after_trace``, ``insert_observers``, ``calibrate``, ``replace`` or
    ``finalize`` lifecycle.
    """

    def validate(self, converter: "ModuleConverter") -> None:
        r"""
        **API Language** - :ref:`中文 <ModuleConversionRecipe.validate-cn>` | :ref:`English <ModuleConversionRecipe.validate-en>`

        ----

        .. _ModuleConversionRecipe.validate-cn:

        * **中文**

        校验 module-tree recipe 的前置条件。默认实现不做任何检查。

        :param converter: 执行当前 recipe 的 module converter。
        :type converter: ModuleConverter

        ----

        .. _ModuleConversionRecipe.validate-en:

        * **English**

        Validate prerequisites for a module-tree recipe. The default
        implementation checks nothing.

        :param converter: Module converter that executes this recipe.
        :type converter: ModuleConverter
        """
        return None

    def convert_module(
        self,
        converter: "ModuleConverter",
        ann: nn.Module,
    ) -> nn.Module:
        r"""
        **API Language** - :ref:`中文 <ModuleConversionRecipe.convert_module-cn>` | :ref:`English <ModuleConversionRecipe.convert_module-en>`

        ----

        .. _ModuleConversionRecipe.convert_module-cn:

        * **中文**

        执行直接 module-tree 转换。默认直接返回 ``ann``。

        :param converter: 执行当前 recipe 的 module converter。
        :type converter: ModuleConverter
        :param ann: 待转换的原始 ANN 或 QANN。
        :type ann: torch.nn.Module
        :return: 转换后的模型。
        :rtype: torch.nn.Module

        ----

        .. _ModuleConversionRecipe.convert_module-en:

        * **English**

        Execute direct module-tree conversion. The default implementation
        returns ``ann`` unchanged.

        :param converter: Module converter that executes this recipe.
        :type converter: ModuleConverter
        :param ann: Original ANN or QANN to convert.
        :type ann: torch.nn.Module
        :return: Converted model.
        :rtype: torch.nn.Module
        """
        return ann
