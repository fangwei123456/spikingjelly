import threading
import types
import warnings
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import fx

from spikingjelly.activation_based.ann2snn.recipes import (
    FXConversionRecipe,
    ModuleConversionRecipe,
    TransformerTDEquivalentRecipe,
)


_FX_TRACE_LOCK = threading.RLock()


def _symbolic_trace(root: nn.Module) -> fx.GraphModule:
    with _FX_TRACE_LOCK:
        original_reshape = torch.reshape

        def proxy_aware_reshape(input, shape):
            if isinstance(input, fx.Proxy):
                return input.tracer.create_proxy(
                    "call_function", original_reshape, (input, shape), {}
                )
            return original_reshape(input, shape)

        # Torch 2.6/2.7 validates reshape's shape before dispatching FX Proxy inputs.
        torch.reshape = proxy_aware_reshape
        try:
            tracer = fx.Tracer()
            graph = tracer.trace(root)
        finally:
            torch.reshape = original_reshape

    # Torch 2.6/2.7 FX codegen cannot register repeated PEP 604 union types.
    for node in graph.nodes:
        if isinstance(node.type, types.UnionType):
            node.type = None
    return fx.GraphModule(tracer.root, graph)


class FXConverter:
    def __init__(
        self,
        recipe: Union[str, FXConversionRecipe],
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <Converter.__init__-cn>` | :ref:`English <Converter.__init__-en>`

        ----

        .. _Converter.__init__-cn:

        * **中文**

        ``FXConverter`` 是 FX graph 路径的 ANN2SNN 转换框架执行器。
        兼容名 ``Converter`` 等价于 ``FXConverter``。它只负责 device 解析、
        FX tracing 和固定转换模板调度；具体算法参数与图变换由 ``recipe`` 定义。

        :param recipe: 转换 recipe。传入
            :class:`~spikingjelly.activation_based.ann2snn.recipes.FXConversionRecipe`
            实例（或兼容名 ``ConversionRecipe``），或稳定的内置 recipe 字符串。
            目前字符串别名仅支持 ``"transformer_td_equivalent"``。
            Rate-coding、STA Transformer 需要显式传入带参数的 recipe 对象。
            SpikeZIP QANN 等 module-tree recipe 使用 :class:`ModuleConverter`。
        :type recipe: str or FXConversionRecipe
        :param device: 转换目标 device。若为 ``None``，从模型参数推断；无参数
            模型使用 CPU。
        :type device: torch.device or str or None

        ----

        .. _Converter.__init__-en:

        * **English**

        ``FXConverter`` is the FX graph ANN2SNN conversion framework executor.
        The compatibility name ``Converter`` is equivalent to ``FXConverter``.
        It only owns device resolution, FX tracing and fixed template
        orchestration; algorithm parameters and graph transforms are defined by
        ``recipe``.

        :param recipe: Conversion recipe. Pass a
            :class:`~spikingjelly.activation_based.ann2snn.recipes.FXConversionRecipe`
            instance (or the compatibility name ``ConversionRecipe``), or a
            stable built-in recipe string. Currently, the only supported string
            alias is ``"transformer_td_equivalent"``. Rate-coding and STA
            Transformer conversion must pass explicit recipe objects.
            Module-tree recipes such as SpikeZIP QANN use :class:`ModuleConverter`
            instead.
        :type recipe: str or FXConversionRecipe
        :param device: Target conversion device. If ``None``, infer it from the
            model parameters; parameterless models use CPU.
        :type device: torch.device or str or None
        """
        self.recipe = self._resolve_recipe(recipe)
        self.device = device

    @staticmethod
    def _resolve_recipe(recipe: Union[str, FXConversionRecipe]) -> FXConversionRecipe:
        if isinstance(recipe, FXConversionRecipe):
            return recipe
        if isinstance(recipe, ModuleConversionRecipe):
            raise TypeError(
                "FXConverter/Converter requires an FXConversionRecipe. "
                "Use ModuleConverter for ModuleConversionRecipe instances."
            )
        if recipe == "transformer_spike_equivalent":
            warnings.warn(
                "The 'transformer_spike_equivalent' recipe string is deprecated; "
                "use 'transformer_td_equivalent' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return TransformerTDEquivalentRecipe()
        if recipe == "transformer_td_equivalent":
            return TransformerTDEquivalentRecipe()
        if recipe == "rate_coding":
            raise ValueError(
                "The rate_coding recipe requires parameters. "
                "Pass RateCodingRecipe(dataloader=...) to Converter."
            )
        if recipe == "sta_transformer":
            raise ValueError(
                "The sta_transformer recipe requires parameters. "
                "Pass STATransformerRecipe(dataloader=..., time_steps=...) "
                "to Converter."
            )
        if isinstance(recipe, str):
            raise ValueError(f"Unknown ann2snn conversion recipe: {recipe!r}.")
        raise TypeError(
            "recipe must be a recipe name string or an FXConversionRecipe "
            f"instance, but got {type(recipe).__name__}."
        )

    def _resolve_device(self, ann: nn.Module) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        try:
            return next(ann.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def convert(self, ann: nn.Module) -> nn.Module:
        r"""
        **API Language** - :ref:`中文 <Converter.convert-cn>` | :ref:`English <Converter.convert-en>`

        ----

        .. _Converter.convert-cn:

        * **中文**

        按当前 ``recipe`` 执行完整 FX ANN2SNN 转换模板。``FXConverter`` 只负责
        device 解析、FX tracing 和步骤调度；recipe 定义每一步的算法行为。
        ``validate`` 在每次转换开始时调用一次，``before_trace`` 在 FX tracing
        前运行。

        :param ann: 待转换的 ANN。
        :type ann: torch.nn.Module
        :return: 转换后的模型。
        :rtype: torch.nn.Module

        ----

        .. _Converter.convert-en:

        * **English**

        Execute the full FX ANN2SNN conversion template with the current
        ``recipe``. ``FXConverter`` only owns device resolution, FX tracing and
        step orchestration; the recipe defines the algorithm behavior of each
        step. ``validate`` is called once at the beginning of each conversion,
        and ``before_trace`` runs before FX tracing.

        :param ann: ANN to be converted.
        :type ann: torch.nn.Module
        :return: Converted model.
        :rtype: torch.nn.Module
        """
        configured_device = self.device
        original_training_modes: dict[nn.Module, bool] = {}
        try:
            original_training_modes = {
                module: module.training for module in ann.modules()
            }
            self.device = self._resolve_device(ann)
            with torch.no_grad():
                self.recipe.validate(self)
                ann = self.recipe.before_trace(self, ann)
                fx_model = _symbolic_trace(ann).to(self.device)
                fx_model = self.recipe.after_trace(self, fx_model)
                fx_model = self.recipe.insert_observers(self, fx_model)
                fx_model = self.recipe.calibrate(self, fx_model)
                fx_model = self.recipe.replace(self, fx_model)
                fx_model = self.recipe.finalize(self, fx_model)
                return fx_model
        finally:
            for module, training in original_training_modes.items():
                module.training = training
            self.device = configured_device


class ModuleConverter:
    def __init__(
        self,
        recipe: ModuleConversionRecipe,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <ModuleConverter.__init__-cn>` | :ref:`English <ModuleConverter.__init__-en>`

        ----

        .. _ModuleConverter.__init__-cn:

        * **中文**

        ``ModuleConverter`` 是直接 ``nn.Module`` tree 路径的 ANN2SNN 转换
        执行器。它不执行 FX tracing，也不是 ``Converter`` 的自动分发分支。
        固定生命周期为 device 解析、保存原始 training 状态、在
        ``torch.no_grad()`` 中调用 ``recipe.validate(self)`` 与
        ``recipe.convert_module(self, ann)``，再把转换产物移动到目标 device。

        :param recipe: module-tree 转换 recipe。
        :type recipe: ModuleConversionRecipe
        :param device: 转换目标 device。若为 ``None``，从模型参数推断；无参数
            模型使用 CPU。
        :type device: torch.device or str or None
        :raises TypeError: ``recipe`` 不是 ``ModuleConversionRecipe``，或传入了
            FX recipe。

        ----

        .. _ModuleConverter.__init__-en:

        * **English**

        ``ModuleConverter`` is the ANN2SNN executor for direct ``nn.Module``
        tree conversion. It does not run FX tracing and is not an automatic
        dispatch branch of ``Converter``. Its fixed lifecycle resolves the
        target device, saves original training states, calls
        ``recipe.validate(self)`` and ``recipe.convert_module(self, ann)``
        under ``torch.no_grad()``, and moves the converted model to the target
        device.

        :param recipe: Module-tree conversion recipe.
        :type recipe: ModuleConversionRecipe
        :param device: Target conversion device. If ``None``, infer it from the
            model parameters; parameterless models use CPU.
        :type device: torch.device or str or None
        :raises TypeError: If ``recipe`` is not a ``ModuleConversionRecipe`` or
            an FX recipe is passed.
        """
        if isinstance(recipe, FXConversionRecipe):
            raise TypeError(
                "ModuleConverter requires a ModuleConversionRecipe. "
                "Use FXConverter/Converter for FXConversionRecipe instances."
            )
        if not isinstance(recipe, ModuleConversionRecipe):
            raise TypeError(
                "recipe must be a ModuleConversionRecipe instance, "
                f"but got {type(recipe).__name__}."
            )
        self.recipe = recipe
        self.device = device

    def _resolve_device(self, ann: nn.Module) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        try:
            return next(ann.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def convert(self, ann: nn.Module) -> nn.Module:
        r"""
        **API Language** - :ref:`中文 <ModuleConverter.convert-cn>` | :ref:`English <ModuleConverter.convert-en>`

        ----

        .. _ModuleConverter.convert-cn:

        * **中文**

        执行直接 module-tree 转换。

        :param ann: 待转换的原始 ANN 或 QANN。
        :type ann: torch.nn.Module
        :return: 转换后的模型。
        :rtype: torch.nn.Module

        ----

        .. _ModuleConverter.convert-en:

        * **English**

        Execute direct module-tree conversion.

        :param ann: Original ANN or QANN to convert.
        :type ann: torch.nn.Module
        :return: Converted model.
        :rtype: torch.nn.Module
        """
        configured_device = self.device
        original_training_modes: dict[nn.Module, bool] = {}
        try:
            original_training_modes = {
                module: module.training for module in ann.modules()
            }
            self.device = self._resolve_device(ann)
            with torch.no_grad():
                self.recipe.validate(self)
                converted = self.recipe.convert_module(self, ann)
                if not isinstance(converted, nn.Module):
                    raise TypeError(
                        "ModuleConversionRecipe.convert_module must return "
                        "a torch.nn.Module, got "
                        f"{type(converted).__name__}."
                    )
                return converted.to(self.device)
        finally:
            for module, training in original_training_modes.items():
                module.training = training
            self.device = configured_device


Converter = FXConverter
