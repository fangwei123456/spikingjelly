from typing import Optional, Union

import torch
import torch.nn as nn
from torch import fx

from spikingjelly.activation_based.ann2snn.recipes import (
    ConversionRecipe,
    TransformerSpikeEquivalentRecipe,
)


class Converter:
    def __init__(
        self,
        recipe: Union[str, ConversionRecipe],
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <Converter.__init__-cn>` | :ref:`English <Converter.__init__-en>`

        ----

        .. _Converter.__init__-cn:

        * **中文**

        ``Converter`` 是 ANN2SNN 转换框架执行器，而不是具体转换算法。
        它只负责 device 解析、FX tracing 和固定转换模板调度；具体算法参数
        与图变换由 ``recipe`` 定义。

        :param recipe: 转换 recipe。传入
            :class:`~spikingjelly.activation_based.ann2snn.recipes.ConversionRecipe`
            实例，或稳定的内置 recipe 字符串。目前字符串别名仅支持
            ``"transformer_spike_equivalent"``。Rate-coding 转换需要显式传入
            ``RateCodingRecipe(dataloader=...)``。
        :type recipe: str or ConversionRecipe
        :param device: 转换目标 device。若为 ``None``，从模型参数推断；无参数
            模型使用 CPU。
        :type device: torch.device or str or None

        ----

        .. _Converter.__init__-en:

        * **English**

        ``Converter`` is the ANN2SNN conversion framework executor, not a
        concrete conversion algorithm. It only owns device resolution, FX
        tracing and fixed template orchestration; algorithm parameters and graph
        transforms are defined by ``recipe``.

        :param recipe: Conversion recipe. Pass a
            :class:`~spikingjelly.activation_based.ann2snn.recipes.ConversionRecipe`
            instance, or a stable built-in recipe string. Currently, the only
            supported string alias is ``"transformer_spike_equivalent"``.
            Rate-coding conversion must pass
            ``RateCodingRecipe(dataloader=...)`` explicitly.
        :type recipe: str or ConversionRecipe
        :param device: Target conversion device. If ``None``, infer it from the
            model parameters; parameterless models use CPU.
        :type device: torch.device or str or None
        """
        self.recipe = self._resolve_recipe(recipe)
        self.device = device

    @staticmethod
    def _resolve_recipe(recipe: Union[str, ConversionRecipe]) -> ConversionRecipe:
        if isinstance(recipe, ConversionRecipe):
            return recipe
        if recipe == "transformer_spike_equivalent":
            return TransformerSpikeEquivalentRecipe()
        if recipe == "rate_coding":
            raise ValueError(
                "The rate_coding recipe requires parameters. "
                "Pass RateCodingRecipe(dataloader=...) to Converter."
            )
        if isinstance(recipe, str):
            raise ValueError(f"Unknown ann2snn conversion recipe: {recipe!r}.")
        raise TypeError(
            "recipe must be a recipe name string or a ConversionRecipe "
            f"instance, but got {type(recipe).__name__}."
        )

    def _resolve_device(self, ann: nn.Module) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        try:
            return next(ann.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def convert(self, ann: nn.Module) -> torch.fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <Converter.convert-cn>` | :ref:`English <Converter.convert-en>`

        ----

        .. _Converter.convert-cn:

        * **中文**

        按当前 ``recipe`` 执行完整 ANN2SNN 转换模板。``Converter`` 只负责
        device 解析、FX tracing 和步骤调度；recipe 定义每一步的算法行为。
        ``validate`` 在每次转换开始时调用一次，``before_trace`` 在 FX tracing
        前运行。

        :param ann: 待转换的 ANN。
        :type ann: torch.nn.Module
        :return: 转换后的 ``GraphModule``。
        :rtype: torch.fx.GraphModule

        ----

        .. _Converter.convert-en:

        * **English**

        Execute the full ANN2SNN conversion template with the current
        ``recipe``. ``Converter`` only owns device resolution, FX tracing and
        step orchestration; the recipe defines the algorithm behavior of each
        step. ``validate`` is called once at the beginning of each conversion,
        and ``before_trace`` runs before FX tracing.

        :param ann: ANN to be converted.
        :type ann: torch.nn.Module
        :return: Converted ``GraphModule``.
        :rtype: torch.fx.GraphModule
        """
        self.recipe.validate(self)
        configured_device = self.device
        self.device = self._resolve_device(ann)
        try:
            ann = self.recipe.before_trace(self, ann)
            fx_model = fx.symbolic_trace(ann).to(self.device)
            fx_model = self.recipe.after_trace(self, fx_model)
            fx_model = self.recipe.insert_observers(self, fx_model)
            fx_model = self.recipe.calibrate(self, fx_model)
            fx_model = self.recipe.replace(self, fx_model)
            fx_model = self.recipe.finalize(self, fx_model)
            return fx_model
        finally:
            self.device = configured_device
