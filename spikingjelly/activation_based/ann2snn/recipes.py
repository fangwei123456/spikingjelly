from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    TYPE_CHECKING,
    Type,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fx
from torch.nn.utils.fusion import fuse_conv_bn_eval
from tqdm import tqdm

from spikingjelly.activation_based.ann2snn.factories import HookFactory, NeuronFactory
from spikingjelly.activation_based.ann2snn.operators import (
    TDGELU,
    TDLayerNorm,
    TDLinear,
    TDModule,
    TDMultiheadAttention,
    TDScaledDotProductAttention,
)
from spikingjelly.activation_based.ann2snn.rules import ActivationRule, ReLURule
from spikingjelly.activation_based.ann2snn.threshold import ThresholdOptimizer

if TYPE_CHECKING:
    from spikingjelly.activation_based.ann2snn.converter import Converter


__all__ = [
    "ConversionRecipe",
    "RateCodingRecipe",
    "TransformerSpikeEquivalentRecipe",
]


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

    def finalize(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <ConversionRecipe.finalize-cn>` | :ref:`English <ConversionRecipe.finalize-en>`

        ----

        .. _ConversionRecipe.finalize-cn:

        * **中文**

        转换结束前的收尾步骤。默认直接返回 ``fx_model``。子类可在此做最终
        graph lint、清理临时模块或恢复状态。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: Converter
        :param fx_model: 当前 ``GraphModule``。
        :type fx_model: torch.fx.GraphModule
        :return: 最终转换结果。
        :rtype: torch.fx.GraphModule

        ----

        .. _ConversionRecipe.finalize-en:

        * **English**

        Final step before returning the converted model. The default
        implementation returns ``fx_model`` unchanged. Subclasses can perform
        final graph linting, clean temporary modules, or restore state here.

        :param converter: Converter that executes this recipe.
        :type converter: Converter
        :param fx_model: Current ``GraphModule``.
        :type fx_model: torch.fx.GraphModule
        :return: Final converted model.
        :rtype: torch.fx.GraphModule
        """
        return fx_model


class RateCodingRecipe(ConversionRecipe):
    def __init__(
        self,
        dataloader: Iterable,
        mode: Union[str, float] = "Max",
        momentum: float = 0.1,
        fuse_flag: bool = True,
        rules: Optional[List[ActivationRule]] = None,
        neuron_factory: Optional[NeuronFactory] = None,
        threshold_optimizer: Optional[ThresholdOptimizer] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <RateCodingRecipe.__init__-cn>` | :ref:`English <RateCodingRecipe.__init__-en>`

        ----

        .. _RateCodingRecipe.__init__-cn:

        * **中文**

        构造传统 rate-coding ReLU→IFNode 转换 recipe。该 recipe 拥有
        rate-coding 算法参数，并执行 Conv-BN 融合、VoltageHook 校准和
        neuron replacement。

        :param dataloader: 校准数据加载器。每个 batch 可为输入 tensor、
            tuple/list 或 dict。
        :type dataloader: Iterable
        :param mode: VoltageHook 统计模式，支持 ``"Max"``、百分位字符串和
            ``0 < mode <= 1`` 的浮点缩放。
        :type mode: str or float
        :param momentum: VoltageHook 动量。
        :type momentum: float
        :param fuse_flag: 是否执行 Conv-BN 融合。
        :type fuse_flag: bool
        :param rules: 激活转换规则。默认 ``[ReLURule()]``。
        :type rules: Optional[List[ActivationRule]]
        :param neuron_factory: 脉冲神经元工厂。
        :type neuron_factory: Optional[NeuronFactory]
        :param threshold_optimizer: 阈值优化器。
        :type threshold_optimizer: Optional[ThresholdOptimizer]

        ----

        .. _RateCodingRecipe.__init__-en:

        * **English**

        Construct a traditional rate-coding ReLU-to-IFNode conversion recipe.
        This recipe owns rate-coding algorithm parameters and performs Conv-BN
        fusion, VoltageHook calibration and neuron replacement.

        :param dataloader: Calibration dataloader. Each batch can be an input
            tensor, tuple/list, or dict.
        :type dataloader: Iterable
        :param mode: VoltageHook statistics mode. Supports ``"Max"``,
            percentile strings, and float scaling with ``0 < mode <= 1``.
        :type mode: str or float
        :param momentum: VoltageHook momentum.
        :type momentum: float
        :param fuse_flag: Whether to fuse Conv-BN modules.
        :type fuse_flag: bool
        :param rules: Activation conversion rules. Defaults to ``[ReLURule()]``.
        :type rules: Optional[List[ActivationRule]]
        :param neuron_factory: Spiking-neuron factory.
        :type neuron_factory: Optional[NeuronFactory]
        :param threshold_optimizer: Threshold optimizer.
        :type threshold_optimizer: Optional[ThresholdOptimizer]
        """
        self.dataloader = dataloader
        self.mode = mode
        self.momentum = momentum
        self.fuse_flag = fuse_flag
        self.rules = rules if rules is not None else [ReLURule()]
        self.neuron_factory = (
            neuron_factory if neuron_factory is not None else NeuronFactory()
        )
        self.threshold_optimizer = (
            threshold_optimizer
            if threshold_optimizer is not None
            else ThresholdOptimizer()
        )

    def validate(self, converter: "Converter") -> None:
        if self.dataloader is None:
            raise ValueError(
                "RateCodingRecipe requires a dataloader. "
                "Pass dataloader to RateCodingRecipe."
            )
        self._check_mode()

    def before_trace(self, converter: "Converter", ann: nn.Module) -> nn.Module:
        ann.eval()
        return ann

    def after_trace(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        return self._fuse(fx_model, fuse_flag=self.fuse_flag).to(converter.device)

    def insert_observers(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        return self._set_voltagehook(fx_model).to(converter.device)

    def calibrate(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        with torch.no_grad():
            for _, data in enumerate(tqdm(self.dataloader)):
                imgs = self._extract_batch_input(data)
                if isinstance(imgs, torch.Tensor):
                    imgs = imgs.to(device=converter.device)
                else:
                    imgs = torch.as_tensor(imgs, device=converter.device)
                fx_model(imgs)
        return fx_model

    def replace(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <RateCodingRecipe.replace-cn>` | :ref:`English <RateCodingRecipe.replace-en>`

        ----

        .. _RateCodingRecipe.replace-cn:

        * **中文**

        将已校准的 activation-hook 节点对替换为 rate-coding SNN 子图，并将
        结果移动到当前转换 device。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: Converter
        :param fx_model: 已插入并校准 ``VoltageHook`` 的 ``GraphModule``。
        :type fx_model: torch.fx.GraphModule
        :return: 替换后的 ``GraphModule``。
        :rtype: torch.fx.GraphModule

        ----

        .. _RateCodingRecipe.replace-en:

        * **English**

        Replace calibrated activation-hook node pairs with rate-coding SNN
        subgraphs, and move the result to the current conversion device.

        :param converter: Converter that executes this recipe.
        :type converter: Converter
        :param fx_model: ``GraphModule`` with inserted and calibrated
            ``VoltageHook`` modules.
        :type fx_model: torch.fx.GraphModule
        :return: Replaced ``GraphModule``.
        :rtype: torch.fx.GraphModule
        """
        return self._replace_by_neurons(fx_model).to(converter.device)

    @staticmethod
    def _extract_batch_input(data):
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, (list, tuple)):
            if not data:
                raise ValueError("Batch data is an empty list or tuple.")
            return RateCodingRecipe._extract_batch_input(data[0])
        if isinstance(data, dict):
            if not data:
                raise ValueError("Batch data is an empty dictionary.")
            for key in (
                "input",
                "image",
                "images",
                "img",
                "x",
                "data",
                "pixel_values",
            ):
                if key in data:
                    return RateCodingRecipe._extract_batch_input(data[key])
            return RateCodingRecipe._extract_batch_input(next(iter(data.values())))
        return data

    def _check_mode(self):
        err_msg = "You have used a non-defined VoltageScale Method."
        if isinstance(self.mode, str):
            if not self.mode:
                raise NotImplementedError(err_msg)
            if self.mode[-1] == "%":
                try:
                    percentile = float(self.mode[:-1])
                    if not (0.0 <= percentile <= 100.0):
                        raise NotImplementedError(err_msg)
                except ValueError as exc:
                    raise NotImplementedError(err_msg) from exc
            elif self.mode.lower() in ["max"]:
                pass
            else:
                raise NotImplementedError(err_msg)
        elif isinstance(self.mode, (int, float)) and not isinstance(self.mode, bool):
            if not (0 < self.mode <= 1):
                raise NotImplementedError(err_msg)
        else:
            raise NotImplementedError(err_msg)

    @staticmethod
    def _fuse(
        fx_model: torch.fx.GraphModule, fuse_flag: bool = True
    ) -> torch.fx.GraphModule:
        if not fuse_flag:
            return fx_model

        def matches_module_pattern(
            pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]
        ) -> bool:
            if len(node.args) == 0:
                return False
            nodes: Tuple[Any, fx.Node] = (node.args[0], node)
            for expected_type, current_node in zip(pattern, nodes):
                if not isinstance(current_node, fx.Node):
                    return False
                if current_node.op != "call_module":
                    return False
                if not isinstance(current_node.target, str):
                    return False
                if current_node.target not in modules:
                    return False
                if not isinstance(modules[current_node.target], expected_type):
                    return False
            return True

        def replace_node_module(
            node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module
        ):
            assert isinstance(node.target, str)
            parent_path, _, child_name = node.target.rpartition(".")
            modules[node.target] = new_module
            setattr(modules[parent_path], child_name, new_module)

        patterns = [
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d),
        ]

        modules = dict(fx_model.named_modules())

        for pattern in patterns:
            for node in list(fx_model.graph.nodes):
                if matches_module_pattern(pattern, node, modules):
                    if len(node.args[0].users) > 1:
                        continue
                    conv = modules[node.args[0].target]
                    bn = modules[node.target]
                    fused_conv = fuse_conv_bn_eval(conv, bn)
                    replace_node_module(node.args[0], modules, fused_conv)
                    node.replace_all_uses_with(node.args[0])
                    fx_model.graph.erase_node(node)
        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()
        fx_model.recompile()
        return fx_model

    def _set_voltagehook(self, fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        hook_factory = HookFactory(mode=self.mode, momentum=self.momentum)
        hook_counts_per_prefix: Dict[str, int] = {}
        modules = dict(fx_model.named_modules())

        for node in list(fx_model.graph.nodes):
            if node.op != "call_module":
                continue
            if node.target not in modules:
                continue
            for rule in self.rules:
                if rule.match(node, modules):
                    rule.insert_hooks(
                        fx_model, node, hook_factory, hook_counts_per_prefix
                    )
                    modules = dict(fx_model.named_modules())
                    break

        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model

    def _replace_by_neurons(
        self, fx_model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        replaced_hooks = set()
        replaced_activations = set()
        for rule in self.rules:
            modules = dict(fx_model.named_modules())
            replacements = list(rule.find_replacements(fx_model, modules))
            for activation_node, hook_node in replacements:
                if (
                    hook_node in replaced_hooks
                    or activation_node in replaced_activations
                ):
                    continue
                replaced_hooks.add(hook_node)
                replaced_activations.add(activation_node)
                rule.replace_with_neurons(
                    fx_model,
                    activation_node,
                    hook_node,
                    self.neuron_factory,
                    self.threshold_optimizer,
                )

        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()
        fx_model.recompile()
        return fx_model


class TransformerSpikeEquivalentRecipe(ConversionRecipe):
    r"""
    **API Language** - :ref:`中文 <TransformerSpikeEquivalentRecipe-cn>` | :ref:`English <TransformerSpikeEquivalentRecipe-en>`

    ----

    .. _TransformerSpikeEquivalentRecipe-cn:

    * **中文**

    Transformer TD / spike-equivalent operator 替换 recipe。该 recipe 不插入
    observer，不运行 dataloader 校准，也不强制切换模型 train/eval 状态；它仅
    将当前支持的 ANN core modules 和窄 attention 子集替换为 TD 等价算子。

    ----

    .. _TransformerSpikeEquivalentRecipe-en:

    * **English**

    Transformer TD / spike-equivalent operator replacement recipe. This recipe
    does not insert observers, does not run dataloader calibration, and does not
    force train/eval mode changes. It only replaces the currently supported ANN
    core modules and narrow attention subset with TD-equivalent operators.
    """

    def replace(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <TransformerSpikeEquivalentRecipe.replace-cn>` | :ref:`English <TransformerSpikeEquivalentRecipe.replace-en>`

        ----

        .. _TransformerSpikeEquivalentRecipe.replace-cn:

        * **中文**

        将当前支持的 Transformer core modules、SDPA 调用和窄
        ``MultiheadAttention`` 调用替换为 TD / spike-equivalent 算子。
        该步骤不插入 observer，也不运行 rate-coding 校准。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: Converter
        :param fx_model: 已 trace 的 ``GraphModule``。
        :type fx_model: torch.fx.GraphModule
        :return: 替换后的 ``GraphModule``。
        :rtype: torch.fx.GraphModule
        :raises ValueError: 当 attention 调用或配置不在当前支持范围内时抛出。

        ----

        .. _TransformerSpikeEquivalentRecipe.replace-en:

        * **English**

        Replace currently supported Transformer core modules, SDPA calls and
        narrow ``MultiheadAttention`` calls with TD / spike-equivalent
        operators. This step does not insert observers or run rate-coding
        calibration.

        :param converter: Converter that executes this recipe.
        :type converter: Converter
        :param fx_model: Traced ``GraphModule``.
        :type fx_model: torch.fx.GraphModule
        :return: Replaced ``GraphModule``.
        :rtype: torch.fx.GraphModule
        :raises ValueError: If an attention call or configuration is outside
            the currently supported subset.
        """
        modules = dict(fx_model.named_modules())
        for node in list(fx_model.graph.nodes):
            if node.op != "call_module":
                continue
            if not isinstance(node.target, str) or node.target not in modules:
                continue
            module = modules[node.target]
            replacement = self._make_td_operator(module, node)
            if replacement is None:
                continue
            self._replace_submodule(fx_model, node.target, replacement)
            modules[node.target] = replacement

        sdpa_index = 0
        for node in list(fx_model.graph.nodes):
            if (
                node.op != "call_function"
                or node.target is not F.scaled_dot_product_attention
            ):
                continue
            sdpa_kwargs = self._parse_sdpa_node(node)
            target = f"td_scaled_dot_product_attention_{sdpa_index}"
            existing_modules = dict(fx_model.named_modules())
            while target in existing_modules:
                sdpa_index += 1
                target = f"td_scaled_dot_product_attention_{sdpa_index}"
            sdpa_index += 1
            fx_model.add_submodule(
                target,
                TDScaledDotProductAttention(
                    is_causal=sdpa_kwargs["is_causal"],
                    scale=sdpa_kwargs["scale"],
                ),
            )
            with fx_model.graph.inserting_after(node):
                new_node = fx_model.graph.call_module(
                    target,
                    args=(
                        sdpa_kwargs["query"],
                        sdpa_kwargs["key"],
                        sdpa_kwargs["value"],
                        sdpa_kwargs["attn_mask"],
                    ),
                )
            node.replace_all_uses_with(new_node)
            fx_model.graph.erase_node(node)

        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()
        fx_model.recompile()
        return fx_model

    @staticmethod
    def _replace_submodule(
        fx_model: torch.fx.GraphModule, target: str, module: nn.Module
    ) -> None:
        parent_name, _, child_name = target.rpartition(".")
        parent = fx_model.get_submodule(parent_name) if parent_name else fx_model
        setattr(parent, child_name, module)

    @staticmethod
    def _get_literal_argument(
        node: fx.Node,
        name: str,
        position: int,
        default: Any,
    ) -> Any:
        if name in node.kwargs:
            return node.kwargs[name]
        if len(node.args) > position:
            return node.args[position]
        return default

    @staticmethod
    def _parse_sdpa_node(node: fx.Node) -> Dict[str, Any]:
        if len(node.args) < 3:
            raise ValueError("SDPA node must have query, key, and value arguments.")
        dropout_p = TransformerSpikeEquivalentRecipe._get_literal_argument(
            node, "dropout_p", 4, 0.0
        )
        if not isinstance(dropout_p, (int, float)) or float(dropout_p) != 0.0:
            raise ValueError(
                "TD SDPA conversion only supports literal dropout_p=0.0, "
                f"but got {dropout_p!r}."
            )
        enable_gqa = TransformerSpikeEquivalentRecipe._get_literal_argument(
            node, "enable_gqa", 7, False
        )
        if enable_gqa is not False:
            raise ValueError("TD SDPA conversion does not support enable_gqa=True.")

        is_causal = TransformerSpikeEquivalentRecipe._get_literal_argument(
            node, "is_causal", 5, False
        )
        if not isinstance(is_causal, bool):
            raise ValueError(
                "TD SDPA conversion only supports literal bool is_causal, "
                f"but got {is_causal!r}."
            )
        scale = TransformerSpikeEquivalentRecipe._get_literal_argument(
            node, "scale", 6, None
        )
        if scale is not None and not isinstance(scale, (int, float)):
            raise ValueError(
                "TD SDPA conversion only supports literal numeric scale or None, "
                f"but got {scale!r}."
            )

        return {
            "query": node.args[0],
            "key": node.args[1],
            "value": node.args[2],
            "attn_mask": TransformerSpikeEquivalentRecipe._get_literal_argument(
                node, "attn_mask", 3, None
            ),
            "is_causal": is_causal,
            "scale": None if scale is None else float(scale),
        }

    @staticmethod
    def _check_mha_node(module: nn.MultiheadAttention, node: fx.Node) -> None:
        if module.dropout != 0.0:
            raise ValueError("TD MHA conversion only supports dropout=0.0.")
        if not module.batch_first:
            raise ValueError("TD MHA conversion only supports batch_first=True.")
        if module.kdim != module.embed_dim or module.vdim != module.embed_dim:
            raise ValueError(
                "TD MHA conversion only supports kdim == vdim == embed_dim."
            )
        if module.in_proj_weight is None:
            raise ValueError("TD MHA conversion requires packed in_proj_weight.")
        if module.bias_k is not None or module.bias_v is not None:
            raise ValueError("TD MHA conversion does not support add_bias_kv.")
        if module.add_zero_attn:
            raise ValueError("TD MHA conversion does not support add_zero_attn.")

        need_weights = TransformerSpikeEquivalentRecipe._get_literal_argument(
            node, "need_weights", 4, True
        )
        if need_weights is not False:
            raise ValueError("TD MHA conversion requires need_weights=False.")
        key_padding_mask = TransformerSpikeEquivalentRecipe._get_literal_argument(
            node, "key_padding_mask", 3, None
        )
        if key_padding_mask is not None:
            raise ValueError("TD MHA conversion does not support key_padding_mask.")
        average_attn_weights = TransformerSpikeEquivalentRecipe._get_literal_argument(
            node, "average_attn_weights", 6, True
        )
        if average_attn_weights is not True:
            raise ValueError(
                "TD MHA conversion does not support average_attn_weights=False."
            )

    @staticmethod
    def _copy_mha_parameters(
        source: nn.MultiheadAttention,
        target: TDMultiheadAttention,
    ) -> None:
        if source.in_proj_weight is None:
            raise ValueError("TD MHA conversion requires packed in_proj_weight.")
        with torch.no_grad():
            q_weight, k_weight, v_weight = source.in_proj_weight.chunk(3, dim=0)
            target.q_proj.weight.copy_(q_weight)
            target.k_proj.weight.copy_(k_weight)
            target.v_proj.weight.copy_(v_weight)
            if source.in_proj_bias is not None:
                q_bias, k_bias, v_bias = source.in_proj_bias.chunk(3, dim=0)
                target.q_proj.bias.copy_(q_bias)
                target.k_proj.bias.copy_(k_bias)
                target.v_proj.bias.copy_(v_bias)
            target.out_proj.weight.copy_(source.out_proj.weight)
            if source.out_proj.bias is not None:
                target.out_proj.bias.copy_(source.out_proj.bias)

    def _make_td_operator(
        self,
        module: nn.Module,
        node: Optional[fx.Node] = None,
    ) -> Optional[nn.Module]:
        if isinstance(module, TDModule):
            return None

        if isinstance(module, nn.Linear):
            td_module = TDLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            with torch.no_grad():
                td_module.weight.copy_(module.weight)
                if module.bias is not None:
                    td_module.bias.copy_(module.bias)
            td_module.train(module.training)
            return td_module

        if isinstance(module, nn.LayerNorm):
            td_module = TDLayerNorm(
                module.normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
                bias=module.bias is not None,
                device=(module.weight.device if module.weight is not None else None),
                dtype=(module.weight.dtype if module.weight is not None else None),
            )
            with torch.no_grad():
                if module.weight is not None:
                    td_module.weight.copy_(module.weight)
                if module.bias is not None:
                    td_module.bias.copy_(module.bias)
            td_module.train(module.training)
            return td_module

        if isinstance(module, nn.GELU):
            td_module = TDGELU(approximate=module.approximate)
            td_module.train(module.training)
            return td_module

        if isinstance(module, nn.MultiheadAttention):
            if node is None:
                raise ValueError("TD MHA conversion requires an FX node.")
            self._check_mha_node(module, node)
            td_module = TDMultiheadAttention(
                module.embed_dim,
                module.num_heads,
                dropout=module.dropout,
                bias=module.in_proj_bias is not None,
                batch_first=module.batch_first,
                device=module.in_proj_weight.device,
                dtype=module.in_proj_weight.dtype,
            )
            self._copy_mha_parameters(module, td_module)
            td_module.train(module.training)
            return td_module

        return None
