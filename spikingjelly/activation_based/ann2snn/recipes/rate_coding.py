from __future__ import annotations

import math
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    TYPE_CHECKING,
    Type,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fx
from torch.nn.utils.fusion import fuse_conv_bn_eval
from tqdm import tqdm

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.ann2snn.modules import (
    ChannelVoltageScaler,
    VoltageHook,
    VoltageScaler,
)
from spikingjelly.activation_based.ann2snn.recipes.base import ConversionRecipe
from spikingjelly.activation_based.ann2snn.recipes.step_mode_adapters import (
    _RATE_CODING_SAFE_MODULE_TYPES,
    _RATE_CODING_STATELESS_MODULE_TYPES,
    adapt_step_mode_graph,
)

if TYPE_CHECKING:
    from spikingjelly.activation_based.ann2snn.converter import Converter


__all__ = ["RateCodingRecipe"]


def _matches_relu(node: fx.Node, modules: Dict[str, nn.Module]) -> bool:
    return (
        node.op == "call_module"
        and isinstance(node.target, str)
        and type(modules.get(node.target)) is nn.ReLU
    )


def _add_module_and_node(
    fx_model: fx.GraphModule,
    target: str,
    after: fx.Node,
    module: nn.Module,
    args: Tuple,
) -> fx.Node:
    fx_model.add_submodule(target=target, m=module)
    with fx_model.graph.inserting_after(after):
        return fx_model.graph.call_module(target, args=args)


def validate_rate_coding_mode(mode: Union[str, float]) -> None:
    err_msg = "You have used a non-defined VoltageScale Method."
    if isinstance(mode, str):
        if not mode:
            raise NotImplementedError(err_msg)
        if mode[-1] == "%":
            try:
                percentile = float(mode[:-1])
                if not (0.0 <= percentile <= 100.0):
                    raise NotImplementedError(err_msg)
            except ValueError as exc:
                raise NotImplementedError(err_msg) from exc
        elif mode.lower() != "max":
            raise NotImplementedError(err_msg)
    elif isinstance(mode, (int, float)) and not isinstance(mode, bool):
        if not (0 < mode <= 1):
            raise NotImplementedError(err_msg)
    else:
        raise NotImplementedError(err_msg)


def _extract_batch_input(data, _inside_input=False):
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, (list, tuple)):
        if not data:
            raise ValueError("Batch data is an empty list or tuple.")
        if len(data) == 1:
            return _extract_batch_input(data[0], _inside_input=_inside_input)
        if not _inside_input:
            return _extract_batch_input(
                data[0], _inside_input=not isinstance(data[0], dict)
            )
        return data
    if isinstance(data, dict):
        if not data:
            raise ValueError("Batch data is an empty dictionary.")
        if _inside_input:
            return data
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
                return _extract_batch_input(data[key], _inside_input=True)
        if len(data) != 1:
            raise ValueError(
                "Batch dictionaries with multiple fields must contain one "
                "of 'input', 'image', 'images', 'img', 'x', 'data', or "
                "'pixel_values'."
            )
        return _extract_batch_input(next(iter(data.values())), _inside_input=True)
    return data


def _fuse_conv_bn(
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
        if not isinstance(node.target, str):
            raise ValueError("FX module replacement requires a string target.")
        parent_path, _, child_name = node.target.rpartition(".")
        modules[node.target] = new_module
        parent = fx_model.get_submodule(parent_path) if parent_path else fx_model
        setattr(parent, child_name, new_module)

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


class ChannelVoltageHook(nn.Module):
    def __init__(
        self,
        mode: Union[str, float] = "Max",
        momentum: float = 0.1,
        channel_dim: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.momentum = momentum
        self.channel_dim = channel_dim
        self.eps = eps
        self.register_buffer("scale", torch.empty(0))
        self.num_batches_tracked = 0

    @staticmethod
    def _normalize_channel_dim(x: torch.Tensor, channel_dim: int) -> int:
        if x.dim() < 2:
            raise ValueError(
                "Channel-wise rate calibration requires activation tensors with "
                "at least 2 dimensions."
            )
        if channel_dim < 0:
            channel_dim += x.dim()
        if channel_dim < 0 or channel_dim >= x.dim():
            raise ValueError("channel_dim is out of range.")
        return channel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        err_msg = "You have used a non-defined VoltageScale Method."
        channel_dim = self._normalize_channel_dim(x, self.channel_dim)
        x_stat = torch.clamp(x.detach(), min=0)
        if x_stat.dtype in [torch.float16, torch.bfloat16]:
            x_stat = x_stat.to(torch.float32)
        channel_values = x_stat.movedim(channel_dim, 0).reshape(
            x_stat.shape[channel_dim], -1
        )

        if isinstance(self.mode, str):
            if not self.mode:
                raise NotImplementedError(err_msg)
            if self.mode[-1] == "%":
                try:
                    quantile = float(self.mode[:-1]) / 100.0
                except ValueError as exc:
                    raise NotImplementedError(err_msg) from exc
                if not (0.0 <= quantile <= 1.0):
                    raise NotImplementedError(err_msg)
                s_t = torch.quantile(channel_values, quantile, dim=1)
            elif self.mode.lower() in ["max"]:
                s_t = channel_values.max(dim=1).values
            else:
                raise NotImplementedError(err_msg)
        elif (
            isinstance(self.mode, (int, float))
            and not isinstance(self.mode, bool)
            and 0 < self.mode <= 1
        ):
            s_t = channel_values.max(dim=1).values * self.mode
        else:
            raise NotImplementedError(err_msg)

        s_t = torch.clamp(s_t.to(device=x.device, dtype=x.dtype), min=self.eps)
        if self.num_batches_tracked == 0:
            self.scale = s_t.detach()
        else:
            self.scale = (
                (1 - self.momentum) * self.scale.to(device=x.device, dtype=x.dtype)
                + self.momentum * s_t
            ).detach()
        self.num_batches_tracked += x.shape[0]
        return x

    def compute_threshold(self) -> torch.Tensor:
        if self.scale.numel() == 0:
            raise ValueError("No calibration activations have been recorded.")
        if not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Channel-wise thresholds must be finite positive values.")
        return self.scale.detach()


class RateCodingRecipe(ConversionRecipe):
    def __init__(
        self,
        dataloader: Iterable,
        mode: Union[str, float] = "Max",
        momentum: float = 0.1,
        fuse_flag: bool = True,
        channel_wise: bool = False,
        channel_dim: int = 1,
        pre_spike_maxpool: bool = False,
        half_threshold: bool = False,
        eps: float = 1e-6,
        neuron_factory: Optional[Callable[[float], nn.Module]] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <RateCodingRecipe.__init__-cn>` | :ref:`English <RateCodingRecipe.__init__-en>`

        ----

        .. _RateCodingRecipe.__init__-cn:

        * **中文**

        构造传统 rate-coding ReLU→IFNode 转换 recipe。该 recipe 拥有
        rate-coding 算法参数，并执行 Conv-BN 融合、VoltageHook 校准和
        neuron replacement。

        :param dataloader: 校准数据加载器。每个 batch 可为单输入 tensor、
            ``(input, target)`` 风格的 tuple/list，或包含 ``"input"`` /
            ``"image"`` / ``"images"`` 等输入键的 dict。默认 recipe 只支持
            单输入校准；多输入模型应通过自定义 recipe 扩展。
        :type dataloader: Iterable
        :param mode: VoltageHook 统计模式，支持 ``"Max"``、百分位字符串和
            ``0 < mode <= 1`` 的浮点缩放。
        :type mode: str or float
        :param momentum: VoltageHook 动量。
        :type momentum: float
        :param fuse_flag: 是否执行 Conv-BN 融合。
        :type fuse_flag: bool
        :param channel_wise: 是否按通道统计 robust 激活尺度并使用 channel-wise
            threshold。默认 ``False`` 以保持原有 layer-wise 行为。
        :type channel_wise: bool
        :param channel_dim: ``channel_wise=True`` 时的通道维。
        :type channel_dim: int
        :param pre_spike_maxpool: ``channel_wise=True`` 时，若匹配到
            ``ReLU -> MaxPool2d``，是否把 MaxPool2d 放到脉冲神经元之前。
        :type pre_spike_maxpool: bool
        :param half_threshold: ``channel_wise=True`` 时，是否使用半阈值膜电位初始化。
        :type half_threshold: bool
        :param eps: ``channel_wise=True`` 时的阈值数值下界。
        :type eps: float
        :param neuron_factory: 接收当前层校准尺度并返回脉冲神经元模块的可调用对象。
            ``None`` 时创建阈值为 ``1.0``、软复位的
            :class:`~spikingjelly.activation_based.neuron.IFNode`。
        :type neuron_factory: Optional[Callable[[float], torch.nn.Module]]

        ----

        .. _RateCodingRecipe.__init__-en:

        * **English**

        Construct a traditional rate-coding ReLU-to-IFNode conversion recipe.
        This recipe owns rate-coding algorithm parameters and performs Conv-BN
        fusion, VoltageHook calibration and neuron replacement.

        :param dataloader: Calibration dataloader. Each batch can be a
            single-input tensor, a ``(input, target)``-style tuple/list, or a
            dict with input-like keys such as ``"input"``, ``"image"``, or
            ``"images"``. The default recipe only supports single-input
            calibration; multi-input models should extend a custom recipe.
        :type dataloader: Iterable
        :param mode: VoltageHook statistics mode. Supports ``"Max"``,
            percentile strings, and float scaling with ``0 < mode <= 1``.
        :type mode: str or float
        :param momentum: VoltageHook momentum.
        :type momentum: float
        :param fuse_flag: Whether to fuse Conv-BN modules.
        :type fuse_flag: bool
        :param channel_wise: If ``True``, collect robust activation scales per
            channel and use channel-wise thresholds. Defaults to ``False`` to
            preserve the original layer-wise behaviour.
        :type channel_wise: bool
        :param channel_dim: Channel dimension used when ``channel_wise=True``.
        :type channel_dim: int
        :param pre_spike_maxpool: When ``channel_wise=True`` and a
            ``ReLU -> MaxPool2d`` pattern is matched, place MaxPool2d before the
            spiking neuron.
        :type pre_spike_maxpool: bool
        :param half_threshold: When ``channel_wise=True``, initialize membrane
            potential at half threshold.
        :type half_threshold: bool
        :param eps: Numeric lower bound for thresholds when ``channel_wise=True``.
        :type eps: float
        :param neuron_factory: Callable that receives the calibrated layer scale
            and returns a spiking-neuron module. ``None`` creates an
            :class:`~spikingjelly.activation_based.neuron.IFNode` with threshold
            ``1.0`` and soft reset.
        :type neuron_factory: Optional[Callable[[float], torch.nn.Module]]
        """
        self.dataloader = dataloader
        self.mode = mode
        self.momentum = momentum
        self.fuse_flag = fuse_flag
        self.channel_wise = channel_wise
        self.channel_dim = channel_dim
        self.pre_spike_maxpool = pre_spike_maxpool
        self.half_threshold = half_threshold
        self.eps = eps
        if channel_wise and half_threshold and neuron_factory is not None:
            raise ValueError(
                "RateCodingRecipe(channel_wise=True, half_threshold=True) does not "
                "support custom neuron_factory."
            )
        self.neuron_factory = neuron_factory

    def validate(self, converter: "Converter") -> None:
        if self.dataloader is None:
            raise ValueError(
                "RateCodingRecipe requires a dataloader. "
                "Pass dataloader to RateCodingRecipe."
            )
        validate_rate_coding_mode(self.mode)
        if (
            not isinstance(self.momentum, (int, float))
            or isinstance(self.momentum, bool)
            or not math.isfinite(float(self.momentum))
            or not 0.0 <= self.momentum <= 1.0
        ):
            raise ValueError("momentum must be a finite number in [0, 1].")
        if not isinstance(self.channel_wise, bool):
            raise ValueError("channel_wise must be bool.")
        if not isinstance(self.channel_dim, int):
            raise ValueError("channel_dim must be int.")
        if not isinstance(self.pre_spike_maxpool, bool):
            raise ValueError("pre_spike_maxpool must be bool.")
        if not isinstance(self.half_threshold, bool):
            raise ValueError("half_threshold must be bool.")
        if not (isinstance(self.eps, (int, float)) and self.eps > 0):
            raise ValueError("eps must be a positive number.")

    def before_trace(self, converter: "Converter", ann: nn.Module) -> nn.Module:
        ann.eval()
        return ann

    def after_trace(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        return _fuse_conv_bn(fx_model, fuse_flag=self.fuse_flag).to(converter.device)

    def insert_observers(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        return self._set_voltagehook(fx_model).to(converter.device)

    def calibrate(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                imgs = _extract_batch_input(data)
                if isinstance(imgs, torch.Tensor):
                    imgs = imgs.to(device=converter.device)
                else:
                    if isinstance(imgs, (list, tuple, dict)):
                        raise ValueError(
                            "RateCodingRecipe supports single-input calibration "
                            "batches only. For multi-input models, subclass "
                            "ConversionRecipe or RateCodingRecipe and override "
                            "calibrate()."
                        )
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
        fx_model = self._replace_by_neurons(fx_model)
        fx_model = adapt_step_mode_graph(
            fx_model,
            context="RateCodingRecipe step-mode backend",
            wrap_module_types=_RATE_CODING_STATELESS_MODULE_TYPES,
            safe_module_types=_RATE_CODING_SAFE_MODULE_TYPES,
            safe_call_functions=(F.dropout,),
        )
        return fx_model.to(converter.device)

    def _set_voltagehook(self, fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        hook_counts_per_prefix: Dict[str, int] = {}
        modules = dict(fx_model.named_modules())

        for node in list(fx_model.graph.nodes):
            if not _matches_relu(node, modules):
                continue
            parent = node.target.rpartition(".")[0]
            key = parent or "__FIRST_LEVEL_OF_MODULE__"
            counter = hook_counts_per_prefix.get(key, 0)
            hook_counts_per_prefix[key] = counter + 1
            if self.channel_wise:
                leaf = f"channel_voltage_hook_{counter}"
                hook = ChannelVoltageHook(
                    mode=self.mode,
                    momentum=self.momentum,
                    channel_dim=self.channel_dim,
                    eps=self.eps,
                )
            else:
                leaf = f"voltage_hook_{counter}"
                hook = VoltageHook(momentum=self.momentum, mode=self.mode)
            target = f"{parent}.{leaf}" if parent else leaf

            hook_input = node
            users = list(node.users)
            if self.channel_wise and self.pre_spike_maxpool and len(users) == 1:
                user = users[0]
                if (
                    user.op == "call_module"
                    and isinstance(user.target, str)
                    and isinstance(modules.get(user.target), nn.MaxPool2d)
                ):
                    hook_input = user

            hook_node = _add_module_and_node(
                fx_model, target, hook_input, hook, (hook_input,)
            )
            if self.channel_wise:
                for user in list(hook_input.users):
                    if user is not hook_node:
                        user.replace_input_with(hook_input, hook_node)
            modules[target] = hook

        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model

    def _replace_by_neurons(
        self, fx_model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        modules = dict(fx_model.named_modules())
        for hook_node in list(fx_model.graph.nodes):
            if hook_node.op != "call_module" or not hook_node.args:
                continue
            hook = modules.get(hook_node.target)
            hook_input = hook_node.args[0]
            if not isinstance(hook_input, fx.Node):
                continue
            if self.channel_wise:
                if not isinstance(hook, ChannelVoltageHook):
                    continue
                activation_node = hook_input
                if not _matches_relu(activation_node, modules):
                    if (
                        not isinstance(modules.get(hook_input.target), nn.MaxPool2d)
                        or not hook_input.args
                        or not isinstance(hook_input.args[0], fx.Node)
                    ):
                        continue
                    activation_node = hook_input.args[0]
                    if not _matches_relu(activation_node, modules):
                        continue
                self._replace_channelwise(fx_model, activation_node, hook_node, hook)
            else:
                if not isinstance(hook, VoltageHook) or not _matches_relu(
                    hook_input, modules
                ):
                    continue
                self._replace_layerwise(fx_model, hook_input, hook_node, hook)

        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()
        fx_model.recompile()
        return fx_model

    def _replace_layerwise(
        self,
        fx_model: fx.GraphModule,
        activation_node: fx.Node,
        hook_node: fx.Node,
        hook: VoltageHook,
    ) -> None:
        scale = float(hook.scale.item())
        if not scale > 0.0 or not math.isfinite(scale):
            raise ValueError(
                f"Threshold must be a finite positive number, got {scale} "
                f"for hook {hook_node.target!r}."
            )
        hook_parent, _, hook_leaf = hook_node.target.rpartition(".")
        spike_leaf = hook_leaf.replace("voltage_hook_", "spiking_")
        prefix = f"{hook_parent}.{spike_leaf}" if hook_parent else spike_leaf

        spiking_neuron = (
            neuron.IFNode(v_reset=None)
            if self.neuron_factory is None
            else self.neuron_factory(scale)
        )
        neuron_threshold = getattr(spiking_neuron, "v_threshold", 1.0)
        if hasattr(neuron_threshold, "item"):
            if not hasattr(neuron_threshold, "numel") or neuron_threshold.numel() == 1:
                neuron_threshold = neuron_threshold.item()
        node0 = _add_module_and_node(
            fx_model,
            f"{prefix}.scaler0",
            hook_node,
            VoltageScaler(neuron_threshold / scale),
            activation_node.args,
        )
        node1 = _add_module_and_node(
            fx_model,
            f"{prefix}.if_node",
            node0,
            spiking_neuron,
            (node0,),
        )
        node2 = _add_module_and_node(
            fx_model,
            f"{prefix}.scaler1",
            node1,
            VoltageScaler(scale),
            (node1,),
        )

        hook_node.replace_all_uses_with(node2)
        activation_node.replace_all_uses_with(node2)
        fx_model.graph.erase_node(hook_node)
        fx_model.graph.erase_node(activation_node)

    def _replace_channelwise(
        self,
        fx_model: fx.GraphModule,
        activation_node: fx.Node,
        hook_node: fx.Node,
        hook: ChannelVoltageHook,
    ) -> None:
        hook_input = hook_node.args[0]
        pre_spike_maxpool = hook_input is not activation_node
        if pre_spike_maxpool:
            hook_input.replace_input_with(activation_node, activation_node.args[0])

        threshold = hook.compute_threshold()
        hook_parent, _, hook_leaf = hook_node.target.rpartition(".")
        spike_leaf = hook_leaf.replace("channel_voltage_hook_", "channel_spiking_")
        prefix = f"{hook_parent}.{spike_leaf}" if hook_parent else spike_leaf

        if self.half_threshold:
            neuron_threshold = 1.0
            spiking_neuron = neuron.HalfThresholdIFNode()
        else:
            spiking_neuron = (
                neuron.IFNode(v_reset=None)
                if self.neuron_factory is None
                else self.neuron_factory(1.0)
            )
            neuron_threshold = getattr(spiking_neuron, "v_threshold", 1.0)
            if hasattr(neuron_threshold, "item"):
                if (
                    not hasattr(neuron_threshold, "numel")
                    or neuron_threshold.numel() == 1
                ):
                    neuron_threshold = neuron_threshold.item()
            if not isinstance(neuron_threshold, (int, float)):
                raise ValueError(
                    "Channel-wise RateCodingRecipe requires a scalar neuron threshold."
                )
            if not neuron_threshold > 0.0 or not math.isfinite(neuron_threshold):
                raise ValueError(
                    "Channel-wise RateCodingRecipe requires a finite positive "
                    f"neuron threshold, got {neuron_threshold}."
                )

        node0 = _add_module_and_node(
            fx_model,
            f"{prefix}.scaler0",
            hook_node,
            ChannelVoltageScaler(
                neuron_threshold / threshold, channel_dim=self.channel_dim
            ),
            (hook_input,) if pre_spike_maxpool else activation_node.args,
        )
        node1 = _add_module_and_node(
            fx_model,
            f"{prefix}.if_node",
            node0,
            spiking_neuron,
            (node0,),
        )
        node2 = _add_module_and_node(
            fx_model,
            f"{prefix}.scaler1",
            node1,
            ChannelVoltageScaler(threshold, channel_dim=self.channel_dim),
            (node1,),
        )

        hook_node.replace_all_uses_with(node2)
        fx_model.graph.erase_node(hook_node)
        if not pre_spike_maxpool:
            activation_node.replace_all_uses_with(node2)
        fx_model.graph.erase_node(activation_node)
