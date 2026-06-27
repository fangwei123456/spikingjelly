from __future__ import annotations

import math
from typing import Dict, Iterable, Iterator, TYPE_CHECKING, Tuple, Union

import torch
import torch.nn as nn
from torch import fx
from tqdm import tqdm

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.ann2snn.modules import ChannelVoltageScaler
from spikingjelly.activation_based.ann2snn.rules import ReLURule
from spikingjelly.activation_based.ann2snn.recipes.base import ConversionRecipe
from spikingjelly.activation_based.ann2snn.recipes.rate_coding import (
    RateCodingRecipe,
    validate_rate_coding_mode,
)

if TYPE_CHECKING:
    from spikingjelly.activation_based.ann2snn.converter import Converter


__all__ = ["LocalThresholdBalancingRecipe"]


class LocalThresholdBalancingHook(nn.Module):
    def __init__(
        self,
        mode: Union[str, float] = "99.9%",
        channel_dim: int = 1,
        threshold_candidates: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
        time_steps: int = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.channel_dim = channel_dim
        self.threshold_candidates = tuple(float(v) for v in threshold_candidates)
        self.time_steps = int(time_steps)
        self.eps = float(eps)
        self.register_buffer("scale", torch.empty(0))
        self.register_buffer("threshold", torch.empty(0))

    @staticmethod
    def _normalize_channel_dim(x: torch.Tensor, channel_dim: int) -> int:
        if x.dim() < 2:
            raise ValueError(
                "LocalThresholdBalancingRecipe requires activation tensors with "
                "at least 2 dimensions."
            )
        if channel_dim < 0:
            channel_dim += x.dim()
        if channel_dim < 0 or channel_dim >= x.dim():
            raise ValueError("channel_dim is out of range.")
        return channel_dim

    @staticmethod
    def _channel_view(values: torch.Tensor, x: torch.Tensor, channel_dim: int):
        shape = [1] * x.dim()
        shape[channel_dim] = values.numel()
        return values.reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_dim = self._normalize_channel_dim(x, self.channel_dim)
        x_nonnegative = torch.clamp(x, min=0)
        x_detached = x_nonnegative.detach()
        if x_detached.dtype in [torch.float16, torch.bfloat16]:
            x_stat = x_detached.to(torch.float32)
        else:
            x_stat = x_detached

        if self.threshold.numel() == 0:
            self.threshold = torch.zeros(
                x.shape[channel_dim], device=x_stat.device, dtype=x_stat.dtype
            )

        threshold = self.threshold.to(device=x_stat.device, dtype=x_stat.dtype)
        threshold_view = self._channel_view(threshold, x_stat, channel_dim)
        reduce_dims = tuple(dim for dim in range(x_stat.dim()) if dim != channel_dim)
        overflow = torch.clamp(x_stat - threshold_view, min=0)
        threshold = threshold + 2.0 * overflow.mean(dim=reduce_dims)
        threshold = torch.clamp(threshold, min=self.eps)
        self.threshold = threshold.detach()
        self.scale = self.threshold

        clipped_threshold = self._channel_view(
            threshold.to(device=x.device, dtype=x.dtype), x, channel_dim
        )
        return torch.minimum(x_nonnegative, clipped_threshold)

    def compute_threshold(self) -> torch.Tensor:
        if self.threshold.numel() > 0:
            threshold = self.threshold
        elif self.scale.numel() > 0:
            threshold = self.scale
        else:
            raise ValueError("No calibration activations have been recorded.")
        if not torch.isfinite(threshold).all() or (threshold <= 0).any():
            raise ValueError("Balanced thresholds must be finite positive values.")
        return threshold.detach()

class LocalThresholdBalancingReLURule:
    def __init__(self, channel_dim: int = 1) -> None:
        self.channel_dim = channel_dim
        self.relu_rule = ReLURule()

    def match(self, node: fx.Node, modules: Dict[str, nn.Module]) -> bool:
        return self.relu_rule.match(node, modules)

    def insert_hooks(
        self,
        fx_model: fx.GraphModule,
        node: fx.Node,
        hook_factory: "LocalThresholdBalancingHookFactory",
        hook_counts_per_prefix: Dict[str, int],
    ) -> fx.Node:
        if not isinstance(node.target, str):
            raise TypeError("node.target must be a module path string.")
        parent, _, _ = node.target.rpartition(".")
        key = parent or "__FIRST_LEVEL_OF_MODULE__"
        counter = hook_counts_per_prefix.get(key, 0)
        hook_counts_per_prefix[key] = counter + 1
        target = (
            f"{parent}.ltb_hook_{counter}" if parent else f"ltb_hook_{counter}"
        )
        modules = dict(fx_model.named_modules())
        hook_input = node
        users = list(node.users)
        if len(users) == 1:
            user = users[0]
            if (
                user.op == "call_module"
                and isinstance(user.target, str)
                and isinstance(modules.get(user.target), nn.MaxPool2d)
            ):
                hook_input = user
        fx_model.add_submodule(target=target, m=hook_factory.create())
        with fx_model.graph.inserting_after(n=hook_input):
            hook_node = fx_model.graph.call_module(
                module_name=target, args=(hook_input,)
            )
        for user in list(hook_input.users):
            if user is not hook_node:
                user.replace_input_with(hook_input, hook_node)
        return hook_node

    def find_replacements(
        self, fx_model: fx.GraphModule, modules: Dict[str, nn.Module]
    ) -> Iterator[Tuple[fx.Node, fx.Node]]:
        for hook_node in fx_model.graph.nodes:
            if hook_node.op != "call_module":
                continue
            if not isinstance(modules.get(hook_node.target), LocalThresholdBalancingHook):
                continue
            if len(hook_node.args) == 0 or not isinstance(hook_node.args[0], fx.Node):
                continue
            hook_input_node = hook_node.args[0]
            if hook_input_node.op != "call_module":
                continue
            if hook_input_node.target not in modules:
                continue
            if self.match(hook_input_node, modules):
                yield hook_input_node, hook_node
                continue
            if not isinstance(modules.get(hook_input_node.target), nn.MaxPool2d):
                continue
            if (
                len(hook_input_node.args) == 0
                or not isinstance(hook_input_node.args[0], fx.Node)
            ):
                continue
            activation_node = hook_input_node.args[0]
            if (
                activation_node.op == "call_module"
                and activation_node.target in modules
                and self.match(activation_node, modules)
            ):
                yield activation_node, hook_node

    def replace_with_neurons(
        self,
        fx_model: fx.GraphModule,
        activation_node: fx.Node,
        hook_node: fx.Node,
    ) -> None:
        if len(activation_node.args) != 1:
            raise ValueError(
                f"The activation node {activation_node.target!r} must have exactly "
                f"1 argument, but got {len(activation_node.args)}."
            )
        if not isinstance(hook_node.target, str):
            raise TypeError("hook_node.target must be a module path string.")
        hook = fx_model.get_submodule(hook_node.target)
        if not isinstance(hook, LocalThresholdBalancingHook):
            raise TypeError("hook_node must target a LocalThresholdBalancingHook.")
        if len(hook_node.args) != 1 or not isinstance(hook_node.args[0], fx.Node):
            raise ValueError("hook_node must have exactly one FX node input.")
        hook_input_node = hook_node.args[0]
        pre_spike_maxpool = False
        if hook_input_node is not activation_node:
            if not (
                hook_input_node.op == "call_module"
                and isinstance(hook_input_node.target, str)
                and isinstance(fx_model.get_submodule(hook_input_node.target), nn.MaxPool2d)
            ):
                raise TypeError(
                    "LocalThresholdBalancingRecipe only supports hooks after "
                    "ReLU or after ReLU -> MaxPool2d."
                )
            pre_spike_maxpool = True
            hook_input_node.replace_input_with(activation_node, activation_node.args[0])
        threshold = hook.compute_threshold()
        hook_parent, _, hook_leaf = hook_node.target.rpartition(".")
        spike_leaf = hook_leaf.replace("ltb_hook_", "ltb_spiking_")
        prefix = f"{hook_parent}.{spike_leaf}" if hook_parent else spike_leaf

        target0 = f"{prefix}.scaler0"
        target1 = f"{prefix}.if_node"
        target2 = f"{prefix}.scaler1"
        m0 = ChannelVoltageScaler(1.0 / threshold, channel_dim=self.channel_dim)
        m1 = neuron.HalfThresholdIFNode()
        m2 = ChannelVoltageScaler(threshold, channel_dim=self.channel_dim)

        fx_model.add_submodule(target=target0, m=m0)
        with fx_model.graph.inserting_after(n=hook_node):
            spike_input_args = (hook_input_node,) if pre_spike_maxpool else activation_node.args
            node0 = fx_model.graph.call_module(target0, args=spike_input_args)
        fx_model.add_submodule(target=target1, m=m1)
        with fx_model.graph.inserting_after(n=node0):
            node1 = fx_model.graph.call_module(target1, args=(node0,))
        fx_model.add_submodule(target=target2, m=m2)
        with fx_model.graph.inserting_after(n=node1):
            node2 = fx_model.graph.call_module(target2, args=(node1,))

        hook_node.replace_all_uses_with(node2)
        fx_model.graph.erase_node(hook_node)
        if not pre_spike_maxpool:
            activation_node.replace_all_uses_with(node2)
        fx_model.graph.erase_node(activation_node)


class LocalThresholdBalancingHookFactory:
    def __init__(
        self,
        mode: Union[str, float],
        channel_dim: int,
        threshold_candidates: Tuple[float, ...],
        time_steps: int,
        eps: float,
    ) -> None:
        self.mode = mode
        self.channel_dim = channel_dim
        self.threshold_candidates = threshold_candidates
        self.time_steps = time_steps
        self.eps = eps

    def create(self) -> LocalThresholdBalancingHook:
        return LocalThresholdBalancingHook(
            mode=self.mode,
            channel_dim=self.channel_dim,
            threshold_candidates=self.threshold_candidates,
            time_steps=self.time_steps,
            eps=self.eps,
        )


class LocalThresholdBalancingRecipe(ConversionRecipe):
    def __init__(
        self,
        dataloader: Iterable,
        time_steps: int = 64,
        mode: Union[str, float] = "99.9%",
        channel_dim: int = 1,
        threshold_candidates: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
        fuse_flag: bool = True,
        eps: float = 1e-6,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <LocalThresholdBalancingRecipe.__init__-cn>` | :ref:`English <LocalThresholdBalancingRecipe.__init__-en>`

        ----

        .. _LocalThresholdBalancingRecipe.__init__-cn:

        * **中文**

        构造 training-free local-threshold-balancing ANN2SNN 转换 recipe。该
        recipe 只使用校准数据在 SNN 侧为 ReLU 输出选择 channel-wise 阈值，不训练
        或修改输入 ANN 参数。

        参考文献：Bu T, Li M, Yu Z. Inference-Scale Complexity in ANN-SNN
        Conversion for High-Performance and Low-Power Applications.
        arXiv:2409.03368, 2024. Accepted by CVPR 2025.

        :param dataloader: 校准数据加载器。
        :type dataloader: Iterable
        :param time_steps: SNN 仿真步数，保留用于 recipe 配置记录。
        :type time_steps: int
        :param mode: 保留用于兼容 RateCodingRecipe 风格的配置校验。
        :type mode: str or float
        :param channel_dim: ReLU 输出的通道维。
        :type channel_dim: int
        :param threshold_candidates: 保留用于兼容早期实验 API；当前实现遵循
            原始 LTB 的逐 batch 阈值平衡更新。
        :type threshold_candidates: Tuple[float, ...]
        :param fuse_flag: 是否执行 Conv-BN 融合。
        :type fuse_flag: bool
        :param eps: 数值下界。
        :type eps: float

        ----

        .. _LocalThresholdBalancingRecipe.__init__-en:

        * **English**

        Construct a training-free local-threshold-balancing ANN2SNN conversion
        recipe. It uses calibration data only to choose channel-wise thresholds
        on the SNN side for ReLU outputs, without training or mutating the input
        ANN parameters.

        Reference: Bu T, Li M, Yu Z. Inference-Scale Complexity in ANN-SNN
        Conversion for High-Performance and Low-Power Applications.
        arXiv:2409.03368, 2024. Accepted by CVPR 2025.

        :param dataloader: Calibration dataloader.
        :type dataloader: Iterable
        :param time_steps: SNN simulation steps, retained as recipe
            configuration metadata.
        :type time_steps: int
        :param mode: Retained for configuration validation in the style of
            RateCodingRecipe.
        :type mode: str or float
        :param channel_dim: Channel dimension of ReLU outputs.
        :type channel_dim: int
        :param threshold_candidates: Retained for compatibility with earlier
            experimental APIs. The implementation follows the original LTB
            per-batch threshold-balancing update.
        :type threshold_candidates: Tuple[float, ...]
        :param fuse_flag: Whether to fuse Conv-BN modules.
        :type fuse_flag: bool
        :param eps: Numeric lower bound.
        :type eps: float
        """
        self.dataloader = dataloader
        self.time_steps = time_steps
        self.mode = mode
        self.channel_dim = channel_dim
        self.threshold_candidates = tuple(float(v) for v in threshold_candidates)
        self.fuse_flag = fuse_flag
        self.eps = eps
        self.rule = LocalThresholdBalancingReLURule(channel_dim=channel_dim)

    def validate(self, converter: "Converter") -> None:
        if self.dataloader is None:
            raise ValueError(
                "LocalThresholdBalancingRecipe requires a dataloader."
            )
        if not isinstance(self.time_steps, int) or self.time_steps <= 0:
            raise ValueError("time_steps must be a positive int.")
        if not isinstance(self.channel_dim, int):
            raise ValueError("channel_dim must be int.")
        if not self.threshold_candidates:
            raise ValueError("threshold_candidates must not be empty.")
        if any(v <= 0 or not math.isfinite(v) for v in self.threshold_candidates):
            raise ValueError("threshold_candidates must contain finite positive values.")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")
        validate_rate_coding_mode(self.mode)

    def before_trace(self, converter: "Converter", ann: nn.Module) -> nn.Module:
        ann.eval()
        return ann

    def after_trace(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        return RateCodingRecipe._fuse(fx_model, fuse_flag=self.fuse_flag).to(
            converter.device
        )

    def insert_observers(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        hook_factory = LocalThresholdBalancingHookFactory(
            mode=self.mode,
            channel_dim=self.channel_dim,
            threshold_candidates=self.threshold_candidates,
            time_steps=self.time_steps,
            eps=self.eps,
        )
        hook_counts_per_prefix: Dict[str, int] = {}
        modules = dict(fx_model.named_modules())
        for node in list(fx_model.graph.nodes):
            if node.op != "call_module":
                continue
            if node.target not in modules:
                continue
            if self.rule.match(node, modules):
                self.rule.insert_hooks(
                    fx_model, node, hook_factory, hook_counts_per_prefix
                )
                modules = dict(fx_model.named_modules())
        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model.to(converter.device)

    def calibrate(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                fx_model(self._move_batch_to_device(data, converter.device))
        return fx_model

    def replace(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        replaced_hooks = set()
        replaced_activations = set()
        modules = dict(fx_model.named_modules())
        for activation_node, hook_node in self.rule.find_replacements(fx_model, modules):
            if hook_node in replaced_hooks or activation_node in replaced_activations:
                continue
            replaced_hooks.add(hook_node)
            replaced_activations.add(activation_node)
            self.rule.replace_with_neurons(fx_model, activation_node, hook_node)
        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()
        fx_model.recompile()
        return fx_model.to(converter.device)

    @staticmethod
    def _move_batch_to_device(data, device: torch.device):
        imgs = RateCodingRecipe._extract_batch_input(data)
        if isinstance(imgs, torch.Tensor):
            return imgs.to(device=device)
        if isinstance(imgs, (list, tuple, dict)):
            raise ValueError(
                "LocalThresholdBalancingRecipe supports single-input calibration "
                "batches only."
            )
        return torch.as_tensor(imgs, device=device)
