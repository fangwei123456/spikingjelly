from __future__ import annotations

from typing import Dict, Iterable, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fx
from tqdm import tqdm

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.ann2snn.modules import ChannelVoltageScaler
from spikingjelly.activation_based.ann2snn.recipes.base import ConversionRecipe
from spikingjelly.activation_based.ann2snn.recipes.rate_coding import (
    _extract_batch_input,
    _fuse_conv_bn,
)
from spikingjelly.activation_based.ann2snn.recipes.step_mode_adapters import (
    _RATE_CODING_SAFE_MODULE_TYPES,
    _RATE_CODING_STATELESS_MODULE_TYPES,
    adapt_step_mode_graph,
)

if TYPE_CHECKING:
    from spikingjelly.activation_based.ann2snn.converter import Converter


__all__ = ["LocalThresholdBalancingRecipe"]


def _matches_relu(node: fx.Node, modules: Dict[str, nn.Module]) -> bool:
    return (
        node.op == "call_module"
        and isinstance(node.target, str)
        and type(modules.get(node.target)) is nn.ReLU
    )


class LocalThresholdBalancingHook(nn.Module):
    def __init__(
        self,
        channel_dim: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channel_dim = channel_dim
        self.eps = float(eps)
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

        clipped_threshold = self._channel_view(
            threshold.to(device=x.device, dtype=x.dtype), x, channel_dim
        )
        return torch.minimum(x_nonnegative, clipped_threshold)

    def compute_threshold(self) -> torch.Tensor:
        if self.threshold.numel() == 0:
            raise ValueError("No calibration activations have been recorded.")
        if not torch.isfinite(self.threshold).all() or (self.threshold <= 0).any():
            raise ValueError("Balanced thresholds must be finite positive values.")
        return self.threshold.detach()


class LocalThresholdBalancingRecipe(ConversionRecipe):
    def __init__(
        self,
        dataloader: Iterable,
        channel_dim: int = 1,
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
        :param channel_dim: ReLU 输出的通道维。
        :type channel_dim: int
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
        :param channel_dim: Channel dimension of ReLU outputs.
        :type channel_dim: int
        :param fuse_flag: Whether to fuse Conv-BN modules.
        :type fuse_flag: bool
        :param eps: Numeric lower bound.
        :type eps: float
        """
        self.dataloader = dataloader
        self.channel_dim = channel_dim
        self.fuse_flag = fuse_flag
        self.eps = eps

    def validate(self, converter: "Converter") -> None:
        if self.dataloader is None:
            raise ValueError("LocalThresholdBalancingRecipe requires a dataloader.")
        if not isinstance(self.channel_dim, int):
            raise ValueError("channel_dim must be int.")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")

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
        hook_counts_per_prefix: Dict[str, int] = {}
        modules = dict(fx_model.named_modules())
        for node in list(fx_model.graph.nodes):
            if not _matches_relu(node, modules):
                continue
            parent = node.target.rpartition(".")[0]
            key = parent or "__FIRST_LEVEL_OF_MODULE__"
            counter = hook_counts_per_prefix.get(key, 0)
            hook_counts_per_prefix[key] = counter + 1
            leaf = f"ltb_hook_{counter}"
            target = f"{parent}.{leaf}" if parent else leaf

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
            hook = LocalThresholdBalancingHook(
                channel_dim=self.channel_dim,
                eps=self.eps,
            )
            fx_model.add_submodule(target, hook)
            with fx_model.graph.inserting_after(hook_input):
                hook_node = fx_model.graph.call_module(target, args=(hook_input,))
            for user in list(hook_input.users):
                if user is not hook_node:
                    user.replace_input_with(hook_input, hook_node)
            modules[target] = hook
        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model.to(converter.device)

    def calibrate(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                fx_model(
                    torch.as_tensor(
                        _extract_batch_input(data),
                        device=converter.device,
                    )
                )
        return fx_model

    def replace(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        modules = dict(fx_model.named_modules())
        for hook_node in list(fx_model.graph.nodes):
            if (
                hook_node.op != "call_module"
                or not isinstance(
                    modules.get(hook_node.target), LocalThresholdBalancingHook
                )
                or not hook_node.args
                or not isinstance(hook_node.args[0], fx.Node)
            ):
                continue
            hook_input = hook_node.args[0]
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
            self._replace_activation(
                fx_model,
                activation_node,
                hook_node,
                modules[hook_node.target],
            )
        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()
        fx_model.recompile()
        fx_model = adapt_step_mode_graph(
            fx_model,
            context="LocalThresholdBalancingRecipe step-mode backend",
            wrap_module_types=_RATE_CODING_STATELESS_MODULE_TYPES,
            safe_module_types=_RATE_CODING_SAFE_MODULE_TYPES,
            safe_call_functions=(F.dropout,),
        )
        return fx_model.to(converter.device)

    def _replace_activation(
        self,
        fx_model: fx.GraphModule,
        activation_node: fx.Node,
        hook_node: fx.Node,
        hook: LocalThresholdBalancingHook,
    ) -> None:
        hook_input = hook_node.args[0]
        pre_spike_maxpool = hook_input is not activation_node
        if pre_spike_maxpool:
            hook_input.replace_input_with(activation_node, activation_node.args[0])

        threshold = hook.compute_threshold()
        hook_parent, _, hook_leaf = hook_node.target.rpartition(".")
        spike_leaf = hook_leaf.replace("ltb_hook_", "ltb_spiking_")
        prefix = f"{hook_parent}.{spike_leaf}" if hook_parent else spike_leaf
        scaler0 = f"{prefix}.scaler0"
        fx_model.add_submodule(
            scaler0,
            ChannelVoltageScaler(1.0 / threshold, channel_dim=self.channel_dim),
        )
        with fx_model.graph.inserting_after(hook_node):
            node0 = fx_model.graph.call_module(
                scaler0,
                args=(hook_input,) if pre_spike_maxpool else activation_node.args,
            )

        if_node = f"{prefix}.if_node"
        fx_model.add_submodule(if_node, neuron.HalfThresholdIFNode())
        with fx_model.graph.inserting_after(node0):
            node1 = fx_model.graph.call_module(if_node, args=(node0,))

        scaler1 = f"{prefix}.scaler1"
        fx_model.add_submodule(
            scaler1,
            ChannelVoltageScaler(threshold, channel_dim=self.channel_dim),
        )
        with fx_model.graph.inserting_after(node1):
            node2 = fx_model.graph.call_module(scaler1, args=(node1,))

        hook_node.replace_all_uses_with(node2)
        fx_model.graph.erase_node(hook_node)
        if not pre_spike_maxpool:
            activation_node.replace_all_uses_with(node2)
        fx_model.graph.erase_node(activation_node)
