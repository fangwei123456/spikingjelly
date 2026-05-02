from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import Tensor, fx
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .. import base, layer, neuron

__all__ = [
    "fuse_conv_bn_eval_modules",
    "pack_conv_bn_train_modules",
]


_TRAIN_PACK_CONV_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    layer.Conv1d,
    layer.Conv2d,
    layer.Conv3d,
)
_TRAIN_PACK_BN_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    layer.BatchNorm1d,
    layer.BatchNorm2d,
    layer.BatchNorm3d,
)
_CONV_BN_PATTERNS = [
    (layer.Conv1d, layer.BatchNorm1d),
    (layer.Conv2d, layer.BatchNorm2d),
    (layer.Conv3d, layer.BatchNorm3d),
    (nn.Conv1d, nn.BatchNorm1d),
    (nn.Conv2d, nn.BatchNorm2d),
    (nn.Conv3d, nn.BatchNorm3d),
]


def _matches_module_pattern(pattern, node: fx.Node, modules) -> bool:
    if len(node.args) == 0:
        return False
    nodes = (node.args[0], node)
    if len(pattern) != len(nodes):
        return False
    for i in range(len(pattern)):
        expected_type = pattern[i]
        current_node = nodes[i]
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


def _replace_node_module(node: fx.Node, modules, new_module: torch.nn.Module) -> None:
    def parent_name(target: str):
        *parent, name = target.rsplit(".", 1)
        return parent[0] if parent else "", name

    assert isinstance(node.target, str)
    parent, name = parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent], name, new_module)


def _collect_conv_bn_matches(fx_model: fx.GraphModule, modules, patterns):
    pair_to_nodes = {}
    conv_to_bn_targets = {}
    conv_target_call_counts = {}
    matched_bn_nodes = set()
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            conv_target_call_counts[node.target] = (
                conv_target_call_counts.get(node.target, 0) + 1
            )
    for pattern in patterns:
        for node in list(fx_model.graph.nodes):
            if node in matched_bn_nodes:
                continue
            if not _matches_module_pattern(pattern, node, modules):
                continue
            conv_node = node.args[0]
            assert isinstance(conv_node, fx.Node)
            if len(conv_node.users) > 1:
                continue
            pair = (conv_node.target, node.target)
            pair_to_nodes.setdefault(pair, []).append(node)
            conv_to_bn_targets.setdefault(conv_node.target, set()).add(node.target)
            matched_bn_nodes.add(node)

    ambiguous_conv_targets = {
        conv_target
        for conv_target, bn_targets in conv_to_bn_targets.items()
        if len(bn_targets) > 1
    }
    return {
        pair: matched_nodes
        for pair, matched_nodes in pair_to_nodes.items()
        if (
            pair[0] not in ambiguous_conv_targets
            and len(matched_nodes) == conv_target_call_counts[pair[0]]
        )
    }


class _EvalFusionTracer(fx.Tracer):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(
            m,
            (
                _TrainConvBnWrapper,
                layer.Conv1d,
                layer.Conv2d,
                layer.Conv3d,
                layer.BatchNorm1d,
                layer.BatchNorm2d,
                layer.BatchNorm3d,
                base.MemoryModule,
                neuron.BaseNode,
            ),
        ):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class _TrainPackTracer(_EvalFusionTracer):
    pass


class _TrainConvBnWrapper(nn.Module):
    def __init__(self, conv: nn.Module, bn: nn.Module):
        super().__init__()
        self.conv = conv
        self.bn = bn

    @contextmanager
    def _single_step_mode(self, module: nn.Module):
        if isinstance(module, base.StepModule):
            old_step_mode = module.step_mode
            module.step_mode = "s"
            try:
                yield
            finally:
                module.step_mode = old_step_mode
        else:
            yield

    def _packed_forward(self, x: Tensor) -> Tensor:
        t, n = x.shape[:2]
        x = x.flatten(0, 1)
        with self._single_step_mode(self.conv):
            x = self.conv(x)

        with self._single_step_mode(self.bn):
            x = self.bn(x)
        return x.reshape(t, n, *x.shape[1:])

    def _expects_multistep_input(self, x: Tensor) -> bool:
        if isinstance(self.conv, (nn.Conv1d, layer.Conv1d)):
            return x.dim() == 4
        if isinstance(self.conv, (nn.Conv2d, layer.Conv2d)):
            return x.dim() == 5
        if isinstance(self.conv, (nn.Conv3d, layer.Conv3d)):
            return x.dim() == 6
        return False

    def forward(self, x: Tensor) -> Tensor:
        if (
            isinstance(self.conv, _TRAIN_PACK_CONV_TYPES)
            and isinstance(self.bn, _TRAIN_PACK_BN_TYPES)
            and getattr(self.conv, "step_mode", "m") == "m"
            and getattr(self.bn, "step_mode", "m") == "m"
            and self._expects_multistep_input(x)
        ):
            return self._packed_forward(x)
        return self.bn(self.conv(x))


def fuse_conv_bn_eval_modules(net: nn.Module) -> fx.GraphModule:
    """
    **API Language:**
    :ref:`中文 <fuse_conv_bn_eval_modules-cn>` | :ref:`English <fuse_conv_bn_eval_modules-en>`

    ----

    .. _fuse_conv_bn_eval_modules-cn:

    * **中文**

    将评估模式下模型中的相邻 ``Conv*`` 与 ``BatchNorm*`` 模块融合为单个卷积模块.
    该函数同时支持原生 ``torch.nn`` 模块以及 SpikingJelly 的
    :class:`spikingjelly.activation_based.layer.Conv1d`,
    :class:`spikingjelly.activation_based.layer.Conv2d`,
    :class:`spikingjelly.activation_based.layer.Conv3d`,
    :class:`spikingjelly.activation_based.layer.BatchNorm1d`,
    :class:`spikingjelly.activation_based.layer.BatchNorm2d`,
    :class:`spikingjelly.activation_based.layer.BatchNorm3d`.

    输入模型必须处于 ``eval()`` 模式; 返回值是融合后的 ``fx.GraphModule``.

    .. _fuse_conv_bn_eval_modules-en:

    * **English**

    Fuse adjacent ``Conv*`` and ``BatchNorm*`` modules in an evaluation-mode model
    into a single convolution module. Both native ``torch.nn`` layers and
    SpikingJelly activation-based ``layer.Conv*`` / ``layer.BatchNorm*`` wrappers
    are supported.

    The input model must be in ``eval()`` mode. The returned value is a fused
    ``fx.GraphModule``.

    :param net: EN: Evaluation-mode module to transform. Chinese: 待变换的评估模式模型, 必须已经调用 ``eval()``。
    :type net: torch.nn.Module
    :return: EN: Fused FX graph module. Chinese: 融合后的 FX 图模块。
    :rtype: torch.fx.GraphModule
    :raises ValueError: EN: Raised when ``net`` is still in training mode, or when a matched BatchNorm module does not track running statistics or lacks valid running mean/variance. Chinese: 当 ``net`` 仍处于训练模式时, 或者匹配到的 BatchNorm 模块未跟踪运行统计量、缺少有效的运行均值/方差时抛出。
    """

    if net.training:
        raise ValueError("fuse_conv_bn_eval_modules only supports eval() models.")

    tracer = _EvalFusionTracer()
    graph = tracer.trace(net)
    fx_model = fx.GraphModule(tracer.root, graph)
    modules = dict(fx_model.named_modules())

    for (conv_target, bn_target), matched_nodes in _collect_conv_bn_matches(
        fx_model, modules, _CONV_BN_PATTERNS
    ).items():
        conv = modules[conv_target]
        bn = modules[bn_target]
        if (
            getattr(bn, "track_running_stats", True) is False
            or bn.running_mean is None
            or bn.running_var is None
        ):
            raise ValueError(
                f"Cannot fuse {bn_target}: BatchNorm must track running stats."
            )
        fused_conv = fuse_conv_bn_eval(conv, bn)
        conv_node = matched_nodes[0].args[0]
        assert isinstance(conv_node, fx.Node)
        _replace_node_module(conv_node, modules, fused_conv)
        for node in matched_nodes:
            node.replace_all_uses_with(node.args[0])
            fx_model.graph.erase_node(node)

    fx_model.graph.lint()
    fx_model.delete_all_unused_submodules()
    fx_model.recompile()
    return fx_model


def pack_conv_bn_train_modules(net: nn.Module) -> fx.GraphModule:
    """
    **API Language:**
    :ref:`中文 <pack_conv_bn_train_modules-cn>` | :ref:`English <pack_conv_bn_train_modules-en>`

    ----

    .. _pack_conv_bn_train_modules-cn:

    * **中文**

    将训练模式下模型中的相邻 ``Conv*`` 与 ``BatchNorm*`` 模块打包为单个 wrapper,
    以减少多步 ``Conv -> BatchNorm`` 路径中的 ``view/flatten`` 往返。

    该函数不会像 ``fuse_conv_bn_eval_modules`` 那样融合权重; 它只是将相邻层包装成一个
    compile-friendly 的训练模块。当前同时支持原生 ``torch.nn`` 的 ``Conv*`` / ``BatchNorm*``
    模块, 以及 SpikingJelly activation-based ``layer.Conv*`` / ``layer.BatchNorm*`` 模块。

    输入模型必须处于 ``train()`` 模式。返回值是变换后的 ``fx.GraphModule``。

    .. _pack_conv_bn_train_modules-en:

    * **English**

    Pack adjacent ``Conv*`` and ``BatchNorm*`` modules in a training-mode model into
    a single wrapper to reduce redundant ``view/flatten`` hops along multi-step
    ``Conv -> BatchNorm`` paths.

    Unlike ``fuse_conv_bn_eval_modules``, this transform does not fuse weights. It
    only rewrites the module graph into a more compile-friendly training structure.
    Both native ``torch.nn`` layers and SpikingJelly activation-based
    ``layer.Conv*`` / ``layer.BatchNorm*`` wrappers are supported.

    The input model must be in ``train()`` mode. The returned value is the packed
    ``fx.GraphModule``.

    :param net: EN: Training-mode module to transform. Chinese: 待变换的训练模式模型, 必须已经调用 ``train()``。
    :type net: torch.nn.Module
    :return: EN: Packed FX graph module. Chinese: 打包后的 FX 图模块。
    :rtype: torch.fx.GraphModule
    :raises ValueError: EN: Raised when ``net`` is not in training mode. Chinese: 当 ``net`` 不处于训练模式时抛出。
    """
    if not net.training:
        raise ValueError("pack_conv_bn_train_modules only supports train() models.")

    tracer = _TrainPackTracer()
    graph = tracer.trace(net)
    fx_model = fx.GraphModule(tracer.root, graph)
    modules = dict(fx_model.named_modules())

    for (conv_target, bn_target), matched_nodes in _collect_conv_bn_matches(
        fx_model, modules, _CONV_BN_PATTERNS
    ).items():
        conv = modules[conv_target]
        bn = modules[bn_target]
        if (
            isinstance(conv, (layer.Conv1d, layer.Conv2d, layer.Conv3d))
            and isinstance(
                bn, (layer.BatchNorm1d, layer.BatchNorm2d, layer.BatchNorm3d)
            )
            and getattr(conv, "step_mode", None) != getattr(bn, "step_mode", None)
        ):
            continue
        packed = _TrainConvBnWrapper(conv, bn)
        conv_node = matched_nodes[0].args[0]
        assert isinstance(conv_node, fx.Node)
        _replace_node_module(conv_node, modules, packed)
        for node in matched_nodes:
            node.replace_all_uses_with(node.args[0])
            fx_model.graph.erase_node(node)

    fx_model.graph.lint()
    fx_model.delete_all_unused_submodules()
    fx_model.recompile()
    return fx_model
