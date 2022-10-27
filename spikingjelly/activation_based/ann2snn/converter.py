import torch.nn as nn
from spikingjelly.activation_based.ann2snn.modules import *
from tqdm import tqdm
from spikingjelly.activation_based import neuron
from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast
from torch import fx
from torch.nn.utils.fusion import fuse_conv_bn_eval


class Converter(nn.Module):

    def __init__(self, dataloader, mode='Max', fuse_flag=True):
        """
        * :ref:`API in English <Converter.__init__-en>`

        .. _Converter.__init__-cn:

        :param dataloader: 数据加载器
        :type dataloader: Dataloader
        :param mode: 转换模式。目前支持三种模式，最大电流转换模式，99.9%电流转换模式，以及缩放转换模式
        :type mode: str, float
        :param fuse_flag: 标志位，设置为True，则进行conv与bn的融合，反之不进行。
        :type mode: bool

        ``Converter`` 用于将ReLU的ANN转换为SNN。这里实现了常见的三种模式。
        最常见的是最大电流转换模式，它利用前后层的激活上限，使发放率最高的情况能够对应激活取得最大值的情况。
        99.9%电流转换模式利用99.9%的激活分位点限制了激活上限。
        缩放转换模式下，用户需要给定缩放参数到模式中，即可利用缩放后的激活最大值对电流进行限制。

        * :ref:`中文API <Converter.__init__-cn>`

        .. _Converter.__init__-en:

        :param dataloader: Dataloader for converting
        :type dataloader: Dataloader
        :param mode: Conversion mode. Now support three mode, MaxNorm, RobustNorm(99.9%), and scaling mode
        :type mode: str, float
        :param fuse_flag: Bool specifying if fusion of the conv and the bn happens, by default it happens.
        :type mode: bool

        ``Converter`` is used to convert ReLU's ANN to SNN. Three common methods are implemented here.
        The most common is the maximum mode, which utilizes the upper activation limits of
        the front and rear layers so that the case with the highest firing rate corresponds to the case where the
        activation achieves the maximum value.
        The 99.9% mode utilizes the 99.9% activation quantile to limit the upper activation limit.
        In the scaling conversion mode, the user needs to specify the scaling parameters into the mode, and the current
        can be limited by the activated maximum value after scaling.

        """
        super().__init__()
        self.mode = mode
        self.fuse_flag = fuse_flag
        self.dataloader = dataloader
        self._check_mode()
        self.device = None

    def forward(self, ann):
        ann = fx.symbolic_trace(ann).to(self.device)
        if self.device is None:
            self.device = next(ann.parameters()).device
        ann.eval()
        ann_fused = self.fuse(ann, fuse_flag=self.fuse_flag).to(self.device)
        ann_with_hook = self.set_voltagehook(ann_fused, mode=self.mode).to(self.device)
        for _, (imgs, _) in enumerate(tqdm(self.dataloader)):
            ann_with_hook(imgs.to(self.device))
        snn = self.replace_by_ifnode(ann_with_hook).to(self.device)
        return snn  # return type: GraphModule

    def _check_mode(self):
        err_msg = 'You have used a non-defined VoltageScale Method.'
        if isinstance(self.mode, str):
            if self.mode[-1] == '%':
                try:
                    float(self.mode[:-1])
                except ValueError:
                    raise NotImplementedError(err_msg)
            elif self.mode.lower() in ['max']:
                pass
            else:
                raise NotImplementedError(err_msg)
        elif isinstance(self.mode, float):
            try:
                assert (self.mode <= 1 and self.mode > 0)
            except AssertionError:
                raise NotImplementedError(err_msg)
        else:
            raise NotImplementedError(err_msg)

    def fuse(self, fx_model: torch.fx.GraphModule, fuse_flag=True) -> torch.fx.GraphModule:
        """
        * :ref:`API in English <Converter.fuse-en>`

        .. _Converter.fuse-cn:

        :param fx_model: 原模型
        :type fx_model: torch.fx.GraphModule
        :param fuse_flag: 标志位，设置为True，则进行conv与bn的融合，反之不进行。
        :type fuse_flag: bool
        :return: conv层和bn层融合后的模型.
        :rtype: torch.fx.GraphModule

        ``fuse`` 用于conv与bn的融合。

        * :ref:`中文API <Converter.fuse-cn>`

        .. _Converter.fuse-en:

        :param fx_model: Original fx_model
        :type fx_model: torch.fx.GraphModule
        :param fuse_flag: Bool specifying if fusion of the conv and the bn happens, by default it happens.
        :type fuse_flag: bool
        :return: fx_model whose conv layer and bn layer have been fused.
        :rtype: torch.fx.GraphModule

        ``fuse`` is used to fuse conv layer and bn layer.

        """
        if not fuse_flag:
            return fx_model
        patterns = [(nn.Conv1d, nn.BatchNorm1d),
                    (nn.Conv2d, nn.BatchNorm2d),
                    (nn.Conv3d, nn.BatchNorm3d)]

        modules = dict(fx_model.named_modules())

        for pattern in patterns:
            for node in fx_model.graph.nodes:
                if self._matches_module_pattern(pattern, node,
                                                modules):
                    if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                        continue
                    conv = modules[node.args[0].target]
                    bn = modules[node.target]
                    fused_conv = fuse_conv_bn_eval(conv, bn)
                    self._replace_node_module(node.args[0], modules,
                                              fused_conv)
                    node.replace_all_uses_with(node.args[0])
                    fx_model.graph.erase_node(node)
        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()  # remove unused bn modules
        fx_model.recompile()
        return fx_model

    def set_voltagehook(self, fx_model: torch.fx.GraphModule, mode='Max') -> torch.fx.GraphModule:
        """
        * :ref:`API in English <Converter.set_voltagehook-en>`

        .. _Converter.set_voltagehook-cn:

        :param fx_model: 原模型
        :type fx_model: torch.fx.GraphModule
        :param mode: 转换模式。目前支持三种模式，最大电流转换模式，99.9%电流转换模式，以及缩放转换模式
        :type mode: str, float
        :return: 带有VoltageHook的模型.
        :rtype: torch.fx.GraphModule

        ``set_voltagehook`` 用于给模型添加VoltageHook模块。这里实现了常见的三种模式，同上。

        * :ref:`中文API <Converter.set_voltagehook-cn>`

        .. _Converter.set_voltagehook-en:

        :param fx_model: Original fx_model
        :type fx_model: torch.fx.GraphModule
        :param mode: Conversion mode. Now support three mode, MaxNorm, RobustNorm(99.9%), and scaling mode
        :type mode: str, float
        :return: fx_model with VoltageHook.
        :rtype: torch.fx.GraphModule

        ``set_voltagehook`` is used to add VoltageHook to fx_model. Three common methods are implemented here, the same as Converter.mode.

        """
        modules = dict(fx_model.named_modules())
        for node in fx_model.graph.nodes:
            if node.op != 'call_module':
                continue
            if type(modules[node.target]) is nn.ReLU:
                seq = nn.Sequential(modules[node.target], VoltageHook(mode=mode))
                self._replace_node_module(node, modules, seq)
        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model

    def replace_by_ifnode(self, fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """
        * :ref:`API in English <Converter.replace_by_ifnode-en>`

        .. _Converter.replace_by_ifnode-cn:

        :param fx_model: 原模型
        :type fx_model: torch.fx.GraphModule
        :return: 将ReLU替换为IF脉冲神经元后的模型.
        :rtype: torch.fx.GraphModule

        ``replace_by_ifnode`` 用于将模型的ReLU替换为IF脉冲神经元。

        * :ref:`中文API <Converter.replace_by_ifnode-cn>`

        .. _Converter.replace_by_ifnode-en:

        :param fx_model: Original fx_model
        :type fx_model: torch.fx.GraphModule
        :return: fx_model whose ReLU has been replaced by IF neuron.
        :rtype: torch.fx.GraphModule

        ``replace_by_ifnode`` is used to replace ReLU with IF neuron.

        """
        modules = dict(fx_model.named_modules())
        for node in fx_model.graph.nodes:  # Seq as one node
            if node.op != 'call_module':
                continue
            if type(modules[node.target]) is nn.Sequential and len(modules[node.target]) == 2 and type(
                    modules[node.target][0]) is nn.ReLU and type(modules[node.target][1]) is VoltageHook:
                s = modules[node.target][1].scale.item()
                seq = nn.Sequential(
                    VoltageScaler(1.0 / s),
                    neuron.IFNode(v_threshold=1., v_reset=None),
                    VoltageScaler(s)
                )
                self._replace_node_module(node, modules, seq)
        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model

    def _replace_node_module(self, node: fx.Node, modules: Dict[str, Any],
                             new_module: torch.nn.Module):
        assert (isinstance(node.target, str))
        parent_name, name = self._parent_name(node.target)
        modules[node.target] = new_module
        setattr(modules[parent_name], name, new_module)

    def _parent_name(self, target: str) -> Tuple[str, str]:
        """
        Splits a qualname into parent path and last atom.
        For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
        """
        *parent, name = target.rsplit('.', 1)
        return parent[0] if parent else '', name

    def _matches_module_pattern(self, pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]) -> bool:
        if len(node.args) == 0:
            return False
        nodes: Tuple[Any, fx.Node] = (node.args[0], node)
        for expected_type, current_node in zip(pattern, nodes):
            if not isinstance(current_node, fx.Node):
                return False
            if current_node.op != 'call_module':
                return False
            if not isinstance(current_node.target, str):
                return False
            if current_node.target not in modules:
                return False
            if type(modules[current_node.target]) is not expected_type:
                return False
        return True
