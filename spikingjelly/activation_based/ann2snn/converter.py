from typing import Type, Dict, Any, Tuple, Iterable
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.ann2snn.modules import *
from torch import fx
from torch.nn.utils.fusion import fuse_conv_bn_eval
from tqdm import tqdm


class Converter(nn.Module):

    def __init__(self, dataloader, device=None, mode='Max', momentum=0.1, fuse_flag=True):
        """
        * :ref:`API in English <Converter.__init__-en>`

        .. _Converter.__init__-cn:

        :param dataloader: 数据加载器
        :type dataloader: Dataloader
        :param device: Device
        :type device: str
        :param mode: 转换模式。目前支持三种模式: 最大电流转换模式mode='max'，99.9%电流转换模式mode='99.9%'，以及缩放转换模式mode=x（0<x<=1）
        :type mode: str, float
        :param momentum: 动量值，用于modules.VoltageHook
        :type momentum: float
        :param fuse_flag: 标志位，设置为True，则进行conv与bn的融合，反之不进行。
        :type fuse_flag: bool

        ``Converter`` 用于将带有ReLU的ANN转换为SNN。

        ANN2SNN教程见此处 `ANN转换SNN <https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based/ann2snn.html>`_ 。

        目前支持三种转换模式，由参数mode进行设置。

        转换后ReLU模块被删除，SNN需要的新模块（包括VoltageScaler、IFNode等)被创建并存放在snn tailor父模块中。

        由于返回值的类型为fx.GraphModule，建议使用print(fx.GraphModule.graph)查看计算图及前向传播关系。更多API参见 `GraphModule <https://pytorch.org/docs/stable/fx.html?highlight=graphmodule#torch.fx.GraphModule>`_ 。

        .. warning::

            必须确保ANN中的 ``ReLU`` 为module而非function。

            您最好在ANN模型中使用平均池化而不是最大池化。否则，可能会损害转换后的SNN模型的性能。

        * :ref:`中文API <Converter.__init__-cn>`

        .. _Converter.__init__-en:

        :param dataloader: Dataloader for converting
        :type dataloader: Dataloader
        :param device: Device
        :type device: str
        :param mode: Conversion mode. Now support three mode, MaxNorm(mode='max'), RobustNorm(mode='99.9%'), and scaling mode(mode=x, where 0<x<=1)
        :type mode: str, float
        :param momentum: Momentum value used by modules.VoltageHook
        :type momentum: float
        :param fuse_flag: Bool specifying if fusion of the conv and the bn happens, by default it happens.
        :type fuse_flag: bool

        ``Converter`` is used to convert ANN with to SNN.

        ANN2SNN tutorial is here `ANN2SNN <https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/ann2snn.html>`_ .

        Three common methods are implemented here, which can be selected by the value of parameter mode.

        After converting, ReLU modules will be removed. And new modules needed by SNN, such as VoltageScaler and IFNode, will be created and stored in the parent module 'snn tailor'.

        Due to the type of the return model is fx.GraphModule, you can use 'print(fx.GraphModule.graph)' to view how modules links and the how the forward method works. More APIs are here `GraphModule <https://pytorch.org/docs/stable/fx.html?highlight=graphmodule#torch.fx.GraphModule>`_ .

        .. warning::

            Make sure that ``ReLU`` is module rather than function.

            You'd better use ``avgpool`` rather than ``maxpool`` in your ann model. If not, the performance of the converted snn model may be ruined.
        """
        super().__init__()
        self.mode = mode
        self.fuse_flag = fuse_flag
        self.dataloader = dataloader
        self._check_mode()
        self.device = device
        self.momentum = momentum

    def forward(self, ann: nn.Module):
        """
        * :ref:`API in English <Converter.forward-en>`

        .. _Converter.forward-cn:
        :param ann: 待转换的ann
        :type ann: torch.nn.Module
        :return: 转换得到的snn
        :rtype: torch.fx.GraphModule

        * :ref:`API in Chinese <Converter.forward-cn>`

        .. _Converter.forward-en:
        :param ann: ann to be converted
        :type ann: torch.nn.Module
        :return: snn
        :rtype: torch.fx.GraphModule

        """
        if self.device is None:
            self.device = next(ann.parameters()).device
        ann = fx.symbolic_trace(ann).to(self.device)
        ann.eval()
        ann_fused = self.fuse(ann, fuse_flag=self.fuse_flag).to(self.device)
        ann_with_hook = self.set_voltagehook(ann_fused, momentum=self.momentum, mode=self.mode).to(self.device)
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

    @staticmethod
    def fuse(fx_model: torch.fx.GraphModule, fuse_flag: bool = True) -> torch.fx.GraphModule:
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

        def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]) -> bool:
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

        def replace_node_module(node: fx.Node, modules: Dict[str, Any],
                                new_module: torch.nn.Module):
            def parent_name(target: str) -> Tuple[str, str]:
                """
                Splits a qualname into parent path and last atom.
                For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
                """
                *parent, name = target.rsplit('.', 1)
                return parent[0] if parent else '', name

            assert (isinstance(node.target, str))
            parent_name, name = parent_name(node.target)
            modules[node.target] = new_module
            setattr(modules[parent_name], name, new_module)

        if not fuse_flag:
            return fx_model
        patterns = [(nn.Conv1d, nn.BatchNorm1d),
                    (nn.Conv2d, nn.BatchNorm2d),
                    (nn.Conv3d, nn.BatchNorm3d)]

        modules = dict(fx_model.named_modules())

        for pattern in patterns:
            for node in fx_model.graph.nodes:
                if matches_module_pattern(pattern, node,
                                          modules):
                    if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                        continue
                    conv = modules[node.args[0].target]
                    bn = modules[node.target]
                    fused_conv = fuse_conv_bn_eval(conv, bn)
                    replace_node_module(node.args[0], modules,
                                        fused_conv)
                    node.replace_all_uses_with(node.args[0])
                    fx_model.graph.erase_node(node)
        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()  # remove unused bn modules
        fx_model.recompile()
        return fx_model

    @staticmethod
    def set_voltagehook(fx_model: torch.fx.GraphModule, mode='Max', momentum=0.1) -> torch.fx.GraphModule:
        """
        * :ref:`API in English <Converter.set_voltagehook-en>`

        .. _Converter.set_voltagehook-cn:

        :param fx_model: 原模型
        :type fx_model: torch.fx.GraphModule
        :param mode: 转换模式。目前支持三种模式，最大电流转换模式，99.9%电流转换模式，以及缩放转换模式
        :type mode: str, float
        :param momentum: 动量值，用于VoltageHook
        :type momentum: float
        :return: 带有VoltageHook的模型.
        :rtype: torch.fx.GraphModule

        ``set_voltagehook`` 用于给模型添加VoltageHook模块。这里实现了常见的三种模式，同上。

        * :ref:`中文API <Converter.set_voltagehook-cn>`

        .. _Converter.set_voltagehook-en:

        :param fx_model: Original fx_model
        :type fx_model: torch.fx.GraphModule
        :param mode: Conversion mode. Now support three mode, MaxNorm, RobustNorm(99.9%), and scaling mode
        :type mode: str, float
        :param momentum: momentum value used by VoltageHook
        :type momentum: float
        :return: fx_model with VoltageHook.
        :rtype: torch.fx.GraphModule

        ``set_voltagehook`` is used to add VoltageHook to fx_model. Three common methods are implemented here, the same as Converter.mode.

        """

        hook_cnt = -1
        for node in fx_model.graph.nodes:
            if node.op != 'call_module':
                continue
            if type(fx_model.get_submodule(node.target)) is nn.ReLU:
                hook_cnt += 1
                target = 'snn tailor.' + str(hook_cnt) + '.0'  # voltage_hook
                m = VoltageHook(momentum=momentum, mode=mode)
                new_node = Converter._add_module_and_node(fx_model, target, node, m
                                                          , (node,))
        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model

    @staticmethod
    def replace_by_ifnode(fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
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

        hook_cnt = -1
        for node in fx_model.graph.nodes:
            if node.op != 'call_module':
                continue
            if type(fx_model.get_submodule(node.target)) is VoltageHook:
                if type(fx_model.get_submodule(node.args[0].target)) is nn.ReLU:
                    hook_cnt += 1
                    hook_node = node
                    relu_node = node.args[0]
                    if len(relu_node.args) != 1:
                        raise NotImplementedError('The number of relu_node.args should be 1.')
                    s = fx_model.get_submodule(node.target).scale.item()
                    target0 = 'snn tailor.' + str(hook_cnt) + '.0'  # voltage_scaler
                    target1 = 'snn tailor.' + str(hook_cnt) + '.1'  # IF_node
                    target2 = 'snn tailor.' + str(hook_cnt) + '.2'  # voltage_scaler
                    m0 = VoltageScaler(1.0 / s)
                    m1 = neuron.IFNode(v_threshold=1., v_reset=None)
                    m2 = VoltageScaler(s)
                    node0 = Converter._add_module_and_node(fx_model, target0, hook_node, m0,
                                                           relu_node.args)
                    node1 = Converter._add_module_and_node(fx_model, target1, node0, m1
                                                           , (node0,))
                    node2 = Converter._add_module_and_node(fx_model, target2, node1, m2, args=(node1,))

                    relu_node.replace_all_uses_with(node2)
                    node2.args = (node1,)
                    fx_model.graph.erase_node(hook_node)
                    fx_model.graph.erase_node(relu_node)
                    fx_model.delete_all_unused_submodules()
        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model

    @staticmethod
    def _add_module_and_node(fx_model: fx.GraphModule, target: str, after: fx.Node, m: nn.Module,
                             args: Tuple) -> fx.Node:
        fx_model.add_submodule(target=target, m=m)
        with fx_model.graph.inserting_after(n=after):
            new_node = fx_model.graph.call_module(module_name=target, args=args)
        return new_node
