from typing import TYPE_CHECKING, Dict, Iterator, Protocol, Tuple

import torch.nn as nn
from torch import fx

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.ann2snn.modules import VoltageHook, VoltageScaler

if TYPE_CHECKING:
    from spikingjelly.activation_based.ann2snn.factories import (
        HookFactory,
        NeuronFactory,
    )
    from spikingjelly.activation_based.ann2snn.threshold import ThresholdOptimizer


class ActivationRule(Protocol):
    r"""
    **API Language:**
    :ref:`中文 <ActivationRule-cn>` | :ref:`English <ActivationRule-en>`

    ----

    .. _ActivationRule-cn:

    * **中文**

    激活函数转换规则协议。实现该协议即可接入新的 ANN→SNN 转换算法。规则需要负责：

    1. 通过 :meth:`match` 判断是否处理某个 ``fx.Node``；
    2. 通过 :meth:`insert_hooks` 在节点后插入校准 hook；
    3. 通过 :meth:`find_replacements` 找到 ``(activation_node, hook_node)`` 对；
    4. 通过 :meth:`replace_with_neurons` 将激活节点与 hook 替换为脉冲神经元结构。

    ----

    .. _ActivationRule-en:

    * **English**

    Protocol for activation-to-neuron conversion rules. Implement this
    protocol to plug a new ANN→SNN algorithm into the converter. A rule
    must:

    1. decide whether it handles a given ``fx.Node`` via :meth:`match`;
    2. insert a calibration hook after the node via :meth:`insert_hooks`;
    3. enumerate ``(activation_node, hook_node)`` pairs to replace via
       :meth:`find_replacements`;
    4. replace the activation + hook pair with spiking neurons via
       :meth:`replace_with_neurons`.
    """

    def match(self, node: fx.Node, modules: Dict[str, nn.Module]) -> bool:
        r"""
        **API Language:**
        :ref:`中文 <ActivationRule.match-cn>` | :ref:`English <ActivationRule.match-en>`

        ----

        .. _ActivationRule.match-cn:

        * **中文**

        判断该规则是否处理给定节点。

        :param node: 待检查的 ``fx.Node``。
        :type node: fx.Node
        :param modules: ``fx.GraphModule.named_modules()`` 得到的模块名字典。
        :type modules: Dict[str, nn.Module]
        :return: 若该规则负责此节点则返回 ``True``。
        :rtype: bool

        ----

        .. _ActivationRule.match-en:

        * **English**

        Return *True* if this rule handles the given graph node.

        :param node: The ``fx.Node`` to check.
        :type node: fx.Node
        :param modules: Module-name dictionary obtained from
            ``fx.GraphModule.named_modules()``.
        :type modules: Dict[str, nn.Module]
        :return: ``True`` if this rule handles the node.
        :rtype: bool
        """
        ...

    def insert_hooks(
        self,
        fx_model: fx.GraphModule,
        node: fx.Node,
        hook_factory: "HookFactory",
        hook_counts_per_prefix: Dict[str, int],
    ) -> fx.Node:
        r"""
        **API Language:**
        :ref:`中文 <ActivationRule.insert_hooks-cn>` | :ref:`English <ActivationRule.insert_hooks-en>`

        ----

        .. _ActivationRule.insert_hooks-cn:

        * **中文**

        在 ``node`` 之后插入一个由 ``hook_factory`` 创建的校准 hook，并将新节点加入
        ``fx_model``。``hook_counts_per_prefix`` 用于在多 hook 场景下生成唯一的目标
        名称。

        :param fx_model: 待修改的 ``GraphModule``。
        :type fx_model: fx.GraphModule
        :param node: 触发 hook 插入的 ``fx.Node``。
        :type node: fx.Node
        :param hook_factory: 校准 hook 工厂。
        :type hook_factory: HookFactory
        :param hook_counts_per_prefix: 用于生成唯一 hook 目标名的前缀计数器。
        :type hook_counts_per_prefix: Dict[str, int]
        :return: 新插入的 hook 节点。
        :rtype: fx.Node

        ----

        .. _ActivationRule.insert_hooks-en:

        * **English**

        Insert a calibration hook created by ``hook_factory`` after ``node`` and
        register the new node inside ``fx_model``. ``hook_counts_per_prefix`` is
        used to generate unique hook target names when multiple hooks are
        inserted.

        :param fx_model: The ``GraphModule`` to modify.
        :type fx_model: fx.GraphModule
        :param node: The ``fx.Node`` after which the hook is inserted.
        :type node: fx.Node
        :param hook_factory: Hook factory used to build the calibration hook.
        :type hook_factory: HookFactory
        :param hook_counts_per_prefix: Per-prefix counters used to build
            unique hook target names.
        :type hook_counts_per_prefix: Dict[str, int]
        :return: The newly inserted hook node.
        :rtype: fx.Node
        """
        ...

    def find_replacements(
        self, fx_model: fx.GraphModule, modules: Dict[str, nn.Module]
    ) -> Iterator[Tuple[fx.Node, fx.Node]]:
        r"""
        **API Language:**
        :ref:`中文 <ActivationRule.find_replacements-cn>` | :ref:`English <ActivationRule.find_replacements-en>`

        ----

        .. _ActivationRule.find_replacements-cn:

        * **中文**

        遍历 ``fx_model``，产出需要被替换的 ``(activation_node, hook_node)`` 对。
        对于非标准图结构的规则，应重写该方法实现自定义遍历。

        :param fx_model: 已插入校准 hook 的 ``GraphModule``。
        :type fx_model: fx.GraphModule
        :param modules: ``fx.GraphModule.named_modules()`` 得到的模块名字典。
        :type modules: Dict[str, nn.Module]
        :return: 形如 ``(activation_node, hook_node)`` 的迭代器。
        :rtype: Iterator[Tuple[fx.Node, fx.Node]]

        ----

        .. _ActivationRule.find_replacements-en:

        * **English**

        Iterate over ``fx_model`` and yield ``(activation_node, hook_node)``
        pairs to replace. Rules with non-standard graph patterns should
        override this method with their own traversal.

        :param fx_model: ``GraphModule`` with calibration hooks already
            inserted.
        :type fx_model: fx.GraphModule
        :param modules: Module-name dictionary obtained from
            ``fx.GraphModule.named_modules()``.
        :type modules: Dict[str, nn.Module]
        :return: Iterator of ``(activation_node, hook_node)`` pairs.
        :rtype: Iterator[Tuple[fx.Node, fx.Node]]
        """
        ...

    def replace_with_neurons(
        self,
        fx_model: fx.GraphModule,
        activation_node: fx.Node,
        hook_node: fx.Node,
        neuron_factory: "NeuronFactory",
        threshold_optimizer: "ThresholdOptimizer",
    ) -> None:
        r"""
        **API Language:**
        :ref:`中文 <ActivationRule.replace_with_neurons-cn>` | :ref:`English <ActivationRule.replace_with_neurons-en>`

        ----

        .. _ActivationRule.replace_with_neurons-cn:

        * **中文**

        将 ``activation_node`` 与 ``hook_node`` 替换为对应的脉冲神经元结构。``threshold``
        由 ``threshold_optimizer`` 基于 hook 校准数据计算得到；神经元由
        ``neuron_factory`` 构造。

        :param fx_model: 待修改的 ``GraphModule``。
        :type fx_model: fx.GraphModule
        :param activation_node: 激活节点。
        :type activation_node: fx.Node
        :param hook_node: 校准 hook 节点。
        :type hook_node: fx.Node
        :param neuron_factory: 脉冲神经元工厂。
        :type neuron_factory: NeuronFactory
        :param threshold_optimizer: 阈值优化器。
        :type threshold_optimizer: ThresholdOptimizer
        :return: ``None``。
        :rtype: None

        ----

        .. _ActivationRule.replace_with_neurons-en:

        * **English**

        Replace the activation + hook pair with the corresponding spiking
        neuron structure. The threshold is computed by ``threshold_optimizer``
        from the calibration hook; the neuron is built by ``neuron_factory``.

        :param fx_model: The ``GraphModule`` to modify.
        :type fx_model: fx.GraphModule
        :param activation_node: The activation node.
        :type activation_node: fx.Node
        :param hook_node: The calibration hook node.
        :type hook_node: fx.Node
        :param neuron_factory: Spiking-neuron factory.
        :type neuron_factory: NeuronFactory
        :param threshold_optimizer: Threshold optimizer.
        :type threshold_optimizer: ThresholdOptimizer
        :return: ``None``.
        :rtype: None
        """
        ...


class ReLURule:
    r"""
    **API Language:**
    :ref:`中文 <ReLURule-cn>` | :ref:`English <ReLURule-en>`

    ----

    .. _ReLURule-cn:

    * **中文**

    ``nn.ReLU`` 转换规则。复现 SpikingJelly 原有行为：将每个 ``nn.ReLU`` 替换为
    ``VoltageScaler(1/s) -> IFNode -> VoltageScaler(s)``，其中 ``s`` 由
    :class:`ThresholdOptimizer` 基于 :class:`VoltageHook` 的校准结果计算。

    ----

    .. _ReLURule-en:

    * **English**

    Conversion rule for ``nn.ReLU`` modules. Reproduces the original
    SpikingJelly behaviour: each ``nn.ReLU`` is replaced by
    ``VoltageScaler(1/s) -> IFNode -> VoltageScaler(s)``, where ``s`` is
    computed by :class:`ThresholdOptimizer` from the
    :class:`VoltageHook` calibration data.
    """

    def match(self, node: fx.Node, modules: Dict[str, nn.Module]) -> bool:
        if node.op != "call_module":
            return False
        return type(modules[node.target]) is nn.ReLU

    def insert_hooks(
        self,
        fx_model: fx.GraphModule,
        node: fx.Node,
        hook_factory: "HookFactory",
        hook_counts_per_prefix: Dict[str, int],
    ) -> fx.Node:
        prefix = str(node.args[0]).split("_")

        if len(prefix) > 1:
            prefix = ".".join(prefix[:-1])
            counter = hook_counts_per_prefix.get(prefix, 0)
            hook_counts_per_prefix[prefix] = counter + 1
            target = f"{prefix}.voltage_hook_{counter}"
        else:
            prefix = "__FIRST_LEVEL_OF_MODULE__"
            counter = hook_counts_per_prefix.get(prefix, 0)
            hook_counts_per_prefix[prefix] = counter + 1
            target = f"voltage_hook_{counter}"

        m = hook_factory.create()
        return _add_module_and_node(fx_model, target, node, m, (node,))

    def find_replacements(
        self, fx_model: fx.GraphModule, modules: Dict[str, nn.Module]
    ) -> Iterator[Tuple[fx.Node, fx.Node]]:
        for hook_node in fx_model.graph.nodes:
            if hook_node.op != "call_module":
                continue
            if not isinstance(modules.get(hook_node.target), VoltageHook):
                continue
            if len(hook_node.args) == 0 or not isinstance(hook_node.args[0], fx.Node):
                continue
            activation_node = hook_node.args[0]
            if activation_node.op != "call_module":
                continue
            if activation_node.target not in modules:
                continue
            if self.match(activation_node, modules):
                yield activation_node, hook_node

    def replace_with_neurons(
        self,
        fx_model: fx.GraphModule,
        activation_node: fx.Node,
        hook_node: fx.Node,
        neuron_factory: "NeuronFactory",
        threshold_optimizer: "ThresholdOptimizer",
    ) -> None:
        # The prefix mirrors the hook target generated in insert_hooks. Keeping
        # the names paired makes exported GraphModules easier to inspect.
        prefix = hook_node.name.replace("voltage_hook_", "spiking")
        prefix = prefix.split("_")
        prefix = ".".join(prefix)

        hook = fx_model.get_submodule(hook_node.target)
        if not isinstance(hook, VoltageHook):
            raise TypeError("hook_node must target a VoltageHook module.")
        s = threshold_optimizer.compute_threshold(hook)
        target0 = f"{prefix}.scaler0"
        target1 = f"{prefix}.if_node"
        target2 = f"{prefix}.scaler1"

        m1 = neuron_factory.create(scale=s)
        neuron_threshold = getattr(m1, "v_threshold", 1.0)
        m0 = VoltageScaler(neuron_threshold / s)
        m2 = VoltageScaler(s)

        node0 = _add_module_and_node(
            fx_model, target0, hook_node, m0, activation_node.args
        )
        node1 = _add_module_and_node(fx_model, target1, node0, m1, (node0,))
        node2 = _add_module_and_node(fx_model, target2, node1, m2, args=(node1,))

        activation_node.replace_all_uses_with(node2)
        node2.args = (node1,)
        fx_model.graph.erase_node(hook_node)
        fx_model.graph.erase_node(activation_node)
        fx_model.delete_all_unused_submodules()


def _add_module_and_node(
    fx_model: fx.GraphModule,
    target: str,
    after: fx.Node,
    m: nn.Module,
    args: Tuple,
) -> fx.Node:
    fx_model.add_submodule(target=target, m=m)
    with fx_model.graph.inserting_after(n=after):
        new_node = fx_model.graph.call_module(module_name=target, args=args)
    return new_node
