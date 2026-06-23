import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

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
    TDMultiheadAttention,
    TDScaledDotProductAttention,
)
from spikingjelly.activation_based.ann2snn.rules import ActivationRule, ReLURule
from spikingjelly.activation_based.ann2snn.threshold import ThresholdOptimizer


class Converter:
    def __init__(
        self,
        dataloader: Iterable,
        device: Optional[Union[torch.device, str]] = None,
        mode: Union[str, float] = "Max",
        momentum: float = 0.1,
        fuse_flag: bool = True,
        rules: Optional[List[ActivationRule]] = None,
        neuron_factory: Optional[NeuronFactory] = None,
        threshold_optimizer: Optional[ThresholdOptimizer] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <Converter.__init__-cn>` | :ref:`English <Converter.__init__-en>`

        ----

        .. _Converter.__init__-cn:

        * **中文**

        ``Converter`` 是 ANN2SNN 转换器对象，而不是用于推理的
        :class:`torch.nn.Module`。它提供显式转换方法：
        :meth:`convert_to_spiking_neurons` 用于传统 ReLU→IFNode 校准转换，
        :meth:`replace_by_td_operators` 用于 Transformer TD core operator
        替换。

        ANN2SNN教程见此处 `ANN转换SNN <https://spikingjelly.readthedocs.io/zh_CN/latest/tutorials/cn/ann2snn.html>`_ 。

        目前支持三种转换模式，由参数mode进行设置。

        ReLU→IFNode 转换后 ReLU 模块被删除，SNN 需要的新模块（包括
        VoltageScaler、IFNode 等）被创建并存放在 snn tailor 父模块中。
        TD operator 替换不使用校准数据，只将支持的 ANN 模块替换为 TD
        等价模块。

        由于返回值的类型为 ``fx.GraphModule``，建议使用
        ``print(fx.GraphModule.graph)`` 查看计算图及前向传播关系。更多 API
        参见 `GraphModule <https://pytorch.org/docs/stable/fx.html?highlight=graphmodule#torch.fx.GraphModule>`_ 。

        .. warning::

            必须确保ANN中的 ``ReLU`` 为module而非function。

            您最好在ANN模型中使用平均池化而不是最大池化。否则，可能会损害转换后的SNN模型的性能。

        :param dataloader: 数据加载器。迭代返回的每个 batch 必须支持
            ``data[0]`` 取出输入张量，例如 ``(input, label)`` 或 ``(input,)``。
        :type dataloader: Iterable
        :param device: Device
        :type device: torch.device or str or None
        :param mode: 转换模式。目前支持三种模式：最大电流转换模式 ``mode="max"``，
            99.9% 电流转换模式 ``mode="99.9%"``，以及缩放转换模式 ``mode=x`` （``0 < x <= 1``）。
        :type mode: str, float
        :param momentum: 动量值，用于modules.VoltageHook
        :type momentum: float
        :param fuse_flag: 标志位，设置为True，则进行conv与bn的融合，反之不进行。
        :type fuse_flag: bool
        :param rules: 激活函数转换规则列表。每个规则必须实现
            ``match``、``insert_hooks``、``find_replacements`` 和
            ``replace_with_neurons`` 。默认使用 ``[ReLURule()]`` 。
        :type rules: Optional[List[ActivationRule]]
        :param neuron_factory: 脉冲神经元工厂。默认使用
            ``NeuronFactory()`` （IFNode, threshold=1.0）。
        :type neuron_factory: Optional[NeuronFactory]
        :param threshold_optimizer: 阈值优化器。默认使用
            ``ThresholdOptimizer(strategy="fixed")``。
        :type threshold_optimizer: Optional[ThresholdOptimizer]

        ----

        .. _Converter.__init__-en:

        * **English**

        ``Converter`` is an ANN2SNN conversion driver, not a
        :class:`torch.nn.Module` for inference. It provides explicit conversion
        methods: :meth:`convert_to_spiking_neurons` for the traditional
        ReLU-to-IFNode calibrated conversion, and :meth:`replace_by_td_operators`
        for Transformer TD core operator replacement.

        ANN2SNN tutorial is here `ANN2SNN <https://spikingjelly.readthedocs.io/zh_CN/latest/tutorials/en/ann2snn.html>`_ .

        Three common methods are implemented here, which can be selected by the value of parameter mode.

        In the ReLU-to-IFNode path, ReLU modules will be removed, and new modules
        needed by SNN, such as VoltageScaler and IFNode, will be created and
        stored in the parent module ``snn tailor``. The TD operator replacement
        path does not use calibration data; it only replaces supported ANN
        modules with TD-equivalent modules.

        Since the converted model is an ``fx.GraphModule``, use
        ``print(fx.GraphModule.graph)`` to inspect the generated computation
        graph. More APIs are here `GraphModule <https://pytorch.org/docs/stable/fx.html?highlight=graphmodule#torch.fx.GraphModule>`_ .

        .. warning::

            Make sure that ``ReLU`` is module rather than function.

            You'd better use ``avgpool`` rather than ``maxpool`` in your ann model. If not, the performance of the converted snn model may be ruined.

        :param dataloader: Dataloader for converting. Each yielded batch must
            support ``data[0]`` as the input tensor, for example
            ``(input, label)`` or ``(input,)``.
        :type dataloader: Iterable
        :param device: Device
        :type device: torch.device or str or None
        :param mode: Conversion mode. Now support three mode,
            MaxNorm (``mode="max"``), RobustNorm (``mode="99.9%"``), and
            scaling mode (``mode=x``, where ``0 < x <= 1``).
        :type mode: str, float
        :param momentum: Momentum value used by modules.VoltageHook
        :type momentum: float
        :param fuse_flag: Bool specifying if fusion of the conv and the bn happens, by default it happens.
        :type fuse_flag: bool
        :param rules: List of activation conversion rules. Each rule must
            implement ``match``, ``insert_hooks``, ``find_replacements`` and
            ``replace_with_neurons``. Defaults to ``[ReLURule()]``.
        :type rules: Optional[List[ActivationRule]]
        :param neuron_factory: Neuron factory. Defaults to ``NeuronFactory()`` (IFNode, threshold=1.0).
        :type neuron_factory: Optional[NeuronFactory]
        :param threshold_optimizer: Threshold optimizer. Defaults to
            ``ThresholdOptimizer(strategy="fixed")``.
        :type threshold_optimizer: Optional[ThresholdOptimizer]
        """
        self.mode = mode
        self.fuse_flag = fuse_flag
        self.dataloader = dataloader
        self._check_mode()
        self.device = device
        self.momentum = momentum
        self.rules = rules if rules is not None else [ReLURule()]
        self.neuron_factory = (
            neuron_factory if neuron_factory is not None else NeuronFactory()
        )
        self.threshold_optimizer = (
            threshold_optimizer
            if threshold_optimizer is not None
            else ThresholdOptimizer()
        )

    def convert_to_spiking_neurons(self, ann: nn.Module) -> torch.fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <Converter.convert_to_spiking_neurons-cn>` | :ref:`English <Converter.convert_to_spiking_neurons-en>`

        ----

        .. _Converter.convert_to_spiking_neurons-cn:

        * **中文**

        将带有 ReLU module 的 ANN 转换为 SNN ``GraphModule``。该方法会执行
        FX tracing、可选 Conv-BN 融合、VoltageHook 校准和神经元替换。

        :param ann: 待转换的 ANN。
        :type ann: torch.nn.Module
        :return: 转换得到的 SNN。
        :rtype: torch.fx.GraphModule

        ----

        .. _Converter.convert_to_spiking_neurons-en:

        * **English**

        Convert an ANN with ReLU modules to an SNN ``GraphModule``. This method
        performs FX tracing, optional Conv-BN fusion, VoltageHook calibration,
        and neuron replacement.

        :param ann: ANN to be converted.
        :type ann: torch.nn.Module
        :return: Converted SNN.
        :rtype: torch.fx.GraphModule
        """
        if self.device is None:
            self.device = next(ann.parameters()).device
        ann = fx.symbolic_trace(ann).to(self.device)
        ann.eval()
        ann_fused = self.fuse(ann, fuse_flag=self.fuse_flag).to(self.device)
        ann_with_hook = self.set_voltagehook(
            ann_fused, momentum=self.momentum, mode=self.mode, rules=self.rules
        ).to(self.device)
        with torch.no_grad():
            for _, data in enumerate(tqdm(self.dataloader)):
                imgs = self._extract_batch_input(data)
                ann_with_hook(
                    torch.as_tensor(imgs).to(device=self.device, dtype=torch.float)
                )
        snn = self.replace_by_neurons(ann_with_hook).to(self.device)
        return snn

    def replace_by_td_operators(self, ann: nn.Module) -> torch.fx.GraphModule:
        r"""
        **API Language:**
        :ref:`中文 <Converter.replace_by_td_operators-cn>` |
        :ref:`English <Converter.replace_by_td_operators-en>`

        ----

        .. _Converter.replace_by_td_operators-cn:

        * **中文**

        将 ANN 中支持的 core modules 和窄 attention 子集替换为
        temporal-difference (TD) 等价算子，并返回 ``GraphModule``。当前自动
        替换 :class:`torch.nn.Linear`、:class:`torch.nn.LayerNorm`、
        :class:`torch.nn.GELU`、literal ``dropout_p=0.0`` 的
        :func:`torch.nn.functional.scaled_dot_product_attention` 调用，以及
        ``dropout=0.0``、``batch_first=True``、``need_weights=False`` 的
        :class:`torch.nn.MultiheadAttention` 调用。该方法不插入
        ``VoltageHook``，不运行 dataloader 校准。返回模型会保留输入模型及
        已替换模块的 training/eval 状态。

        该转换路径面向完整时间序列输入，约定转换后模型的输入张量使用第
        0 维作为时间维，形状通常为 ``[T, ...]`` 且 ``T > 0``。TD 算子输出
        浮点差分值，不是二值脉冲，也不表示 fully spike-driven 在线执行。
        dtype、device 与后端行为跟随被替换算子的 PyTorch 实现；当前没有
        CuPy / Triton 专用路径。该方法不改变输入模型本身，而是返回 tracing
        后的 ``GraphModule``。

        :param ann: 待转换的 ANN。
        :type ann: torch.nn.Module
        :return: 已替换 core TD operators 的 ``GraphModule``。
        :rtype: torch.fx.GraphModule
        :raises ValueError: 若 FX 图中包含当前不支持的 TD attention 配置，例如
            非零 SDPA dropout、动态 SDPA 配置、``enable_gqa=True``、
            ``nn.MultiheadAttention`` 的 ``dropout != 0``、
            ``batch_first=False``、``need_weights=True``、
            ``key_padding_mask`` 或非 packed q/k/v 参数。

        ----

        .. _Converter.replace_by_td_operators-en:

        * **English**

        Replace supported core modules and a narrow attention subset in an ANN
        with temporal-difference (TD) equivalent operators and return a
        ``GraphModule``. Currently, :class:`torch.nn.Linear`,
        :class:`torch.nn.LayerNorm`, :class:`torch.nn.GELU`, literal
        ``dropout_p=0.0``
        :func:`torch.nn.functional.scaled_dot_product_attention` calls, and
        :class:`torch.nn.MultiheadAttention` calls with ``dropout=0.0``,
        ``batch_first=True`` and ``need_weights=False`` are replaced
        automatically. This method does not insert ``VoltageHook`` and does not
        run dataloader calibration. The returned model preserves the
        training/eval state of the input model and replaced modules.

        This conversion path targets complete time-sequence inputs. Converted
        models conventionally use dimension 0 as the time dimension, with shape
        ``[T, ...]`` and ``T > 0``. TD operators output floating-point
        differential values; they are not binary spikes and do not represent
        fully spike-driven online execution. Dtype, device, and backend behavior
        follow the PyTorch implementation of each replaced operator; there is no
        CuPy / Triton specific path currently. This method does not mutate the
        input model itself; it returns a traced ``GraphModule``.

        :param ann: ANN to be converted.
        :type ann: torch.nn.Module
        :return: ``GraphModule`` with core TD operators replaced.
        :rtype: torch.fx.GraphModule
        :raises ValueError: If the FX graph contains unsupported TD attention
            configurations, such as nonzero SDPA dropout, dynamic SDPA
            configuration, ``enable_gqa=True``, ``nn.MultiheadAttention`` with
            ``dropout != 0``, ``batch_first=False``, ``need_weights=True``,
            ``key_padding_mask``, or non-packed q/k/v parameters.
        """
        device = self.device
        if device is None:
            try:
                device = next(ann.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        fx_model = fx.symbolic_trace(ann).to(device)

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
    def _extract_batch_input(data):
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, (list, tuple)):
            if not data:
                raise ValueError("Batch data is an empty list or tuple.")
            return data[0]
        if isinstance(data, dict):
            if not data:
                raise ValueError("Batch data is an empty dictionary.")
            for key in ("input", "image", "img", "x", "data", "pixel_values"):
                if key in data:
                    return data[key]
            return next(iter(data.values()))
        return data

    def _check_mode(self):
        err_msg = "You have used a non-defined VoltageScale Method."
        if isinstance(self.mode, str):
            if self.mode[-1] == "%":
                try:
                    float(self.mode[:-1])
                except ValueError:
                    raise NotImplementedError(err_msg)
            elif self.mode.lower() in ["max"]:
                pass
            else:
                raise NotImplementedError(err_msg)
        elif isinstance(self.mode, float):
            try:
                assert self.mode <= 1 and self.mode > 0
            except AssertionError:
                raise NotImplementedError(err_msg)
        else:
            raise NotImplementedError(err_msg)

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
        dropout_p = Converter._get_literal_argument(node, "dropout_p", 4, 0.0)
        if not isinstance(dropout_p, (int, float)) or float(dropout_p) != 0.0:
            raise ValueError(
                "TD SDPA conversion only supports literal dropout_p=0.0, "
                f"but got {dropout_p!r}."
            )
        enable_gqa = Converter._get_literal_argument(node, "enable_gqa", 7, False)
        if enable_gqa is not False:
            raise ValueError("TD SDPA conversion does not support enable_gqa=True.")

        is_causal = Converter._get_literal_argument(node, "is_causal", 5, False)
        if not isinstance(is_causal, bool):
            raise ValueError(
                "TD SDPA conversion only supports literal bool is_causal, "
                f"but got {is_causal!r}."
            )
        scale = Converter._get_literal_argument(node, "scale", 6, None)
        if scale is not None and not isinstance(scale, (int, float)):
            raise ValueError(
                "TD SDPA conversion only supports literal numeric scale or None, "
                f"but got {scale!r}."
            )

        return {
            "query": node.args[0],
            "key": node.args[1],
            "value": node.args[2],
            "attn_mask": Converter._get_literal_argument(node, "attn_mask", 3, None),
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
        if module.bias_k is not None or module.bias_v is not None:
            raise ValueError("TD MHA conversion does not support add_bias_kv.")
        if module.add_zero_attn:
            raise ValueError("TD MHA conversion does not support add_zero_attn.")

        need_weights = Converter._get_literal_argument(node, "need_weights", 4, True)
        if need_weights is not False:
            raise ValueError("TD MHA conversion requires need_weights=False.")
        key_padding_mask = Converter._get_literal_argument(
            node, "key_padding_mask", 3, None
        )
        if key_padding_mask is not None:
            raise ValueError("TD MHA conversion does not support key_padding_mask.")
        average_attn_weights = Converter._get_literal_argument(
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

    @staticmethod
    def _make_td_operator(
        module: nn.Module,
        node: Optional[fx.Node] = None,
    ) -> Optional[nn.Module]:
        if type(module) is nn.Linear:
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

        if type(module) is nn.LayerNorm:
            td_module = TDLayerNorm(
                module.normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
                bias=module.bias is not None,
                device=(
                    module.weight.device
                    if module.weight is not None
                    else None
                ),
                dtype=(
                    module.weight.dtype
                    if module.weight is not None
                    else None
                ),
            )
            with torch.no_grad():
                if module.weight is not None:
                    td_module.weight.copy_(module.weight)
                if module.bias is not None:
                    td_module.bias.copy_(module.bias)
            td_module.train(module.training)
            return td_module

        if type(module) is nn.GELU:
            td_module = TDGELU(approximate=module.approximate)
            td_module.train(module.training)
            return td_module

        if type(module) is nn.MultiheadAttention:
            if node is None:
                raise ValueError("TD MHA conversion requires an FX node.")
            Converter._check_mha_node(module, node)
            td_module = TDMultiheadAttention(
                module.embed_dim,
                module.num_heads,
                dropout=module.dropout,
                bias=module.in_proj_bias is not None,
                batch_first=module.batch_first,
                device=module.in_proj_weight.device,
                dtype=module.in_proj_weight.dtype,
            )
            Converter._copy_mha_parameters(module, td_module)
            td_module.train(module.training)
            return td_module

        return None

    @staticmethod
    def fuse(
        fx_model: torch.fx.GraphModule, fuse_flag: bool = True
    ) -> torch.fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <Converter.fuse-cn>` | :ref:`English <Converter.fuse-en>`

        ----

        .. _Converter.fuse-cn:

        * **中文**

        ``fuse`` 用于conv与bn的融合。

        :param fx_model: 原模型
        :type fx_model: torch.fx.GraphModule
        :param fuse_flag: 标志位，设置为True，则进行conv与bn的融合，反之不进行。
        :type fuse_flag: bool
        :return: conv层和bn层融合后的模型.
        :rtype: torch.fx.GraphModule

        ----

        .. _Converter.fuse-en:

        * **English**

        ``fuse`` is used to fuse conv layer and bn layer.

        :param fx_model: Original fx_model
        :type fx_model: torch.fx.GraphModule
        :param fuse_flag: Bool specifying if fusion of the conv and the bn happens, by default it happens.
        :type fuse_flag: bool
        :return: fx_model whose conv layer and bn layer have been fused.
        :rtype: torch.fx.GraphModule
        """

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
                if type(modules[current_node.target]) is not expected_type:
                    return False
            return True

        def replace_node_module(
            node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module
        ):
            def parent_name(target: str) -> Tuple[str, str]:
                *parent, name = target.rsplit(".", 1)
                return parent[0] if parent else "", name

            assert isinstance(node.target, str)
            parent_name, name = parent_name(node.target)
            modules[node.target] = new_module
            setattr(modules[parent_name], name, new_module)

        patterns = [
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d),
        ]

        modules = dict(fx_model.named_modules())

        for pattern in patterns:
            for node in fx_model.graph.nodes:
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

    @staticmethod
    def set_voltagehook(
        fx_model: torch.fx.GraphModule,
        mode="Max",
        momentum=0.1,
        rules: Optional[List[ActivationRule]] = None,
    ) -> torch.fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <Converter.set_voltagehook-cn>` | :ref:`English <Converter.set_voltagehook-en>`

        ----

        .. _Converter.set_voltagehook-cn:

        * **中文**

        ``set_voltagehook`` 用于给模型添加VoltageHook模块。这里实现了常见的三种模式，同上。

        :param fx_model: 原模型
        :type fx_model: torch.fx.GraphModule
        :param mode: 转换模式。目前支持三种模式，最大电流转换模式，99.9%电流转换模式，以及缩放转换模式
        :type mode: str, float
        :param momentum: 动量值，用于VoltageHook
        :type momentum: float
        :param rules: 自定义的激活匹配规则列表。默认值为 ``None``，此时使用 ``[ReLURule()]``，即匹配
            ``ReLU`` 并为其前后插入 ``VoltageHook``。传入自定义规则可扩展匹配的激活类型或调整 hook 插入位置。
        :type rules: Optional[List[ActivationRule]]
        :return: 带有VoltageHook的模型.
        :rtype: torch.fx.GraphModule

        ----

        .. _Converter.set_voltagehook-en:

        * **English**

        ``set_voltagehook`` is used to add VoltageHook to fx_model. Three common methods are implemented here, the same as Converter.mode.

        :param fx_model: Original fx_model
        :type fx_model: torch.fx.GraphModule
        :param mode: Conversion mode. Now support three mode, MaxNorm, RobustNorm(99.9%), and scaling mode
        :type mode: str, float
        :param momentum: momentum value used by VoltageHook
        :type momentum: float
        :param rules: Optional list of activation matching rules. When ``None`` (default) ``[ReLURule()]`` is used,
            which matches ``ReLU`` and wraps it with ``VoltageHook``.
            Pass custom rules to match additional activation types or to change where hooks are inserted.
        :type rules: Optional[List[ActivationRule]]
        :return: fx_model with VoltageHook.
        :rtype: torch.fx.GraphModule
        """
        hook_factory = HookFactory(mode=mode, momentum=momentum)
        hook_counts_per_prefix: Dict[str, int] = {}
        modules = dict(fx_model.named_modules())
        active_rules = rules if rules is not None else [ReLURule()]

        for node in list(fx_model.graph.nodes):
            if node.op != "call_module":
                continue
            if node.target not in modules:
                continue
            for rule in active_rules:
                if rule.match(node, modules):
                    rule.insert_hooks(
                        fx_model, node, hook_factory, hook_counts_per_prefix
                    )
                    modules = dict(fx_model.named_modules())
                    break

        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model

    def replace_by_neurons(
        self, fx_model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <Converter.replace_by_neurons-cn>` | :ref:`English <Converter.replace_by_neurons-en>`

        ----

        .. _Converter.replace_by_neurons-cn:

        * **中文**

        将 ``self.rules`` 匹配到的激活节点替换为脉冲神经元，并按 ``self.threshold_optimizer`` 计算出的阈值
        完成 ``VoltageScaler(1/v_threshold) -> Neuron -> VoltageScaler(v_threshold)`` 的等价变换。
        默认规则、神经元工厂与阈值优化器可复现原 ``replace_by_ifnode`` 的 ``ReLU -> IFNode`` 替换语义。

        :param fx_model: 已插入校准 hook 的 ``GraphModule``
        :type fx_model: torch.fx.GraphModule
        :return: 激活节点被替换为脉冲神经元后的模型
        :rtype: torch.fx.GraphModule

        ----

        .. _Converter.replace_by_neurons-en:

        * **English**

        Replace activations matched by ``self.rules`` with spiking neurons, applying the
        ``VoltageScaler(1/v_threshold) -> Neuron -> VoltageScaler(v_threshold)`` transformation
        using the threshold computed by ``self.threshold_optimizer``. With the default rule,
        neuron factory and threshold optimizer this reproduces the original
        ``replace_by_ifnode`` ``ReLU -> IFNode`` replacement semantics.

        :param fx_model: ``GraphModule`` with calibration hooks already inserted.
        :type fx_model: torch.fx.GraphModule
        :return: Model with activations replaced by spiking neurons.
        :rtype: torch.fx.GraphModule
        """
        return Converter._replace_by_neurons_impl(
            fx_model,
            self.rules,
            self.neuron_factory,
            self.threshold_optimizer,
        )

    @staticmethod
    def _replace_by_neurons_impl(
        fx_model: torch.fx.GraphModule,
        rules: List[ActivationRule],
        neuron_factory: NeuronFactory,
        threshold_optimizer: ThresholdOptimizer,
    ) -> torch.fx.GraphModule:
        replaced_hooks = set()
        replaced_activations = set()
        for rule in rules:
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
                    neuron_factory,
                    threshold_optimizer,
                )

        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()
        fx_model.recompile()
        return fx_model

    @staticmethod
    def replace_by_ifnode(fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        r"""
        Replace ReLU with IF neurons (legacy API, use :meth:`replace_by_neurons` instead).

        :deprecated: Use :meth:`replace_by_neurons` instead.
        :param fx_model: Model with calibration hooks inserted.
        :type fx_model: torch.fx.GraphModule
        :return: Model with ReLU replaced by IF neurons.
        :rtype: torch.fx.GraphModule
        """
        warnings.warn(
            "replace_by_ifnode is deprecated, use replace_by_neurons instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Converter._replace_by_neurons_impl(
            fx_model,
            [ReLURule()],
            NeuronFactory(),
            ThresholdOptimizer(),
        )
