from __future__ import annotations

import operator
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fx

from spikingjelly.activation_based.ann2snn.operators import (
    SNNMatrixOperator,
    TDConv2d,
    TDGELU,
    TDLayerNorm,
    TDLinear,
    TDModule,
    TDMultiheadAttention,
    TDScaledDotProductAttention,
    TDSoftmax,
)
from spikingjelly.activation_based.ann2snn.recipes.base import ConversionRecipe
from spikingjelly.activation_based.ann2snn.recipes.step_mode_adapters import (
    _SHAPE_ONLY_MODULE_TYPES,
    _TRANSFORMER_SAFE_MODULE_TYPES,
    adapt_step_mode_graph,
)

if TYPE_CHECKING:
    from spikingjelly.activation_based.ann2snn.converter import Converter


__all__ = ["TransformerTDEquivalentRecipe"]


def _td_softmax_dim(dim: int) -> int:
    # `dim` indexes the ANN tensor. TD prepends a time axis, so non-negative
    # dims shift by +1. Negative dims stay negative because TDSoftmax resolves
    # them against the TD tensor rank, preserving the same feature axis.
    return dim + 1 if dim >= 0 else dim


class _TDTanh(TDModule):
    def ann_forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self._td_sequence_forward((x_seq,), torch.tanh)


class TransformerTDEquivalentRecipe(ConversionRecipe):
    r"""
    **API Language** - :ref:`中文 <TransformerTDEquivalentRecipe-cn>` | :ref:`English <TransformerTDEquivalentRecipe-en>`

    ----

    .. _TransformerTDEquivalentRecipe-cn:

    * **中文**

    Transformer TD-equivalent operator 替换 recipe。该 recipe 不插入
    observer，不运行 dataloader 校准，也不强制切换模型 train/eval 状态；它仅
    将当前支持的 ANN core modules 和窄 attention 子集替换为 TD 等价算子。

    ----

    .. _TransformerTDEquivalentRecipe-en:

    * **English**

    Transformer TD-equivalent operator replacement recipe. This recipe
    does not insert observers, does not run dataloader calibration, and does not
    force train/eval mode changes. It only replaces the currently supported ANN
    core modules and narrow attention subset with TD-equivalent operators.
    """

    def __init__(self, time_steps: Optional[int] = None) -> None:
        self.time_steps = time_steps

    def validate(self, converter: "Converter") -> None:
        if self.time_steps is None:
            return
        if (
            not isinstance(self.time_steps, int)
            or isinstance(self.time_steps, bool)
            or self.time_steps <= 0
        ):
            raise ValueError("time_steps must be a positive integer when set.")

    def replace(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        r"""
        **API Language** - :ref:`中文 <TransformerTDEquivalentRecipe.replace-cn>` | :ref:`English <TransformerTDEquivalentRecipe.replace-en>`

        ----

        .. _TransformerTDEquivalentRecipe.replace-cn:

        * **中文**

        将当前支持的 Transformer core modules、SDPA 调用和窄
        ``MultiheadAttention`` 调用替换为 TD-equivalent 算子。
        该步骤不插入 observer，也不运行 rate-coding 校准。

        :param converter: 执行当前 recipe 的转换器。
        :type converter: Converter
        :param fx_model: 已 trace 的 ``GraphModule``。
        :type fx_model: torch.fx.GraphModule
        :return: 替换后的 ``GraphModule``。
        :rtype: torch.fx.GraphModule
        :raises ValueError: 当 attention 调用或配置不在当前支持范围内时抛出。

        ----

        .. _TransformerTDEquivalentRecipe.replace-en:

        * **English**

        Replace currently supported Transformer core modules, SDPA calls and
        narrow ``MultiheadAttention`` calls with TD-equivalent
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

        self._replace_functional_td_ops(fx_model)

        sdpa_index = 0
        existing_modules = set(dict(fx_model.named_modules()).keys())
        for node in list(fx_model.graph.nodes):
            if (
                node.op != "call_function"
                or node.target is not F.scaled_dot_product_attention
            ):
                continue
            sdpa_kwargs = self._parse_sdpa_node(node)
            target = f"td_scaled_dot_product_attention_{sdpa_index}"
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
            existing_modules.add(target)
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

        return adapt_step_mode_graph(
            fx_model,
            context="TransformerTDEquivalentRecipe step-mode backend",
            wrap_module_types=_SHAPE_ONLY_MODULE_TYPES,
            safe_module_types=_TRANSFORMER_SAFE_MODULE_TYPES,
            safe_call_functions=(
                torch.div,
                operator.truediv,
            ),
        )

    def finalize(self, converter: "Converter", fx_model: fx.GraphModule) -> nn.Module:
        object.__setattr__(fx_model, "ann2snn_recipe", "transformer_td_equivalent")
        if self.time_steps is not None:
            object.__setattr__(fx_model, "time_steps", self.time_steps)
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
    def _get_tensor_argument(node: fx.Node, name: str, position: int) -> Any:
        if len(node.args) > position:
            return node.args[position]
        if name in node.kwargs:
            return node.kwargs[name]
        raise ValueError(
            f"TD conversion got malformed {node.target!r} node: missing {name!r}."
        )

    @staticmethod
    def _parse_sdpa_node(node: fx.Node) -> Dict[str, Any]:
        if len(node.args) < 3:
            raise ValueError("SDPA node must have query, key, and value arguments.")
        dropout_p = TransformerTDEquivalentRecipe._get_literal_argument(
            node, "dropout_p", 4, 0.0
        )
        if not isinstance(dropout_p, (int, float)) or float(dropout_p) != 0.0:
            raise ValueError(
                "TD SDPA conversion only supports literal dropout_p=0.0, "
                f"but got {dropout_p!r}."
            )
        enable_gqa = TransformerTDEquivalentRecipe._get_literal_argument(
            node, "enable_gqa", 7, False
        )
        if enable_gqa is not False:
            raise ValueError("TD SDPA conversion does not support enable_gqa=True.")

        is_causal = TransformerTDEquivalentRecipe._get_literal_argument(
            node, "is_causal", 5, False
        )
        if not isinstance(is_causal, bool):
            raise ValueError(
                "TD SDPA conversion only supports literal bool is_causal, "
                f"but got {is_causal!r}."
            )
        scale = TransformerTDEquivalentRecipe._get_literal_argument(
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
            "attn_mask": TransformerTDEquivalentRecipe._get_literal_argument(
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

        need_weights = TransformerTDEquivalentRecipe._get_literal_argument(
            node, "need_weights", 4, True
        )
        if need_weights is not False:
            raise ValueError("TD MHA conversion requires need_weights=False.")
        key_padding_mask = TransformerTDEquivalentRecipe._get_literal_argument(
            node, "key_padding_mask", 3, None
        )
        if key_padding_mask is not None:
            raise ValueError("TD MHA conversion does not support key_padding_mask.")
        average_attn_weights = TransformerTDEquivalentRecipe._get_literal_argument(
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
                if target.q_proj.bias is not None:
                    target.q_proj.bias.copy_(q_bias)
                if target.k_proj.bias is not None:
                    target.k_proj.bias.copy_(k_bias)
                if target.v_proj.bias is not None:
                    target.v_proj.bias.copy_(v_bias)
            target.out_proj.weight.copy_(source.out_proj.weight)
            if source.out_proj.bias is not None and target.out_proj.bias is not None:
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
            td_module.weight.requires_grad = module.weight.requires_grad
            if module.bias is not None:
                td_module.bias.requires_grad = module.bias.requires_grad
            td_module.train(module.training)
            return td_module

        if isinstance(module, nn.Conv2d):
            td_module = TDConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            with torch.no_grad():
                td_module.weight.copy_(module.weight)
                if module.bias is not None:
                    td_module.bias.copy_(module.bias)
            td_module.weight.requires_grad = module.weight.requires_grad
            if module.bias is not None:
                td_module.bias.requires_grad = module.bias.requires_grad
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
            td_module = TDGELU(approximate=getattr(module, "approximate", "none"))
            td_module.train(module.training)
            return td_module

        if isinstance(module, nn.Tanh):
            td_module = _TDTanh()
            td_module.train(module.training)
            return td_module

        if isinstance(module, nn.Softmax):
            dim = module.dim
            if not isinstance(dim, int):
                raise ValueError(
                    "TD softmax conversion requires nn.Softmax dim to be a literal int."
                )
            return TDSoftmax(dim=_td_softmax_dim(dim))

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

    @staticmethod
    def _insert_call_module_after(
        fx_model: fx.GraphModule,
        node: fx.Node,
        module: nn.Module,
        prefix: str,
        index: int,
        args: tuple[Any, ...],
    ) -> int:
        existing = set(dict(fx_model.named_modules()).keys())
        target = f"{prefix}_{index}"
        while target in existing:
            index += 1
            target = f"{prefix}_{index}"
        fx_model.add_submodule(target, module)
        with fx_model.graph.inserting_after(node):
            new_node = fx_model.graph.call_module(target, args=args)
        node.replace_all_uses_with(new_node)
        fx_model.graph.erase_node(node)
        return index + 1

    def _replace_functional_td_ops(self, fx_model: fx.GraphModule) -> None:
        softmax_index = 0
        matmul_index = 0
        gelu_index = 0
        tanh_index = 0
        for node in list(fx_model.graph.nodes):
            if node.op == "call_function" and node.target is F.gelu:
                approximate = self._get_literal_argument(node, "approximate", 1, "none")
                if approximate not in ("none", "tanh"):
                    raise ValueError(
                        "TD GELU conversion requires approximate to be 'none' or 'tanh'."
                    )
                gelu_index = self._insert_call_module_after(
                    fx_model,
                    node,
                    TDGELU(approximate=approximate),
                    "td_gelu",
                    gelu_index,
                    (self._get_tensor_argument(node, "input", 0),),
                )
                continue
            if node.op == "call_function" and node.target in (torch.tanh, F.tanh):
                tanh_index = self._insert_call_module_after(
                    fx_model,
                    node,
                    _TDTanh(),
                    "td_tanh",
                    tanh_index,
                    (self._get_tensor_argument(node, "input", 0),),
                )
                continue
            if node.op == "call_function" and node.target in (F.softmax, torch.softmax):
                dim = self._get_literal_argument(node, "dim", 1, None)
                if not isinstance(dim, int):
                    raise ValueError("TD softmax conversion requires literal int dim.")
                softmax_index = self._insert_call_module_after(
                    fx_model,
                    node,
                    TDSoftmax(dim=_td_softmax_dim(dim)),
                    "td_softmax",
                    softmax_index,
                    (self._get_tensor_argument(node, "input", 0),),
                )
                continue
            if node.op == "call_method" and node.target == "softmax":
                dim = self._get_literal_argument(node, "dim", 1, None)
                if not isinstance(dim, int):
                    raise ValueError(
                        "TD tensor.softmax conversion requires literal int dim."
                    )
                softmax_index = self._insert_call_module_after(
                    fx_model,
                    node,
                    TDSoftmax(dim=_td_softmax_dim(dim)),
                    "td_softmax",
                    softmax_index,
                    (node.args[0],),
                )
                continue
            if node.op == "call_function" and node.target in (
                torch.matmul,
                operator.matmul,
            ):
                matmul_index = self._insert_call_module_after(
                    fx_model,
                    node,
                    SNNMatrixOperator(),
                    "td_matmul",
                    matmul_index,
                    (
                        self._get_tensor_argument(node, "input", 0),
                        self._get_tensor_argument(node, "other", 1),
                    ),
                )
                continue
            if node.op == "call_method" and node.target == "matmul":
                if len(node.args) < 2:
                    raise ValueError(
                        "TD tensor.matmul conversion got malformed matmul node."
                    )
                matmul_index = self._insert_call_module_after(
                    fx_model,
                    node,
                    SNNMatrixOperator(),
                    "td_matmul",
                    matmul_index,
                    (node.args[0], node.args[1]),
                )
                continue

        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()
        fx_model.recompile()
