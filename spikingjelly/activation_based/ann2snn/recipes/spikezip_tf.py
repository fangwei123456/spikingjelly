from __future__ import annotations

import operator
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fx

from spikingjelly.activation_based.ann2snn.operators import (
    SNNMatrixOperator,
    TDGELU,
    TDModule,
    TDSoftmax,
)
from spikingjelly.activation_based.ann2snn.recipes.base import ConversionRecipe
from spikingjelly.activation_based.ann2snn.recipes.step_mode_adapters import (
    _TRANSFORMER_SAFE_MODULE_TYPES,
    adapt_step_mode_graph,
)
from spikingjelly.activation_based.ann2snn.recipes.transformer_spike_equivalent import (
    TransformerSpikeEquivalentRecipe,
)

if TYPE_CHECKING:
    from spikingjelly.activation_based.ann2snn.converter import Converter


__all__ = ["SpikeZIPTFRecipe"]


class _TDTanh(TDModule):
    def ann_forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self._td_sequence_forward((x_seq,), torch.tanh)


class SpikeZIPTFRecipe(ConversionRecipe):
    def __init__(
        self,
        time_steps: int = 8,
        supported_model: str = "bert",
        strict: bool = True,
    ) -> None:
        r"""
        **API Language** -
        :ref:`中文 <SpikeZIPTFRecipe.__init__-cn>` |
        :ref:`English <SpikeZIPTFRecipe.__init__-en>`

        ----

        .. _SpikeZIPTFRecipe.__init__-cn:

        * **中文**

        SpikeZIP-TF 风格的窄版 Transformer 语言模型转换 recipe。该 recipe
        面向 embedding 之后的 BERT-style encoder/classifier wrapper，将
        Linear、LayerNorm、GELU、Softmax 和矩阵乘法替换为累积-差分 TD 算子。
        产物遵循 SpikingJelly ``step_mode`` 语义：``step_mode="s"`` 时用户
        显式逐步调用，``step_mode="m"`` 时输入第 0 维是时间维，readout 由用户
        显式完成。

        该实现只提供 SpikeZIP-TF-style TD 等价窄版，不实现完整论文中的
        activation quantization、ST-BIF+、SESA、Spike-Softmax 或
        Spike-LayerNorm，也不支持 autoregressive generation、KV cache 或
        decoder/cross-attention。

        :param time_steps: 转换后示例/校验使用的时间步数，会记录到转换产物。
        :type time_steps: int
        :param supported_model: 当前只支持 ``"bert"``。
        :type supported_model: str
        :param strict: 是否启用严格边界检查。当前必须为 ``True``。
        :type strict: bool
        :raises ValueError: 若参数非法或请求 unsupported model / non-strict
            fallback。

        ----

        .. _SpikeZIPTFRecipe.__init__-en:

        * **English**

        Narrow SpikeZIP-TF-style Transformer language-model conversion recipe.
        This recipe targets a BERT-style encoder/classifier wrapper after the
        embedding layer, and replaces Linear, LayerNorm, GELU, Softmax and
        matrix multiplication with cumulative-difference TD operators. The
        converted product follows SpikingJelly ``step_mode`` semantics:
        ``step_mode="s"`` uses an explicit user loop, ``step_mode="m"`` consumes
        inputs whose first dimension is time, and readout is explicit.

        This implementation is only a narrow SpikeZIP-TF-style TD-equivalent
        path. It does not implement the full paper's activation quantization,
        ST-BIF+, SESA, Spike-Softmax or Spike-LayerNorm, and it does not support
        autoregressive generation, KV cache, decoder attention or cross
        attention.

        :param time_steps: Number of timesteps used by examples/checks and
            recorded on the converted module.
        :type time_steps: int
        :param supported_model: Currently only ``"bert"`` is supported.
        :type supported_model: str
        :param strict: Whether to use strict boundary checks. It must currently
            be ``True``.
        :type strict: bool
        :raises ValueError: If parameters are invalid or unsupported model /
            non-strict fallback is requested.
        """
        self.time_steps = time_steps
        self.supported_model = supported_model
        self.strict = strict
        self._td_recipe = TransformerSpikeEquivalentRecipe()

    def validate(self, converter: "Converter") -> None:
        if (
            not isinstance(self.time_steps, int)
            or isinstance(self.time_steps, bool)
            or self.time_steps <= 0
        ):
            raise ValueError("time_steps must be a positive integer.")
        if not isinstance(self.supported_model, str):
            raise ValueError("supported_model must be str.")
        if self.supported_model.lower() != "bert":
            raise ValueError(
                "SpikeZIPTFRecipe currently supports only supported_model='bert'."
            )
        if not isinstance(self.strict, bool):
            raise ValueError("strict must be bool.")
        if not self.strict:
            raise ValueError(
                "SpikeZIPTFRecipe currently requires strict=True; unsupported "
                "language-model branches must fail explicitly."
            )

    def before_trace(self, converter: "Converter", ann: nn.Module) -> nn.Module:
        ann.eval()
        return ann

    def replace(
        self, converter: "Converter", fx_model: fx.GraphModule
    ) -> fx.GraphModule:
        modules = dict(fx_model.named_modules())
        for node in list(fx_model.graph.nodes):
            if node.op != "call_module":
                continue
            if not isinstance(node.target, str) or node.target not in modules:
                continue
            module = modules[node.target]
            replacement = self._make_td_operator(module)
            if replacement is None:
                continue
            self._replace_submodule(fx_model, node.target, replacement)
            modules[node.target] = replacement

        self._replace_functional_td_ops(fx_model)
        return adapt_step_mode_graph(
            fx_model,
            context="SpikeZIPTFRecipe step-mode backend",
            safe_module_types=_TRANSFORMER_SAFE_MODULE_TYPES,
            safe_call_functions=(
                torch.div,
                operator.truediv,
            ),
        )

    def finalize(self, converter: "Converter", fx_model: fx.GraphModule) -> nn.Module:
        object.__setattr__(fx_model, "time_steps", self.time_steps)
        object.__setattr__(fx_model, "ann2snn_recipe", "spikezip_tf")
        object.__setattr__(fx_model, "ann2snn_supported_model", "bert")
        return fx_model

    @staticmethod
    def _replace_submodule(
        fx_model: fx.GraphModule, target: str, module: nn.Module
    ) -> None:
        parent_name, _, child_name = target.rpartition(".")
        parent = fx_model.get_submodule(parent_name) if parent_name else fx_model
        setattr(parent, child_name, module)

    def _make_td_operator(self, module: nn.Module) -> Optional[nn.Module]:
        if isinstance(module, TDModule):
            return None
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.GELU)):
            return self._td_recipe._make_td_operator(module)
        if isinstance(module, nn.Tanh):
            td_module = _TDTanh()
            td_module.train(module.training)
            return td_module
        if isinstance(module, nn.Softmax):
            dim = module.dim
            if not isinstance(dim, int):
                raise ValueError(
                    "SpikeZIPTFRecipe requires nn.Softmax dim to be a literal int."
                )
            return TDSoftmax(dim=dim)
        return None

    @staticmethod
    def _literal_arg(
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
        for node in list(fx_model.graph.nodes):
            if node.op == "call_function" and node.target is F.gelu:
                approximate = self._literal_arg(node, "approximate", 1, "none")
                if approximate not in ("none", "tanh"):
                    raise ValueError(
                        "SpikeZIPTFRecipe requires GELU approximate to be "
                        "'none' or 'tanh'."
                    )
                gelu_index = self._insert_call_module_after(
                    fx_model,
                    node,
                    TDGELU(approximate=approximate),
                    "spikezip_tf_gelu",
                    gelu_index,
                    (node.args[0],),
                )
                continue
            if node.op == "call_function" and node.target in (
                F.softmax,
                torch.softmax,
            ):
                dim = self._literal_arg(node, "dim", 1, None)
                if not isinstance(dim, int):
                    raise ValueError(
                        "SpikeZIPTFRecipe requires softmax dim to be a literal int."
                    )
                softmax_index = self._insert_call_module_after(
                    fx_model,
                    node,
                    TDSoftmax(dim=dim),
                    "spikezip_tf_softmax",
                    softmax_index,
                    (node.args[0],),
                )
                continue
            if node.op == "call_method" and node.target == "softmax":
                dim = self._literal_arg(node, "dim", 1, None)
                if not isinstance(dim, int):
                    raise ValueError(
                        "SpikeZIPTFRecipe requires tensor.softmax dim to be "
                        "a literal int."
                    )
                softmax_index = self._insert_call_module_after(
                    fx_model,
                    node,
                    TDSoftmax(dim=dim),
                    "spikezip_tf_softmax",
                    softmax_index,
                    (node.args[0],),
                )
                continue
            if node.op == "call_function" and node.target in (
                torch.matmul,
                operator.matmul,
            ):
                if len(node.args) < 2:
                    raise ValueError("SpikeZIPTFRecipe got malformed matmul node.")
                matmul_index = self._insert_call_module_after(
                    fx_model,
                    node,
                    SNNMatrixOperator(),
                    "spikezip_tf_matmul",
                    matmul_index,
                    (node.args[0], node.args[1]),
                )
                continue
            if node.op == "call_method" and node.target == "matmul":
                if len(node.args) < 2:
                    raise ValueError(
                        "SpikeZIPTFRecipe got malformed tensor.matmul node."
                    )
                matmul_index = self._insert_call_module_after(
                    fx_model,
                    node,
                    SNNMatrixOperator(),
                    "spikezip_tf_matmul",
                    matmul_index,
                    (node.args[0], node.args[1]),
                )

        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()
        fx_model.recompile()
