from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch.nn as nn

from spikingjelly.activation_based import base


LinearLike = (nn.Linear,)


@dataclass
class SNNDistributedAnalysis:
    r"""
    **API Language** - :ref:`中文 <SNNDistributedAnalysis-cn>` | :ref:`English <SNNDistributedAnalysis-en>`

    ----

    .. _SNNDistributedAnalysis-cn:

    * **中文**

    SNN 分布式训练分析器。分析模型结构并推荐并行策略。

    ----

    .. _SNNDistributedAnalysis-en:

    * **English**

    SNN distributed training analyzer.
    """

    memory_module_names: Tuple[str, ...]
    tensor_parallel_candidate_names: Tuple[str, ...]
    unsupported_tensor_parallel_names: Tuple[str, ...]
    notes: Tuple[str, ...]
    tensor_parallel_roots: Optional[Tuple[str, ...]] = None


SNNDistributedAnalysis.__init__.__doc__ = r"""Initialize distributed capability analysis results.

.. admonition:: Chinese

    初始化 SNN 分布式能力分析结果，包括状态模块、张量并行候选模块和提示信息。

:param memory_module_names: Names of stateful memory modules.
:type memory_module_names: tuple[str, ...]
:param tensor_parallel_candidate_names: Names of modules that can use tensor parallelism.
:type tensor_parallel_candidate_names: tuple[str, ...]
:param unsupported_tensor_parallel_names: Names seen under tensor-parallel roots but not supported.
:type unsupported_tensor_parallel_names: tuple[str, ...]
:param notes: Human-readable analysis notes.
:type notes: tuple[str, ...]
:param tensor_parallel_roots: Roots used by the analysis.
:type tensor_parallel_roots: tuple[str, ...] or None
"""


def _iter_named_modules_under_roots(
    module: nn.Module,
    roots: Optional[Sequence[str]] = None,
) -> Iterable[Tuple[str, nn.Module]]:
    if not roots:
        for name, child in module.named_modules():
            if name:
                yield name, child
        return

    named_children = dict(module.named_modules())
    seen = set()
    for root in roots:
        if not root:
            raise ValueError(
                "tensor_parallel_roots entries must be non-empty module paths; "
                "pass None or omit the argument to scan the full model."
            )
        if root not in named_children:
            raise KeyError(
                f"tensor_parallel_roots contains unknown module path '{root}'."
            )

        root_module = named_children[root]
        for sub_name, child in root_module.named_modules():
            full_name = root if not sub_name else f"{root}.{sub_name}"
            if full_name in seen:
                continue
            seen.add(full_name)
            yield full_name, child


def analyze_snn_distributed_capability(
    module: nn.Module,
    tensor_parallel_roots: Optional[Sequence[str]] = None,
) -> SNNDistributedAnalysis:
    r"""
    **API Language** - :ref:`中文 <analyze_snn_distributed_capability-cn>` | :ref:`English <analyze_snn_distributed_capability-en>`

    ----

    .. _analyze_snn_distributed_capability-cn:

    * **中文**

    分析 SNN 模型的分布式训练能力。

    ----

    .. _analyze_snn_distributed_capability-en:

    * **English**

    Analyze SNN distributed capability.
    """
    memory_modules: List[str] = []
    tensor_parallel_candidates: List[str] = []
    unsupported_tp: List[str] = []
    notes: List[str] = []

    for name, child in module.named_modules():
        if not name:
            continue
        if isinstance(child, base.MemoryModule):
            memory_modules.append(name)

    for name, child in _iter_named_modules_under_roots(module, tensor_parallel_roots):
        if isinstance(child, LinearLike):
            tensor_parallel_candidates.append(name)
        elif isinstance(
            child,
            (nn.Conv1d, nn.Conv2d, nn.Conv3d),
        ):
            unsupported_tp.append(name)

    if memory_modules:
        notes.append(
            "Stateful neuron modules remain local/replicated in this first DTensor-ready layer."
        )
    if unsupported_tp:
        notes.append(
            "Conv tensor parallel is not enabled in this first implementation; only Linear-like modules "
            "are auto-parallelized."
        )
    if not tensor_parallel_candidates:
        notes.append(
            "No Linear-like tensor-parallel candidates were found under the selected roots."
        )

    return SNNDistributedAnalysis(
        memory_module_names=tuple(memory_modules),
        tensor_parallel_candidate_names=tuple(tensor_parallel_candidates),
        unsupported_tensor_parallel_names=tuple(unsupported_tp),
        notes=tuple(notes),
        tensor_parallel_roots=(
            tuple(tensor_parallel_roots) if tensor_parallel_roots is not None else None
        ),
    )
