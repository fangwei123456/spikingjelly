from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch


@dataclass
class SNNDistributedConfig:
    r"""
    **API Language** - :ref:`中文 <SNNDistributedConfig-cn>` | :ref:`English <SNNDistributedConfig-en>`

    ----

    .. _SNNDistributedConfig-cn:

    * **中文**

    SNN 分布式训练配置。包含数据/张量/流水线并行设置。

    ----

    .. _SNNDistributedConfig-en:

    * **English**

    SNN distributed training configuration.
    """

    device_type: str = "cuda"
    mesh_shape: Optional[Tuple[int, ...]] = None
    device_mesh: Optional[Any] = None
    tp_mesh_dim: int = 0
    dp_mesh_dim: Optional[int] = None
    enable_data_parallel: bool = False
    enable_fsdp2: bool = False
    tensor_parallel_roots: Optional[Sequence[str]] = None
    tensor_parallel_plan: Optional[Mapping[str, Union[str, Any]]] = None
    auto_tensor_parallel: bool = True
    experimental_conv_tensor_parallel: bool = False
    conv_tensor_parallel_roots: Optional[Sequence[str]] = None
    experimental_spikformer_tensor_parallel: bool = False
    spikformer_tensor_parallel_roots: Optional[Sequence[str]] = None
    experimental_spikformer_patch_stem_tensor_parallel: bool = False
    spikformer_patch_stem_tensor_parallel_roots: Optional[Sequence[str]] = None
    broadcast_buffers: bool = False
    find_unused_parameters: bool = False
    static_graph: bool = False
    fsdp_shard_roots: Optional[Sequence[str]] = None
    fsdp_shard_module_root: bool = True
    fsdp_root_reshard_after_forward: Optional[bool] = False
    fsdp_param_dtype: Optional[torch.dtype] = None
    fsdp_reduce_dtype: Optional[torch.dtype] = None
    fsdp_output_dtype: Optional[torch.dtype] = None


SNNDistributedConfig.__init__.__doc__ = r"""Initialize SNN distributed training configuration.

.. admonition:: Chinese

    初始化 SNN 分布式训练配置，包括 mesh、数据并行、张量并行和 FSDP2 设置。

:param device_type: Device type used by distributed mesh construction.
:type device_type: str
:param mesh_shape: Optional logical mesh shape.
:type mesh_shape: tuple[int, ...] or None
:param device_mesh: Optional pre-built device mesh.
:param tp_mesh_dim: Tensor-parallel mesh dimension.
:type tp_mesh_dim: int
:param dp_mesh_dim: Data-parallel mesh dimension.
:type dp_mesh_dim: int or None
:param enable_data_parallel: Whether to wrap the model with data parallelism.
:type enable_data_parallel: bool
:param enable_fsdp2: Whether to apply FSDP2 sharding.
:type enable_fsdp2: bool
:param tensor_parallel_roots: Optional roots for linear tensor parallelism.
:type tensor_parallel_roots: sequence[str] or None
"""


@dataclass(frozen=True)
class EagerParallelPolicy:
    linear_tensor_parallel_roots: Tuple[str, ...] = ()
    conv_tensor_parallel_roots: Tuple[str, ...] = ()
    spikformer_tensor_parallel_roots: Tuple[str, ...] = ()
    spikformer_patch_stem_tensor_parallel_roots: Tuple[str, ...] = ()
    fsdp_shard_roots: Tuple[str, ...] = ()
    fsdp2_tp_shard_roots: Tuple[str, ...] = ()
    fsdp_shard_module_root: bool = True
    fsdp2_tp_shard_module_root: bool = False
