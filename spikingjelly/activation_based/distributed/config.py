from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple, Union

import torch

if TYPE_CHECKING:
    from torch.distributed._tensor import DeviceMesh


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
    device_mesh: Optional["DeviceMesh"] = None
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
    fsdp_root_reshard_after_forward: bool = False
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
:type device_mesh: torch.distributed._tensor.DeviceMesh or None
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
:param tensor_parallel_plan: Explicit tensor-parallel plan mapping module names to styles.
:type tensor_parallel_plan: mapping[str, str or Any] or None
:param auto_tensor_parallel: Whether to build a linear tensor-parallel plan automatically.
:type auto_tensor_parallel: bool
:param experimental_conv_tensor_parallel: Whether to enable experimental convolution TP.
:type experimental_conv_tensor_parallel: bool
:param conv_tensor_parallel_roots: Optional roots for convolution TP.
:type conv_tensor_parallel_roots: sequence[str] or None
:param experimental_spikformer_tensor_parallel: Whether to enable experimental Spikformer block TP.
:type experimental_spikformer_tensor_parallel: bool
:param spikformer_tensor_parallel_roots: Optional Spikformer block TP roots.
:type spikformer_tensor_parallel_roots: sequence[str] or None
:param experimental_spikformer_patch_stem_tensor_parallel: Whether to enable experimental Spikformer patch-stem TP.
:type experimental_spikformer_patch_stem_tensor_parallel: bool
:param spikformer_patch_stem_tensor_parallel_roots: Optional Spikformer patch-stem TP roots.
:type spikformer_patch_stem_tensor_parallel_roots: sequence[str] or None
:param broadcast_buffers: Whether data-parallel wrappers broadcast buffers.
:type broadcast_buffers: bool
:param find_unused_parameters: Whether data-parallel wrappers find unused parameters.
:type find_unused_parameters: bool
:param static_graph: Whether data-parallel wrappers assume a static graph.
:type static_graph: bool
:param fsdp_shard_roots: Optional roots for FSDP2 sharding.
:type fsdp_shard_roots: sequence[str] or None
:param fsdp_shard_module_root: Whether to shard the root module with FSDP2.
:type fsdp_shard_module_root: bool
:param fsdp_root_reshard_after_forward: Whether the FSDP2 root reshards after forward.
:type fsdp_root_reshard_after_forward: bool
:param fsdp_param_dtype: Optional FSDP2 parameter dtype policy.
:type fsdp_param_dtype: torch.dtype or None
:param fsdp_reduce_dtype: Optional FSDP2 reduction dtype policy.
:type fsdp_reduce_dtype: torch.dtype or None
:param fsdp_output_dtype: Optional FSDP2 output dtype policy.
:type fsdp_output_dtype: torch.dtype or None
"""


@dataclass(frozen=True)
class EagerParallelPolicy:
    linear_tensor_parallel_roots: Tuple[str, ...] = ()
    conv_tensor_parallel_roots: Tuple[str, ...] = ()
    spikformer_tensor_parallel_roots: Tuple[str, ...] = ()
    spikformer_patch_stem_tensor_parallel_roots: Tuple[str, ...] = ()
    fsdp_shard_roots: Tuple[str, ...] = ()
    fsdp2_tp_shard_roots: Optional[Tuple[str, ...]] = None
    fsdp_shard_module_root: bool = True
    fsdp2_tp_shard_module_root: bool = False
