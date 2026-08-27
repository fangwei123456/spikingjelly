from __future__ import annotations

import functools
import math
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from spikingjelly.activation_based import quantize
from spikingjelly.activation_based.ann2snn.recipes.qwen2 import (
    Qwen2SNNCalibration,
    Qwen2SNNConfig,
)
from spikingjelly.activation_based.distributed.llm.temporal import (
    pack_time_batch,
    unpack_time_batch,
)
from spikingjelly.activation_based.distributed.llm import (
    ModelBuilder,
    ModelConfig,
    SGLangExportStage,
    export_sglang_artifact,
)
from ._causal_lm import forward_step

if TYPE_CHECKING:
    from collections.abc import Callable

    from megatron.core.transformer import MegatronModule, TransformerConfig


@dataclass(frozen=True, kw_only=True)
class Qwen2Config(ModelConfig):
    builder: ClassVar[str] = "benchmark.snn_llm.qwen2.Qwen2Builder"
    source_path: Path
    calibration_path: Path


Qwen2Config.__init__.__doc__ = r"""Configure Qwen2 QCFS-SG fine-tuning.

**API Language** - 中文 | English

**中文：** 组合 MCore Transformer、Qwen2 模型事实、Hugging Face 权重目录和
SpikingJelly calibration 文件。builder 从这些有类型字段构建当前 TP/PP shard。

**English:** Combine the MCore Transformer, Qwen2 model facts, Hugging Face
weights directory, and SpikingJelly calibration file. The builder constructs the
current TP/PP shard from these typed fields.

:param transformer: MCore Transformer 配置。 / MCore Transformer configuration.
:type transformer: megatron.core.transformer.TransformerConfig
:param vocab_size: 词表大小。 / Vocabulary size.
:type vocab_size: int
:param max_sequence_length: 最大上下文长度。 / Maximum context length.
:type max_sequence_length: int
:param time_steps: SNN 时间步。 / SNN time steps.
:type time_steps: int
:param share_embeddings_and_output_weights: 是否共享 embedding 与输出权重。 / Whether to tie embedding and output weights.
:type share_embeddings_and_output_weights: bool
:param position_embedding_type: 位置编码类型。 / Position-embedding type.
:type position_embedding_type: str
:param source_path: Hugging Face 权重目录。 / Hugging Face weights directory.
:type source_path: pathlib.Path
:param calibration_path: calibration 文件。 / Calibration file.
:type calibration_path: pathlib.Path
:raises ValueError: 模型公共字段无效。 / If common model fields are invalid.
"""


def _exact_sequence(value: torch.Tensor, time_steps: int) -> torch.Tensor:
    zeros = torch.zeros_like(value).unsqueeze(0).expand(time_steps - 1, *value.shape)
    return torch.cat((value.unsqueeze(0), zeros), dim=0)


def _qcfs(value: torch.Tensor, scale: torch.Tensor, time_steps: int) -> torch.Tensor:
    scale = scale.to(value)
    shape = [1] * value.dim()
    shape[-1] = scale.numel()
    scale = scale.reshape(shape)
    positive = quantize.multi_level_spike_count(torch.relu(value) / scale, time_steps)
    negative = quantize.multi_level_spike_count(torch.relu(-value) / scale, time_steps)
    return (positive - negative) * scale


def _encode_envelope(
    hidden: torch.Tensor,
    scale: torch.Tensor,
    time_steps: int,
    checkpoint_enabled: bool = False,
) -> torch.Tensor:
    if checkpoint_enabled and torch.is_grad_enabled():
        return torch_checkpoint(
            lambda value: _encode_envelope(value, scale, time_steps),
            hidden,
            use_reentrant=True,
            preserve_rng_state=False,
        )
    shape = hidden.shape
    flat = hidden.flatten(2)
    dense = unpack_time_batch(flat, time_steps).sum(0)
    encoded = _exact_sequence(_qcfs(dense, scale, time_steps), time_steps)
    return pack_time_batch(encoded).reshape(shape)


def _encode_input(
    hidden: torch.Tensor, scale: torch.Tensor, time_steps: int
) -> torch.Tensor:
    sequence = unpack_time_batch(hidden, time_steps)
    dense = sequence.mean(0)
    encoded = _exact_sequence(_qcfs(dense, scale, time_steps), time_steps)
    return pack_time_batch(encoded)


class _InputQCFSRMSNorm(nn.RMSNorm):
    def __init__(
        self,
        *,
        config: "TransformerConfig",
        hidden_size: int,
        eps: float,
        scale: torch.Tensor,
        time_steps: int,
        checkpoint_enabled: bool,
    ) -> None:
        del config
        super().__init__(hidden_size, eps=eps)
        self.time_steps = time_steps
        self.register_buffer("qcfs_scale", scale.detach().clone())
        self.checkpoint_enabled = checkpoint_enabled

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_enabled and torch.is_grad_enabled():
            hidden = torch_checkpoint(
                lambda value: _encode_input(value, self.qcfs_scale, self.time_steps),
                hidden,
                use_reentrant=True,
                preserve_rng_state=False,
            )
        else:
            hidden = _encode_input(hidden, self.qcfs_scale, self.time_steps)
        return super().forward(hidden)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
        from megatron.core.transformer.utils import (
            make_sharded_tensors_for_checkpoint,
        )

        # The input scale exists only on global layer 0, so it has no PP layer axis.
        return {
            **make_sharded_tensors_for_checkpoint(
                {"weight": self.weight}, prefix, {}, sharded_offsets
            ),
            **make_sharded_tensors_for_checkpoint(
                {"qcfs_scale": self.qcfs_scale}, prefix, {}, ()
            ),
        }


def _rms_norm(
    *, config: "TransformerConfig", hidden_size: int, eps: float
) -> nn.RMSNorm:
    del config
    return nn.RMSNorm(hidden_size, eps=eps)


def _attention_builder(base: type) -> type:
    from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

    class QCFSAttention(base):
        def __init__(
            self,
            *args: Any,
            query_scale: torch.Tensor,
            key_scale: torch.Tensor,
            value_scale: torch.Tensor,
            time_steps: int,
            checkpoint_enabled: bool,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.time_steps = time_steps
            self.register_buffer("query_scale", query_scale.detach().clone())
            self.register_buffer("key_scale", key_scale.detach().clone())
            self.register_buffer("value_scale", value_scale.detach().clone())
            self.checkpoint_enabled = checkpoint_enabled

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            query = _encode_envelope(
                query, self.query_scale, self.time_steps, self.checkpoint_enabled
            )
            key = _encode_envelope(
                key, self.key_scale, self.time_steps, self.checkpoint_enabled
            )
            value = _encode_envelope(
                value, self.value_scale, self.time_steps, self.checkpoint_enabled
            )
            return super().forward(query, key, value, *args, **kwargs)

        def sharded_state_dict(
            self, prefix: str = "", sharded_offsets=(), metadata=None
        ):
            state = self.state_dict(prefix="", keep_vars=True)
            return make_sharded_tensors_for_checkpoint(
                state,
                prefix,
                {"query_scale": 0, "key_scale": 0, "value_scale": 0},
                sharded_offsets,
                tp_group=getattr(self, "tp_group", getattr(self, "_tp_group", None)),
                dp_cp_group=metadata["dp_cp_group"],
            )

    QCFSAttention.__name__ = f"QCFS{base.__name__}"
    return QCFSAttention


def _row_linear_builder(base: type) -> type:
    from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

    class QCFSRowLinear(base):
        def __init__(
            self,
            *args: Any,
            scale: torch.Tensor,
            time_steps: int,
            checkpoint_enabled: bool,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.time_steps = time_steps
            self.register_buffer("qcfs_scale", scale.detach().clone())
            self.checkpoint_enabled = checkpoint_enabled

        def forward(self, hidden: torch.Tensor, *args: Any, **kwargs: Any):
            hidden = _encode_envelope(
                hidden, self.qcfs_scale, self.time_steps, self.checkpoint_enabled
            )
            return super().forward(hidden, *args, **kwargs)

        def sharded_state_dict(
            self, prefix: str = "", sharded_offsets=(), metadata=None
        ):
            state = self.state_dict(prefix="", keep_vars=True)
            return make_sharded_tensors_for_checkpoint(
                state,
                prefix,
                {"weight": 1, "qcfs_scale": 0},
                sharded_offsets,
                tp_group=getattr(self, "tp_group", getattr(self, "_tp_group", None)),
                dp_cp_group=metadata["dp_cp_group"],
            )

    QCFSRowLinear.__name__ = f"QCFS{base.__name__}"
    return QCFSRowLinear


def _local_chunk(value: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    if value.numel() % world_size:
        raise ValueError("Calibration channels must be divisible by TP size.")
    return value.flatten().chunk(world_size)[rank].contiguous()


def _fuse_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_attention_heads: int,
    num_query_groups: int,
) -> torch.Tensor:
    head_dim = query.shape[0] // num_attention_heads
    heads_per_group = num_attention_heads // num_query_groups
    tail = query.shape[1:]
    query = query.reshape(num_query_groups, heads_per_group * head_dim, *tail)
    key = key.reshape(num_query_groups, head_dim, *tail)
    value = value.reshape(num_query_groups, head_dim, *tail)
    return torch.cat((query, key, value), dim=1).flatten(0, 1)


def _copy(target: torch.Tensor, source: torch.Tensor) -> None:
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def _qwen2_rotary_base(source_config: Any) -> float:
    rope = source_config.rope_parameters
    if rope.get("rope_type") != "default":
        raise ValueError("Qwen-SNN currently supports only default RoPE.")
    return float(rope["rope_theta"])


def _qwen2_architecture(source_config: Any) -> dict[str, Any]:
    return {
        "model_type": source_config.model_type,
        "vocab_size": int(source_config.vocab_size),
        "max_position_embeddings": int(source_config.max_position_embeddings),
        "num_hidden_layers": int(source_config.num_hidden_layers),
        "hidden_size": int(source_config.hidden_size),
        "num_attention_heads": int(source_config.num_attention_heads),
        "num_key_value_heads": int(source_config.num_key_value_heads),
        "intermediate_size": int(source_config.intermediate_size),
        "rms_norm_eps": float(source_config.rms_norm_eps),
        "tie_word_embeddings": bool(source_config.tie_word_embeddings),
        "rope_parameters": dict(source_config.rope_parameters),
    }


def _validate_qwen2_calibration(
    calibration: Qwen2SNNCalibration, source_config: Any
) -> None:
    head_dim = source_config.hidden_size // source_config.num_attention_heads
    expected_sizes = {
        "query": source_config.hidden_size,
        "key": source_config.num_key_value_heads * head_dim,
        "value": source_config.num_key_value_heads * head_dim,
        "mlp": source_config.intermediate_size,
    }
    scales = [("input", calibration.input_scale, source_config.hidden_size)]
    for index, layer in enumerate(calibration.layer_scales):
        if set(layer) != set(expected_sizes):
            raise ValueError(f"Calibration layer {index} has invalid scale names.")
        scales.extend(
            (f"layer {index} {name}", layer[name], size)
            for name, size in expected_sizes.items()
        )
    for name, scale, size in scales:
        if scale.numel() != size or not torch.all(torch.isfinite(scale) & (scale > 0)):
            raise ValueError(f"Calibration {name} scale must contain {size} positives.")


@torch.no_grad()
def load_hf_qwen2_weights(model: "MegatronModule", source: nn.Module) -> None:
    r"""Import Hugging Face Qwen2 weights into the current MCore shards.

    **API Language** - 中文 | English

    **中文：** 确定性地将 Hugging Face Qwen2 权重导入当前 PP/TP shard。该函数只
    支持结构匹配的 Qwen2 causal LM，并在 shape 不一致时直接失败。

    :param model: 当前 PP stage 的 MCore GPTModel。
    :type model: megatron.core.transformer.MegatronModule
    :param source: Hugging Face Qwen2 causal LM。
    :type source: torch.nn.Module

    **English:** Deterministically import a matching Hugging Face Qwen2 causal LM
    into the current MCore PP/TP shard. Shape mismatches fail immediately.

    :param model: MCore GPTModel for the current PP stage.
    :type model: megatron.core.transformer.MegatronModule
    :param source: Hugging Face Qwen2 causal LM.
    :type source: torch.nn.Module
    """
    from megatron.core import parallel_state

    rank = parallel_state.get_tensor_model_parallel_rank()
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    source_model = source.model
    if model.pre_process:
        _copy(
            model.embedding.word_embeddings.weight,
            source_model.embed_tokens.weight.chunk(world_size)[rank],
        )
    for layer in model.decoder.layers:
        source_layer = source_model.layers[layer.layer_number - 1]
        attention = layer.self_attention
        source_attention = source_layer.self_attn
        fused_weight = _fuse_qkv(
            source_attention.q_proj.weight,
            source_attention.k_proj.weight,
            source_attention.v_proj.weight,
            source.config.num_attention_heads,
            source.config.num_key_value_heads,
        )
        _copy(attention.linear_qkv.weight, fused_weight.chunk(world_size)[rank])
        query_bias = source_attention.q_proj.bias
        key_bias = source_attention.k_proj.bias
        value_bias = source_attention.v_proj.bias
        if attention.linear_qkv.bias is None:
            if query_bias is not None or key_bias is not None or value_bias is not None:
                raise ValueError("The source and target QKV bias settings must match.")
        else:
            if query_bias is None or key_bias is None or value_bias is None:
                raise ValueError("The source and target QKV bias settings must match.")
            fused_bias = _fuse_qkv(
                query_bias,
                key_bias,
                value_bias,
                source.config.num_attention_heads,
                source.config.num_key_value_heads,
            )
            _copy(attention.linear_qkv.bias, fused_bias.chunk(world_size)[rank])
        _copy(
            attention.linear_proj.weight,
            source_attention.o_proj.weight.chunk(world_size, dim=1)[rank],
        )
        _copy(layer.input_layernorm.weight, source_layer.input_layernorm.weight)
        _copy(
            layer.pre_mlp_layernorm.weight,
            source_layer.post_attention_layernorm.weight,
        )
        gate = source_layer.mlp.gate_proj.weight.chunk(world_size)[rank]
        up = source_layer.mlp.up_proj.weight.chunk(world_size)[rank]
        _copy(layer.mlp.linear_fc1.weight, torch.cat((gate, up)))
        _copy(
            layer.mlp.linear_fc2.weight,
            source_layer.mlp.down_proj.weight.chunk(world_size, dim=1)[rank],
        )
    if model.post_process:
        _copy(model.decoder.final_layernorm.weight, source_model.norm.weight)
        if model.output_layer.weight is not None:
            output = source.lm_head.weight.chunk(world_size)[rank]
            _copy(model.output_layer.weight, output)


def model_provider(
    source_path: Path | None,
    source_config: Any,
    calibration: Qwen2SNNCalibration,
    conversion: Qwen2SNNConfig,
    config: Qwen2Config,
    memopt_level: int,
    memopt_checkpoint_budget: str,
    pre_process: bool,
    post_process: bool,
) -> "MegatronModule":
    r"""Build and initialize one MCore Qwen-SNN pipeline stage.

    **API Language** - 中文 | English

    **中文：** 使用现有 Qwen 校准产物构建 ``qcfs_sg`` MCore ModuleSpec，并从
    Hugging Face checkpoint 单向导入当前 PP/TP shard。FP8 配置必须使用 TE，
    不做 BF16 fallback。

    :param source_path: 首次导入的 Hugging Face Qwen2 checkpoint 目录；从 MCore
        checkpoint 加载时为 ``None``。
    :type source_path: pathlib.Path or None
    :param source_config: Hugging Face Qwen2 config。
    :type source_config: Any
    :param calibration: 已有 Qwen2 SNN 校准产物。
    :type calibration: Qwen2SNNCalibration
    :param conversion: ``qcfs_sg`` 转换配置。
    :type conversion: Qwen2SNNConfig
    :param config: 完整 Qwen2 模型配置。
    :type config: Qwen2Config
    :param memopt_level: SpikingJelly memopt 级别。
    :type memopt_level: int
    :param memopt_checkpoint_budget: checkpoint 数量预设。
    :type memopt_checkpoint_budget: str
    :param pre_process: 当前 stage 是否拥有 embedding。
    :type pre_process: bool
    :param post_process: 当前 stage 是否拥有 LM head。
    :type post_process: bool
    :return: 已导入 HF 权重的当前 MCore stage。
    :rtype: megatron.core.transformer.MegatronModule
    :raises ValueError: 校准、转换或模型结构不一致。
    :raises ImportError: FP8 配置缺少 Transformer Engine。

    **English:** Build a ``qcfs_sg`` MCore ModuleSpec from the existing Qwen
    calibration and import the current PP/TP shard from a Hugging Face checkpoint.
    FP8 requires TE and never falls back to BF16.

    :param source_path: Hugging Face Qwen2 checkpoint directory for initial import,
        or ``None`` when loading an MCore checkpoint.
    :type source_path: pathlib.Path or None
    :param source_config: Hugging Face Qwen2 configuration.
    :type source_config: Any
    :param calibration: Existing Qwen2 SNN calibration artifact.
    :type calibration: Qwen2SNNCalibration
    :param conversion: ``qcfs_sg`` conversion configuration.
    :type conversion: Qwen2SNNConfig
    :param config: Complete Qwen2 model configuration.
    :type config: Qwen2Config
    :param memopt_level: SpikingJelly memopt level.
    :type memopt_level: int
    :param memopt_checkpoint_budget: Checkpoint-count preset.
    :type memopt_checkpoint_budget: str
    :param pre_process: Whether this stage owns the embedding.
    :type pre_process: bool
    :param post_process: Whether this stage owns the LM head.
    :type post_process: bool
    :return: Current MCore stage initialized from HF weights.
    :rtype: megatron.core.transformer.MegatronModule
    :raises ValueError: If calibration, conversion, or model structures disagree.
    :raises ImportError: If FP8 is configured without Transformer Engine.
    """
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.layers import RowParallelLinear
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.transformer_block import (
        TransformerBlockSubmodules,
        get_num_layers_to_build,
    )
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    if source_config.model_type != "qwen2":
        raise ValueError("Qwen-SNN requires a Hugging Face Qwen2 configuration.")
    if conversion.time_steps != calibration.time_steps:
        raise ValueError("Conversion and calibration time_steps must match.")
    if (
        config.vocab_size != int(source_config.vocab_size)
        or config.max_sequence_length != int(source_config.max_position_embeddings)
        or config.time_steps != conversion.time_steps
        or config.share_embeddings_and_output_weights
        != bool(source_config.tie_word_embeddings)
        or config.position_embedding_type != "rope"
    ):
        raise ValueError("Qwen2Config does not match Qwen2 or conversion.")
    transformer_config = config.transformer
    if len(calibration.layer_scales) != transformer_config.num_layers:
        raise ValueError("Calibration layer count must match TransformerConfig.")
    _validate_qwen2_calibration(calibration, source_config)
    if transformer_config.normalization != "RMSNorm":
        raise ValueError("Qwen-SNN requires MCore RMSNorm.")
    if not transformer_config.gated_linear_unit:
        raise ValueError("Qwen-SNN requires MCore gated_linear_unit=True.")
    if source_config.use_sliding_window:
        raise ValueError("Qwen-SNN does not support sliding-window attention.")
    expected = {
        "num_layers": source_config.num_hidden_layers,
        "hidden_size": source_config.hidden_size,
        "num_attention_heads": source_config.num_attention_heads,
        "num_query_groups": source_config.num_key_value_heads,
        "kv_channels": source_config.hidden_size // source_config.num_attention_heads,
        "ffn_hidden_size": source_config.intermediate_size,
        "layernorm_epsilon": source_config.rms_norm_eps,
    }
    mismatched = [
        name
        for name, value in expected.items()
        if getattr(transformer_config, name) != value
    ]
    if mismatched:
        raise ValueError(
            "TransformerConfig does not match Qwen2 fields: " + ", ".join(mismatched)
        )

    use_te = (
        transformer_config.fp8 is not None
        or transformer_config.context_parallel_size > 1
    )
    if use_te:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TEDotProductAttention,
            TERowParallelLinear,
        )

        attention_base = TEDotProductAttention
        column_base = TEColumnParallelLinear
        row_base = TERowParallelLinear
    else:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        attention_base = DotProductAttention
        column_base = ColumnParallelLinear
        row_base = RowParallelLinear
    attention_builder = _attention_builder(attention_base)
    row_builder = _row_linear_builder(row_base)
    local_layers = get_num_layers_to_build(transformer_config)
    checkpoint_layers = math.ceil(
        local_layers
        * {"speed": 0.5, "balanced": 0.75, "memory": 1.0}[memopt_checkpoint_budget]
    )
    rank = parallel_state.get_tensor_model_parallel_rank()
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    offset = get_transformer_layer_offset(transformer_config)
    specs = []
    for local_index in range(local_layers):
        global_index = offset + local_index
        checkpoint_enabled = bool(memopt_level and local_index < checkpoint_layers)
        attention_checkpoint = bool(
            checkpoint_enabled
            and transformer_config.recompute_granularity != "selective"
        )
        scales = calibration.layer_scales[global_index]
        spec = (
            get_gpt_layer_with_transformer_engine_spec()
            if use_te
            else get_gpt_layer_local_spec(normalization="RMSNorm")
        )
        spec.submodules.input_layernorm = (
            functools.partial(
                _InputQCFSRMSNorm,
                scale=calibration.input_scale,
                time_steps=conversion.time_steps,
                checkpoint_enabled=checkpoint_enabled,
            )
            if global_index == 0
            else _rms_norm
        )
        spec.submodules.pre_mlp_layernorm = _rms_norm
        attention = spec.submodules.self_attention.submodules
        attention.linear_qkv = column_base
        attention.core_attention = functools.partial(
            attention_builder,
            query_scale=_local_chunk(scales["query"], rank, world_size),
            key_scale=_local_chunk(scales["key"], rank, world_size),
            value_scale=_local_chunk(scales["value"], rank, world_size),
            time_steps=conversion.time_steps,
            checkpoint_enabled=attention_checkpoint,
        )
        spec.submodules.mlp = functools.partial(
            MLP.as_mlp_submodule,
            submodules=MLPSubmodules(
                linear_fc1=column_base,
                linear_fc2=functools.partial(
                    row_builder,
                    scale=_local_chunk(scales["mlp"], rank, world_size),
                    time_steps=conversion.time_steps,
                    checkpoint_enabled=checkpoint_enabled,
                ),
            ),
        )
        specs.append(spec)
    block_spec = TransformerBlockSubmodules(layer_specs=specs, layer_norm=_rms_norm)
    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=block_spec,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=True,
        share_embeddings_and_output_weights=config.share_embeddings_and_output_weights,
        position_embedding_type=config.position_embedding_type,
        rotary_base=_qwen2_rotary_base(source_config),
    )
    if source_path is not None:
        from transformers import AutoModelForCausalLM

        source = AutoModelForCausalLM.from_pretrained(source_path, dtype="auto")
        load_hf_qwen2_weights(model, source)
        del source
    model.snn_model_config = config
    model.snn_memopt_level = memopt_level
    model.temporal_output_reduction = "sum"
    model.checkpoint_metadata = {
        "recipe_name": "qwen2-qcfs-sg",
        "model_config": {
            "architecture": _qwen2_architecture(source_config),
            "conversion": asdict(conversion),
            "calibration_levels": calibration.calibration_levels,
        },
    }
    return model


class Qwen2Builder(ModelBuilder):
    def build(
        self,
        *,
        memopt_level: int = 0,
        memopt_checkpoint_budget: str = "memory",
        resume: bool,
    ) -> tuple["Callable", "Callable"]:
        from transformers import AutoConfig

        if not isinstance(self.config, Qwen2Config):
            raise TypeError("Qwen2Builder requires Qwen2Config.")
        source_config = AutoConfig.from_pretrained(self.config.source_path)
        calibration = Qwen2SNNCalibration.from_state_dict(
            torch.load(
                self.config.calibration_path, map_location="cpu", weights_only=True
            )
        )
        conversion = Qwen2SNNConfig(
            time_steps=calibration.time_steps,
            calibration_levels=calibration.calibration_levels,
            calibration_quantile=calibration.calibration_quantile,
            calibration_reservoir_size=calibration.calibration_reservoir_size,
            calibration_seed=calibration.calibration_seed,
        )
        return (
            functools.partial(
                model_provider,
                None if resume else self.config.source_path,
                source_config,
                calibration,
                conversion,
                self.config,
                memopt_level,
                memopt_checkpoint_budget,
            ),
            forward_step,
        )


def _source_layers(stage: SGLangExportStage) -> list[int]:
    layers = sorted(
        {
            int(match.group(1))
            for name in stage.tensor_names()
            if (match := re.match(r"decoder\.layers\.(\d+)\.", name))
        }
    )
    if len(layers) != stage.local_layer_count:
        raise ValueError("MCore PP stage does not contain the expected layer count.")
    return layers


def _reorder_qkv(value: torch.Tensor, heads: int, kv_heads: int) -> torch.Tensor:
    shape = value.shape
    head_dim = shape[0] // (heads + 2 * kv_heads)
    heads_per_group = heads // kv_heads
    grouped = value.reshape(kv_heads, heads_per_group + 2, head_dim, *shape[1:])
    query = grouped[:, :heads_per_group].reshape(-1, *shape[1:])
    key = grouped[:, heads_per_group].reshape(-1, *shape[1:])
    values = grouped[:, heads_per_group + 1].reshape(-1, *shape[1:])
    return torch.cat((query, key, values))


def _gated_tensor(stage: SGLangExportStage, name: str) -> torch.Tensor:
    chunks = [value.chunk(2, dim=0) for value in stage.tensor_shards(name)]
    return torch.cat(
        [chunk[0] for chunk in chunks] + [chunk[1] for chunk in chunks], dim=0
    )


def _sglang_tensors(
    config: Qwen2Config,
    source_config: Any,
    stage: SGLangExportStage,
) -> Iterator[tuple[str, torch.Tensor]]:
    input_scale = stage.merge_tensor(
        "decoder.layers.0.input_layernorm.qcfs_scale", pipeline_rank=0
    )
    embedding_weight = stage.merge_tensor(
        "embedding.word_embeddings.weight", dim=0, pipeline_rank=0
    )
    if stage.is_first:
        yield "model.embedding.weight", embedding_weight
    for position, source_index in enumerate(_source_layers(stage)):
        source = f"decoder.layers.{source_index}."
        target = f"model.layers.{stage.layer_offset + position}."
        mapping = {
            "input_layernorm.weight": ("input_norm.weight", None),
            "self_attention.core_attention.query_scale": ("attn.query_scale", 0),
            "self_attention.core_attention.key_scale": ("attn.key_scale", 0),
            "self_attention.core_attention.value_scale": ("attn.value_scale", 0),
            "self_attention.linear_proj.weight": ("attn.proj.weight", 1),
            "pre_mlp_layernorm.weight": ("mlp_norm.weight", None),
            "mlp.linear_fc2.weight": ("mlp.down.weight", 1),
            "mlp.linear_fc2.qcfs_scale": ("mlp.scale", 0),
        }
        yield target + "input_scale", input_scale
        for source_name, (target_name, dim) in mapping.items():
            yield (
                target + target_name,
                stage.merge_tensor(source + source_name, dim=dim),
            )
        yield (
            target + "mlp.gate_up.weight",
            _gated_tensor(stage, source + "mlp.linear_fc1.weight"),
        )
        qkv = stage.merge_tensor(source + "self_attention.linear_qkv.weight", dim=0)
        heads = int(source_config.num_attention_heads)
        kv_heads = int(source_config.num_key_value_heads)
        yield target + "attn.qkv.weight", _reorder_qkv(qkv, heads, kv_heads)
        if bool(getattr(source_config, "attention_bias", True)):
            qkv_bias = stage.merge_tensor(
                source + "self_attention.linear_qkv.bias", dim=0
            )
            yield target + "attn.qkv.bias", _reorder_qkv(qkv_bias, heads, kv_heads)
    if stage.is_last:
        yield (
            "model.final_norm.weight",
            stage.merge_tensor("decoder.final_layernorm.weight"),
        )
        output_weight = (
            embedding_weight
            if config.share_embeddings_and_output_weights
            else stage.merge_tensor("output_layer.weight", dim=0)
        )
        yield "lm_head.weight", output_weight


def export_sglang(
    config: Qwen2Config,
    model_provider: "Callable[[bool, bool], MegatronModule]",
    checkpoint: Path,
    output: Path,
    *,
    tokenizer: Path | None = None,
) -> None:
    r"""Export a Qwen2 checkpoint with the generic distributed SGLang exporter.

    **API Language** - 中文 | English

    **中文：** 模型 recipe 从 Hugging Face config 构造 SGLang config 并负责
    Qwen2 权重映射；通用导出器负责分布式生命周期和文件发布。

    **English:** The model recipe builds the SGLang configuration from the
    Hugging Face config and owns Qwen2 weight mapping; the generic exporter owns
    the distributed lifecycle and artifact publication.

    :param config: Qwen2 模型配置。 / Qwen2 model configuration.
    :type config: Qwen2Config
    :param model_provider: MCore model provider。 / MCore model provider.
    :type model_provider: Callable
    :param checkpoint: MCore checkpoint 目录。 / MCore checkpoint directory.
    :type checkpoint: pathlib.Path
    :param output: 新 artifact 目录。 / New artifact directory.
    :type output: pathlib.Path
    :param tokenizer: 可选 tokenizer 目录。 / Optional tokenizer directory.
    :type tokenizer: Optional[pathlib.Path]
    :return: None.
    :rtype: None
    """
    from transformers import AutoConfig

    source_config = AutoConfig.from_pretrained(config.source_path)
    head_dim = source_config.hidden_size // source_config.num_attention_heads
    artifact_config = {
        "architectures": ["SpikingJellyQwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": source_config.vocab_size,
        "hidden_size": source_config.hidden_size,
        "intermediate_size": source_config.intermediate_size,
        "num_hidden_layers": source_config.num_hidden_layers,
        "num_attention_heads": source_config.num_attention_heads * config.time_steps,
        "num_key_value_heads": source_config.num_key_value_heads * config.time_steps,
        "snn_num_attention_heads": source_config.num_attention_heads,
        "snn_num_key_value_heads": source_config.num_key_value_heads,
        "snn_time_steps": config.time_steps,
        "head_dim": head_dim,
        "max_position_embeddings": source_config.max_position_embeddings,
        "rms_norm_eps": source_config.rms_norm_eps,
        "rope_theta": _qwen2_rotary_base(source_config),
        "attention_bias": bool(getattr(source_config, "attention_bias", True)),
        "tie_word_embeddings": bool(source_config.tie_word_embeddings),
        "bos_token_id": source_config.bos_token_id,
        "eos_token_id": source_config.eos_token_id,
        "torch_dtype": "bfloat16",
    }
    export_sglang_artifact(
        config.transformer,
        model_provider,
        checkpoint,
        output,
        artifact_config=artifact_config,
        stage_tensors=functools.partial(_sglang_tensors, config, source_config),
        tokenizer=tokenizer,
    )


__all__ = [
    "Qwen2Builder",
    "Qwen2Config",
    "export_sglang",
    "forward_step",
    "load_hf_qwen2_weights",
    "model_provider",
]
