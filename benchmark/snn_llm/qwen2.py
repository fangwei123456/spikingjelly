from __future__ import annotations

import functools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn as nn

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
)
from spikingjelly.activation_based.memopt import (
    NullSpikeCompressor,
    input_compressed_gc,
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
    compressor: NullSpikeCompressor | None = None,
) -> torch.Tensor:
    if compressor is not None and torch.is_grad_enabled():
        return input_compressed_gc(
            lambda value: _encode_envelope(value, scale, time_steps),
            compressor,
            hidden,
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
        use_snn_memopt: bool,
    ) -> None:
        del config
        super().__init__(hidden_size, eps=eps)
        self.time_steps = time_steps
        self.register_buffer("qcfs_scale", scale.detach().clone())
        self.compressor = NullSpikeCompressor() if use_snn_memopt else None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.compressor is not None and torch.is_grad_enabled():
            hidden = input_compressed_gc(
                lambda value: _encode_input(value, self.qcfs_scale, self.time_steps),
                self.compressor,
                hidden,
            )
        else:
            hidden = _encode_input(hidden, self.qcfs_scale, self.time_steps)
        return super().forward(hidden)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
        from megatron.core.transformer.utils import (
            make_sharded_tensors_for_checkpoint,
        )

        kwargs = {
            "tp_group": getattr(self, "tp_group", getattr(self, "_tp_group", None)),
            "dp_cp_group": metadata["dp_cp_group"],
        }
        return {
            **make_sharded_tensors_for_checkpoint(
                {"weight": self.weight}, prefix, {}, sharded_offsets, **kwargs
            ),
            **make_sharded_tensors_for_checkpoint(
                {"qcfs_scale": self.qcfs_scale}, prefix, {}, (), **kwargs
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
            use_snn_memopt: bool,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.time_steps = time_steps
            self.register_buffer("query_scale", query_scale.detach().clone())
            self.register_buffer("key_scale", key_scale.detach().clone())
            self.register_buffer("value_scale", value_scale.detach().clone())
            self.compressor = NullSpikeCompressor() if use_snn_memopt else None

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            query = _encode_envelope(
                query, self.query_scale, self.time_steps, self.compressor
            )
            key = _encode_envelope(
                key, self.key_scale, self.time_steps, self.compressor
            )
            value = _encode_envelope(
                value, self.value_scale, self.time_steps, self.compressor
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
            use_snn_memopt: bool,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.time_steps = time_steps
            self.register_buffer("qcfs_scale", scale.detach().clone())
            self.compressor = NullSpikeCompressor() if use_snn_memopt else None

        def forward(self, hidden: torch.Tensor, *args: Any, **kwargs: Any):
            hidden = _encode_envelope(
                hidden, self.qcfs_scale, self.time_steps, self.compressor
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
    use_snn_memopt: bool,
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
    :param use_snn_memopt: 是否 checkpoint 确定性 QCFS 变换。
    :type use_snn_memopt: bool
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
    :param use_snn_memopt: Whether deterministic QCFS transforms are checkpointed.
    :type use_snn_memopt: bool
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
    attention_memopt = (
        use_snn_memopt and transformer_config.recompute_granularity != "selective"
    )
    rank = parallel_state.get_tensor_model_parallel_rank()
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    offset = get_transformer_layer_offset(transformer_config)
    specs = []
    for local_index in range(get_num_layers_to_build(transformer_config)):
        global_index = offset + local_index
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
                use_snn_memopt=use_snn_memopt,
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
            use_snn_memopt=attention_memopt,
        )
        spec.submodules.mlp = functools.partial(
            MLP.as_mlp_submodule,
            submodules=MLPSubmodules(
                linear_fc1=column_base,
                linear_fc2=functools.partial(
                    row_builder,
                    scale=_local_chunk(scales["mlp"], rank, world_size),
                    time_steps=conversion.time_steps,
                    use_snn_memopt=use_snn_memopt,
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
    model.snn_memopt_enabled = use_snn_memopt
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
        self, *, use_snn_memopt: bool, resume: bool
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
                use_snn_memopt,
            ),
            forward_step,
        )


__all__ = [
    "Qwen2Builder",
    "Qwen2Config",
    "forward_step",
    "load_hf_qwen2_weights",
    "model_provider",
]
