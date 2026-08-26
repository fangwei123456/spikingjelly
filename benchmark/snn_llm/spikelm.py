from __future__ import annotations

import functools
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from spikingjelly.activation_based import base
from spikingjelly.activation_based.distributed.llm import (
    ModelBuilder,
    ModelConfig,
    SGLangExportStage,
    export_sglang_artifact,
)
from spikingjelly.activation_based.distributed.llm.temporal import (
    pack_time_batch,
    run_functional_sequence,
    unpack_time_batch,
)
from ._causal_lm import forward_step

if TYPE_CHECKING:
    from collections.abc import Callable

    from megatron.core.transformer import MegatronModule, TransformerConfig


@dataclass(frozen=True, kw_only=True)
class SpikeLMConfig(ModelConfig):
    builder: ClassVar[str] = "benchmark.snn_llm.spikelm.SpikeLMBuilder"
    spike_decay: float = 0.25
    spike_amplitude: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.spike_decay <= 1.0:
            raise ValueError("spike_decay must lie in [0, 1].")
        if self.spike_amplitude <= 0.0:
            raise ValueError("spike_amplitude must be positive.")


SpikeLMConfig.__init__.__doc__ = r"""Initialize the MCore-native SpikeLM recipe.

**API Language** - :ref:`中文 <SpikeLMConfig-cn>` | :ref:`English <SpikeLMConfig-en>`

----

.. _SpikeLMConfig-cn:

* **中文**

定义完整的 SpikeLM 模型配置：MCore Transformer 结构、词表、上下文、时间步和
elastic bi-spiking 参数。

:param transformer: MCore Transformer 配置。
:type transformer: megatron.core.transformer.TransformerConfig
:param vocab_size: 词表大小。
:type vocab_size: int
:param max_sequence_length: 最大上下文长度。
:type max_sequence_length: int
:param time_steps: SNN 时间步。
:type time_steps: int
:param share_embeddings_and_output_weights: 是否共享 embedding 与输出权重。
:type share_embeddings_and_output_weights: bool
:param position_embedding_type: 位置编码类型。
:type position_embedding_type: str
:param spike_decay: 膜电位残留系数。
:type spike_decay: float
:param spike_amplitude: 双向脉冲幅值。
:type spike_amplitude: float
:raises ValueError: 数值范围无效。

----

.. _SpikeLMConfig-en:

* **English**

Defines the complete SpikeLM model configuration: MCore Transformer structure,
vocabulary, context, time steps, and elastic bi-spiking parameters.

:param transformer: MCore Transformer configuration.
:type transformer: megatron.core.transformer.TransformerConfig
:param vocab_size: Vocabulary size.
:type vocab_size: int
:param max_sequence_length: Maximum context length.
:type max_sequence_length: int
:param time_steps: SNN time steps.
:type time_steps: int
:param share_embeddings_and_output_weights: Whether to tie embedding and output weights.
:type share_embeddings_and_output_weights: bool
:param position_embedding_type: Position-embedding type.
:type position_embedding_type: str
:param spike_decay: Membrane carry coefficient.
:type spike_decay: float
:param spike_amplitude: Bidirectional spike amplitude.
:type spike_amplitude: float
:raises ValueError: If a numeric range is invalid.
"""


class _ElasticBiSpike(base.MemoryModule):
    def __init__(self, time_steps: int, decay: float, amplitude: float) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.decay = decay
        self.register_buffer("amplitude", torch.full((time_steps,), amplitude))
        self.register_memory("v", 0.0)
        self.step_mode = "m"

    def materialize_states(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[Any, ...],
        step_mode: str,
    ) -> tuple[torch.Tensor, ...]:
        del step_mode
        voltage = states[0]
        if not isinstance(voltage, torch.Tensor):
            voltage = torch.zeros_like(inputs[0][0])
        return (voltage,)

    def multi_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[torch.Tensor, ...],
        **kwargs: Any,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        del kwargs
        sequence = inputs[0]
        if sequence.shape[0] != self.time_steps:
            raise ValueError(
                f"Expected T={self.time_steps}, got sequence length {sequence.shape[0]}."
            )
        voltage = states[0]
        previous_spike = torch.zeros_like(voltage)
        spikes = []
        for step, current in enumerate(sequence):
            amplitude = self.amplitude[step].to(dtype=current.dtype)
            if step == 0:
                voltage = voltage + current
            else:
                voltage = (
                    voltage
                    * self.decay
                    * (
                        self.amplitude[step - 1].to(dtype=current.dtype)
                        - previous_spike.detach()
                    )
                    + current
                )
            scaled = (voltage / amplitude).clamp(-1.0, 1.0)
            rounded = scaled.round()
            previous_spike = ((rounded - scaled).detach() + scaled) * amplitude
            spikes.append(previous_spike)
        return (torch.stack(spikes),), (voltage,)


class _SpikingLayerNorm(nn.LayerNorm):
    def __init__(
        self,
        *,
        config: "TransformerConfig",
        hidden_size: int,
        eps: float,
        time_steps: int,
        decay: float,
        amplitude: float,
        checkpoint_spike: bool,
    ) -> None:
        del config
        super().__init__(hidden_size, eps=eps)
        self.time_steps = time_steps
        self.spike = _ElasticBiSpike(time_steps, decay, amplitude)
        self.checkpoint_spike = checkpoint_spike

    def _spike(self, hidden: torch.Tensor) -> torch.Tensor:
        sequence = unpack_time_batch(hidden, self.time_steps)
        spikes = run_functional_sequence(self.spike, (sequence,))[0]
        return pack_time_batch(spikes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = super().forward(hidden)
        if self.checkpoint_spike and torch.is_grad_enabled():
            return torch_checkpoint(
                self._spike, hidden, use_reentrant=True, preserve_rng_state=False
            )
        return self._spike(hidden)


def _layer_norm(
    *, config: "TransformerConfig", hidden_size: int, eps: float
) -> nn.LayerNorm:
    del config
    return nn.LayerNorm(hidden_size, eps=eps)


def model_provider(
    config: SpikeLMConfig,
    memopt_level: int,
    memopt_checkpoint_budget: str,
    memopt_compress_inputs: bool,
    pre_process: bool,
    post_process: bool,
) -> "MegatronModule":
    r"""Build the local MCore pipeline stage for SpikeLM.

    **API Language** - :ref:`中文 <spikelm-provider-cn>` | :ref:`English <spikelm-provider-en>`

    ----

    .. _spikelm-provider-cn:

    * **中文**

    通过公开 ``ModuleSpec`` 在 attention 与 MLP 投影前注入 functional elastic
    bi-spiking transition；FP8 配置使用 TE GPT layer spec，且不会静默回退。

    :param config: 完整 SpikeLM 模型配置。
    :type config: SpikeLMConfig
    :param memopt_level: SpikingJelly memopt 级别。
    :type memopt_level: int
    :param memopt_checkpoint_budget: checkpoint 数量预设。
    :type memopt_checkpoint_budget: str
    :param memopt_compress_inputs: 是否压缩二值输入。
    :type memopt_compress_inputs: bool
    :param pre_process: 当前 PP stage 是否拥有 embedding。
    :type pre_process: bool
    :param post_process: 当前 PP stage 是否拥有 LM head。
    :type post_process: bool
    :return: 当前 PP stage 的 MCore GPT model。
    :rtype: megatron.core.transformer.MegatronModule
    :raises ValueError: 模型 context 与训练配置不一致，或 normalization 不受支持。
    :raises ImportError: 启用 FP8 但 Transformer Engine 不可用。

    ----

    .. _spikelm-provider-en:

    * **English**

    Injects functional elastic bi-spiking transitions before attention and MLP
    projections through public ``ModuleSpec``. FP8 uses the TE GPT layer spec and
    never silently falls back.

    :param config: Complete SpikeLM model configuration.
    :type config: SpikeLMConfig
    :param memopt_level: SpikingJelly memopt level.
    :type memopt_level: int
    :param memopt_checkpoint_budget: Checkpoint-count preset.
    :type memopt_checkpoint_budget: str
    :param memopt_compress_inputs: Whether binary inputs are compressed.
    :type memopt_compress_inputs: bool
    :param pre_process: Whether this PP stage owns the embedding.
    :type pre_process: bool
    :param post_process: Whether this PP stage owns the LM head.
    :type post_process: bool
    :return: MCore GPT model for the current PP stage.
    :rtype: megatron.core.transformer.MegatronModule
    :raises ValueError: If model context disagrees with training or normalization is unsupported.
    :raises ImportError: If FP8 is enabled without Transformer Engine.
    """
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.transformer_block import (
        TransformerBlockSubmodules,
        get_num_layers_to_build,
    )

    transformer_config = config.transformer
    if transformer_config.normalization != "LayerNorm":
        raise ValueError("SpikeLM currently requires MCore LayerNorm.")
    use_te = (
        transformer_config.fp8 is not None
        or transformer_config.context_parallel_size > 1
    )
    del memopt_compress_inputs
    local_layers = get_num_layers_to_build(transformer_config)
    checkpoint_layers = math.ceil(
        local_layers
        * {"speed": 0.5, "balanced": 0.75, "memory": 1.0}[memopt_checkpoint_budget]
    )
    layer_specs = []
    for index in range(local_layers):
        layer_spec = (
            get_gpt_layer_with_transformer_engine_spec()
            if use_te
            else get_gpt_layer_local_spec()
        )
        spiking_norm = functools.partial(
            _SpikingLayerNorm,
            time_steps=config.time_steps,
            decay=config.spike_decay,
            amplitude=config.spike_amplitude,
            checkpoint_spike=bool(memopt_level and index < checkpoint_layers),
        )
        layer_spec.submodules.input_layernorm = spiking_norm
        layer_spec.submodules.pre_mlp_layernorm = spiking_norm
        if use_te:
            from megatron.core.extensions.transformer_engine import (
                TEColumnParallelLinear,
            )

            layer_spec.submodules.self_attention.submodules.linear_qkv = (
                TEColumnParallelLinear
            )
            mlp_submodules = layer_spec.submodules.mlp.keywords["submodules"]
            layer_spec.submodules.mlp = functools.partial(
                MLP.as_mlp_submodule,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=mlp_submodules.linear_fc2,
                ),
            )
        layer_specs.append(layer_spec)
    block_spec = TransformerBlockSubmodules(
        layer_specs=layer_specs,
        layer_norm=_layer_norm,
    )
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
    )
    model.snn_model_config = config
    model.snn_memopt_level = memopt_level
    model.temporal_output_reduction = "mean"
    model.checkpoint_metadata = {
        "recipe_name": "spikelm",
        "model_config": {
            "spike_decay": config.spike_decay,
            "spike_amplitude": config.spike_amplitude,
            "transformer": {
                "num_layers": transformer_config.num_layers,
                "hidden_size": transformer_config.hidden_size,
                "num_attention_heads": transformer_config.num_attention_heads,
                "ffn_hidden_size": transformer_config.ffn_hidden_size,
                "normalization": transformer_config.normalization,
                "layernorm_epsilon": transformer_config.layernorm_epsilon,
            },
        },
    }
    return model


class SpikeLMBuilder(ModelBuilder):
    def build(
        self,
        *,
        memopt_level: int,
        memopt_checkpoint_budget: str,
        memopt_compress_inputs: bool,
        resume: bool,
    ) -> tuple["Callable", "Callable"]:
        del resume
        if not isinstance(self.config, SpikeLMConfig):
            raise TypeError("SpikeLMBuilder requires SpikeLMConfig.")
        return (
            functools.partial(
                model_provider,
                self.config,
                memopt_level,
                memopt_checkpoint_budget,
                memopt_compress_inputs,
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


def _sglang_tensors(
    config: SpikeLMConfig, stage: SGLangExportStage
) -> Iterator[tuple[str, torch.Tensor]]:
    embedding_weight = None
    if stage.is_first or (stage.is_last and config.share_embeddings_and_output_weights):
        embedding_weight = stage.merge_tensor(
            "embedding.word_embeddings.weight", dim=0, pipeline_rank=0
        )
    if stage.is_first:
        yield "model.embedding.weight", embedding_weight
    for position, source_index in enumerate(_source_layers(stage)):
        source = f"decoder.layers.{source_index}."
        target = f"model.layers.{stage.layer_offset + position}."
        mapping = {
            "input_layernorm.weight": ("attn_norm.norm.weight", None),
            "input_layernorm.bias": ("attn_norm.norm.bias", None),
            "input_layernorm.spike.amplitude": ("attn_norm.amplitude", None),
            "self_attention.linear_proj.weight": ("attn.proj.weight", 1),
            "self_attention.linear_proj.bias": ("attn.proj.bias", None),
            "pre_mlp_layernorm.weight": ("mlp_norm.norm.weight", None),
            "pre_mlp_layernorm.bias": ("mlp_norm.norm.bias", None),
            "pre_mlp_layernorm.spike.amplitude": ("mlp_norm.amplitude", None),
            "mlp.linear_fc1.weight": ("mlp.fc1.weight", 0),
            "mlp.linear_fc1.bias": ("mlp.fc1.bias", 0),
            "mlp.linear_fc2.weight": ("mlp.fc2.weight", 1),
            "mlp.linear_fc2.bias": ("mlp.fc2.bias", None),
        }
        for source_name, (target_name, dim) in mapping.items():
            yield (
                target + target_name,
                stage.merge_tensor(source + source_name, dim=dim),
            )
        for suffix in ("weight", "bias"):
            name = source + f"self_attention.linear_qkv.{suffix}"
            value = stage.merge_tensor(name, dim=0)
            heads = config.transformer.num_attention_heads
            head_dim = value.shape[0] // (3 * heads)
            grouped = value.reshape(heads, 3, head_dim, *value.shape[1:])
            yield (
                target + f"attn.qkv.{suffix}",
                torch.cat(
                    tuple(
                        grouped[:, index].reshape(-1, *value.shape[1:])
                        for index in range(3)
                    )
                ),
            )
    if stage.is_last:
        yield (
            "model.final_norm.weight",
            stage.merge_tensor("decoder.final_layernorm.weight"),
        )
        yield (
            "model.final_norm.bias",
            stage.merge_tensor("decoder.final_layernorm.bias"),
        )
        yield (
            "lm_head.weight",
            embedding_weight
            if config.share_embeddings_and_output_weights
            else stage.merge_tensor("output_layer.weight", dim=0),
        )


def export_sglang(
    config: SpikeLMConfig,
    model_provider: "Callable[[bool, bool], MegatronModule]",
    checkpoint: Path,
    output: Path,
    *,
    tokenizer: Path | None = None,
) -> None:
    r"""Export a SpikeLM checkpoint with the generic distributed SGLang exporter.

    **API Language** - 中文 | English

    **中文：** 模型 recipe 负责 SpikeLM 权重映射和 artifact config；
    :func:`export_sglang_artifact` 负责分布式生命周期和文件发布。

    **English:** The model recipe owns SpikeLM weight mapping and artifact
    configuration, while :func:`export_sglang_artifact` owns the distributed
    lifecycle and artifact publication.

    :param config: SpikeLM 模型配置。 / SpikeLM model configuration.
    :type config: SpikeLMConfig
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
    transformer = config.transformer
    heads = transformer.num_attention_heads
    artifact_config = {
        "architectures": ["SpikingJellySpikeLMForCausalLM"],
        "model_type": "gpt2",
        "vocab_size": config.vocab_size,
        "n_embd": transformer.hidden_size,
        "n_layer": transformer.num_layers,
        "n_head": heads * config.time_steps,
        "n_inner": transformer.ffn_hidden_size,
        "n_positions": config.max_sequence_length,
        "hidden_size": transformer.hidden_size,
        "num_hidden_layers": transformer.num_layers,
        "num_attention_heads": heads * config.time_steps,
        "num_key_value_heads": heads * config.time_steps,
        "head_dim": transformer.hidden_size // heads,
        "intermediate_size": transformer.ffn_hidden_size,
        "max_position_embeddings": config.max_sequence_length,
        "layer_norm_epsilon": transformer.layernorm_epsilon,
        "rope_theta": 10000.0,
        "tie_word_embeddings": config.share_embeddings_and_output_weights,
        "torch_dtype": "bfloat16",
        "bos_token_id": None,
        "eos_token_id": None,
        "snn_time_steps": config.time_steps,
        "snn_num_attention_heads": heads,
        "snn_spike_decay": config.spike_decay,
        "snn_spike_amplitude": config.spike_amplitude,
    }
    export_sglang_artifact(
        transformer,
        model_provider,
        checkpoint,
        output,
        artifact_config=artifact_config,
        stage_tensors=functools.partial(_sglang_tensors, config),
        tokenizer=tokenizer,
    )


__all__ = [
    "SpikeLMBuilder",
    "SpikeLMConfig",
    "export_sglang",
    "forward_step",
    "model_provider",
]
