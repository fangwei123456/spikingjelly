"""Configuration for distributed SNN language-model training."""

from __future__ import annotations

import abc
import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.transformer import MegatronModule, TransformerConfig


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    r"""Describe one SNN language-model architecture.

    **API Language** - 中文 | English

    **中文：** LLM 模型配置的基类。具体模型子类通过 ``builder`` 类变量绑定
    architecture-specific builder，并在本对象内组合 MCore ``TransformerConfig``、
    词表、上下文、时间步和模型族专有字段。

    :param transformer: MCore Transformer 结构、精度与并行配置。
    :type transformer: megatron.core.transformer.TransformerConfig
    :param vocab_size: 词表大小。
    :type vocab_size: int
    :param max_sequence_length: 最大 token context 长度。
    :type max_sequence_length: int
    :param time_steps: 每个语义样本的 SNN 时间步 ``T``。
    :type time_steps: int
    :param share_embeddings_and_output_weights: 是否共享 embedding 与输出权重。
    :type share_embeddings_and_output_weights: bool
    :param position_embedding_type: ``"rope"`` 或 ``"learned_absolute"``。
    :type position_embedding_type: str
    :raises ValueError: 正整数或位置编码类型无效。

    **English:** Base configuration for SNN language models. Concrete subclasses
    bind an architecture-specific builder through the ``builder`` class variable
    and combine MCore ``TransformerConfig`` with vocabulary, context, temporal,
    and model-family-specific fields.

    :param transformer: MCore Transformer structure, precision, and parallelism.
    :type transformer: megatron.core.transformer.TransformerConfig
    :param vocab_size: Vocabulary size.
    :type vocab_size: int
    :param max_sequence_length: Maximum token context length.
    :type max_sequence_length: int
    :param time_steps: SNN time steps ``T`` per semantic sample.
    :type time_steps: int
    :param share_embeddings_and_output_weights: Whether to tie embedding and output weights.
    :type share_embeddings_and_output_weights: bool
    :param position_embedding_type: ``"rope"`` or ``"learned_absolute"``.
    :type position_embedding_type: str
    :raises ValueError: If a positive integer or position-embedding type is invalid.
    """

    builder: ClassVar[str]
    transformer: "TransformerConfig"
    vocab_size: int
    max_sequence_length: int
    time_steps: int
    share_embeddings_and_output_weights: bool = False
    position_embedding_type: Literal["rope", "learned_absolute"] = "rope"

    def __post_init__(self) -> None:
        for name in ("vocab_size", "max_sequence_length", "time_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.position_embedding_type not in {"rope", "learned_absolute"}:
            raise ValueError(
                "position_embedding_type must be 'rope' or 'learned_absolute'."
            )

    def get_builder_cls(self) -> type[ModelBuilder]:
        r"""Resolve the model builder declared by the concrete configuration.

        **中文：** 返回 ``builder`` 导入路径指向的模型 builder。
        **English:** Return the model builder identified by the ``builder`` path.

        :return: 模型 builder 类。 / Model-builder class.
        :rtype: type
        :raises ImportError: builder 模块无法导入。 / If its module cannot be imported.
        :raises AttributeError: builder 类不存在。 / If the class does not exist.
        :raises TypeError: builder 未继承 :class:`ModelBuilder`。 / If the class
            does not inherit :class:`ModelBuilder`.
        """
        module_name, class_name = self.builder.rsplit(".", 1)
        builder_cls = getattr(importlib.import_module(module_name), class_name)
        if not isinstance(builder_cls, type) or not issubclass(
            builder_cls, ModelBuilder
        ):
            raise TypeError("model builder must inherit llm.ModelBuilder.")
        return builder_cls

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "builder": self.builder,
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "time_steps": self.time_steps,
            "share_embeddings_and_output_weights": self.share_embeddings_and_output_weights,
            "position_embedding_type": self.position_embedding_type,
        }


class ModelBuilder(abc.ABC):
    r"""Build one SNN language-model architecture for MCore training.

    **API Language** - 中文 | English

    **中文：** model config 与 architecture-specific MCore 构造逻辑之间的 seam。
    实现返回 pipeline 使用的模型 provider 和 forward-step；训练生命周期由
    :func:`train` 拥有。

    **English:** Seam between model configuration and architecture-specific MCore
    construction. Implementations return the model provider and forward-step used
    by the pipeline while :func:`train` owns the training lifecycle.

    :param config: 模型专项配置。 / Architecture-specific model configuration.
    :type config: ModelConfig
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def build(
        self,
        *,
        memopt_level: int = 0,
        memopt_checkpoint_budget: Literal["speed", "balanced", "memory"] = "memory",
        resume: bool,
    ) -> tuple[
        Callable[[bool, bool], "MegatronModule"],
        Callable[[Iterator[Any], "MegatronModule"], tuple[torch.Tensor, Callable]],
    ]:
        r"""Build the MCore callbacks for this architecture.

        **中文：** 返回模型 provider 与 forward-step。
        **English:** Return the model provider and forward-step.

        :param memopt_level: SpikingJelly memopt 级别。 / SpikingJelly memopt level.
        :type memopt_level: int
        :param memopt_checkpoint_budget: checkpoint 数量预设。 / Checkpoint-count preset.
        :type memopt_checkpoint_budget: Literal["speed", "balanced", "memory"]
        :param resume: 是否从 MCore checkpoint 恢复。 / Whether this run resumes
            from an MCore checkpoint.
        :type resume: bool
        :return: ``(model_provider, forward_step)``。
        :rtype: tuple[Callable, Callable]
        """
        raise NotImplementedError


@dataclass(frozen=True)
class EvaluationConfig:
    model: ModelConfig
    checkpoint: Path
    dataset_builder: str
    sequence_length: int
    micro_batch_size: int
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int = 1234
    timing_warmup_batches: int = 0
    pipeline_microbatches: int = 1

    def __post_init__(self) -> None:
        if "." not in self.dataset_builder:
            raise ValueError("dataset_builder must be a full import path.")
        if (
            self.sequence_length <= 0
            or self.micro_batch_size <= 0
            or self.pipeline_microbatches <= 0
        ):
            raise ValueError(
                "sequence_length, micro_batch_size, and pipeline_microbatches "
                "must be positive."
            )
        if self.timing_warmup_batches < 0:
            raise ValueError("timing_warmup_batches must be non-negative.")
        if self.sequence_length > self.model.max_sequence_length:
            raise ValueError("sequence_length cannot exceed model.max_sequence_length.")


EvaluationConfig.__init__.__doc__ = r"""Configure standalone MCore loss and perplexity evaluation.

**API Language** - 中文 | English

**中文：** 从 optimizer-boundary checkpoint 仅恢复 model，并使用
``ModelConfig.transformer`` 中的 DP/TP/PP/CP 拓扑评测完整 token dataset。
dataset builder 必须返回非空 ``Dataset``，元素包含 ``input_ids``、``labels``
和可选 ``loss_mask``。

**English:** Restore model state only from an optimizer-boundary checkpoint and
evaluate a complete token dataset with the DP/TP/PP/CP topology in
``ModelConfig.transformer``. The non-empty dataset must provide ``input_ids``,
``labels``, and an optional ``loss_mask``.

:param model: MCore SNN 模型配置。 / MCore SNN model configuration.
:type model: ModelConfig
:param checkpoint: 已完成的训练 checkpoint。 / Completed training checkpoint.
:type checkpoint: pathlib.Path
:param dataset_builder: 返回一个 Dataset 的完整导入路径。 / Full import path
    returning one Dataset.
:type dataset_builder: str
:param sequence_length: token 序列长度 ``S``。 / Token sequence length ``S``.
:type sequence_length: int
:param micro_batch_size: 每个 pipeline microbatch 的语义样本数。 / Semantic
    samples per pipeline microbatch.
:type micro_batch_size: int
:param dataset_kwargs: dataset builder 参数。 / Dataset-builder arguments.
:type dataset_kwargs: dict[str, Any]
:param seed: sampler 与 MCore model seed。 / Sampler and MCore model seed.
:type seed: int
:param timing_warmup_batches: 计时前从 dataset 起点重复执行、但不计入指标的
    schedule batch 数。 / Schedule batches repeatedly run from the dataset start
    before timing and excluded from metrics.
:type timing_warmup_batches: int
:param pipeline_microbatches: 每次 pipeline schedule 的 microbatch 数；与
    ``micro_batch_size`` 的乘积是每个 DP rank 的本地 schedule batch。 /
    Microbatches per pipeline schedule; their product with ``micro_batch_size``
    is the local schedule batch per DP rank.
:type pipeline_microbatches: int
:raises ValueError: 尺寸或导入路径无效。 / If a size or import path is invalid.
"""


@dataclass(frozen=True)
class MCoreGenerationConfig:
    model: ModelConfig
    checkpoint: Path
    max_new_tokens: int
    eos_token_id: Optional[int] = None
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if (
            self.eos_token_id is not None
            and not 0 <= self.eos_token_id < self.model.vocab_size
        ):
            raise ValueError("eos_token_id must lie in the model vocabulary.")
        if self.model.transformer.context_parallel_size != 1:
            raise ValueError(
                "MCore cached generation requires context_parallel_size=1."
            )
        if self.model.transformer.sequence_parallel:
            raise ValueError(
                "MCore cached generation requires sequence_parallel=False."
            )


MCoreGenerationConfig.__init__.__doc__ = r"""Configure offline MCore cached generation.

**API Language** - 中文 | English

**中文：** 从 checkpoint 恢复 MCore model，使用 TP/PP 执行 cached greedy
generation，并沿 DP 切分 prompt batch。MCore cached generation 要求
``context_parallel_size=1``。

**English:** Restore an MCore model, run cached greedy generation with TP/PP,
and shard the prompt batch over DP replicas. MCore cached generation requires
``context_parallel_size=1``.

:param model: MCore SNN 模型配置。 / MCore SNN model configuration.
:type model: ModelConfig
:param checkpoint: 已完成的训练 checkpoint。 / Completed training checkpoint.
:type checkpoint: pathlib.Path
:param max_new_tokens: 最大生成 token 数。 / Maximum generated token count.
:type max_new_tokens: int
:param eos_token_id: 可选 EOS token ID。 / Optional EOS token ID.
:type eos_token_id: Optional[int]
:param seed: MCore model seed。 / MCore model seed.
:type seed: int
:raises ValueError: 生成参数或 MCore 拓扑无效。 / If a generation value or
    MCore topology is invalid.
"""


@dataclass(frozen=True)
class SGLangEngineConfig:
    artifact: Path
    external_model_package: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    memory_fraction: float = 0.9
    seed: int = 1234
    tokenizer: Optional[Path] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.external_model_package, str)
            or not self.external_model_package.strip()
        ):
            raise ValueError("external_model_package must be non-empty.")
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 < self.memory_fraction < 1.0:
            raise ValueError("memory_fraction must lie in (0, 1).")


SGLangEngineConfig.__init__.__doc__ = r"""Configure a managed SGLang offline Engine.

**API Language** - 中文 | English

**中文：** 在独立 Python 环境中使用 SGLang offline ``Engine`` 加载
SpikingJelly artifact。首版支持单节点 NVIDIA BF16 TP/PP/DP，不支持 context
parallel。采样、变长 token IDs、异步生成和 streaming 直接使用原生 Engine 接口。

**English:** Load a SpikingJelly artifact with SGLang's offline ``Engine`` in a
separate Python environment. The first release supports single-node NVIDIA BF16
TP/PP/DP and no context parallelism. Sampling, variable-length token IDs,
asynchronous generation, and streaming use the native Engine interface.

:param artifact: SGLang/Hugging Face 风格的模型目录。 / SGLang/Hugging Face
    style model directory.
:type artifact: pathlib.Path
:param external_model_package: SGLang external model package 的显式导入路径。 /
    Explicit import path of the SGLang external model package.
:type external_model_package: str
:param tensor_parallel_size: SGLang TP 大小。 / SGLang TP size.
:type tensor_parallel_size: int
:param pipeline_parallel_size: SGLang PP 大小。 / SGLang PP size.
:type pipeline_parallel_size: int
:param data_parallel_size: SGLang DP 大小。 / SGLang DP size.
:type data_parallel_size: int
:param memory_fraction: SGLang 预留的空闲 GPU 显存比例。 / Fraction of free GPU
    memory reserved by SGLang.
:type memory_fraction: float
:param seed: SGLang sampling seed。 / SGLang sampling seed.
:type seed: int
:param tokenizer: 可选 tokenizer 目录；``None`` 使用 token-in/token-out。 /
    Optional tokenizer directory; ``None`` uses token-in/token-out.
:type tokenizer: Optional[pathlib.Path]
:raises ValueError: 外部包、拓扑或显存比例无效。 / If the external package,
    topology, or memory fraction is invalid.
"""


@dataclass
class TrainingConfig:
    model: ModelConfig
    optimizer: "OptimizerConfig"
    dataset_builder: str
    sequence_length: int
    micro_batch_size: int
    global_batch_size: int
    train_steps: int
    timing_warmup_steps: int = 0
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    eval_interval: int = 0
    eval_steps: int = 0
    log_interval: int = 10
    lr_warmup_steps: int = 0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "cosine"
    checkpoint_dir: Optional[Path] = None
    checkpoint_interval: int = 0
    resume: Optional[Path] = None
    seed: int = 1234
    memopt_level: int = 0
    memopt_checkpoint_budget: Literal["speed", "balanced", "memory"] = "memory"

    def __post_init__(self) -> None:
        if "." not in self.dataset_builder:
            raise ValueError("dataset_builder must be a full import path.")
        for name in (
            "sequence_length",
            "micro_batch_size",
            "global_batch_size",
            "train_steps",
            "log_interval",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.global_batch_size % self.micro_batch_size:
            raise ValueError("global_batch_size must be divisible by micro_batch_size.")
        if not 0 <= self.timing_warmup_steps < self.train_steps:
            raise ValueError("timing_warmup_steps must lie in [0, train_steps).")
        if self.sequence_length > self.model.max_sequence_length:
            raise ValueError("sequence_length cannot exceed model.max_sequence_length.")
        if (self.eval_interval == 0) != (self.eval_steps == 0):
            raise ValueError(
                "eval_interval and eval_steps must both be zero or positive."
            )
        if self.eval_interval < 0 or self.eval_steps < 0:
            raise ValueError("eval_interval and eval_steps cannot be negative.")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval cannot be negative.")
        if self.checkpoint_interval and self.checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir is required when checkpointing is enabled."
            )
        if self.lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps cannot be negative.")
        if self.lr_decay_steps is None:
            self.lr_decay_steps = self.train_steps
        if self.lr_decay_steps <= self.lr_warmup_steps:
            raise ValueError("lr_decay_steps must be greater than lr_warmup_steps.")
        if self.lr_decay_style not in {
            "constant",
            "cosine",
            "inverse-square-root",
            "linear",
        }:
            raise ValueError(f"Unsupported lr_decay_style={self.lr_decay_style!r}.")

        transformer = self.model.transformer
        if transformer.context_parallel_size > 1 and self.sequence_length % (
            2 * transformer.context_parallel_size
        ):
            raise ValueError(
                "sequence_length must be divisible by 2 * context_parallel_size."
            )
        if transformer.expert_model_parallel_size != 1:
            raise ValueError("Expert parallelism is not supported.")
        if not 0 <= self.memopt_level <= 4:
            raise ValueError("memopt_level must lie in [0, 4].")
        if self.memopt_checkpoint_budget not in {"speed", "balanced", "memory"}:
            raise ValueError(
                "memopt_checkpoint_budget must be 'speed', 'balanced', or 'memory'."
            )
        recompute = transformer.recompute_granularity
        if self.memopt_level and recompute == "full":
            raise ValueError("SNN memopt cannot overlap MCore full recompute.")
        if self.memopt_level and recompute == "selective":
            if set(transformer.recompute_modules or ()) != {"core_attn"}:
                raise ValueError(
                    "SNN memopt only supports MCore selective core_attn recompute."
                )
            if (
                transformer.recompute_method is not None
                or transformer.recompute_num_layers is not None
            ):
                raise ValueError(
                    "MCore selective recompute must not set method or num_layers."
                )
        if not transformer.calculate_per_token_loss:
            raise ValueError("MCore calculate_per_token_loss must be enabled.")
        if not self.optimizer.use_distributed_optimizer:
            raise ValueError("MCore distributed optimizer must be enabled.")
        if transformer.fp16 != self.optimizer.fp16:
            raise ValueError("transformer.fp16 and optimizer.fp16 must match.")
        if transformer.bf16 != self.optimizer.bf16:
            raise ValueError("transformer.bf16 and optimizer.bf16 must match.")
        if transformer.params_dtype != self.optimizer.params_dtype:
            raise ValueError("Transformer and optimizer params_dtype must match.")
        expected_dtype = (
            torch.float16
            if transformer.fp16
            else torch.bfloat16
            if transformer.bf16
            else torch.float32
        )
        if transformer.params_dtype != expected_dtype:
            raise ValueError(
                f"params_dtype must be {expected_dtype} for the selected precision."
            )
        if (
            transformer.pipeline_model_parallel_size > 1
            and transformer.pipeline_dtype != expected_dtype
        ):
            raise ValueError(
                "pipeline_dtype must match params_dtype when PP is enabled."
            )
        if self.optimizer.lr is None or self.optimizer.min_lr is None:
            raise ValueError("optimizer.lr and optimizer.min_lr are required.")


TrainingConfig.__init__.__doc__ = r"""Initialize large-scale SNN LLM training.

**API Language** - 中文 | English

**中文：** 组合 architecture-specific :class:`ModelConfig`、MCore optimizer、数据、
batch、训练进度、学习率和 checkpoint 策略。TP、PP、CP、sequence parallel 与
Transformer 精度只由 ``model.transformer`` 拥有；DP 由运行时 world size 推导。

**English:** Combine an architecture-specific :class:`ModelConfig` with the MCore
optimizer, data, batch, progress, learning-rate, and checkpoint policies. TP, PP,
CP, sequence parallelism, and Transformer precision have one source of truth in
``model.transformer``; DP is derived from the runtime world size.

:param model: 模型专项配置。 / Architecture-specific model configuration.
:type model: ModelConfig
:param optimizer: MCore distributed optimizer 配置。 / MCore distributed optimizer configuration.
:type optimizer: megatron.core.optimizer.OptimizerConfig
:param dataset_builder: dataset provider 完整导入路径。 / Full dataset-provider import path.
:type dataset_builder: str
:param sequence_length: token 长度 ``S``。 / Token length ``S``.
:type sequence_length: int
:param micro_batch_size: 每个 DP rank 的真实样本数。 / Real samples per DP rank and microbatch.
:type micro_batch_size: int
:param global_batch_size: 每个 optimizer step 的全局样本数，不乘 ``T``。 / Global samples per optimizer step, excluding ``T``.
:type global_batch_size: int
:param train_steps: optimizer step 数。 / Number of optimizer steps.
:type train_steps: int
:param timing_warmup_steps: 不计入性能指标的前置 optimizer steps。 / Initial
    optimizer steps excluded from performance metrics.
:type timing_warmup_steps: int
:param dataset_kwargs: dataset provider 参数。 / Dataset-provider arguments.
:type dataset_kwargs: dict[str, Any]
:param eval_interval: 验证间隔；``0`` 禁用。 / Validation interval; ``0`` disables it.
:type eval_interval: int
:param eval_steps: 每次验证的 step 数。 / Validation steps per interval.
:type eval_steps: int
:param log_interval: 指标输出间隔。 / Metric reporting interval.
:type log_interval: int
:param lr_warmup_steps: warmup optimizer steps。 / Warmup optimizer steps.
:type lr_warmup_steps: int
:param lr_decay_steps: 衰减 steps；``None`` 使用 ``train_steps``。 / Decay steps; ``None`` uses ``train_steps``.
:type lr_decay_steps: Optional[int]
:param lr_decay_style: MCore 学习率衰减样式。 / MCore learning-rate decay style.
:type lr_decay_style: str
:param checkpoint_dir: checkpoint 根目录。 / Checkpoint root.
:type checkpoint_dir: Optional[pathlib.Path]
:param checkpoint_interval: 保存间隔；``0`` 禁用。 / Save interval; ``0`` disables it.
:type checkpoint_interval: int
:param resume: 要恢复的 checkpoint。 / Checkpoint to resume.
:type resume: Optional[pathlib.Path]
:param seed: MCore 随机种子。 / MCore random seed.
:type seed: int
:param memopt_level: SpikingJelly memopt 级别。 / SpikingJelly memopt level.
:type memopt_level: int
:param memopt_checkpoint_budget: checkpoint 数量预设。 / Checkpoint-count preset.
:type memopt_checkpoint_budget: Literal["speed", "balanced", "memory"]
:raises ValueError: 配置不一致。 / If configuration values are inconsistent.
"""


__all__ = [
    "EvaluationConfig",
    "MCoreGenerationConfig",
    "ModelBuilder",
    "ModelConfig",
    "SGLangEngineConfig",
    "TrainingConfig",
]
