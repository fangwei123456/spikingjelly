from __future__ import annotations

import abc
import importlib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import torch.nn as nn


def _encode(value: Any) -> Any:
    if is_dataclass(value):
        result = {"_target_": f"{type(value).__module__}.{type(value).__qualname__}"}
        result.update(
            {item.name: _encode(getattr(value, item.name)) for item in fields(value)}
        )
        return result
    if isinstance(value, Path):
        return {"_path_": str(value)}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"_path_"}:
        return Path(value["_path_"])
    if "_target_" not in value:
        return {key: _decode(item) for key, item in value.items()}

    target_name = value["_target_"]
    config_types = _config_types()
    if (
        isinstance(target_name, str)
        and target_name not in config_types
        and target_name.startswith("spikingjelly.activation_based.distributed.vision.")
    ):
        target_name = "spikingjelly.activation_based.model." + target_name.removeprefix(
            "spikingjelly.activation_based.distributed.vision."
        )
    if isinstance(target_name, str) and target_name not in config_types:
        if not target_name.startswith("spikingjelly.activation_based.model."):
            raise ValueError(f"Unsupported config target {target_name!r}.")
        try:
            importlib.import_module(target_name.rsplit(".", 1)[0])
        except ImportError as error:
            raise ValueError(f"Unsupported config target {target_name!r}.") from error
        config_types = _config_types()
    if not isinstance(target_name, str) or target_name not in config_types:
        raise ValueError(
            f"Unsupported config target {target_name!r}; target must be a ModelConfig, "
            "PredictionConfig, or TrainingConfig subclass."
        )
    target = config_types[target_name]
    kwargs = {key: _decode(item) for key, item in value.items() if key != "_target_"}
    return target(**kwargs)


@dataclass(frozen=True)
class ModelConfig:
    r"""Serializable configuration owned by a vision-model builder.

    **API Language** - 中文 | English

    **中文：** 视觉 SNN 模型配置的基类。子类通过 ``builder`` 类变量声明可导入的
    builder，并至少定义时间步、步进模式和类别数。builder 负责模型结构及其张量并行策略。

    :param time_steps: 每个图像样本的 SNN 时间步数。
    :type time_steps: int
    :param num_classes: 分类类别数。
    :type num_classes: int
    :param step_mode: ``"s"`` （单步）或 ``"m"`` （多步）。
    :type step_mode: str
    :raises ValueError: 参数不是正整数或步进模式无效。

    **English:** Base configuration for vision SNNs. Subclasses declare an
    importable ``builder`` class variable and define at least the time-step and
    step mode, and class counts. The builder owns model construction and tensor
    parallelism.

    :param time_steps: SNN time steps for each image sample.
    :type time_steps: int
    :param num_classes: Number of classification classes.
    :type num_classes: int
    :param step_mode: ``"s"`` (single-step) or ``"m"`` (multi-step).
    :type step_mode: str
    :raises ValueError: If a numeric value is not positive or the step mode is invalid.
    """

    builder: ClassVar[str]
    time_steps: int = 4
    num_classes: int = 1000
    step_mode: Literal["s", "m"] = "m"

    def __post_init__(self) -> None:
        if self.time_steps <= 0 or self.num_classes <= 0:
            raise ValueError("time_steps and num_classes must be positive.")
        if self.step_mode not in {"s", "m"}:
            raise ValueError("step_mode must be 's' or 'm'.")

    def get_builder_cls(self) -> type[ModelBuilder]:
        r"""Resolve the declared model builder.

        **中文：** 返回 ``builder`` 导入路径指向的类。
        **English:** Return the class identified by the ``builder`` import path.

        :return: 模型 builder 类。 / The model builder class.
        :rtype: type
        :raises ImportError: builder 模块无法导入。 / If the module cannot be imported.
        :raises AttributeError: builder 类不存在。 / If the class does not exist.
        :raises TypeError: builder 未继承 :class:`ModelBuilder`。 / If the class
            does not inherit :class:`ModelBuilder`.
        """
        module_name, class_name = self.builder.rsplit(".", 1)
        builder_cls = getattr(importlib.import_module(module_name), class_name)
        if not isinstance(builder_cls, type) or not issubclass(
            builder_cls, ModelBuilder
        ):
            raise TypeError("model builder must inherit vision.ModelBuilder.")
        return builder_cls

    def as_dict(self) -> dict[str, Any]:
        r"""Serialize this model configuration.

        **中文：** 返回包含具体 config 类型的 JSON 兼容 mapping。
        **English:** Return a JSON-compatible mapping containing the concrete
        configuration type.

        :return: 可序列化配置。 / Serializable configuration.
        :rtype: dict[str, Any]
        """
        return _encode(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        r"""Restore a model configuration created by :meth:`as_dict`.

        **中文：** 按 ``_target_`` 恢复具体 model config。内置模型会自动导入；
        外部 config 类必须由调用方预先导入。
        **English:** Restore a concrete model configuration from its ``_target_``
        path. Built-in models are imported automatically; callers must pre-import
        external config classes.

        :param data: 已序列化配置。 / Serialized configuration.
        :type data: dict[str, Any]
        :return: 模型配置。 / Model configuration.
        :rtype: ModelConfig
        :raises TypeError: 结果不是 ``ModelConfig``。 / If the result is not a ModelConfig.
        """
        config = _decode(data)
        if not isinstance(config, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(config).__name__}.")
        return config


class ModelBuilder(abc.ABC):
    r"""Build and partition one vision architecture.

    **API Language** - 中文 | English

    **中文：** model config 与 architecture-specific 实现之间的 seam。实现只负责
    构建当前 PP stage、应用 TP/memopt，并返回适合原生 FSDP2 逐层分片的模块路径；
    返回模型的 forward 必须遵守 ``config.step_mode``：单步接收 ``[N, ...]``，
    多步接收 ``[T, N, ...]``。训练生命周期由
    :func:`spikingjelly.activation_based.distributed.vision.train_classification`
    负责。

    **English:** Seam between model configuration and architecture-specific
    implementation. Implementations build the current PP stage, apply TP/memopt,
    and return module paths suitable for native FSDP2. The returned model must obey
    ``config.step_mode``: single-step input is ``[N, ...]`` and multi-step input is
    ``[T, N, ...]``. The training lifecycle belongs to
    :func:`spikingjelly.activation_based.distributed.vision.train_classification`.

    ``_pipeline_stage`` 必须复用 canonical model 的参数对象。 / ``_pipeline_stage``
    must reuse the canonical model's parameter objects.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def _build_canonical_model(self) -> nn.Module:
        raise NotImplementedError(
            f"{type(self).__name__} does not support topology-independent artifacts."
        )

    def _pipeline_stage(self, model: nn.Module, rank: int, size: int) -> nn.Module:
        raise NotImplementedError(
            f"{type(self).__name__} does not support topology-independent artifacts."
        )

    def _canonical_key_map(
        self,
        pipeline_rank: int,
        pipeline_size: int,
        model: Optional[nn.Module] = None,
    ) -> dict[str, str]:
        if model is None:
            model = self._build_canonical_model()
        full_names = {
            id(value): name for name, value in model.state_dict(keep_vars=True).items()
        }
        stage = self._pipeline_stage(model, pipeline_rank, pipeline_size)
        return {
            local_name: full_names[id(value)]
            for local_name, value in stage.state_dict(keep_vars=True).items()
        }

    def _merge_tensor_parallel_shards(
        self,
        name: str,
        shards: Sequence[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        del name
        if shards[0].shape == reference.shape:
            if any(not torch.equal(shards[0], shard) for shard in shards[1:]):
                raise ValueError("Replicated tensor differs across TP ranks.")
            return shards[0]
        shard_dims = [
            dim
            for dim in range(reference.ndim)
            if reference.shape[dim] == shards[0].shape[dim] * len(shards)
            and all(
                reference.shape[other] == shards[0].shape[other]
                for other in range(reference.ndim)
                if other != dim
            )
        ]
        if len(shard_dims) != 1:
            raise ValueError(
                f"Cannot reconstruct tensor with shape {tuple(reference.shape)} "
                f"from TP shards shaped {tuple(shards[0].shape)}."
            )
        return torch.cat(tuple(shards), dim=shard_dims[0])

    def _shard_tensor_parallel_tensor(
        self,
        name: str,
        value: torch.Tensor,
        target: torch.Tensor,
        tensor_rank: int,
        tensor_size: int,
    ) -> torch.Tensor:
        del name
        if value.shape == target.shape:
            return value
        shard_dims = [
            dim
            for dim in range(value.ndim)
            if value.shape[dim] == target.shape[dim] * tensor_size
            and all(
                value.shape[other] == target.shape[other]
                for other in range(value.ndim)
                if other != dim
            )
        ]
        if len(shard_dims) != 1:
            raise ValueError(
                f"Cannot shard tensor {tuple(value.shape)} for target "
                f"{tuple(target.shape)}."
            )
        return value.chunk(tensor_size, dim=shard_dims[0])[tensor_rank].contiguous()

    def merge_state_dicts(
        self,
        shards: Sequence[tuple[int, int, Mapping[str, torch.Tensor]]],
        *,
        pipeline_size: int,
        tensor_size: int,
    ) -> dict[str, torch.Tensor]:
        r"""Merge local PP/TP states into the canonical model state.

        **中文：** 输入 ``(pp_rank, tp_rank, state_dict)`` 序列，在 CPU 上还原
        与非并行模型同名同形的 state dict。

        **English:** Merge ``(pp_rank, tp_rank, state_dict)`` entries on CPU into
        a state dict matching the unpartitioned model.

        :param shards: Local states ordered by PP and TP rank.
        :type shards: Sequence[tuple[int, int, Mapping[str, torch.Tensor]]]
        :param pipeline_size: Source PP size. / Source PP size.
        :type pipeline_size: int
        :param tensor_size: Source TP size. / Source TP size.
        :type tensor_size: int
        :return: Canonical CPU state dict. / Canonical CPU state dict.
        :rtype: dict[str, torch.Tensor]
        :raises ValueError: A rank, key, or shard shape is inconsistent.
        """
        canonical = self._build_canonical_model()
        reference = canonical.state_dict()
        key_maps = {
            rank: self._canonical_key_map(rank, pipeline_size, canonical)
            for rank in range(pipeline_size)
        }
        by_name: dict[str, list[tuple[int, torch.Tensor]]] = {}
        for pipeline_rank, tensor_rank, state in shards:
            if (
                not 0 <= pipeline_rank < pipeline_size
                or not 0 <= tensor_rank < tensor_size
            ):
                raise ValueError("Invalid PP or TP rank in state shards.")
            key_map = key_maps[pipeline_rank]
            if set(state) != set(key_map):
                raise ValueError("Local state keys do not match the pipeline stage.")
            for local_name, value in state.items():
                by_name.setdefault(key_map[local_name], []).append(
                    (tensor_rank, value.detach().cpu())
                )

        result = {}
        if set(by_name) != set(reference):
            raise ValueError("Pipeline shards do not cover the canonical model state.")
        for name, entries in by_name.items():
            entries.sort(key=lambda item: item[0])
            if [rank for rank, _ in entries] != list(range(tensor_size)):
                raise ValueError(f"Tensor {name!r} does not contain every TP rank.")
            result[name] = self._merge_tensor_parallel_shards(
                name, [value for _, value in entries], reference[name]
            )
        return result

    def build_for_inference(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        process_group: Optional[ProcessGroup],
        pipeline_rank: int,
        pipeline_size: int,
        pipeline_microbatches: int,
        device: torch.device,
        micro_batch_size: int,
    ) -> tuple[
        nn.Module,
        tuple[str, ...],
        Optional[tuple[int, ...]],
        Optional[tuple[int, ...]],
    ]:
        r"""Build a target PP/TP stage from a canonical state dict.

        **中文：** 先构建目标拓扑的本地 stage，再按当前 TP rank 从
        canonical state 取出对应 shard。

        **English:** Build the local stage for the target topology and select its
        TP shards from the canonical state.

        :param state_dict: Canonical model state. / Canonical model state.
        :type state_dict: Mapping[str, torch.Tensor]
        :param process_group: TP 进程组；TP=1 时为 ``None``。 / TP process
            group; ``None`` when TP=1.
        :type process_group: Optional[ProcessGroup]
        :param pipeline_rank: 当前 PP rank。 / Current PP rank.
        :type pipeline_rank: int
        :param pipeline_size: PP rank 数。 / Number of PP ranks.
        :type pipeline_size: int
        :param pipeline_microbatches: 每个本地 batch 的 pipeline microbatch 数。 /
            Pipeline microbatches per local batch.
        :type pipeline_microbatches: int
        :param device: 当前 rank 的设备。 / Device for the current rank.
        :type device: torch.device
        :param micro_batch_size: 当前 DP rank 的图像 batch size。 / Image batch
            size on the current DP rank.
        :type micro_batch_size: int
        :return: The same tuple as :meth:`build`. / Same tuple as :meth:`build`.
        :rtype: tuple
        :raises ValueError: artifact state 或目标 TP/PP shard 无效。 / If the
            artifact state or target TP/PP shard is invalid.
        """
        built = self.build(
            process_group=process_group,
            memopt_process_group=None,
            pipeline_rank=pipeline_rank,
            pipeline_size=pipeline_size,
            pipeline_microbatches=pipeline_microbatches,
            device=device,
            micro_batch_size=micro_batch_size,
            memopt_level=0,
            memopt_compress_inputs=False,
            memopt_checkpoint_budget="memory",
        )
        model = built[0]
        local_state = model.state_dict()
        canonical = self._build_canonical_model()
        key_map = self._canonical_key_map(pipeline_rank, pipeline_size, canonical)
        if set(state_dict) != set(canonical.state_dict()):
            raise ValueError("Artifact state does not match the configured model.")
        tensor_rank = dist.get_rank(process_group) if process_group is not None else 0
        tensor_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        model.load_state_dict(
            {
                local_name: self._shard_tensor_parallel_tensor(
                    canonical_name,
                    state_dict[canonical_name],
                    target,
                    tensor_rank,
                    tensor_size,
                )
                for local_name, target in local_state.items()
                for canonical_name in (key_map[local_name],)
            }
        )
        return built

    @abc.abstractmethod
    def build(
        self,
        *,
        process_group: Optional[ProcessGroup],
        memopt_process_group: Optional[ProcessGroup],
        pipeline_rank: int,
        pipeline_size: int,
        pipeline_microbatches: int,
        device: torch.device,
        micro_batch_size: int,
        memopt_level: int,
        memopt_compress_inputs: bool,
        memopt_checkpoint_budget: Literal["speed", "balanced", "memory"],
    ) -> tuple[
        nn.Module,
        tuple[str, ...],
        Optional[tuple[int, ...]],
        Optional[tuple[int, ...]],
    ]:
        r"""Build the local model shard.

        **中文：** 返回当前 PP stage（已应用 architecture-specific TP 和 memopt）、
        按由内到外顺序排列的 FSDP2 module 路径，以及 PP 边界张量形状。
        **English:** Return the current PP stage after architecture-specific TP and
        memopt, FSDP2 module paths ordered from inner to outer, and PP boundary shapes.

        :param process_group: TP 进程组；TP=1 时为 ``None``。
        :type process_group: Optional[ProcessGroup]
        :param memopt_process_group: 当前 PP stage 的 DP x TP 进程组。
        :type memopt_process_group: Optional[ProcessGroup]
        :param pipeline_rank: 当前 PP rank。 / Current PP rank.
        :type pipeline_rank: int
        :param pipeline_size: PP rank 数。 / Number of PP ranks.
        :type pipeline_size: int
        :param pipeline_microbatches: 每个本地 batch 的 pipeline microbatch 数。 /
            Pipeline microbatches per local batch.
        :type pipeline_microbatches: int
        :param device: 当前 rank 的设备。
        :type device: torch.device
        :param micro_batch_size: 当前 DP rank 的图像 batch size。
        :type micro_batch_size: int
        :param memopt_level: SpikingJelly memopt level。
        :type memopt_level: int
        :param memopt_compress_inputs: 是否压缩 checkpoint 输入。
        :type memopt_compress_inputs: bool
        :param memopt_checkpoint_budget: checkpoint 数量预设。
        :type memopt_checkpoint_budget: Literal["speed", "balanced", "memory"]
        :return: model、FSDP2 roots 以及 PP input/output shapes。 / Model, FSDP2
            roots, and PP input/output shapes.
        :rtype: tuple
        """
        raise NotImplementedError


@dataclass(frozen=True)
class PredictionConfig:
    r"""Configure distributed vision classification prediction.

    **API Language** - 中文 | English

    **中文：** 从 topology-independent artifact 构建视觉 SNN，并使用
    PyTorch replicated DP、FSDP2、architecture-specific TP 与 PP 执行离线
    预测。dataset builder 必须返回一个 ``Dataset``；元素可以是 image 或
    ``(image, target)``，其中 target 被忽略。PP 只支持 multi-step artifact。

    **English:** Build a vision SNN from a topology-independent artifact and run
    offline prediction with PyTorch replicated DP, FSDP2, architecture-specific
    TP, and PP. The dataset builder must return one ``Dataset`` whose items may
    be images or ``(image, target)``; targets are ignored. PP requires a
    multi-step artifact.

    :param artifact: :func:`export_inference_artifact` 生成的 artifact。 /
        Artifact created by :func:`export_inference_artifact`.
    :type artifact: pathlib.Path
    :param dataset_builder: 返回一个 Dataset 的完整导入路径。 / Full
        import path returning one Dataset.
    :type dataset_builder: str
    :param dataset_kwargs: dataset builder 参数。 / Dataset-builder kwargs.
    :type dataset_kwargs: dict[str, Any]
    :param input_layout: ``"NCHW"`` 或 ``"NTCHW"``。 / DataLoader input layout.
    :type input_layout: str
    :param batch_size: 每个 DP rank 的 batch size。 / Batch size per DP rank.
    :type batch_size: int
    :param workers: 每个 DataLoader 的 worker 数。 / DataLoader workers.
    :type workers: int
    :param tensor_parallel_size: TP rank 数。 / Number of TP ranks.
    :type tensor_parallel_size: int
    :param pipeline_parallel_size: PP rank 数。 / Number of PP ranks.
    :type pipeline_parallel_size: int
    :param pipeline_microbatches: 每个 batch 的 pipeline microbatch 数。 /
        Pipeline microbatches per batch.
    :type pipeline_microbatches: int
    :param data_parallel: ``"replicate"`` 或 ``"fsdp2"``。 / Replicated or
        FSDP2 data parallelism.
    :type data_parallel: str
    :param precision: ``"fp32"``、``"bf16"`` 或 ``"fp16"``。 / Arithmetic precision.
    :type precision: str
    :param compile: 是否使用 ``torch.compile``；当前仅支持无 PP 的
        replicated 模式。 / Whether to use ``torch.compile``; currently limited
        to replicated execution without PP.
    :type compile: bool
    :param seed: 模型与数据随机种子。 / Model and data seed.
    :type seed: int
    :raises ValueError: 配置值或导入路径无效。 / If a value or import path is invalid.
    """

    artifact: Path
    dataset_builder: str
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    input_layout: Literal["NCHW", "NTCHW"] = "NCHW"
    batch_size: int = 32
    workers: int = 4
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    pipeline_microbatches: int = 1
    data_parallel: Literal["replicate", "fsdp2"] = "replicate"
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    compile: bool = False
    seed: int = 1234

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "pipeline_microbatches",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.batch_size % self.pipeline_microbatches:
            raise ValueError("batch_size must be divisible by pipeline_microbatches.")
        if self.workers < 0:
            raise ValueError("workers cannot be negative.")
        if self.input_layout not in {"NCHW", "NTCHW"}:
            raise ValueError("input_layout must be 'NCHW' or 'NTCHW'.")
        if self.data_parallel not in {"replicate", "fsdp2"}:
            raise ValueError("data_parallel must be 'replicate' or 'fsdp2'.")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be 'fp32', 'bf16', or 'fp16'.")
        if self.pipeline_parallel_size > 1 and self.precision == "fp16":
            raise ValueError("Vision PP currently supports fp32 and bf16.")
        if self.compile and (
            self.pipeline_parallel_size > 1 or self.data_parallel != "replicate"
        ):
            raise ValueError("compile requires replicated execution without PP.")
        if "." not in self.dataset_builder:
            raise ValueError("dataset_builder must be a full import path.")


@dataclass(frozen=True)
class EvaluationConfig(PredictionConfig):
    loss_function: str = "torch.nn.functional.cross_entropy"
    loss_kwargs: dict[str, Any] = field(default_factory=dict)
    timing_warmup_batches: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if "." not in self.loss_function:
            raise ValueError("loss_function must be a full import path.")
        if self.timing_warmup_batches < 0:
            raise ValueError("timing_warmup_batches cannot be negative.")


EvaluationConfig.__init__.__doc__ = r"""Configure distributed vision classification evaluation.

**API Language** - 中文 | English

**中文：** 从 topology-independent artifact 构建视觉 SNN，并使用 PyTorch
replicated DP、FSDP2、architecture-specific TP 与 PP 评测分类 dataset。
每个元素必须是 ``(image, target)``；返回全局 loss、accuracy 与性能指标。

**English:** Build a vision SNN from a topology-independent artifact and evaluate
a classification dataset with PyTorch replicated DP, FSDP2, architecture-specific
TP, and PP. Every item must be ``(image, target)``. The result contains global
loss, accuracy, and performance metrics.

:param artifact: :func:`export_inference_artifact` 生成的 artifact。 / Artifact
    created by :func:`export_inference_artifact`.
:type artifact: pathlib.Path
:param dataset_builder: 返回一个 Dataset 的完整导入路径。 / Full import path
    returning one Dataset.
:type dataset_builder: str
:param dataset_kwargs: dataset builder 参数。 / Dataset-builder arguments.
:type dataset_kwargs: dict[str, Any]
:param input_layout: ``"NCHW"`` 或 ``"NTCHW"``。 / DataLoader input layout.
:type input_layout: str
:param batch_size: 每个 DP rank 的 batch size。 / Batch size per DP rank.
:type batch_size: int
:param workers: 每个 DataLoader 的 worker 数。 / DataLoader workers.
:type workers: int
:param tensor_parallel_size: TP rank 数。 / Number of TP ranks.
:type tensor_parallel_size: int
:param pipeline_parallel_size: PP rank 数。 / Number of PP ranks.
:type pipeline_parallel_size: int
:param pipeline_microbatches: 每个 batch 的 pipeline microbatch 数。 / Pipeline
    microbatches per batch.
:type pipeline_microbatches: int
:param data_parallel: ``"replicate"`` 或 ``"fsdp2"``。 / Replicated or FSDP2
    data parallelism.
:type data_parallel: str
:param precision: ``"fp32"``、``"bf16"`` 或 ``"fp16"``。 / Arithmetic precision.
:type precision: str
:param compile: 是否使用 ``torch.compile``；当前仅支持无 PP 的 replicated 模式。 /
    Whether to use ``torch.compile``; currently limited to replicated execution
    without PP.
:type compile: bool
:param seed: 模型与数据随机种子。 / Model and data seed.
:type seed: int
:param loss_function: 分类 loss 函数的完整导入路径。 / Full loss-function
    import path.
:type loss_function: str
:param loss_kwargs: loss 参数。 / Loss-function arguments.
:type loss_kwargs: dict[str, Any]
:param timing_warmup_batches: 计时前执行且不计入指标的 batch 数。 / Batches
    executed before timing and excluded from metrics.
:type timing_warmup_batches: int
:raises ValueError: 配置值或导入路径无效。 / If a value or import path is invalid.
"""


@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig
    dataset_builder: str
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    input_layout: Literal["NCHW", "NTCHW"] = "NCHW"
    epochs: int = 1
    batch_size: int = 32
    workers: int = 4
    optimizer: str = "torch.optim.AdamW"
    optimizer_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "weight_decay": 0.0}
    )
    loss_function: str = "torch.nn.functional.cross_entropy"
    loss_kwargs: dict[str, Any] = field(default_factory=dict)
    mixup_alpha: float = 0.0
    scheduler: Optional[str] = None
    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    pipeline_microbatches: int = 1
    data_parallel: Literal["ddp", "fsdp2"] = "ddp"
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    memopt_level: int = 0
    memopt_compress_inputs: bool = True
    memopt_checkpoint_budget: Literal["speed", "balanced", "memory"] = "memory"
    max_steps: Optional[int] = None
    timing_warmup_steps: int = 0
    checkpoint_dir: Optional[Path] = None
    checkpoint_interval: int = 0
    resume: Optional[Path] = None
    seed: int = 1234

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "batch_size",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "pipeline_microbatches",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.batch_size % self.pipeline_microbatches:
            raise ValueError("batch_size must be divisible by pipeline_microbatches.")
        if self.workers < 0:
            raise ValueError("workers cannot be negative.")
        if self.input_layout not in {"NCHW", "NTCHW"}:
            raise ValueError("input_layout must be 'NCHW' or 'NTCHW'.")
        if self.mixup_alpha < 0:
            raise ValueError("mixup_alpha cannot be negative.")
        if self.data_parallel not in {"ddp", "fsdp2"}:
            raise ValueError("data_parallel must be 'ddp' or 'fsdp2'.")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be 'fp32', 'bf16', or 'fp16'.")
        if self.pipeline_parallel_size > 1 and self.precision == "fp16":
            raise ValueError("Vision PP currently supports fp32 and bf16.")
        if self.model.step_mode == "s" and self.pipeline_parallel_size > 1:
            raise ValueError("Vision PP currently requires step_mode='m'.")
        if not 0 <= self.memopt_level <= 4:
            raise ValueError("memopt_level must lie in [0, 4].")
        if self.memopt_checkpoint_budget not in {"speed", "balanced", "memory"}:
            raise ValueError(
                "memopt_checkpoint_budget must be 'speed', 'balanced', or 'memory'."
            )
        if self.model.step_mode == "s" and self.memopt_level:
            raise ValueError("Vision memopt currently requires step_mode='m'.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when set.")
        if self.timing_warmup_steps < 0:
            raise ValueError("timing_warmup_steps cannot be negative.")
        if self.max_steps is not None and self.timing_warmup_steps >= self.max_steps:
            raise ValueError("timing_warmup_steps must be smaller than max_steps.")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval cannot be negative.")
        if self.checkpoint_interval and self.checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir is required when checkpoint_interval is positive."
            )
        for name, value in (
            ("dataset_builder", self.dataset_builder),
            ("optimizer", self.optimizer),
            ("loss_function", self.loss_function),
        ):
            if "." not in value:
                raise ValueError(f"{name} must be a full import path.")
        if self.scheduler is not None and "." not in self.scheduler:
            raise ValueError("scheduler must be a full import path.")

    def as_dict(self) -> dict[str, Any]:
        r"""Serialize this configuration to a JSON-compatible dictionary.

        **中文：** 返回包含 config 与 builder 类型信息的 JSON 兼容字典。
        **English:** Return a JSON-compatible dictionary including config and
        builder type metadata.

        :return: 可序列化配置。 / Serializable configuration.
        :rtype: dict[str, Any]
        """
        return _encode(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        r"""Restore a configuration created by :meth:`as_dict`.

        **中文：** 从 ``as_dict`` 结果按 ``_target_`` 恢复具体 config 子类。内置模型
        会自动导入；外部 config 类必须预先导入；不可用或非 config 类型会被拒绝。
        **English:** Restore concrete config subclasses from the ``_target_`` path
        produced by ``as_dict``. Built-in models are imported automatically;
        external config classes must already be imported; unavailable or non-config
        targets are rejected.

        :param data: 已序列化配置。 / Serialized configuration.
        :type data: dict[str, Any]
        :return: 训练配置。 / Training configuration.
        :rtype: TrainingConfig
        :raises TypeError: 反序列化结果不是 ``TrainingConfig``。
        """
        config = _decode(data)
        if not isinstance(config, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(config).__name__}.")
        return config


def _config_types() -> dict[str, type]:
    pending = [ModelConfig, PredictionConfig, TrainingConfig]
    result = {}
    while pending:
        config_type = pending.pop()
        result[f"{config_type.__module__}.{config_type.__qualname__}"] = config_type
        pending.extend(config_type.__subclasses__())
    return result


TrainingConfig.__init__.__doc__ = r"""Initialize distributed vision training.

**API Language** - 中文 | English

**中文：** 配置图像数据、loss、优化器、训练进度、DP/FSDP2、通道 TP、PP、
精度、SpikingJelly memopt 与 checkpoint。全局 batch 为 ``batch_size * DP``，
不乘 TP 或时间步。dataset builder 接收 ``dataset_kwargs`` 并返回
train/validation Dataset。

**English:** Configure image data, loss, optimization, progress, DP/FSDP2, channel
TP, PP, precision, SpikingJelly memopt, and checkpoints. Global batch size is
``batch_size * DP`` and does not include TP or SNN time steps. The dataset builder
receives ``dataset_kwargs`` and returns train and validation datasets.

:param model: 包含时间步和步进模式的模型专项配置。 / Architecture-specific model
    configuration including time steps and step mode.
:type model: ModelConfig
:param dataset_builder: 返回 train/validation Dataset 的完整导入路径。 / Full
    import path returning train and validation datasets.
:type dataset_builder: str
:param dataset_kwargs: 传给 dataset builder 的关键字参数。 / Dataset-builder kwargs.
:type dataset_kwargs: dict[str, Any]
:param input_layout: DataLoader batch 的输入布局；``"NCHW"`` 表示静态图像，
    ``"NTCHW"`` 表示 batch-first 时间序列。 / DataLoader batch layout;
    ``"NCHW"`` denotes static images and ``"NTCHW"`` denotes batch-first
    temporal sequences.
:type input_layout: str
:param epochs: 最大训练 epoch 数。 / Maximum training epochs.
:type epochs: int
:param batch_size: 每个 DP rank 的图像 batch size。 / Image batch size per DP rank.
:type batch_size: int
:param workers: 每个 DataLoader 的 worker 数。 / DataLoader workers.
:type workers: int
:param optimizer: optimizer 类的完整导入路径。 / Full optimizer-class import path.
:type optimizer: str
:param optimizer_kwargs: optimizer 关键字参数。 / Optimizer kwargs.
:type optimizer_kwargs: dict[str, Any]
:param loss_function: 接收 ``(logits, targets)`` 并返回 batch-mean 标量张量的
    loss 函数完整导入路径。 / Full import path of a loss function that accepts
    ``(logits, targets)`` and returns a batch-mean scalar tensor.
:type loss_function: str
:param loss_kwargs: 每次调用 loss 函数时传入的关键字参数。 / Keyword arguments
    passed to every loss-function call.
:type loss_kwargs: dict[str, Any]
:param mixup_alpha: batch-level mixup 的 beta 分布参数；``0`` 禁用。 / Beta
    distribution parameter for batch-level mixup; ``0`` disables mixup.
:type mixup_alpha: float
:param scheduler: scheduler 类的完整导入路径；``None`` 禁用。 / Full scheduler
    import path; ``None`` disables scheduling.
:type scheduler: Optional[str]
:param scheduler_kwargs: scheduler 关键字参数。 / Scheduler kwargs.
:type scheduler_kwargs: dict[str, Any]
:param tensor_parallel_size: architecture-specific TP rank 数。 / Number of
    architecture-specific TP ranks.
:type tensor_parallel_size: int
:param pipeline_parallel_size: architecture-specific PP rank 数。 / Number of
    architecture-specific PP ranks.
:type pipeline_parallel_size: int
:param pipeline_microbatches: 每个 batch 的 pipeline microbatch 数。 / Pipeline
    microbatches per batch.
:type pipeline_microbatches: int
:param data_parallel: ``"ddp"`` 或 ``"fsdp2"``。 / ``"ddp"`` or ``"fsdp2"``.
:type data_parallel: str
:param precision: ``"fp32"``、``"bf16"`` 或 ``"fp16"``。 / Arithmetic precision.
:type precision: str
:param memopt_level: SpikingJelly memopt level，范围 ``[0, 4]``。 / SpikingJelly
    memopt level in ``[0, 4]``.
:type memopt_level: int
:param memopt_compress_inputs: 是否压缩 checkpoint 输入。 / Whether to compress
    checkpoint inputs.
:type memopt_compress_inputs: bool
:param memopt_checkpoint_budget: ``"speed"``、``"balanced"`` 或 ``"memory"``
    checkpoint 数量预设。 / ``"speed"``, ``"balanced"``, or ``"memory"``
    checkpoint-count preset.
:type memopt_checkpoint_budget: Literal["speed", "balanced", "memory"]
:param max_steps: 可选的 optimizer-step 上限。 / Optional optimizer-step limit.
:type max_steps: Optional[int]
:param timing_warmup_steps: 不计入性能指标的前置 optimizer steps。 / Initial
    optimizer steps excluded from performance metrics.
:type timing_warmup_steps: int
:param checkpoint_dir: checkpoint 根目录。 / Checkpoint root.
:type checkpoint_dir: Optional[pathlib.Path]
:param checkpoint_interval: optimizer-step 保存间隔；``0`` 禁用。 / Save interval;
    ``0`` disables saving.
:type checkpoint_interval: int
:param resume: 要恢复的 checkpoint 目录。 / Checkpoint directory to resume.
:type resume: Optional[pathlib.Path]
:param seed: 数据 sampler 与模型初始化种子。 / Model and sampler seed.
:type seed: int
:raises ValueError: 配置值、导入路径或 checkpoint 组合无效。 / If a value,
    import path, or checkpoint combination is invalid.
"""
