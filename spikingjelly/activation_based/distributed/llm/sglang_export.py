from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

from .inference import load_for_inference

if TYPE_CHECKING:
    from megatron.core.transformer import MegatronModule, TransformerConfig

_MAX_SHARD_SIZE_BYTES = 5 * 1024**3


def _json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _recipe_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _sync_error(error: Optional[BaseException], device: torch.device) -> None:
    failed = torch.tensor(int(error is not None), device=device)
    torch.distributed.all_reduce(failed)
    if failed.item():
        if error is not None:
            raise error
        raise RuntimeError("Another rank failed while exporting the SGLang artifact.")


def _copy_tokenizer(source: Optional[Path], output: Path) -> None:
    if source is None:
        return
    if not source.exists():
        raise ValueError(f"Tokenizer directory does not exist: {source}")
    if not source.is_dir():
        raise ValueError(f"Tokenizer path is not a directory: {source}")

    def ignored(_path: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name.startswith(".")
            or name
            in {
                "config.json",
                "model.safetensors.index.json",
                "spikingjelly_sglang.json",
            }
            or Path(name).suffix in {".pt", ".pth", ".safetensors"}
            or name.startswith(("pytorch_model", "tf_model", "flax_model"))
        }

    shutil.copytree(source, output, dirs_exist_ok=True, ignore=ignored)


def _write_tensor_shards(
    tensors: Iterator[tuple[str, torch.Tensor]],
    output: Path,
    prefix: str,
    max_shard_size_bytes: int,
) -> tuple[dict[str, str], int]:
    from safetensors.torch import save_file

    weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_index = 0
    parameter_count = 0

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_index
        if not shard:
            return
        shard_index += 1
        filename = f"model-{prefix}-{shard_index:05d}.safetensors"
        save_file(shard, output / filename)
        weight_map.update(dict.fromkeys(shard, filename))
        shard = {}
        shard_bytes = 0

    for name, value in tensors:
        parameter_count += value.numel()
        if name in weight_map or name in shard:
            raise ValueError(f"Duplicate exported tensor: {name}")
        value = value.detach().cpu().contiguous().clone()
        size = value.numel() * value.element_size()
        if shard and shard_bytes + size > max_shard_size_bytes:
            flush()
        shard[name] = value
        shard_bytes += size
    flush()
    return weight_map, parameter_count


class SGLangExportStage:
    def __init__(
        self,
        *,
        pipeline_rank: int,
        is_first: bool,
        is_last: bool,
        layer_offset: int,
        local_layer_count: int,
        readers: Callable[[int], Sequence[Any]],
    ) -> None:
        r"""Expose one MCore pipeline stage to a model-owned export callback.

        **API Language** - 中文 | English

        **中文：** 提供当前 PP stage 的拓扑信息，并按需读取任意 PP stage 上的
        TP safetensors shards。该对象只在 ``stage_tensors`` 回调执行期间有效。

        **English:** Provide topology metadata for the current PP stage and
        on-demand access to TP safetensors shards from any PP stage. The object
        is valid only while the ``stage_tensors`` callback is running.

        :param pipeline_rank: 当前 PP rank。 / Current PP rank.
        :type pipeline_rank: int
        :param is_first: 是否为首个 PP stage。 / Whether this is the first PP stage.
        :type is_first: bool
        :param is_last: 是否为末个 PP stage。 / Whether this is the last PP stage.
        :type is_last: bool
        :param layer_offset: 当前 stage 的全局 Transformer layer offset。 /
            Global Transformer-layer offset for this stage.
        :type layer_offset: int
        :param local_layer_count: 当前 stage 的 Transformer layer 数。 /
            Number of Transformer layers in this stage.
        :type local_layer_count: int
        :param readers: 内部 shard reader resolver。 / Internal shard-reader resolver.
        :type readers: Callable
        """
        self._pipeline_rank = pipeline_rank
        self.is_first = is_first
        self.is_last = is_last
        self.layer_offset = layer_offset
        self.local_layer_count = local_layer_count
        self._readers = readers

    def tensor_names(self) -> tuple[str, ...]:
        r"""Return source tensor names on one PP stage.

        **中文：** 返回当前 PP stage 的源权重名。
        **English:** Return source tensor names for the current PP stage.

        :return: 源权重名。 / Source tensor names.
        :rtype: tuple[str, ...]
        """
        return tuple(self._readers(self._pipeline_rank)[0].keys())

    def tensor_shards(self, name: str) -> tuple[torch.Tensor, ...]:
        r"""Read all TP shards for one source tensor.

        **中文：** 按 TP rank 顺序读取当前 PP stage 上的全部 shards。
        **English:** Read every shard on the current PP stage in TP-rank order.

        :param name: 源权重名。 / Source tensor name.
        :type name: str
        :return: 按 TP rank 排列的 tensors。 / Tensors ordered by TP rank.
        :rtype: tuple[torch.Tensor, ...]
        """
        return tuple(
            reader.get_tensor(name) for reader in self._readers(self._pipeline_rank)
        )

    def merge_tensor(
        self,
        name: str,
        dim: Optional[int] = None,
        pipeline_rank: Optional[int] = None,
    ) -> torch.Tensor:
        r"""Merge a replicated or concatenated TP tensor.

        **中文：** ``dim=None`` 时校验所有 TP shard 完全相同，否则沿 ``dim`` 拼接。
        **English:** With ``dim=None``, require identical TP shards; otherwise
        concatenate them along ``dim``.

        :param name: 源权重名。 / Source tensor name.
        :type name: str
        :param dim: 拼接维度；``None`` 表示 replicated tensor。 / Concatenation
            dimension; ``None`` denotes a replicated tensor.
        :type dim: Optional[int]
        :param pipeline_rank: 可选源 PP rank。 / Optional source PP rank.
        :type pipeline_rank: Optional[int]
        :return: 合并后的 tensor。 / Merged tensor.
        :rtype: torch.Tensor
        :raises ValueError: Replicated TP shards 不一致。 / If replicated TP shards differ.
        """
        values = (
            self.tensor_shards(name)
            if pipeline_rank is None
            else tuple(
                reader.get_tensor(name) for reader in self._readers(pipeline_rank)
            )
        )
        if dim is not None:
            return torch.cat(values, dim=dim)
        first = values[0]
        if any(not torch.equal(first, value) for value in values[1:]):
            raise ValueError(f"Replicated TP tensor differs across ranks: {name}")
        return first


def export_sglang_artifact(
    transformer_config: "TransformerConfig",
    model_provider: Callable[[bool, bool], "MegatronModule"],
    checkpoint: Path,
    output: Path,
    *,
    artifact_config: Mapping[str, Any],
    stage_tensors: Callable[[SGLangExportStage], Iterable[tuple[str, torch.Tensor]]],
    tokenizer: Optional[Path] = None,
    max_shard_size_bytes: int = _MAX_SHARD_SIZE_BYTES,
) -> None:
    r"""Export an MCore checkpoint as a topology-independent SGLang artifact.

    **API Language** - 中文 | English

    **中文：** 在源 TP/PP/CP 拓扑上加载 model shards，并为每个 PP stage 调用
    ``stage_tensors`` 完成模型专属的权重映射。框架负责分片写入、索引、manifest、
    tokenizer、跨 rank 错误同步和原子发布，不要求任意 GPU 持有完整模型。源配置
    必须使用 BF16，并且 ``expert_model_parallel_size`` 必须为 1。

    **English:** Load model shards on the source TP/PP/CP topology and invoke
    ``stage_tensors`` for model-owned weight mapping on each PP stage. The
    framework owns sharded writes, indexing, the manifest, tokenizer files,
    cross-rank error synchronization, and atomic publication without requiring
    any GPU to hold the complete model. The source configuration must use BF16
    with ``expert_model_parallel_size=1``.

    :param transformer_config: 源 MCore Transformer 配置。 / Source MCore configuration.
    :type transformer_config: megatron.core.transformer.TransformerConfig
    :param model_provider: MCore model provider。 / MCore model provider.
    :type model_provider: Callable
    :param checkpoint: MCore checkpoint 目录。 / MCore checkpoint directory.
    :type checkpoint: pathlib.Path
    :param output: 不存在的目标 artifact 目录。 / New artifact directory.
    :type output: pathlib.Path
    :param artifact_config: 写入 ``config.json`` 的模型专属配置。 / Model-owned
        configuration written to ``config.json``.
    :type artifact_config: Mapping[str, Any]
    :param stage_tensors: 将通用 stage 视图转换为目标权重的回调。 / Callback mapping
        a generic stage view to target tensors.
    :type stage_tensors: Callable
    :param tokenizer: 可选 tokenizer 目录。 / Optional tokenizer directory.
    :type tokenizer: Optional[pathlib.Path]
    :param max_shard_size_bytes: 单个目标 shard 的最大字节数。 / Maximum target
        shard size in bytes.
    :type max_shard_size_bytes: int
    :return: None.
    :rtype: None
    :raises RuntimeError: CUDA 不可用。 / If CUDA is unavailable.
    :raises ValueError: 精度、配置或拓扑无效。 / If precision, configuration, or
        topology is invalid.
    :raises FileNotFoundError: checkpoint 不存在。 / If the checkpoint is missing.
    :raises FileExistsError: output 已存在。 / If output already exists.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("SGLang artifact export requires CUDA.")
    if max_shard_size_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be positive.")
    checkpoint, output = Path(checkpoint), Path(output)
    if not transformer_config.bf16 or transformer_config.params_dtype != torch.bfloat16:
        raise ValueError("SGLang artifact export currently requires MCore bfloat16.")
    if transformer_config.expert_model_parallel_size != 1:
        raise ValueError("SGLang artifact export does not support expert parallelism.")
    architectures = artifact_config.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or not isinstance(architectures[0], str)
        or not architectures[0]
    ):
        raise ValueError("artifact_config must contain one non-empty architecture.")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    if output.exists():
        raise FileExistsError(f"Artifact output already exists: {output}")

    try:
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import (
            get_transformer_layer_offset,
        )
        from megatron.core.utils import unwrap_model
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise ImportError(
            "SGLang export requires Python 3.12 and "
            "spikingjelly[megatron,sglang-export]."
        ) from error

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    owns_distributed = not torch.distributed.is_initialized()
    if owns_distributed:
        torch.distributed.init_process_group("nccl", device_id=device)
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    temporary: Optional[Path] = None
    try:
        if owns_model_parallel:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=transformer_config.tensor_model_parallel_size,
                pipeline_model_parallel_size=transformer_config.pipeline_model_parallel_size,
                context_parallel_size=transformer_config.context_parallel_size,
                expert_model_parallel_size=1,
            )
        model_parallel_size = (
            transformer_config.tensor_model_parallel_size
            * transformer_config.pipeline_model_parallel_size
            * transformer_config.context_parallel_size
        )
        if torch.distributed.get_world_size() != model_parallel_size:
            raise ValueError("Export world size must equal source TP * PP * CP.")
        model_parallel_cuda_manual_seed(1234)
        model = load_for_inference(transformer_config, model_provider, checkpoint)
        unwrapped = unwrap_model(model)
        model_config = unwrapped.snn_model_config
        recipe = getattr(unwrapped, "checkpoint_metadata", None)
        if not isinstance(recipe, dict):
            raise ValueError("Model checkpoint_metadata must be a dictionary.")
        recipe = dict(recipe)
        if not isinstance(recipe.get("recipe_name"), str) or not recipe["recipe_name"]:
            raise ValueError(
                "Model checkpoint_metadata must declare a non-empty recipe_name."
            )
        reduction = str(unwrapped.temporal_output_reduction)

        payload = [None]
        error = None
        if torch.distributed.get_rank() == 0:
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                payload[0] = tempfile.mkdtemp(
                    prefix=f".{output.name}.", dir=output.parent
                )
            except BaseException as exception:
                error = exception
        _sync_error(error, device)
        torch.distributed.broadcast_object_list(payload, src=0)
        temporary = Path(payload[0])

        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        cp_rank = parallel_state.get_context_parallel_rank()
        local_path = temporary / f"local-pp{pp_rank:04d}-tp{tp_rank:04d}.safetensors"
        error = None
        if cp_rank == 0:
            try:
                state = {
                    name: value.detach().cpu().contiguous()
                    for name, value in unwrapped.state_dict().items()
                    if isinstance(value, torch.Tensor)
                }
                save_file(state, local_path)
            except BaseException as exception:
                error = exception
        _sync_error(error, device)
        torch.distributed.barrier()

        error = None
        if cp_rank == 0 and tp_rank == 0:
            try:
                with ExitStack() as stack:
                    opened: dict[int, tuple[Any, ...]] = {}

                    def readers(rank: int) -> tuple[Any, ...]:
                        if (
                            not 0
                            <= rank
                            < transformer_config.pipeline_model_parallel_size
                        ):
                            raise ValueError(
                                "pipeline_rank is outside the source topology."
                            )
                        if rank not in opened:
                            opened[rank] = tuple(
                                stack.enter_context(
                                    safe_open(
                                        temporary
                                        / f"local-pp{rank:04d}-tp{tensor_rank:04d}.safetensors",
                                        framework="pt",
                                        device="cpu",
                                    )
                                )
                                for tensor_rank in range(
                                    transformer_config.tensor_model_parallel_size
                                )
                            )
                        return opened[rank]

                    layer_offset = get_transformer_layer_offset(transformer_config)
                    local_layers = get_num_layers_to_build(transformer_config)
                    stage = SGLangExportStage(
                        pipeline_rank=pp_rank,
                        is_first=pp_rank == 0,
                        is_last=(
                            pp_rank
                            == transformer_config.pipeline_model_parallel_size - 1
                        ),
                        layer_offset=layer_offset,
                        local_layer_count=local_layers,
                        readers=readers,
                    )
                    weight_map, parameter_count = _write_tensor_shards(
                        iter(stage_tensors(stage)),
                        temporary,
                        f"pp{pp_rank:04d}",
                        max_shard_size_bytes,
                    )
                (temporary / f"weights-pp{pp_rank:04d}.json").write_text(
                    _json_dump(
                        {
                            "parameter_count": parameter_count,
                            "weight_map": weight_map,
                        }
                    ),
                    encoding="utf-8",
                )
            except BaseException as exception:
                error = exception
        _sync_error(error, device)
        torch.distributed.barrier()

        error = None
        if torch.distributed.get_rank() == 0:
            try:
                weight_map: dict[str, str] = {}
                parameter_count = 0
                for rank in range(transformer_config.pipeline_model_parallel_size):
                    stage = json.loads(
                        (temporary / f"weights-pp{rank:04d}.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    stage_map = stage["weight_map"]
                    duplicates = set(weight_map) & set(stage_map)
                    if duplicates:
                        raise ValueError(
                            f"Duplicate tensors across PP stages: {sorted(duplicates)}"
                        )
                    weight_map.update(stage_map)
                    parameter_count += int(stage["parameter_count"])
                index = {
                    "metadata": {
                        "parameter_count": parameter_count,
                        "total_size": sum(
                            path.stat().st_size
                            for path in temporary.glob("model-*.safetensors")
                        ),
                    },
                    "weight_map": weight_map,
                }
                (temporary / "model.safetensors.index.json").write_text(
                    _json_dump(index), encoding="utf-8"
                )
                (temporary / "config.json").write_text(
                    _json_dump(dict(artifact_config)), encoding="utf-8"
                )
                manifest = {
                    "schema_version": 2,
                    "recipe_name": recipe["recipe_name"],
                    "recipe_sha256": _recipe_digest(recipe),
                    "time_steps": model_config.time_steps,
                    "temporal_output_reduction": reduction,
                    "dtype": "bfloat16",
                    "spikingjelly_version": importlib.metadata.version("spikingjelly"),
                    "sglang_version": "0.5.17",
                }
                (temporary / "spikingjelly_sglang.json").write_text(
                    _json_dump(manifest), encoding="utf-8"
                )
                _copy_tokenizer(tokenizer, temporary)
                for path in temporary.glob("local-*.safetensors"):
                    path.unlink()
                for path in temporary.glob("weights-pp*.json"):
                    path.unlink()
                temporary.replace(output)
                temporary = None
            except BaseException as exception:
                error = exception
        _sync_error(error, device)
        torch.distributed.barrier()
    finally:
        if temporary is not None and torch.distributed.get_rank() == 0:
            shutil.rmtree(temporary, ignore_errors=True)
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


__all__ = ["SGLangExportStage", "export_sglang_artifact"]
