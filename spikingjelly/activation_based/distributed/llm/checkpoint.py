"""Megatron Core checkpoint integration for SNN language models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from megatron.core.optimizer import MegatronOptimizer
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.transformer import MegatronModule


def _state(model: Any, optimizer: Any, scheduler: Any, is_loading: bool) -> dict:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.dist_checkpointing.mapping import ShardedObject
    from megatron.core.utils import unwrap_model

    metadata = {
        "distrib_optim_sharding_type": "dp_reshardable",
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
        "dp_cp_group": parallel_state.get_data_parallel_group(
            with_context_parallel=True
        ),
    }
    model_state = unwrap_model(model).sharded_state_dict(metadata=metadata)
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    cp_rank = parallel_state.get_context_parallel_rank()
    dp_rank = parallel_state.get_data_parallel_rank()
    rng = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "tracker": tensor_parallel.get_cuda_rng_tracker().get_states(),
    }
    return {
        "model": model_state,
        "optimizer": optimizer.sharded_state_dict(
            model_state, is_loading=is_loading, metadata=metadata
        ),
        "scheduler": scheduler.state_dict(),
        "rng": ShardedObject(
            "rng",
            rng,
            (
                parallel_state.get_pipeline_model_parallel_world_size(),
                parallel_state.get_tensor_model_parallel_world_size(),
                parallel_state.get_context_parallel_world_size(),
                parallel_state.get_data_parallel_world_size(),
            ),
            (pp_rank, tp_rank, cp_rank, dp_rank),
        ),
    }


def save_checkpoint(
    checkpoint_dir: Path,
    model: "MegatronModule",
    optimizer: "MegatronOptimizer",
    scheduler: "OptimizerParamScheduler",
    optimizer_step: int,
    consumed_samples: int,
    recipe: dict[str, Any],
) -> None:
    r"""Save an optimizer-boundary MCore distributed checkpoint.

    **API Language** - :ref:`中文 <save-snn-checkpoint-cn>` | :ref:`English <save-snn-checkpoint-en>`

    ----

    .. _save-snn-checkpoint-cn:

    * **中文**

    保存 MCore sharded model、distributed optimizer、scheduler、RNG、训练进度和
    recipe 元数据。``MemoryModule`` 的临时膜电位不属于 ``state_dict``，因此不会保存。
    目标目录必须为空，避免覆盖已完成的 checkpoint。

    :param checkpoint_dir: 当前 optimizer step 的新 checkpoint 目录。
    :type checkpoint_dir: pathlib.Path
    :param model: MCore DDP 包装后的模型。
    :type model: megatron.core.transformer.MegatronModule
    :param optimizer: MCore distributed optimizer。
    :type optimizer: megatron.core.optimizer.MegatronOptimizer
    :param scheduler: MCore optimizer scheduler。
    :type scheduler: megatron.core.optimizer_param_scheduler.OptimizerParamScheduler
    :param optimizer_step: 已完成的 optimizer step。
    :type optimizer_step: int
    :param consumed_samples: 已消费的真实样本数，不乘 ``T``。
    :type consumed_samples: int
    :param recipe: 可序列化的 recipe 名称、模型配置和时间语义元数据。
    :type recipe: dict[str, Any]
    :raises FileExistsError: 目标目录非空。
    :raises OSError: rank 0 无法检查或创建目标目录。
    :raises RuntimeError: 其他 rank 收到 rank 0 的目录设置失败状态。

    ----

    .. _save-snn-checkpoint-en:

    * **English**

    Saves MCore sharded model, distributed optimizer, scheduler, RNG, progress,
    and recipe metadata. Ephemeral ``MemoryModule`` membrane state is absent from
    ``state_dict`` and therefore excluded. The destination must be empty to avoid
    overwriting a completed checkpoint.

    :param checkpoint_dir: New checkpoint directory for the current optimizer step.
    :type checkpoint_dir: pathlib.Path
    :param model: Model wrapped by MCore DDP.
    :type model: megatron.core.transformer.MegatronModule
    :param optimizer: MCore distributed optimizer.
    :type optimizer: megatron.core.optimizer.MegatronOptimizer
    :param scheduler: MCore optimizer scheduler.
    :type scheduler: megatron.core.optimizer_param_scheduler.OptimizerParamScheduler
    :param optimizer_step: Completed optimizer step.
    :type optimizer_step: int
    :param consumed_samples: Consumed real samples, excluding ``T``.
    :type consumed_samples: int
    :param recipe: Serializable recipe name, model configuration, and temporal metadata.
    :type recipe: dict[str, Any]
    :raises FileExistsError: If the destination directory is non-empty.
    :raises OSError: If rank 0 cannot inspect or create the destination.
    :raises RuntimeError: If another rank receives rank 0's setup failure status.
    """
    from megatron.core import dist_checkpointing

    checkpoint_dir = Path(checkpoint_dir)
    setup_status = torch.zeros(
        (), dtype=torch.uint8, device=torch.cuda.current_device()
    )
    setup_error = None
    if torch.distributed.get_rank() == 0:
        try:
            if (
                checkpoint_dir.exists()
                and next(checkpoint_dir.iterdir(), None) is not None
            ):
                setup_status.fill_(1)
            else:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            setup_status.fill_(2)
            setup_error = error
    torch.distributed.broadcast(setup_status, src=0)
    if setup_status.item() == 1:
        raise FileExistsError(f"Checkpoint directory is not empty: {checkpoint_dir}")
    if setup_status.item() == 2:
        if setup_error is not None:
            raise setup_error
        raise RuntimeError(
            f"Rank 0 could not prepare checkpoint directory: {checkpoint_dir}"
        )
    torch.distributed.barrier()

    state = _state(model, optimizer, scheduler, is_loading=False)
    state["progress"] = {
        "optimizer_step": optimizer_step,
        "consumed_samples": consumed_samples,
    }
    state["recipe"] = recipe
    dist_checkpointing.save(state, str(checkpoint_dir))


def load_checkpoint(
    checkpoint_dir: Path,
    model: "MegatronModule",
    optimizer: "MegatronOptimizer",
    scheduler: "OptimizerParamScheduler",
) -> dict[str, Any]:
    r"""Load an MCore distributed checkpoint into an initialized training stack.

    **API Language** - :ref:`中文 <load-snn-checkpoint-cn>` | :ref:`English <load-snn-checkpoint-en>`

    ----

    .. _load-snn-checkpoint-cn:

    * **中文**

    按当前 TP/PP/CP/DP 拓扑生成 sharded state 模板并恢复 model、distributed optimizer、
    scheduler 和 RNG。

    :param checkpoint_dir: 已完成的 checkpoint 目录。
    :type checkpoint_dir: pathlib.Path
    :param model: MCore DDP 包装后的模型。
    :type model: megatron.core.transformer.MegatronModule
    :param optimizer: MCore distributed optimizer。
    :type optimizer: megatron.core.optimizer.MegatronOptimizer
    :param scheduler: MCore optimizer scheduler。
    :type scheduler: megatron.core.optimizer_param_scheduler.OptimizerParamScheduler
    :return: ``progress`` 与 ``recipe`` 元数据。
    :rtype: dict[str, Any]
    :raises FileNotFoundError: checkpoint 目录不存在。

    ----

    .. _load-snn-checkpoint-en:

    * **English**

    Builds sharded-state templates for the current TP/PP/CP/DP topology and restores
    model, distributed optimizer, scheduler, and RNG state.

    :param checkpoint_dir: Completed checkpoint directory.
    :type checkpoint_dir: pathlib.Path
    :param model: Model wrapped by MCore DDP.
    :type model: megatron.core.transformer.MegatronModule
    :param optimizer: MCore distributed optimizer.
    :type optimizer: megatron.core.optimizer.MegatronOptimizer
    :param scheduler: MCore optimizer scheduler.
    :type scheduler: megatron.core.optimizer_param_scheduler.OptimizerParamScheduler
    :return: ``progress`` and ``recipe`` metadata.
    :rtype: dict[str, Any]
    :raises FileNotFoundError: If the checkpoint directory does not exist.
    """
    from megatron.core import dist_checkpointing
    from megatron.core.utils import unwrap_model

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )
    state = dist_checkpointing.load(
        _state(model, optimizer, scheduler, is_loading=True), str(checkpoint_dir)
    )
    unwrap_model(model).load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"]["torch"])
    torch.cuda.set_rng_state(state["rng"]["cuda"])
    from megatron.core import tensor_parallel

    tensor_parallel.get_cuda_rng_tracker().set_states(state["rng"]["tracker"])
    return {"progress": state["progress"], "recipe": state["recipe"]}
