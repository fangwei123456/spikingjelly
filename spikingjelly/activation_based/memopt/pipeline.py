import copy
import math
from collections.abc import Callable, Sequence
from typing import Literal, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.utils._pytree import tree_flatten

from spikingjelly.logger import logger

from .. import base
from .checkpointing import (
    _checkpoint_options,
    _unwrap_checkpoint_module,
    checkpoint_module,
)
from .compress import BitSpikeCompressor

__all__ = ["optimize_memory"]


_BUDGET_RATIOS = {"speed": 0.5, "balanced": 0.75, "memory": 1.0}


def _clone_value(value):
    return (
        value.detach().clone()
        if isinstance(value, torch.Tensor)
        else copy.deepcopy(value)
    )


def _capture_runtime(model: nn.Module):
    return {
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "buffers": [(buffer, buffer.detach().clone()) for buffer in model.buffers()],
        "memories": [_clone_value(value) for value in base.extract_memories(model)],
        "grads": [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()
        ],
    }


def _restore_runtime(model: nn.Module, state) -> None:
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], torch.cuda.current_device())
    with torch.no_grad():
        for buffer, saved in state["buffers"]:
            buffer.copy_(saved)
    base.load_memories(model, [_clone_value(value) for value in state["memories"]])
    for parameter, grad in zip(model.parameters(), state["grads"]):
        parameter.grad = None if grad is None else grad.detach().clone()


def _backward_outputs(outputs: object) -> None:
    leaves, _ = tree_flatten(outputs)
    tensors = [
        leaf
        for leaf in leaves
        if isinstance(leaf, torch.Tensor)
        and leaf.is_floating_point()
        and leaf.requires_grad
    ]
    if not tensors:
        raise ValueError("example_forward must return a differentiable tensor leaf.")
    torch.stack([tensor.float().sum() for tensor in tensors]).sum().backward()


def _group_device(process_group: Optional[ProcessGroup]) -> torch.device:
    if process_group is not None and dist.get_backend(process_group) == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _group_max(value: float, process_group: Optional[ProcessGroup]) -> float:
    if process_group is None:
        return value
    tensor = torch.tensor(
        value, dtype=torch.float64, device=_group_device(process_group)
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=process_group)
    return float(tensor.item())


def _group_values(
    values: list[float], op: dist.ReduceOp, process_group: Optional[ProcessGroup]
) -> list[float]:
    if process_group is None or not values:
        return values
    tensor = torch.tensor(
        values, dtype=torch.float64, device=_group_device(process_group)
    )
    dist.all_reduce(tensor, op=op, group=process_group)
    return tensor.cpu().tolist()


def _run_trial(function: Callable[[], float], process_group: Optional[ProcessGroup]):
    error = None
    status = 0
    value = 0.0
    try:
        value = function()
    except torch.cuda.OutOfMemoryError as exc:
        status, error = 1, exc
        torch.cuda.empty_cache()
    except Exception as exc:
        status, error = 2, exc

    if process_group is not None:
        status_tensor = torch.tensor(
            status, dtype=torch.int64, device=_group_device(process_group)
        )
        dist.all_reduce(status_tensor, op=dist.ReduceOp.MAX, group=process_group)
        group_status = int(status_tensor.item())
    else:
        group_status = status

    if group_status == 2:
        if error is not None and status == 2:
            raise error
        raise RuntimeError("A process-group rank failed while profiling memopt.")
    if group_status == 1:
        return None
    return _group_max(value, process_group)


def _execute_example(
    model: nn.Module,
    example_forward: Callable[[nn.Module], object],
    *,
    backward: bool,
) -> object:
    for parameter in model.parameters():
        parameter.grad = None
    outputs = example_forward(model)
    if backward:
        _backward_outputs(outputs)
    return outputs


def _measure_peak(
    model: nn.Module,
    example_forward: Callable[[nn.Module], object],
) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("memopt levels 2-4 require CUDA memory profiling.")
    tensor = next(iter(model.parameters()), None)
    if tensor is None:
        tensor = next(iter(model.buffers()), None)
    if tensor is not None and tensor.device.type != "cuda":
        raise RuntimeError("memopt levels 2-4 require the model on CUDA.")
    state = _capture_runtime(model)
    try:
        _execute_example(model, example_forward, backward=True)
        _restore_runtime(model, state)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _execute_example(model, example_forward, backward=True)
        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated())
    finally:
        _restore_runtime(model, state)


def _module_path_map(model: nn.Module) -> dict[nn.Module, str]:
    return {module: name for name, module in model.named_modules() if name}


def _probe_inputs(
    model: nn.Module,
    example_forward: Callable[[nn.Module], object],
    process_group: Optional[ProcessGroup],
):
    modules = list(model.modules())
    activation_bytes = {module: 0 for module in modules}
    binary = {module: True for module in modules}
    sequence_lengths: dict[nn.Module, int] = {}
    seen = set()
    hooks = []

    def hook(module: nn.Module, args: tuple[object, ...]) -> None:
        tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
        if tensor is None:
            return
        seen.add(module)
        activation_bytes[module] = max(
            activation_bytes[module], tensor.numel() * tensor.element_size()
        )
        binary[module] = binary[module] and bool(
            torch.all((tensor == 0) | (tensor == 1)).item()
        )
        if tensor.ndim:
            sequence_lengths[module] = tensor.shape[0]

    for module in modules:
        hooks.append(module.register_forward_pre_hook(hook))
    state = _capture_runtime(model)
    try:
        example_forward(model)
    finally:
        for handle in hooks:
            handle.remove()
        _restore_runtime(model, state)

    byte_values = _group_values(
        [float(activation_bytes[module]) for module in modules],
        dist.ReduceOp.MAX,
        process_group,
    )
    binary_values = _group_values(
        [1.0 if module in seen and binary[module] else 0.0 for module in modules],
        dist.ReduceOp.MIN,
        process_group,
    )
    length_values = _group_values(
        [float(sequence_lengths.get(module, 0)) for module in modules],
        dist.ReduceOp.MIN,
        process_group,
    )
    for module, value, is_binary, length in zip(
        modules, byte_values, binary_values, length_values
    ):
        activation_bytes[module] = int(value)
        binary[module] = bool(is_binary)
        if length:
            sequence_lengths[module] = int(length)
        else:
            sequence_lengths.pop(module, None)
    return activation_bytes, binary, sequence_lengths


def _selected_paths(
    model: nn.Module,
    targets: type | tuple[type, ...],
    activation_bytes: dict[nn.Module, int],
    checkpoint_budget: str,
) -> list[str]:
    path_map = _module_path_map(model)
    candidates = []
    candidate_paths = []
    for path, module in model.named_modules():
        if not path or not isinstance(module, targets):
            continue
        if any(path.startswith(f"{parent}.") for parent in candidate_paths):
            continue
        candidates.append(module)
        candidate_paths.append(path)
    count = math.ceil(len(candidates) * _BUDGET_RATIOS[checkpoint_budget])
    order = {module: index for index, module in enumerate(candidates)}
    candidates.sort(key=lambda module: (-activation_bytes[module], order[module]))
    return [path_map[module] for module in candidates[:count]]


def _compressor_for(module: nn.Module, binary: dict[nn.Module, bool], compress: bool):
    return BitSpikeCompressor() if compress and binary.get(module, False) else None


def _checkpoint_leaves(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (path, module)
        for path, module in model.named_modules()
        if path and isinstance(module, CheckpointWrapper)
    ]


def _split_parts(module: nn.Module, split_fn) -> list[tuple[str, nn.Module]]:
    parts = list(split_fn(module))
    if len(parts) < 2:
        return []
    paths = _module_path_map(module)
    if any(part not in paths for part in parts):
        raise ValueError("split_fn must return registered descendants of its input.")
    part_paths = [paths[part] for part in parts]
    for index, path in enumerate(part_paths):
        if any(
            other.startswith(f"{path}.") or path.startswith(f"{other}.")
            for other in part_paths[index + 1 :]
        ):
            raise ValueError("split_fn descendants must not overlap.")
    return list(zip(part_paths, parts))


def _replace_split(
    model: nn.Module,
    path: str,
    wrapper: nn.Module,
    parts: list[tuple[str, nn.Module]],
    binary: dict[nn.Module, bool],
    compress: bool,
) -> None:
    original = _unwrap_checkpoint_module(wrapper)
    model.set_submodule(path, original)
    for relative_path, part in parts:
        original.set_submodule(
            relative_path,
            checkpoint_module(part, compressor=_compressor_for(part, binary, compress)),
        )


def _restore_split(
    model: nn.Module,
    path: str,
    wrapper: nn.Module,
    parts: list[tuple[str, nn.Module]],
) -> None:
    original = _unwrap_checkpoint_module(wrapper)
    for relative_path, part in parts:
        original.set_submodule(relative_path, part)
    model.set_submodule(path, wrapper)


def _spatial_split(
    model: nn.Module,
    example_forward,
    split_fn,
    activation_bytes,
    binary,
    compress: bool,
    process_group,
) -> float:
    baseline = _run_trial(lambda: _measure_peak(model, example_forward), process_group)
    best = float("inf") if baseline is None else baseline
    blocked = set()
    while True:
        candidates = []
        for path, wrapper in _checkpoint_leaves(model):
            if path in blocked:
                continue
            original = _unwrap_checkpoint_module(wrapper)
            parts = _split_parts(original, split_fn)
            if parts:
                candidates.append(
                    (activation_bytes.get(original, 0), path, wrapper, parts)
                )
        candidates.sort(key=lambda item: (-item[0], item[1]))
        accepted = False
        for _, path, wrapper, parts in candidates:
            _replace_split(model, path, wrapper, parts, binary, compress)
            try:
                peak = _run_trial(
                    lambda: _measure_peak(model, example_forward), process_group
                )
            except Exception:
                _restore_split(model, path, wrapper, parts)
                raise
            if peak is not None and peak < best:
                best, accepted = peak, True
                blocked.clear()
                break
            _restore_split(model, path, wrapper, parts)
            blocked.add(path)
        if not accepted:
            return best


def _temporal_split(
    model: nn.Module,
    example_forward,
    can_chunk,
    activation_bytes,
    sequence_lengths,
    baseline: float,
    process_group,
) -> float:
    eligible = {
        path
        for path, wrapper in _checkpoint_leaves(model)
        if can_chunk(_unwrap_checkpoint_module(wrapper))
    }
    blocked = set()
    while True:
        candidates = []
        for path, wrapper in _checkpoint_leaves(model):
            if path not in eligible or path in blocked:
                continue
            original = _unwrap_checkpoint_module(wrapper)
            options = _checkpoint_options(wrapper)
            next_chunks = int(options["chunks"]) * 2
            if next_chunks <= sequence_lengths.get(original, 0):
                candidates.append(
                    (activation_bytes.get(original, 0), path, wrapper, next_chunks)
                )
        candidates.sort(key=lambda item: (-item[0], item[1]))
        accepted = False
        for _, path, wrapper, next_chunks in candidates:
            original = _unwrap_checkpoint_module(wrapper)
            options = _checkpoint_options(wrapper)
            candidate = checkpoint_module(
                original,
                compressor=options["compressor"],
                chunks=next_chunks,
                chunked_args=options["chunked_args"],
                time_dim=options["time_dim"],
            )
            model.set_submodule(path, candidate)
            try:
                peak = _run_trial(
                    lambda: _measure_peak(model, example_forward), process_group
                )
            except Exception:
                model.set_submodule(path, wrapper)
                raise
            if peak is not None and peak < baseline:
                baseline, accepted = peak, True
                blocked.clear()
                break
            model.set_submodule(path, wrapper)
            blocked.add(path)
        if not accepted:
            return baseline


def _forward_costs(
    model: nn.Module,
    example_forward,
    process_group,
) -> dict[str, float]:
    leaves = _checkpoint_leaves(model)
    events = {path: [] for path, _ in leaves}
    starts = {}
    handles = []

    def pre_hook(path):
        def hook(module, args):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            starts[path] = event

        return hook

    def post_hook(path):
        def hook(module, args, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events[path].append((starts[path], end))

        return hook

    state = _capture_runtime(model)
    try:
        for _ in range(5):
            example_forward(model)
            _restore_runtime(model, state)
        for path, wrapper in leaves:
            original = _unwrap_checkpoint_module(wrapper)
            handles.append(original.register_forward_pre_hook(pre_hook(path)))
            handles.append(original.register_forward_hook(post_hook(path)))
        for _ in range(10):
            example_forward(model)
            _restore_runtime(model, state)
        torch.cuda.synchronize()
        costs = {
            path: sum(start.elapsed_time(end) for start, end in pairs)
            for path, pairs in events.items()
        }
    finally:
        for handle in handles:
            handle.remove()
        _restore_runtime(model, state)

    values = _group_values(list(costs.values()), dist.ReduceOp.MAX, process_group)
    return dict(zip(costs, values))


def _greedy_unwrap(
    model: nn.Module,
    example_forward,
    baseline: float,
    process_group,
) -> None:
    costs = _forward_costs(model, example_forward, process_group)
    for path in sorted(costs, key=lambda name: (-costs[name], name)):
        wrapper = model.get_submodule(path)
        original = _unwrap_checkpoint_module(wrapper)
        model.set_submodule(path, original)
        try:
            peak = _run_trial(
                lambda: _measure_peak(model, example_forward), process_group
            )
        except Exception:
            model.set_submodule(path, wrapper)
            raise
        if peak is None or peak > baseline:
            model.set_submodule(path, wrapper)
        else:
            baseline = peak


def optimize_memory(
    model: nn.Module,
    targets: type | tuple[type, ...],
    example_forward: Optional[Callable[[nn.Module], object]] = None,
    *,
    level: Literal[0, 1, 2, 3, 4],
    checkpoint_budget: Literal["speed", "balanced", "memory"] = "memory",
    compress: bool = True,
    split_fn: Optional[Callable[[nn.Module], Sequence[nn.Module]]] = None,
    can_chunk: Optional[Callable[[nn.Module], bool]] = None,
    process_group: Optional[ProcessGroup] = None,
) -> nn.Module:
    r"""Apply the SpikingJelly memory-optimization preset in place.

    **中文：** 该高层预设实现论文中的四级累进策略：选择性 checkpoint、空间
    切分、时间切分和贪心解包。``level=0`` 严格关闭优化。用户自定义检查点结构时，
    应直接组合 :func:`checkpoint` 和 :func:`checkpoint_module`。

    **English:** This high-level preset implements the paper's four progressive
    stages: selective checkpointing, spatial splitting, temporal splitting, and
    greedy unwrapping. ``level=0`` strictly disables optimization. Compose
    :func:`checkpoint` and :func:`checkpoint_module` for custom structures.

    :param model: model mutated in place / 原地修改的模型
    :type model: nn.Module
    :param targets: module types considered by level 1 / level 1 候选模块类型
    :type targets: type or tuple[type, ...]
    :param example_forward: representative ``callable(model)`` / 代表性前向回调
    :type example_forward: Optional[Callable[[nn.Module], object]]
    :param level: optimization level from 0 to 4 / 0 到 4 的优化级别
    :type level: Literal[0, 1, 2, 3, 4]
    :param checkpoint_budget: target ratio preset / 目标比例预设
    :type checkpoint_budget: Literal["speed", "balanced", "memory"]
    :param compress: auto-compress strictly binary first inputs / 自动压缩严格二值首输入
    :type compress: bool
    :param split_fn: ordered descendant selector for level 2 / level 2 有序子模块选择函数
    :type split_fn: Optional[Callable[[nn.Module], Sequence[nn.Module]]]
    :param can_chunk: temporal-separability predicate for final leaves; this preset
        chunks dimension 0 / 最终片段时间可分判断；本预设固定切分第 0 维
    :type can_chunk: Optional[Callable[[nn.Module], bool]]
    :param process_group: ranks sharing one stage structure / 共享当前 stage 结构的进程组
    :type process_group: Optional[ProcessGroup]
    :return: the same ``model`` object / 同一个 ``model`` 对象
    :rtype: nn.Module
    :raises ValueError: for invalid options or a missing example at level 1-4 / 参数无效或缺少样本

    Reference: "Towards Lossless Memory-efficient Training of Spiking Neural
    Networks via Gradient Checkpointing and Spike Compression", ICLR 2026.
    """
    if level not in (0, 1, 2, 3, 4):
        raise ValueError(f"level must be one of 0, 1, 2, 3, 4; got {level}.")
    if level == 0:
        return model
    if example_forward is None:
        raise ValueError("example_forward is required when level > 0.")
    if checkpoint_budget not in _BUDGET_RATIOS:
        raise ValueError(
            f"checkpoint_budget must be one of {tuple(_BUDGET_RATIOS)}, "
            f"got {checkpoint_budget!r}."
        )
    if isinstance(model, targets):
        raise ValueError("optimize_memory cannot replace the root model in place.")
    if process_group is not None and not dist.is_initialized():
        raise RuntimeError("process_group requires initialized torch.distributed.")

    activation_bytes, binary, sequence_lengths = _probe_inputs(
        model, example_forward, process_group
    )
    selected = _selected_paths(model, targets, activation_bytes, checkpoint_budget)
    for path in selected:
        module = model.get_submodule(path)
        model.set_submodule(
            path,
            checkpoint_module(
                module, compressor=_compressor_for(module, binary, compress)
            ),
        )
    logger.info("memopt level 1 checkpointed {} modules", len(selected))

    if level == 1 or not selected:
        return model

    if split_fn is not None:
        baseline = _spatial_split(
            model,
            example_forward,
            split_fn,
            activation_bytes,
            binary,
            compress,
            process_group,
        )
    elif level == 4 or (level >= 3 and can_chunk is not None):
        measured = _run_trial(
            lambda: _measure_peak(model, example_forward), process_group
        )
        baseline = float("inf") if measured is None else measured
    else:
        return model

    if level >= 3 and can_chunk is not None:
        baseline = _temporal_split(
            model,
            example_forward,
            can_chunk,
            activation_bytes,
            sequence_lengths,
            baseline,
            process_group,
        )
    if level == 4:
        _greedy_unwrap(model, example_forward, baseline, process_group)
    return model
