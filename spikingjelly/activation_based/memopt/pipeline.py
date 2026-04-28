import copy
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from .. import functional, neuron
from ..profiler import LayerWiseFPCUDATimeProfiler, LayerWiseMemoryProfiler
from . import compress
from .checkpointing import GCContainer, TCGCContainer
from .compress import BitSpikeCompressor, NullSpikeCompressor

__all__ = [
    "MEMOPT_PROFILES",
    "MEMOPT_CHECKPOINT_BUDGETS",
    "MemOptSummary",
    "resolve_device",
    "apply_gc",
    "get_module_and_parent",
    "memory_optimization",
]

TCGC_FORBIDDEN_MODULES = [neuron.PSN, neuron.MaskedPSN, neuron.SlidingPSN]
MEMOPT_PROFILES = ("safe", "balanced", "memory", "exhaustive")
MEMOPT_CHECKPOINT_BUDGETS = ("speed", "balanced", "memory")


@dataclass
class MemOptSummary:
    profile: Optional[str]
    checkpoint_budget: Optional[str]
    device: str
    requested_level: int
    applied_level: int
    compress_x: bool
    allow_expensive_profiling: bool
    applied_steps: list = field(default_factory=list)
    skipped_steps: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    gc_wrap_count: int = 0
    manual_compressor_count: int = 0
    bit_compressor_count: int = 0
    null_compressor_count: int = 0
    spatial_split_count: int = 0
    temporal_split_count: int = 0
    unwrap_count: int = 0
    gc_container_count: int = 0
    tcgc_container_count: int = 0
    options: dict = field(default_factory=dict)
    gc_candidate_count: int = 0
    gc_selected_count: int = 0
    gc_selection_policy: str = "all_candidates"


def _build_compressor_from_spec(spec):
    if isinstance(spec, str):
        return getattr(compress, spec)()
    return copy.deepcopy(spec)


def resolve_device() -> str:
    r"""
    **API Language:**
    :ref:`中文 <resolve_device-cn>` | :ref:`English <resolve_device-en>`

    ----

    .. _resolve_device-cn:

    * **中文**

    解析当前进程的逻辑设备。

    优先级：

    1. 若CUDA不可用，则返回 ``"cpu"``
    2. 环境变量 ``LOCAL_RANK`` / ``SLURM_LOCALID`` / ``OMPI_COMM_WORLD_LOCAL_RANK``
    3. 如果 torch.distributed 已初始化，则使用 ``rank % ngpus``
    4. ``torch.cuda.current_device()``
    5. 回退到 ``"cuda"``

    :return: 设备字符串，例如 ``"cpu"`` 或 ``"cuda:0"``
    :rtype: str

    ----

    .. _resolve_device-en:

    * **English**

    Resolve the logical device for the current process.

    Priority:

    1. If CUDA is not available, return ``"cpu"``
    2. Environment variables ``LOCAL_RANK`` / ``SLURM_LOCALID`` / ``OMPI_COMM_WORLD_LOCAL_RANK``
    3. If ``torch.distributed`` is initialized, use ``rank % ngpus``
    4. ``torch.cuda.current_device()``
    5. Fallback to ``"cuda"``

    :return: device string, e.g., ``"cpu"`` or ``"cuda:0"``
    :rtype: str
    """
    if not torch.cuda.is_available():
        return "cpu"

    # common env vars
    for k in (
        "LOCAL_RANK",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
    ):
        v = os.environ.get(k)
        if v is not None:
            try:
                return f"cuda:{int(v)}"
            except Exception:
                pass

    # if dist inited, use rank % n_gpus
    try:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            n_gpu = torch.cuda.device_count()
            if n_gpu > 0:
                return f"cuda:{rank % n_gpu}"
    except Exception:
        pass

    # fallback to current_device (logical ID after CUDA_VISIBLE_DEVICES)
    try:
        return f"cuda:{torch.cuda.current_device()}"
    except Exception:
        return "cuda"


def _dummy_input_to_device(dummy_input, device):
    if isinstance(dummy_input, torch.Tensor):
        return dummy_input.to(device)
    elif isinstance(dummy_input, (tuple, list)):
        return type(dummy_input)(_dummy_input_to_device(t, device) for t in dummy_input)
    elif isinstance(dummy_input, dict):
        return {k: _dummy_input_to_device(v, device) for k, v in dummy_input.items()}
    else:
        # Non-tensor inputs (e.g., None, int, etc.)
        return dummy_input


def _randomize_input_like(dummy_input):
    def _generate_tensor_like(x: torch.Tensor):
        choice = torch.randint(0, 3, ()).item()
        if choice == 0:
            return torch.randn_like(x)
        elif choice == 1:
            return torch.empty_like(x).uniform_(-5, 5)
        else:
            return torch.empty_like(x).bernoulli_(p=0.5)

    if isinstance(dummy_input, torch.Tensor):
        return _generate_tensor_like(dummy_input)
    elif isinstance(dummy_input, (tuple, list)):
        return type(dummy_input)(_randomize_input_like(t) for t in dummy_input)
    elif isinstance(dummy_input, dict):
        return {k: _randomize_input_like(v) for k, v in dummy_input.items()}
    else:
        # Non-tensor inputs (e.g., None, int, etc.)
        return dummy_input


def _probe_binary_inputs(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: tuple,
    n_trials: int = 5,
    target_modules: Optional[Tuple[nn.Module, ...]] = None,
) -> Dict[nn.Module, bool]:
    """Run dummy forward and record whether target modules receive binary inputs."""
    is_binary = defaultdict(lambda: True)
    hooks = []
    if target_modules is None:
        target_modules = tuple(
            m for m in net.modules() if isinstance(m, instance)
        )

    def hook_fn(m, inputs: tuple):
        x = inputs[0]  # assume the first input is the one to be checked
        binary = torch.all((x == 0) | (x == 1)).item()
        is_binary[m] = is_binary[m] and binary

    # register hooks
    for m in target_modules:
        hooks.append(m.register_forward_pre_hook(hook_fn))

    is_training = net.training
    net.eval()
    with torch.no_grad():
        if n_trials > 0:
            _ = net(*dummy_input)
            functional.reset_net(net)
            if target_modules and all(not is_binary[m] for m in target_modules):
                n_trials = 1

        for _ in range(1, n_trials):
            new_input = _randomize_input_like(dummy_input)
            _ = net(*new_input)
            functional.reset_net(net)
            if target_modules and all(not is_binary[m] for m in target_modules):
                break

    net.train(is_training)
    for h in hooks:
        h.remove()
    return dict(is_binary)


def _estimate_module_input_bytes(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: tuple,
    target_modules: Optional[Tuple[nn.Module, ...]] = None,
):
    """Estimate per-module input activation size (in bytes) from one dry run."""
    estimated_bytes = defaultdict(int)
    hooks = []
    if target_modules is None:
        target_modules = tuple(m for m in net.modules() if isinstance(m, instance))

    def hook_fn(m, inputs: tuple):
        if not inputs:
            return
        x = inputs[0]
        if isinstance(x, torch.Tensor):
            estimated_bytes[m] = max(estimated_bytes[m], x.numel() * x.element_size())

    for m in target_modules:
        hooks.append(m.register_forward_pre_hook(hook_fn))

    is_training = net.training
    net.eval()
    with torch.no_grad():
        _ = net(*dummy_input)
        functional.reset_net(net)
    net.train(is_training)
    for h in hooks:
        h.remove()
    return dict(estimated_bytes)


def _resolve_gc_selection_targets(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: Optional[tuple],
    *,
    device: str,
    max_gc_wrapped_modules: Optional[int],
    gc_target_budget_ratio: Optional[float],
):
    candidates = [m for m in net.modules() if isinstance(m, instance)]
    candidate_count = len(candidates)
    if candidate_count == 0:
        return None, candidate_count, candidate_count, "no_candidates"

    limits = []
    if max_gc_wrapped_modules is not None:
        if max_gc_wrapped_modules <= 0:
            return tuple(), candidate_count, 0, "explicit_zero_budget"
        limits.append(min(max_gc_wrapped_modules, candidate_count))
    if gc_target_budget_ratio is not None:
        if gc_target_budget_ratio <= 0:
            return tuple(), candidate_count, 0, "zero_ratio_budget"
        ratio_limit = max(1, min(candidate_count, ceil(candidate_count * gc_target_budget_ratio)))
        limits.append(ratio_limit)

    if not limits:
        return None, candidate_count, candidate_count, "all_candidates"

    selected_count = min(limits)
    if selected_count >= candidate_count:
        return None, candidate_count, candidate_count, "budget_covers_all"

    if dummy_input is None:
        return tuple(candidates[:selected_count]), candidate_count, selected_count, "fallback_module_order"

    net = net.to(device)
    moved_input = _dummy_input_to_device(dummy_input, device)
    estimated_bytes = _estimate_module_input_bytes(
        net,
        instance,
        moved_input,
        target_modules=tuple(candidates),
    )
    ranked = sorted(
        candidates,
        key=lambda m: (estimated_bytes.get(m, 0), -candidates.index(m)),
        reverse=True,
    )
    return tuple(ranked[:selected_count]), candidate_count, selected_count, "largest_input_activations"


def _resolve_memory_optimization_options(
    level: Optional[int],
    profile: Optional[str],
    dummy_input,
    compress_x: bool,
    max_split_rounds: Optional[int],
    max_candidates_per_round: Optional[int],
    warmup_in_main_process: bool,
    warmup_in_profile_workers: bool,
    allow_expensive_profiling: Optional[bool],
):
    if profile is not None and profile not in MEMOPT_PROFILES:
        raise ValueError(
            f"Unsupported memopt profile {profile!r}. Expected one of {MEMOPT_PROFILES}."
        )

    defaults = {
        "safe": dict(
            level=1,
            compress_x=True,
            max_split_rounds=0,
            max_candidates_per_round=0,
            warmup_in_main_process=False,
            warmup_in_profile_workers=False,
            allow_expensive_profiling=False,
        ),
        "balanced": dict(
            level=2,
            compress_x=True,
            max_split_rounds=1,
            max_candidates_per_round=1,
            warmup_in_main_process=False,
            warmup_in_profile_workers=False,
            allow_expensive_profiling=False,
        ),
        "memory": dict(
            level=3,
            compress_x=True,
            max_split_rounds=2,
            max_candidates_per_round=2,
            warmup_in_main_process=False,
            warmup_in_profile_workers=True,
            allow_expensive_profiling=True,
        ),
        "exhaustive": dict(
            level=4,
            compress_x=True,
            max_split_rounds=None,
            max_candidates_per_round=None,
            warmup_in_main_process=True,
            warmup_in_profile_workers=True,
            allow_expensive_profiling=True,
        ),
    }

    if profile is None:
        resolved_level = 0 if level is None else level
        resolved = dict(
            level=resolved_level,
            compress_x=compress_x,
            max_split_rounds=max_split_rounds,
            max_candidates_per_round=max_candidates_per_round,
            warmup_in_main_process=warmup_in_main_process,
            warmup_in_profile_workers=warmup_in_profile_workers,
            allow_expensive_profiling=(
                True if allow_expensive_profiling is None else allow_expensive_profiling
            ),
        )
        notes = []
    else:
        preset = defaults[profile]
        resolved = dict(
            level=preset["level"] if level is None else level,
            compress_x=compress_x if compress_x is not None else preset["compress_x"],
            max_split_rounds=(
                preset["max_split_rounds"]
                if max_split_rounds is None
                else max_split_rounds
            ),
            max_candidates_per_round=(
                preset["max_candidates_per_round"]
                if max_candidates_per_round is None
                else max_candidates_per_round
            ),
            warmup_in_main_process=(
                preset["warmup_in_main_process"]
                if warmup_in_main_process is True
                and level is None
                and max_split_rounds is None
                and max_candidates_per_round is None
                else warmup_in_main_process
            ),
            warmup_in_profile_workers=(
                preset["warmup_in_profile_workers"]
                if warmup_in_profile_workers is True
                and max_split_rounds is None
                and max_candidates_per_round is None
                else warmup_in_profile_workers
            ),
            allow_expensive_profiling=(
                preset["allow_expensive_profiling"]
                if allow_expensive_profiling is None
                else allow_expensive_profiling
            ),
        )
        notes = [f"profile:{profile}"]

    if not resolved["allow_expensive_profiling"] and resolved["level"] > 1:
        resolved["max_split_rounds"] = 1
        resolved["max_candidates_per_round"] = 1
        resolved["warmup_in_profile_workers"] = False

    if dummy_input is None and resolved["level"] > 1:
        if profile is None:
            raise ValueError("dummy_input must be provided for memory profiling.")
        notes.append("fallback:level>1_requires_dummy_input")
        resolved["level"] = 1
        resolved["max_split_rounds"] = 0
        resolved["max_candidates_per_round"] = 0
        resolved["warmup_in_profile_workers"] = False

    return resolved, notes


def _resolve_checkpoint_budget_options(
    checkpoint_budget: Optional[str],
    max_gc_wrapped_modules: Optional[int],
    gc_target_budget_ratio: Optional[float],
):
    if checkpoint_budget is not None and checkpoint_budget not in MEMOPT_CHECKPOINT_BUDGETS:
        raise ValueError(
            "Unsupported checkpoint_budget "
            f"{checkpoint_budget!r}. Expected one of {MEMOPT_CHECKPOINT_BUDGETS}."
        )

    resolved_max = max_gc_wrapped_modules
    resolved_ratio = gc_target_budget_ratio
    notes = []

    if checkpoint_budget is not None:
        defaults = {
            "speed": 0.5,
            "balanced": 0.75,
            "memory": 1.0,
        }
        if resolved_ratio is None and resolved_max is None:
            resolved_ratio = defaults[checkpoint_budget]
            notes.append(f"checkpoint_budget:{checkpoint_budget}")

    return resolved_max, resolved_ratio, notes


def apply_gc(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: Optional[tuple] = None,
    compress_x: bool = True,
    device: str = "cuda",
    checkpoint_budget: Optional[str] = None,
    max_gc_wrapped_modules: Optional[int] = None,
    gc_target_budget_ratio: Optional[float] = None,
    return_summary: bool = False,
) -> Union[nn.Module, Tuple[nn.Module, dict]]:
    r"""
    **API Language:**
    :ref:`中文 <_apply_gc-cn>` | :ref:`English <_apply_gc-en>`

    ----

    .. _apply_gc-cn:

    * **中文**

    对网络中的制定模块应用带输入压缩的梯度检查点（GC）。

    :param net: 目标神经网络模块
    :type net: torch.nn.Module

    :param instance: 要应用 GC 的模块类型或类型元组
    :type instance: Union[type, Tuple[type]]

    :param dummy_input: 用于探测输入的虚拟输入数据
    :type dummy_input: Optional[tuple]

    :param compress_x: 是否压缩输入数据
    :type compress_x: bool

    :param device: 设备类型，例如 "cuda" 或 "cpu"
    :type device: str

    :return: 应用 GC 后的网络模块
    :rtype: torch.nn.Module

    ----

    .. _apply_gc-en:

    * **English**

    Apply gradient checkpointing (GC) with input compression to the specified network module.

    :param net: Target neural network module
    :type net: torch.nn.Module

    :param instance: Module type or tuple of types to apply GC
    :type instance: Union[type, Tuple[type]]

    :param dummy_input: Dummy input data for probing inputs
    :type dummy_input: Optional[tuple]

    :param compress_x: Whether to compress input data
    :type compress_x: bool

    :param device: Device type, e.g., "cuda" or "cpu"
    :type device: str

    :param checkpoint_budget: High-level selective checkpoint preset. One of
        ``"speed"``, ``"balanced"``, or ``"memory"``
    :type checkpoint_budget: Optional[str]

    :param max_gc_wrapped_modules: Optional upper bound on how many matching
        modules should be wrapped. When set, the modules with the largest
        observed input activations are preferred if ``dummy_input`` is given
    :type max_gc_wrapped_modules: Optional[int]

    :param gc_target_budget_ratio: Optional ratio in ``(0, 1]`` controlling the
        fraction of matching modules to wrap. When used together with
        ``max_gc_wrapped_modules``, the smaller budget wins
    :type gc_target_budget_ratio: Optional[float]

    :return: Network module with GC applied
    :rtype: torch.nn.Module
    """
    is_binary_input = {}
    apply_summary = dict(
        gc_wrap_count=0,
        manual_compressor_count=0,
        bit_compressor_count=0,
        null_compressor_count=0,
        gc_candidate_count=0,
        gc_selected_count=0,
        gc_selection_policy="all_candidates",
    )
    (
        max_gc_wrapped_modules,
        gc_target_budget_ratio,
        budget_notes,
    ) = _resolve_checkpoint_budget_options(
        checkpoint_budget,
        max_gc_wrapped_modules,
        gc_target_budget_ratio,
    )
    selected_targets, candidate_count, selected_count, selection_policy = (
        _resolve_gc_selection_targets(
            net,
            instance,
            dummy_input,
            device=device,
            max_gc_wrapped_modules=max_gc_wrapped_modules,
            gc_target_budget_ratio=gc_target_budget_ratio,
        )
    )
    apply_summary["gc_candidate_count"] = candidate_count
    apply_summary["gc_selected_count"] = selected_count
    apply_summary["gc_selection_policy"] = selection_policy
    apply_summary["checkpoint_budget"] = checkpoint_budget
    apply_summary["budget_notes"] = budget_notes
    if compress_x and dummy_input is not None:
        probe_targets = tuple(
            m
            for m in net.modules()
            if isinstance(m, instance)
            and (selected_targets is None or m in selected_targets)
            and getattr(m, "x_compressor", None) is None
        )
        if probe_targets:
            net = net.to(device)
            dummy_input = _dummy_input_to_device(dummy_input, device)
            is_binary_input = _probe_binary_inputs(
                net,
                instance,
                dummy_input,
                target_modules=probe_targets,
            )

    def _get_compressor(module: nn.Module, is_binary_input: bool):
        spec = getattr(module, "x_compressor", None)
        if compress_x:
            if spec is None:  # auto-detect
                x_compressor = (
                    BitSpikeCompressor() if is_binary_input else NullSpikeCompressor()
                )
                if isinstance(x_compressor, BitSpikeCompressor):
                    apply_summary["bit_compressor_count"] += 1
                else:
                    apply_summary["null_compressor_count"] += 1
            else:  # manually specified
                x_compressor = _build_compressor_from_spec(spec)
                apply_summary["manual_compressor_count"] += 1
        else:  # disable compression
            x_compressor = NullSpikeCompressor()
            apply_summary["null_compressor_count"] += 1
        return x_compressor

    def _replace(subnet: nn.Module):
        for name, child in list(subnet.named_children()):
            if isinstance(child, instance) and (
                selected_targets is None or child in selected_targets
            ):
                x_compressor = _get_compressor(child, is_binary_input.get(child, False))
                setattr(subnet, name, GCContainer(x_compressor, child))
                apply_summary["gc_wrap_count"] += 1
            elif not isinstance(child, GCContainer):
                _replace(child)

    _replace(net)
    if return_summary:
        return net, apply_summary
    return net


def _save_bn_states(net: nn.Module):
    bn_modules = [
        m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    saved_bn_states = []
    for m in bn_modules:
        saved_bn_states.append(
            {
                "running_mean": m.running_mean.clone()
                if m.running_mean is not None
                else None,
                "running_var": m.running_var.clone()
                if m.running_var is not None
                else None,
                "num_batches_tracked": m.num_batches_tracked.clone()
                if m.num_batches_tracked is not None
                else None,
            }
        )
    return saved_bn_states


def _load_bn_states(net: nn.Module, saved_bn_states: list):
    bn_modules = [
        m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    for m, state in zip(bn_modules, saved_bn_states):
        if state["running_mean"] is not None:
            m.running_mean.copy_(state["running_mean"])
        else:
            m.running_mean = None
        if state["running_var"] is not None:
            m.running_var.copy_(state["running_var"])
        else:
            m.running_var = None
        if state["num_batches_tracked"] is not None:
            m.num_batches_tracked.copy_(state["num_batches_tracked"])
        else:
            m.num_batches_tracked = None


def _dummy_train_step(
    net: nn.Module,
    dummy_input: Union[torch.Tensor, tuple, list, dict],
    restore_bn: bool = False,
):
    net.train()
    net.zero_grad(set_to_none=True)
    saved_bn_states = []
    if restore_bn:
        saved_bn_states = _save_bn_states(net)

    def _prepare_dummy_input(dummy_input):  # clone, detach, requires grad
        if isinstance(dummy_input, torch.Tensor):
            return dummy_input.clone().detach().requires_grad_(True)
        elif isinstance(dummy_input, (tuple, list)):
            return type(dummy_input)(_prepare_dummy_input(t) for t in dummy_input)
        elif isinstance(dummy_input, dict):
            return {k: _prepare_dummy_input(v) for k, v in dummy_input.items()}
        else:
            # Non-tensor inputs (e.g., None, int, etc.)
            return dummy_input

    dummy_input = _prepare_dummy_input(dummy_input)
    out = net(*dummy_input)

    def _calculate_loss(out):
        if isinstance(out, torch.Tensor):
            return out.sum()
        elif isinstance(out, (tuple, list)):
            return sum(_calculate_loss(t) for t in out)
        elif isinstance(out, dict):
            return sum(_calculate_loss(t) for t in out.values())
        else:
            return 0.0

    loss = _calculate_loss(out)
    loss.backward()
    net.zero_grad(set_to_none=True)
    functional.reset_net(net)

    if restore_bn:
        _load_bn_states(net, saved_bn_states)


def _train_memory_profile_worker(
    net,
    dummy_input,
    q,
    device,
    worker_warmup=True,
    return_peak=False,
):
    """`net` and `dummy_input` should be a deep copy of the original model and
    should be located on CPU, since they must be pickle-able.
    """
    net = net.to(device)
    dummy_input = _dummy_input_to_device(dummy_input, device)

    # Warmup to trigger Triton autotune & JIT compilation in this subprocess.
    # Without this, the peak memory of the 1st and last layers will be strange!
    if worker_warmup:
        _dummy_train_step(net, dummy_input)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with LayerWiseMemoryProfiler(
        (net,),
        model_names=("net",),
        search_mode=("submodules",),
        instances=(GCContainer,),
        device=device,
    ) as prof:
        _dummy_train_step(net, dummy_input)
    results = prof.export(output=False)
    if return_peak:
        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        q.put((results, peak_allocated, peak_reserved))
    else:
        q.put(results)


def _train_memory_profile(
    net, dummy_input, ctx, device, worker_warmup=True, return_peak=False
):
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_train_memory_profile_worker,
        args=(
            copy.deepcopy(net).cpu(),
            _dummy_input_to_device(dummy_input, "cpu"),
            q,
            device,
            worker_warmup,
            return_peak,
        ),
    )
    p.start()
    results = q.get()
    p.join()
    return results


def _train_peak_memory_worker(net, dummy_input, q, device, worker_warmup=True):
    """Profile the peak training memory usage of the entire net.

    `net` and `dummy_input` should be deep copies located on CPU,
    since they must be pickle-able for multiprocessing.
    """
    net = net.to(device)
    dummy_input = _dummy_input_to_device(dummy_input, device)
    # Warmup to trigger Triton autotune & JIT compilation in this subprocess.
    # Without this, the peak memory of the 1st and last layers will be strange!
    if worker_warmup:
        _dummy_train_step(net, dummy_input)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    _dummy_train_step(net, dummy_input)
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    q.put((peak_allocated, peak_reserved))


def _train_peak_memory(net, dummy_input, ctx, device, worker_warmup=True):
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_train_peak_memory_worker,
        args=(
            copy.deepcopy(net).cpu(),
            _dummy_input_to_device(dummy_input, "cpu"),
            q,
            device,
            worker_warmup,
        ),
    )
    p.start()
    results = q.get()
    p.join()
    return results


def _inference_time_profile_worker(net, dummy_input, q, device, N=50):
    """`net` and `dummy_input` should be a deep copy of the original model and
    should be located on CPU, since they must be pickle-able.
    """
    net = net.to(device)
    dummy_input = _dummy_input_to_device(dummy_input, device)

    net.eval()
    with torch.no_grad(), LayerWiseFPCUDATimeProfiler(
        (net,),
        model_names=("net",),
        search_mode=("submodules",),
        instances=(GCContainer,),
    ) as prof:
        for _ in range(N):
            _ = net(*dummy_input)
            functional.reset_net(net)
    results = prof.export(output=False)
    prof.close()
    q.put(results)


def _inference_time_profile(net, dummy_input, ctx, device):
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_inference_time_profile_worker,
        args=(
            copy.deepcopy(net).cpu(),
            _dummy_input_to_device(dummy_input, "cpu"),
            q,
            device,
        ),
        kwargs={"N": 50},
    )
    p.start()
    results = q.get()
    p.join()
    return results


def get_module_and_parent(
    net: nn.Module, module_name: str
) -> Tuple[nn.Module, nn.Module, str]:
    r"""
    **API Language:**
    :ref:`中文 <_get_module_and_parent-cn>` | :ref:`English <_get_module_and_parent-en>`

    ----

    .. _get_module_and_parent-cn:

    * **中文**

    根据模块路径（例如 ``"layer1.0.conv1"`` ，不包括顶层模块名称）返回目标模块、父模块以及目标模块的名称。

    :param net: 神经网络模型
    :type net: nn.Module

    :param module_name: 模块路径字符串
    :type module_name: str

    :return: 目标模块、父模块和目标模块名称
    :rtype: Tuple[nn.Module, nn.Module, str]

    ----

    .. _get_module_and_parent-en:

    * **English**

    Given a module path (e.g., ``“layer1.0.conv1"`` , excluding the top-level module name), return the target module, parent module, and target module name.

    :param net: Neural network model
    :type net: nn.Module

    :param module_name: Module path string
    :type module_name: str

    :return: target module, parent module, and target module name
    :rtype: Tuple[nn.Module, nn.Module, str]
    """
    module_name = module_name.split(" ")[-1]
    parts = module_name.split(".")
    parent = net
    for p in parts[:-1]:
        parent = getattr(parent, p)
    child_name = parts[-1]
    module = getattr(parent, child_name)
    return module, parent, child_name


def _spatially_split_gc_container(block: GCContainer, compress_x: bool = True):
    assert isinstance(block, GCContainer)

    if len(block) > 1:
        sub_blocks = block
    elif len(block) == 1 and hasattr(block[0], "__spatial_split__"):
        sub_blocks = block[0].__spatial_split__()
    else:  # not split-able
        return None
    x_compressor = block.x_compressor

    def _get_compressor(module: nn.Module, use_original_compressor: bool):
        spec = getattr(module, "x_compressor", None)
        if compress_x:
            if spec is None:  # auto-detect
                c = (
                    copy.deepcopy(x_compressor)
                    if use_original_compressor
                    else NullSpikeCompressor()
                )
            else:  # manually specified
                c = _build_compressor_from_spec(spec)
        else:  # disable compression
            c = NullSpikeCompressor()
        return c

    l = []
    for i, sub in enumerate(sub_blocks):
        c = _get_compressor(sub, i == 0)
        l.append(GCContainer(c, sub))
    return nn.Sequential(*l)


def _can_spatially_split(block: GCContainer) -> bool:
    return len(block) > 1 or (len(block) == 1 and hasattr(block[0], "__spatial_split__"))


def _cannot_temporally_split(block: GCContainer):
    for m in block.modules():
        if isinstance(m, tuple(TCGC_FORBIDDEN_MODULES)):
            return True
    return False


def _temporally_split_gc_container(block: GCContainer, factor: int = 2):
    assert isinstance(block, GCContainer)

    if _cannot_temporally_split(block):
        return None

    x_compressor = block.x_compressor
    n_chunk = getattr(block, "n_chunk", 1)
    n_seq_inputs = getattr(block[0], "n_seq_inputs", 1)
    n_outputs = getattr(block[-1], "n_outputs", 1)
    return TCGCContainer(
        x_compressor,
        *block,
        n_chunk=n_chunk * factor,
        n_seq_inputs=n_seq_inputs,
        n_outputs=n_outputs,
    )


def _can_temporally_split(block: GCContainer) -> bool:
    return not _cannot_temporally_split(block)


def _unwrap_gc_container(block: GCContainer) -> nn.Module:
    assert isinstance(block, GCContainer)

    if len(block) == 1:
        return block[0]
    else:
        return nn.Sequential(*block)


def _cprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def _candidate_entries(results, max_candidates_per_round: Optional[int]):
    if max_candidates_per_round is None or max_candidates_per_round <= 0:
        return results
    return results[:max_candidates_per_round]


def _gc_container_count(net: nn.Module) -> int:
    return sum(1 for m in net.modules() if isinstance(m, GCContainer))


def _has_split_candidate(net: nn.Module, predicate) -> bool:
    return any(predicate(m) for m in net.modules() if isinstance(m, GCContainer))


def memory_optimization(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: Optional[tuple] = None,
    compress_x: bool = True,
    level: Optional[int] = None,
    verbose: bool = False,
    temporal_split_factor: int = 2,
    max_split_rounds: Optional[int] = None,
    max_candidates_per_round: Optional[int] = None,
    warmup_in_main_process: bool = True,
    warmup_in_profile_workers: bool = True,
    profile: Optional[str] = None,
    allow_expensive_profiling: Optional[bool] = None,
    checkpoint_budget: Optional[str] = None,
    max_gc_wrapped_modules: Optional[int] = None,
    gc_target_budget_ratio: Optional[float] = None,
    return_summary: bool = False,
) -> Union[nn.Module, Tuple[nn.Module, MemOptSummary]]:
    r"""
    **API Language:**
    :ref:`中文 <memory_optimization-cn>` | :ref:`English <memory_optimization-en>`

    ----

    .. _memory_optimization-cn:

    * **中文**

    使用梯度检查点和脉冲压缩进行训练显存优化。

    此函数通过以下逐步优化策略转换给定的网络：

    - ``level=0`` : 无优化。
    - ``level=1`` : 将匹配的模块包装在 :class:`GCContainer <spikingjelly.activation_based.memopt.checkpointing.GCContainer>` 中以进行逐层梯度检查点（GC），可选输入压缩。
    - ``level=2`` : 如果支持，则沿空间维度拆分显存消耗巨大的的 ``GCContainer`` 。
    - ``level=3`` : 如果支持，则沿时间维度进一步显存消耗巨大的 ``GCContainer`` 。
    - ``level=4`` : 如果不会增加内存占用，则贪婪地解包部分 ``GCContainer`` 以减少训练时间成本。

    :param net: 要优化的模型
    :type net: nn.Module

    :param instance: 要包装的模块类或模块类元组
    :type instance: Union[type, Tuple[type]]

    :param dummy_input: 用于内存分析的输入， ``level > 1`` 时必需给出。需使用元组包装。
    :type dummy_input: Optional[tuple]

    :param compress_x: 是否应用输入脉冲压缩
    :type compress_x: bool

    :param level: 优化级别。若为 ``None`` 且指定 ``profile`` ，则使用预设推荐值
    :type level: Optional[int]

    :param verbose: 是否打印优化过程日志
    :type verbose: bool

    :param temporal_split_factor: 沿时间拆分检查点片段时所使用的倍增因子
    :type temporal_split_factor: int

    :param max_split_rounds: 每个 split 阶段允许的最大 profiling 轮数。 ``None`` 表示不限制
    :type max_split_rounds: Optional[int]

    :param max_candidates_per_round: 每轮 profiling 至多尝试的候选 ``GCContainer`` 数量。 ``None`` 表示不限制
    :type max_candidates_per_round: Optional[int]

    :param warmup_in_main_process: 是否在主进程中对优化后的模型执行一次 dummy train step，
        以避免首次使用时的额外开销。默认开启
    :type warmup_in_main_process: bool

    :param warmup_in_profile_workers: 是否在 profiling 子进程中执行预热 dummy train step。
        默认开启；关闭后可以减少优化耗时，但可能增加测量噪声
    :type warmup_in_profile_workers: bool

    :param profile: 高层预设策略，可选 ``"safe"`` 、 ``"balanced"`` 、 ``"memory"`` 、 ``"exhaustive"``
    :type profile: Optional[str]

    :param allow_expensive_profiling: 是否允许高开销 profiling。关闭后会自动收紧 split 搜索预算
    :type allow_expensive_profiling: Optional[bool]

    :param checkpoint_budget: 高层选择性 checkpoint 预算策略，可选
        ``"speed"`` 、 ``"balanced"`` 、 ``"memory"``
    :type checkpoint_budget: Optional[str]

    :param max_gc_wrapped_modules: 选择性 checkpoint 的上限。若给定，则只包装最多这么多个匹配模块。
        当 ``dummy_input`` 可用时，优先选择输入激活更大的模块
    :type max_gc_wrapped_modules: Optional[int]

    :param gc_target_budget_ratio: 选择性 checkpoint 的比例预算，取值应在 ``(0, 1]`` 之间。
        当与 ``max_gc_wrapped_modules`` 同时给定时，较小的预算生效
    :type gc_target_budget_ratio: Optional[float]

    :param return_summary: 是否同时返回结构化优化摘要
    :type return_summary: bool

    :return: 优化后的模型；当 ``return_summary=True`` 时，返回 ``(model, summary)``
    :rtype: Union[nn.Module, Tuple[nn.Module, MemOptSummary]]

    ----

    .. _memory_optimization-en:

    * **English**

    Memory optimization using gradient checkpointing and spike compression.

    This function progressively transforms the given network by applying the following optimization strategies:

    - ``level=0`` : no optimization.
    - ``level=1`` : wrap matching modules in :class:`GCContainer <spikingjelly.activation_based.memopt.checkpointing.GCContainer>` for layer-wise gradient checkpointing (GC), with optional input compression.
    - ``level=2`` : recursively split heavy ``GCContainer`` into multiple sub-containers along the spatial dimension, if supported.
    - ``level=3`` : further split heavy ``GCContainer`` along the temporal dimension, if supported.
    - ``level=4`` : greedily unwrap some ``GCContainer`` to reduce training time cost if doing so does not increase the memory footprint.

    :param net: the model to be optimized
    :type net: nn.Module

    :param instance: module classes or tuple of classes to wrap
    :type instance: Union[type, Tuple[type]]

    :param dummy_input: input for memory profiling, required if ``level > 1`` . Should be wrapped by a tuple.
    :type dummy_input: Optional[tuple]

    :param compress_x: whether to apply input spike compression
    :type compress_x: bool

    :param level: optimization level. If ``None`` and ``profile`` is specified,
        the recommended preset level will be used
    :type level: Optional[int]

    :param verbose: whether to print logs
    :type verbose: bool

    :param temporal_split_factor: factor to increase the number of chunks when splitting GC segments temporally
    :type temporal_split_factor: int

    :param max_split_rounds: maximum number of profiling rounds allowed for each split stage.
        ``None`` means no limit
    :type max_split_rounds: Optional[int]

    :param max_candidates_per_round: maximum number of GCContainer candidates to try in each profiling round.
        ``None`` means no limit
    :type max_candidates_per_round: Optional[int]

    :param warmup_in_main_process: whether to run one dummy train step for the optimized
        model in the main process to hide first-use overhead. Default to ``True``
    :type warmup_in_main_process: bool

    :param warmup_in_profile_workers: whether to run a warmup dummy train step in
        profiling subprocesses. Default to ``True``; disabling it can reduce
        optimization latency at the cost of noisier measurements
    :type warmup_in_profile_workers: bool

    :param profile: high-level preset strategy. One of ``"safe"``, ``"balanced"``,
        ``"memory"``, or ``"exhaustive"``
    :type profile: Optional[str]

    :param allow_expensive_profiling: whether to allow expensive profiling.
        Disabling this automatically tightens split search budgets
    :type allow_expensive_profiling: Optional[bool]

    :param checkpoint_budget: high-level selective checkpoint budget preset.
        One of ``"speed"``, ``"balanced"``, or ``"memory"``
    :type checkpoint_budget: Optional[str]

    :param max_gc_wrapped_modules: upper bound for selective checkpointing.
        When provided, at most this many matching modules are wrapped; modules
        with larger observed input activations are preferred when ``dummy_input`` is available
    :type max_gc_wrapped_modules: Optional[int]

    :param gc_target_budget_ratio: ratio budget for selective checkpointing in ``(0, 1]``.
        When used together with ``max_gc_wrapped_modules``, the smaller budget wins
    :type gc_target_budget_ratio: Optional[float]

    :param return_summary: whether to also return a structured optimization summary
    :type return_summary: bool

    :return: the optimized model, or ``(model, summary)`` when ``return_summary=True``
    :rtype: Union[nn.Module, Tuple[nn.Module, MemOptSummary]]
    """
    st = time.time()
    ctx = mp.get_context("spawn")
    device = resolve_device()
    preset_levels = {
        "safe": 1,
        "balanced": 2,
        "memory": 3,
        "exhaustive": 4,
    }
    requested_level = (
        level if level is not None else preset_levels.get(profile, 0)
    )
    resolved, resolution_notes = _resolve_memory_optimization_options(
        level=level,
        profile=profile,
        dummy_input=dummy_input,
        compress_x=compress_x,
        max_split_rounds=max_split_rounds,
        max_candidates_per_round=max_candidates_per_round,
        warmup_in_main_process=warmup_in_main_process,
        warmup_in_profile_workers=warmup_in_profile_workers,
        allow_expensive_profiling=allow_expensive_profiling,
    )
    level = resolved["level"]
    compress_x = resolved["compress_x"]
    max_split_rounds = resolved["max_split_rounds"]
    max_candidates_per_round = resolved["max_candidates_per_round"]
    warmup_in_main_process = resolved["warmup_in_main_process"]
    warmup_in_profile_workers = resolved["warmup_in_profile_workers"]
    allow_expensive_profiling = resolved["allow_expensive_profiling"]

    summary = MemOptSummary(
        profile=profile,
        checkpoint_budget=checkpoint_budget,
        device=device,
        requested_level=requested_level,
        applied_level=0 if level is None else level,
        compress_x=compress_x,
        allow_expensive_profiling=allow_expensive_profiling,
        notes=list(resolution_notes),
        options=dict(
            temporal_split_factor=temporal_split_factor,
            max_split_rounds=max_split_rounds,
            max_candidates_per_round=max_candidates_per_round,
            warmup_in_main_process=warmup_in_main_process,
            warmup_in_profile_workers=warmup_in_profile_workers,
            checkpoint_budget=checkpoint_budget,
            max_gc_wrapped_modules=max_gc_wrapped_modules,
            gc_target_budget_ratio=gc_target_budget_ratio,
        ),
    )
    summary.applied_level = level
    _cprint(verbose, f"Optimizing memory on device {device}")
    peak_allocated = -1.0

    if level > 0:
        _cprint(verbose, "Level 1: layer-wise GC with input spike compression")
        net, apply_summary = apply_gc(
            net,
            instance,
            dummy_input,
            compress_x,
            device,
            checkpoint_budget=checkpoint_budget,
            max_gc_wrapped_modules=max_gc_wrapped_modules,
            gc_target_budget_ratio=gc_target_budget_ratio,
            return_summary=True,
        )
        summary.applied_steps.append("level1_gc")
        for k, v in apply_summary.items():
            if v is None:
                continue
            if isinstance(v, str):
                setattr(summary, k, v)
            elif isinstance(v, list):
                summary.notes.extend(v)
            else:
                setattr(summary, k, getattr(summary, k) + v)

    if level > 1:  # spatial split
        if _gc_container_count(net) == 0:
            _cprint(verbose, "Level 2: no GCContainers found, skip spatial split")
            summary.skipped_steps.append("level2:no_gccontainers")
        elif not _has_split_candidate(net, _can_spatially_split):
            _cprint(verbose, "Level 2: no spatially splittable GCContainers, skip")
            summary.skipped_steps.append("level2:no_spatial_candidates")
        else:
            _cprint(verbose, "Level 2: split GCContainers spatially")
            summary.applied_steps.append("level2_spatial_split")
            peak_allocated, _ = _train_peak_memory(
                net,
                dummy_input,
                ctx,
                device,
                worker_warmup=warmup_in_profile_workers,
            )
            split_rounds = 0
            blocked_candidates = set()

            while True:
                if max_split_rounds is not None and split_rounds >= max_split_rounds:
                    _cprint(verbose, "\tReached max_split_rounds for spatial split.")
                    break
                if not _has_split_candidate(net, _can_spatially_split):
                    _cprint(verbose, "\tNo spatially splittable GCContainers remain.")
                    break
                split_rounds += 1
                results = _train_memory_profile(
                    net,
                    dummy_input,
                    ctx,
                    device,
                    worker_warmup=warmup_in_profile_workers,
                )
                if not results:
                    _cprint(verbose, "\tNo more GCContainers to split.")
                    break
                filtered_results = [
                    row for row in results if row[0].split(" ")[-1] not in blocked_candidates
                ]
                if not filtered_results:
                    _cprint(verbose, "\tNo eligible spatial split candidates remain.")
                    break
                improved = False
                for row in _candidate_entries(filtered_results, max_candidates_per_round):
                    cb_name = row[0]
                    cb_path = cb_name.split(" ")[-1]
                    cb, parent, child_name = get_module_and_parent(net, cb_path)

                    split_cb = _spatially_split_gc_container(cb)
                    if split_cb is None:
                        _cprint(verbose, f"\t{cb_name}: can't be spatially split")
                        blocked_candidates.add(cb_path)
                        continue
                    setattr(parent, child_name, split_cb)

                    new_peak_allocated, _ = _train_peak_memory(
                        net,
                        dummy_input,
                        ctx,
                        device,
                        worker_warmup=warmup_in_profile_workers,
                    )
                    if new_peak_allocated >= peak_allocated:
                        _cprint(
                            verbose,
                            f"\t{cb_name}: no reduction in memory, revert "
                            f"({peak_allocated} -> {new_peak_allocated})",
                        )
                        setattr(parent, child_name, cb)
                        blocked_candidates.add(cb_path)
                        continue

                    _cprint(
                        verbose,
                        f"\t{cb_name}: successfully split "
                        f"({peak_allocated} -> {new_peak_allocated})",
                    )
                    peak_allocated = new_peak_allocated
                    summary.spatial_split_count += 1
                    improved = True
                    blocked_candidates.clear()
                    break

                if not improved:
                    _cprint(verbose, "\tNo spatial split candidate improved memory.")
                    summary.skipped_steps.append("level2:no_improving_candidate")
                    break

    if level > 2:  # temporal split
        if _gc_container_count(net) == 0:
            _cprint(verbose, "Level 3: no GCContainers found, skip temporal split")
            summary.skipped_steps.append("level3:no_gccontainers")
        elif not _has_split_candidate(net, _can_temporally_split):
            _cprint(verbose, "Level 3: no temporally splittable GCContainers, skip")
            summary.skipped_steps.append("level3:no_temporal_candidates")
        else:
            _cprint(verbose, "Level 3: split GCContainers temporally")
            summary.applied_steps.append("level3_temporal_split")
            if peak_allocated < 0:
                peak_allocated, _ = _train_peak_memory(
                    net,
                    dummy_input,
                    ctx,
                    device,
                    worker_warmup=warmup_in_profile_workers,
                )
            split_rounds = 0
            blocked_candidates = set()

            while True:
                if max_split_rounds is not None and split_rounds >= max_split_rounds:
                    _cprint(verbose, "\tReached max_split_rounds for temporal split.")
                    break
                if not _has_split_candidate(net, _can_temporally_split):
                    _cprint(verbose, "\tNo temporally splittable GCContainers remain.")
                    break
                split_rounds += 1
                results = _train_memory_profile(
                    net,
                    dummy_input,
                    ctx,
                    device,
                    worker_warmup=warmup_in_profile_workers,
                )
                if not results:
                    _cprint(verbose, "\tNo more GCContainers to split.")
                    break
                filtered_results = [
                    row for row in results if row[0].split(" ")[-1] not in blocked_candidates
                ]
                if not filtered_results:
                    _cprint(verbose, "\tNo eligible temporal split candidates remain.")
                    break
                improved = False
                for row in _candidate_entries(filtered_results, max_candidates_per_round):
                    cb_name = row[0]
                    cb_path = cb_name.split(" ")[-1]
                    cb, parent, child_name = get_module_and_parent(net, cb_path)

                    split_cb = _temporally_split_gc_container(cb, temporal_split_factor)
                    if split_cb is None:
                        _cprint(verbose, f"\t{cb_name}: can't be temporally split")
                        blocked_candidates.add(cb_path)
                        continue
                    setattr(parent, child_name, split_cb)

                    new_peak_allocated, _ = _train_peak_memory(
                        net,
                        dummy_input,
                        ctx,
                        device,
                        worker_warmup=warmup_in_profile_workers,
                    )
                    if new_peak_allocated >= peak_allocated:
                        _cprint(
                            verbose,
                            f"\t{cb_name}: no reduction in memory, revert "
                            f"({peak_allocated} -> {new_peak_allocated})",
                        )
                        setattr(parent, child_name, cb)
                        blocked_candidates.add(cb_path)
                        continue

                    _cprint(
                        verbose,
                        f"\t{cb_name}: successfully split "
                        f"({peak_allocated} -> {new_peak_allocated})",
                    )
                    peak_allocated = new_peak_allocated
                    summary.temporal_split_count += 1
                    improved = True
                    blocked_candidates.clear()
                    break

                if not improved:
                    _cprint(verbose, "\tNo temporal split candidate improved memory.")
                    summary.skipped_steps.append("level3:no_improving_candidate")
                    break

    if level > 3:
        if _gc_container_count(net) == 0:
            _cprint(verbose, "Level 4: no GCContainers found, skip greedy unwrap")
            summary.skipped_steps.append("level4:no_gccontainers")
        else:
            if peak_allocated < 0:
                peak_allocated, _ = _train_peak_memory(
                    net,
                    dummy_input,
                    ctx,
                    device,
                    worker_warmup=warmup_in_profile_workers,
                )
            _cprint(verbose, "Level 4: greedily disable GCContainers")
            summary.applied_steps.append("level4_greedy_unwrap")
            results = _inference_time_profile(net, dummy_input, ctx, device)

            for r in results:
                cb_name = r[0]
                cb, parent, child_name = get_module_and_parent(net, cb_name.split(" ")[-1])

                # try to unwrap the GCContainer
                ucb = _unwrap_gc_container(cb)
                setattr(parent, child_name, ucb)

                # if the peak memory increases, revert; otherwise, keep the change
                new_peak_allocated, _ = _train_peak_memory(
                    net,
                    dummy_input,
                    ctx,
                    device,
                    worker_warmup=warmup_in_profile_workers,
                )
                if new_peak_allocated > peak_allocated:
                    _cprint(
                        verbose,
                        f"\t{cb_name}: keep GCContainer "
                        f"({peak_allocated} -> {new_peak_allocated})",
                    )
                    setattr(parent, child_name, cb)
                else:
                    _cprint(
                        verbose,
                        f"\t{cb_name}: disable GCContainer "
                        f"({peak_allocated} -> {new_peak_allocated})",
                    )
                    peak_allocated = new_peak_allocated  # update the peak memory
                    summary.unwrap_count += 1

    if warmup_in_main_process and dummy_input is not None:
        # Warm up in the main process to avoid 1st-time overhead.
        net = net.to(device)
        dummy_input = _dummy_input_to_device(dummy_input, device)
        _dummy_train_step(net, dummy_input, restore_bn=True)

    et = time.time()
    _cprint(verbose, f"Total time: {et - st:.2f}s")
    net = net.cpu()  # must return a model on CPU
    summary.gc_container_count = sum(
        1 for m in net.modules() if isinstance(m, GCContainer)
    )
    summary.tcgc_container_count = sum(
        1 for m in net.modules() if isinstance(m, TCGCContainer)
    )
    if return_summary:
        return net, summary
    return net
