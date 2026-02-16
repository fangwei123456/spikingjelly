from typing import Optional, Tuple, Union, Dict
import copy
import time
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from ..profiler import *
from .. import functional, neuron
from . import compress
from .compress import BitSpikeCompressor, NullSpikeCompressor
from .checkpointing import GCContainer, TCGCContainer


__all__ = ["resolve_device", "apply_gc", "get_module_and_parent", "memory_optimization"]

TCGC_FORBIDDEN_MODULES = [neuron.PSN, neuron.MaskedPSN, neuron.SlidingPSN]



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


def _probe_binary_inputs(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: tuple,
    n_trials: int = 5,
) -> Dict[nn.Module, bool]:
    """Run dummy forward and record whether target modules receive binary inputs."""
    is_binary = defaultdict(lambda: True)
    hooks = []

    def hook_fn(m, inputs: tuple):
        x = inputs[0]  # assume the first input is the one to be checked
        binary = torch.all((x == 0) | (x == 1)).item()
        is_binary[m] = is_binary[m] and binary

    # register hooks
    for m in net.modules():
        if isinstance(m, instance):
            hooks.append(m.register_forward_pre_hook(hook_fn))

    def _generate_tensor_like(x: torch.Tensor):
        choice = torch.randint(0, 3, ()).item()
        if choice == 0:
            return torch.randn_like(x)
        elif choice == 1:
            return torch.empty_like(x).uniform_(-5, 5)
        else:
            return torch.empty_like(x).bernoulli_(p=0.5)

    def _generate_input_like(dummy_input):
        if isinstance(dummy_input, torch.Tensor):
            return _generate_tensor_like(dummy_input)
        elif isinstance(dummy_input, (tuple, list)):
            return type(dummy_input)(_generate_input_like(t) for t in dummy_input)
        elif isinstance(dummy_input, dict):
            return {k: _generate_input_like(v) for k, v in dummy_input.items()}
        else:
            # Non-tensor inputs (e.g., None, int, etc.)
            return dummy_input

    is_training = net.training
    net.eval()
    with torch.no_grad():
        for _ in range(n_trials):
            new_input = _generate_input_like(dummy_input)
            _ = net(*new_input)
            functional.reset_net(net)

    net.train(is_training)
    for h in hooks:
        h.remove()
    return dict(is_binary)


def apply_gc(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: Optional[tuple] = None,
    compress_x: bool = True,
    device: str = "cuda",
) -> nn.Module:
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

    :return: Network module with GC applied
    :rtype: torch.nn.Module
    """
    is_binary_input = {}
    if compress_x and dummy_input is not None:
        net = net.to(device)
        dummy_input = _dummy_input_to_device(dummy_input, device)
        is_binary_input = _probe_binary_inputs(net, instance, dummy_input)

    def _get_compressor(module: nn.Module, is_binary_input: bool):
        spec = getattr(module, "x_compressor", None)
        if compress_x:
            if spec is None:  # auto-detect
                x_compressor = (
                    BitSpikeCompressor() if is_binary_input else NullSpikeCompressor()
                )
            else:  # manually specified
                x_compressor = (
                    getattr(compress, spec)() if isinstance(spec, str) else spec
                )
        else:  # disable compression
            x_compressor = NullSpikeCompressor()
        return x_compressor

    def _replace(subnet: nn.Module):
        for name, child in list(subnet.named_children()):
            if isinstance(child, instance):
                x_compressor = _get_compressor(child, is_binary_input.get(child, False))
                setattr(subnet, name, GCContainer(x_compressor, child))
            elif not isinstance(child, GCContainer):
                _replace(child)

    _replace(net)
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
    net: nn.Module, dummy_input: Union[tuple], restore_bn: bool = False
):
    net.train()
    net.zero_grad(set_to_none=True)
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


def _train_memory_profile_worker(net, dummy_input, q, device):
    """`net` and `dummy_input` should be a deep copy of the original model and
    should be located on CPU, since they must be pickle-able.
    """
    net = net.to(device)
    dummy_input = _dummy_input_to_device(dummy_input, device)

    # Warmup to trigger Triton autotune & JIT compilation in this subprocess.
    # Without this, the peak memory of the 1st and last layers will be strange!
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
    q.put(results)


def _train_memory_profile(net, dummy_input, ctx, device):
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_train_memory_profile_worker,
        args=(
            copy.deepcopy(net).cpu(),
            _dummy_input_to_device(dummy_input, "cpu"),
            q,
            device,
        ),
    )
    p.start()
    results = q.get()
    p.join()
    return results


def _train_peak_memory_worker(net, dummy_input, q, device):
    """Profile the peak training memory usage of the entire net.

    `net` and `dummy_input` should be deep copies located on CPU,
    since they must be pickle-able for multiprocessing.
    """
    net = net.to(device)
    dummy_input = _dummy_input_to_device(dummy_input, device)
    # Warmup to trigger Triton autotune & JIT compilation in this subprocess.
    # Without this, the peak memory of the 1st and last layers will be strange!
    _dummy_train_step(net, dummy_input)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    _dummy_train_step(net, dummy_input)
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    q.put((peak_allocated, peak_reserved))


def _train_peak_memory(net, dummy_input, ctx, device):
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_train_peak_memory_worker,
        args=(
            copy.deepcopy(net).cpu(),
            _dummy_input_to_device(dummy_input, "cpu"),
            q,
            device,
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
    with (
        torch.no_grad(),
        LayerWiseFPCUDATimeProfiler(
            (net,),
            model_names=("net",),
            search_mode=("submodules",),
            instances=(GCContainer,),
        ) as prof,
    ):
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
                c = x_compressor if use_original_compressor else NullSpikeCompressor()
            else:  # manually specified
                c = getattr(compress, spec)() if isinstance(spec, str) else spec
        else:  # disable compression
            c = NullSpikeCompressor()
        return c

    l = []
    for i, sub in enumerate(sub_blocks):
        c = _get_compressor(sub, i == 0)
        l.append(GCContainer(c, sub))
    return nn.Sequential(*l)


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


def _unwrap_gc_container(block: GCContainer) -> nn.Module:
    assert isinstance(block, GCContainer)

    if len(block) == 1:
        return block[0]
    else:
        return nn.Sequential(*block)


def _cprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def memory_optimization(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: Optional[tuple] = None,
    compress_x: bool = True,
    level: int = 0,
    verbose: bool = False,
    temporal_split_factor: int = 2,
):
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

    :param level: 优化级别
    :type level: int

    :param verbose: 是否打印优化过程日志
    :type verbose: bool

    :param temporal_split_factor: 沿时间拆分检查点片段时所使用的倍增因子
    :type temporal_split_factor: int

    :return: 优化后的模型
    :rtype: nn.Module

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

    :param level: optimization level
    :type level: int

    :param verbose: whether to print logs
    :type verbose: bool

    :param temporal_split_factor: factor to increase the number of chunks when splitting GC segments temporally
    :type temporal_split_factor: int

    :return: the optimized model
    :rtype: nn.Module
    """
    st = time.time()
    ctx = mp.get_context("spawn")
    device = resolve_device()
    _cprint(verbose, f"Optimizing memory on device {device}")

    if level > 0:
        _cprint(verbose, "Level 1: layer-wise GC with input spike compression")
        net = apply_gc(net, instance, dummy_input, compress_x, device)

    if level > 1:  # spatial split
        if dummy_input is None:
            raise ValueError("dummy_input must be provided for memory profiling.")

        _cprint(verbose, "Level 2: split GCContainers spatially")
        peak_allocated, _ = _train_peak_memory(net, dummy_input, ctx, device)

        while True:
            results = _train_memory_profile(net, dummy_input, ctx, device)
            if not results:
                _cprint(verbose, "\tNo more GCContainers to split.")
                break
            cb_name = results[0][0]  # GCContainer with the highest mem.
            cb, parent, child_name = get_module_and_parent(net, cb_name.split(" ")[-1])

            # try to spatially split the GCContainer
            # if not split-able, break
            split_cb = _spatially_split_gc_container(cb)
            if split_cb is None:
                _cprint(verbose, f"\t{cb_name}: can't be spatially split")
                break
            setattr(parent, child_name, split_cb)

            # if the peak memory does not reduces, revert and break;
            # otherwise, keep the change and continue
            new_peak_allocated, _ = _train_peak_memory(net, dummy_input, ctx, device)
            if new_peak_allocated >= peak_allocated:
                _cprint(
                    verbose,
                    f"\t{cb_name}: no reduction in memory, revert "
                    f"({peak_allocated} -> {new_peak_allocated})",
                )
                setattr(parent, child_name, cb)
                break
            else:
                _cprint(
                    verbose,
                    f"\t{cb_name}: successfully split "
                    f"({peak_allocated} -> {new_peak_allocated})",
                )
                peak_allocated = new_peak_allocated  # update the peak memory

    if level > 2:  # temporal split
        _cprint(verbose, "Level 3: split GCContainers temporally")

        while True:
            results = _train_memory_profile(net, dummy_input, ctx, device)
            if not results:
                _cprint(verbose, "\tNo more GCContainers to split.")
                break
            cb_name = results[0][0]  # GCContainer with the highest mem.
            cb, parent, child_name = get_module_and_parent(net, cb_name.split(" ")[-1])

            # try to temporally split the GCContainer
            # if not split-able, break
            split_cb = _temporally_split_gc_container(cb, temporal_split_factor)
            if split_cb is None:
                _cprint(verbose, f"\t{cb_name}: can't be temporally split")
                break
            setattr(parent, child_name, split_cb)

            # if the peak memory does not reduces, revert and break;
            # otherwise, keep the change and continue
            new_peak_allocated, _ = _train_peak_memory(net, dummy_input, ctx, device)
            if new_peak_allocated >= peak_allocated:
                _cprint(
                    verbose,
                    f"\t{cb_name}: no reduction in memory, revert "
                    f"({peak_allocated} -> {new_peak_allocated})",
                )
                setattr(parent, child_name, cb)
                break
            else:
                _cprint(
                    verbose,
                    f"\t{cb_name}: successfully split "
                    f"({peak_allocated} -> {new_peak_allocated})",
                )
                peak_allocated = new_peak_allocated  # update the peak memory

    if level > 3:
        _cprint(verbose, "Level 4: greedily disable GCContainers")
        results = _inference_time_profile(net, dummy_input, ctx, device)

        for r in results:
            cb_name = r[0]
            cb, parent, child_name = get_module_and_parent(net, cb_name.split(" ")[-1])

            # try to unwrap the GCContainer
            ucb = _unwrap_gc_container(cb)
            setattr(parent, child_name, ucb)

            # if the peak memory increases, revert; otherwise, keep the change
            new_peak_allocated, _ = _train_peak_memory(net, dummy_input, ctx, device)
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

    # Warm up in the main process to avoid 1st-time overhead
    net = net.to(device)
    dummy_input = _dummy_input_to_device(dummy_input, device)
    _dummy_train_step(net, dummy_input, restore_bn=True)

    et = time.time()
    _cprint(verbose, f"Total time: {et - st:.2f}s")
    return net.cpu()  # must return a model on CPU
