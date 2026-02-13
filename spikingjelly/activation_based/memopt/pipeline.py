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
from .. import functional
from . import compress
from .compress import BaseSpikeCompressor, BitSpikeCompressor, NullSpikeCompressor
from .checkpointing import GCContainer, TCGCContainer


__all__ = ["resolve_device", "memory_optimization"]


def resolve_device() -> str:
    """Resolve the logical device for the current process.

    Priority:

    #. If no cuda available -> cpu
    #. LOCAL_RANK / SLURM_LOCALID / OMPI_COMM_WORLD_LOCAL_RANK env
    #. If torch.distributed initialized -> use rank % ngpus
    #. torch.cuda.current_device()
    #. fallback to cuda
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


def _probe_binary_inputs(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: Union[torch.Tensor, tuple],
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
            return type(dummy_input)(
                _generate_input_like(t) for t in dummy_input
            )
        elif isinstance(dummy_input, dict):
            return {
                k: _generate_input_like(v)
                for k, v in dummy_input.items()
            }
        else:
            # Non-tensor inputs (e.g., None, int, etc.)
            return dummy_input

    is_training = net.training
    net.eval()
    with torch.no_grad():
        for _ in range(n_trials):
            new_input = _generate_input_like(dummy_input)
            if isinstance(new_input, (tuple, list)):
                _ = net(*new_input)
            elif isinstance(new_input, dict):
                _ = net(**new_input)
            else:
                _ = net(new_input)
            functional.reset_net(net)

    net.train(is_training)
    for h in hooks:
        h.remove()
    return dict(is_binary)


def _apply_gc(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: Optional[Union[torch.Tensor, tuple]] = None,
    compress_x: bool = True,
    device: str = "cuda",
) -> nn.Module:
    net = net.to(device)
    dummy_input = dummy_input.to(device)

    is_binary_input = {}
    if compress_x and dummy_input is not None:
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
    net = net.cpu()
    return net


def _dummy_train_step(net: nn.Module, dummy_input: Union[torch.Tensor, tuple]):
    net.train()
    net.zero_grad(set_to_none=True)

    def _prepare_dummy_input(dummy_input):
        """
        Clone, detach, requires grad.
        """
        if isinstance(dummy_input, torch.Tensor):
            return dummy_input.clone().detach().requires_grad_(True)
        elif isinstance(dummy_input, (tuple, list)):
            return type(dummy_input)(
                _prepare_dummy_input(t) for t in dummy_input
            )
        elif isinstance(dummy_input, dict):
            return {
                k: _prepare_dummy_input(v)
                for k, v in dummy_input.items()
            }
        else:
            # Non-tensor inputs (e.g., None, int, etc.)
            return dummy_input

    dummy_input = _prepare_dummy_input(dummy_input)
    out = net(dummy_input)

    def _calculate_loss(out):
        if isinstance(out, torch.Tensor):
            return out.sum()
        elif isinstance(out, (tuple, list)):
            l = 0.
            for t in out:
                l = l + _calculate_loss(out)
            return l
        elif isinstance(dummy_input, dict):
            l = 0.
            for t in out.values():
                l = l + _calculate_loss(out)
            return l
        else:
            return 0.

    loss = _calculate_loss(out)
    loss.backward()
    functional.reset_net(net)


def _dummy_input_to_device(dummy_input, device):
    if isinstance(dummy_input, torch.Tensor):
        return dummy_input.to(device)
    elif isinstance(dummy_input, (tuple, list)):
        return type(dummy_input)(
            _dummy_input_to_device(t, device) for t in dummy_input
        )
    elif isinstance(dummy_input, dict):
        return {
            k: _dummy_input_to_device(v, device)
            for k, v in dummy_input.items()
        }
    else:
        # Non-tensor inputs (e.g., None, int, etc.)
        return dummy_input


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
        args=(copy.deepcopy(net).cpu(), _dummy_input_to_device("cpu"), q, device),
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
        args=(copy.deepcopy(net).cpu(), _dummy_input_to_device(dummy_input, "cpu"), q, device),
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
            _ = net(dummy_input)
            functional.reset_net(net)
    results = prof.export(output=False)
    prof.close()
    q.put(results)


def _inference_time_profile(net, dummy_input, ctx, device):
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_inference_time_profile_worker,
        args=(copy.deepcopy(net).cpu(), _dummy_input_to_device(dummy_input, "cpu"), q, device),
        kwargs={"N": 50},
    )
    p.start()
    results = q.get()
    p.join()
    return results


def _get_module_and_parent(
    net: nn.Module, module_name: str
) -> Tuple[nn.Module, nn.Module, str]:
    """
    Given a dotted module path (e.g., 'layer1.0.conv1', not including the
    top-level module name), return (target_module, parent_module, child_name).

    Example:
        m, parent, child_name = get_module_and_parent(net, "layer1.0.conv1")
        # parent.child_name == m
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
    else: # not split-able
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


def _temporally_split_gc_container(block: GCContainer, factor: int = 2):
    assert isinstance(block, GCContainer)

    x_compressor = block.x_compressor
    n_chunk = getattr(block, "n_chunk", 1)
    return TCGCContainer(x_compressor, *block, n_chunk=n_chunk * factor) # TODO: support n_seq_inputs and n_outputs


def _unwrap_gc_container(block: GCContainer) -> nn.Module:
    assert isinstance(block, GCContainer)

    if len(block) == 1:
        return block[0]
    else:
        return nn.Sequential(*block)


def _cprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


# TODO: check this function
def memory_optimization(
    net: nn.Module,
    instance: Union[type, Tuple[type]],
    dummy_input: torch.Tensor = None,
    compress_x: bool = True,
    level: int = 0,
    verbose: bool = False,
    temporal_split_factor: int = 2,
):
    """Memory optimization using gradient checkpointing and spike compression.

    This function progressively transforms the given network by wrapping
    specified layers in `GCContainer`s and applying several optimization
    strategies (in increasing order of aggressiveness):
    - Level 0: No optimization.
    - Level 1: Wrap matching modules in `GCContainer` for layer-wise
      gradient checkpointing (GC), with optional input compression.
    - Level 2: Recursively split heavy `GCContainer`s into multiple
      sub-containers along the spatial (layer-wise) dimension, if supported.
    - Level 3: Further split heavy `GCContainer`s along the temporal
      dimension (chunking the time axis), if supported.
    - Level 4: Greedily unwrap some `GCContainer`s to reduce training time cost
      if doing so does not increase the memory footprint.

    Args:
        net (nn.Module): the model to be optimized.
        instance (type or tuple of types): module classes to wrap.
        dummy_input (Tensor, optional): input for memory profiling, required
            if level > 1.
        compress_x (bool): whether to apply input spike compression.
        level (int): optimization level:
            0 - no optimization
            1 - layer-wise GC
            2 - add spatial splitting
            3 - add temporal splitting
            4 - greedily disable GC
        verbose (bool): whether to print logs.
        temporal_split_factor (int): factor to increase the number of chunks
            when splitting temporally.

    Returns:
        nn.Module: the optimized model.

    Notes:
        To support spatial splitting (level 2), modules must implement:
            __spatial_split__() -> List[nn.Module]

        To support temporal splitting (level 3), modules must implement:
            __tc_init_states__(x: Tensor) -> List[Tensor]
            __tc_forward__(x_chunk: Tensor, *states, *args) -> (y_chunk, *states)
    """
    st = time.time()
    ctx = mp.get_context("spawn")
    device = resolve_device()
    _cprint(verbose, f"Optimizing memory on device {device}")

    if level > 0:
        _cprint(verbose, "Level 1: layer-wise GC with input spike compression")
        net = _apply_gc(net, instance, dummy_input, compress_x, device)

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
            cb, parent, child_name = _get_module_and_parent(net, cb_name.split(" ")[-1])

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
            cb, parent, child_name = _get_module_and_parent(net, cb_name.split(" ")[-1])

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
            cb, parent, child_name = _get_module_and_parent(net, cb_name.split(" ")[-1])

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

    et = time.time()
    _cprint(verbose, f"Total time: {et - st:.2f}s")
    return net
