"""Device dispatch for explicit-state neuron operators."""

import hashlib
import importlib
import threading
from functools import cache, wraps
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily

from spikingjelly.logger import logger


_CACHE_TAG_LOCK = threading.Lock()


def _include_cache_tag(tag):
    from torch.compiler import config

    with _CACHE_TAG_LOCK:
        if tag not in config.cache_key_tag:
            config.cache_key_tag += tag


@cache
def _source_cache_tag(namespace, scope, module, trainable):
    owner = Path(module.__file__)
    sources = [
        owner,
        owner.with_name("autograd.py" if trainable else "validation.py"),
        Path(__file__),
    ]
    if owner.stem == "triton":
        sources.append(Path(__file__).with_name("triton_surrogate.py"))
    elif owner.stem in {"cpu", "reference"}:
        torch_reference = owner.with_name("reference.py")
        if torch_reference.is_file():
            sources.append(torch_reference)
        sources.append(Path(__file__).with_name("surrogate.py"))
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    # AOT otherwise hashes only the shared operator name, not its chosen provider.
    return f"|sj-ops:{namespace}:{scope}:{digest.hexdigest()}"


def _update_cache_tag(
    namespace, index, module, selected, *, capability, priority, execution
):
    if execution == "compile":
        tag = _source_cache_tag(
            namespace,
            f"{index}:{execution}:{selected.name}",
            module,
            selected.trace_backward is not None,
        )
        _include_cache_tag(tag)
        selected = selected._replace(cache_tag=tag)
    logger.info(
        "ops selection operator={} device=cuda:{} ({}) execution={} capability={} priority={} implementation={} forward={} backward={} unavailable={}",
        namespace,
        index,
        torch.cuda.get_device_name(index),
        execution,
        capability,
        priority,
        selected.name,
        getattr(
            selected.eager_forward,
            "__qualname__",
            type(selected.eager_forward).__name__,
        ),
        getattr(
            selected.eager_backward,
            "__qualname__",
            type(selected.eager_backward).__name__,
        ),
        selected.unavailable,
    )
    return selected


def _register_dispatch(namespace, cpu, selection):
    library = torch.library.Library(namespace, "FRAGMENT")
    definitions = []
    trainable = selection._cpu_backward is not None
    cpu_forward = selection._cpu_forward
    validation = importlib.import_module(
        f"{cpu.__package__}.{'autograd' if trainable else 'validation'}"
    )
    for direction in ("forward", "backward"):
        schema = getattr(cpu, f"_{direction}_impl", None)
        reference = cpu_forward if direction == "forward" else selection._cpu_backward
        if reference is None:
            continue
        select = (
            selection.get_trace_forward
            if direction == "forward"
            else selection.get_trace_backward
        )
        select_cuda = (
            selection.get_cuda_forward
            if direction == "forward"
            else selection.get_cuda_backward
        )

        def trace_impl(select, schema):
            @wraps(schema)
            def call(*args, **kwargs):
                return select(args[0].device)(*args, **kwargs)

            return call

        def cuda_impl(select_cuda):
            def call(*args, **kwargs):
                return select_cuda(args[0].device)(*args, **kwargs)

            return call

        # Triton-op decomposition keeps provider kernels visible to torch.compile.
        # Explicit device kernels avoid its additional eager Python wrapper.
        definition = torch.library.triton_op(
            f"{namespace}::{direction}",
            trace_impl(select, schema),
            mutates_args=(),
        )

        def fake_impl(reference_fake):
            def call(*args, **kwargs):
                device = args[0].device
                tracing = torch._guards.TracingContext.try_get() is not None
                if device.type == "cpu" and tracing:
                    _include_cache_tag(
                        _source_cache_tag(namespace, "cpu", cpu, trainable)
                    )
                elif device.type == "cuda" and tracing:
                    with unset_fake_temporarily():
                        selection.get_trace_forward(device)
                    _include_cache_tag(
                        selection._get_compile_selection(device).cache_tag
                    )
                return reference_fake(*args, **kwargs)

            return call

        definition.register_fake(fake_impl(getattr(validation, f"_{direction}_fake")))
        definitions.append(definition)
        library.impl(direction, reference, "CPU")
        library.impl(direction, cuda_impl(select_cuda), "CUDA")
    if trainable:
        validation._register_ops(
            f"{namespace}::forward", f"{namespace}::backward", register_fake=False
        )
        torch_reference = Path(cpu.__file__).with_name("reference.py")
        if torch_reference.is_file():
            library.impl("forward", cpu_forward, "AutogradCPU")
    if getattr(cpu, "_forward_impl", None) is not None:
        logger.info(
            "ops binding operator={} device=cpu implementation=torch-reference",
            namespace,
        )
    # Libraries own registrations; retain both the explicit and triton_op owners.
    return library, definitions
