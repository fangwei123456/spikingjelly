"""Device dispatch for explicit-state neuron operators."""

import hashlib
import importlib
import threading
from functools import wraps
from pathlib import Path

import torch


_CACHE_TAG_LOCK = threading.Lock()


def _include_cache_tag(tag):
    from torch.compiler import config

    with _CACHE_TAG_LOCK:
        if tag not in config.cache_key_tag:
            config.cache_key_tag += tag


def _source_cache_tag(namespace, scope, module, trainable):
    owner = Path(module.__file__)
    sources = [
        owner,
        owner.with_name("autograd.py" if trainable else "validation.py"),
        Path(__file__),
    ]
    if owner.stem == "triton":
        sources.append(Path(__file__).with_name("triton_surrogate.py"))
    elif owner.stem == "cpu":
        sources.append(Path(__file__).with_name("surrogate.py"))
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    # AOT otherwise hashes only the shared operator name, not its chosen provider.
    return f"|sj-ops:{namespace}:{scope}:{digest.hexdigest()}"


def _update_cache_tag(namespace, index, module, selected):
    tag = _source_cache_tag(
        namespace,
        f"{index}:{selected.name}",
        module,
        selected.trace_backward is not None,
    )
    _include_cache_tag(tag)
    return selected._replace(cache_tag=tag)


def _register_dispatch(namespace, cpu, selection):
    library = torch.library.Library(namespace, "FRAGMENT")
    definitions = []
    trainable = hasattr(cpu, "_backward_impl")
    cpu_tag = _source_cache_tag(namespace, "cpu", cpu, trainable)
    _include_cache_tag(cpu_tag)
    validation = importlib.import_module(
        f"{cpu.__package__}.{'autograd' if trainable else 'validation'}"
    )
    for direction in ("forward", "backward"):
        reference = getattr(cpu, f"_{direction}_impl", None)
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

        def trace_impl(select, reference):
            @wraps(reference)
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
            f"{namespace}::{direction}", trace_impl(select, reference), mutates_args=()
        )

        def fake_impl(reference_fake):
            def call(*args, **kwargs):
                device = args[0].device
                if device.type == "cpu":
                    _include_cache_tag(cpu_tag)
                elif device.type == "cuda":
                    selected = selection._selections.get(device.index)
                    if selected is not None:
                        _include_cache_tag(selected.cache_tag)
                    elif torch._guards.TracingContext.try_get() is not None:
                        raise RuntimeError(
                            "Warm up the registered neuron on this CUDA device before torch.compile."
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
    # Libraries own registrations; retain both the explicit and triton_op owners.
    return library, definitions
