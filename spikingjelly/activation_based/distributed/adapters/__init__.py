from __future__ import annotations

from typing import Dict, Optional

from .base import SNNDistributedAdapter, infer_model_family
from .cifar10dvs_vgg import CIFAR10DVSVGGAdapter, build_cifar10dvs_vgg_eager_policy
from .spikformer import SpikformerAdapter, build_spikformer_eager_policy

_ADAPTER_REGISTRY: Dict[str, SNNDistributedAdapter] = {
    "cifar10dvs_vgg": CIFAR10DVSVGGAdapter(),
    "spikformer": SpikformerAdapter(),
}

__all__ = (
    "CIFAR10DVSVGGAdapter",
    "SNNDistributedAdapter",
    "SpikformerAdapter",
    "build_cifar10dvs_vgg_eager_policy",
    "build_spikformer_eager_policy",
    "get_adapter",
    "infer_model_family",
    "list_adapters",
    "resolve_adapter",
)


def get_adapter(name: str) -> SNNDistributedAdapter:
    if name not in _ADAPTER_REGISTRY:
        raise KeyError(f"Unknown distributed adapter '{name}'.")
    return _ADAPTER_REGISTRY[name]


def resolve_adapter(
    model, model_family: Optional[str] = None
) -> Optional[SNNDistributedAdapter]:
    family = model_family if model_family != "generic" else None
    family = family or infer_model_family(model)
    if family is None:
        return None
    return _ADAPTER_REGISTRY.get(family)


def list_adapters():
    return tuple(sorted(_ADAPTER_REGISTRY.keys()))
