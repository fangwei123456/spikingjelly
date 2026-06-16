from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Tuple


@dataclass(frozen=True)
class SNNDistributedTopology:
    world_size: int
    dims: Mapping[str, int]

    def __post_init__(self):
        frozen_dims = MappingProxyType(dict(self.dims))
        object.__setattr__(self, "dims", frozen_dims)
        if not isinstance(self.world_size, Integral) or isinstance(
            self.world_size, bool
        ):
            raise TypeError(
                f"world_size must be an integer, but got {type(self.world_size).__name__}."
            )
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, but got {self.world_size}.")
        if not self.dims:
            raise ValueError("dims must not be empty.")
        volume = 1
        for name, size in frozen_dims.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"Topology dimension names must be strings, but got {type(name).__name__}."
                )
            if not name:
                raise ValueError("Topology dimension names must be non-empty.")
            if not isinstance(size, Integral) or isinstance(size, bool):
                raise TypeError(
                    f"Topology dimension '{name}' must be an integer, but got {type(size).__name__}."
                )
            if size <= 0:
                raise ValueError(
                    f"Topology dimension '{name}' must be positive, but got {size}."
                )
            volume *= size
        if volume != self.world_size:
            raise ValueError(
                f"Topology dims {dict(frozen_dims)} multiply to {volume}, but world_size={self.world_size}."
            )

    @property
    def ordered_dim_names(self) -> Tuple[str, ...]:
        preferred = ("dp", "tp", "pp", "vpp")
        names = list(self.dims.keys())
        ordered = [name for name in preferred if name in self.dims]
        ordered.extend(name for name in names if name not in ordered)
        return tuple(ordered)

    @property
    def mesh_shape(self) -> Tuple[int, ...]:
        return tuple(int(self.dims[name]) for name in self.ordered_dim_names)

    @classmethod
    def from_mapping(
        cls,
        dims: Mapping[str, int],
        *,
        world_size: int | None = None,
    ) -> "SNNDistributedTopology":
        normalized_dims = {}
        for key, value in dims.items():
            if isinstance(value, bool):
                raise TypeError(
                    f"Topology dimension '{key}' must be an integer, but got bool."
                )
            if isinstance(value, Integral):
                normalized_dims[key] = int(value)
                continue
            if isinstance(value, float) and value.is_integer():
                normalized_dims[key] = int(value)
                continue
            if isinstance(value, float):
                raise TypeError(
                    f"Topology dimension '{key}' must be an integer, but got float."
                )
            raise TypeError(
                f"Topology dimension '{key}' must be an integer, but got {type(value).__name__}."
            )
        if world_size is None:
            volume = 1
            for size in normalized_dims.values():
                volume *= size
            world_size = volume
        else:
            if isinstance(world_size, bool):
                raise TypeError("world_size must be an integer, but got bool.")
            if isinstance(world_size, Integral):
                world_size = int(world_size)
            elif isinstance(world_size, float) and world_size.is_integer():
                world_size = int(world_size)
            elif isinstance(world_size, float):
                raise TypeError("world_size must be an integer, but got float.")
            else:
                raise TypeError(
                    f"world_size must be an integer, but got {type(world_size).__name__}."
                )
        return cls(world_size=int(world_size), dims=normalized_dims)
