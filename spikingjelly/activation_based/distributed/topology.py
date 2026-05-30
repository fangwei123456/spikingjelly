from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class SNNDistributedTopology:
    world_size: int
    dims: Mapping[str, int]

    def __post_init__(self):
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, but got {self.world_size}.")
        if not self.dims:
            raise ValueError("dims must not be empty.")
        volume = 1
        for name, size in self.dims.items():
            if not name:
                raise ValueError("Topology dimension names must be non-empty.")
            if size <= 0:
                raise ValueError(
                    f"Topology dimension '{name}' must be positive, but got {size}."
                )
            volume *= size
        if volume != self.world_size:
            raise ValueError(
                f"Topology dims {dict(self.dims)} multiply to {volume}, but world_size={self.world_size}."
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
        if world_size is None:
            volume = 1
            for size in dims.values():
                volume *= int(size)
            world_size = volume
        return cls(world_size=int(world_size), dims=dict(dims))
