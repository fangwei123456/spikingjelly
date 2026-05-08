from __future__ import annotations

import math
from typing import Any, Mapping

from .config import SpikeSimEnergyConfig

__all__ = [
    "compute_spikesim_event_energy_breakdown",
]


def compute_spikesim_event_energy_breakdown(
    stats: Mapping[str, Any],
    metadata: Mapping[str, Any],
    config: SpikeSimEnergyConfig,
) -> dict[str, float]:
    p_i = math.ceil(metadata["in_channels"] / config.xbar_size)

    patch_control = (
        float(metadata["out_channel_tiles"])
        * float(stats["active_patch_tile_count"])
        * config.patch_control_energy_pj
    )
    active_row_count_by_tile = stats.get("active_row_count_by_tile")
    input_tile_channels = metadata.get("input_tile_channels")
    if active_row_count_by_tile is not None and input_tile_channels is not None:
        if len(active_row_count_by_tile) != len(input_tile_channels):
            raise ValueError(
                "active_row_count_by_tile and input_tile_channels must have the "
                f"same length, but got len(active_row_count_by_tile)="
                f"{len(active_row_count_by_tile)} and len(input_tile_channels)="
                f"{len(input_tile_channels)}."
            )
        xbar_rows = 0.0
        for row_count, tile_channels in zip(
            active_row_count_by_tile, input_tile_channels
        ):
            xbar_rows += float(row_count) * config.xbar_row_energy_pj(
                int(tile_channels)
            )
        xbar = float(metadata["out_channel_tiles"]) * xbar_rows
    else:
        xbar = (
            float(metadata["out_channel_tiles"])
            * float(stats["active_row_count"])
            * config.xbar_row_energy_pj(metadata["in_channels"])
        )
    neuron = (
        float(p_i) * float(stats["active_output_tile_site_count"]) * config.neuron_pj
    )

    return {
        "patch_control_pj": patch_control,
        "xbar_pj": xbar,
        "neuron_pj": neuron,
        "total_pj": patch_control + xbar + neuron,
    }


def _compute_spikesim_dense_stage_energy(
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    num_sites: int,
    config: SpikeSimEnergyConfig,
) -> float:
    p_i = math.ceil(in_channels / config.xbar_size)
    q_i = math.ceil(out_channels / config.xbar_size)
    dense_pe_cycles = p_i * q_i * num_sites
    k_h, k_w = kernel_size
    pe_cycle_energy = (
        config.patch_control_energy_pj
        + config.neuron_pj
        + (config.xbar_size / 8.0) * k_h * k_w * config.xbar_array_energy_pj
    )
    return dense_pe_cycles * pe_cycle_energy
