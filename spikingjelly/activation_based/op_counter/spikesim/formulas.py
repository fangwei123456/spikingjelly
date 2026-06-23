from __future__ import annotations

import math
from typing import Any, Mapping

from .config import SpikeSimEnergyConfig

__all__ = [
    "compute_spikesim_dense_energy_breakdown",
    "compute_spikesim_event_energy_breakdown",
]


def compute_spikesim_dense_energy_breakdown(
    stats: Mapping[str, Any],
    metadata: Mapping[str, Any],
    config: SpikeSimEnergyConfig,
) -> dict[str, float]:
    dense_pe_cycles = float(stats["dense_pe_cycle_count"])
    pe_cycle_energy = config.pe_cycle_energy_for_kernel_pj(metadata["kernel_size"])
    total = dense_pe_cycles * pe_cycle_energy
    return {
        "pe_cycle_pj": total,
        "total_pj": total,
    }


def compute_spikesim_event_energy_breakdown(
    stats: Mapping[str, Any],
    metadata: Mapping[str, Any],
    config: SpikeSimEnergyConfig,
) -> dict[str, float]:
    """Compute SpikeSim event-based energy breakdown for a layer.

    **API Language** - :ref:`中文 <compute_spikesim_event_energy_breakdown-cn>` | :ref:`English <compute_spikesim_event_energy_breakdown-en>`

    ----

    .. _compute_spikesim_event_energy_breakdown-cn:

    * **中文**

    compute spikesim event energy breakdown 函数

    :param stats: Operation statistics from SpikeSim tracing
    :type stats: ``Mapping[str, Any]``
    :param metadata: Layer metadata (channels, tile configuration)
    :type metadata: ``Mapping[str, Any]``
    :param config: SpikeSim energy configuration
    :type config: ``SpikeSimEnergyConfig``
    :return: Dictionary of energy components in picojoules
    :rtype: dict[str, float]
    :raises ValueError: If ``active_row_count_by_tile`` and ``input_tile_channels`` are inconsistently specified
    Calculates patch control, crossbar array, and row energy contributions
    based on operation statistics and hardware configuration.

    ----

    .. _compute_spikesim_event_energy_breakdown-en:

    * **English**

    Compute Spikesim Event Energy Breakdown function

    :param stats: Operation statistics from SpikeSim tracing
    :param metadata: Layer metadata (channels, tile configuration)
    :param config: SpikeSim energy configuration
    :type stats: ``Mapping[str, Any]``
    :type metadata: ``Mapping[str, Any]``
    :type config: ``SpikeSimEnergyConfig``
    :raises ValueError: If ``active_row_count_by_tile`` and ``input_tile_channels`` are inconsistently specified
    :return: Dictionary of energy components in picojoules
    :rtype: dict[str, float]
    """
    if config.activity_mode != "event":
        return compute_spikesim_dense_energy_breakdown(stats, metadata, config)

    p_i = math.ceil(metadata["in_channels"] / config.xbar_size)

    patch_control = (
        float(metadata["out_channel_tiles"])
        * float(stats["active_patch_tile_count"])
        * config.patch_control_energy_pj
    )
    active_row_count_by_tile = stats.get("active_row_count_by_tile")
    input_tile_channels = metadata.get("input_tile_channels")
    if (active_row_count_by_tile is None) != (input_tile_channels is None):
        raise ValueError(
            "active_row_count_by_tile and input_tile_channels must either both "
            "be present or both be absent."
        )
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
