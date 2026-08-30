from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil

from .config import MemoryHierarchyConfig, MemoryInstanceSpec


@dataclass
class _Fragment:
    stage: str
    phase: str
    op_name: str
    core_type: str
    process_key: str
    loop_dims: dict[str, int]
    input_precision_bits: int
    weight_precision_bits: int
    output_precision_bits: int
    input_numel: int
    weight_numel: int
    output_numel: int
    mac_count: int
    conv_type: str = "--"
    b_type: int = 0
    t_type: int = 0
    source: str = "trace"
    optimizer_has_momentum: bool = False
    optimizer_has_weight_decay: bool = False
    optimizer_has_momentum_buffer: bool = False


def _accumulate_memory(
    totals,
    energy,
    *,
    level: str,
    direction: str,
    bits: int,
    spec: MemoryInstanceSpec,
    config: MemoryHierarchyConfig,
) -> None:
    if bits <= 0:
        return
    if level == "dram" and config.zero_dram_in_paper_energy:
        return
    if level == "noc" and config.zero_noc_in_paper_energy:
        return
    if (
        level == "sram"
        and config.zero_sram_high_directions
        and direction in {"rl2h", "wh2l"}
    ):
        return
    totals[level][direction] += bits
    bandwidth = spec.r_bw if direction.startswith("r") else spec.w_bw
    if bandwidth <= 0:
        return
    cost = spec.r_cost if direction.startswith("r") else spec.w_cost
    energy[level][direction] += bits / bandwidth * cost


def _map_base_memory(fragment: _Fragment, config: MemoryHierarchyConfig):
    totals = defaultdict(lambda: defaultdict(int))
    energy = defaultdict(lambda: defaultdict(float))
    if fragment.mac_count == 0:
        return totals, energy

    cfg = config.memory_instances
    if fragment.core_type == "fp_soma":
        reg_i, reg_w, reg_o = cfg["reg_1b"], cfg["reg_16b"], cfg["reg_16b"]
        sram_i, sram_w, sram_o = (
            cfg["sram_fp_conv_in_s"],
            cfg["sram_fp_conv_in_w"],
            cfg["sram_fp_conv_out_xi"],
        )
    elif fragment.core_type == "ann_fe":
        reg_i = reg_w = reg_o = cfg["reg_16b"]
        sram_i, sram_w, sram_o = (
            cfg["sram_fp_conv_in_w"],
            cfg["sram_fp_conv_in_w"],
            cfg["sram_fp_conv_out_xi"],
        )
    elif fragment.core_type in {"bp_grad", "ann_be"}:
        reg_i = reg_w = reg_o = cfg["reg_16b"]
        sram_i, sram_w, sram_o = (
            cfg["sram_bp_conv_in_du"],
            cfg["sram_bp_conv_in_w"],
            cfg["sram_bp_conv_out_res"],
        )
    elif fragment.core_type == "wg":
        reg_i, reg_w, reg_o = cfg["reg_1b"], cfg["reg_16b"], cfg["reg_16b"]
        sram_i, sram_w, sram_o = (
            cfg["sram_wg_conv_in_s"],
            cfg["sram_wg_conv_in_du"],
            cfg["sram_wg_conv_out_dw"],
        )
    elif fragment.core_type == "ann_we":
        reg_i = reg_w = reg_o = cfg["reg_16b"]
        sram_i, sram_w, sram_o = (
            cfg["sram_wg_conv_in_du"],
            cfg["sram_wg_conv_in_du"],
            cfg["sram_wg_conv_out_dw"],
        )
    else:
        return totals, energy

    def add(level, direction, elements, precision, spec):
        _accumulate_memory(
            totals,
            energy,
            level=level,
            direction=direction,
            bits=elements * precision,
            spec=spec,
            config=config,
        )

    mac = fragment.mac_count
    output_elements = fragment.output_numel

    if fragment.core_type in {"wg", "ann_we"}:
        input_spatial_factor = min(fragment.loop_dims["K"], 16)
        weight_spatial_factor = min(fragment.loop_dims["C"], 16)
        input_sram_elements = (mac + input_spatial_factor - 1) // input_spatial_factor
        weight_sram_elements = (
            mac + weight_spatial_factor - 1
        ) // weight_spatial_factor
        for direction in ("rh2l", "wh2l"):
            add("reg", direction, mac, fragment.input_precision_bits, reg_i)
            add("reg", direction, mac, fragment.weight_precision_bits, reg_w)
        add("reg", "rl2h", output_elements, fragment.output_precision_bits, reg_o)
        add(
            "reg",
            "rh2l",
            mac - output_elements,
            fragment.output_precision_bits,
            reg_o,
        )
        add("reg", "wl2h", mac, fragment.output_precision_bits, reg_o)
        add(
            "sram",
            "rh2l",
            input_sram_elements,
            fragment.input_precision_bits,
            sram_i,
        )
        add(
            "sram",
            "rh2l",
            weight_sram_elements,
            fragment.weight_precision_bits,
            sram_w,
        )
        add(
            "sram",
            "wl2h",
            output_elements,
            fragment.output_precision_bits,
            sram_o,
        )
        return totals, energy

    reduction_dim = (
        fragment.loop_dims["K"]
        if fragment.core_type in {"bp_grad", "ann_be"}
        else fragment.loop_dims["C"]
    )
    reduction_chunks = (
        ceil(reduction_dim / 16) * fragment.loop_dims["FY"] * fragment.loop_dims["FX"]
    )
    partial_elements = output_elements * (reduction_chunks - 1)
    streamed_elements = output_elements * reduction_chunks

    for direction in ("rh2l", "wh2l"):
        add("reg", direction, mac, fragment.input_precision_bits, reg_i)
    reuse_weight = (fragment.b_type or fragment.t_type) and (
        fragment.weight_numel * fragment.weight_precision_bits <= sram_w.size_bits
    )
    if not reuse_weight:
        for direction in ("rh2l", "wh2l"):
            add(
                "reg",
                direction,
                fragment.weight_numel,
                fragment.weight_precision_bits,
                reg_w,
            )
        add(
            "sram",
            "rh2l",
            fragment.weight_numel,
            fragment.weight_precision_bits,
            sram_w,
        )
    add("reg", "rl2h", mac, fragment.output_precision_bits, reg_o)
    add(
        "reg",
        "rh2l",
        mac - output_elements,
        fragment.output_precision_bits,
        reg_o,
    )
    add("reg", "wh2l", mac, fragment.output_precision_bits, reg_o)
    add("reg", "wl2h", partial_elements, fragment.output_precision_bits, reg_o)
    add(
        "sram",
        "rh2l",
        streamed_elements,
        fragment.input_precision_bits,
        sram_i,
    )
    add(
        "sram",
        "rh2l",
        partial_elements,
        fragment.output_precision_bits,
        sram_o,
    )
    add(
        "sram",
        "wl2h",
        streamed_elements,
        fragment.output_precision_bits,
        sram_o,
    )
    return totals, energy
