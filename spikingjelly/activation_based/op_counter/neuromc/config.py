from __future__ import annotations

import warnings
from dataclasses import dataclass, field

__all__ = ["MemoryInstanceSpec", "MemoryHierarchyConfig"]


@dataclass(frozen=True)
class MemoryInstanceSpec:
    name: str
    size_bits: int
    datawidth: int
    r_bw: int
    w_bw: int
    r_cost: float
    w_cost: float


_MEMORY_INSTANCES = {
    "reg_1b": MemoryInstanceSpec("reg_1b", 1, 1, 1, 1, 0.00901 / 7.1 * 2, 0.00901 / 7.1 * 2),
    "reg_16b": MemoryInstanceSpec(
        "reg_16b", 16, 16, 16, 16, 0.14416 / 7.1 * 2, 0.14416 / 7.1 * 2
    ),
    "reg_32b": MemoryInstanceSpec(
        "reg_32b", 32, 32, 32, 32, 0.14416 * 2 / 7.1 * 2, 0.14416 * 2 / 7.1 * 2
    ),
    "sram_fp_conv_in_s": MemoryInstanceSpec(
        "sram_fp_conv_in_s", 18496, 1, 1, 1, 0.44854798, 0.44854798
    ),
    "sram_fp_conv_in_w": MemoryInstanceSpec(
        "sram_fp_conv_in_w", 18432, 16, 16, 16, 5.302477904, 5.302477904
    ),
    "sram_fp_conv_out_xi": MemoryInstanceSpec(
        "sram_fp_conv_out_xi", 65536, 16, 16, 16, 5.792676768, 5.792676768
    ),
    "sram_fp_soma_u": MemoryInstanceSpec(
        "sram_fp_soma_u", 1048576, 16, 256, 256, 92.68282828, 92.68282828
    ),
    "sram_fp_soma_s": MemoryInstanceSpec(
        "sram_fp_soma_s", 262144, 1, 16, 16, 7.176767677, 7.176767677
    ),
    "sram_fp_soma_smask": MemoryInstanceSpec(
        "sram_fp_soma_smask", 65536, 1, 16, 16, 7.176767677, 7.176767677
    ),
    "sram_bp_conv_in_du": MemoryInstanceSpec(
        "sram_bp_conv_in_du", 73984, 16, 16, 16, 5.792676768, 5.792676768
    ),
    "sram_bp_conv_in_w": MemoryInstanceSpec(
        "sram_bp_conv_in_w", 18432, 16, 16, 16, 5.302477904, 5.302477904
    ),
    "sram_bp_conv_out_res": MemoryInstanceSpec(
        "sram_bp_conv_out_res", 65536, 16, 16, 16, 5.792676768, 5.792676768
    ),
    "sram_bp_grad_in_u": MemoryInstanceSpec(
        "sram_bp_grad_in_u", 1048576, 16, 256, 256, 92.68282828, 92.68282828
    ),
    "sram_bp_grad_in_smask": MemoryInstanceSpec(
        "sram_bp_grad_in_smask", 65536, 1, 16, 16, 7.176767677, 7.176767677
    ),
    "sram_bp_grad_in_s": MemoryInstanceSpec(
        "sram_bp_grad_in_s", 65536, 1, 16, 16, 7.176767677, 7.176767677
    ),
    "sram_bp_grad_out_du": MemoryInstanceSpec(
        "sram_bp_grad_out_du", 1048576, 16, 256, 256, 92.68282828, 92.68282828
    ),
    "sram_wg_conv_in_s": MemoryInstanceSpec(
        "sram_wg_conv_in_s", 4624, 1, 1, 1, 0.44854798, 0.44854798
    ),
    "sram_wg_conv_in_du": MemoryInstanceSpec(
        "sram_wg_conv_in_du", 65536, 16, 16, 16, 5.792676768, 5.792676768
    ),
    "sram_wg_conv_out_dw": MemoryInstanceSpec(
        "sram_wg_conv_out_dw", 18432, 16, 16, 16, 5.302477904, 5.302477904
    ),
    "sram_2MB": MemoryInstanceSpec("sram_2MB", 1048576, 16, 256, 256, 46.976, 45.96),
    "sram_6MB": MemoryInstanceSpec(
        "sram_6MB", 50331648, 16, 256, 256, 458.1830181, 375.0422185
    ),
    "dram": MemoryInstanceSpec("dram", 134217728 * 64, 16, 64, 64, 1300.0, 1300.0),
}


@dataclass
class MemoryHierarchyConfig:
    r"""
    Exact NeuroMC v1 hardware preset used by the runtime profiler.
    """

    preset_name: str = "neuromc_like_v1"
    technology_nm: int = 32
    level_order: tuple[str, str, str, str] = ("dram", "sram", "reg", "noc")
    memory_instances: dict[str, MemoryInstanceSpec] = field(
        default_factory=lambda: dict(_MEMORY_INSTANCES)
    )
    zero_dram_in_paper_energy: bool = True
    zero_noc_in_paper_energy: bool = True
    zero_sram_high_directions: bool = True

    @classmethod
    def neuromc_like_v1(cls, memory_model: str | None = None):
        if memory_model is not None:
            warnings.warn(
                "memory_model is deprecated and ignored by exact NeuroMC v2.",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(
            preset_name="neuromc_like_v1",
            technology_nm=32,
            level_order=("dram", "sram", "reg", "noc"),
            memory_instances=dict(_MEMORY_INSTANCES),
            zero_dram_in_paper_energy=True,
            zero_noc_in_paper_energy=True,
            zero_sram_high_directions=True,
        )

    def copy(self):
        return MemoryHierarchyConfig(
            preset_name=self.preset_name,
            technology_nm=self.technology_nm,
            level_order=tuple(self.level_order),
            memory_instances=dict(self.memory_instances),
            zero_dram_in_paper_energy=self.zero_dram_in_paper_energy,
            zero_noc_in_paper_energy=self.zero_noc_in_paper_energy,
            zero_sram_high_directions=self.zero_sram_high_directions,
        )

    def validate(self):
        if self.preset_name != "neuromc_like_v1":
            raise ValueError(
                f"Only preset_name='neuromc_like_v1' is supported, got {self.preset_name}."
            )
        if self.technology_nm != 32:
            raise ValueError("Exact NeuroMC v1 only supports 32nm hardware constants.")
        required = {
            "reg_1b",
            "reg_16b",
            "reg_32b",
            "sram_fp_conv_in_s",
            "sram_fp_conv_in_w",
            "sram_fp_conv_out_xi",
            "sram_fp_soma_u",
            "sram_fp_soma_s",
            "sram_fp_soma_smask",
            "sram_bp_conv_in_du",
            "sram_bp_conv_in_w",
            "sram_bp_conv_out_res",
            "sram_bp_grad_in_u",
            "sram_bp_grad_in_s",
            "sram_bp_grad_in_smask",
            "sram_bp_grad_out_du",
            "sram_wg_conv_in_s",
            "sram_wg_conv_in_du",
            "sram_wg_conv_out_dw",
            "sram_2MB",
            "sram_6MB",
            "dram",
        }
        missing = sorted(required.difference(self.memory_instances.keys()))
        if missing:
            raise ValueError(f"Missing required NeuroMC memory instances: {missing}.")
