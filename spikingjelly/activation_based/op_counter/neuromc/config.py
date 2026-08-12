from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["MemoryInstanceSpec", "MemoryHierarchyConfig"]


@dataclass(frozen=True)
class MemoryInstanceSpec:
    """Specification for a single memory instance in the memory hierarchy.

    **API Language** - :ref:`中文 <MemoryInstanceSpec-cn>` | :ref:`English <MemoryInstanceSpec-en>`

    ----

    .. _MemoryInstanceSpec-cn:

    * **中文**

    内存层次结构中单个内存级别的规格描述。

    定义内存级别的容量（以元素数或字节数计）、每个数据元素的位宽
    以及每次访问的能耗（皮焦耳）。多个 ``MemoryInstanceSpec`` 组合
    构成一个完整的 :class:`MemoryHierarchyConfig`。

    :param capacity: 内存容量（元素数或字节数）
    :type capacity: int
    :param data_bit: Bit width of each data element
    :type data_bit: int
    :param memory_cost_pj: Energy cost per access in picojoules
    :type memory_cost_pj: float
    Describes the properties of one level in the memory hierarchy, including
    capacity, bit width, and access energy cost.

    ----

    .. _MemoryInstanceSpec-en:

    * **English**

    MemoryInstanceSpec class

    :param capacity: Capacity of the memory level (in elements or bytes)
    :param data_bit: Bit width of each data element
    :param memory_cost_pj: Energy cost per access in picojoules
    :type capacity: int
    :type data_bit: int
    :type memory_cost_pj: float
    """

    name: str
    size_bits: int
    datawidth: int
    r_bw: int
    w_bw: int
    r_cost: float
    w_cost: float


_MEMORY_INSTANCES = {
    "reg_1b": MemoryInstanceSpec(
        "reg_1b", 1, 1, 1, 1, 0.00901 / 7.1 * 2, 0.00901 / 7.1 * 2
    ),
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
    **API Language** - :ref:`中文 <MemoryHierarchyConfig-cn>` | :ref:`English <MemoryHierarchyConfig-en>`

    ----

    .. _MemoryHierarchyConfig-cn:

    * **中文**

    NeuroMC v1 硬件预设配置。``preset_name`` 和 ``technology_nm`` 是固定的
    报告元数据，不是可配置项。

    ----

    .. _MemoryHierarchyConfig-en:

    * **English**

    NeuroMC v1 hardware constants and memory hierarchy. ``preset_name`` and
    ``technology_nm`` are fixed report metadata rather than configuration knobs.
    """

    level_order: tuple[str, str, str, str] = ("dram", "sram", "reg", "noc")
    memory_instances: dict[str, MemoryInstanceSpec] = field(
        default_factory=lambda: dict(_MEMORY_INSTANCES)
    )
    zero_dram_in_paper_energy: bool = True
    zero_noc_in_paper_energy: bool = True
    zero_sram_high_directions: bool = True

    @property
    def preset_name(self) -> str:
        return "neuromc_like_v1"

    @property
    def technology_nm(self) -> int:
        return 32

    def validate(self):
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
