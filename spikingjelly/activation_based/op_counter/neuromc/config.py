from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["MemoryHierarchyConfig"]


_DEFAULT_MEMORY_COST_PJ_PER_BIT = {
    "reg": 0.00901,
    "sram": 92.68282828 / 256.0,
    "dram": 1300.0 / 64.0,
}

_DEFAULT_CAPACITY_BITS = {
    "reg": float(256 * 1024 * 8),
    "sram": float(38 * 1024 * 1024 * 8),
    "dram": float("inf"),
}


@dataclass
class MemoryHierarchyConfig:
    r"""
    **API Language:**
    :ref:`中文 <MemoryHierarchyConfig-cn>` | :ref:`English <MemoryHierarchyConfig-en>`

    ----

    .. _MemoryHierarchyConfig-cn:

    * **中文**

    NeuroMC runtime 能耗估计中的访存层级配置。
    当前实现固定使用 ``reg/sram/dram`` 三层层级；可通过该配置控制：

    - 使用驻留模拟模型（``memory_model="residency"``）还是加权近似模型（``memory_model="weighted"``）；
    - 各层容量（bit）；
    - 各层单位 bit 的能耗成本（pJ/bit）；
    - 淘汰策略（当前仅支持 ``LRU``）。

    :param memory_model: 访存模型类型，支持 ``"residency"`` 或 ``"weighted"``
    :type memory_model: str

    :param level_order: 层级顺序，v1 固定为 ``("reg", "sram", "dram")``
    :type level_order: tuple[str, str, str]

    :param capacity_bits: 各层容量（bit）
    :type capacity_bits: dict[str, float]

    :param energy_pj_per_bit: 各层单位 bit 能耗（pJ/bit）
    :type energy_pj_per_bit: dict[str, float]

    :param eviction_policy: 淘汰策略，v1 仅支持 ``"LRU"``
    :type eviction_policy: str

    ----

    .. _MemoryHierarchyConfig-en:

    * **English**

    Memory-hierarchy configuration for NeuroMC runtime energy estimation.
    Currently, the hierarchy is fixed to ``reg/sram/dram``. This config controls:

    - memory model mode: residency simulation (``"residency"``) or weighted approximation (``"weighted"``);
    - per-level capacity in bits;
    - per-level energy cost in pJ/bit;
    - eviction policy (currently only ``"LRU"``).

    :param memory_model: memory model type, either ``"residency"`` or ``"weighted"``
    :type memory_model: str

    :param level_order: hierarchy order, fixed to ``("reg", "sram", "dram")`` in v1
    :type level_order: tuple[str, str, str]

    :param capacity_bits: per-level capacity in bits
    :type capacity_bits: dict[str, float]

    :param energy_pj_per_bit: per-level energy cost in pJ/bit
    :type energy_pj_per_bit: dict[str, float]

    :param eviction_policy: eviction policy, only ``"LRU"`` is supported in v1
    :type eviction_policy: str
    """

    memory_model: str = "residency"
    level_order: tuple[str, str, str] = ("reg", "sram", "dram")
    capacity_bits: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_CAPACITY_BITS)
    )
    energy_pj_per_bit: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_MEMORY_COST_PJ_PER_BIT)
    )
    eviction_policy: str = "LRU"

    @classmethod
    def neuromc_like_v1(cls, memory_model: str = "residency"):
        r"""
        **API Language:**
        :ref:`中文 <MemoryHierarchyConfig.neuromc_like_v1-cn>` | :ref:`English <MemoryHierarchyConfig.neuromc_like_v1-en>`

        ----

        .. _MemoryHierarchyConfig.neuromc_like_v1-cn:

        * **中文**

        构造一份 NeuroMC-like v1 的默认访存配置：

        - ``level_order = ("reg", "sram", "dram")``
        - ``capacity_bits`` 使用当前项目默认硬件档位
        - ``energy_pj_per_bit`` 使用当前默认单位成本
        - ``eviction_policy = "LRU"``

        :param memory_model: 访存模型类型，支持 ``"residency"`` 或 ``"weighted"``
        :type memory_model: str

        :return: 默认配置对象
        :rtype: MemoryHierarchyConfig

        ----

        .. _MemoryHierarchyConfig.neuromc_like_v1-en:

        * **English**

        Build a default NeuroMC-like v1 memory config with:

        - ``level_order = ("reg", "sram", "dram")``
        - project-default hardware capacities
        - project-default per-bit energy costs
        - ``eviction_policy = "LRU"``

        :param memory_model: memory model type, either ``"residency"`` or ``"weighted"``
        :type memory_model: str

        :return: default configuration object
        :rtype: MemoryHierarchyConfig
        """
        return cls(
            memory_model=memory_model,
            level_order=("reg", "sram", "dram"),
            capacity_bits=dict(_DEFAULT_CAPACITY_BITS),
            energy_pj_per_bit=dict(_DEFAULT_MEMORY_COST_PJ_PER_BIT),
            eviction_policy="LRU",
        )

    def copy(self):
        r"""
        **API Language:**
        :ref:`中文 <MemoryHierarchyConfig.copy-cn>` | :ref:`English <MemoryHierarchyConfig.copy-en>`

        ----

        .. _MemoryHierarchyConfig.copy-cn:

        * **中文**

        复制当前配置，返回一个新的 ``MemoryHierarchyConfig`` 对象。
        内部 ``dict`` 字段会做浅层复制，便于后续安全修改。

        :return: 配置副本
        :rtype: MemoryHierarchyConfig

        ----

        .. _MemoryHierarchyConfig.copy-en:

        * **English**

        Return a copied ``MemoryHierarchyConfig`` instance.
        Internal dictionary fields are shallow-copied for safe downstream edits.

        :return: copied config
        :rtype: MemoryHierarchyConfig
        """
        return MemoryHierarchyConfig(
            memory_model=self.memory_model,
            level_order=tuple(self.level_order),
            capacity_bits=dict(self.capacity_bits),
            energy_pj_per_bit=dict(self.energy_pj_per_bit),
            eviction_policy=self.eviction_policy,
        )

    def validate(self):
        r"""
        **API Language:**
        :ref:`中文 <MemoryHierarchyConfig.validate-cn>` | :ref:`English <MemoryHierarchyConfig.validate-en>`

        ----

        .. _MemoryHierarchyConfig.validate-cn:

        * **中文**

        校验配置是否合法；不合法时抛出 ``ValueError``。
        当前 v1 会检查：

        - ``memory_model`` 是否在 ``{"residency", "weighted"}``；
        - ``level_order`` 是否为固定三层 ``("reg", "sram", "dram")``；
        - ``eviction_policy`` 是否为 ``"LRU"``。

        ----

        .. _MemoryHierarchyConfig.validate-en:

        * **English**

        Validate this configuration and raise ``ValueError`` on invalid values.
        In v1, it checks:

        - ``memory_model`` is one of ``{"residency", "weighted"}``;
        - ``level_order`` equals fixed three levels ``("reg", "sram", "dram")``;
        - ``eviction_policy`` equals ``"LRU"``.
        """
        if self.memory_model not in ("residency", "weighted"):
            raise ValueError(
                f"memory_model must be 'residency' or 'weighted', got {self.memory_model}."
            )
        if self.level_order != ("reg", "sram", "dram"):
            raise ValueError("level_order must be ('reg', 'sram', 'dram').")
        if self.eviction_policy != "LRU":
            raise ValueError("Only LRU eviction_policy is supported in v1.")
