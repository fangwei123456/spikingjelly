from __future__ import annotations

import math
from dataclasses import dataclass, replace

__all__ = ["SpikeSimEnergyConfig"]
_SPIKESIM_XBAR_ROW_DIVISOR = 8.0


@dataclass
class SpikeSimEnergyConfig:
    r"""
    **API Language:**
    :ref:`中文 <SpikeSimEnergyConfig-cn>` | :ref:`English <SpikeSimEnergyConfig-en>`

    ----

    .. _SpikeSimEnergyConfig-cn:

    * **中文**

    Runtime energy configuration for the event-driven SpikeSim estimator.

    The default coefficients intentionally match the released ``ela_spikesim.py``
    energy path, while the counting logic is replaced by runtime event analysis.

    ----

    .. _SpikeSimEnergyConfig-en:

    * **English**

    Runtime energy configuration for the event-driven SpikeSim estimator.

    The default coefficients intentionally match the released ``ela_spikesim.py``
    energy path, while the counting logic is replaced by runtime event analysis.
    """

    xbar_size: int = 64
    device: str = "rram"
    tile_buffer_pj: float = 397.0
    temp_buffer_pj: float = 0.2
    sub_pj: float = 1.15e-6
    adc_pj: float = 2.03084
    htree_pj: float = 19.64 * 8.0
    mux_pj: float = 0.094245
    mem_fetch_pj: float = 4.64
    neuron_pj: float = 1.274 * 4.0
    rram_xbar_pj: float = 1.76423
    sram_xbar_pj: float = 671.089

    def copy(self) -> "SpikeSimEnergyConfig":
        r"""
        **API Language:**
        :ref:`中文 <SpikeSimEnergyConfig.copy-cn>` |
        :ref:`English <SpikeSimEnergyConfig.copy-en>`

        ----

        .. _SpikeSimEnergyConfig.copy-cn:

        * **中文**

        复制当前配置并返回新对象。

        ----

        .. _SpikeSimEnergyConfig.copy-en:

        * **English**

        Return a copied config object.
        """
        return replace(self)

    def validate(self) -> None:
        r"""
        **API Language:**
        :ref:`中文 <SpikeSimEnergyConfig.validate-cn>` |
        :ref:`English <SpikeSimEnergyConfig.validate-en>`

        ----

        .. _SpikeSimEnergyConfig.validate-cn:

        * **中文**

        校验配置是否合法；不合法时抛出 ``ValueError``。

        ----

        .. _SpikeSimEnergyConfig.validate-en:

        * **English**

        Validate the config and raise ``ValueError`` on invalid values.
        """
        if self.xbar_size <= 0:
            raise ValueError(f"xbar_size must be positive, got {self.xbar_size}.")
        if self.device not in ("rram", "sram"):
            raise ValueError(f"device must be 'rram' or 'sram', got {self.device}.")

    @property
    def xbar_array_energy_pj(self) -> float:
        if self.device == "rram":
            return self.rram_xbar_pj
        return self.sram_xbar_pj

    @property
    def patch_control_energy_pj(self) -> float:
        b = float(self.xbar_size)
        return (
            self.htree_pj
            + self.mem_fetch_pj
            + self.tile_buffer_pj
            + (b / 8.0) * 16.0 * self.sub_pj
            + (b / 8.0) * self.temp_buffer_pj
            + (b / 8.0) * (b / 8.0) * (self.adc_pj + self.mux_pj)
        )

    def xbar_row_energy_pj(self, tile_channels: int) -> float:
        if tile_channels <= 0:
            raise ValueError(f"tile_channels must be positive, got {tile_channels}.")
        padded_ratio = math.ceil(tile_channels / self.xbar_size)
        padded_ratio = padded_ratio * self.xbar_size / float(tile_channels)
        # Keep the row-level basis aligned with the released SpikeSim dense
        # energy path, whose xbar contribution scales as (xbar_size / 8) * k^2.
        return (self.xbar_array_energy_pj / _SPIKESIM_XBAR_ROW_DIVISOR) * padded_ratio
