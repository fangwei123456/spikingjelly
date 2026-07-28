from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class PrecisionConfig:
    mode: str = "fp32"
    strictness: str = "warn"
    fp8_recipe: str = "auto"
    device: str | None = None

    def __post_init__(self):
        if self.mode is not None:
            object.__setattr__(self, "mode", str(self.mode).lower())
        else:
            object.__setattr__(self, "mode", "fp32")
        if self.device is not None and not isinstance(self.device, str):
            object.__setattr__(self, "device", str(self.device))

    @classmethod
    def from_any(
        cls,
        config: "PrecisionConfig | str | dict | None",
        default_device: str | None = None,
    ) -> "PrecisionConfig":
        if config is None:
            return cls(device=default_device)
        if isinstance(config, cls):
            if config.device is None and default_device is not None:
                return dataclasses.replace(config, device=default_device)
            return config
        if isinstance(config, str):
            return cls(mode=config.lower(), device=default_device)
        if isinstance(config, dict):
            data = dict(config)
            if "device" not in data:
                data["device"] = default_device
            elif data["device"] is not None:
                data["device"] = str(data["device"])
            return cls(**data)

        raise TypeError(
            "PrecisionConfig.from_any() expects None, a PrecisionConfig, str, or dict."
        )
