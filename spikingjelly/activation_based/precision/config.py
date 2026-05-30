from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrecisionConfig:
    mode: str = "fp32"
    strictness: str = "warn"
    fp8_recipe: str = "auto"
    report: bool = True
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
        config: "PrecisionConfig | str | dict | Any",
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
            if "precision" in data and "mode" not in data:
                data["mode"] = data.pop("precision")
            if "precision_strict" in data and "strictness" not in data:
                data["strictness"] = data.pop("precision_strict")
            if "fp8_report" in data and "report" not in data:
                data["report"] = data.pop("fp8_report")
            valid_fields = {f.name for f in dataclasses.fields(cls)}
            filtered_data = {k: v for k, v in data.items() if k in valid_fields}
            return cls(**filtered_data)

        precision = getattr(config, "precision", None)
        if precision is not None:
            device_attr = getattr(config, "device", None)
            device_value = device_attr if device_attr is not None else default_device
            return cls(
                mode=str(precision).lower(),
                strictness=getattr(config, "precision_strict", "warn"),
                fp8_recipe=getattr(config, "fp8_recipe", "auto"),
                report=getattr(config, "fp8_report", True),
                device=str(device_value) if device_value is not None else None,
            )

        if hasattr(config, "disable_amp") or hasattr(config, "device"):
            device_attr = getattr(config, "device", None)
            device_value = device_attr if device_attr is not None else default_device
            if getattr(config, "disable_amp", False):
                mode = "fp32"
            else:
                device = str(device_value or "cpu")
                mode = "fp16" if device.startswith("cuda") else "fp32"
            return cls(
                mode=mode,
                device=str(device_value) if device_value is not None else None,
            )

        raise TypeError(
            "PrecisionConfig.from_any() expects a PrecisionConfig, str, dict, or an object "
            "with precision/disable_amp/device-style attributes."
        )
