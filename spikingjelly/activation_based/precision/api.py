from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch

from .config import PrecisionConfig
from .runtime import resolve_precision_policy


@dataclass
class PrecisionArtifacts:
    requested_config: PrecisionConfig
    effective_config: PrecisionConfig
    policy: Any
    model: torch.nn.Module
    scaler: Any
    fallback_reason: str | None = None

    def autocast_context(self) -> AbstractContextManager:
        return self.policy.autocast_context()

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad_norm: float | None = None,
        parameters: Iterable[torch.nn.Parameter] | None = None,
        step_optimizer: bool = True,
    ) -> float | None:
        if self.scaler is None:
            loss.backward()
            grad_norm = None
            if clip_grad_norm is not None:
                if parameters is None:
                    raise ValueError("parameters must be provided when clip_grad_norm is set.")
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
            if step_optimizer:
                optimizer.step()
            return None if grad_norm is None else float(grad_norm)

        self.scaler.scale(loss).backward()
        grad_norm = None
        if clip_grad_norm is not None:
            if not step_optimizer:
                raise ValueError(
                    "clip_grad_norm with step_optimizer=False is not supported when a grad scaler is active."
                )
            if parameters is None:
                raise ValueError("parameters must be provided when clip_grad_norm is set.")
            self.scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
        if step_optimizer:
            self.scaler.step(optimizer)
            self.scaler.update()
        return None if grad_norm is None else float(grad_norm)

    def describe(self) -> dict[str, Any]:
        return {
            "requested_config": asdict(self.requested_config),
            "effective_config": asdict(self.effective_config),
            "policy": self.policy.describe(),
            "capability_report": self.policy.capability_report(),
            "conversion_report": self.policy.conversion_report(),
            "fallback_reason": self.fallback_reason,
        }


def prepare_model_for_precision(
    model: torch.nn.Module,
    device: torch.device | str,
    config: PrecisionConfig | str | dict | Any,
) -> PrecisionArtifacts:
    device = torch.device(device)
    requested = PrecisionConfig.from_any(config, default_device=str(device))
    policy = resolve_precision_policy(requested)
    policy.check_capability(model, device)
    prepared_model = policy.prepare_model(model)
    effective = requested  # Fallback resolution is intentionally deferred for now.
    scaler = policy.create_grad_scaler()
    return PrecisionArtifacts(
        requested_config=requested,
        effective_config=effective,
        policy=policy,
        model=prepared_model,
        scaler=scaler,
        fallback_reason=None,
    )


def save_precision_reports(artifacts: PrecisionArtifacts, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    payload = artifacts.describe()
    runtime_summary = {
        "requested_config": payload["requested_config"],
        "effective_config": payload["effective_config"],
        "fallback_reason": payload["fallback_reason"],
    }
    for filename, key in (
        ("precision_policy.json", "policy"),
        ("capability_report.json", "capability_report"),
        ("conversion_report.json", "conversion_report"),
        ("precision_runtime.json", "runtime_summary"),
    ):
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                runtime_summary if key == "runtime_summary" else payload[key],
                f,
                indent=2,
                sort_keys=True,
            )
