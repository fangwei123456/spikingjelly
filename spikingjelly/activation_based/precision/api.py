from __future__ import annotations

import json
import os
import warnings
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Iterable

import torch

from .config import PrecisionConfig
from .policy import FP32Policy
from .runtime import resolve_precision_policy

if TYPE_CHECKING:
    from .policy import PrecisionPolicy


@dataclass
class PrecisionArtifacts:
    requested_config: PrecisionConfig
    effective_config: PrecisionConfig
    policy: PrecisionPolicy
    model: torch.nn.Module
    scaler: torch.amp.GradScaler | None = None
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
        if clip_grad_norm is not None and parameters is None:
            parameters = self.model.parameters()

        grad_norm = None
        if self.scaler is None:
            loss.backward()
            if clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
            if step_optimizer:
                optimizer.step()
        else:
            self.scaler.scale(loss).backward()
            if clip_grad_norm is not None:
                if not step_optimizer:
                    raise ValueError(
                        "clip_grad_norm with step_optimizer=False is not supported "
                        "when a grad scaler is active."
                    )
                self.scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
            if step_optimizer:
                self.scaler.step(optimizer)
                self.scaler.update()

        return float(grad_norm) if grad_norm is not None else None

    def describe(self) -> dict:
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
    config: PrecisionConfig | str | dict,
) -> PrecisionArtifacts:
    """Prepare a model for the requested precision mode.
    Note:
        This may replace modules in the model. Call it before constructing the
        optimizer so the optimizer does not keep references to stale
        parameters.
    """
    device = torch.device(device)
    requested = PrecisionConfig.from_any(config, default_device=str(device))
    policy = resolve_precision_policy(requested)
    fallback_reason = None
    try:
        policy.check_capability(model, device)
    except RuntimeError as exc:
        if requested.strictness == "strict":
            raise
        fallback_reason = str(exc)
        if requested.strictness == "warn":
            warnings.warn(
                f"precision={requested.mode!r} is falling back to fp32: {fallback_reason}",
                RuntimeWarning,
                stacklevel=2,
            )
        fallback_policy = FP32Policy()
        fallback_policy.set_capability_report(policy.capability_report())
        policy = fallback_policy
    prepared_model = policy.prepare_model(model)
    effective = requested
    if fallback_reason is not None:
        effective = replace(requested, mode="fp32")
    scaler = policy.create_grad_scaler()
    return PrecisionArtifacts(
        requested_config=requested,
        effective_config=effective,
        policy=policy,
        model=prepared_model,
        scaler=scaler,
        fallback_reason=fallback_reason,
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
