"""Tests for ``set_step_mode`` / ``set_backend`` diagnostics (issue #632)."""

import torch.nn as nn

from spikingjelly.activation_based import base, functional, neuron
from spikingjelly.logger import logger

_WARNING_LEVEL = logger.level("WARNING").no


class _PlainStepModeModule(nn.Module):
    """Carries a ``step_mode`` attribute but is not a ``base.StepModule``."""

    def __init__(self):
        super().__init__()
        self.step_mode = "s"


class _StepModuleContainer(nn.Module, base.StepModule):
    """Follows the ``StepModule`` contract; ``step_mode`` is validated."""

    def __init__(self):
        super().__init__()
        self.step_mode = "s"
        self.sn = neuron.IFNode()


def _warnings(records: list[dict]) -> list[dict]:
    return [r for r in records if r["level"].no >= _WARNING_LEVEL]


def test_set_step_mode_plain_module_warns_with_remedy_and_assigns(loguru_records):
    net = nn.Sequential(_PlainStepModeModule())
    functional.set_step_mode(net, "m")

    # Behaviour is unchanged: the value is still assigned.
    assert net[0].step_mode == "m"

    records = _warnings(loguru_records)
    assert len(records) == 1
    assert records[0]["name"].startswith("spikingjelly")
    message = records[0]["message"]
    assert "which is not a StepModule" in message
    assert "step_mode=m" in message
    assert "Inherit from spikingjelly.activation_based.base.StepModule" in message


def test_set_step_mode_step_module_container_does_not_warn(loguru_records):
    net = _StepModuleContainer()
    functional.set_step_mode(net, "m")

    assert _warnings(loguru_records) == []
    assert net.step_mode == "m"
    assert net.sn.step_mode == "m"


def test_set_backend_before_multi_step_warns_and_keeps_torch(loguru_records):
    sn = neuron.ParametricLIFNode()
    net = nn.Sequential(sn)
    assert sn.step_mode == "s"
    assert sn.backend == "torch"
    assert sn.supported_backends == ("torch",)

    functional.set_backend(net, "cupy")

    # Behaviour is unchanged: the unsupported backend is rejected and kept.
    assert sn.backend == "torch"
    assert sn.step_mode == "s"

    records = _warnings(loguru_records)
    assert len(records) == 1
    assert records[0]["name"].startswith("spikingjelly")
    message = records[0]["message"]
    assert "does not support backend=cupy" in message
    assert "while step_mode=s" in message
    assert "supported_backends=('torch',)" in message
    assert "continue using backend=torch" in message
    assert "set_step_mode() before set_backend()" in message

    # The remedy named in the warning works: multi-step mode exposes cupy.
    # (Do not assign backend="cupy" here; CuPy is not installed on CPU CI.)
    functional.set_step_mode(net, "m")
    assert sn.step_mode == "m"
    assert "cupy" in sn.supported_backends

    # A supported backend is applied silently.
    loguru_records.clear()
    functional.set_backend(net, "torch")
    assert _warnings(loguru_records) == []
    assert sn.backend == "torch"
