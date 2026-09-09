"""Tests for ``set_step_mode`` / ``set_backend`` behavior (issue #632)."""

import torch.nn as nn

from spikingjelly.activation_based import base, functional, neuron


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


def test_set_step_mode_plain_module_assigns_value():
    net = nn.Sequential(_PlainStepModeModule())
    functional.set_step_mode(net, "m")

    assert net[0].step_mode == "m"


def test_set_step_mode_step_module_container_configures_children():
    net = _StepModuleContainer()
    functional.set_step_mode(net, "m")

    assert net.step_mode == "m"
    assert net.sn.step_mode == "m"


def test_set_backend_before_multi_step_keeps_torch_until_mode_changes():
    sn = neuron.ParametricLIFNode()
    net = nn.Sequential(sn)
    assert sn.step_mode == "s"
    assert sn.backend == "torch"
    assert sn.supported_backends == ("torch",)

    functional.set_backend(net, "cupy")

    assert sn.backend == "torch"
    assert sn.step_mode == "s"

    functional.set_step_mode(net, "m")
    assert sn.step_mode == "m"
    assert "cupy" in sn.supported_backends

    functional.set_backend(net, "torch")
    assert sn.backend == "torch"
