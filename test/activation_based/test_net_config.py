"""Tests for step-mode configuration."""

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
