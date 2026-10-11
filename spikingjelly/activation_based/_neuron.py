from copy import deepcopy
from typing import Any, Callable, Dict, Optional

from . import neuron


def _make_multi_step_neuron(
    spiking_neuron: Optional[Callable[..., neuron.BaseNode]],
    kwargs: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None,
) -> neuron.BaseNode:
    if kwargs.get("step_mode", "m") != "m":
        raise ValueError("Spiking neurons require step_mode='m'.")
    parameters = deepcopy(defaults) if spiking_neuron is None and defaults else {}
    parameters.update(deepcopy(kwargs))
    parameters.update(step_mode="m")
    return (neuron.LIFNode if spiking_neuron is None else spiking_neuron)(**parameters)
