from typing import Optional


def voltage(v, spike, threshold: float, reset: Optional[float], detach: bool):
    if detach:
        spike = spike.detach()
    if reset is None:
        return v - spike * threshold
    return spike * reset + (1.0 - spike) * v
