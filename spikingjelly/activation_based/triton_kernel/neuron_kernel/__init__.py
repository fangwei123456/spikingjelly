from spikingjelly.logger import logger

try:
    from . import activation_aware_if, ilif, stbif
    from .integrate_and_fire import *
    from .lif import *
    from .plif import *
except (ImportError, OSError) as e:
    logger.debug("Optional neuron kernels unavailable: {}", e)
    activation_aware_if = None
    ilif = None
    integrate_and_fire = None
    stbif = None
    lif = None
    plif = None
