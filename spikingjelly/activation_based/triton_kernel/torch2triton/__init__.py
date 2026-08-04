from spikingjelly.logger import logger

try:
    from .torch2graph import *
    from .graph2triton import *
except (ImportError, OSError) as e:
    logger.debug("spikingjelly.activation_based.triton_kernel.torch2triton: %s", e)
    torch2graph = None
    graph2triton = None
