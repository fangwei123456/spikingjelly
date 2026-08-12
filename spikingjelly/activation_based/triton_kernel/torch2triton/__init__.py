from spikingjelly.logger import logger

try:
    from .torch2graph import *
    from .graph2triton import *
except (ImportError, OSError) as e:
    logger.debug("Optional conversion dependency unavailable: {}", e)
    torch2graph = None
    graph2triton = None
