from spikingjelly.logger import logger

try:
    from . import info as info
    from . import hop as hop
    from . import kernel as kernel
    from . import template as template
    from . import wrapper as wrapper
    from . import custom_ops as custom_ops
except Exception as e:
    logger.debug("spikingjelly.activation_based.triton_kernel.flexsn: %s", e)
    info = None
    hop = None
    kernel = None
    template = None
    wrapper = None
    custom_ops = None
else:
    from .info import *
    from .hop import *
    from .kernel import *
    from .template import *
    from .wrapper import *
