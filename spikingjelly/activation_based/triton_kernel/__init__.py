try:
    from . import flexsn
    from .neuron_kernel import *
    from . import torch2triton
except BaseException as e:
    import logging

    logging.info(f"spikingjelly.activation_based.triton_kernel: {e}")
    flexsn = None
    torch2triton = None
    neuron_kernel = None
