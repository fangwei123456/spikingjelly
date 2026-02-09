try:
    from .torch2graph import *
    from .graph2triton import *
except BaseException as e:
    import logging

    logging.info(f"spikingjelly.activation_based.triton_kernel.torch2triton: {e}")
    torch2graph = None
    graph2triton = None
