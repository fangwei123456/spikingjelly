try:
    from .hop import flex_sn_scan, FlexSNScan, eager_scan
except BaseException as e:
    import logging

    logging.info(f"spikingjelly.activation_based.triton_kernel.flex_sn_inductor: {e}")
    flex_sn_scan = None
    FlexSNScan = None
    eager_scan = None
