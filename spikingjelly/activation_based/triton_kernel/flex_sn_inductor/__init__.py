try:
    from .hop import (
        FlexSNScan,
        eager_scan,
        eager_scan_final_state,
        flex_sn_scan,
        lowerable_scan,
        lowerable_scan_available,
        lowerable_scan_final_state,
        lowerable_while_loop_available,
        lowerable_while_loop_scan,
        lowerable_while_loop_scan_final_state,
    )
except Exception as e:
    import logging

    logging.info(f"spikingjelly.activation_based.triton_kernel.flex_sn_inductor: {e}")
    flex_sn_scan = None
    FlexSNScan = None
    eager_scan = None
    eager_scan_final_state = None
    lowerable_scan = None
    lowerable_scan_final_state = None
    lowerable_while_loop_scan = None
    lowerable_while_loop_scan_final_state = None
    lowerable_scan_available = None
    lowerable_while_loop_available = None
