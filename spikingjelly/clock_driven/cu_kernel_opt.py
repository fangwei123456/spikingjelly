try:
    import cupy
    import torch
    nvcc_options = ('--use_fast_math',)

    threads = 1024

    def cal_blocks(numel):
        return (numel + threads - 1) // threads

    def check_contiguous(*args):
        for item in args:
            item = item.contiguous()

    def check_device(device, *args):
        for item in args:
            assert item.get_device() == device







except ImportError:
    pass