try:
    import cupy
    import torch
    import time
    import numpy as np


    def cal_fun_t(n, device, f, *args, **kwargs):
        if n <= 2:
            torch.cuda.synchronize(device)
            t_start = time.perf_counter()
            f(*args, **kwargs)
            torch.cuda.synchronize(device)
            return (time.perf_counter() - t_start)
        # warm up
        f(*args, **kwargs)
        torch.cuda.synchronize(device)

        t_list = []
        for _ in range(n * 2):
            torch.cuda.synchronize(device)
            t_start = time.perf_counter()
            f(*args, **kwargs)
            torch.cuda.synchronize(device)
            t_list.append(time.perf_counter() - t_start)
        t_list = np.asarray(t_list)
        return t_list[n:].mean()


    nvcc_options = ('--use_fast_math',)

    threads = 1024

    def cal_blocks(numel: int):
        return (numel + threads - 1) // threads

    def get_contiguous(*args):
        ret_list = []
        for item in args:
            if isinstance(item, torch.Tensor):
                ret_list.append(item.contiguous())

            elif isinstance(item, cupy.ndarray):
                ret_list.append(cupy.ascontiguousarray(item))

            else:
                raise TypeError
        return ret_list

    def wrap_args_to_raw_kernel(device: int, *args):
        # note that the input must be contiguous
        # check device and get data_ptr from tensor
        ret_list = []
        for item in args:
            if isinstance(item, torch.Tensor):
                assert item.get_device() == device
                assert item.is_contiguous()
                ret_list.append(item.data_ptr())

            elif isinstance(item, cupy.ndarray):
                assert item.device.id == device
                assert item.flags['C_CONTIGUOUS']
                ret_list.append(item)

            else:
                raise TypeError

        return tuple(ret_list)










except ImportError:
    pass