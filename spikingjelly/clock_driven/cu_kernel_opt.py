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

    def wrap_args_to_raw_kernel(device: int, args_list: list):
        # check device and contiguous
        ret_list = []
        for i in range(args_list.__len__()):
            if isinstance(args_list[i], torch.Tensor):
                assert args_list[i].get_device() == device
                args_list[i] = args_list[i].contiguous()
                ret_list.append(args_list[i].data_ptr())
            elif isinstance(args_list[i], cupy.ndarray):
                assert args_list[i].device.id == device
                args_list[i] = cupy.ascontiguousarray(args_list[i])
                ret_list.append(args_list[i])
            else:
                raise TypeError

        return tuple(ret_list)










except ImportError:
    pass